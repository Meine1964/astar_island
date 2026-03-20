"""Astar Island — Offline optimization using local ground truth data.

Designed for speed: vectorized scoring, minimal sims, 1 seed/round.

Three stages:
  1. Calibrate simulator HiddenParams against GT (coordinate descent)
  2. Cross-validate blending weights (alpha, decay, spatial blend)
  3. Per-cell error analysis

All computation is local — no API calls, no query budget used.

Usage:  uv run python optimize.py
"""
from __future__ import annotations
import json
import os
import sys
import time
import numpy as np
from dataclasses import asdict, replace

from astar_island_simulator.env import (
    AstarIslandSimulator, HiddenParams, Settlement as SimSettlement,
    CODE_TO_CLASS, NUM_CLASSES,
)
from strategy import (
    compute_features, domain_prior, OutcomeModel,
    _load_bias_data, MIN_PROB,
)

# ── Data loading ──────────────────────────────────────────────────────

def load_all_gt_data():
    """Load all local ground truth data from data/ directory."""
    entries = []
    for rnd in range(1, 20):
        rdir = f"data/round_{rnd:02d}"
        detail_path = os.path.join(rdir, "round_detail.json")
        if not os.path.exists(detail_path):
            continue
        with open(detail_path) as f:
            detail = json.load(f)

        W, H = detail["map_width"], detail["map_height"]

        for si in range(detail["seeds_count"]):
            gt_path = os.path.join(rdir, "analysis", f"seed_{si}_ground_truth.npy")
            if not os.path.exists(gt_path):
                continue
            gt = np.load(gt_path)
            if gt.shape != (H, W, NUM_CLASSES):
                continue

            state = detail["initial_states"][si]
            setts = [s for s in state["settlements"] if s.get("alive", True)]
            entries.append({
                "round_num": rnd, "seed_index": si,
                "init_grid": state["grid"], "init_settlements": setts,
                "gt": gt, "W": W, "H": H,
            })

    print(f"Loaded {len(entries)} seed-rounds of ground truth")
    return entries


def subsample(entries, per_round=1):
    """Pick `per_round` seeds per round for fast evaluation."""
    by_round = {}
    for e in entries:
        by_round.setdefault(e["round_num"], []).append(e)
    out = []
    for rn in sorted(by_round):
        out.extend(by_round[rn][:per_round])
    return out


# ── Vectorised sim + scoring ─────────────────────────────────────────

def simulate_fast(init_grid, init_settlements, W, H, params, n_sims=5):
    """Run simulator with minimal overhead, return (H,W,6) prior."""
    grid_np = np.array(init_grid, dtype=int)

    sett_objs = []
    for i, s in enumerate(init_settlements):
        sett_objs.append(SimSettlement(
            x=s["x"], y=s["y"],
            population=10, food=5.0, wealth=1.0, defense=3.0, tech_level=1.0,
            has_port=s.get("has_port", False), owner_id=i,
        ))

    sim = AstarIslandSimulator.__new__(AstarIslandSimulator)
    sim.map_seed = 0
    sim.params = params
    sim.width = W
    sim.height = H
    sim.base_grid = grid_np
    sim.base_settlements = sett_objs

    counts = np.zeros((H, W, NUM_CLASSES))
    for i in range(n_sims):
        final_grid, _ = sim.run(sim_seed=i)
        # Vectorised class mapping
        flat = final_grid.ravel().astype(int)
        # Build lookup: code -> class
        max_code = max(max(flat), 11) + 1
        lut = np.zeros(max_code + 1, dtype=int)
        for code, cls in CODE_TO_CLASS.items():
            if code <= max_code:
                lut[code] = cls
        cls_grid = lut[flat].reshape(H, W)
        for c in range(NUM_CLASSES):
            counts[:, :, c] += (cls_grid == c)

    prior = counts / n_sims
    prior = np.maximum(prior, MIN_PROB)
    prior /= prior.sum(axis=2, keepdims=True)
    return prior


def kl_score(pred, gt):
    """Entropy-weighted KL divergence over dynamic cells (vectorised)."""
    eps = 1e-10
    entropy = -np.sum(gt * np.log(gt + eps), axis=2)  # (H,W)
    dynamic = entropy > 0.01
    if not dynamic.any():
        return 0.0, 0
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)  # (H,W)
    return float(kl[dynamic].sum()), int(dynamic.sum())


def eval_params(entries, params, n_sims=5):
    """Compute mean KL across entries with given params."""
    total_kl, total_cells = 0.0, 0
    for e in entries:
        pred = simulate_fast(
            e["init_grid"], e["init_settlements"],
            e["W"], e["H"], params, n_sims
        )
        kl, nc = kl_score(pred, e["gt"])
        total_kl += kl
        total_cells += nc
    return total_kl / max(total_cells, 1)


# ── Stage 1: Simulator Calibration ───────────────────────────────────

def calibrate_simulator(entries):
    """Coordinate descent on key simulator parameters."""
    print("\n" + "=" * 60)
    print("STAGE 1: Simulator Calibration")
    print("=" * 60)

    sample = subsample(entries, per_round=1)
    print(f"Using {len(sample)} seed-rounds (1/round), 5 sims each")

    base = HiddenParams()
    best_params = base
    best_loss = eval_params(sample, base, n_sims=5)
    print(f"Baseline loss: {best_loss:.4f}")

    # Parameters and their search values — ordered by expected impact
    search_space = [
        ("winter_base_severity",    [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]),
        ("collapse_food_threshold", [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]),
        ("food_per_forest",         [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        ("food_per_plains",         [0.2, 0.3, 0.5, 0.7, 1.0]),
        ("expansion_prob",          [0.02, 0.04, 0.06, 0.08, 0.12]),
        ("expansion_pop_threshold", [10, 15, 20, 25, 30, 35]),
        ("port_creation_prob",      [0.05, 0.10, 0.15, 0.20, 0.30]),
        ("ruin_reclaim_prob",       [0.03, 0.06, 0.10, 0.15, 0.20]),
        ("forest_regrowth_prob",    [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]),
        ("plains_regrowth_prob",    [0.10, 0.15, 0.20, 0.25, 0.30]),
        ("winter_severity_variance",[1.0, 1.5, 2.0, 2.5, 3.0]),
        ("raid_strength_factor",    [0.15, 0.25, 0.30, 0.40, 0.50]),
        ("desperate_food_threshold",[2.0, 3.0, 4.0, 5.0]),
    ]

    for iteration in range(3):
        improved = False
        print(f"\n--- Pass {iteration+1} ---")

        for param_name, values in search_space:
            current_val = getattr(best_params, param_name)
            best_val = current_val
            t0 = time.time()

            for val in values:
                if isinstance(val, float) and abs(val - current_val) < 1e-6:
                    continue
                if isinstance(val, int) and val == current_val:
                    continue
                trial = replace(best_params, **{param_name: val})
                loss = eval_params(sample, trial, n_sims=5)
                if loss < best_loss:
                    best_loss = loss
                    best_val = val

            dt = time.time() - t0
            if best_val != current_val:
                best_params = replace(best_params, **{param_name: best_val})
                improved = True
                print(f"  {param_name}: {current_val} -> {best_val} "
                      f"(loss={best_loss:.4f}, {dt:.0f}s)")
            else:
                print(f"  {param_name}: {current_val} (unchanged, {dt:.0f}s)")

        if not improved:
            print("  No improvement — stopping early")
            break

    # Validate on all data with more sims
    print("\n--- Validation (all data, 10 sims) ---")
    baseline_full = eval_params(entries, HiddenParams(), n_sims=10)
    final_loss = eval_params(entries, best_params, n_sims=10)
    pct = (baseline_full - final_loss) / baseline_full * 100
    print(f"  Baseline:   {baseline_full:.4f}")
    print(f"  Calibrated: {final_loss:.4f}  ({pct:+.1f}%)")

    param_dict = asdict(best_params)
    with open("calibrated_params.json", "w") as f:
        json.dump(param_dict, f, indent=2)
    print(f"Saved calibrated_params.json")

    return best_params, final_loss


# ── Stage 2: Blending Weight Optimisation ─────────────────────────────

def build_pred_with_weights(feats, grid_list, H, W, sim_prior,
                            alpha, decay, sp_unobs, sp_obs):
    """Build prediction from sim_prior + bias correction + spatial (no obs)."""
    pred = sim_prior.copy()

    # Bias correction (vectorised per-bucket)
    data = _load_bias_data()
    bias = data["bias_by_dist"]
    if bias and alpha > 0:
        for db_val, gt_target in bias.items():
            mask = (feats[:, :, 1] == db_val)
            # exclude static cells
            for y in range(H):
                for x in range(W):
                    if mask[y, x] and grid_list[y][x] not in (5, 10):
                        a = alpha  # no obs, so decay doesn't matter
                        pred[y, x] = (1.0 - a) * pred[y, x] + a * gt_target

    pred = np.maximum(pred, MIN_PROB)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


def eval_weights(sample, sim_priors, alpha, decay, sp_unobs, sp_obs):
    """Score blending weights against GT quickly (sim priors pre-computed)."""
    total_kl, total_cells = 0.0, 0
    for i, e in enumerate(sample):
        pred = build_pred_with_weights(
            e["_feats"], e["init_grid"], e["H"], e["W"],
            sim_priors[i], alpha, decay, sp_unobs, sp_obs,
        )
        kl, nc = kl_score(pred, e["gt"])
        total_kl += kl
        total_cells += nc
    return total_kl / max(total_cells, 1)


def cross_validate_weights(entries, sim_params):
    """Search blending weights with pre-computed sim priors for speed."""
    print("\n" + "=" * 60)
    print("STAGE 2: Blending Weight Optimisation")
    print("=" * 60)

    sample = subsample(entries, per_round=1)

    # Pre-compute sim priors once
    print(f"Pre-computing sim priors for {len(sample)} seed-rounds...")
    sim_priors = []
    for e in sample:
        e["_feats"] = compute_features(e["init_grid"], e["init_settlements"],
                                       e["W"], e["H"])
        sp = simulate_fast(e["init_grid"], e["init_settlements"],
                          e["W"], e["H"], sim_params, n_sims=10)
        sim_priors.append(sp)
    print("  Done.")

    # No-correction baseline
    base_loss = eval_weights(sample, sim_priors, 0.0, 1.0, 0.0, 0.0)
    print(f"  No correction: {base_loss:.4f}")

    # Search alpha
    print("\n--- Searching alpha ---")
    best_loss = base_loss
    best_alpha = 0.0
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        loss = eval_weights(sample, sim_priors, alpha, 0.5, 0.0, 0.0)
        marker = " *" if loss < best_loss else ""
        print(f"  alpha={alpha:.2f}: {loss:.4f}{marker}")
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha

    # Search decay (with best alpha)
    print(f"\n--- Searching decay (alpha={best_alpha:.2f}) ---")
    best_decay = 0.5
    for decay in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
        loss = eval_weights(sample, sim_priors, best_alpha, decay, 0.0, 0.0)
        marker = " *" if loss < best_loss else ""
        print(f"  decay={decay:.1f}: {loss:.4f}{marker}")
        if loss < best_loss:
            best_loss = loss
            best_decay = decay

    weights = {
        "alpha": best_alpha,
        "decay": best_decay,
        "spatial_blend_unobs": 0.15,  # keep defaults (spatial needs obs data)
        "spatial_blend_obs": 0.08,
    }
    print(f"\n  Best: alpha={best_alpha:.2f}, decay={best_decay:.1f}")
    print(f"  Loss: {best_loss:.4f} (baseline was {base_loss:.4f})")

    with open("optimized_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    print(f"  Saved to optimized_weights.json")

    return weights


# ── Stage 3: Error Analysis ──────────────────────────────────────────

def error_analysis(entries, sim_params):
    """Identify systematic errors by distance bucket and GT class."""
    print("\n" + "=" * 60)
    print("STAGE 3: Error Analysis")
    print("=" * 60)

    eps = 1e-10
    errors_by_dist = {}
    errors_by_class = np.zeros(NUM_CLASSES)
    counts_by_class = np.zeros(NUM_CLASSES)

    sample = subsample(entries, per_round=2)
    print(f"Analysing {len(sample)} seed-rounds...")

    for e in sample:
        W, H = e["W"], e["H"]
        gt = e["gt"]
        feats = compute_features(e["init_grid"], e["init_settlements"], W, H)
        pred = simulate_fast(e["init_grid"], e["init_settlements"],
                            W, H, sim_params, n_sims=10)

        entropy = -np.sum(gt * np.log(gt + eps), axis=2)
        kl_map = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
        dynamic = entropy > 0.01

        for y in range(H):
            for x in range(W):
                if not dynamic[y, x]:
                    continue
                db = int(feats[y, x, 1])
                kl = kl_map[y, x]

                errors_by_dist.setdefault(db, [0.0, 0])
                errors_by_dist[db][0] += kl
                errors_by_dist[db][1] += 1

                gt_cls = np.argmax(gt[y, x])
                errors_by_class[gt_cls] += kl
                counts_by_class[gt_cls] += 1

    print("\n--- Error by distance bucket ---")
    for db in sorted(errors_by_dist):
        s, n = errors_by_dist[db]
        print(f"  Bucket {db}: avg_kl={s/n:.4f}  ({n} cells)")

    print("\n--- Error by GT class ---")
    names = ["plains/ocean", "settlement", "port", "ruin", "forest", "mountain"]
    for cls in range(NUM_CLASSES):
        if counts_by_class[cls] > 0:
            avg = errors_by_class[cls] / counts_by_class[cls]
            print(f"  {names[cls]:>12}: avg_kl={avg:.4f}  ({int(counts_by_class[cls])} cells)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    start = time.time()

    print("Loading ground truth data...")
    entries = load_all_gt_data()
    if not entries:
        print("No ground truth data found. Run download_history.py first.")
        return

    # Stage 1: Calibrate simulator
    best_params, _ = calibrate_simulator(entries)

    # Stage 2: Cross-validate blending weights
    weights = cross_validate_weights(entries, best_params)

    # Stage 3: Error analysis
    error_analysis(entries, best_params)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Optimization complete in {elapsed:.0f}s")
    print(f"  calibrated_params.json  — use with resubmit.py")
    print(f"  optimized_weights.json  — update strategy.py constants")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
