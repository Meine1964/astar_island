"""Overnight autonomous round handler.

Monitors for new rounds, evaluates completed rounds, auto-tunes parameters,
and submits predictions — all in a loop.

Usage:  uv run python overnight.py
"""
import requests
import numpy as np
import json
import time
import os
import sys
import traceback
import datetime
import truststore
truststore.inject_into_ssl()

import data_store
from strategy import (
    OutcomeModel, analyze_seeds, build_prediction,
    calibrate_from_history, compute_simulator_prior,
    execute_adaptive_queries, NUM_CLASSES,
    estimate_settlement_regime, self_consistency_tune,
)

# ── Configuration ──────────────────────────────────────────────────────
BASE = "https://api.ainm.no/astar-island"
TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJlMDcyNjM1ZC1mYTNmLTQ5MzMtOGMwNC1lMmJmYmM4ZDhiZDEi"
    "LCJlbWFpbCI6Im1laW5lLnZhbi5kZXIubWV1bGVuQGdtYWlsLmNvbSIsImlz"
    "X2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTEyMDI4fQ."
    "QMd3aqRnowq1zyiyFDnOu0bXSNSAZ2rEMaIoCkOQLJ4"
)

POLL_INTERVAL = 120          # seconds between status checks
LOG_FILE = "overnight_log.txt"

# ── State tracking ─────────────────────────────────────────────────────
submitted_rounds = set()     # round IDs we've already submitted for
evaluated_rounds = set()     # round IDs we've already evaluated


def log(msg):
    """Print and append to log file."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_session():
    s = requests.Session()
    s.cookies.set("access_token", TOKEN)
    return s


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATE: Download GT, compute KL, identify error patterns
# ═══════════════════════════════════════════════════════════════════════
CLASS_NAMES = ["Empty/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def evaluate_round(session, round_info):
    """Evaluate a completed round: download GT, compute per-class KL, log findings."""
    rid = round_info["id"]
    rnum = round_info["round_number"]
    log(f"=== EVALUATING Round #{rnum} ===")

    # Get our rank
    my_rounds = session.get(f"{BASE}/my-rounds").json()
    my_round = next((r for r in my_rounds if r.get("round_number") == rnum), None)
    rank = my_round.get("rank") if my_round else None
    log(f"  Rank: {rank}")

    # Download round detail if not saved
    detail = data_store.load_round_detail(rnum)
    if detail is None:
        detail = session.get(f"{BASE}/rounds/{rid}").json()
        data_store.save_round_detail(rnum, detail)
    seeds = detail["seeds_count"]
    W, H = detail["map_width"], detail["map_height"]

    # Download GT and scores for each seed
    gt_data = {}
    scores = {}
    for si in range(seeds):
        try:
            resp = session.get(f"{BASE}/analysis/{rid}/{si}")
            if resp.status_code != 200:
                log(f"  Seed {si}: analysis not available ({resp.status_code})")
                continue
            a = resp.json()
            gt = np.array(a["ground_truth"])
            score = a.get("score")
            gt_data[si] = gt
            scores[si] = score
            data_store.save_analysis(rnum, si, gt, score)
            time.sleep(0.3)
        except Exception as e:
            log(f"  Seed {si}: error downloading GT - {e}")

    if not gt_data:
        log("  No GT data available, skipping evaluation")
        return None

    # Load our predictions
    preds = {}
    for si in range(seeds):
        p = data_store.load_prediction(rnum, si)
        if p is not None:
            preds[si] = p

    if not preds:
        log("  No predictions saved for this round, skipping comparison")
        return {"rank": rank, "scores": scores}

    # Per-class KL analysis
    eps = 1e-10
    class_kl_totals = np.zeros(6)
    total_kl_sum = 0
    n_seeds_eval = 0

    for si in range(seeds):
        if si not in gt_data or si not in preds:
            continue
        gt, pred = gt_data[si], preds[si]
        kl_per_cell = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
        mean_kl = kl_per_cell.mean()
        total_kl_sum += mean_kl
        n_seeds_eval += 1

        for c in range(6):
            kl_c = (gt[:, :, c] * np.log((gt[:, :, c] + eps) / (pred[:, :, c] + eps))).mean()
            class_kl_totals[c] += kl_c

        log(f"  Seed {si}: score={scores.get(si, '?'):.4f}, mean_KL={mean_kl:.5f}")

    avg_kl = total_kl_sum / n_seeds_eval if n_seeds_eval > 0 else 0
    class_kl_avg = class_kl_totals / n_seeds_eval if n_seeds_eval > 0 else class_kl_totals

    log(f"  Average KL: {avg_kl:.5f}")
    log(f"  Per-class KL breakdown:")
    for c in range(6):
        gt_avg = np.mean([gt_data[si][:, :, c].mean() for si in gt_data])
        pred_avg = np.mean([preds[si][:, :, c].mean() for si in preds if si in gt_data])
        log(f"    {CLASS_NAMES[c]:15s}: KL={class_kl_avg[c]:+.5f}  "
            f"gt={gt_avg:.4f}  pred={pred_avg:.4f}")

    # Settlement survival analysis
    for si in range(seeds):
        if si not in gt_data:
            continue
        state = detail["initial_states"][si]
        setts = [s for s in state["settlements"] if s.get("alive", True)]
        survived = sum(1 for s in setts if gt_data[si][s["y"], s["x"], 1] > 0.3)
        log(f"  Seed {si}: {len(setts)} settlements, {survived} survived "
            f"({survived / len(setts) * 100:.0f}%)" if setts else "")

    return {
        "rank": rank,
        "avg_kl": avg_kl,
        "class_kl": class_kl_avg.tolist(),
        "scores": {si: s for si, s in scores.items()},
        "n_seeds": n_seeds_eval,
    }


# ═══════════════════════════════════════════════════════════════════════
#  AUTO-TUNE: Backtest parameter variations across all rounds with GT
# ═══════════════════════════════════════════════════════════════════════

def load_all_rounds_with_gt():
    """Load all rounds that have both predictions and ground truth."""
    rounds_data = []
    for rnum in range(1, 30):
        detail_file = f"data/round_{rnum:02d}/round_detail.json"
        if not os.path.exists(detail_file):
            continue
        detail = json.load(open(detail_file))
        seeds = detail["seeds_count"]
        W, H = detail["map_width"], detail["map_height"]

        seed_data = []
        for si in range(seeds):
            gt_file = f"data/round_{rnum:02d}/analysis/seed_{si}_ground_truth.npy"
            pred_file = f"data/round_{rnum:02d}/predictions/seed_{si}.npy"
            if not os.path.exists(gt_file) or not os.path.exists(pred_file):
                continue
            gt = np.load(gt_file)
            pred = np.load(pred_file)
            grid = detail["initial_states"][si]["grid"]
            seed_data.append((si, gt, pred, grid))

        if seed_data:
            rounds_data.append((rnum, W, H, seed_data))
    return rounds_data


def compute_kl(gt, pred):
    eps = 1e-10
    return np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2).mean()


def backtest_ruin_floor(rounds_data):
    """Find optimal RUIN_FLOOR by backtesting on all rounds with GT."""
    candidates = [0.002, 0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025]
    best_floor = 0.010
    best_kl = float("inf")

    for floor in candidates:
        total_kl = 0
        count = 0
        for rnum, W, H, seed_data in rounds_data:
            for si, gt, pred, grid in seed_data:
                p = pred.copy()
                for y in range(H):
                    for x in range(W):
                        cell = grid[y][x]
                        if cell == 5 or cell == 10:
                            continue
                        if p[y, x, 3] < floor:
                            delta = floor - p[y, x, 3]
                            p[y, x, 3] = floor
                            largest = np.argmax(p[y, x])
                            if largest != 3:
                                p[y, x, largest] -= delta
                            p[y, x] = np.maximum(p[y, x], 0.002)
                            p[y, x] /= p[y, x].sum()
                total_kl += compute_kl(gt, p)
                count += 1
        if count > 0:
            avg = total_kl / count
            if avg < best_kl:
                best_kl = avg
                best_floor = floor

    return best_floor, best_kl


def auto_tune(eval_result):
    """Run parameter tuning based on all available GT data.

    Returns dict of parameter changes to apply, or None if no improvement found.
    """
    log("--- AUTO-TUNE: backtesting parameters ---")
    rounds_data = load_all_rounds_with_gt()
    if len(rounds_data) < 3:
        log("  Not enough rounds with GT for tuning")
        return None

    log(f"  Backtesting on {len(rounds_data)} rounds")

    # 1. Tune RUIN_FLOOR
    best_floor, best_kl = backtest_ruin_floor(rounds_data)

    # Get current baseline KL
    baseline_kl = 0
    count = 0
    for rnum, W, H, seed_data in rounds_data:
        for si, gt, pred, grid in seed_data:
            baseline_kl += compute_kl(gt, pred)
            count += 1
    baseline_kl /= count if count > 0 else 1

    log(f"  Baseline avg KL: {baseline_kl:.5f}")
    log(f"  Best RUIN_FLOOR={best_floor:.3f} → avg KL={best_kl:.5f}")

    changes = {}
    # Only apply if it's an improvement
    current_floor = _read_current_ruin_floor()
    if best_floor != current_floor:
        improvement = (baseline_kl - best_kl) / baseline_kl * 100
        if improvement > 0.5:  # at least 0.5% improvement
            changes["RUIN_FLOOR"] = best_floor
            log(f"  → Will update RUIN_FLOOR: {current_floor} → {best_floor} "
                f"({improvement:+.1f}%)")
        else:
            log(f"  → RUIN_FLOOR change too small ({improvement:+.1f}%), keeping {current_floor}")
    else:
        log(f"  → RUIN_FLOOR already optimal at {current_floor}")

    return changes if changes else None


def _read_current_ruin_floor():
    """Read current RUIN_FLOOR from strategy.py."""
    with open("strategy.py", "r") as f:
        for line in f:
            if "RUIN_FLOOR" in line and "=" in line and not line.strip().startswith("#"):
                try:
                    val = float(line.split("=")[1].strip())
                    return val
                except (ValueError, IndexError):
                    pass
    return 0.010  # default


def apply_changes(changes):
    """Apply parameter changes to strategy.py."""
    if not changes:
        return

    with open("strategy.py", "r") as f:
        content = f.read()

    if "RUIN_FLOOR" in changes:
        old_val = _read_current_ruin_floor()
        old_str = f"RUIN_FLOOR = {old_val}"
        new_str = f"RUIN_FLOOR = {changes['RUIN_FLOOR']}"
        if old_str in content:
            content = content.replace(old_str, new_str)
            log(f"  Applied RUIN_FLOOR: {old_val} → {changes['RUIN_FLOOR']}")
        else:
            log(f"  WARNING: Could not find '{old_str}' in strategy.py")

    with open("strategy.py", "w") as f:
        f.write(content)


# ═══════════════════════════════════════════════════════════════════════
#  SUBMIT: Run full prediction pipeline for an active round
# ═══════════════════════════════════════════════════════════════════════

def submit_round(session, round_info):
    """Run the full prediction pipeline: sim priors, adaptive queries, submit."""
    rid = round_info["id"]
    rnum = round_info["round_number"]
    log(f"=== SUBMITTING Round #{rnum} ===")

    # Round details
    detail = session.get(f"{BASE}/rounds/{rid}").json()
    W, H, seeds = detail["map_width"], detail["map_height"], detail["seeds_count"]
    data_store.save_round_detail(rnum, detail)
    log(f"  Map {W}x{H}, {seeds} seeds")

    # Budget
    budget = session.get(f"{BASE}/budget").json()
    used = budget.get("queries_used", 0)
    total = budget.get("queries_max", 50)
    remaining = total - used
    log(f"  Budget: {used}/{total} used, {remaining} remaining")

    if remaining <= 0:
        log("  No queries remaining — running resubmit instead")
        resubmit_round(session, round_info)
        return

    # Historical calibration
    model = OutcomeModel()
    n_hist = calibrate_from_history(session, BASE, model)
    log(f"  Historical calibration: {n_hist} seed-rounds, {len(model.counts)} buckets")

    # Analyze seeds
    seed_info = analyze_seeds(detail, seeds)

    # Simulator priors
    log("  Computing simulator priors (100 sims/seed)...")
    sim_priors = {}
    for si in range(seeds):
        state = detail["initial_states"][si]
        setts = [s for s in state["settlements"] if s.get("alive", True)]
        sim_priors[si] = compute_simulator_prior(
            state["grid"], setts, W, H, n_sims=100
        )
        n_dynamic = int((sim_priors[si].max(axis=2) < 0.95).sum())
        log(f"    Seed {si}: {n_dynamic} dynamic cells")

    # Load saved observations or start fresh
    obs, obs_n, had_saved = data_store.load_observations(rnum, seeds, H, W, NUM_CLASSES)
    if had_saved:
        total_saved = sum(int((obs_n[si] > 0).sum()) for si in range(seeds))
        log(f"  Resumed {total_saved} previously observed cells")
    settlement_stats = {}

    # Baseline submission (free)
    log("  Submitting baseline (sim prior only)...")
    for si in range(seeds):
        pred = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W,
                                sim_prior=sim_priors[si])
        resp = session.post(f"{BASE}/submit", json={
            "round_id": rid, "seed_index": si,
            "prediction": pred.tolist(),
        })
        time.sleep(0.5)
    log("  Baseline submitted")

    # Adaptive queries with periodic resubmission
    submit_count = [0]

    def submit_all(query_count):
        submit_count[0] += 1
        regime = estimate_settlement_regime(obs, obs_n, seed_info, seeds, H, W)
        rate_str = f"{regime['observed_rate']:.1%}" if regime['observed_rate'] else "N/A"
        log(f"  Resubmit #{submit_count[0]} after {query_count}q "
            f"(rate={rate_str}, scale={regime['scale']:.2f})")
        for sj in range(seeds):
            pred = build_prediction(seed_info[sj], obs[sj], obs_n[sj], model, H, W,
                                    sim_prior=sim_priors[sj],
                                    settlement_stats=settlement_stats.get(sj),
                                    regime=regime)
            resp = session.post(f"{BASE}/submit", json={
                "round_id": rid, "seed_index": sj,
                "prediction": pred.tolist(),
            })
            time.sleep(0.5)

    log(f"  Running {remaining} adaptive queries...")
    total_q = execute_adaptive_queries(
        session, BASE, rid, seed_info,
        obs, obs_n, model, sim_priors,
        seeds, H, W, budget=remaining, delay=0.15,
        submit_fn=submit_all, submit_every=10,
        query_log_fn=lambda qn, si, vp, res: data_store.append_query(
            rnum, qn, si, vp, res),
        settlement_stats=settlement_stats,
    )
    log(f"  Queries executed: {total_q}")
    data_store.save_observations(rnum, obs, obs_n, seeds)

    # Final submission with self-consistency tuning
    regime = estimate_settlement_regime(obs, obs_n, seed_info, seeds, H, W)
    base_scale = regime["scale"]
    regime = self_consistency_tune(seed_info, obs, obs_n, model, seeds, H, W, regime)
    log(f"  Final scale: {base_scale:.2f} → {regime['scale']:.2f}")

    for si in range(seeds):
        pred = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W,
                                sim_prior=sim_priors[si],
                                settlement_stats=settlement_stats.get(si),
                                regime=regime)
        data_store.save_prediction(rnum, si, pred)
        resp = session.post(f"{BASE}/submit", json={
            "round_id": rid, "seed_index": si,
            "prediction": pred.tolist(),
        })
        n_obs = int((obs_n[si] > 0).sum())
        log(f"    Seed {si}: {resp.status_code} | {n_obs} cells observed")
        time.sleep(0.5)

    log(f"  Round #{rnum} submission complete!")


def resubmit_round(session, round_info):
    """Resubmit with zero queries — uses saved observations + sim priors."""
    rid = round_info["id"]
    rnum = round_info["round_number"]
    log(f"  Resubmitting Round #{rnum} (0 queries)")

    detail = session.get(f"{BASE}/rounds/{rid}").json()
    W, H, seeds = detail["map_width"], detail["map_height"], detail["seeds_count"]

    model = OutcomeModel()
    calibrate_from_history(session, BASE, model)
    seed_info = analyze_seeds(detail, seeds)

    sim_priors = {}
    for si in range(seeds):
        state = detail["initial_states"][si]
        setts = [s for s in state["settlements"] if s.get("alive", True)]
        sim_priors[si] = compute_simulator_prior(
            state["grid"], setts, W, H, n_sims=100, ensemble=True
        )

    obs, obs_n, loaded = data_store.load_observations(rnum, seeds, H, W, NUM_CLASSES)
    regime = estimate_settlement_regime(obs, obs_n, seed_info, seeds, H, W)
    if loaded:
        regime = self_consistency_tune(seed_info, obs, obs_n, model, seeds, H, W, regime)

    for si in range(seeds):
        pred = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W,
                                sim_prior=sim_priors[si], regime=regime)
        data_store.save_prediction(rnum, si, pred)
        resp = session.post(f"{BASE}/submit", json={
            "round_id": rid, "seed_index": si,
            "prediction": pred.tolist(),
        })
        log(f"    Seed {si}: {resp.status_code}")
        time.sleep(0.5)

    log(f"  Resubmission complete")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("OVERNIGHT AUTONOMOUS HANDLER — Starting")
    log("=" * 70)

    # Load previously submitted/evaluated rounds from disk
    state_file = "overnight_state.json"
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        submitted_rounds.update(state.get("submitted", []))
        evaluated_rounds.update(state.get("evaluated", []))
        log(f"  Loaded state: {len(submitted_rounds)} submitted, "
            f"{len(evaluated_rounds)} evaluated")

    def save_state():
        with open(state_file, "w") as f:
            json.dump({
                "submitted": list(submitted_rounds),
                "evaluated": list(evaluated_rounds),
            }, f)

    cycle = 0
    while True:
        cycle += 1
        try:
            session = get_session()

            # Get all rounds
            rounds = session.get(f"{BASE}/rounds").json()
            active = [r for r in rounds if r["status"] == "active"]
            completed = [r for r in rounds if r["status"] == "completed"]

            # ── Step 1: Evaluate newly completed rounds ───────────────
            for rnd in completed:
                rid = rnd["id"]
                rnum = rnd["round_number"]
                if rid in evaluated_rounds:
                    continue

                log(f"\n{'─' * 60}")
                log(f"Round #{rnum} completed — evaluating...")

                eval_result = evaluate_round(session, rnd)
                evaluated_rounds.add(rid)
                save_state()

                if eval_result:
                    log(f"  Evaluation done: rank={eval_result.get('rank')}")

                    # ── Step 2: Auto-tune based on new data ───────────
                    changes = auto_tune(eval_result)
                    if changes:
                        log("  Applying parameter changes...")
                        apply_changes(changes)
                        log("  Parameters updated!")

                        # Reimport strategy module to pick up changes
                        import importlib
                        import strategy
                        importlib.reload(strategy)
                        # Re-import the functions we use
                        globals().update({
                            'build_prediction': strategy.build_prediction,
                            'analyze_seeds': strategy.analyze_seeds,
                            'calibrate_from_history': strategy.calibrate_from_history,
                            'compute_simulator_prior': strategy.compute_simulator_prior,
                            'execute_adaptive_queries': strategy.execute_adaptive_queries,
                            'estimate_settlement_regime': strategy.estimate_settlement_regime,
                            'self_consistency_tune': strategy.self_consistency_tune,
                            'OutcomeModel': strategy.OutcomeModel,
                        })
                        log("  Strategy module reloaded with new params")

            # ── Step 3: Submit for active rounds ──────────────────────
            for rnd in active:
                rid = rnd["id"]
                rnum = rnd["round_number"]
                if rid in submitted_rounds:
                    continue

                log(f"\n{'─' * 60}")
                log(f"Round #{rnum} is active — submitting...")

                submit_round(session, rnd)
                submitted_rounds.add(rid)
                save_state()
                log(f"Round #{rnum} submitted successfully!")

            # ── Sleep ─────────────────────────────────────────────────
            if cycle == 1:
                log(f"\nCycle {cycle} complete. Polling every {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log("\nStopped by user (Ctrl+C)")
            save_state()
            break
        except Exception as e:
            log(f"\n!!! ERROR in cycle {cycle}: {e}")
            log(traceback.format_exc())
            log(f"Sleeping 60s before retry...")
            time.sleep(60)


if __name__ == "__main__":
    main()
