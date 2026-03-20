"""Astar Island — simulator calibration from real round feedback.

Pulls completed-round data from the live API (observations + analysis)
and tunes the local simulator's HiddenParams to better match reality.

Workflow:
  1. Fetch all completed rounds and their initial states
  2. For each seed, collect observation statistics from our queries
  3. If analysis endpoint provides ground truth, compare directly
  4. Otherwise, compare class-frequency distributions in observed regions
  5. Search HiddenParams space to minimise divergence from real data
  6. Save calibrated params to calibrated_params.json

Usage:  uv run python calibrate.py
"""
import json
import time
import itertools
import requests
import numpy as np
import truststore
truststore.inject_into_ssl()

from astar_island_simulator.env import (
    AstarIslandSimulator, HiddenParams, CODE_TO_CLASS, NUM_CLASSES,
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
PARAMS_FILE = "calibrated_params.json"


# ── Helpers ────────────────────────────────────────────────────────────

def load_saved_params() -> HiddenParams:
    """Load previously calibrated params, or return defaults."""
    try:
        with open(PARAMS_FILE) as f:
            d = json.load(f)
        return HiddenParams(**{k: v for k, v in d.items()
                               if k in HiddenParams.__dataclass_fields__})
    except FileNotFoundError:
        return HiddenParams()


def save_params(params: HiddenParams):
    """Save calibrated params to JSON."""
    from dataclasses import asdict
    with open(PARAMS_FILE, "w") as f:
        json.dump(asdict(params), f, indent=2)
    print(f"Saved calibrated params to {PARAMS_FILE}")


def class_distribution(grid_2d, W, H):
    """Compute per-cell class from a 2D terrain grid."""
    dist = np.zeros(NUM_CLASSES)
    for y in range(H):
        for x in range(W):
            cls = CODE_TO_CLASS.get(int(grid_2d[y][x]) if isinstance(grid_2d[y][x], (int, float, np.integer))
                                    else int(grid_2d[y][x]), 0)
            dist[cls] += 1
    return dist / dist.sum()


def region_class_distribution(grid_2d, region, W, H):
    """Compute class distribution within a viewport region."""
    dist = np.zeros(NUM_CLASSES)
    rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
    for y in range(ry, min(ry + rh, H)):
        for x in range(rx, min(rx + rw, W)):
            cls = CODE_TO_CLASS.get(int(grid_2d[y][x]) if isinstance(grid_2d[y][x], (int, float, np.integer))
                                    else int(grid_2d[y][x]), 0)
            dist[cls] += 1
    total = dist.sum()
    return dist / total if total > 0 else dist


# ── Data collection ───────────────────────────────────────────────────

def fetch_round_data(session):
    """Fetch all completed rounds and their details."""
    rounds = session.get(f"{BASE}/rounds").json()
    completed = [r for r in rounds if r["status"] == "completed"]
    print(f"Found {len(completed)} completed round(s)")

    round_data = []
    for rnd in completed:
        rid = rnd["id"]
        rnum = rnd["round_number"]

        # Round details (initial states)
        detail = session.get(f"{BASE}/rounds/{rid}").json()
        W, H = detail["map_width"], detail["map_height"]
        seeds = detail["seeds_count"]

        # Our predictions and their scores
        try:
            my_preds = session.get(f"{BASE}/my-predictions/{rid}").json()
        except Exception:
            my_preds = None

        # Per-seed analysis (ground truth if available)
        seed_analyses = {}
        for si in range(seeds):
            try:
                resp = session.get(f"{BASE}/analysis/{rid}/{si}")
                time.sleep(0.25)
                if resp.status_code == 200:
                    seed_analyses[si] = resp.json()
            except Exception:
                pass

        rd = {
            "round_id": rid,
            "round_number": rnum,
            "detail": detail,
            "W": W, "H": H, "seeds": seeds,
            "my_predictions": my_preds,
            "analyses": seed_analyses,
        }
        round_data.append(rd)
        print(f"  Round #{rnum}: {W}x{H}, {seeds} seeds, "
              f"{len(seed_analyses)} analyses fetched")

    return round_data


def extract_observation_stats(round_data):
    """Extract per-seed statistics we can compare against simulator output.

    Returns a list of dicts with:
      - initial grid/settlements
      - observed class frequencies (from analysis or predictions)
      - settlement survival rates, port counts, ruin counts, etc.
    """
    stats = []

    for rd in round_data:
        detail = rd["detail"]
        W, H = rd["W"], rd["H"]

        for si in range(rd["seeds"]):
            state = detail["initial_states"][si]
            init_grid = state["grid"]
            init_setts = state["settlements"]
            n_initial_alive = sum(1 for s in init_setts if s.get("alive", True))

            entry = {
                "round_number": rd["round_number"],
                "seed_index": si,
                "W": W, "H": H,
                "init_grid": init_grid,
                "init_settlements": init_setts,
                "n_initial_settlements": n_initial_alive,
                "ground_truth": None,
                "class_freq": None,
                "settlement_count": None,
                "port_count": None,
                "ruin_count": None,
            }

            # Try to extract ground truth from analysis
            analysis = rd["analyses"].get(si)
            if analysis:
                # Look for ground truth data in various possible formats
                gt = None
                for key in ["ground_truth", "truth", "ground_truth_grid", "gt", "actual",
                            "true_distribution", "distribution"]:
                    gt = analysis.get(key)
                    if gt is not None:
                        break

                if gt is not None:
                    try:
                        gt_arr = np.array(gt)
                        if gt_arr.ndim == 3 and gt_arr.shape == (H, W, NUM_CLASSES):
                            entry["ground_truth"] = gt_arr
                            # Extract marginal stats
                            argmax = gt_arr.argmax(axis=2)
                            entry["class_freq"] = np.bincount(
                                argmax.ravel(), minlength=NUM_CLASSES
                            ).astype(float) / (H * W)
                        elif gt_arr.ndim == 2 and gt_arr.shape == (H, W):
                            # Class grid
                            entry["class_freq"] = np.bincount(
                                gt_arr.ravel().astype(int), minlength=NUM_CLASSES
                            ).astype(float) / (H * W)
                    except Exception:
                        pass

                # Look for summary stats in analysis
                for key in ["settlement_count", "n_settlements", "settlements_alive"]:
                    val = analysis.get(key)
                    if val is not None:
                        entry["settlement_count"] = int(val)
                        break

                for key in ["port_count", "n_ports"]:
                    val = analysis.get(key)
                    if val is not None:
                        entry["port_count"] = int(val)
                        break

                for key in ["ruin_count", "n_ruins"]:
                    val = analysis.get(key)
                    if val is not None:
                        entry["ruin_count"] = int(val)
                        break

                # If we have score info, note it
                if "score" in analysis:
                    entry["real_score"] = analysis["score"]

            stats.append(entry)

    print(f"\nExtracted stats for {len(stats)} seed-round(s)")
    # Summarize what we got
    n_gt = sum(1 for s in stats if s["ground_truth"] is not None)
    n_freq = sum(1 for s in stats if s["class_freq"] is not None)
    print(f"  Ground truth distributions: {n_gt}")
    print(f"  Class frequency data: {n_freq}")

    return stats


# ── Simulator comparison ──────────────────────────────────────────────

def simulate_with_params(init_grid, init_setts, W, H, params, n_sims=50):
    """Run our simulator on a given initial state and measure outcomes.

    Since we can't set the exact map_seed to match the server's map
    (our generator differs), we instead:
    - Use the real initial grid and settlements
    - Run the simulation phases only
    - Compare outcome distributions
    """
    from astar_island_simulator.env import (
        Settlement, OCEAN, PLAINS, EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN
    )

    # Build numpy grid from the initial state
    grid_np = np.array(init_grid, dtype=int)

    # Build settlement objects
    settlements_base = []
    for i, s in enumerate(init_setts):
        if not s.get("alive", True):
            continue
        settlements_base.append(Settlement(
            x=s["x"], y=s["y"],
            population=10,  # unknown — use default
            food=5.0,
            wealth=1.0,
            defense=3.0,
            tech_level=1.0,
            has_port=s.get("has_port", False),
            owner_id=i,
        ))

    # Create a simulator but override its base state
    sim = AstarIslandSimulator.__new__(AstarIslandSimulator)
    sim.map_seed = 0
    sim.params = params
    sim.width = W
    sim.height = H
    sim.base_grid = grid_np
    sim.base_settlements = settlements_base

    # Run multiple simulations and aggregate
    class_counts = np.zeros((H, W, NUM_CLASSES))
    settlement_counts = []
    port_counts = []
    ruin_counts = []

    for i in range(n_sims):
        final_grid, final_setts = sim.run(sim_seed=1000 + i)
        for y in range(H):
            for x in range(W):
                cls = CODE_TO_CLASS.get(int(final_grid[y, x]), 0)
                class_counts[y, x, cls] += 1

        n_alive = sum(1 for s in final_setts if s.alive)
        n_ports = sum(1 for s in final_setts if s.alive and s.has_port)
        n_ruins = sum(1 for y in range(H) for x in range(W) if final_grid[y, x] == RUIN)
        settlement_counts.append(n_alive)
        port_counts.append(n_ports)
        ruin_counts.append(n_ruins)

    pred_dist = class_counts / n_sims
    return {
        "distribution": pred_dist,
        "class_freq": pred_dist.sum(axis=(0, 1)) / (H * W),
        "mean_settlements": np.mean(settlement_counts),
        "mean_ports": np.mean(port_counts),
        "mean_ruins": np.mean(ruin_counts),
    }


def compute_divergence(real_stats, sim_result):
    """Compute divergence between real observation data and simulator output.

    Returns a scalar loss: lower is better.
    """
    loss = 0.0
    n_comparisons = 0

    # Compare ground truth distribution if available
    if real_stats["ground_truth"] is not None:
        gt = real_stats["ground_truth"]
        pred = sim_result["distribution"]
        eps = 1e-10
        # Mean KL divergence over dynamic cells
        H, W = gt.shape[0], gt.shape[1]
        for y in range(H):
            for x in range(W):
                q = gt[y, x]
                p = pred[y, x]
                entropy = -np.sum(q * np.log(q + eps))
                if entropy > 0.01:  # dynamic cell
                    kl = np.sum(q * np.log((q + eps) / (p + eps)))
                    loss += kl
                    n_comparisons += 1

    # Compare class frequency distribution
    elif real_stats["class_freq"] is not None:
        real_freq = real_stats["class_freq"]
        sim_freq = sim_result["class_freq"]
        # Jensen-Shannon divergence
        m = 0.5 * (real_freq + sim_freq)
        eps = 1e-10
        js = 0.5 * np.sum(real_freq * np.log((real_freq + eps) / (m + eps)))
        js += 0.5 * np.sum(sim_freq * np.log((sim_freq + eps) / (m + eps)))
        loss += js * 1000  # scale up to be comparable
        n_comparisons += 1

    # Compare settlement/port/ruin counts if available
    if real_stats["settlement_count"] is not None:
        diff = abs(real_stats["settlement_count"] - sim_result["mean_settlements"])
        loss += diff * 2.0
        n_comparisons += 1

    if real_stats["port_count"] is not None:
        diff = abs(real_stats["port_count"] - sim_result["mean_ports"])
        loss += diff * 2.0
        n_comparisons += 1

    if real_stats["ruin_count"] is not None:
        diff = abs(real_stats["ruin_count"] - sim_result["mean_ruins"])
        loss += diff * 1.0
        n_comparisons += 1

    return loss, n_comparisons


# ── Parameter search ──────────────────────────────────────────────────

def build_param_grid():
    """Define the parameter search space.

    We focus on the parameters most likely to differ from our defaults
    and that most affect the outcome distribution.
    """
    return {
        "winter_base_severity": [1.0, 1.5, 2.0, 2.5, 3.0],
        "winter_severity_variance": [1.0, 1.5, 2.0],
        "collapse_food_threshold": [-5.0, -3.0, -1.0, 0.0],
        "food_per_forest": [2.0, 3.0, 4.0],
        "food_per_plains": [0.5, 1.0, 1.5],
        "desperate_food_threshold": [2.0, 3.0, 5.0],
        "raid_strength_factor": [0.2, 0.3, 0.4],
        "forest_regrowth_prob": [0.05, 0.10, 0.15, 0.20],
        "expansion_pop_threshold": [10, 15, 20],
    }


def calibrate(stats, base_params=None, n_sims_per_eval=15):
    """Two-phase calibration: coarse grid search then local refinement.

    Phase 1: Grid search over most impactful parameters (2-3 at a time)
    Phase 2: Coordinate descent on all parameters
    """
    if base_params is None:
        base_params = HiddenParams()

    # Subsample: pick one seed per round for speed
    seen_rounds = set()
    sampled = []
    for s in stats:
        rn = s["round_number"]
        if rn not in seen_rounds:
            seen_rounds.add(rn)
            sampled.append(s)
    print(f"Using {len(sampled)} seed-rounds (1 per round) for calibration")
    stats = sampled

    print("\n=== Phase 1: Coarse grid search ===")

    # Focus on the 3 most impactful parameters first
    coarse_grid = {
        "winter_base_severity": [1.0, 2.5, 4.0],
        "collapse_food_threshold": [-5.0, -2.0, 0.0],
        "food_per_forest": [2.0, 3.0, 4.0],
    }

    best_loss = float("inf")
    best_params = base_params
    combos = list(itertools.product(*coarse_grid.values()))
    print(f"Testing {len(combos)} parameter combinations...")

    for idx, combo in enumerate(combos):
        from dataclasses import replace
        trial = replace(base_params,
                        winter_base_severity=combo[0],
                        collapse_food_threshold=combo[1],
                        food_per_forest=combo[2])

        total_loss = 0.0
        total_comparisons = 0
        for s in stats:
            result = simulate_with_params(
                s["init_grid"], s["init_settlements"],
                s["W"], s["H"], trial, n_sims=n_sims_per_eval
            )
            loss, nc = compute_divergence(s, result)
            total_loss += loss
            total_comparisons += nc

        avg_loss = total_loss / max(total_comparisons, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = trial
            print(f"  [{idx+1}/{len(combos)}] New best: loss={avg_loss:.4f} "
                  f"(winter={combo[0]}, collapse={combo[1]}, food_forest={combo[2]})")

    print(f"\nPhase 1 best loss: {best_loss:.4f}")

    # Phase 2: Coordinate descent on finer grid
    print("\n=== Phase 2: Local refinement ===")
    fine_grid = build_param_grid()
    improved = True
    iteration = 0

    while improved and iteration < 2:
        improved = False
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        for param_name, values in fine_grid.items():
            current_val = getattr(best_params, param_name)
            best_val = current_val

            for val in values:
                if val == current_val:
                    continue

                from dataclasses import replace
                trial = replace(best_params, **{param_name: val})
                total_loss = 0.0
                total_comparisons = 0

                for s in stats:
                    result = simulate_with_params(
                        s["init_grid"], s["init_settlements"],
                        s["W"], s["H"], trial, n_sims=n_sims_per_eval
                    )
                    loss, nc = compute_divergence(s, result)
                    total_loss += loss
                    total_comparisons += nc

                avg_loss = total_loss / max(total_comparisons, 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_val = val
                    improved = True

            if best_val != current_val:
                from dataclasses import replace
                best_params = replace(best_params, **{param_name: best_val})
                print(f"  {param_name}: {current_val} -> {best_val} (loss={best_loss:.4f})")

    return best_params, best_loss


# ── Report ────────────────────────────────────────────────────────────

def print_comparison(stats, params, n_sims=30):
    """Print a detailed comparison table for the calibrated simulator."""
    print("\n=== Calibration results ===\n")
    from dataclasses import asdict
    d = asdict(params)
    defaults = asdict(HiddenParams())
    changed = {k: v for k, v in d.items() if v != defaults[k]}
    if changed:
        print("Parameters changed from defaults:")
        for k, v in changed.items():
            print(f"  {k}: {defaults[k]} -> {v}")
    else:
        print("All parameters at defaults (no real data to calibrate against)")

    print(f"\n{'Round':>6} {'Seed':>4} | {'Real freq':>40} | {'Sim freq':>40} | {'Loss':>8}")
    print("-" * 110)

    for s in stats:
        result = simulate_with_params(
            s["init_grid"], s["init_settlements"],
            s["W"], s["H"], params, n_sims=n_sims
        )
        loss, _ = compute_divergence(s, result)

        real_freq = s["class_freq"]
        sim_freq = result["class_freq"]

        real_str = ", ".join(f"{v:.3f}" for v in real_freq) if real_freq is not None else "N/A"
        sim_str = ", ".join(f"{v:.3f}" for v in sim_freq)

        print(f"  R{s['round_number']:>4} S{s['seed_index']:>2} | "
              f"{real_str:>40} | {sim_str:>40} | {loss:>8.4f}")

        # Settlement stats
        real_sc = s.get("settlement_count")
        if real_sc is not None:
            print(f"{'':>14} | settlements: real={real_sc}, sim={result['mean_settlements']:.1f} | "
                  f"ports: sim={result['mean_ports']:.1f} | ruins: sim={result['mean_ruins']:.1f}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    session = requests.Session()
    session.cookies.set("access_token", TOKEN)

    print("=== Astar Island Simulator Calibration ===\n")

    # 1. Fetch data
    round_data = fetch_round_data(session)
    if not round_data:
        print("\nNo completed rounds yet. Run this after a round finishes.")
        print("In the meantime, using default parameters.")
        save_params(HiddenParams())
        return

    # 2. Extract observation stats
    stats = extract_observation_stats(round_data)
    usable = [s for s in stats
              if s["ground_truth"] is not None or s["class_freq"] is not None
              or s["settlement_count"] is not None]

    if not usable:
        print("\nNo usable comparison data found in analyses.")
        print("The analysis endpoints may not return ground truth yet.")
        print("Saving default parameters for now.")
        save_params(HiddenParams())

        # Still run comparison with defaults so we see the simulator's output
        print("\n--- Simulator output with defaults (for reference) ---")
        for s in stats[:3]:  # show first 3
            result = simulate_with_params(
                s["init_grid"], s["init_settlements"],
                s["W"], s["H"], HiddenParams(), n_sims=30
            )
            print(f"  R{s['round_number']} S{s['seed_index']}: "
                  f"settlements={result['mean_settlements']:.1f}, "
                  f"ports={result['mean_ports']:.1f}, "
                  f"ruins={result['mean_ruins']:.1f}, "
                  f"freq=[{', '.join(f'{v:.3f}' for v in result['class_freq'])}]")
        return

    # 3. Calibrate
    base = load_saved_params()
    print(f"\nStarting calibration with {len(usable)} usable seed-round(s)...")
    best_params, best_loss = calibrate(usable, base_params=base)

    # 4. Report
    print_comparison(usable, best_params)

    # 5. Save
    save_params(best_params)
    print(f"\nFinal loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
