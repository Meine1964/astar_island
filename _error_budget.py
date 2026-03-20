"""Error budget analysis: quantify where score is lost and what can be improved.

Computes per-cell KL divergence loss and breaks it down by:
1. Distance from settlements
2. Class (what GT says vs what we predict)
3. Observed vs unobserved cells
4. Sim-prior-only vs observations-available
"""
import numpy as np
import json
import os
from astar_island_simulator.env import (
    AstarIslandSimulator, HiddenParams, Settlement as SimSettlement,
    CODE_TO_CLASS, NUM_CLASSES,
)
from strategy import (
    compute_features, domain_prior, OutcomeModel, compute_simulator_prior,
    build_prediction, _load_bias_data, MIN_PROB,
)

class_names = ["plains", "sett", "port", "ruin", "forest", "mtn"]

def kl_per_cell(gt, pred, eps=1e-10):
    """Compute KL divergence per cell: sum_c gt[c] * log(gt[c] / pred[c])."""
    gt_safe = np.clip(gt, eps, 1.0)
    pred_safe = np.clip(pred, eps, 1.0)
    return np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=-1)

def entropy_per_cell(gt, eps=1e-10):
    """Compute entropy per cell."""
    gt_safe = np.clip(gt, eps, 1.0)
    return -np.sum(gt_safe * np.log(gt_safe), axis=-1)


# Load all completed rounds with GT
print("=== Error Budget Analysis ===\n")
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
        entries.append((rnd, si, detail, gt))

print(f"Loaded {len(entries)} seed-rounds\n")

# Aggregate across all seeds
total_kl_by_source = {
    "sim_only": 0.0,
    "optimal": 0.0,    # theoretical best (GT itself)
    "uniform": 0.0,    # uniform prediction
    "domain_prior": 0.0,
}
total_kl_by_dist = {}  # dist_bucket -> total KL
total_kl_by_gt_class = np.zeros(NUM_CLASSES)
total_entropy = 0.0
total_cells = 0
total_kl_breakdown = {  # per-class contribution to KL
    "sim": np.zeros(NUM_CLASSES),
    "domain": np.zeros(NUM_CLASSES),
}

# Also track per-cell error patterns
error_patterns = []  # list of (kl, gt_argmax, sim_argmax, dist_bucket)

for rnd, si, detail, gt in entries[:20]:  # limit for speed
    state = detail["initial_states"][si]
    W, H = detail["map_width"], detail["map_height"]
    grid_init = np.array(state["grid"], dtype=int)
    setts = [s for s in state["settlements"] if s.get("alive", True)]

    # Compute features
    features = compute_features(state["grid"], setts, W, H)

    # Compute sim prior
    sett_objs = []
    for i, s in enumerate(setts):
        if not s.get("alive", True):
            continue
        sett_objs.append(SimSettlement(
            x=s["x"], y=s["y"], population=10, food=5.0, wealth=1.0,
            defense=3.0, tech_level=1.0,
            has_port=s.get("has_port", False), owner_id=i,
        ))

    sim = AstarIslandSimulator.__new__(AstarIslandSimulator)
    sim.map_seed = 0
    sim.params = HiddenParams()
    sim.width = W
    sim.height = H
    sim.base_grid = grid_init
    sim.base_settlements = sett_objs

    sim_counts = np.zeros((H, W, NUM_CLASSES))
    n_sims = 50
    for seed in range(n_sims):
        final_grid, _ = sim.run(sim_seed=seed)
        for y in range(H):
            for x in range(W):
                cls = CODE_TO_CLASS.get(int(final_grid[y, x]), 0)
                sim_counts[y, x, cls] += 1
    sim_prior = sim_counts / n_sims
    # Apply MIN_PROB floor
    sim_prior = np.clip(sim_prior, MIN_PROB, None)
    sim_prior /= sim_prior.sum(axis=-1, keepdims=True)

    for y in range(H):
        for x in range(W):
            ic = int(grid_init[y, x])
            if ic in (10, 5):  # static
                continue

            gt_cell = gt[y, x]
            ent = entropy_per_cell(gt_cell)
            if ent < 0.01:  # near-deterministic cells contribute little
                continue

            total_entropy += ent
            total_cells += 1

            _, db, coastal, _ = features[y][x]

            # Sim-only prediction
            sim_cell = sim_prior[y, x]
            kl_sim = float(kl_per_cell(gt_cell, sim_cell))
            total_kl_by_source["sim_only"] += kl_sim * ent

            # Domain prior prediction
            dp = domain_prior(ic, db, coastal)
            dp = np.clip(dp, MIN_PROB, None)
            dp /= dp.sum()
            kl_dp = float(kl_per_cell(gt_cell, dp))
            total_kl_by_source["domain_prior"] += kl_dp * ent

            # Uniform prediction
            uniform = np.ones(NUM_CLASSES) / NUM_CLASSES
            kl_uni = float(kl_per_cell(gt_cell, uniform))
            total_kl_by_source["uniform"] += kl_uni * ent

            # Track by distance bucket
            if db not in total_kl_by_dist:
                total_kl_by_dist[db] = {"sim": 0.0, "domain": 0.0, "n": 0, "entropy": 0.0}
            total_kl_by_dist[db]["sim"] += kl_sim * ent
            total_kl_by_dist[db]["domain"] += kl_dp * ent
            total_kl_by_dist[db]["n"] += 1
            total_kl_by_dist[db]["entropy"] += ent

            # Track by GT argmax class
            gt_cls = int(np.argmax(gt_cell))
            total_kl_by_gt_class[gt_cls] += kl_sim * ent

            # Per-class KL contribution
            for c in range(NUM_CLASSES):
                if gt_cell[c] > 0.01:
                    c_kl = gt_cell[c] * np.log(gt_cell[c] / max(sim_cell[c], 1e-10))
                    total_kl_breakdown["sim"][c] += c_kl * ent
                    c_kl_dp = gt_cell[c] * np.log(gt_cell[c] / max(dp[c], 1e-10))
                    total_kl_breakdown["domain"][c] += c_kl_dp * ent

            sim_cls = int(np.argmax(sim_cell))
            if gt_cls != sim_cls:
                error_patterns.append((kl_sim, gt_cls, sim_cls, db))

# Print results
print(f"Total entropy-weighted cells: {total_cells}")
print(f"Total entropy: {total_entropy:.1f}")
print()

print("=== Prediction Method Comparison (entropy-weighted KL) ===")
print(f"  Score = max_score - KL_loss")
print(f"  Max possible score (perfect): {total_entropy:.1f}")
for method, kl in sorted(total_kl_by_source.items(), key=lambda x: x[1]):
    score = total_entropy - kl
    pct = 100 * score / total_entropy
    print(f"  {method:20s}: KL_loss={kl:8.1f}  score={score:8.1f}  ({pct:.1f}% of max)")

print()
print("=== KL Loss by Distance Bucket (sim-only) ===")
for db in sorted(total_kl_by_dist.keys()):
    d = total_kl_by_dist[db]
    print(f"  Bucket {db}: KL_loss={d['sim']:.1f}  ({100*d['sim']/total_kl_by_source['sim_only']:.1f}% of total)  "
          f"n={d['n']}  avg_entropy={d['entropy']/d['n']:.3f}")

print()
print("=== KL Loss by GT Dominant Class (sim-only) ===")
for c in range(NUM_CLASSES):
    if total_kl_by_gt_class[c] > 0.1:
        pct = 100 * total_kl_by_gt_class[c] / total_kl_by_source["sim_only"]
        print(f"  {class_names[c]:10s}: KL_loss={total_kl_by_gt_class[c]:8.1f}  ({pct:.1f}% of total)")

print()
print("=== Per-Class KL Contribution (where GT probability is mispredicted) ===")
print(f"  {'Class':10s} {'Sim-only':>10s} {'Domain-prior':>12s}")
for c in range(NUM_CLASSES):
    s = total_kl_breakdown["sim"][c]
    d = total_kl_breakdown["domain"][c]
    if s > 0.1 or d > 0.1:
        print(f"  {class_names[c]:10s} {s:10.1f} {d:12.1f}")

print()
print("=== Top Error Patterns (sim argmax != GT argmax) ===")
from collections import Counter
pattern_counts = Counter((p[1], p[2]) for p in error_patterns)
for (gt_c, sim_c), count in pattern_counts.most_common(10):
    avg_kl = np.mean([p[0] for p in error_patterns if p[1] == gt_c and p[2] == sim_c])
    print(f"  GT={class_names[gt_c]:7s} Sim={class_names[sim_c]:7s}: {count:4d} cells, avg_KL={avg_kl:.4f}")

print()
print("=== Improvement Potential ===")
sim_score = total_entropy - total_kl_by_source["sim_only"]
dp_score = total_entropy - total_kl_by_source["domain_prior"]
print(f"  Current sim-only: {sim_score:.1f} / {total_entropy:.1f}")
print(f"  Domain prior: {dp_score:.1f} / {total_entropy:.1f}")
print(f"  Observations add {total_kl_by_source['sim_only'] - total_kl_by_source.get('sim_plus_obs', total_kl_by_source['sim_only']):.1f} score points")
print()
print("  The gap between sim-only and perfect is the improvement ceiling.")
print(f"  Improvement ceiling (sim-only): {total_kl_by_source['sim_only']:.1f} KL points")
print(f"  Of which observations + bias correction can recover some fraction.")
print(f"  Realistic target: recover 40-60% of gap through observations.")
