"""Backtest new regime rescaling against R14 GT data."""
import numpy as np
import json
from strategy import (
    OutcomeModel, build_prediction, compute_features,
    estimate_settlement_regime, NUM_CLASSES, MIN_PROB,
)

# Load R14 data
with open("data/round_14/round_detail.json") as f:
    detail = json.load(f)

W, H = detail["map_width"], detail["map_height"]
seeds = detail["seeds_count"]

# Load observations
data = np.load("data/round_14/observations.npz")
obs = {}
obs_n = {}
for si in range(seeds):
    obs[si] = data[f"obs_{si}"]
    obs_n[si] = data[f"obs_n_{si}"]

# Build seed info
seed_info = []
for si in range(seeds):
    state = detail["initial_states"][si]
    setts = [s for s in state["settlements"] if s.get("alive", True)]
    feats = compute_features(state["grid"], setts, W, H)
    seed_info.append({"grid": state["grid"], "settlements": setts, "features": feats})

model = OutcomeModel()

# Estimate regime from R14 observations
regime = estimate_settlement_regime(obs, obs_n, seed_info, seeds, H, W)
rate = regime["observed_rate"]
print(f"Regime: rate={rate:.1%}, scale={regime['scale']:.2f}, cells={regime['observed_cells']}")


def score_pred(pred, gt):
    """Approximate the competition scoring."""
    eps = 1e-10
    total_kl = 0.0
    total_weight = 0.0
    for y in range(H):
        for x in range(W):
            gt_dist = gt[y, x]
            p_dist = pred[y, x]
            # Entropy of GT
            ent = -np.sum(gt_dist * np.log(gt_dist + eps))
            weight = max(ent, 0.1)
            # KL divergence
            kl = np.sum(gt_dist * np.log((gt_dist + eps) / (p_dist + eps)))
            total_kl += kl * weight
            total_weight += weight
    weighted_kl = total_kl / total_weight if total_weight > 0 else 0.0
    return max(0, 120 - weighted_kl * 200)


print("\n=== WITHOUT regime (old approach) ===")
total_old = 0
for si in range(seeds):
    gt = np.load(f"data/round_14/analysis/seed_{si}_ground_truth.npy")
    pred_old = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W)
    s = score_pred(pred_old, gt)
    total_old += s
    print(f"  Seed {si}: {s:.1f}")
print(f"  Total: {total_old:.1f}")

print("\n=== WITH regime (new approach) ===")
total_new = 0
for si in range(seeds):
    gt = np.load(f"data/round_14/analysis/seed_{si}_ground_truth.npy")
    pred_new = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W,
                                regime=regime)
    s = score_pred(pred_new, gt)
    total_new += s
    print(f"  Seed {si}: {s:.1f}")
print(f"  Total: {total_new:.1f}")

print(f"\nImprovement: {total_new - total_old:+.1f} ({(total_new-total_old)/max(total_old,1)*100:+.1f}%)")

# Also test with sim priors
from strategy import compute_simulator_prior
print("\n=== WITH regime + sim priors ===")
total_sim = 0
for si in range(seeds):
    state = detail["initial_states"][si]
    setts = [s for s in state["settlements"] if s.get("alive", True)]
    sim_prior = compute_simulator_prior(state["grid"], setts, W, H, n_sims=100)
    gt = np.load(f"data/round_14/analysis/seed_{si}_ground_truth.npy")
    pred_sim = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W,
                                sim_prior=sim_prior, regime=regime)
    s = score_pred(pred_sim, gt)
    total_sim += s
    print(f"  Seed {si}: {s:.1f}")
print(f"  Total: {total_sim:.1f}")
print(f"  vs old: {total_sim - total_old:+.1f}")
