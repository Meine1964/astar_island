"""Backtest regime estimation across multiple rounds."""
import numpy as np
import json
import os
from strategy import (
    OutcomeModel, build_prediction, compute_features,
    estimate_settlement_regime, NUM_CLASSES, MIN_PROB,
)


def score_pred(pred, gt, H, W):
    eps = 1e-10
    total_kl = 0.0
    total_weight = 0.0
    for y in range(H):
        for x in range(W):
            gt_dist = gt[y, x]
            p_dist = pred[y, x]
            ent = -np.sum(gt_dist * np.log(gt_dist + eps))
            weight = max(ent, 0.1)
            kl = np.sum(gt_dist * np.log((gt_dist + eps) / (p_dist + eps)))
            total_kl += kl * weight
            total_weight += weight
    weighted_kl = total_kl / total_weight if total_weight > 0 else 0.0
    return max(0, 120 - weighted_kl * 200)


for rnd in [9, 10, 12, 13, 14]:
    obs_path = f"data/round_{rnd:02d}/observations.npz"
    detail_path = f"data/round_{rnd:02d}/round_detail.json"
    gt_path = f"data/round_{rnd:02d}/analysis/seed_0_ground_truth.npy"
    if not os.path.exists(obs_path) or not os.path.exists(gt_path):
        continue

    with open(detail_path) as f:
        detail = json.load(f)
    W, H = detail["map_width"], detail["map_height"]
    seeds = detail["seeds_count"]

    data = np.load(obs_path)
    obs = {}
    obs_n = {}
    for si in range(seeds):
        obs[si] = data[f"obs_{si}"]
        obs_n[si] = data[f"obs_n_{si}"]

    seed_info = []
    for si in range(seeds):
        state = detail["initial_states"][si]
        setts = [s for s in state["settlements"] if s.get("alive", True)]
        feats = compute_features(state["grid"], setts, W, H)
        seed_info.append({"grid": state["grid"], "settlements": setts, "features": feats})

    model = OutcomeModel()
    regime = estimate_settlement_regime(obs, obs_n, seed_info, seeds, H, W)

    total_old = 0
    total_new = 0
    for si in range(seeds):
        gt = np.load(f"data/round_{rnd:02d}/analysis/seed_{si}_ground_truth.npy")
        pred_old = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W)
        pred_new = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W, regime=regime)
        total_old += score_pred(pred_old, gt, H, W)
        total_new += score_pred(pred_new, gt, H, W)

    rate = regime.get("observed_rate")
    expected = regime.get("expected_rate")
    rate_str = f"{rate:.1%}" if rate is not None else "N/A"
    exp_str = f"{expected:.1%}" if expected is not None else "N/A"
    delta = total_new - total_old
    print(f"R{rnd:2d}: obs_rate={rate_str:>6s}  exp_rate={exp_str:>6s}  "
          f"scale={regime['scale']:.2f}  "
          f"old={total_old:.1f}  new={total_new:.1f}  delta={delta:+.1f}")
