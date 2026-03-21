"""Validate observation-adaptive prior scaling on all GT rounds.

Simulates what would happen if we had observations covering ~50% of cells
and tested with/without the round-adaptive scaling.
"""
import numpy as np, json, os
from strategy import (
    domain_prior, compute_features, NUM_CLASSES, MIN_PROB,
    OutcomeModel, build_prediction, _load_bias_data,
)

def kl_score(gt, pred, grid_init, eps=1e-10):
    H, W = gt.shape[:2]
    total_ent = total_kl = 0.0
    for y in range(H):
        for x in range(W):
            if grid_init[y, x] in (10, 5):
                continue
            g = gt[y, x]
            ent = -np.sum(np.where(g > eps, g * np.log(g), 0))
            if ent < 0.001:
                continue
            p = np.clip(pred[y, x], eps, 1.0)
            p /= p.sum()
            kl = np.sum(np.where(g > eps, g * np.log(g / p), 0))
            total_ent += ent
            total_kl += kl
    return total_ent - total_kl, total_ent

print("=== Observation-Adaptive Prior Validation ===\n")
print(f"{'Round':>6} {'sett%':>6} {'no-obs':>8} {'with-obs':>9} {'adaptive':>9} {'max':>8} {'adapt-gain':>11}")
print("-" * 70)

total_noobs = 0
total_withobs = 0
total_adaptive = 0
total_max = 0

for rnd in range(1, 13):
    rdir = f"data/round_{rnd:02d}"
    detail_path = os.path.join(rdir, "round_detail.json")
    if not os.path.exists(detail_path):
        continue
    gt_path = os.path.join(rdir, "analysis", "seed_0_ground_truth.npy")
    if not os.path.exists(gt_path):
        continue

    detail = json.load(open(detail_path))
    W, H = detail["map_width"], detail["map_height"]
    gt = np.load(gt_path)
    state = detail["initial_states"][0]
    grid_init = np.array(state["grid"], dtype=int)
    setts = [s for s in state["settlements"] if s.get("alive", True)]
    feats = compute_features(state["grid"], setts, W, H)

    # GT settlement rate
    n_dynamic = 0
    sett_gt = 0
    for y in range(H):
        for x in range(W):
            if grid_init[y, x] not in (10, 5):
                n_dynamic += 1
                sett_gt += gt[y, x, 1]
    sett_pct = 100 * sett_gt / n_dynamic

    seed_info = {
        "grid": state["grid"],
        "settlements": setts,
        "features": feats,
        "viewports": [],
    }

    # Create empty model
    model = OutcomeModel()

    # Simulate observations: randomly observe ~50% of dynamic cells
    # Use GT as the observation source (one sample per observation)
    rng = np.random.RandomState(42)
    obs_si = np.zeros((H, W, NUM_CLASSES))
    obs_n_si = np.zeros((H, W))

    # Create a realistic observation pattern (viewports near settlements)
    observed_mask = np.zeros((H, W), dtype=bool)
    for s in setts:
        # Mark a 10x10 area around each settlement as "observed"
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                ny, nx = s["y"] + dy, s["x"] + dx
                if 0 <= ny < H and 0 <= nx < W:
                    observed_mask[ny, nx] = True

    for y in range(H):
        for x in range(W):
            if grid_init[y, x] in (5, 10):
                continue
            if observed_mask[y, x]:
                # Sample from GT distribution
                gt_dist = gt[y, x]
                cls = rng.choice(NUM_CLASSES, p=gt_dist)
                obs_si[y, x, cls] += 1
                obs_n_si[y, x] += 1
                model.observe(tuple(feats[y, x]), cls)

    n_obs = int((obs_n_si > 0).sum())

    # Test 1: No observations (prior only)
    obs_empty = np.zeros((H, W, NUM_CLASSES))
    obs_n_empty = np.zeros((H, W))
    model_empty = OutcomeModel()
    pred_noobs = build_prediction(seed_info, obs_empty, obs_n_empty, model_empty, H, W)
    s_noobs, max_s = kl_score(gt, pred_noobs, grid_init)

    # Test 2: With observations (current build_prediction, has adaptive scaling)
    pred_withobs = build_prediction(seed_info, obs_si, obs_n_si, model, H, W)
    s_withobs, _ = kl_score(gt, pred_withobs, grid_init)

    # Test 3: With observations but force no round scaling (monkey-patch)
    # Temporarily set obs to look like < 30 cells observed, then restore
    # Actually, let's just compute with a modified obs_n that underreports
    # Instead, let's manually build prediction without scaling
    pred_basic = np.zeros((H, W, NUM_CLASSES))
    for y in range(H):
        for x in range(W):
            cell = grid_init[y, x]
            if cell == 5:
                p = np.full(NUM_CLASSES, MIN_PROB)
                p[5] = 1.0
            elif cell == 10:
                p = np.full(NUM_CLASSES, MIN_PROB)
                p[0] = 1.0
            else:
                ic, db, co, dn = feats[y, x]
                fkey = tuple(feats[y, x])
                prior = domain_prior(ic, db, co)
                model_pred = model.predict(fkey, prior)
                n = obs_n_si[y, x]
                if n >= 8:
                    emp = obs_si[y, x] / n
                    w = n / (n + 2)
                    p = w * emp + (1 - w) * model_pred
                elif n > 0:
                    emp = obs_si[y, x] / n
                    w = n / (n + 8)
                    p = w * emp + (1 - w) * model_pred
                else:
                    p = model_pred
            pred_basic[y, x] = p
    pred_basic = np.maximum(pred_basic, MIN_PROB)
    pred_basic /= pred_basic.sum(axis=2, keepdims=True)
    s_basic, _ = kl_score(gt, pred_basic, grid_init)

    gain = s_withobs - s_basic
    print(f"R{rnd:2d}    {sett_pct:5.1f}  {s_noobs:8.1f} {s_basic:9.1f} {s_withobs:9.1f} {max_s:8.1f}   {gain:+9.1f}")

    total_noobs += s_noobs
    total_withobs += s_withobs
    total_adaptive += s_basic
    total_max += max_s

print("-" * 70)
print(f"{'Total':>6}        {total_noobs:8.1f} {total_adaptive:9.1f} {total_withobs:9.1f} {total_max:8.1f}   {total_withobs-total_adaptive:+9.1f}")
print(f"\nAdaptive scaling adds {total_withobs - total_adaptive:+.1f} points across {12} rounds")
print(f"  (no-obs: {total_noobs:.1f}, basic+obs: {total_adaptive:.1f}, adaptive+obs: {total_withobs:.1f}, max: {total_max:.1f})")
