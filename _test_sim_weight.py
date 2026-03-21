"""Test different sim_weight values through the full build_prediction pipeline.

Uses simulated observations (from GT) to test the complete prediction flow.
"""
import numpy as np, json, os
from strategy import (
    domain_prior, compute_features, NUM_CLASSES, MIN_PROB,
    OutcomeModel, build_prediction,
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

print("=== sim_weight Optimization (full pipeline with observations) ===\n")

# Collect round data
round_data = []
for rnd in range(1, 13):
    rdir = f"data/round_{rnd:02d}"
    gt_path = os.path.join(rdir, "analysis", "seed_0_ground_truth.npy")
    if not os.path.exists(gt_path):
        continue
    detail = json.load(open(os.path.join(rdir, "round_detail.json")))
    W, H = detail["map_width"], detail["map_height"]
    gt = np.load(gt_path)
    state = detail["initial_states"][0]
    grid_init = np.array(state["grid"], dtype=int)
    setts = [s for s in state["settlements"] if s.get("alive", True)]
    feats = compute_features(state["grid"], setts, W, H)

    # Simulate observations near settlements
    rng = np.random.RandomState(42)
    obs_si = np.zeros((H, W, NUM_CLASSES))
    obs_n_si = np.zeros((H, W))
    model = OutcomeModel()
    
    for s in setts:
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                ny, nx = s["y"] + dy, s["x"] + dx
                if 0 <= ny < H and 0 <= nx < W and grid_init[ny, nx] not in (5, 10):
                    if obs_n_si[ny, nx] == 0:
                        cls = rng.choice(NUM_CLASSES, p=gt[ny, nx])
                        obs_si[ny, nx, cls] += 1
                        obs_n_si[ny, nx] += 1
                        model.observe(tuple(feats[ny, nx]), cls)

    # Create synthetic sim prior (flat ~11% settlement, similar to what our sim produces)
    sim_prior = np.zeros((H, W, 6))
    for y in range(H):
        for x in range(W):
            if grid_init[y, x] == 10:
                sim_prior[y, x] = [1.0, 0, 0, 0, 0, 0]
            elif grid_init[y, x] == 5:
                sim_prior[y, x] = [0, 0, 0, 0, 0, 1.0]
            else:
                ic = feats[y, x, 0]
                if ic == 4:
                    sim_prior[y, x] = [0.08, 0.11, 0.005, 0.01, 0.79, 0.005]
                elif ic == 1:
                    sim_prior[y, x] = [0.45, 0.28, 0.005, 0.025, 0.23, 0.005]
                else:
                    sim_prior[y, x] = [0.62, 0.11, 0.005, 0.01, 0.25, 0.005]

    sett_gt = sum(gt[y, x, 1] for y in range(H) for x in range(W) if grid_init[y, x] not in (10, 5))
    n_dyn = sum(1 for y in range(H) for x in range(W) if grid_init[y, x] not in (10, 5))
    
    round_data.append({
        "rnd": rnd, "gt": gt, "grid_init": grid_init, "feats": feats,
        "obs_si": obs_si, "obs_n_si": obs_n_si, "model": model,
        "sim_prior": sim_prior, "sett_pct": 100*sett_gt/n_dyn,
        "seed_info": {"grid": state["grid"], "settlements": setts,
                      "features": feats, "viewports": []},
        "W": W, "H": H,
    })

# Test different sim weights by patching build_prediction
import strategy

original_code = strategy.build_prediction.__code__

sim_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
weight_scores = {w: 0 for w in sim_weights}
weight_scores_per_round = {w: [] for w in sim_weights}

for sw in sim_weights:
    total = 0
    for rd in round_data:
        # Manually build prediction with custom sim_weight
        grid = rd["seed_info"]["grid"]
        feats = rd["feats"]
        obs_si = rd["obs_si"]
        obs_n_si = rd["obs_n_si"]
        model = rd["model"]
        sim_prior = rd["sim_prior"]
        H, W = rd["H"], rd["W"]
        gt = rd["gt"]
        grid_init = rd["grid_init"]

        pred = np.zeros((H, W, NUM_CLASSES))
        for y in range(H):
            for x in range(W):
                ic, db, co, dn = feats[y, x]
                fkey = tuple(feats[y, x])
                cell = grid[y][x]
                if cell == 5:
                    p = np.full(NUM_CLASSES, MIN_PROB); p[5] = 1.0
                elif cell == 10:
                    p = np.full(NUM_CLASSES, MIN_PROB); p[0] = 1.0
                else:
                    dp = domain_prior(ic, db, co)
                    prior = sw * sim_prior[y, x] + (1-sw) * dp
                    prior = np.maximum(prior, MIN_PROB)
                    prior /= prior.sum()
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
                pred[y, x] = p

        pred = np.maximum(pred, MIN_PROB)
        pred /= pred.sum(axis=2, keepdims=True)
        score, max_s = kl_score(gt, pred, grid_init)
        total += score
        weight_scores_per_round[sw].append((rd["rnd"], score, max_s, rd["sett_pct"]))

    weight_scores[sw] = total

# Display results
max_total = sum(m for _, _, m, _ in weight_scores_per_round[0.0])
print(f"{'sim_w':>6} {'Total':>10} {'Pct':>6}")
print("-" * 25)
best_w = 0.0
best_total = -999999
for sw in sim_weights:
    t = weight_scores[sw]
    pct = 100 * t / max_total
    marker = " <-- current" if sw == 0.5 else ""
    print(f"  {sw:.1f}  {t:10.1f} {pct:5.1f}%{marker}")
    if t > best_total:
        best_total = t
        best_w = sw

print(f"\nBest: sim_w={best_w} ({best_total:.1f})")

# Per-round comparison: best vs current (0.5)
print(f"\n=== Per-Round: sim_w={best_w} vs sim_w=0.5 ===")
print(f"{'Round':>6} {'sett%':>6} {'best':>8} {'0.5':>8} {'max':>8} {'diff':>7}")
for i in range(len(round_data)):
    rn, sb, mb, sp = weight_scores_per_round[best_w][i]
    _, s5, m5, _ = weight_scores_per_round[0.5][i]
    print(f"R{rn:2d}    {sp:5.1f}  {sb:8.1f} {s5:8.1f} {mb:8.1f} {sb-s5:+7.1f}")
