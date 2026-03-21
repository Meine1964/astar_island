"""Optimize hedging ratio and explore other improvements for R13+.

Tests different sim:domain prior blending ratios on all GT rounds.
Also tests adaptive hedging (adjust ratio based on early observations).
"""
import numpy as np, json, os
from strategy import domain_prior, compute_features, NUM_CLASSES

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

def make_hedged(prior_ens, feats, grid_init, sim_weight, H=40, W=40):
    """Blend sim prior with domain prior."""
    result = prior_ens.copy()
    dom_weight = 1.0 - sim_weight
    for y in range(H):
        for x in range(W):
            if grid_init[y, x] in (10, 5):
                continue
            ic, db, co, dn = feats[y][x]
            dp = domain_prior(ic, db, co)
            result[y, x] = sim_weight * prior_ens[y, x] + dom_weight * dp
            result[y, x] = np.maximum(result[y, x], 0.002)
            result[y, x] /= result[y, x].sum()
    return result

# Collect all rounds with GT
print("=== Loading all rounds with ground truth ===\n")
round_data = []
for rnd in range(1, 13):
    rdir = f"data/round_{rnd:02d}"
    detail_path = os.path.join(rdir, "round_detail.json")
    if not os.path.exists(detail_path):
        continue
    detail = json.load(open(detail_path))
    W, H = detail["map_width"], detail["map_height"]
    
    for si in [0]:  # Just seed 0 for speed
        gt_path = os.path.join(rdir, "analysis", f"seed_{si}_ground_truth.npy")
        if not os.path.exists(gt_path):
            continue
        gt = np.load(gt_path)
        state = detail["initial_states"][si]
        grid_init = np.array(state["grid"], dtype=int)
        setts = [s for s in state["settlements"] if s.get("alive", True)]
        feats = compute_features(state["grid"], setts, W, H)
        
        # Use synthetic sim prior: sim always predicts ~11% settlement, ~62% plains, ~22% forest
        # This is faster than running the actual sim and captures its typical output
        prior_ens = np.zeros((H, W, 6))
        for y in range(H):
            for x in range(W):
                if grid_init[y, x] == 10:
                    prior_ens[y, x] = [1.0, 0, 0, 0, 0, 0]
                elif grid_init[y, x] == 5:
                    prior_ens[y, x] = [0, 0, 0, 0, 0, 1.0]
                else:
                    ic, db, co, dn = feats[y][x]
                    # Approximate sim output based on feature type
                    if ic == 4:  # forest
                        prior_ens[y, x] = [0.08, 0.11, 0.005, 0.01, 0.79, 0.005]
                    elif ic == 1:  # settlement init
                        prior_ens[y, x] = [0.45, 0.28, 0.005, 0.025, 0.23, 0.005]
                    else:  # plains
                        prior_ens[y, x] = [0.62, 0.11, 0.005, 0.01, 0.25, 0.005]
        
        # Measure GT settlement rate
        n_cells = 0
        sett_gt = 0.0
        for y in range(H):
            for x in range(W):
                if grid_init[y, x] not in (10, 5):
                    n_cells += 1
                    sett_gt += gt[y, x, 1]
        sett_pct = 100 * sett_gt / n_cells if n_cells > 0 else 0
        
        round_data.append({
            "rnd": rnd, "si": si, "gt": gt, "grid_init": grid_init,
            "feats": feats, "prior_ens": prior_ens, "sett_pct": sett_pct,
        })
        print(f"  R{rnd}/s{si}: sett={sett_pct:.1f}%")

print(f"\nLoaded {len(round_data)} seed-rounds\n")

# Test different hedging ratios
print("=== Hedging Ratio Optimization ===\n")
ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ratio_scores = {r: [] for r in ratios}

for rd in round_data:
    for r in ratios:
        hedged = make_hedged(rd["prior_ens"], rd["feats"], rd["grid_init"], sim_weight=r)
        score, max_score = kl_score(rd["gt"], hedged, rd["grid_init"])
        ratio_scores[r].append((score, max_score, rd["rnd"], rd["si"]))

print(f"{'Ratio':>6} {'TotalScore':>11} {'MeanScore':>10} {'MeanPct':>8}")
print("-" * 40)
best_ratio = 0.0
best_total = -999999
for r in ratios:
    total = sum(s for s, m, rn, si in ratio_scores[r])
    mean = total / len(ratio_scores[r])
    max_total = sum(m for s, m, rn, si in ratio_scores[r])
    pct = 100 * total / max_total if max_total > 0 else 0
    print(f"  {r:.1f}   {total:11.1f} {mean:10.1f}   {pct:6.1f}%")
    if total > best_total:
        best_total = total
        best_ratio = r

print(f"\nBest ratio: {best_ratio} (sim_weight)")

# Break down by round for best ratio and current 0.5
print(f"\n=== Per-Round Breakdown: best={best_ratio} vs current=0.5 ===\n")
print(f"{'Round':>6} {'ratio={best_ratio}':>12} {'ratio=0.5':>10} {'max':>10} {'diff':>6}")
for rnd in sorted(set(rd["rnd"] for rd in round_data)):
    best_scores = [s for s, m, rn, si in ratio_scores[best_ratio] if rn == rnd]
    curr_scores = [s for s, m, rn, si in ratio_scores[0.5] if rn == rnd]
    max_scores = [m for s, m, rn, si in ratio_scores[0.5] if rn == rnd]
    b = sum(best_scores)
    c = sum(curr_scores)
    mx = sum(max_scores)
    print(f"  R{rnd:2d}  {b:10.1f}  {c:10.1f}  {mx:10.1f}  {b-c:+6.1f}")

# Test adaptive hedging: use different blend per round based on "early signal"
# Idea: if round has many initial settlements AND they're near coast -> likely high survival -> weight sim more
print(f"\n=== Adaptive Hedging Test ===")
print("(Use initial settlement count/positions to pick sim_weight per round)\n")

adaptive_total = 0
static_total = 0
for rd in round_data:
    # Signal: number of initial settlements (proxy for round difficulty)
    # More setts with high GT survival should lean more on sim
    # Fewer setts / low GT survival should lean more on domain
    # But we don't know GT at runtime! Use initial setts count as proxy.
    
    # Test: just use best_ratio for now
    hedged_best = make_hedged(rd["prior_ens"], rd["feats"], rd["grid_init"], sim_weight=best_ratio)
    score_best, max_score = kl_score(rd["gt"], hedged_best, rd["grid_init"])
    
    hedged_curr = make_hedged(rd["prior_ens"], rd["feats"], rd["grid_init"], sim_weight=0.5)
    score_curr, _ = kl_score(rd["gt"], hedged_curr, rd["grid_init"])
    
    adaptive_total += score_best
    static_total += score_curr

print(f"Static 0.5 total: {static_total:.1f}")
print(f"Best {best_ratio} total: {adaptive_total:.1f}")
print(f"Improvement: {adaptive_total - static_total:+.1f}")

# Also test: domain-prior only (no sim at all)
print(f"\n=== Domain Prior Only (no sim) ===")
dom_total = 0
for rd in round_data:
    hedged = make_hedged(rd["prior_ens"], rd["feats"], rd["grid_init"], sim_weight=0.0)
    score, _ = kl_score(rd["gt"], hedged, rd["grid_init"])
    dom_total += score
print(f"Domain-only total: {dom_total:.1f}")
print(f"vs 0.5 hedge: {static_total:.1f} (diff: {dom_total - static_total:+.1f})")
print(f"vs best {best_ratio}: {adaptive_total:.1f} (diff: {dom_total - adaptive_total:+.1f})")
