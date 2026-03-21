"""Quick validation: does wider ensemble improve sim prior on extreme rounds?
Tests R3 (0.3% sett), R9 (14.2% sett), R11 (28.4% sett)."""
import numpy as np, json
from astar_island_simulator.env import (
    AstarIslandSimulator, HiddenParams, Settlement as SimSettlement,
    CODE_TO_CLASS, NUM_CLASSES,
)
from strategy import compute_simulator_prior

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

for rnd in [3, 8, 9, 10, 11]:
    detail = json.load(open(f"data/round_{rnd:02d}/round_detail.json"))
    gt = np.load(f"data/round_{rnd:02d}/analysis/seed_0_ground_truth.npy")
    state = detail["initial_states"][0]
    grid_init = np.array(state["grid"], dtype=int)
    setts = [s for s in state["settlements"] if s.get("alive", True)]

    # Compute sim prior with new wide ensemble
    prior_ens = compute_simulator_prior(state["grid"], setts,
                                     detail["map_width"], detail["map_height"],
                                     n_sims=100, ensemble=True)

    # Also compute a "hedged" prior: blend sim with domain prior
    from strategy import domain_prior, compute_features
    feats = compute_features(state["grid"], setts, detail["map_width"], detail["map_height"])
    prior_hedged = prior_ens.copy()
    for y in range(40):
        for x in range(40):
            if grid_init[y, x] in (10, 5):
                continue
            ic, db, co, dn = feats[y][x]
            dp = domain_prior(ic, db, co)
            # 50/50 blend of sim and domain prior
            prior_hedged[y, x] = 0.5 * prior_ens[y, x] + 0.5 * dp
            prior_hedged[y, x] = np.maximum(prior_hedged[y, x], 0.002)
            prior_hedged[y, x] /= prior_hedged[y, x].sum()

    score_ens, max_score = kl_score(gt, prior_ens, grid_init)
    score_hedged, _ = kl_score(gt, prior_hedged, grid_init)

    # Check settlement prediction
    sett_pred = 0.0
    sett_gt = 0.0
    n = 0
    for y in range(40):
        for x in range(40):
            if grid_init[y, x] in (10, 5):
                continue
            n += 1
            sett_pred += prior_ens[y, x, 1]
            sett_gt += gt[y, x, 1]

    print(f"R{rnd:2d}: ens={score_ens:.1f} hedged={score_hedged:.1f} / {max_score:.1f}  "
          f"sett: GT={100*sett_gt/n:.1f}% pred={100*sett_pred/n:.1f}%")
