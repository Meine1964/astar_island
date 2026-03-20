"""Estimate how much observations actually help by simulating full pipeline on GT data."""
import numpy as np
import json
import os
from astar_island_simulator.env import (
    AstarIslandSimulator, HiddenParams, Settlement as SimSettlement,
    CODE_TO_CLASS, NUM_CLASSES,
)
from strategy import (
    compute_features, domain_prior, OutcomeModel, build_prediction,
    _load_bias_data, MIN_PROB,
)

def kl_score(gt, pred, eps=1e-10):
    """Compute KL-based score: max_score - KL_loss."""
    H, W = gt.shape[:2]
    total_ent = 0.0
    total_kl = 0.0
    for y in range(H):
        for x in range(W):
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


def make_sim_prior(detail, si, n_sims=50):
    state = detail["initial_states"][si]
    grid_np = np.array(state["grid"], dtype=int)
    setts = [s for s in state["settlements"] if s.get("alive", True)]
    W, H = detail["map_width"], detail["map_height"]

    sett_objs = []
    for i, s in enumerate(setts):
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
    sim.base_grid = grid_np
    sim.base_settlements = sett_objs

    counts = np.zeros((H, W, NUM_CLASSES))
    for seed in range(n_sims):
        final_grid, _ = sim.run(sim_seed=seed)
        for y in range(H):
            for x in range(W):
                cls = CODE_TO_CLASS.get(int(final_grid[y, x]), 0)
                counts[y, x, cls] += 1
    prior = counts / n_sims
    prior = np.clip(prior, MIN_PROB, None)
    prior /= prior.sum(axis=-1, keepdims=True)
    return prior


def simulate_observations(gt, grid_init, n_obs_sims=200):
    """Simulate what observations would look like from GT distribution."""
    H, W, C = gt.shape
    obs = np.zeros((H, W, C))
    obs_n = np.zeros((H, W), dtype=int)

    rng = np.random.default_rng(42)
    for _ in range(n_obs_sims):
        for y in range(H):
            for x in range(W):
                if grid_init[y, x] in (10, 5):
                    continue
                # Sample from GT distribution
                cls = rng.choice(C, p=gt[y, x])
                obs[y, x, cls] += 1
                obs_n[y, x] += 1

    # Normalize
    for y in range(H):
        for x in range(W):
            if obs_n[y, x] > 0:
                obs[y, x] /= obs_n[y, x]
    return obs, obs_n


# Test on a few seeds
print("=== Observation Impact Analysis ===\n")

# Different observation counts simulate different query budgets
obs_counts = [0, 1, 3, 5, 10, 50, 200]

for rnd in [1, 5, 9]:
    rdir = f"data/round_{rnd:02d}"
    detail = json.load(open(os.path.join(rdir, "round_detail.json")))
    W, H = detail["map_width"], detail["map_height"]

    for si in [0]:
        gt = np.load(os.path.join(rdir, "analysis", f"seed_{si}_ground_truth.npy"))
        state = detail["initial_states"][si]
        grid_init = np.array(state["grid"], dtype=int)

        sim_prior = make_sim_prior(detail, si, n_sims=50)

        print(f"\nR{rnd} S{si}:")

        # Score with sim-prior only
        sim_pred = sim_prior.copy()
        score_sim, max_score = kl_score(gt, sim_pred)
        print(f"  Max score: {max_score:.1f}")
        print(f"  Sim-only: {score_sim:.1f} ({100*score_sim/max_score:.1f}%)")

        # Score with N observations per cell
        for n_obs in obs_counts[1:]:
            rng = np.random.default_rng(42)
            obs = np.zeros((H, W, NUM_CLASSES))
            obs_n = np.zeros((H, W), dtype=int)

            for y in range(H):
                for x in range(W):
                    if grid_init[y, x] in (10, 5):
                        continue
                    for _ in range(n_obs):
                        cls = rng.choice(NUM_CLASSES, p=gt[y, x])
                        obs[y, x, cls] += 1
                        obs_n[y, x] += 1

            # Build prediction using build_prediction logic
            pred = np.zeros((H, W, NUM_CLASSES))
            setts = [s for s in state["settlements"] if s.get("alive", True)]
            features = compute_features(state["grid"], setts, W, H)
            model = OutcomeModel()

            # Feed observations to model
            for y in range(H):
                for x in range(W):
                    if obs_n[y, x] > 0:
                        fkey = tuple(features[y][x])
                        for c in range(NUM_CLASSES):
                            if obs[y, x, c] > 0:
                                model.observe(fkey, c, obs[y, x, c])

            # Build prediction
            seed_info = {
                "grid": state["grid"],
                "settlements": setts,
                "map_width": W,
                "map_height": H,
                "features": features,
            }
            pred = build_prediction(seed_info, obs, obs_n, model, H, W, sim_prior=sim_prior)
            score_obs, _ = kl_score(gt, pred)
            gain = score_obs - score_sim
            print(f"  {n_obs:3d} obs/cell: {score_obs:.1f} ({100*score_obs/max_score:.1f}%) gain={gain:+.1f}")


print("\n\n=== Key Question: What happens with ACTUAL query observations? ===")
print("In practice, we get ~50 queries × ~225 cells = ~11,250 observed cells")
print("But 40×40 = 1600 cells, so some cells get ~7 observations, many get 0")
print("The sim-prior is critical for unobserved cells (maybe 30-60% of dynamic cells)")
