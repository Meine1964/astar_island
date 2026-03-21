"""Analyze R12 results and test what hedging would have scored.

Also tests domain-prior-only vs various blends using REAL sim prior
from the actual observations we collected.
"""
import numpy as np, json, os
from strategy import (
    domain_prior, compute_features, NUM_CLASSES, OutcomeModel,
    build_prediction, calibrate_from_history, compute_simulator_prior,
)
import requests, truststore
truststore.inject_into_ssl()

BASE = "https://api.ainm.no/astar-island"
TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJlMDcyNjM1ZC1mYTNmLTQ5MzMtOGMwNC1lMmJmYmM4ZDhiZDEi"
    "LCJlbWFpbCI6Im1laW5lLnZhbi5kZXIubWV1bGVuQGdtYWlsLmNvbSIsImlz"
    "X2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTEyMDI4fQ."
    "QMd3aqRnowq1zyiyFDnOu0bXSNSAZ2rEMaIoCkOQLJ4"
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

# ── Load R12 data ──
rnd = 12
detail = json.load(open(f"data/round_{rnd:02d}/round_detail.json"))
W, H = detail["map_width"], detail["map_height"]
seeds = detail["seeds_count"]

# Load observations we made in R12
obs_data = np.load(f"data/round_{rnd:02d}/observations.npz")
obs = {si: obs_data[f"obs_{si}"] for si in range(seeds)}
obs_n = {si: obs_data[f"obs_n_{si}"] for si in range(seeds)}

# Pre-train model from history (same as server_run.py does)
session = requests.Session()
session.cookies.set("access_token", TOKEN)
model = OutcomeModel()
n_hist = calibrate_from_history(session, BASE, model)
print(f"Historical calibration: {n_hist} seed-rounds")

# R12 GT settlement rate
print("\n=== R12 Ground Truth Analysis ===")
for si in range(seeds):
    gt = np.load(f"data/round_{rnd:02d}/analysis/seed_{si}_ground_truth.npy")
    grid_init = np.array(detail["initial_states"][si]["grid"], dtype=int)
    n = 0; sett = 0
    for y in range(H):
        for x in range(W):
            if grid_init[y, x] not in (10, 5):
                n += 1
                sett += gt[y, x, 1]
    print(f"  Seed {si}: sett={100*sett/n:.1f}%")

# Now test different strategies
print("\n=== Strategy Comparison for R12 ===")
print(f"{'Strategy':>30} {'Total':>8} {'Max':>8} {'Pct':>6}")
print("-" * 55)

# 1. What we actually submitted (saved predictions)
total_actual = 0
total_max = 0
for si in range(seeds):
    gt = np.load(f"data/round_{rnd:02d}/analysis/seed_{si}_ground_truth.npy")
    pred = np.load(f"data/round_{rnd:02d}/predictions/seed_{si}.npy")
    grid_init = np.array(detail["initial_states"][si]["grid"], dtype=int)
    score, max_s = kl_score(gt, pred, grid_init)
    total_actual += score
    total_max += max_s
print(f"{'Actual submission (sim-only)':>30} {total_actual:8.1f} {total_max:8.1f} {100*total_actual/total_max:5.1f}%")

# 2. Rebuild with hedging at different ratios
for sim_w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    total = 0
    for si in range(seeds):
        state = detail["initial_states"][si]
        setts = [s for s in state["settlements"] if s.get("alive", True)]
        feats = compute_features(state["grid"], setts, W, H)
        gt = np.load(f"data/round_{rnd:02d}/analysis/seed_{si}_ground_truth.npy")
        grid_init = np.array(state["grid"], dtype=int)
        
        # Compute real sim prior
        sim_prior = compute_simulator_prior(state["grid"], setts, W, H, n_sims=50, ensemble=True)

        # Create hedged prior manually
        hedged_prior = np.zeros((H, W, 6))
        for y in range(H):
            for x in range(W):
                if grid_init[y, x] in (10, 5):
                    hedged_prior[y, x] = sim_prior[y, x]
                else:
                    ic, db, co, dn = feats[y][x]
                    dp = domain_prior(ic, db, co)
                    hedged_prior[y, x] = sim_w * sim_prior[y, x] + (1-sim_w) * dp
                    hedged_prior[y, x] = np.maximum(hedged_prior[y, x], 0.002)
                    hedged_prior[y, x] /= hedged_prior[y, x].sum()
        
        # Build full prediction with this hedged prior
        seed_info = {
            "grid": state["grid"],
            "settlements": setts,
            "features": feats,
            "viewports": [],
        }
        pred = build_prediction(seed_info, obs[si], obs_n[si], model, H, W,
                               sim_prior=hedged_prior)
        score, max_s = kl_score(gt, pred, grid_init)
        total += score
    
    label = f"sim_w={sim_w:.1f}" if sim_w > 0 else "domain-only"
    print(f"{label:>30} {total:8.1f} {total_max:8.1f} {100*total/total_max:5.1f}%")
