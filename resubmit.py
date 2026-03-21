"""Resubmit predictions for an active round using simulator priors.

Uses zero queries — only reads round data (free) and submits (free).
Ideal when queries are already spent but we have a better model.

Usage:  uv run python resubmit.py
"""
import requests
import numpy as np
import time
import truststore
truststore.inject_into_ssl()

from strategy import (
    OutcomeModel, analyze_seeds, build_prediction,
    calibrate_from_history, compute_simulator_prior, NUM_CLASSES,
    estimate_settlement_regime, self_consistency_tune,
)
import data_store

# ── Configuration ──────────────────────────────────────────────────────
BASE = "https://api.ainm.no/astar-island"
TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJlMDcyNjM1ZC1mYTNmLTQ5MzMtOGMwNC1lMmJmYmM4ZDhiZDEi"
    "LCJlbWFpbCI6Im1laW5lLnZhbi5kZXIubWV1bGVuQGdtYWlsLmNvbSIsImlz"
    "X2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTEyMDI4fQ."
    "QMd3aqRnowq1zyiyFDnOu0bXSNSAZ2rEMaIoCkOQLJ4"
)

session = requests.Session()
session.cookies.set("access_token", TOKEN)

# ── Find active round ─────────────────────────────────────────────────
rounds = session.get(f"{BASE}/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)
if not active:
    print("No active round found.")
    exit()
round_id = active["id"]
print(f"Round #{active['round_number']} ({round_id})")

# ── Round details ─────────────────────────────────────────────────────
detail = session.get(f"{BASE}/rounds/{round_id}").json()
W, H, seeds = detail["map_width"], detail["map_height"], detail["seeds_count"]
print(f"Map {W}x{H}, {seeds} seeds")

# ── Budget check ──────────────────────────────────────────────────────
budget = session.get(f"{BASE}/budget").json()
print(f"Budget: {budget.get('queries_used', 0)}/{budget.get('queries_max', 50)} used")
print("(This script uses ZERO queries — only free submissions)")

# ── Pre-train model from completed rounds ─────────────────────────────
model = OutcomeModel()
n_hist = calibrate_from_history(session, BASE, model)
print(f"Historical calibration: {n_hist} seed-rounds, {len(model.counts)} buckets")

# ── Analyze seeds ─────────────────────────────────────────────────────
seed_info = analyze_seeds(detail, seeds)

# ── Compute simulator priors ──────────────────────────────────────────
print("\nComputing ensemble simulator priors (100 sims per seed)...")
sim_priors = {}
for si in range(seeds):
    state = detail["initial_states"][si]
    setts = [s for s in state["settlements"] if s.get("alive", True)]
    sim_priors[si] = compute_simulator_prior(
        state["grid"], setts, W, H, n_sims=100, ensemble=True
    )
    n_dynamic = int((sim_priors[si].max(axis=2) < 0.95).sum())
    print(f"  Seed {si}: {n_dynamic} dynamic cells")

# ── Build and submit predictions ──────────────────────────────────────
round_num = active["round_number"]
obs, obs_n, loaded = data_store.load_observations(round_num, seeds, H, W, NUM_CLASSES)
if not loaded:
    print("No saved observations found -- using sim priors only")

print("\n-- Submitting predictions --")
regime = estimate_settlement_regime(obs, obs_n, seed_info, seeds, H, W)
base_scale = regime['scale']
if loaded:
    regime = self_consistency_tune(seed_info, obs, obs_n, model, seeds, H, W, regime)
rate_str = f"{regime['observed_rate']:.1%}" if regime['observed_rate'] is not None else "N/A"
print(f"Regime: rate={rate_str}, scale={base_scale:.2f}->{regime['scale']:.2f}")
for si in range(seeds):
    pred = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W,
                            sim_prior=sim_priors[si], regime=regime)
    resp = session.post(f"{BASE}/submit", json={
        "round_id": round_id, "seed_index": si,
        "prediction": pred.tolist(),
    })
    print(f"  Seed {si}: {resp.status_code} | {resp.text[:200]}")
    time.sleep(0.5)

print("\nDone! Predictions submitted with simulator priors (0 queries used).")
