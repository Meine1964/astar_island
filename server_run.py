"""Astar Island — live API submission.

Connects to the real competition server, runs queries, and submits predictions.
Use main.py for local simulator testing.

Usage:  uv run python server_run.py
"""
import requests
import numpy as np
import time
import truststore
truststore.inject_into_ssl()

from strategy import (
    OutcomeModel, CODE_TO_CLASS, analyze_seeds,
    build_prediction, calibrate_from_history, compute_simulator_prior,
    execute_adaptive_queries, print_summary, NUM_CLASSES,
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

# ── Safety prompt ──────────────────────────────────────────────────────
print("=== LIVE API SUBMISSION ===")
print("This will use real queries and submit real predictions.\n")

# ── Session setup ──────────────────────────────────────────────────────
session = requests.Session()
session.cookies.set("access_token", TOKEN)

# ── Step 1: Active round ──────────────────────────────────────────────
rounds = session.get(f"{BASE}/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)
if not active:
    print("No active round found.")
    exit()
round_id = active["id"]
print(f"Round #{active['round_number']} ({round_id})")

# ── Step 2: Round details ─────────────────────────────────────────────
detail = session.get(f"{BASE}/rounds/{round_id}").json()
W, H, seeds = detail["map_width"], detail["map_height"], detail["seeds_count"]
print(f"Map {W}x{H}, {seeds} seeds")
round_num = active["round_number"]
data_store.save_round_detail(round_num, detail)

# ── Step 3: Budget ────────────────────────────────────────────────────
budget = session.get(f"{BASE}/budget").json()
used = budget.get("queries_used", 0)
total = budget.get("queries_max", 50)
remaining = total - used
print(f"Budget: {used}/{total} used, {remaining} remaining")

if remaining <= 0:
    print("No queries remaining — cannot proceed.")
    exit()

# ── Step 4: Pre-train cross-seed model from history ───────────────────
model = OutcomeModel()
n_hist = calibrate_from_history(session, BASE, model)
print(f"Historical calibration: {n_hist} seed-rounds, {len(model.counts)} buckets")

# ── Step 5: Analyze seeds ─────────────────────────────────────────────
seed_info = analyze_seeds(detail, seeds)

# ── Step 5b: Compute simulator prior for each seed ────────────────────
print("Computing simulator priors (100 sims per seed)...")
sim_priors = {}
for si in range(seeds):
    state = detail["initial_states"][si]
    setts = [s for s in state["settlements"] if s.get("alive", True)]
    sim_priors[si] = compute_simulator_prior(
        state["grid"], setts, W, H, n_sims=100
    )
    n_dynamic = int((sim_priors[si].max(axis=2) < 0.95).sum())
    print(f"  Seed {si}: {n_dynamic} dynamic cells in simulator prior")

# ── Step 6: Submit simulator-prior baseline (free safety net) ─────────
obs = {i: np.zeros((H, W, NUM_CLASSES)) for i in range(seeds)}
obs_n = {i: np.zeros((H, W)) for i in range(seeds)}
settlement_stats = {}  # seed_index -> {(x,y) -> stats}

print("\n-- Submission #0: simulator-prior baseline --")
for si in range(seeds):
    pred = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W,
                            sim_prior=sim_priors[si])
    resp = session.post(f"{BASE}/submit", json={
        "round_id": round_id, "seed_index": si,
        "prediction": pred.tolist(),
    })
    print(f"  Seed {si}: {resp.status_code} | {resp.json().get('status', resp.text[:80])}")
    time.sleep(0.5)

# ── Step 7: Adaptive queries with periodic resubmission ───────────────
submit_count = 0

def submit_all(query_count):
    """Callback: resubmit all seeds with current data."""
    global submit_count
    submit_count += 1
    print(f"\n-- Submission #{submit_count}: after {query_count} queries --")
    for sj in range(seeds):
        pred = build_prediction(seed_info[sj], obs[sj], obs_n[sj], model, H, W,
                                sim_prior=sim_priors[sj],
                                settlement_stats=settlement_stats.get(sj))
        resp = session.post(f"{BASE}/submit", json={
            "round_id": round_id, "seed_index": sj,
            "prediction": pred.tolist(),
        })
        n_obs = int((obs_n[sj] > 0).sum())
        print(f"  Seed {sj}: {resp.status_code} | {n_obs} cells observed")
        time.sleep(0.5)
    print()

print(f"\nAdaptive query selection ({remaining} queries)...")
total_q = execute_adaptive_queries(
    session, BASE, round_id, seed_info,
    obs, obs_n, model, sim_priors,
    seeds, H, W, budget=remaining, delay=0.15,
    submit_fn=submit_all, submit_every=10,
    query_log_fn=lambda qn, si, vp, res: data_store.append_query(
        round_num, qn, si, vp, res),
    settlement_stats=settlement_stats,
)
print(f"\nQueries executed: {total_q}")
data_store.save_observations(round_num, obs, obs_n, seeds)

# ── Step 8: Final submission ──────────────────────────────────────────
submit_count += 1
print(f"\n-- Final submission #{submit_count}: after {total_q} queries --")
for si in range(seeds):
    pred = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W,
                            sim_prior=sim_priors[si],
                            settlement_stats=settlement_stats.get(si))
    data_store.save_prediction(round_num, si, pred)
    resp = session.post(f"{BASE}/submit", json={
        "round_id": round_id, "seed_index": si,
        "prediction": pred.tolist(),
    })
    n_obs = int((obs_n[si] > 0).sum())
    print(f"  Seed {si}: {resp.status_code} | {n_obs} cells observed | {resp.text[:200]}")
    time.sleep(0.5)

# ── Summary ───────────────────────────────────────────────────────────
print_summary(obs_n, model, seeds)
print("\nDone!")
