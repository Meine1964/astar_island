"""Astar Island — local development & testing.

Runs the prediction pipeline against the local simulator.
Use server_run.py for live API submissions.

Usage:  uv run python main.py
"""
import json
import numpy as np

from astar_island_simulator import LocalAPI, HiddenParams
from strategy import (
    OutcomeModel, analyze_seeds, execute_adaptive_queries,
    build_prediction, compute_simulator_prior,
    print_summary, NUM_CLASSES, estimate_settlement_regime,
)


# ── Local simulator setup ─────────────────────────────────────────────
# Load calibrated params if available, otherwise use defaults
try:
    with open("calibrated_params.json") as f:
        params = HiddenParams(**{k: v for k, v in json.load(f).items()
                                 if k in HiddenParams.__dataclass_fields__})
    print("Using calibrated parameters from calibrated_params.json")
except FileNotFoundError:
    params = HiddenParams()
    print("Using default parameters (run calibrate.py after a round completes)")

api = LocalAPI(n_seeds=5, map_width=40, map_height=40,
               queries_max=50, base_map_seed=42, params=params)
session = api.get_session()
BASE = "http://local/astar-island"

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

# ── Step 3: Budget ────────────────────────────────────────────────────
budget = session.get(f"{BASE}/budget").json()
used = budget.get("queries_used", 0)
total = budget.get("queries_max", 50)
remaining = total - used
print(f"Budget: {used}/{total} used, {remaining} remaining")

# ── Step 4: Cross-seed model (no history in local mode) ───────────────
model = OutcomeModel()
print("Local mode: no historical calibration")

# ── Step 5: Analyze seeds ─────────────────────────────────────────────
seed_info = analyze_seeds(detail, seeds)

# ── Step 5b: Compute simulator prior for each seed ────────────────────
print("Computing simulator priors (100 sims per seed)...")
sim_priors = {}
for si in range(seeds):
    state = detail["initial_states"][si]
    setts = [s for s in state["settlements"] if s.get("alive", True)]
    sim_priors[si] = compute_simulator_prior(
        state["grid"], setts, W, H, n_sims=100, params=params, ensemble=False
    )
    n_dynamic = int((sim_priors[si].max(axis=2) < 0.95).sum())
    print(f"  Seed {si}: {n_dynamic} dynamic cells in simulator prior")

# ── Step 6: Adaptive query execution with periodic scoring ────────────
obs = {i: np.zeros((H, W, NUM_CLASSES)) for i in range(seeds)}
obs_n = {i: np.zeros((H, W)) for i in range(seeds)}
settlement_stats = {}

score_count = 0

def score_all(query_count):
    """Callback: score all seeds with current data to track progress."""
    global score_count
    score_count += 1
    regime = estimate_settlement_regime(obs, obs_n, seed_info, seeds, H, W)
    total_score = 0
    for sj in range(seeds):
        pred = build_prediction(seed_info[sj], obs[sj], obs_n[sj], model, H, W,
                                sim_prior=sim_priors[sj],
                                settlement_stats=settlement_stats.get(sj),
                                regime=regime)
        result = api.score_prediction(seed_index=sj, prediction=pred, n_sims=10)
        total_score += result["score"]
    rate_str = f"{regime['observed_rate']:.1%}" if regime['observed_rate'] is not None else "N/A"
    print(f"\n  >> Checkpoint #{score_count} after {query_count}Q: "
          f"total_score={total_score:.1f} (avg {total_score/seeds:.1f}/seed) "
          f"regime: rate={rate_str}, scale={regime['scale']:.2f}\n")

print(f"\nAdaptive query selection ({remaining} queries)...")
total_q = execute_adaptive_queries(
    session, BASE, round_id, seed_info,
    obs, obs_n, model, sim_priors,
    seeds, H, W, budget=remaining, delay=0,
    submit_fn=score_all, submit_every=10,
    settlement_stats=settlement_stats,
)
print(f"\nQueries executed: {total_q}")
print(f"Cross-seed model: {len(model.counts)} buckets, "
      f"{sum(c.sum() for c in model.counts.values()):.0f} total observations")

# ── Step 7: Build predictions and submit ──────────────────────────────
regime = estimate_settlement_regime(obs, obs_n, seed_info, seeds, H, W)
rate_str = f"{regime['observed_rate']:.1%}" if regime['observed_rate'] is not None else "N/A"
print(f"\nRegime: rate={rate_str}, scale={regime['scale']:.2f}")
for si in range(seeds):
    pred = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W,
                            sim_prior=sim_priors[si], regime=regime)

    resp = session.post(f"{BASE}/submit", json={
        "round_id": round_id, "seed_index": si,
        "prediction": pred.tolist(),
    })
    n_obs = int((obs_n[si] > 0).sum())
    print(f"Seed {si}: {resp.status_code} | {n_obs} cells observed | {resp.text[:200]}")

# ── Step 8: Score against local ground truth ──────────────────────────
print("\n-- Local scoring (10 sims per seed) --")
for si in range(seeds):
    pred = build_prediction(seed_info[si], obs[si], obs_n[si], model, H, W,
                            sim_prior=sim_priors[si], regime=regime)
    result = api.score_prediction(seed_index=si, prediction=pred, n_sims=10)
    print(f"  Seed {si}: score={result['score']:.2f}, "
          f"weighted_kl={result['weighted_kl']:.4f}, "
          f"dynamic_cells={result['n_dynamic_cells']}")

# ── Summary ───────────────────────────────────────────────────────────
print_summary(obs_n, model, seeds)
print("\nDone!")
