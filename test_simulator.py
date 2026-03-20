"""Test the local Astar Island simulator and score a prediction.

Run with:  uv run python test_simulator.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, ".")
from astar_island_simulator import AstarIslandSimulator, HiddenParams, LocalAPI
from astar_island_simulator.simulator import CODE_TO_CLASS, NUM_CLASSES

print("=== Astar Island Local Simulator Test ===\n")

# 1. Create a simulator and inspect the map
params = HiddenParams()
sim = AstarIslandSimulator(map_seed=42, params=params)
grid, setts = sim.initial_state()
H, W = len(grid), len(grid[0])
print(f"Map: {W}x{H}")
print(f"Settlements: {len(setts)}")

# Count terrain types
terrain_counts = {}
for row in grid:
    for cell in row:
        terrain_counts[cell] = terrain_counts.get(cell, 0) + 1
print(f"Terrain: {terrain_counts}")

# 2. Run a single simulation
t0 = time.time()
final_grid, final_setts = sim.run(sim_seed=123)
dt = time.time() - t0
alive = [s for s in final_setts if s.alive]
print(f"\nSingle sim: {dt:.3f}s, {len(alive)}/{len(final_setts)} settlements alive")

# 3. Run ground truth distribution (10 sims for quick test)
print("\nComputing ground truth (10 sims)...")
t0 = time.time()
gt = sim.ground_truth_distribution(n_sims=10)
dt = time.time() - t0
print(f"Ground truth: {dt:.1f}s")

# Count dynamic cells (entropy > 0.05)
entropy = np.zeros((H, W))
eps = 1e-10
for y in range(H):
    for x in range(W):
        q = gt[y, x]
        entropy[y, x] = -np.sum(q * np.log(q + eps))
dynamic = (entropy > 0.05).sum()
print(f"Dynamic cells: {dynamic}/{H*W} ({100*dynamic/(H*W):.1f}%)")
print(f"Max entropy: {entropy.max():.3f}")

# 4. Test the LocalAPI mock
print("\n=== LocalAPI Test ===")
api = LocalAPI(n_seeds=5, queries_max=50, base_map_seed=42, params=params)
session = api.get_session()

rounds = session.get("http://local/astar-island/rounds").json()
print(f"Rounds: {len(rounds)}, status={rounds[0]['status']}")

detail = session.get(f"http://local/astar-island/rounds/{rounds[0]['id']}").json()
print(f"Seeds: {detail['seeds_count']}, map: {detail['map_width']}x{detail['map_height']}")

budget = session.get("http://local/astar-island/budget").json()
print(f"Budget: {budget['queries_used']}/{budget['queries_max']}")

# Run a simulate query
result = session.post("http://local/astar-island/simulate", json={
    "round_id": rounds[0]["id"],
    "seed_index": 0,
    "viewport_x": 10, "viewport_y": 10,
    "viewport_w": 15, "viewport_h": 15,
}).json()
vp = result["viewport"]
print(f"Simulate: viewport ({vp['x']},{vp['y']}) {vp['w']}x{vp['h']}, "
      f"budget {result['queries_used']}/{result['queries_max']}")
print(f"  Settlements in viewport: {len(result['settlements'])}")

# 5. Test scoring: uniform prediction vs ground truth
print("\n=== Scoring Test ===")
uniform = np.full((H, W, NUM_CLASSES), 1.0 / NUM_CLASSES)
score_uniform = api.score_prediction(seed_index=0, prediction=uniform, n_sims=10)
print(f"Uniform prediction: score={score_uniform['score']}, "
      f"weighted_kl={score_uniform['weighted_kl']}, "
      f"dynamic_cells={score_uniform['n_dynamic_cells']}")

# Now test with a "cheating" prediction (ground truth itself)
gt_pred = gt.copy()
gt_pred = np.maximum(gt_pred, 0.005)
gt_pred /= gt_pred.sum(axis=2, keepdims=True)
score_perfect = api.score_prediction(seed_index=0, prediction=gt_pred, n_sims=10)
print(f"Near-perfect pred: score={score_perfect['score']}, "
      f"weighted_kl={score_perfect['weighted_kl']}")

print("\n=== All tests passed! ===")
