import requests
import numpy as np
import os
import truststore
truststore.inject_into_ssl()

BASE = "https://api.ainm.no"
TOKEN = os.environ.get("AINM_TOKEN", "YOUR_JWT_TOKEN")  # Set AINM_TOKEN env var or paste token here

session = requests.Session()
session.cookies.set("access_token", TOKEN)

# --- Step 1: Find the active round ---
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)
if not active:
    print("No active round found.")
    exit()

round_id = active["id"]
print(f"Active round: #{active['round_number']} ({round_id})")

# --- Step 2: Get round details (initial states for all seeds) ---
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
width = detail["map_width"]
height = detail["map_height"]
seeds = detail["seeds_count"]
print(f"Map: {width}x{height}, {seeds} seeds")

for i, state in enumerate(detail["initial_states"]):
    grid = state["grid"]
    settlements = state["settlements"]
    print(f"  Seed {i}: {len(settlements)} settlements")

# --- Step 3: Check query budget ---
budget = session.get(f"{BASE}/astar-island/budget").json()
print(f"Budget response: {budget}")
queries_used = budget.get("queries_used", budget.get("used", 0))
queries_max = budget.get("queries_max", budget.get("max", budget.get("total", 50)))
print(f"Budget: {queries_used}/{queries_max} queries used")

# --- Step 4: Run simulation queries (use budget wisely!) ---
# Terrain code -> prediction class mapping
CODE_TO_CLASS = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
NUM_CLASSES = 6
MIN_PROB = 0.01  # Never assign 0 probability (KL divergence becomes infinite)

observations = {seed_idx: np.zeros((height, width, NUM_CLASSES)) for seed_idx in range(seeds)}
obs_counts = {seed_idx: np.zeros((height, width)) for seed_idx in range(seeds)}

# Example: tile the map with 15x15 viewports across seeds
queries_per_seed = queries_max // seeds
for seed_idx in range(seeds):
    for q in range(queries_per_seed):
        # Simple tiling strategy: cover different parts of the map
        vx = (q * 15) % width
        vy = ((q * 15) // width * 15) % height
        vw, vh = min(15, width - vx), min(15, height - vy)
        if vw < 5 or vh < 5:
            continue

        result = session.post(f"{BASE}/astar-island/simulate", json={
            "round_id": round_id,
            "seed_index": seed_idx,
            "viewport_x": vx,
            "viewport_y": vy,
            "viewport_w": vw,
            "viewport_h": vh,
        }).json()

        if "error" in result:
            print(f"  Query error: {result['error']}")
            break

        vp = result["viewport"]
        sim_grid = result["grid"]
        for row_i, row in enumerate(sim_grid):
            for col_i, cell in enumerate(row):
                gy = vp["y"] + row_i
                gx = vp["x"] + col_i
                cls = CODE_TO_CLASS.get(cell, 0)
                observations[seed_idx][gy, gx, cls] += 1
                obs_counts[seed_idx][gy, gx] += 1

        print(f"  Seed {seed_idx} query {q}: viewport ({vp['x']},{vp['y']}) {vp['w']}x{vp['h']}, "
              f"budget {result['queries_used']}/{result['queries_max']}")

        if result["queries_used"] >= result["queries_max"]:
            break
    if result.get("queries_used", 0) >= result.get("queries_max", 50):
        break

# --- Step 5: Build predictions and submit ---
for seed_idx in range(seeds):
    prediction = np.full((height, width, NUM_CLASSES), 1.0 / NUM_CLASSES)  # uniform baseline

    # Where we have observations, use empirical distribution
    observed = obs_counts[seed_idx] > 0
    for y in range(height):
        for x in range(width):
            if obs_counts[seed_idx][y, x] > 0:
                prediction[y, x] = observations[seed_idx][y, x] / obs_counts[seed_idx][y, x]

    # Use initial state to set known static terrain (mountains, ocean, forest)
    init_grid = detail["initial_states"][seed_idx]["grid"]
    for y in range(height):
        for x in range(width):
            cell = init_grid[y][x]
            if cell == 5:  # Mountain - never changes
                prediction[y, x] = np.zeros(NUM_CLASSES)
                prediction[y, x, 5] = 1.0
            elif cell == 10:  # Ocean - never changes
                prediction[y, x] = np.zeros(NUM_CLASSES)
                prediction[y, x, 0] = 1.0

    # Enforce minimum probability floor and renormalize
    prediction = np.maximum(prediction, MIN_PROB)
    prediction /= prediction.sum(axis=2, keepdims=True)

    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    })
    print(f"Seed {seed_idx} submit: {resp.status_code} - {resp.text[:200]}")

print("Done!")
