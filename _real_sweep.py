"""Sweep params against REAL initial states from API rounds."""
import numpy as np
import json
from astar_island_simulator.env import (
    AstarIslandSimulator, HiddenParams, Settlement as SimSettlement,
    CODE_TO_CLASS, NUM_CLASSES,
)

# Load real initial states
real_states = []
for rnd in [1, 3, 5, 7, 9]:
    try:
        detail = json.load(open(f"data/round_{rnd:02d}/round_detail.json"))
        gt = np.load(f"data/round_{rnd:02d}/analysis/seed_0_ground_truth.npy")
        real_states.append((detail, gt))
    except FileNotFoundError:
        pass

print(f"Loaded {len(real_states)} rounds")

# GT settlement %
gt_pcts = []
for detail, gt in real_states:
    state = detail["initial_states"][0]
    grid_init = np.array(state["grid"], dtype=int)
    dynamic = 0
    sett_prob = 0.0
    for y in range(40):
        for x in range(40):
            if grid_init[y, x] in (10, 5):
                continue
            dynamic += 1
            sett_prob += gt[y, x, 1]
    gt_pcts.append(sett_prob / dynamic)
print(f"GT settlement %: {100*np.mean(gt_pcts):.1f}%\n")


def run_on_real_state(detail, params, n_sims=20):
    state = detail["initial_states"][0]
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
    sim.params = params
    sim.width = W
    sim.height = H
    sim.base_grid = grid_np
    sim.base_settlements = sett_objs

    sett_pcts = []
    for seed in range(n_sims):
        final_grid, _ = sim.run(sim_seed=seed)
        dynamic = 0
        sett = 0
        for y in range(H):
            for x in range(W):
                if grid_np[y, x] in (10, 5):
                    continue
                dynamic += 1
                cls = CODE_TO_CLASS.get(int(final_grid[y, x]), 0)
                if cls == 1:
                    sett += 1
        sett_pcts.append(sett / dynamic if dynamic > 0 else 0)
    return np.mean(sett_pcts)


test_configs = [
    {"name": "Sev 3.0 exp20/0.12", "winter_base_severity": 3.0, "expansion_pop_threshold": 20, "expansion_prob": 0.12},
    {"name": "Sev 3.0 exp20/0.13", "winter_base_severity": 3.0, "expansion_pop_threshold": 20, "expansion_prob": 0.13},
    {"name": "Sev 3.0 exp22/0.13", "winter_base_severity": 3.0, "expansion_pop_threshold": 22, "expansion_prob": 0.13},
    {"name": "Sev 3.0 exp22/0.14", "winter_base_severity": 3.0, "expansion_pop_threshold": 22, "expansion_prob": 0.14},
    {"name": "Sev 3.0 exp25/0.14", "winter_base_severity": 3.0, "expansion_pop_threshold": 25, "expansion_prob": 0.14},
    {"name": "Sev 3.0 exp25/0.15", "winter_base_severity": 3.0, "expansion_pop_threshold": 25, "expansion_prob": 0.15},
    {"name": "Sev 3.0 exp25/0.13", "winter_base_severity": 3.0, "expansion_pop_threshold": 25, "expansion_prob": 0.13},
    {"name": "Sev 3.2 exp20/0.13", "winter_base_severity": 3.2, "expansion_pop_threshold": 20, "expansion_prob": 0.13},
    {"name": "Sev 3.2 exp20/0.15", "winter_base_severity": 3.2, "expansion_pop_threshold": 20, "expansion_prob": 0.15},
    {"name": "Sev 3.2 exp25/0.15", "winter_base_severity": 3.2, "expansion_pop_threshold": 25, "expansion_prob": 0.15},
]

for cfg in test_configs:
    name = cfg.pop("name")
    p = HiddenParams(**cfg)

    pcts = []
    for detail, gt in real_states:
        pct = run_on_real_state(detail, p, n_sims=15)
        pcts.append(pct)

    avg = np.mean(pcts)
    print(f"  {name:25s}: {100*avg:.1f}%  (target ~{100*np.mean(gt_pcts):.1f}%)")
