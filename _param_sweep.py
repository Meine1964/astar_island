"""Quick parameter sweep to find expansion params that give ~15% settlement."""
import numpy as np
import json
from astar_island_simulator.env import AstarIslandSimulator, HiddenParams, CODE_TO_CLASS

# Load GT data for comparison
gt_sett_pcts = []
for r in [1, 3, 5, 7, 9]:
    detail = json.load(open(f"data/round_{r:02d}/round_detail.json"))
    seed0 = detail["initial_states"][0]
    grid_init = np.array(seed0["grid"], dtype=int)
    gt = np.load(f"data/round_{r:02d}/analysis/seed_0_ground_truth.npy")  # (40, 40, 6)

    dynamic = 0
    sett_prob = 0.0
    for y in range(40):
        for x in range(40):
            if grid_init[y, x] in (10, 5):
                continue
            dynamic += 1
            sett_prob += gt[y, x, 1]  # class 1 = settlement

    gt_sett_pcts.append(sett_prob / dynamic)

gt_avg = np.mean(gt_sett_pcts)
print(f"GT average settlement %: {gt_avg:.3f} ({100*gt_avg:.1f}%)")
print()

# Test parameter combinations
test_configs = [
    {"name": "Sev 2.0 base", "winter_base_severity": 2.0},
    {"name": "Sev 1.5 base", "winter_base_severity": 1.5},
    {"name": "Sev 1.5 + expand", "winter_base_severity": 1.5, "expansion_pop_threshold": 20, "expansion_prob": 0.20},
    {"name": "Sev 1.5 + aggr exp", "winter_base_severity": 1.5, "expansion_pop_threshold": 15, "expansion_prob": 0.30, "expansion_radius": 5},
    {"name": "Sev 2.0 + aggr exp", "winter_base_severity": 2.0, "expansion_pop_threshold": 15, "expansion_prob": 0.30, "expansion_radius": 5},
    {"name": "Sev 1.0", "winter_base_severity": 1.0},
    {"name": "Sev 1.0 + expand", "winter_base_severity": 1.0, "expansion_pop_threshold": 20, "expansion_prob": 0.20},
    {"name": "Sev 2.0 exp+reclaim", "winter_base_severity": 2.0, "expansion_pop_threshold": 20, "expansion_prob": 0.20, "ruin_reclaim_prob": 0.08},
    {"name": "Sev 1.5 exp+reclaim", "winter_base_severity": 1.5, "expansion_pop_threshold": 20, "expansion_prob": 0.20, "ruin_reclaim_prob": 0.08},
    {"name": "Sev 2.0 linear 0.01", "winter_base_severity": 2.0, "expansion_pop_threshold": 20, "expansion_prob": 0.20},
]

n_sims = 20  # per map seed
map_seeds = [42, 123, 7, 999, 314]  # diverse test maps

for cfg in test_configs:
    name = cfg.pop("name")
    p = HiddenParams(**cfg)

    all_sett_pcts = []
    for ms in map_seeds:
        sim = AstarIslandSimulator(map_seed=ms, params=p)
        grid_init = sim.base_grid.copy()

        for sim_seed in range(n_sims):
            grid, setts = sim.run(sim_seed)
            dynamic = 0
            sett = 0
            for y in range(40):
                for x in range(40):
                    if grid_init[y, x] in (10, 5):
                        continue
                    dynamic += 1
                    cls = CODE_TO_CLASS.get(int(grid[y, x]), 0)
                    if cls == 1:
                        sett += 1
            all_sett_pcts.append(sett / dynamic if dynamic > 0 else 0)

    avg = np.mean(all_sett_pcts)
    std = np.std(all_sett_pcts)
    print(f"  {name:25s}: settlement={100*avg:.1f}% +/- {100*std:.1f}%  (target ~{100*gt_avg:.1f}%)")
