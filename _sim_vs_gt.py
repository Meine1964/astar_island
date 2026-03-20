"""Compare simulator output vs real ground truth to find systematic errors.
Uses saved data from data/ directory.
"""
import numpy as np
import json
import os
import data_store
from astar_island_simulator.env import (
    AstarIslandSimulator, HiddenParams, Settlement,
    CODE_TO_CLASS, NUM_CLASSES, OCEAN, PLAINS, RUIN, PORT, SETTLEMENT, MOUNTAIN
)

# Load default params
params = HiddenParams()

# Check rounds 7 and 8 (most recent, most relevant)
for rnum in [7, 8]:
    detail = data_store.load_round_detail(rnum)
    if detail is None:
        print(f"Round {rnum}: no saved detail")
        continue

    W, H = detail["map_width"], detail["map_height"]
    seeds = detail["seeds_count"]
    print(f"\n{'='*60}")
    print(f"Round #{rnum}: {W}x{H}, {seeds} seeds")
    print(f"{'='*60}")

    for si in range(min(seeds, 2)):  # just check 2 seeds per round for speed
        gt, score = data_store.load_analysis(rnum, si)
        if gt is None:
            print(f"  Seed {si}: no ground truth")
            continue

        state = detail["initial_states"][si]
        init_grid = state["grid"]
        init_setts = [s for s in state["settlements"] if s.get("alive", True)]

        # Run our simulator
        grid_np = np.array(init_grid, dtype=int)
        settlements_base = []
        for i, s in enumerate(init_setts):
            settlements_base.append(Settlement(
                x=s["x"], y=s["y"],
                population=10, food=5.0, wealth=1.0, defense=3.0,
                tech_level=1.0, has_port=s.get("has_port", False),
                owner_id=i,
            ))

        sim = AstarIslandSimulator.__new__(AstarIslandSimulator)
        sim.map_seed = 0
        sim.params = params
        sim.width = W
        sim.height = H
        sim.base_grid = grid_np
        sim.base_settlements = settlements_base

        # Run 50 sims
        sim_dist = np.zeros((H, W, NUM_CLASSES))
        n_sims = 50
        for i in range(n_sims):
            final_grid, final_setts = sim.run(sim_seed=2000 + i)
            for y in range(H):
                for x in range(W):
                    cls = CODE_TO_CLASS.get(int(final_grid[y, x]), 0)
                    sim_dist[y, x, cls] += 1
        sim_dist /= n_sims

        # Compare GT vs sim on dynamic cells
        eps = 1e-10
        gt_entropy = -np.sum(gt * np.log(gt + eps), axis=2)
        dynamic = gt_entropy > 0.01
        n_dynamic = dynamic.sum()

        gt_dyn = gt[dynamic]
        sim_dyn = sim_dist[dynamic]

        gt_avg = gt_dyn.mean(axis=0)
        sim_avg = sim_dyn.mean(axis=0)
        diff = sim_avg - gt_avg

        print(f"\n  Seed {si}: {n_dynamic} dynamic cells, real score={score}")
        print(f"  Class:     [plains,  field,   ruin,    port,    sett,    mount]")
        print(f"  GT avg:    [{', '.join(f'{v:.4f}' for v in gt_avg)}]")
        print(f"  Sim avg:   [{', '.join(f'{v:.4f}' for v in sim_avg)}]")
        print(f"  Diff:      [{', '.join(f'{v:+.4f}' for v in diff)}]")

        # Per distance bucket analysis
        dist_map = np.full((H, W), 999)
        for s in init_setts:
            sx, sy = s["x"], s["y"]
            for y in range(H):
                for x in range(W):
                    dist_map[y, x] = min(dist_map[y, x], abs(x - sx) + abs(y - sy))

        print(f"\n  By distance to settlement:")
        for db, (lo, hi) in enumerate([(0,1),(2,3),(4,5),(6,7),(8,10),(11,99)]):
            mask = dynamic & (dist_map >= lo) & (dist_map <= hi)
            if mask.sum() < 10:
                continue
            gt_b = gt[mask].mean(axis=0)
            sim_b = sim_dist[mask].mean(axis=0)
            d = sim_b - gt_b
            print(f"    d={lo}-{hi} (n={mask.sum():>4}): "
                  f"GT=[{gt_b[0]:.3f},{gt_b[1]:.3f},{gt_b[2]:.3f},{gt_b[3]:.3f},{gt_b[4]:.3f}] "
                  f"Sim=[{sim_b[0]:.3f},{sim_b[1]:.3f},{sim_b[2]:.3f},{sim_b[3]:.3f},{sim_b[4]:.3f}] "
                  f"err=[{d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f},{d[3]:+.3f},{d[4]:+.3f}]")

        # Specific checks: settlement survival and expansion
        gt_argmax = gt.argmax(axis=2)
        sim_argmax = sim_dist.argmax(axis=2)

        # Count cells where settlement is most likely class
        gt_sett_cells = (gt_argmax == 4).sum()
        sim_sett_cells = (sim_argmax == 4).sum()
        gt_field_cells = (gt_argmax == 1).sum()
        sim_field_cells = (sim_argmax == 1).sum()
        gt_ruin_cells = (gt_argmax == 2).sum()
        sim_ruin_cells = (sim_argmax == 2).sum()
        gt_port_cells = (gt_argmax == 3).sum()
        sim_port_cells = (sim_argmax == 3).sum()

        print(f"\n  Argmax cell counts:")
        print(f"    Settlements: GT={gt_sett_cells}, Sim={sim_sett_cells}")
        print(f"    Fields:      GT={gt_field_cells}, Sim={sim_field_cells}")
        print(f"    Ruins:       GT={gt_ruin_cells}, Sim={sim_ruin_cells}")
        print(f"    Ports:       GT={gt_port_cells}, Sim={sim_port_cells}")
