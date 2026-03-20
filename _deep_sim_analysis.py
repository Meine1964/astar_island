"""Deep sim-vs-GT analysis to find concrete simulator improvements."""
import numpy as np, json, os
from astar_island_simulator.env import (
    AstarIslandSimulator, HiddenParams, Settlement as SimSettlement,
    CODE_TO_CLASS, NUM_CLASSES,
)
from strategy import compute_features, MIN_PROB

def simulate_seed(detail, si, n_sims=30):
    """Run simulator on a real round's initial state."""
    state = detail["initial_states"][si]
    init_grid = state["grid"]
    setts = [s for s in state["settlements"] if s.get("alive", True)]
    W, H = detail["map_width"], detail["map_height"]

    grid_np = np.array(init_grid, dtype=int)
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
    sim.width = W; sim.height = H
    sim.base_grid = grid_np
    sim.base_settlements = sett_objs

    counts = np.zeros((H, W, NUM_CLASSES))
    sett_survival = []
    for i in range(n_sims):
        final_grid, final_setts = sim.run(sim_seed=i)
        alive = sum(1 for s in final_setts if s.alive)
        sett_survival.append(alive / max(len(sett_objs), 1))
        for y in range(H):
            for x in range(W):
                cls = CODE_TO_CLASS.get(int(final_grid[y, x]), 0)
                counts[y, x, cls] += 1

    return counts / n_sims, np.mean(sett_survival)


# Load a few rounds and compare
print("=== Simulator vs Ground Truth Analysis ===\n")

eps = 1e-10
class_names = ["plains", "sett", "port", "ruin", "forest", "mtn"]

# Aggregate class frequencies: sim vs GT
sim_class_totals = np.zeros(NUM_CLASSES)
gt_class_totals = np.zeros(NUM_CLASSES)
sim_class_dynamic = np.zeros(NUM_CLASSES)
gt_class_dynamic = np.zeros(NUM_CLASSES)
n_dynamic_total = 0
n_cells_total = 0

# Per-distance-bucket comparison
dist_sim = {}  # bucket -> (6,) avg sim dist
dist_gt = {}   # bucket -> (6,) avg gt dist
dist_n = {}

for rnd in [1, 3, 5, 7, 9]:
    rdir = f"data/round_{rnd:02d}"
    if not os.path.exists(os.path.join(rdir, "round_detail.json")):
        continue
    detail = json.load(open(os.path.join(rdir, "round_detail.json")))
    W, H = detail["map_width"], detail["map_height"]

    for si in [0]:  # 1 seed per round for speed
        gt_path = os.path.join(rdir, "analysis", f"seed_{si}_ground_truth.npy")
        if not os.path.exists(gt_path):
            continue
        gt = np.load(gt_path)
        sim_dist, surv_rate = simulate_seed(detail, si, n_sims=30)

        state = detail["initial_states"][si]
        setts = [s for s in state["settlements"] if s.get("alive", True)]
        feats = compute_features(state["grid"], setts, W, H)

        gt_entropy = -np.sum(gt * np.log(gt + eps), axis=2)

        for y in range(H):
            for x in range(W):
                sim_class_totals += sim_dist[y, x]
                gt_class_totals += gt[y, x]
                n_cells_total += 1

                if gt_entropy[y, x] > 0.01:
                    sim_class_dynamic += sim_dist[y, x]
                    gt_class_dynamic += gt[y, x]
                    n_dynamic_total += 1

                    db = int(feats[y, x, 1])
                    if db not in dist_sim:
                        dist_sim[db] = np.zeros(NUM_CLASSES)
                        dist_gt[db] = np.zeros(NUM_CLASSES)
                        dist_n[db] = 0
                    dist_sim[db] += sim_dist[y, x]
                    dist_gt[db] += gt[y, x]
                    dist_n[db] += 1

        # Per-round summary
        gt_argmax = gt.argmax(axis=2)
        sim_argmax = sim_dist.argmax(axis=2)
        dynamic = gt_entropy > 0.01
        agree = (gt_argmax == sim_argmax) & dynamic
        n_dyn = dynamic.sum()
        print(f"R{rnd} S{si}: {int(n_dyn)} dynamic cells, "
              f"argmax agreement={agree.sum()}/{n_dyn} ({agree.sum()/max(n_dyn,1)*100:.0f}%), "
              f"survival rate={surv_rate:.2f}, "
              f"n_setts={len(setts)}")

# Overall class distribution comparison
print("\n--- Overall class frequencies (dynamic cells) ---")
print(f"{'Class':>10}  {'GT':>8}  {'Sim':>8}  {'Diff':>8}")
sim_norm = sim_class_dynamic / max(n_dynamic_total, 1)
gt_norm = gt_class_dynamic / max(n_dynamic_total, 1)
for c in range(NUM_CLASSES):
    diff = sim_norm[c] - gt_norm[c]
    marker = " **" if abs(diff) > 0.02 else ""
    print(f"{class_names[c]:>10}  {gt_norm[c]:>8.4f}  {sim_norm[c]:>8.4f}  {diff:>+8.4f}{marker}")

# Per-distance comparison
print("\n--- Class distribution by distance bucket (dynamic cells) ---")
for db in sorted(dist_gt):
    n = dist_n[db]
    gn = dist_gt[db] / n
    sn = dist_sim[db] / n
    print(f"\n  Bucket {db} ({n} cells):")
    for c in range(NUM_CLASSES):
        if gn[c] > 0.01 or sn[c] > 0.01:
            diff = sn[c] - gn[c]
            marker = " !!" if abs(diff) > 0.05 else ""
            print(f"    {class_names[c]:>8}: GT={gn[c]:.3f} Sim={sn[c]:.3f} diff={diff:+.3f}{marker}")

# Check what cells sim gets most wrong
print("\n--- Top error patterns (sim argmax != gt argmax) ---")
error_patterns = {}  # (sim_cls, gt_cls) -> count
for rnd in [1, 3, 5, 7, 9]:
    rdir = f"data/round_{rnd:02d}"
    if not os.path.exists(os.path.join(rdir, "round_detail.json")):
        continue
    detail = json.load(open(os.path.join(rdir, "round_detail.json")))
    W, H = detail["map_width"], detail["map_height"]
    for si in [0]:
        gt_path = os.path.join(rdir, "analysis", f"seed_{si}_ground_truth.npy")
        if not os.path.exists(gt_path):
            continue
        gt = np.load(gt_path)
        sim_dist, _ = simulate_seed(detail, si, n_sims=20)
        gt_entropy = -np.sum(gt * np.log(gt + eps), axis=2)
        for y in range(H):
            for x in range(W):
                if gt_entropy[y, x] > 0.01:
                    sc = sim_dist[y, x].argmax()
                    gc = gt[y, x].argmax()
                    if sc != gc:
                        key = (class_names[sc], class_names[gc])
                        error_patterns[key] = error_patterns.get(key, 0) + 1

sorted_errors = sorted(error_patterns.items(), key=lambda kv: -kv[1])
for (sim_cls, gt_cls), count in sorted_errors[:10]:
    print(f"  Sim predicts {sim_cls:>8}, GT is {gt_cls:>8}: {count} cells")
