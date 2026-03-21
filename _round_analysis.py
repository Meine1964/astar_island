"""Analyze all completed rounds to find patterns and lessons from missed rounds."""
import numpy as np
import json
import os

class_names = ["plains", "sett", "port", "ruin", "forest", "mtn"]

print("=== Round-by-Round Ground Truth Analysis ===\n")

all_class_avgs = []
for rnd in range(1, 13):
    rdir = f"data/round_{rnd:02d}"
    detail_path = os.path.join(rdir, "round_detail.json")
    if not os.path.exists(detail_path):
        continue

    detail = json.load(open(detail_path))
    W, H = detail["map_width"], detail["map_height"]
    n_seeds = detail["seeds_count"]
    n_setts_per_seed = []
    for si in range(n_seeds):
        setts = [s for s in detail["initial_states"][si]["settlements"] if s.get("alive", True)]
        n_setts_per_seed.append(len(setts))

    # Check for GT
    gt_avgs = np.zeros(6)
    gt_count = 0
    seed_scores = []
    for si in range(n_seeds):
        gt_path = os.path.join(rdir, "analysis", f"seed_{si}_ground_truth.npy")
        score_path = os.path.join(rdir, "analysis", f"seed_{si}_score.json")
        if not os.path.exists(gt_path):
            continue
        gt = np.load(gt_path)
        grid_init = np.array(detail["initial_states"][si]["grid"], dtype=int)

        for y in range(H):
            for x in range(W):
                if grid_init[y, x] in (10, 5):
                    continue
                gt_avgs += gt[y, x]
                gt_count += 1

        if os.path.exists(score_path):
            sc = json.load(open(score_path))
            if sc is not None and isinstance(sc, dict) and sc.get("score") is not None:
                seed_scores.append(sc["score"])
            elif sc is not None and isinstance(sc, (int, float)):
                seed_scores.append(sc)

    if gt_count > 0:
        gt_avgs /= gt_count
        all_class_avgs.append((rnd, gt_avgs))

    # Our participation
    my_queries = 0
    obs_path = os.path.join(rdir, "observations.npz")
    if os.path.exists(obs_path):
        my_queries = 50  # we used them

    score_str = ""
    if seed_scores:
        score_str = f"  our_score={sum(seed_scores):.1f} (avg/seed={np.mean(seed_scores):.1f})"

    status = "PLAYED" if my_queries > 0 else "MISSED"
    print(f"R{rnd:2d} [{status}]: {n_seeds} seeds, setts={n_setts_per_seed}, "
          f"class_dist=[{', '.join(f'{v:.3f}' for v in gt_avgs)}]{score_str}")

# Cross-round variation
print("\n=== Class Distribution Variation Across Rounds ===")
print(f"{'Round':>6} {'plains':>8} {'sett':>8} {'port':>8} {'ruin':>8} {'forest':>8} {'mtn':>8}")
for rnd, avgs in all_class_avgs:
    print(f"R{rnd:2d}     {avgs[0]:8.3f} {avgs[1]:8.3f} {avgs[2]:8.3f} {avgs[3]:8.3f} {avgs[4]:8.3f} {avgs[5]:8.3f}")

avgs_array = np.array([a for _, a in all_class_avgs])
print(f"{'Mean':>6} {avgs_array[:,0].mean():8.3f} {avgs_array[:,1].mean():8.3f} "
      f"{avgs_array[:,2].mean():8.3f} {avgs_array[:,3].mean():8.3f} "
      f"{avgs_array[:,4].mean():8.3f} {avgs_array[:,5].mean():8.3f}")
print(f"{'Std':>6} {avgs_array[:,0].std():8.3f} {avgs_array[:,1].std():8.3f} "
      f"{avgs_array[:,2].std():8.3f} {avgs_array[:,3].std():8.3f} "
      f"{avgs_array[:,4].std():8.3f} {avgs_array[:,5].std():8.3f}")

# Check if rounds we missed had unusual distributions
print("\n=== Key Observations ===")
mean_sett = avgs_array[:, 1].mean()
for rnd, avgs in all_class_avgs:
    sett_diff = avgs[1] - mean_sett
    if abs(sett_diff) > 0.03:
        print(f"  R{rnd}: settlement {avgs[1]:.3f} ({'HIGH' if sett_diff > 0 else 'LOW'}, "
              f"{sett_diff:+.3f} from mean {mean_sett:.3f})")

# Rebuild bias correction with all 11 rounds now
print(f"\n=== Bias Correction Data ===")
print(f"  Current bias_correction.json uses data from how many rounds?")
bc = json.load(open("bias_correction.json"))
print(f"  n_seed_rounds in file: {bc.get('n_seed_rounds', '?')}")
print(f"  Available GT seed-rounds: {sum(1 for _, _ in all_class_avgs) * 5}")
print(f"  -> Should rebuild bias_correction.json with R10+R11 data!")
