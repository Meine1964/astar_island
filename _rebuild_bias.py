"""Rebuild bias_correction.json with finer distance buckets from local data.

New buckets: [0], [1], [2], [3], [4-5], [6-7], [8+]
Also computes per-(init_class, dist_bucket) priors for the learned domain prior.
"""
import json
import numpy as np
from pathlib import Path

CODE_TO_CLASS = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
NUM_CLASSES = 6


def dist_to_bucket(d):
    """Finer distance buckets: 0,1,2,3, 4-5, 6-7, 8+."""
    if d <= 3:
        return d
    elif d <= 5:
        return 4
    elif d <= 7:
        return 5
    else:
        return 6


def compute_features_fine(init_grid, settlements, W, H):
    """Compute per-cell features with finer distance buckets."""
    dist_map = np.full((H, W), 999, dtype=int)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        for y in range(H):
            for x in range(W):
                dist_map[y, x] = min(dist_map[y, x], abs(x - sx) + abs(y - sy))

    coastal = np.zeros((H, W), dtype=int)
    for y in range(H):
        for x in range(W):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and init_grid[ny][nx] == 10:
                    coastal[y, x] = 1
                    break

    features = {}
    for y in range(H):
        for x in range(W):
            ic = CODE_TO_CLASS.get(init_grid[y][x], 0)
            db = dist_to_bucket(dist_map[y, x])
            co = coastal[y, x]
            features[(y, x)] = (ic, db, co)
    return features


def main():
    data_dir = Path("data")

    # Collect per-bucket distributions from all completed rounds
    by_dist_bucket = {}      # db -> list of class counts
    by_ic_db = {}            # (init_class, dist_bucket) -> list of class probs
    by_ic_db_co = {}         # (init_class, dist_bucket, coastal) -> list
    n_seed_rounds = 0

    for round_dir in sorted(data_dir.glob("round_*")):
        detail_path = round_dir / "round_detail.json"
        analysis_dir = round_dir / "analysis"
        if not detail_path.exists() or not analysis_dir.exists():
            continue

        with open(detail_path) as f:
            detail = json.load(f)
        W, H = detail["map_width"], detail["map_height"]

        for si in range(detail["seeds_count"]):
            gt_path = analysis_dir / f"seed_{si}_ground_truth.npy"
            if not gt_path.exists():
                continue

            gt = np.load(gt_path)
            state = detail["initial_states"][si]
            setts = [s for s in state["settlements"] if s.get("alive", True)]
            feats = compute_features_fine(state["grid"], setts, W, H)

            for y in range(H):
                for x in range(W):
                    cell = state["grid"][y][x]
                    if cell == 5 or cell == 10:
                        continue  # static cells
                    ic, db, co = feats[(y, x)]

                    # Get GT distribution for this cell
                    if gt.ndim == 3:
                        gt_dist = gt[y, x]
                    else:
                        gt_dist = np.zeros(NUM_CLASSES)
                        gt_dist[CODE_TO_CLASS.get(int(gt[y, x]), 0)] = 1.0

                    # By distance bucket
                    if db not in by_dist_bucket:
                        by_dist_bucket[db] = np.zeros(NUM_CLASSES)
                    by_dist_bucket[db] += gt_dist

                    # By (init_class, dist_bucket)
                    key = (ic, db)
                    if key not in by_ic_db:
                        by_ic_db[key] = {"sum": np.zeros(NUM_CLASSES), "n": 0}
                    by_ic_db[key]["sum"] += gt_dist
                    by_ic_db[key]["n"] += 1

                    # By (init_class, dist_bucket, coastal)
                    key3 = (ic, db, co)
                    if key3 not in by_ic_db_co:
                        by_ic_db_co[key3] = {"sum": np.zeros(NUM_CLASSES), "n": 0}
                    by_ic_db_co[key3]["sum"] += gt_dist
                    by_ic_db_co[key3]["n"] += 1

            n_seed_rounds += 1

    print(f"Processed {n_seed_rounds} seed-rounds")

    # Normalize and build output
    gt_by_dist = {}
    total_dynamic = np.zeros(NUM_CLASSES)
    total_count = 0
    for db in sorted(by_dist_bucket):
        s = by_dist_bucket[db]
        if s.sum() > 0:
            gt_by_dist[str(db)] = (s / s.sum()).tolist()
            total_dynamic += s
            total_count += s.sum()
            print(f"  Bucket {db}: n={s.sum():.0f} -> "
                  f"[{', '.join(f'{v:.4f}' for v in s / s.sum())}]")

    gt_dynamic_avg = (total_dynamic / total_dynamic.sum()).tolist() if total_dynamic.sum() > 0 else [0] * 6

    # Learned priors: (init_class, dist_bucket) -> probabilities
    learned_priors = {}
    print("\nLearned priors by (init_class, dist_bucket):")
    for key in sorted(by_ic_db):
        d = by_ic_db[key]
        if d["n"] >= 10:
            probs = d["sum"] / d["sum"].sum()
            key_str = f"{key[0]}_{key[1]}"
            learned_priors[key_str] = probs.tolist()
            print(f"  ic={key[0]} db={key[1]}: n={d['n']} -> "
                  f"[{', '.join(f'{v:.3f}' for v in probs)}]")

    # Learned priors with coastal: (init_class, dist_bucket, coastal)
    learned_priors_coastal = {}
    print("\nLearned priors by (init_class, dist_bucket, coastal):")
    for key in sorted(by_ic_db_co):
        d = by_ic_db_co[key]
        if d["n"] >= 5:
            probs = d["sum"] / d["sum"].sum()
            key_str = f"{key[0]}_{key[1]}_{key[2]}"
            learned_priors_coastal[key_str] = probs.tolist()
            count_str = f"n={d['n']}"
            print(f"  ic={key[0]} db={key[1]} co={key[2]}: {count_str} -> "
                  f"[{', '.join(f'{v:.3f}' for v in probs)}]")

    output = {
        "n_seed_rounds": n_seed_rounds,
        "gt_dynamic_avg": gt_dynamic_avg,
        "gt_by_dist_bucket": gt_by_dist,
        "learned_priors": learned_priors,
        "learned_priors_coastal": learned_priors_coastal,
    }

    with open("bias_correction.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved bias_correction.json with {len(learned_priors)} priors, "
          f"{len(learned_priors_coastal)} coastal priors")


if __name__ == "__main__":
    main()
