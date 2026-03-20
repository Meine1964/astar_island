"""Compute precise bias correction factors from all completed round ground truth.
Looks at ALL seeds from ALL completed rounds (not just ones we participated in).
"""
import requests, truststore, numpy as np, time, json
truststore.inject_into_ssl()

s = requests.Session()
s.cookies.set("access_token",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJlMDcyNjM1ZC1mYTNmLTQ5MzMtOGMwNC1lMmJmYmM4ZDhiZDEi"
    "LCJlbWFpbCI6Im1laW5lLnZhbi5kZXIubWV1bGVuQGdtYWlsLmNvbSIsImlz"
    "X2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTEyMDI4fQ."
    "QMd3aqRnowq1zyiyFDnOu0bXSNSAZ2rEMaIoCkOQLJ4"
)
B = "https://api.ainm.no/astar-island"

rounds = s.get(f"{B}/rounds").json()
completed = [r for r in rounds if r["status"] == "completed"]

# Accumulate ground truth class frequencies for dynamic cells across ALL rounds
all_gt_dynamic = []  # list of (6,) arrays: avg GT prob per dynamic cell
all_gt_by_dist = {}  # dist_bucket -> list of (6,) GT probs

print(f"Analyzing {len(completed)} completed rounds...")

for rnd in sorted(completed, key=lambda r: r["round_number"]):
    rid = rnd["id"]
    rnum = rnd["round_number"]
    detail = s.get(f"{B}/rounds/{rid}").json()
    W, H = detail["map_width"], detail["map_height"]
    seeds = detail["seeds_count"]

    for si in range(seeds):
        try:
            a = s.get(f"{B}/analysis/{rid}/{si}").json()
        except:
            continue
        gt = np.array(a["ground_truth"])
        init_grid = np.array(a["initial_grid"])

        eps = 1e-10
        gt_entropy = -np.sum(gt * np.log(gt + eps), axis=2)
        dynamic = gt_entropy > 0.01

        if dynamic.sum() > 0:
            gt_dyn = gt[dynamic]
            all_gt_dynamic.append(gt_dyn.mean(axis=0))

            # Also look at per-cell detail
            setts = detail["initial_states"][si]["settlements"]
            alive_setts = [ss for ss in setts if ss.get("alive", True)]

            # dist map
            dist_map = np.full((H, W), 999)
            for ss in alive_setts:
                sx, sy = ss["x"], ss["y"]
                for y in range(H):
                    for x in range(W):
                        dist_map[y, x] = min(dist_map[y, x], abs(x - sx) + abs(y - sy))

            # For each dynamic cell, record GT by distance bucket
            for y in range(H):
                for x in range(W):
                    if dynamic[y, x]:
                        d = dist_map[y, x]
                        if d <= 1:
                            db = 0
                        elif d <= 3:
                            db = 1
                        elif d <= 5:
                            db = 2
                        elif d <= 7:
                            db = 3
                        elif d <= 10:
                            db = 4
                        else:
                            db = 5
                        if db not in all_gt_by_dist:
                            all_gt_by_dist[db] = []
                        all_gt_by_dist[db].append(gt[y, x])

        time.sleep(0.15)
    print(f"  Round #{rnum}: processed {seeds} seeds")

# Aggregate
gt_avg = np.mean(all_gt_dynamic, axis=0)
print(f"\n=== Average ground truth for dynamic cells (across {len(all_gt_dynamic)} seed-rounds) ===")
print(f"  Class names: [plains, field, ruin, port, settlement, mountain]")
print(f"  GT average:  [{', '.join(f'{v:.4f}' for v in gt_avg)}]")

print(f"\n=== Ground truth by distance to nearest settlement ===")
for db in sorted(all_gt_by_dist.keys()):
    arr = np.array(all_gt_by_dist[db])
    avg = arr.mean(axis=0)
    print(f"  dist_bucket={db}: n={len(arr):>5}, "
          f"[{', '.join(f'{v:.3f}' for v in avg)}]")

# Save the correction factors
correction = {
    "gt_dynamic_avg": gt_avg.tolist(),
    "n_seed_rounds": len(all_gt_dynamic),
    "gt_by_dist_bucket": {str(db): np.array(v).mean(axis=0).tolist()
                          for db, v in all_gt_by_dist.items()},
}
with open("bias_correction.json", "w") as f:
    json.dump(correction, f, indent=2)
print("\nSaved to bias_correction.json")
