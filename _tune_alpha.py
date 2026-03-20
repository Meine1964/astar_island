"""Find optimal bias correction strength using Round 7 ground truth.
Tests different alpha values with observation-aware weighting.
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

with open("bias_correction.json") as f:
    bc = json.load(f)
gt_by_dist = {int(k): np.array(v) for k, v in bc["gt_by_dist_bucket"].items()}

eps = 1e-10
MIN_PROB = 0.005

def score_pred(pred, gt):
    gt_ent = -np.sum(gt * np.log(gt + eps), axis=2)
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
    return (gt_ent.sum() - (gt_ent * kl).sum())

# Fetch Round 7 data
rounds = s.get(f"{B}/rounds").json()
r7 = next(r for r in rounds if r["round_number"] == 7)
r7id = r7["id"]
detail = s.get(f"{B}/rounds/{r7id}").json()
W, H = detail["map_width"], detail["map_height"]

# Cache all seed data
seed_data = []
for si in range(5):
    a = s.get(f"{B}/analysis/{r7id}/{si}").json()
    gt = np.array(a["ground_truth"])
    pred = np.array(a["prediction"])
    init_grid = np.array(a["initial_grid"])
    
    setts = detail["initial_states"][si]["settlements"]
    alive = [ss for ss in setts if ss.get("alive", True)]
    dist_map = np.full((H, W), 999)
    for ss in alive:
        sx, sy = ss["x"], ss["y"]
        for y in range(H):
            for x in range(W):
                dist_map[y, x] = min(dist_map[y, x], abs(x - sx) + abs(y - sy))

    # Compute dist_bucket map
    db_map = np.zeros((H, W), dtype=int)
    for y in range(H):
        for x in range(W):
            d = dist_map[y, x]
            if d <= 1: db_map[y, x] = 0
            elif d <= 3: db_map[y, x] = 1
            elif d <= 5: db_map[y, x] = 2
            elif d <= 7: db_map[y, x] = 3
            elif d <= 10: db_map[y, x] = 4
            else: db_map[y, x] = 5

    seed_data.append({"gt": gt, "pred": pred, "init_grid": init_grid,
                      "db_map": db_map})
    time.sleep(0.3)

# Test different alpha values — using observation-count-aware formula
# alpha = base_alpha / (1 + n * decay)
# We don't have exact n, but can test the total effect on unobserved cells
# by applying correction only to cells where the model is uncertain

print("=== Testing bias correction alpha values ===")
print(f"{'alpha':>8} {'decay':>6} | {'S0':>8} {'S1':>8} {'S2':>8} {'S3':>8} {'S4':>8} | {'Total':>8} {'Delta':>8}")
print("-" * 85)

# Baseline: no correction
baseline_total = 0
for sd in seed_data:
    baseline_total += score_pred(sd["pred"], sd["gt"])
scores_base = [score_pred(sd["pred"], sd["gt"]) for sd in seed_data]
print(f"{'none':>8} {'--':>6} | {' '.join(f'{s:>8.2f}' for s in scores_base)} | {baseline_total:>8.2f} {'0.00':>8}")

for base_alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
    for decay in [0.3, 0.5, 1.0]:
        scores = []
        for sd in seed_data:
            pred_c = sd["pred"].copy()
            for y in range(H):
                for x in range(W):
                    cell = sd["init_grid"][y, x]
                    if cell == 5 or cell == 10:
                        continue
                    db = sd["db_map"][y, x]
                    if db not in gt_by_dist:
                        continue
                    target = gt_by_dist[db]
                    # Estimate n: if confidence is high, cell was likely observed
                    max_p = pred_c[y, x].max()
                    # Higher confidence -> higher estimated n
                    est_n = max(0, (max_p - 0.5) * 10) if max_p > 0.5 else 0
                    alpha = base_alpha / (1.0 + est_n * decay)
                    pred_c[y, x] = (1 - alpha) * pred_c[y, x] + alpha * target
            pred_c = np.maximum(pred_c, MIN_PROB)
            pred_c /= pred_c.sum(axis=2, keepdims=True)
            scores.append(score_pred(pred_c, sd["gt"]))
        
        total = sum(scores)
        delta = total - baseline_total
        marker = " <-- best" if delta == max(0.01, delta) else ""
        print(f"{base_alpha:>8.2f} {decay:>6.1f} | {' '.join(f'{s:>8.2f}' for s in scores)} | {total:>8.2f} {delta:>+8.2f}{marker}")

# Also test: only correct LOW-confidence cells (< threshold)
print("\n=== Threshold-based correction (only correct uncertain cells) ===")
for conf_thresh in [0.5, 0.6, 0.7, 0.8]:
    for alpha in [0.15, 0.25, 0.35]:
        scores = []
        for sd in seed_data:
            pred_c = sd["pred"].copy()
            for y in range(H):
                for x in range(W):
                    cell = sd["init_grid"][y, x]
                    if cell == 5 or cell == 10:
                        continue
                    if pred_c[y, x].max() > conf_thresh:
                        continue  # skip confident cells
                    db = sd["db_map"][y, x]
                    if db not in gt_by_dist:
                        continue
                    pred_c[y, x] = (1 - alpha) * pred_c[y, x] + alpha * gt_by_dist[db]
            pred_c = np.maximum(pred_c, MIN_PROB)
            pred_c /= pred_c.sum(axis=2, keepdims=True)
            scores.append(score_pred(pred_c, sd["gt"]))
        total = sum(scores)
        delta = total - baseline_total
        print(f"  conf<{conf_thresh}, alpha={alpha:.2f}: "
              f"total={total:.2f}, delta={delta:+.2f}")
