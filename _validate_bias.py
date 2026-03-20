"""Validate bias correction against Round 7 ground truth.
Computes what our score WOULD HAVE BEEN with bias correction applied
to our actual Round 7 predictions.
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

# Load bias correction
with open("bias_correction.json") as f:
    bc = json.load(f)
gt_by_dist = {int(k): np.array(v) for k, v in bc["gt_by_dist_bucket"].items()}

rounds = s.get(f"{B}/rounds").json()
r7 = next(r for r in rounds if r["round_number"] == 7)
r7id = r7["id"]
detail = s.get(f"{B}/rounds/{r7id}").json()
W, H = detail["map_width"], detail["map_height"]

eps = 1e-10
MIN_PROB = 0.005

def score_prediction(pred, gt, H, W):
    """Compute entropy-weighted KL divergence score."""
    gt_entropy = -np.sum(gt * np.log(gt + eps), axis=2)
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
    weighted_kl = (gt_entropy * kl).sum()
    max_score = gt_entropy.sum()
    return max_score - weighted_kl

print("=== Round 7: original vs bias-corrected scores ===\n")

for si in range(5):
    a = s.get(f"{B}/analysis/{r7id}/{si}").json()
    gt = np.array(a["ground_truth"])
    pred_orig = np.array(a["prediction"])
    init_grid = np.array(a["initial_grid"])

    # Compute distance map for this seed
    setts = detail["initial_states"][si]["settlements"]
    alive = [ss for ss in setts if ss.get("alive", True)]
    dist_map = np.full((H, W), 999)
    for ss in alive:
        sx, sy = ss["x"], ss["y"]
        for y in range(H):
            for x in range(W):
                dist_map[y, x] = min(dist_map[y, x], abs(x - sx) + abs(y - sy))

    # Apply bias correction to our original prediction
    pred_corrected = pred_orig.copy()
    for y in range(H):
        for x in range(W):
            cell = init_grid[y, x]
            if cell == 5 or cell == 10:
                continue
            d = dist_map[y, x]
            if d <= 1: db = 0
            elif d <= 3: db = 1
            elif d <= 5: db = 2
            elif d <= 7: db = 3
            elif d <= 10: db = 4
            else: db = 5
            if db not in gt_by_dist:
                continue
            target = gt_by_dist[db]
            # Same alpha as in build_prediction, assume n~2 avg for observed
            # But we don't know n here, so use moderate alpha=0.2
            alpha = 0.2
            pred_corrected[y, x] = (1 - alpha) * pred_corrected[y, x] + alpha * target

    pred_corrected = np.maximum(pred_corrected, MIN_PROB)
    pred_corrected /= pred_corrected.sum(axis=2, keepdims=True)

    score_orig = score_prediction(pred_orig, gt, H, W)
    score_corr = score_prediction(pred_corrected, gt, H, W)
    delta = score_corr - score_orig

    print(f"  Seed {si}: original={score_orig:.2f}, corrected={score_corr:.2f}, delta={delta:+.2f}")

    time.sleep(0.3)

print("\nNote: bias correction trained on Rounds 1-7, so Round 7 is in-sample.")
print("Real improvement on future rounds will be smaller but directionally similar.")
