"""Analyze ground truth vs our predictions for completed rounds.
Find systematic biases in our model.
"""
import requests, truststore, numpy as np, time
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

# Check Round 7 in detail (the one we actually participated in properly)
print("=== Round 7 per-seed analysis ===")
r7 = next(r for r in completed if r["round_number"] == 7)
r7id = r7["id"]
detail = s.get(f"{B}/rounds/{r7id}").json()
W, H = detail["map_width"], detail["map_height"]

for si in range(5):
    a = s.get(f"{B}/analysis/{r7id}/{si}").json()
    gt = np.array(a["ground_truth"])
    pred = np.array(a["prediction"])
    score = a["score"]
    init = np.array(a["initial_grid"])

    # Entropy of ground truth (tells us which cells are dynamic)
    eps = 1e-10
    gt_entropy = -np.sum(gt * np.log(gt + eps), axis=2)
    n_dynamic = (gt_entropy > 0.01).sum()
    avg_dynamic_entropy = gt_entropy[gt_entropy > 0.01].mean() if n_dynamic > 0 else 0

    # KL divergence per cell (gt || pred)
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
    weighted_kl = gt_entropy * kl
    total_weighted_kl = weighted_kl.sum()

    # What does ground truth look like? Class distribution
    gt_argmax = gt.argmax(axis=2)
    gt_class_counts = np.bincount(gt_argmax.ravel(), minlength=6)

    # What did we predict? 
    pred_argmax = pred.argmax(axis=2)
    pred_class_counts = np.bincount(pred_argmax.ravel(), minlength=6)

    # Where is the biggest error?
    worst_cells = np.unravel_index(np.argsort(weighted_kl.ravel())[-5:], (H, W))

    print(f"\n  Seed {si}: score={score:.2f}")
    print(f"    Dynamic cells: {n_dynamic}, avg entropy: {avg_dynamic_entropy:.3f}")
    print(f"    Total weighted KL: {total_weighted_kl:.4f}")
    print(f"    GT class dist:   {gt_class_counts} (plains={gt_class_counts[0]}, "
          f"field={gt_class_counts[1]}, ruin={gt_class_counts[2]}, "
          f"port={gt_class_counts[3]}, sett={gt_class_counts[4]}, mount={gt_class_counts[5]})")
    print(f"    Pred class dist: {pred_class_counts}")

    # For dynamic cells: what's the average predicted vs actual probability per class?
    dynamic_mask = gt_entropy > 0.01
    if dynamic_mask.sum() > 0:
        gt_dynamic = gt[dynamic_mask]  # (n_dynamic, 6)
        pred_dynamic = pred[dynamic_mask]
        print(f"    Dynamic cells avg GT:   [{', '.join(f'{v:.3f}' for v in gt_dynamic.mean(axis=0))}]")
        print(f"    Dynamic cells avg Pred: [{', '.join(f'{v:.3f}' for v in pred_dynamic.mean(axis=0))}]")
        diff = pred_dynamic.mean(axis=0) - gt_dynamic.mean(axis=0)
        print(f"    Bias (pred - gt):       [{', '.join(f'{v:+.3f}' for v in diff)}]")

    time.sleep(0.3)

# Summary across all 7 completed rounds
print("\n\n=== All rounds scores ===")
for rnd in sorted(completed, key=lambda r: r["round_number"]):
    rid = rnd["id"]
    scores = []
    for si in range(5):
        try:
            a = s.get(f"{B}/analysis/{rid}/{si}").json()
            scores.append(a.get("score", "?"))
        except:
            scores.append("?")
        time.sleep(0.25)
    total = sum(sc for sc in scores if isinstance(sc, (int, float)))
    print(f"  Round #{rnd['round_number']}: seeds={scores}, total={total:.1f}")
