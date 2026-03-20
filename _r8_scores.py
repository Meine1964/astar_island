"""Quick Round 8 score check."""
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
r8 = next(r for r in rounds if r["round_number"] == 8)
r8id = r8["id"]

print("=== Round 8 per-seed scores ===")
total = 0
for si in range(5):
    a = s.get(f"{B}/analysis/{r8id}/{si}").json()
    score = a.get("score", "N/A")
    gt = np.array(a["ground_truth"])
    pred = np.array(a["prediction"])
    gt_entropy = -np.sum(gt * np.log(gt + 1e-10), axis=2)
    n_dynamic = (gt_entropy > 0.01).sum()
    if isinstance(score, (int, float)):
        total += score
    print(f"  Seed {si}: score={score}, dynamic_cells={n_dynamic}")
    time.sleep(0.25)

print(f"\n  TOTAL: {total:.1f}")
print(f"  Round 7 was: 299.4 (rank 85)")
print(f"  Round 8 rank: 66")

# Also check Round 7 for comparison
r7 = next(r for r in rounds if r["round_number"] == 7)
r7id = r7["id"]
print("\n=== Round 7 per-seed scores (for comparison) ===")
for si in range(5):
    a = s.get(f"{B}/analysis/{r7id}/{si}").json()
    print(f"  Seed {si}: score={a.get('score')}")
    time.sleep(0.25)
