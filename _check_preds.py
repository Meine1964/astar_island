"""Temp: check our Round 8 predictions and scores."""
import requests, truststore, numpy as np
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
RID = "c5cdf100-a876-4fb7-b5d8-757162c97989"

print("=== Our Round 8 predictions ===")
preds = s.get(f"{B}/my-predictions/{RID}").json()
for e in preds:
    si = e["seed_index"]
    score = e.get("score")
    conf = np.array(e["confidence_grid"]) if e.get("confidence_grid") else None
    argmax = np.array(e["argmax_grid"]) if e.get("argmax_grid") else None
    print(f"  Seed {si}: score={score}, submitted={e.get('submitted_at','?')}")
    if conf is not None:
        print(f"    confidence: shape={conf.shape}, mean={conf.mean():.3f}, min={conf.min():.3f}")
    if argmax is not None:
        print(f"    argmax: shape={argmax.shape}, unique classes={np.unique(argmax).tolist()}")

# Also check: what does the analysis look like for completed round 7?
print("\n=== Round 7 analysis sample (seed 0) ===")
# Find round 7 ID
rounds = s.get(f"{B}/rounds").json()
r7 = next(r for r in rounds if r["round_number"] == 7)
r7id = r7["id"]
analysis = s.get(f"{B}/analysis/{r7id}/0").json()
print(f"  Keys: {list(analysis.keys())}")
for k, v in analysis.items():
    if isinstance(v, list):
        arr = np.array(v)
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    elif isinstance(v, (int, float, str)):
        print(f"  {k}: {v}")
