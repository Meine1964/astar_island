"""Download ground truth for completed rounds we don't have yet."""
import requests, numpy as np, json, os, truststore
truststore.inject_into_ssl()
import data_store

BASE = "https://api.ainm.no/astar-island"
TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJlMDcyNjM1ZC1mYTNmLTQ5MzMtOGMwNC1lMmJmYmM4ZDhiZDEi"
    "LCJlbWFpbCI6Im1laW5lLnZhbi5kZXIubWV1bGVuQGdtYWlsLmNvbSIsImlz"
    "X2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTEyMDI4fQ."
    "QMd3aqRnowq1zyiyFDnOu0bXSNSAZ2rEMaIoCkOQLJ4"
)

s = requests.Session()
s.cookies.set("access_token", TOKEN)

rounds = s.get(f"{BASE}/rounds").json()
completed = [r for r in rounds if r["status"] == "completed"]

for rnd in completed:
    rid = rnd["id"]
    rnum = rnd["round_number"]
    
    # Check if we already have GT for this round
    gt_path = f"data/round_{rnum:02d}/analysis/seed_0_ground_truth.npy"
    if os.path.exists(gt_path):
        continue
    
    print(f"Downloading R{rnum} ground truth...")
    
    # Get round detail
    detail = s.get(f"{BASE}/rounds/{rid}").json()
    data_store.save_round_detail(rnum, detail)
    
    for si in range(detail["seeds_count"]):
        resp = s.get(f"{BASE}/analysis/{rid}/{si}")
        if resp.status_code != 200:
            print(f"  Seed {si}: no analysis (status {resp.status_code})")
            continue
        analysis = resp.json()
        
        gt = analysis.get("ground_truth")
        if gt is None:
            print(f"  Seed {si}: no ground_truth in response")
            continue
        
        gt_arr = np.array(gt)
        score = analysis.get("score")
        data_store.save_analysis(rnum, si, gt_arr, score)
        print(f"  Seed {si}: shape={gt_arr.shape}, score={score}")

print("Done.")
