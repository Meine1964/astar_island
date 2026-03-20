"""Download and save ground truth from all completed rounds.

Downloads analysis data (ground truth distributions, scores, initial states)
for every completed round and saves to data/ directory.

Usage:  uv run python download_history.py
"""
import requests
import numpy as np
import time
import truststore
truststore.inject_into_ssl()

import data_store

# ── Configuration ──────────────────────────────────────────────────────
BASE = "https://api.ainm.no/astar-island"
TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJlMDcyNjM1ZC1mYTNmLTQ5MzMtOGMwNC1lMmJmYmM4ZDhiZDEi"
    "LCJlbWFpbCI6Im1laW5lLnZhbi5kZXIubWV1bGVuQGdtYWlsLmNvbSIsImlz"
    "X2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTEyMDI4fQ."
    "QMd3aqRnowq1zyiyFDnOu0bXSNSAZ2rEMaIoCkOQLJ4"
)

session = requests.Session()
session.cookies.set("access_token", TOKEN)

# ── Fetch all rounds ──────────────────────────────────────────────────
rounds = session.get(f"{BASE}/rounds").json()
completed = [r for r in rounds if r["status"] == "completed"]
print(f"Found {len(completed)} completed rounds\n")

for rnd in sorted(completed, key=lambda r: r["round_number"]):
    rid = rnd["id"]
    rnum = rnd["round_number"]
    print(f"=== Round #{rnum} ({rid[:8]}...) ===")

    # Save round detail
    detail = session.get(f"{BASE}/rounds/{rid}").json()
    data_store.save_round_detail(rnum, detail)
    W, H = detail["map_width"], detail["map_height"]
    seeds = detail["seeds_count"]
    time.sleep(0.25)

    # Download and save analysis for each seed
    for si in range(seeds):
        try:
            resp = session.get(f"{BASE}/analysis/{rid}/{si}")
            if resp.status_code != 200:
                print(f"  Seed {si}: analysis not available ({resp.status_code})")
                continue
            a = resp.json()
            gt = np.array(a["ground_truth"])
            score = a.get("score")
            data_store.save_analysis(rnum, si, gt, score)
            print(f"  Seed {si}: score={score}, gt shape={gt.shape}")
        except Exception as e:
            print(f"  Seed {si}: error - {e}")
        time.sleep(0.25)

    print()

print("Done! All available data saved to data/ directory.")
