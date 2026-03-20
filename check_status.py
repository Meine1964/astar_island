"""Quick read-only status check. No queries burned, no submissions."""
import requests, truststore
truststore.inject_into_ssl()

s = requests.Session()
s.cookies.set("access_token",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJlMDcyNjM1ZC1mYTNmLTQ5MzMtOGMwNC1lMmJmYmM4ZDhiZDEi"
    "LCJlbWFpbCI6Im1laW5lLnZhbi5kZXIubWV1bGVuQGdtYWlsLmNvbSIsImlz"
    "X2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NTEyMDI4fQ."
    "QMd3aqRnowq1zyiyFDnOu0bXSNSAZ2rEMaIoCkOQLJ4"
)
BASE = "https://api.ainm.no/astar-island"

print("=== Rounds ===")
rounds = s.get(f"{BASE}/rounds").json()
for r in rounds:
    print(f"  Round #{r['round_number']}: {r['status']}")

print("\n=== My Rounds ===")
my = s.get(f"{BASE}/my-rounds").json()
for r in my:
    print(f"  Round #{r.get('round_number','?')}: "
          f"score={r.get('score')}, rank={r.get('rank')}, "
          f"queries={r.get('queries_used')}/{r.get('queries_max')}")

print("\n=== Budget ===")
budget = s.get(f"{BASE}/budget").json()
print(f"  {budget}")

print("\n=== Leaderboard (top 10) ===")
lb = s.get(f"{BASE}/leaderboard").json()
for i, entry in enumerate(lb[:10]):
    print(f"  {i+1}. {entry.get('team_name', entry.get('email','?'))}: "
          f"score={entry.get('total_score', entry.get('score'))}")
