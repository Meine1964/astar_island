# How Astar Island Works — The Simple Version

## What is this?

A competition where we predict what a Viking map looks like after 50 years of simulation. We don't get to see the full result — we only get to peek at small parts of it (50 peeks total). Then we guess the rest.

## The Map

- 40x40 grid
- 6 types of terrain: plains, settlement, port, ruin, forest, mountain
- Mountains and ocean never change — we know those for free
- Everything else can change over 50 simulated years

## The Challenge

The server runs 5 random simulations on the same starting map. For each one, we must predict: "what's the probability of each terrain type in every cell?"

We get **50 queries** — each reveals a 15x15 window of one simulation's result. That's our only information. Submissions are free and unlimited.

## Our Approach (3 layers)

### Layer 1: Run our own simulation (free, no queries needed)

We know the starting map. We built a copy of the server's simulator and run it 100 times ourselves. This gives us a rough probability map: "this cell is 60% likely to be plains, 30% settlement, 10% forest..."

This alone gets us roughly halfway to a perfect score.

### Layer 2: Use our 50 queries wisely

Instead of peeking at random spots, we look at the most uncertain areas first. After each peek, we update our predictions and pick the next most useful spot.

Observations from one simulation help predict the others too, since they all start from the same map.

### Layer 3: Learn from past rounds

We've saved the correct answers from 9 previous rounds. This teaches us patterns like:
- "Cells close to settlements have a 25% chance of becoming settlements"
- "Cells far from everything are usually just plains"

These patterns fill in gaps where we couldn't peek and our simulator is uncertain.

## The Files

| File | What it does |
|------|-------------|
| `strategy.py` | The brain — all prediction logic |
| `server_run.py` | Runs the full pipeline against the live API |
| `resubmit.py` | Resubmit improved predictions for free |
| `astar_island_simulator/env.py` | Our copy of the Viking simulator |
| `data/` | Saved results from all rounds |

## How Scoring Works

The competition uses **KL divergence** — it measures how far our predicted probabilities are from the true ones. Cells where the answer is uncertain (50/50) count more than cells where it's obvious (99/1). So getting the tricky cells right matters most.

## Score History

| Round | Rank | What changed |
|-------|------|-------------|
| 6 | 180 | Basic guessing from terrain type |
| 7 | 85 | Added our own simulator |
| 8 | 66 | Smarter query selection + learning from history |
| 9 | 106 | Competition got tighter (same score, more competitors) |
| 10 | ? | Fixed simulator — settlements now survive realistically |
