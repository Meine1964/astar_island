# How Astar Island Works

A comprehensive guide to how our prediction system works, explained in plain language.

---

## Table of Contents

1. [The Competition](#the-competition)
2. [The Viking World](#the-viking-world)
3. [What We Have to Do](#what-we-have-to-do)
4. [Our Strategy — The Big Picture](#our-strategy--the-big-picture)
5. [Layer 1: Running Our Own Simulator](#layer-1-running-our-own-simulator)
6. [Layer 2: Learning from Past Rounds](#layer-2-learning-from-past-rounds)
7. [Layer 3: Peeking Strategically](#layer-3-peeking-strategically)
8. [Layer 4: Building the Final Prediction](#layer-4-building-the-final-prediction)
9. [Layer 5: Post-Processing Refinements](#layer-5-post-processing-refinements)
10. [The Submission Pipeline](#the-submission-pipeline)
11. [Overnight Automation](#overnight-automation)
12. [How Scoring Works](#how-scoring-works)
13. [Score History and Lessons Learned](#score-history-and-lessons-learned)
14. [Project Files](#project-files)

---

## The Competition

Astar Island is part of the **NM i AI 2026** ("Norwegian Championship in AI") competition. Teams compete to predict the outcome of a simulated Viking civilisation. The competition runs over many rounds — a new round starts every few hours, and each round presents a fresh map with new challenges.

The core idea: a server runs a stochastic (random) simulation of Norse settlements growing, fighting, trading, and collapsing over 50 simulated years. You don't get to see the full result. Instead, you get to "peek" at small parts of it through a keyhole. Then you have to predict what the entire result looks like — not just the terrain type in each cell, but the *probability* of each terrain type.

Teams are ranked on how close their predicted probabilities are to the truth. The better your probability estimates, the higher you rank.

---

## The Viking World

### The Map

Each round takes place on a **40×40 grid**. Think of it like a top-down view of a small Norse island. Every cell in this grid contains one of 6 terrain types:

| Class | Terrain | What It Does |
|-------|---------|-------------|
| 0 | **Empty / Plains / Ocean** | The background — flat land or water. Ocean forms the borders and fjords. Plains are buildable land. These never change on their own. |
| 1 | **Settlement** | An active Norse village. Has population, food, wealth, defences. Can grow, expand, fight neighbours, and collapse. |
| 2 | **Port** | A coastal settlement with a harbour. Can trade with other ports and build longships for raiding. |
| 3 | **Ruin** | A collapsed settlement. What's left when a village starves or gets conquered. Slowly reclaimed by nature. |
| 4 | **Forest** | Provides food to nearby settlements. Mostly stays the same, but can grow back over ruins. |
| 5 | **Mountain** | Impassable terrain. Completely static — never changes. |

**Mountains and ocean are freebies.** We know exactly where they are from the starting map, and they never change. So we can assign 100% probability to them immediately. The real challenge is predicting what happens to the dynamic cells — the settlements, the land near settlements, and the forests.

### The Simulation

Each map is generated from a "map seed" (a random number that determines the terrain layout). The server then runs the simulation **5 times** on the same map, each with a different "sim seed" (a different roll of the dice). This means the 5 results share the same geography but evolve differently due to randomness.

Every simulated year goes through 5 phases:

1. **Growth** — Settlements produce food from nearby forest and plains. Well-fed settlements grow in population, develop ports on coastlines, build longships, and eventually found new settlements on nearby land.

2. **Conflict** — Desperate (hungry) settlements raid their neighbours. Successful raids steal food and wealth. If a settlement takes enough damage, it gets conquered and changes allegiance. Longships let settlements raid from further away.

3. **Trade** — Ports within range of each other can trade, exchanging food and wealth. Technology also spreads between trading partners, making them both stronger.

4. **Winter** — Each year ends with a winter of varying harshness. All settlements lose food. Settlements that run out of food collapse into ruins, scattering their population to nearby friendly villages.

5. **Environment** — Nature reclaims abandoned land. Thriving settlements near ruins may rebuild them as outposts. Unclaimed ruins slowly turn back into forest or open plains.

After 50 years of this cycle, the simulation stops and the final state of the map is the "ground truth" — what we're trying to predict.

### Hidden Parameters

The simulation is controlled by about 26 hidden parameters (things like "how harsh are the winters?", "how much food does a forest provide?", "how likely is a settlement to expand?"). These parameters are **the same for all 5 seeds within a round**, but they **change between rounds**. This means each round has a different personality — some rounds have mild winters where settlements thrive, others have brutal conditions where almost everything collapses.

We never get to see these parameters directly. We have to figure out what kind of round we're in by observing what happens.

---

## What We Have to Do

For each of the 5 simulation seeds, we must submit a probability distribution: a 40×40×6 array where each cell has 6 numbers (one per terrain class) that sum to 1.0. For example, a cell might be predicted as:

```
[0.45, 0.25, 0.05, 0.02, 0.22, 0.01]
  ^      ^     ^     ^     ^     ^
  |      |     |     |     |     Mountain
  |      |     |     |     Forest
  |      |     |     Ruin
  |      |     Port
  |      Settlement
  Empty/Plains
```

This says: "We think there's a 45% chance this cell is empty plains, a 25% chance it's a settlement, etc."

### Our Budget

We get **50 queries per round**, shared across all 5 seeds. Each query lets us peek at a window of up to **15×15 cells** of one seed's final state. That's our only way to gather data. Each peek costs one query and shows one random stochastic outcome of the simulation (so peeking at the same spot twice might show different results, because the simulation is random).

**Submissions are free and unlimited.** We can submit as many times as we want — only the latest submission counts. This is a crucial insight: we submit early and often, improving our predictions as we gather more data.

---

## Our Strategy — The Big Picture

Our system layers multiple sources of information on top of each other, from cheapest to most expensive:

```
┌─────────────────────────────────────────────────────┐
│  Layer 5: Post-processing refinements               │
│  (bias correction, port logic, ruin floor, regime)  │
├─────────────────────────────────────────────────────┤
│  Layer 4: Blend everything into final prediction    │
│  (weighted combination based on observation count)  │
├─────────────────────────────────────────────────────┤
│  Layer 3: Peek at the real results (50 queries)     │
│  (focused-seed strategy, adaptive viewport picking) │
├─────────────────────────────────────────────────────┤
│  Layer 2: Learn from past rounds' ground truth      │
│  (cross-seed outcome model, historical calibration) │
├─────────────────────────────────────────────────────┤
│  Layer 1: Run our own simulator (free, 0 queries)   │
│  (100 ensemble sims per seed on the real map)       │
└─────────────────────────────────────────────────────┘
```

Each layer builds on the one below it. The bottom layer (our simulator) provides a baseline for free. The top layer (post-processing) fine-tunes the details. Together, they produce a prediction that's much better than any single layer alone.

---

## Layer 1: Running Our Own Simulator

### The Idea

When a round starts, we know the starting map — the terrain, the positions of all settlements, and which ones are ports. We *don't* know the hidden parameters, but we have reasonable guesses for them.

So we built our own copy of the server's simulation engine (`astar_island_simulator/env.py`). We feed it the real starting map and run it 100 times with different random seeds. The result: for every cell, we get a count of how many times it ended up as each terrain type. Divide by the number of runs, and you get a probability distribution.

For example, if out of 100 simulation runs a cell ends up as settlement 30 times, plains 50 times, and forest 20 times, our "simulator prior" for that cell would be approximately `[0.50, 0.30, 0, 0, 0.20, 0]`.

### Parameter Ensemble

Since we don't know the server's exact hidden parameters, we don't just use one set — we use an **ensemble** of parameter variations. We take our best-guess parameters and vary 5 key ones (winter severity, food from forests, food from plains, collapse threshold, expansion probability) by factors of 0.3× to 2.0×. This creates about 21 different parameter sets, and we distribute our 100 simulation runs across all of them.

Why? Because in some rounds, winters are harsh (0.3× food) and almost all settlements die. In others, winters are mild (2.0× food) and settlements flourish. By simulating across this full range, our prior naturally becomes a wide distribution that covers both scenarios. It's like hedging our bets.

### The Impact

This alone was our single biggest improvement. In Round 7, just adding our simulator (with no other changes) jumped our rank from 180th to 85th place. It captures the actual spatial structure of the map — which settlements are coastal, which are isolated, which forests feed which villages — in a way that no simple statistical model can.

---

## Layer 2: Learning from Past Rounds

### The Outcome Model

We maintain a **cross-seed outcome model** — a dictionary that maps "cell features" to "observed class counts". When we observe that a cell with certain characteristics turned out to be a settlement, we record that. Over time, the model accumulates statistics like "cells that started as plains, are located 2 cells from a settlement, and are coastal, ended up as settlements 25% of the time."

The features we track for each cell are:

1. **Initial terrain class** — what the cell was at the start (plains, forest, settlement, etc.)
2. **Distance bucket** — how far the cell is from the nearest initial settlement, grouped into buckets: 0, 1, 2, 3, 4–5, 6–7, 8+
3. **Coastal** — is the cell next to ocean? (0 or 1)
4. **Density** — how many settlements are within a radius of 8 cells (capped at 3)

These 4 features create hundreds of possible combinations (feature "buckets"). The model learns the probability distribution for each bucket.

### Why Cross-Seed?

This is a key insight: all 5 seeds share the same map. A cell at position (15, 20) has the same initial terrain, the same distance to settlements, and the same coastal status in all 5 seeds. So when we observe what happens to that type of cell in seed 0, we can use that information to improve our predictions for seeds 1–4 too.

This means every query teaches us something about *all* seeds, not just the one we queried. It's like getting 5× the value from each peek.

### Historical Calibration

Before spending any queries, we also pre-train the model from **all previously completed rounds**. After each round finishes, the server reveals the ground truth (the actual outcome). We download this data and feed it into our model.

By Round 17, we have ground truth from about 17 completed rounds × 5 seeds = 85 seed-rounds of data. This gives us strong baseline statistics before we even start peeking at the current round. The historical data is weighted at half strength (0.5×) because the hidden parameters change between rounds, so past rounds are informative but not perfectly representative.

### Domain Prior

For feature buckets where we don't have enough data (too rare), we fall back to a **hand-tuned domain prior**. This is a set of carefully crafted probability distributions based on our understanding of the game mechanics:

- Mountains are always mountains (100%)
- Ocean is always ocean (100%)
- Settlements near other settlements are more likely to become settlements themselves
- Cells far from all settlements are most likely to remain empty plains
- Coastal cells have a chance of becoming ports; inland cells don't
- Forest cells tend to stay as forests, especially far from settlements

We also load an empirical version of this from `bias_correction.json` — a file that contains exact probability averages computed from all our ground truth data, broken down by distance bucket and coastal status.

---

## Layer 3: Peeking Strategically

### The Query Budget

We have 50 queries total, shared across all 5 seeds. Each query reveals a 15×15 window (225 cells) of one seed's final state. Since each query runs a fresh random simulation, peeking at the same spot twice gives different results — which is actually useful, because averaging multiple peeks gives better probability estimates.

With 50 queries × 225 cells = 11,250 cell-observations, and a total map of 5 seeds × 1,600 cells = 8,000 cells, we can potentially observe most cells at least once. But not all cells are equally important — we want to spend our budget where it matters most.

### The Focused-Seed Strategy

We don't spread our queries evenly. Instead, we use a **3-phase focused-seed strategy**:

**Phase 1 (Queries 1–6): Regime Detection**
We send one query to each of the 5 seeds, plus one extra. This gives us a quick read on the landscape — are settlements thriving or dying? How much forest regrowth is there? Two of these queries intentionally target areas *far from settlements*, because what happens in those "boring" regions is actually very diagnostic: if settlements appeared far from any starting position, it means the round has aggressive expansion; if nothing changed, it's a conservative round.

**Phase 2 (Queries 7–46): Deep Focus**
Based on the Phase 1 results, we pick the **2 most informative seeds** (the ones with the most settlements, which have the most dynamic and uncertain terrain) and pour almost all remaining queries into them — about 20 queries per seed. This gives us deep coverage: roughly 4–5 observations per cell in the focus seeds. The cross-seed model then transfers what we learn to the other 3 seeds for free.

**Phase 3 (Queries 47–50): Final Spread**
We use the last few queries across all 5 seeds for final seed-specific tuning.

Why this works: the cross-seed model means that deeply observing 2 seeds teaches us almost as much as shallowly observing all 5. In backtesting, this focused strategy beats even spreading by about 11.6%.

### Adaptive Viewport Selection

Within each phase, we don't pre-plan where to look. Before every single query, we:

1. **Compute our current prediction** for the relevant seeds (using all information gathered so far)
2. **Calculate the entropy** (uncertainty) of each cell — high entropy means we're unsure
3. **Scan all possible 15×15 viewport positions** (on a grid with step size 3) across the eligible seeds
4. **Pick the one with the highest total uncertainty** weighted by an observation bonus (cells we've never seen are worth more)

This is like a photographer deciding where to point their camera: always toward the most interesting (uncertain) part of the scene. As we gather more data, the uncertainty shifts, and our queries naturally flow to wherever there's still useful information to gain.

### Early Stopping

If the information gain from queries drops below a threshold for 5 consecutive queries and we've used at least 60% of our budget, we stop early. There's no point peeking at cells where we're already confident.

---

## Layer 4: Building the Final Prediction

### The Blending Formula

For each cell, we combine three sources of information, weighted by how many times we've observed that cell:

**Well-observed cells (many observations):**
The empirical data (what we actually saw) dominates. The formula is:

```
weight = observations / (observations + 14)
prediction = weight × empirical + (1 - weight) × model_prediction
```

So with 14 observations, the empirical data gets 50% weight. With 28 observations, it gets 67%. We need quite a lot of observations before we fully trust the raw data, because each observation is just one random simulation run.

**Sparsely observed cells (a few observations):**
A blend of what we saw and what the cross-seed model predicts. The model gets more weight because a few observations aren't enough to be confident.

**Unobserved cells (zero observations):**
Pure model prediction. The model itself is a blend of our simulator prior (30%) and our domain/historical prior (70%):

```
model_base = 0.3 × simulator_prior + 0.7 × domain_prior
```

This mix gives the simulator credit for understanding spatial dynamics while keeping the data-driven prior as the primary influence.

### Settlement Regime Estimation

Not all rounds are created equal. In some rounds, settlements flourish (25% of dynamic cells become settlements). In others, harsh winters kill nearly everything (under 3% settlements). Our predictions need to reflect the *current* round's personality.

After Phase 1 of querying, we estimate the **settlement regime** — a "scale factor" that tells us how settlement-friendly this round is compared to the historical average. We compute this by comparing the observed settlement rate against what our priors expected for those same cells.

If we observe settlements at twice the expected rate, we scale up settlement probabilities everywhere. If settlements are dying off, we scale them down. This adjustment only applies to cells we haven't observed many times (the well-observed cells are already accurate from direct data).

### Self-Consistency Tuning

After all queries are done, we do a final calibration step. We search through a range of possible regime scales and find the one that minimises the disagreement between our predictions and our observations on moderately-observed cells (cells with 2–5 observations). This is like asking: "which version of our model most closely matches what we actually saw?"

This tuning step catches cases where our initial regime estimate was off, and it's been worth a significant score improvement across many rounds.

---

## Layer 5: Post-Processing Refinements

After the core prediction is built, we apply several corrections that enforce real-world constraints the model might miss:

### Bias Correction

Our predictions tend to drift slightly from the true averages (e.g., predicting too much settlement or too little forest for cells at a given distance from settlements). We compute the correction:

```
correction = ground_truth_average - our_prediction
prediction += alpha × correction
```

where `alpha` starts at 0.05 and decays as we have more observations (we trust actual observations more than historical averages). We also only correct classes that already have meaningful probability — we don't inflate a class from near-zero just because the average says it should be slightly higher.

### Port Logic

Ports can only exist next to ocean. This is a hard physical constraint:
- **Coastal cells**: the combined settlement + port probability is split into 45% settlement and 55% port (since coastal settlements are more likely to have ports).
- **Inland cells**: port probability is forced to near-zero and redistributed to empty and forest.

### Ruin Floor

Through analysis of ground truth data, we discovered that our model was suppressing ruin probability too aggressively — predicting nearly 0% when the true value is typically 1–2%. We now enforce a minimum ruin probability of 1% for cells we haven't observed well. Excess ruin mass above this floor is redistributed to empty (70%) and forest (30%). This single fix improved our scores by 5.7% on average.

### Settlement Health Adjustment

When we peek at a settlement, the query response includes detailed stats: population, food, wealth, defence. A healthy settlement (lots of people and food) makes it more likely that *nearby* cells are also settlements (because healthy settlements expand). We use this signal to slightly boost settlement probability in a 3-cell radius around observed healthy settlements.

---

## The Submission Pipeline

Here's what happens start to finish when a round begins. This is orchestrated by `server_run.py`:

```
1. Fetch the active round details (map, settlements, seeds)
2. Check our query budget (50 queries)
3. Pre-train the cross-seed model from all completed rounds' ground truth
4. Analyse all 5 seeds (extract features, plan initial viewports)
5. Compute simulator priors (100 ensemble sims per seed, ~15 seconds)
6. Submit baseline prediction using only simulator priors (0 queries used)
   └── This is our safety net: if something goes wrong later, at least we
       submitted something reasonable
7. Run 50 adaptive queries:
   ├── Phase 1 (Q1-6): All seeds, regime detection
   ├── Phase 2 (Q7-46): 2 focus seeds, deep coverage
   ├── Phase 3 (Q47-50): All seeds, final tuning
   └── Resubmit every 10 queries with updated predictions
8. Estimate final settlement regime + self-consistency tuning
9. Build final predictions with all refinements
10. Submit final predictions for all 5 seeds
11. Save everything to disk (observations, predictions, query log)
```

### Free Resubmission

Because submissions are free, we also have `resubmit.py` — a tool that takes saved observations from a previous run and builds new predictions with the latest model improvements. This means we can retroactively improve our score for any still-active round just by improving our strategy code and resubmitting. No queries needed.

---

## Overnight Automation

The `overnight.py` script runs an autonomous loop that handles rounds while we sleep:

```
┌─────────────────────────────────────┐
│         Main Loop (every 120s)      │
├─────────────────────────────────────┤
│ 1. Check for newly completed rounds │
│    └── Download ground truth        │
│    └── Evaluate: per-class KL,      │
│        settlement survival, rank    │
│                                     │
│ 2. Auto-tune parameters             │
│    └── Backtest RUIN_FLOOR values   │
│        across ALL rounds with GT    │
│    └── Apply if >0.5% improvement   │
│    └── Reload strategy module       │
│                                     │
│ 3. Submit for active rounds         │
│    └── Full pipeline: sim priors,   │
│        adaptive queries, submit     │
│                                     │
│ 4. Sleep 120 seconds, repeat        │
└─────────────────────────────────────┘
```

The script keeps track of which rounds it has already submitted and evaluated in `overnight_state.json`, so it can restart safely without duplicating work. Logs go to `overnight_log.txt`.

The auto-tuning is key: every time a round completes and reveals ground truth, we re-evaluate parameters like RUIN_FLOOR by backtesting across all historical data. If a different value would have performed better, we automatically apply it. This means the system gradually improves itself over time.

---

## How Scoring Works

### KL Divergence

The competition uses **entropy-weighted KL divergence** to score predictions. KL divergence measures how different two probability distributions are. If we predict `[0.5, 0.3, 0.2, 0, 0, 0]` and the truth is `[0.5, 0.3, 0.2, 0, 0, 0]`, the KL divergence is 0 (perfect). The more our prediction differs, the higher the KL divergence (worse score).

### Entropy Weighting — Why Some Cells Matter More

The "entropy-weighted" part is crucial. Each cell's contribution to the total score is weighted by how uncertain the true distribution is:

- **Static cells** (ocean, mountains, plains far from any settlement) have near-zero entropy — they're almost certainly one thing. These cells barely affect the score, even if we get them slightly wrong.
- **Dynamic cells** (settlements, land near settlements, ports) where the outcome is genuinely uncertain (maybe 40% settlement, 30% plains, 30% forest) have high entropy. Getting these cells right is where nearly all of our score comes from.

This means our entire ranking is determined by how well we predict the ~200–400 "interesting" cells near settlements — not the ~1,200 boring background cells.

### The Zero Probability Trap

A critical rule: **never assign 0 probability to any class**. If the ground truth has even a tiny probability for a class we marked as zero, KL divergence becomes infinite for that cell, which can destroy our score. We enforce a minimum probability floor of 0.2% (`MIN_PROB = 0.002`) for every class in every cell.

---

## Score History and Lessons Learned

| Round | Rank | What Changed |
|-------|------|-------------|
| 6 | 180 | First submission — basic terrain-type guessing |
| 7 | 85 | Added our own simulator (biggest single leap) |
| 8 | 66 | Adaptive query selection + historical learning + bias correction |
| 9 | 106 | Competition tightened (similar score, more competitive field) |
| 10 | 116 | Fixed simulator (winter formula was killing all settlements) |
| 12 | 42 | Self-consistency tuning, refined parameters |
| 13 | 53 | Continued improvement across metrics |
| 14 | 96 | Tough round — harsh hidden parameters |
| 15 | 149 | Regression — ruin suppression bug hurt badly |
| 16 | 124 | Incremental improvements |
| 17 | 146 | Identified ruin floor bug (57% of error from suppressing ruin class) |
| 18+ | TBD | Applied ruin fix + focused-seed query strategy |

### Key Lessons

- **Spatial smoothing hurts:** Blending unobserved cells toward their neighbours' predictions sounds clever, but backtesting showed it costs about 4 points per round. Removed.
- **Ruin is not zero:** Our model was forcing ruin probability to nearly zero (0.2%). Ground truth shows ~1–2% ruin on average. Fixing this one bug accounted for over half of our error in some rounds.
- **Focus beats spreading:** Giving 2 seeds ~90% of the query budget and relying on the cross-seed model for the other 3 works better than splitting queries evenly (11.6% improvement).
- **The simulator is powerful but imperfect:** Our local simulator captures spatial structure beautifully but can be 2× off on settlement survival rates (different hidden parameters). That's why we blend it with the data-driven model rather than relying on it alone.
- **Submissions are truly free:** Submitting early and often with partial data means we always have a safety net. If the API goes down or something breaks mid-round, our latest submission still counts.

---

## Project Files

| File | Purpose |
|------|---------|
| `strategy.py` | The brain — all prediction logic: features, priors, model, query selection, prediction building |
| `server_run.py` | Live API pipeline: fetches round, runs queries, submits predictions |
| `overnight.py` | Autonomous overnight handler: evaluate, tune, submit in a loop |
| `resubmit.py` | Free resubmission using saved data + improved model |
| `main.py` | Local testing against our simulator (no API calls) |
| `calibrate.py` | Grid search for simulator hidden parameters |
| `check_status.py` | Read-only API status viewer (rounds, budget, leaderboard) |
| `download_history.py` | Downloads ground truth from completed rounds |
| `data_store.py` | Saves/loads all round data (observations, predictions, ground truth) |
| `astar_island_simulator/env.py` | Our copy of the Viking simulation engine |
| `astar_island_simulator/local_api.py` | Mock API for offline testing |
| `bias_correction.json` | Learned probability averages from all ground truth data |
| `calibrated_params.json` | Tuned simulator parameters |
| `overnight_state.json` | Tracks which rounds the overnight script has handled |
| `data/round_XX/` | Saved data per round: detail, observations, predictions, ground truth |
