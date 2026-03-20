# Astar Island — Architecture & How It Works

Prediction pipeline for the **NM i AI 2026** "Astar Island" competition, where
we predict terrain distributions on a 40×40 grid after a stochastic Norse
civilisation simulator runs for 50 years.

## The Competition

Each round the server generates a map with settlements, forests, mountains, and
ocean. Five random seeds produce five different 50-year simulation runs. We get
50 queries (each reveals a 15×15 viewport of one seed's final state) and must
predict the probability distribution over 6 terrain classes for every cell of
every seed. Scoring uses entropy-weighted KL divergence — uncertain cells count
more. Submissions are free and unlimited; only queries cost budget.

### Terrain Classes

| Class | Meaning | Behaviour |
|-------|---------|-----------|
| 0 | Plains / Ocean / Empty | Static background |
| 1 | Settlement | Active Norse settlement — can grow, collapse, expand |
| 2 | Port | Coastal settlement with harbour |
| 3 | Ruin | Collapsed settlement, slowly reclaimed by nature |
| 4 | Forest | Mostly static; can regrow over ruins |
| 5 | Mountain | Completely static |

## Project Structure

```
├── strategy.py              Core prediction algorithm (shared)
├── server_run.py            Live API submission pipeline
├── main.py                  Local testing against simulator
├── resubmit.py              Free resubmission with improved model
├── calibrate.py             Simulator parameter tuning
├── check_status.py          Read-only API status viewer
├── download_history.py      Backfill ground truth from completed rounds
├── data_store.py            Persistence layer for all round data
├── test_simulator.py        Simulator unit tests
├── bias_correction.json     Learned priors from ground truth analysis
├── calibrated_params.json   Tuned simulator parameters (when available)
├── astar_island_simulator/
│   ├── env.py               Local simulation engine
│   └── local_api.py         Mock API for offline testing
├── data/
│   └── round_XX/            Per-round saved data
│       ├── round_detail.json
│       ├── observations.npz
│       ├── queries.jsonl
│       ├── predictions/seed_N.npy
│       └── analysis/seed_N_ground_truth.npy
└── task_description/        Official competition docs
```

## How the Prediction Pipeline Works

### 1. Feature Extraction (`compute_features`)

Every cell gets a 4D feature tuple:

- **init_class** — terrain type on the initial map (0–5)
- **dist_bucket** — Manhattan distance to nearest settlement, bucketed as
  `[0], [1], [2], [3], [4-5], [6-7], [8+]`
- **coastal** — 1 if adjacent to ocean, 0 otherwise
- **density** — number of settlements within radius 8 (capped at 3)

These features drive both the cross-seed model and domain priors.

### 2. Simulator Prior (`compute_simulator_prior`)

The biggest single improvement in our pipeline. We reconstruct the server's
initial map state and run our local simulator with **ensemble parameter
variation** (±30% on 5 key parameters × ~9 sims each ≈ 100 total runs). Each
run produces a 40×40 final terrain grid; averaging across all runs gives an
(H, W, 6) probability tensor per cell.

This costs **zero queries** and captures the actual spatial dynamics of *this
specific map* — which settlements are coastal, which are isolated, which forests
block expansion, etc.

### 3. Domain Prior (`domain_prior`)

Data-driven fallback from 40 seed-rounds of ground truth. For each
(init_class, dist_bucket, coastal) combination we have empirically-measured
class probabilities. For rare combinations (< 5 data points) we fall back to
hand-tuned values.

Loaded from `bias_correction.json`, which also contains per-distance-bucket
averages used in bias correction.

### 4. Cross-Seed Outcome Model (`OutcomeModel`)

A dictionary mapping feature bucket → class count vector. When we observe cell
outcomes from queries, we record them in the model keyed by the cell's feature
tuple. Because all 5 seeds share the same map layout, observations from *any*
seed improve predictions for *all* seeds via this shared model.

Also pre-trained from completed rounds' ground truth
(`calibrate_from_history`), giving us ~100+ feature buckets before spending a
single query.

### 5. Adaptive Query Selection (`execute_adaptive_queries`)

Instead of pre-planned viewports, we pick the best (seed, viewport) pair before
each query by:

1. Computing current predictions for all seeds
2. Calculating per-cell entropy
3. Scanning all possible 15×15 viewport positions (step=3)
4. Selecting the viewport with maximum total entropy × observation-need

This focuses our 50-query budget on the most uncertain regions and typically
achieves 70-85% cell coverage.

**Budget-aware early stopping**: if information gain drops below threshold for 5
consecutive queries after 60% of budget is spent, remaining queries are saved.

### 6. Prediction Building (`build_prediction`)

For each cell, predictions are blended from three sources:

```
Well-observed (n ≥ 5):  w = n/(n+2)   →  mostly empirical
Sparse (0 < n < 5):     w = n/(n+5)   →  blend empirical + model
Unobserved (n = 0):                   →  pure model/prior
```

Where the "model" prediction comes from the cross-seed OutcomeModel with the
simulator prior as fallback.

Three post-processing steps then refine the output:

1. **Bias correction** — nudge each cell toward empirical ground-truth averages
   for its distance bucket. Strength `α = 0.15 / (1 + n × 0.5)` decays with
   observation count.

2. **Spatial propagation** — unobserved cells blend 8–15% toward nearby observed
   cells' predictions (radius 2, distance-weighted).

3. **Settlement health adjustment** — if we observed settlement stats
   (population, food) from queries, healthy settlements boost the settlement
   class probability for nearby cells.

## Submission Strategy

```
server_run.py flow:
  1. Safety confirmation prompt
  2. Fetch active round, check budget
  3. Pre-train model from completed-round history (free API calls)
  4. Analyse all seeds (features, viewports)
  5. Compute ensemble simulator priors (100 sims × 11 param sets per seed)
  6. Submit baseline (sim priors only, 0 queries — free safety net)
  7. Execute 50 adaptive queries with resubmission every 10
  8. Final submission with all observations
  9. Save everything to data/round_XX/
```

After the round completes, `download_history.py` pulls ground truth for post-hoc
analysis, and `_rebuild_bias.py` can regenerate the learned priors.

`resubmit.py` allows infinite free resubmissions using saved observation data
plus improved model logic — no queries needed.

## Local Simulator (`astar_island_simulator/env.py`)

An approximate reimplementation of the server's Norse civilisation simulator.
Each year cycles through 5 phases:

1. **Growth** — food production from adjacent terrain → population growth →
   port/longship development → settlement expansion
2. **Conflict** — desperate settlements raid neighbours; successful raids loot
   resources; conquests flip allegiance
3. **Trade** — ports exchange food/wealth; technology diffuses between partners
4. **Winter** — stochastic severity drains food; starvation collapses settlements
   into ruins
5. **Environment** — thriving settlements reclaim nearby ruins; unreclaimed ruins
   regrow into forest or plains

Key tunable parameters live in `HiddenParams` (26 fields). The server's hidden
parameters are unknown — we approximate them from ground truth analysis.

## Score Progression

| Round | Rank | Notes |
|-------|------|-------|
| 6 | 180 | First submission, basic domain prior |
| 7 | 85 | Added simulator priors |
| 8 | 66 | + adaptive queries, bias correction, data persistence |
| 9 | TBD | + learned priors, finer buckets, ensemble sims, spatial propagation |

## Key Files Explained

| File | Purpose |
|------|---------|
| `strategy.py` | All prediction logic — features, model, query selection, priors |
| `server_run.py` | Live pipeline with safety checks and data persistence |
| `main.py` | Offline testing (same logic, local simulator) |
| `resubmit.py` | Free resubmission with improved model (0 queries) |
| `calibrate.py` | Grid search for simulator HiddenParams |
| `data_store.py` | Save/load observations, predictions, ground truth |
| `check_status.py` | View rounds, scores, leaderboard (read-only) |
| `download_history.py` | Pull ground truth after rounds complete |
| `bias_correction.json` | Empirical priors per (init_class, dist_bucket, coastal) |
| `env.py` | Norse civilisation simulator (5-phase lifecycle × 50 years) |
| `local_api.py` | Mock API wrapping the simulator for offline use |
