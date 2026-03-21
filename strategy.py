"""Shared prediction strategy for Astar Island.

Contains all the reusable logic: feature extraction, domain priors,
cross-seed model, viewport planning, and prediction building.

Used by both main.py (local simulator) and server_run.py (live API).
"""
from __future__ import annotations
import json
import numpy as np

CODE_TO_CLASS = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
NUM_CLASSES = 6
MIN_PROB = 0.002
DYNAMIC_RADIUS = 7

# ── Empirical bias correction from ground truth ───────────────────────
# Loaded once; maps distance_bucket -> (6,) probability target
# Also carries learned priors per (init_class, dist_bucket, coastal)
_BIAS_DATA = None

def _load_bias_data():
    global _BIAS_DATA
    if _BIAS_DATA is not None:
        return _BIAS_DATA
    try:
        with open("bias_correction.json") as f:
            data = json.load(f)
        bias_by_dist = {}
        for db_str, probs in data.get("gt_by_dist_bucket", {}).items():
            bias_by_dist[int(db_str)] = np.array(probs)
        learned_priors = {}
        for key_str, probs in data.get("learned_priors_coastal", {}).items():
            parts = key_str.split("_")
            learned_priors[(int(parts[0]), int(parts[1]), int(parts[2]))] = np.array(probs)
        # Fallback: priors without coastal dimension
        for key_str, probs in data.get("learned_priors", {}).items():
            parts = key_str.split("_")
            key = (int(parts[0]), int(parts[1]))
            if key not in learned_priors:  # don't overwrite coastal-specific
                learned_priors[key] = np.array(probs)
        _BIAS_DATA = {"bias_by_dist": bias_by_dist, "learned_priors": learned_priors}
        return _BIAS_DATA
    except FileNotFoundError:
        _BIAS_DATA = {"bias_by_dist": {}, "learned_priors": {}}
        return _BIAS_DATA


# ── Feature extraction ─────────────────────────────────────────────────

def _dist_to_bucket(d):
    """Finer distance buckets: 0,1,2,3, 4-5, 6-7, 8+."""
    if d <= 3:
        return d
    elif d <= 5:
        return 4
    elif d <= 7:
        return 5
    else:
        return 6


def compute_features(init_grid, settlements, W, H):
    """Per-cell feature tuple (init_class, dist_bucket, coastal, density)."""
    dist_map = np.full((H, W), 999, dtype=int)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        for y in range(H):
            for x in range(W):
                dist_map[y, x] = min(dist_map[y, x], abs(x - sx) + abs(y - sy))

    coastal = np.zeros((H, W), dtype=int)
    for y in range(H):
        for x in range(W):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and init_grid[ny][nx] == 10:
                    coastal[y, x] = 1
                    break

    density = np.zeros((H, W), dtype=int)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        for y in range(max(0, sy - 8), min(H, sy + 9)):
            for x in range(max(0, sx - 8), min(W, sx + 9)):
                if abs(x - sx) + abs(y - sy) <= 8:
                    density[y, x] += 1

    features = np.zeros((H, W, 4), dtype=int)
    for y in range(H):
        for x in range(W):
            ic = CODE_TO_CLASS.get(init_grid[y][x], 0)
            db = _dist_to_bucket(dist_map[y, x])
            features[y, x] = [ic, db, coastal[y, x], min(density[y, x], 3)]
    return features


# ── Domain-aware prior ─────────────────────────────────────────────────

_prior_cache = {}

def domain_prior(init_class, dist_bucket, is_coastal):
    """Data-driven prior from ground truth, with hand-tuned fallback."""
    key = (init_class, dist_bucket, is_coastal)
    if key in _prior_cache:
        return _prior_cache[key].copy()

    # Try learned prior from GT data (most specific first)
    data = _load_bias_data()
    learned = data["learned_priors"]
    if key in learned:
        p = learned[key].copy()
        p = np.maximum(p, MIN_PROB)
        p /= p.sum()
        _prior_cache[key] = p.copy()
        return p
    # Try without coastal
    key2 = (init_class, dist_bucket)
    if key2 in learned:
        p = learned[key2].copy()
        p = np.maximum(p, MIN_PROB)
        p /= p.sum()
        _prior_cache[key] = p.copy()
        return p

    # Hand-tuned fallback for rare combinations
    p = np.full(NUM_CLASSES, MIN_PROB)
    ic, db, co = init_class, dist_bucket, is_coastal

    if ic == 5:
        p[5] = 1.0
    elif ic == 0:
        if db >= 5:
            p[0] = 0.94; p[4] = 0.04
        elif db >= 3:
            p[0] = 0.80; p[4] = 0.06; p[1] = 0.06; p[3] = 0.04
            p[2] = 0.03 if co else 0.01
        elif db >= 1:
            p[0] = 0.72; p[1] = 0.18; p[3] = 0.02; p[4] = 0.05
            p[2] = 0.02 if co else 0.005
        else:
            p[0] = 0.46; p[1] = 0.29; p[3] = 0.02; p[4] = 0.22
            p[2] = 0.01 if co else 0.005
    elif ic == 1:
        if co:
            p[0] = 0.48; p[1] = 0.09; p[2] = 0.16; p[3] = 0.02; p[4] = 0.23
        else:
            p[0] = 0.46; p[1] = 0.30; p[3] = 0.02; p[4] = 0.22
    elif ic == 2:
        p[0] = 0.48; p[1] = 0.09; p[2] = 0.17; p[3] = 0.02; p[4] = 0.23
    elif ic == 3:
        p[0] = 0.50; p[4] = 0.20; p[3] = 0.12; p[1] = 0.15
        p[2] = 0.03 if co else 0.01
    elif ic == 4:
        if db >= 5:
            p[4] = 0.94; p[0] = 0.02
        elif db >= 3:
            p[4] = 0.77; p[0] = 0.08; p[1] = 0.13; p[3] = 0.01
        elif db >= 1:
            p[4] = 0.63; p[0] = 0.14; p[1] = 0.20; p[3] = 0.02
        else:
            p[4] = 0.22; p[0] = 0.46; p[1] = 0.29; p[3] = 0.01

    p = np.maximum(p, MIN_PROB)
    p /= p.sum()
    _prior_cache[key] = p.copy()
    return p


# ── Cross-seed outcome model ──────────────────────────────────────────

class OutcomeModel:
    """Aggregates outcome counts by feature bucket across all seeds."""

    def __init__(self):
        self.counts = {}

    def observe(self, fkey, cls, weight=1.0):
        if fkey not in self.counts:
            self.counts[fkey] = np.zeros(NUM_CLASSES)
        self.counts[fkey][cls] += weight

    def observe_distribution(self, fkey, dist, weight=1.0):
        if fkey not in self.counts:
            self.counts[fkey] = np.zeros(NUM_CLASSES)
        self.counts[fkey] += dist * weight

    def predict(self, fkey, fallback):
        if fkey in self.counts:
            c = self.counts[fkey]
            if c.sum() >= 3:
                return (c + 0.5) / (c.sum() + 0.5 * NUM_CLASSES)
        return fallback

    def confidence(self, fkey):
        if fkey in self.counts:
            return self.counts[fkey].sum()
        return 0


# ── Viewport planning ─────────────────────────────────────────────────

def plan_viewports(init_grid, settlements, W, H, max_vp=4):
    """Greedy 15x15 placement maximising weighted dynamic-cell coverage."""
    dyn = np.zeros((H, W), dtype=float)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        for dy in range(-DYNAMIC_RADIUS, DYNAMIC_RADIUS + 1):
            for dx in range(-DYNAMIC_RADIUS, DYNAMIC_RADIUS + 1):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < H and 0 <= nx < W and init_grid[ny][nx] not in (5, 10):
                    dyn[ny, nx] = max(dyn[ny, nx], 1.0 / (1 + abs(dx) + abs(dy)))

    vps = []
    for _ in range(max_vp):
        best, bpos = 0.0, (0, 0)
        for vy in range(H - 4):
            for vx in range(W - 4):
                vw = min(15, W - vx)
                vh = min(15, H - vy)
                s = float(dyn[vy:vy + vh, vx:vx + vw].sum())
                if s > best:
                    best, bpos = s, (vx, vy)
        if best < 0.3:
            break
        vx, vy = bpos
        vw, vh = min(15, W - vx), min(15, H - vy)
        vps.append({"x": vx, "y": vy, "w": vw, "h": vh, "score": best})
        dyn[vy:vy + vh, vx:vx + vw] = 0
    return vps


# ── Query execution ───────────────────────────────────────────────────

def allocate_queries(seed_info, seeds, remaining):
    """Distribute queries across seeds proportional to settlement count."""
    q_per_seed = {}
    if remaining <= 0:
        return {i: 0 for i in range(seeds)}

    sett_counts = [len(seed_info[i]["settlements"]) for i in range(seeds)]
    total_setts = max(sum(sett_counts), 1)
    for i in range(seeds):
        q_per_seed[i] = max(2, round(remaining * sett_counts[i] / total_setts))
    while sum(q_per_seed.values()) > remaining:
        richest = max(q_per_seed, key=q_per_seed.get)
        q_per_seed[richest] -= 1
    while sum(q_per_seed.values()) < remaining:
        poorest = min(q_per_seed, key=q_per_seed.get)
        q_per_seed[poorest] += 1
    return q_per_seed


def execute_queries(session, base_url, round_id, seed_info, q_per_seed,
                    obs, obs_n, model, remaining, seeds, delay=0.0):
    """Run simulation queries and accumulate observations."""
    total_q = 0
    for si in range(seeds):
        vps = seed_info[si]["viewports"]
        feats = seed_info[si]["features"]
        budget_si = q_per_seed.get(si, 0)
        if budget_si <= 0 or not vps:
            continue

        scores = [vp["score"] for vp in vps]
        ts = sum(scores)
        vp_q = [max(1, round(budget_si * s / ts)) for s in scores]
        while sum(vp_q) > budget_si:
            vp_q[vp_q.index(max(vp_q))] -= 1
        while sum(vp_q) < budget_si:
            vp_q[0] += 1

        for vi, vp in enumerate(vps):
            for rep in range(vp_q[vi]):
                result = session.post(f"{base_url}/simulate", json={
                    "round_id": round_id, "seed_index": si,
                    "viewport_x": vp["x"], "viewport_y": vp["y"],
                    "viewport_w": vp["w"], "viewport_h": vp["h"],
                }).json()

                if "error" in result:
                    print(f"  Error: {result['error']}")
                    break

                rv = result["viewport"]
                for ri, row in enumerate(result["grid"]):
                    for ci, cell in enumerate(row):
                        gy, gx = rv["y"] + ri, rv["x"] + ci
                        cls = CODE_TO_CLASS.get(cell, 0)
                        obs[si][gy, gx, cls] += 1
                        obs_n[si][gy, gx] += 1
                        model.observe(tuple(feats[gy, gx]), cls)

                total_q += 1
                print(f"  S{si} VP{vi} r{rep}: ({rv['x']},{rv['y']}) "
                      f"{rv['w']}x{rv['h']}  [{result['queries_used']}/{result['queries_max']}]")

                if result["queries_used"] >= result["queries_max"]:
                    break
                if delay > 0:
                    import time
                    time.sleep(delay)

            if result.get("queries_used", 0) >= result.get("queries_max", 50):
                break
        if total_q >= remaining:
            break
    return total_q


# ── Adaptive query selection ──────────────────────────────────────────

def compute_cell_entropy(pred):
    """Compute per-cell entropy from an (H, W, 6) prediction tensor."""
    eps = 1e-10
    return -np.sum(pred * np.log(pred + eps), axis=2)


def current_prediction_for_seed(seed_info_entry, obs_si, obs_n_si, model, H, W,
                                 sim_prior=None):
    """Fast per-cell prediction for uncertainty estimation."""
    grid = seed_info_entry["grid"]
    feats = seed_info_entry["features"]
    pred = np.zeros((H, W, NUM_CLASSES))

    for y in range(H):
        for x in range(W):
            cell = grid[y][x]
            if cell == 5:
                p = np.full(NUM_CLASSES, MIN_PROB); p[5] = 1.0
            elif cell == 10:
                p = np.full(NUM_CLASSES, MIN_PROB); p[0] = 1.0
            else:
                ic, db, co, dn = feats[y, x]
                fkey = tuple(feats[y, x])
                if sim_prior is not None:
                    prior = sim_prior[y, x].copy()
                else:
                    prior = domain_prior(ic, db, co)
                model_pred = model.predict(fkey, prior)
                n = obs_n_si[y, x]
                if n >= 8:
                    emp = obs_si[y, x] / n
                    w = n / (n + 2)
                    p = w * emp + (1 - w) * model_pred
                elif n > 0:
                    emp = obs_si[y, x] / n
                    w = n / (n + 8)
                    p = w * emp + (1 - w) * model_pred
                else:
                    p = model_pred
            pred[y, x] = p

    pred = np.maximum(pred, MIN_PROB)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


def select_best_viewport(seed_info, obs, obs_n, model, sim_priors,
                         seeds, H, W, step=3):
    """Pick the (seed, viewport) that maximises entropy-weighted information gain.

    Scans all seeds and viewport positions (on a grid with spacing `step`)
    and returns the one covering the most uncertain cells.
    """
    best_score = -1.0
    best_seed = 0
    best_vp = {"x": 0, "y": 0, "w": 15, "h": 15}

    for si in range(seeds):
        pred = current_prediction_for_seed(
            seed_info[si], obs[si], obs_n[si], model, H, W,
            sim_prior=sim_priors.get(si)
        )
        entropy = compute_cell_entropy(pred)

        # Bonus for cells with few observations (more room to improve)
        obs_bonus = np.maximum(1.0 - obs_n[si] / 5.0, 0.0)
        value_map = entropy * (1.0 + obs_bonus)

        # Scan viewport positions
        for vy in range(0, H - 4, step):
            for vx in range(0, W - 4, step):
                vw = min(15, W - vx)
                vh = min(15, H - vy)
                score = float(value_map[vy:vy + vh, vx:vx + vw].sum())
                if score > best_score:
                    best_score = score
                    best_seed = si
                    best_vp = {"x": vx, "y": vy, "w": vw, "h": vh}

    return best_seed, best_vp, best_score


def execute_adaptive_queries(session, base_url, round_id, seed_info,
                              obs, obs_n, model, sim_priors,
                              seeds, H, W, budget, delay=0.0,
                              submit_fn=None, submit_every=10,
                              query_log_fn=None,
                              settlement_stats=None):
    """Execute queries adaptively, choosing the best viewport after each query.

    Args:
        submit_fn: Optional callback(query_count) called periodically
                   to submit intermediate predictions.
        submit_every: How often to call submit_fn.
        query_log_fn: Optional callback(query_num, seed_index, viewport, result)
                      called after each successful query for logging.
        settlement_stats: Optional dict to accumulate settlement stats per seed.
                          Maps seed_index -> list of {x, y, population, food, ...}.
    """
    total_q = 0
    low_info_streak = 0  # for early stopping

    for q in range(budget):
        si, vp, info_score = select_best_viewport(
            seed_info, obs, obs_n, model, sim_priors, seeds, H, W
        )

        feats = seed_info[si]["features"]
        result = session.post(f"{base_url}/simulate", json={
            "round_id": round_id, "seed_index": si,
            "viewport_x": vp["x"], "viewport_y": vp["y"],
            "viewport_w": vp["w"], "viewport_h": vp["h"],
        }).json()

        if "error" in result:
            print(f"  Error: {result['error']}")
            break

        rv = result["viewport"]
        for ri, row in enumerate(result["grid"]):
            for ci, cell in enumerate(row):
                gy, gx = rv["y"] + ri, rv["x"] + ci
                cls = CODE_TO_CLASS.get(cell, 0)
                obs[si][gy, gx, cls] += 1
                obs_n[si][gy, gx] += 1
                model.observe(tuple(feats[gy, gx]), cls)

        # Extract settlement stats from response
        if settlement_stats is not None and "settlements" in result:
            if si not in settlement_stats:
                settlement_stats[si] = {}
            for s in result["settlements"]:
                skey = (s["x"], s["y"])
                settlement_stats[si][skey] = s

        total_q += 1
        n_observed = int((obs_n[si] > 0).sum())
        print(f"  Q{total_q}: S{si} ({rv['x']},{rv['y']}) {rv['w']}x{rv['h']} "
              f"info={info_score:.1f} obs={n_observed} "
              f"[{result['queries_used']}/{result['queries_max']}]")

        if query_log_fn:
            query_log_fn(total_q, si, vp, result)

        if result["queries_used"] >= result["queries_max"]:
            break

        # Budget-aware early stopping: if info gain is consistently low, stop
        if info_score < 20.0:
            low_info_streak += 1
            if low_info_streak >= 5 and total_q >= budget * 0.6:
                print(f"  Early stop: info gain below threshold for "
                      f"{low_info_streak} queries")
                break
        else:
            low_info_streak = 0

        # Periodic submission
        if submit_fn and total_q % submit_every == 0:
            submit_fn(total_q)

        if delay > 0:
            import time
            time.sleep(delay)

    return total_q


# ── Simulator prior ────────────────────────────────────────────────────

def compute_simulator_prior(init_grid, init_settlements, W, H, n_sims=100,
                            params=None, ensemble=True):
    """Run our local simulator on the real initial state to get a per-cell prior.

    Returns an (H, W, 6) probability distribution — much better than hand-tuned
    domain_prior because it uses the actual map layout and settlement positions.

    If ensemble=True, runs with ±20% variations on key parameters to be robust
    against server parameter drift.
    """
    from astar_island_simulator.env import (
        AstarIslandSimulator, HiddenParams, Settlement as SimSettlement,
    )
    from dataclasses import asdict, fields

    if params is None:
        try:
            with open("calibrated_params.json") as f:
                params = HiddenParams(**{k: v for k, v in json.load(f).items()
                                         if k in HiddenParams.__dataclass_fields__})
        except FileNotFoundError:
            params = HiddenParams()

    grid_np = np.array(init_grid, dtype=int)

    settlements_base = []
    for i, s in enumerate(init_settlements):
        if not s.get("alive", True):
            continue
        settlements_base.append(SimSettlement(
            x=s["x"], y=s["y"],
            population=10, food=5.0, wealth=1.0, defense=3.0, tech_level=1.0,
            has_port=s.get("has_port", False),
            owner_id=i,
        ))

    # Build parameter ensemble: baseline + variations
    param_sets = [params]
    if ensemble:
        # Wide ensemble to cover the huge round-to-round variation in hidden params.
        # GT settlement rates range from 0-3% (harsh winters) to 25-28% (mild).
        # We need factors that span this full range.
        vary_keys = ["winter_base_severity", "food_per_forest", "food_per_plains",
                     "collapse_food_threshold", "expansion_prob"]
        base_dict = asdict(params)
        for key in vary_keys:
            base_val = base_dict[key]
            # Wide range: 0.3x to 2.0x covers harsh-to-mild scenarios
            for factor in [0.3, 0.6, 1.5, 2.0]:
                varied = base_dict.copy()
                varied[key] = base_val * factor
                field_names = {f.name for f in fields(HiddenParams)}
                param_sets.append(HiddenParams(**{k: v for k, v in varied.items()
                                                   if k in field_names}))

    # Distribute sims across parameter sets
    sims_per_param = max(10, n_sims // len(param_sets))
    counts = np.zeros((H, W, NUM_CLASSES))
    total_sims = 0

    for pi, p in enumerate(param_sets):
        sim = AstarIslandSimulator.__new__(AstarIslandSimulator)
        sim.map_seed = 0
        sim.params = p
        sim.width = W
        sim.height = H
        sim.base_grid = grid_np
        sim.base_settlements = settlements_base

        for i in range(sims_per_param):
            final_grid, _ = sim.run(sim_seed=pi * 1000 + i)
            for y in range(H):
                for x in range(W):
                    cls = CODE_TO_CLASS.get(int(final_grid[y, x]), 0)
                    counts[y, x, cls] += 1
            total_sims += 1

    prior = counts / total_sims
    prior = np.maximum(prior, MIN_PROB)
    prior /= prior.sum(axis=2, keepdims=True)
    return prior


# ── Prediction building ───────────────────────────────────────────────

def build_prediction(seed_info_entry, obs_si, obs_n_si, model, H, W,
                     sim_prior=None, settlement_stats=None):
    """Build the HxWx6 prediction tensor for one seed.

    If sim_prior (HxWx6) is provided, it replaces the hand-tuned domain_prior
    as the fallback for unobserved cells.
    If settlement_stats is provided (dict of (x,y) -> stat_dict), use settlement
    health to weight nearby cell predictions.
    """
    grid = seed_info_entry["grid"]
    feats = seed_info_entry["features"]
    pred = np.zeros((H, W, NUM_CLASSES))

    for y in range(H):
        for x in range(W):
            ic, db, co, dn = feats[y, x]
            fkey = tuple(feats[y, x])
            cell = grid[y][x]

            if cell == 5:
                p = np.full(NUM_CLASSES, MIN_PROB)
                p[5] = 1.0
            elif cell == 10:
                p = np.full(NUM_CLASSES, MIN_PROB)
                p[0] = 1.0
            else:
                dp = domain_prior(ic, db, co)
                if sim_prior is not None:
                    prior = 0.3 * sim_prior[y, x] + 0.7 * dp
                    prior = np.maximum(prior, MIN_PROB)
                    prior /= prior.sum()
                else:
                    prior = dp

                model_pred = model.predict(fkey, prior)
                n = obs_n_si[y, x]

                if n >= 8:
                    emp = obs_si[y, x] / n
                    w = n / (n + 2)
                    p = w * emp + (1 - w) * model_pred
                elif n > 0:
                    emp = obs_si[y, x] / n
                    w = n / (n + 8)
                    p = w * emp + (1 - w) * model_pred
                else:
                    p = model_pred

            pred[y, x] = p

    # ── Bias correction: nudge toward empirical ground truth ──────────
    data = _load_bias_data()
    bias = data["bias_by_dist"]
    if bias:
        for y in range(H):
            for x in range(W):
                cell = grid[y][x]
                if cell == 5 or cell == 10:
                    continue  # skip static cells
                db = int(feats[y, x, 1])  # distance bucket
                if db not in bias:
                    continue
                gt_target = bias[db]
                n = obs_n_si[y, x]
                # Selective bias: only correct classes where pred already
                # has meaningful mass (>= 0.01).  Don't inflate rare classes.
                alpha = 0.20 / (1.0 + n * 0.5)
                correction = gt_target - pred[y, x]
                # Zero out corrections that would inflate near-zero classes
                for c in range(NUM_CLASSES):
                    if pred[y, x, c] < 0.01 and correction[c] > 0:
                        correction[c] = 0.0
                pred[y, x] += alpha * correction
                pred[y, x] = np.maximum(pred[y, x], 0.0)

    # ── Spatial propagation: smooth predictions using observed neighbors ──
    # If a cell is unobserved but has observed neighbors, blend toward neighbors
    smoothed = pred.copy()
    for y in range(H):
        for x in range(W):
            cell = grid[y][x]
            if cell == 5 or cell == 10:
                continue
            if obs_n_si[y, x] >= 3:
                continue  # well-observed cells don't need smoothing
            # Gather observed neighbor predictions
            neighbor_sum = np.zeros(NUM_CLASSES)
            neighbor_weight = 0.0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and obs_n_si[ny, nx] >= 2:
                        d = abs(dy) + abs(dx)
                        w = obs_n_si[ny, nx] / (d * d)
                        neighbor_sum += pred[ny, nx] * w
                        neighbor_weight += w
            if neighbor_weight > 1.0:
                neighbor_pred = neighbor_sum / neighbor_weight
                # Light blend: 10-20% toward neighbors for unobserved cells
                blend = 0.15 if obs_n_si[y, x] == 0 else 0.08
                smoothed[y, x] = (1.0 - blend) * pred[y, x] + blend * neighbor_pred
    pred = smoothed

    # ── Settlement stats adjustment ──────────────────────────────────
    # If we observed settlement health, adjust nearby cells' settlement prob
    if settlement_stats:
        for (sx, sy), stats in settlement_stats.items():
            pop = stats.get("population", 10)
            food = stats.get("food", 5.0)
            alive = stats.get("alive", True)
            if not alive:
                continue
            # Healthy settlement (high pop+food) → boost settlement class nearby
            health = min(1.0, (pop / 25.0 + food / 15.0) / 2.0)
            # Adjust cells within radius 3
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        cell_n = grid[ny][nx]
                        if cell_n == 5 or cell_n == 10:
                            continue
                        d = abs(dy) + abs(dx)
                        if d == 0 or d > 3:
                            continue
                        # Small adjustment: shift toward settlement class
                        adj = 0.05 * health / d
                        pred[ny, nx, 1] += adj
                        pred[ny, nx, 0] -= adj * 0.5
                        pred[ny, nx, 4] -= adj * 0.5

    pred = np.maximum(pred, MIN_PROB)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


# ── Historical calibration ────────────────────────────────────────────

def calibrate_from_history(session, base_url, model):
    """Pre-train the outcome model from completed rounds' ground truth."""
    rounds = session.get(f"{base_url}/rounds").json()
    completed = [r for r in rounds if r["status"] == "completed"]
    trained = 0

    for rnd in completed:
        rid = rnd["id"]
        try:
            detail = session.get(f"{base_url}/rounds/{rid}").json()
        except Exception:
            continue
        rW, rH = detail["map_width"], detail["map_height"]

        for si in range(detail["seeds_count"]):
            state = detail["initial_states"][si]
            setts = [s for s in state["settlements"] if s.get("alive", True)]
            feats = compute_features(state["grid"], setts, rW, rH)

            try:
                resp = session.get(f"{base_url}/analysis/{rid}/{si}")
                if resp.status_code != 200:
                    continue
                analysis = resp.json()
            except Exception:
                continue

            gt = analysis.get("ground_truth", analysis.get("truth", None))
            if gt is None:
                for key in ["ground_truth_grid", "gt", "actual"]:
                    gt = analysis.get(key)
                    if gt is not None:
                        break
            if gt is None:
                continue

            try:
                gt_arr = np.array(gt)
            except Exception:
                continue

            if gt_arr.ndim == 3 and gt_arr.shape == (rH, rW, NUM_CLASSES):
                for y in range(rH):
                    for x in range(rW):
                        fkey = tuple(feats[y, x])
                        model.observe_distribution(fkey, gt_arr[y, x], weight=0.5)
                trained += 1
            elif gt_arr.ndim == 2 and gt_arr.shape == (rH, rW):
                for y in range(rH):
                    for x in range(rW):
                        cls = CODE_TO_CLASS.get(int(gt_arr[y, x]), 0)
                        fkey = tuple(feats[y, x])
                        model.observe(fkey, cls, weight=0.5)
                trained += 1

    return trained


# ── Analysis helpers ──────────────────────────────────────────────────

def analyze_seeds(detail, seeds):
    """Extract features, viewports, and settlement info for all seeds."""
    W, H = detail["map_width"], detail["map_height"]
    seed_info = []
    for si in range(seeds):
        state = detail["initial_states"][si]
        setts = [s for s in state["settlements"] if s.get("alive", True)]
        feats = compute_features(state["grid"], setts, W, H)
        vps = plan_viewports(state["grid"], setts, W, H)
        seed_info.append({
            "grid": state["grid"], "settlements": setts,
            "features": feats, "viewports": vps,
        })
        print(f"  Seed {si}: {len(setts)} settlements, {len(vps)} viewports "
              f"(scores: {[round(v['score'], 1) for v in vps]})")
    return seed_info


def print_summary(obs_n, model, seeds):
    """Print summary statistics."""
    print("\n-- Summary --")
    for si in range(seeds):
        n_obs = int((obs_n[si] > 0).sum())
        avg_reps = obs_n[si][obs_n[si] > 0].mean() if n_obs > 0 else 0
        print(f"  Seed {si}: {n_obs} cells observed, avg {avg_reps:.1f} reps/cell")
    print(f"  Model: {len(model.counts)} feature buckets")
    for fkey in sorted(model.counts, key=lambda k: model.counts[k].sum(), reverse=True)[:5]:
        c = model.counts[fkey]
        dist = c / c.sum()
        print(f"    {fkey}: n={c.sum():.0f} -> [{', '.join(f'{d:.2f}' for d in dist)}]")
