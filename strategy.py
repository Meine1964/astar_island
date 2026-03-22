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


# ── Settlement regime estimation ───────────────────────────────────────

def estimate_settlement_regime(obs, obs_n, seed_info, seeds, H, W):
    """Pool observations across all seeds to estimate this round's settlement rate.

    Corrects for spatial selection bias (queries target high-entropy areas near
    settlements) by comparing observed rate against the expected rate for the
    same cells from the domain prior.

    Uses two signals for dead-round detection:
    1. Settlement survival at initial settlement positions (stochastic, noisy)
    2. Settlement rate at distance >= 3 from initial settlements (highly diagnostic:
       in dead rounds, far cells almost never become settlements)

    Returns a dict with:
      - 'observed_rate': fraction of observed dynamic cells that are settlements
      - 'expected_rate': what the domain prior predicts for those same cells
      - 'observed_cells': total dynamic cells observed
      - 'scale': ratio of observed/expected — multiply settlement probs by this
      - 'dead_round': True if settlements appear to have collapsed
    """
    total_obs_settle = 0.0
    total_prior_settle = 0.0
    total_dynamic_obs = 0.0
    total_dynamic = 0

    # Far-cell signal: settlements at distance >= 3 from initial positions
    far_obs_settle = 0.0
    far_obs_total = 0.0
    far_prior_settle = 0.0

    # Initial settlement survival signal
    init_sett_observed = 0
    init_sett_still_alive = 0

    for si in range(seeds):
        grid = seed_info[si]["grid"]
        feats = seed_info[si]["features"]
        settlements = seed_info[si]["settlements"]

        init_sett_pos = set()
        for s in settlements:
            init_sett_pos.add((s["x"], s["y"]))

        for y in range(H):
            for x in range(W):
                n = obs_n[si][y, x]
                if n == 0:
                    continue
                cell = grid[y][x]
                if cell == 5 or cell == 10:
                    continue
                total_dynamic += 1
                total_dynamic_obs += n
                total_obs_settle += obs[si][y, x, 1]

                ic, db, co, dn = feats[y, x]
                dp = domain_prior(ic, db, co)
                total_prior_settle += dp[1] * n

                # Check initial settlement positions
                if (x, y) in init_sett_pos:
                    init_sett_observed += 1
                    sett_frac = obs[si][y, x, 1] / n
                    if sett_frac > 0.3:
                        init_sett_still_alive += 1

                # Far-cell observations (informational, for potential future use)
                if db >= 3:
                    far_obs_settle += obs[si][y, x, 1]
                    far_obs_total += n
                    far_prior_settle += dp[1] * n

    if total_dynamic_obs < 100:
        return {"observed_rate": None, "expected_rate": None,
                "observed_cells": total_dynamic, "scale": 1.0,
                "dead_round": False}

    observed_rate = total_obs_settle / total_dynamic_obs
    expected_rate = total_prior_settle / total_dynamic_obs

    # Dead-round detection: initial settlement survival
    # Only catches extreme die-offs (like R10 where survival was 8%)
    # Stochastic variance means mild dead rounds still show ~40-50% survival
    dead_round = False
    if init_sett_observed >= 3:
        survival_rate = init_sett_still_alive / init_sett_observed
        if survival_rate < 0.25:
            dead_round = True

    if dead_round:
        # Force scale very low — settlements have collapsed
        scale = max(0.05, min(observed_rate * 2.0, 0.3))
    else:
        if expected_rate > 0.001:
            scale = observed_rate / expected_rate
        else:
            scale = 1.0
        scale = max(0.05, min(scale, 3.0))

    return {
        "observed_rate": observed_rate,
        "expected_rate": expected_rate,
        "observed_cells": total_dynamic,
        "scale": scale,
        "dead_round": dead_round,
    }


def self_consistency_tune(seed_info, obs, obs_n, model, seeds, H, W, regime):
    """Refine regime scale by minimising KL(obs || pred) on moderately-observed cells.

    Searches a grid of scales around the initial estimate and picks the one
    where predictions best match observations (cells with 2-5 observations,
    where the model still dominates the blended prediction).
    """
    base_scale = regime["scale"]
    lo = max(0.03, base_scale * 0.3)
    hi = min(3.0, base_scale * 3.0)
    candidates = sorted(set(
        [round(s, 3) for s in np.linspace(lo, hi, 30)] +
        [0.03, 0.05, 0.1, 0.2, 0.3, base_scale]
    ))

    best_kl = float("inf")
    best_scale = base_scale

    for s in candidates:
        test_r = dict(regime)
        test_r["scale"] = s
        test_r["dead_round"] = s < 0.15

        kl_sum = 0.0
        count = 0
        for si in range(seeds):
            pred = build_prediction(
                seed_info[si], obs[si], obs_n[si], model, H, W, regime=test_r)
            for y in range(H):
                for x in range(W):
                    n = obs_n[si][y, x]
                    if n < 2 or n > 5:
                        continue
                    cell = seed_info[si]["grid"][y][x]
                    if cell == 5 or cell == 10:
                        continue
                    emp = obs[si][y, x] / n
                    emp = np.maximum(emp, 1e-6)
                    emp /= emp.sum()
                    p = np.maximum(pred[y, x], 1e-6)
                    p /= p.sum()
                    kl_sum += np.sum(emp * np.log(emp / p))
                    count += 1
        if count > 0:
            avg_kl = kl_sum / count
            if avg_kl < best_kl:
                best_kl = avg_kl
                best_scale = s

    tuned = dict(regime)
    tuned["scale"] = best_scale
    tuned["dead_round"] = best_scale < 0.15
    return tuned


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
                                 sim_prior=None, regime=None):
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
                    prior = 0.3 * sim_prior[y, x] + 0.7 * domain_prior(ic, db, co)
                    prior = np.maximum(prior, MIN_PROB)
                    prior /= prior.sum()
                else:
                    prior = domain_prior(ic, db, co)
                model_pred = model.predict(fkey, prior)
                n = obs_n_si[y, x]
                if n > 0:
                    emp = obs_si[y, x] / n
                    w = n / (n + 14)
                    p = w * emp + (1 - w) * model_pred
                else:
                    p = model_pred
            pred[y, x] = p

    # Apply lightweight regime rescaling for better entropy estimation
    if regime is not None and regime.get("observed_rate") is not None:
        scale = regime["scale"]
        for y in range(H):
            for x in range(W):
                cell = grid[y][x]
                if cell == 5 or cell == 10:
                    continue
                if obs_n_si[y, x] >= 4:
                    continue
                p = pred[y, x]
                old_s = p[1]
                new_s = max(MIN_PROB, min(old_s * scale, 0.95))
                delta = new_s - old_s
                mass = p[0] + p[4]
                if mass > 0.01 and abs(delta) > 0.001:
                    p[1] = new_s
                    p[0] -= delta * (p[0] / mass)
                    p[4] -= delta * (p[4] / mass)

    pred = np.maximum(pred, MIN_PROB)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


def _pick_exploration_viewports(seed_info, seeds, H, W):
    """Pick 2 viewports in areas far from all initial settlements, for regime detection.

    Returns list of (seed_index, viewport_dict) targeting areas with the most
    dynamic cells at distance >= 4 from any settlement.
    """
    candidates = []
    for si in range(seeds):
        feats = seed_info[si]["features"]
        grid = seed_info[si]["grid"]
        # Score each viewport by number of far dynamic cells
        for vy in range(0, H - 4, 5):
            for vx in range(0, W - 4, 5):
                vw = min(15, W - vx)
                vh = min(15, H - vy)
                far_count = 0
                for dy in range(vh):
                    for dx in range(vw):
                        y, x = vy + dy, vx + dx
                        cell = grid[y][x]
                        if cell == 5 or cell == 10:
                            continue
                        db = feats[y, x, 1]
                        if db >= 3:  # distance bucket >= 3 = far from settlements
                            far_count += 1
                if far_count > 20:
                    candidates.append((far_count, si, {"x": vx, "y": vy, "w": vw, "h": vh}))
    candidates.sort(key=lambda c: c[0], reverse=True)
    # Return top 2, preferring different seeds
    result = []
    used_seeds = set()
    for score, si, vp in candidates:
        if len(result) >= 2:
            break
        if si not in used_seeds or len(result) < 2:
            result.append((si, vp))
            used_seeds.add(si)
    return result


def select_best_viewport(seed_info, obs, obs_n, model, sim_priors,
                         seeds, H, W, step=3, regime=None,
                         focus_seeds=None):
    """Pick the (seed, viewport) that maximises entropy-weighted information gain.

    Scans all seeds and viewport positions (on a grid with spacing `step`)
    and returns the one covering the most uncertain cells.

    If focus_seeds is provided (set of seed indices), only consider those seeds.
    """
    best_score = -1.0
    best_seed = 0
    best_vp = {"x": 0, "y": 0, "w": 15, "h": 15}

    candidate_seeds = focus_seeds if focus_seeds is not None else range(seeds)

    for si in candidate_seeds:
        pred = current_prediction_for_seed(
            seed_info[si], obs[si], obs_n[si], model, H, W,
            sim_prior=sim_priors.get(si), regime=regime
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

    Strategy: Focus queries on 2 "focus seeds" for deep coverage (~4-5 obs/cell)
    while using the other 3 seeds for cross-seed model learning.
    Phase 1 (first 10 queries): 1 query each across all 5 seeds for regime detection,
        then focus on the 2 most informative seeds.
    Phase 2 (queries 11-40): Deep coverage on focus seeds.
    Phase 3 (queries 41-50): Spread remaining queries across all seeds for final tuning.

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

    # Pre-compute exploration viewports: areas far from settlements for regime detection
    explore_vps = _pick_exploration_viewports(seed_info, seeds, H, W)

    # Pick 2 focus seeds: the ones with the most settlements (most dynamic)
    sett_counts = [(len(seed_info[si]["settlements"]), si) for si in range(seeds)]
    sett_counts.sort(reverse=True)
    focus_seeds = set([sett_counts[0][1], sett_counts[1][1]])
    print(f"  Focus seeds: {sorted(focus_seeds)} "
          f"(settlements: {sett_counts[0][0]}, {sett_counts[1][0]})")

    for q in range(budget):
        # Estimate regime from observations so far (after enough data)
        regime = None
        if total_q >= 6:
            regime = estimate_settlement_regime(obs, obs_n, seed_info, seeds, H, W)

        # Use exploration viewports for queries 4 and 6 (far from settlements)
        if total_q in (3, 5) and explore_vps:
            si, vp = explore_vps.pop(0)
            info_score = 0.0
            print(f"  [exploration query targeting far-from-settlement area]")
        else:
            # Phase-based query allocation (~45% each focus, ~3% each other):
            # Phase 1 (Q1-6):   all seeds — regime detection, 1 query each
            # Phase 2 (Q7-46):  focus seeds only — deep coverage, ~20 each
            # Phase 3 (Q47-50): all seeds — final spread, ~1 per non-focus
            if total_q < 6 or total_q >= 46:
                active_seeds = None  # all seeds
            else:
                active_seeds = focus_seeds

            si, vp, info_score = select_best_viewport(
                seed_info, obs, obs_n, model, sim_priors, seeds, H, W,
                regime=regime, focus_seeds=active_seeds
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
            try:
                final_grid, _ = sim.run(sim_seed=pi * 1000 + i)
            except Exception:
                continue
            for y in range(H):
                for x in range(W):
                    cls = CODE_TO_CLASS.get(int(final_grid[y, x]), 0)
                    counts[y, x, cls] += 1
            total_sims += 1

    if total_sims == 0:
        # Fallback: domain prior only
        total_sims = 1
        for y in range(H):
            for x in range(W):
                counts[y, x, 0] = 1  # plains default

    prior = counts / total_sims
    prior = np.maximum(prior, MIN_PROB)
    prior /= prior.sum(axis=2, keepdims=True)
    return prior


# ── Prediction building ───────────────────────────────────────────────

def build_prediction(seed_info_entry, obs_si, obs_n_si, model, H, W,
                     sim_prior=None, settlement_stats=None,
                     regime=None):
    """Build the HxWx6 prediction tensor for one seed.

    If sim_prior (HxWx6) is provided, it replaces the hand-tuned domain_prior
    as the fallback for unobserved cells.
    If settlement_stats is provided (dict of (x,y) -> stat_dict), use settlement
    health to weight nearby cell predictions.
    If regime is provided (from estimate_settlement_regime), rescale settlement
    probabilities to match the observed round regime.
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

                if n > 0:
                    emp = obs_si[y, x] / n
                    w = n / (n + 14)
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
                alpha = 0.05 / (1.0 + n * 0.5)
                correction = gt_target - pred[y, x]
                # Zero out corrections that would inflate near-zero classes
                for c in range(NUM_CLASSES):
                    if pred[y, x, c] < 0.01 and correction[c] > 0:
                        correction[c] = 0.0
                pred[y, x] += alpha * correction
                pred[y, x] = np.maximum(pred[y, x], MIN_PROB)

    # ── Regime rescaling: adjust settlement probs based on observed rate ──
    if regime is not None and regime.get("observed_rate") is not None:
        scale = regime["scale"]
        obs_rate = regime["observed_rate"]
        for y in range(H):
            for x in range(W):
                cell = grid[y][x]
                if cell == 5 or cell == 10:
                    continue
                if obs_n_si[y, x] >= 4:
                    continue  # well-observed cells are already accurate
                p = pred[y, x].copy()
                old_settle = p[1]
                # Scale settlement probability toward observed rate
                new_settle = old_settle * scale
                new_settle = max(MIN_PROB, min(new_settle, 0.95))
                delta = new_settle - old_settle
                # Redistribute delta from empty(0) and forest(4)
                mass_0_4 = p[0] + p[4]
                if mass_0_4 > 0.01 and abs(delta) > 0.001:
                    p[1] = new_settle
                    p[0] -= delta * (p[0] / mass_0_4)
                    p[4] -= delta * (p[4] / mass_0_4)
                    pred[y, x] = np.maximum(p, MIN_PROB)

    # ── Cap ruins at a small floor (GT averages ~1-2% ruin probability) ──
    RUIN_FLOOR = 0.0080
    for y in range(H):
        for x in range(W):
            cell = grid[y][x]
            if cell == 5 or cell == 10:
                continue
            if obs_n_si[y, x] >= 4:
                continue  # trust observations
            ruin_mass = pred[y, x, 3] - RUIN_FLOOR
            if ruin_mass > 0.001:
                # Redistribute excess ruin mass to empty and forest
                pred[y, x, 3] = RUIN_FLOOR
                pred[y, x, 0] += ruin_mass * 0.7
                pred[y, x, 4] += ruin_mass * 0.3

    # ── Port redistribution: ports only exist on coastal cells ────────
    # Coastal cells: split settlement+port mass 45%/55% (settlement/port)
    # Non-coastal cells: zero out port probability entirely
    PORT_SHARE = 0.55
    for y in range(H):
        for x in range(W):
            cell = grid[y][x]
            if cell == 5 or cell == 10:
                continue
            is_coastal = feats[y, x, 2]
            if is_coastal:
                total_active = pred[y, x, 1] + pred[y, x, 2]
                if total_active > 0.01:
                    pred[y, x, 2] = max(MIN_PROB, total_active * PORT_SHARE)
                    pred[y, x, 1] = max(MIN_PROB, total_active * (1 - PORT_SHARE))
            else:
                port_mass = pred[y, x, 2] - MIN_PROB
                if port_mass > 0.001:
                    pred[y, x, 2] = MIN_PROB
                    pred[y, x, 0] += port_mass * 0.7
                    pred[y, x, 4] += port_mass * 0.3

    # ── Spatial propagation: DISABLED — backtest shows it hurts by ~-4/round ──
    # smoothed = pred.copy()
    # ...
    # pred = smoothed

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
