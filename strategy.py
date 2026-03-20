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
MIN_PROB = 0.005
DYNAMIC_RADIUS = 7

# ── Empirical bias correction from ground truth ───────────────────────
# Loaded once; maps distance_bucket -> (6,) probability target
_BIAS_CORRECTION = None

def _load_bias_correction():
    global _BIAS_CORRECTION
    if _BIAS_CORRECTION is not None:
        return _BIAS_CORRECTION
    try:
        with open("bias_correction.json") as f:
            data = json.load(f)
        _BIAS_CORRECTION = {}
        for db_str, probs in data["gt_by_dist_bucket"].items():
            _BIAS_CORRECTION[int(db_str)] = np.array(probs)
        return _BIAS_CORRECTION
    except FileNotFoundError:
        _BIAS_CORRECTION = {}
        return _BIAS_CORRECTION


# ── Feature extraction ─────────────────────────────────────────────────

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
            d = dist_map[y, x]
            db = d if d <= 3 else (4 if d <= 6 else 5)
            features[y, x] = [ic, db, coastal[y, x], min(density[y, x], 3)]
    return features


# ── Domain-aware prior ─────────────────────────────────────────────────

_prior_cache = {}

def domain_prior(init_class, dist_bucket, is_coastal):
    """Hand-tuned fallback prior."""
    key = (init_class, dist_bucket, is_coastal)
    if key in _prior_cache:
        return _prior_cache[key].copy()

    p = np.full(NUM_CLASSES, MIN_PROB)
    ic, db, co = init_class, dist_bucket, is_coastal

    if ic == 5:
        p[5] = 1.0
    elif ic == 0:
        if db >= 4:
            p[0] = 0.94; p[4] = 0.04
        elif db >= 2:
            p[0] = 0.70; p[4] = 0.09; p[1] = 0.08; p[3] = 0.07
            p[2] = 0.04 if co else 0.01
        else:
            p[0] = 0.38; p[1] = 0.22; p[3] = 0.15; p[4] = 0.10
            p[2] = 0.13 if co else 0.02
    elif ic == 1:
        if co:
            p[1] = 0.24; p[2] = 0.28; p[3] = 0.24; p[0] = 0.16; p[4] = 0.06
        else:
            p[1] = 0.34; p[3] = 0.28; p[0] = 0.18; p[4] = 0.14; p[2] = 0.04
    elif ic == 2:
        p[2] = 0.36; p[3] = 0.26; p[1] = 0.16; p[0] = 0.14; p[4] = 0.06
    elif ic == 3:
        p[0] = 0.24; p[4] = 0.24; p[3] = 0.22; p[1] = 0.18
        p[2] = 0.10 if co else 0.03
    elif ic == 4:
        if db >= 4:
            p[4] = 0.94; p[0] = 0.04
        elif db >= 2:
            p[4] = 0.76; p[0] = 0.10; p[1] = 0.06; p[3] = 0.06
        else:
            p[4] = 0.46; p[0] = 0.15; p[1] = 0.15; p[3] = 0.12
            p[2] = 0.09 if co else 0.02

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
                if n >= 5:
                    emp = obs_si[y, x] / n
                    w = n / (n + 2)
                    p = w * emp + (1 - w) * model_pred
                elif n > 0:
                    emp = obs_si[y, x] / n
                    w = n / (n + 5)
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
                              query_log_fn=None):
    """Execute queries adaptively, choosing the best viewport after each query.

    Args:
        submit_fn: Optional callback(query_count) called periodically
                   to submit intermediate predictions.
        submit_every: How often to call submit_fn.
        query_log_fn: Optional callback(query_num, seed_index, viewport, result)
                      called after each successful query for logging.
    """
    total_q = 0

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

        total_q += 1
        n_observed = int((obs_n[si] > 0).sum())
        print(f"  Q{total_q}: S{si} ({rv['x']},{rv['y']}) {rv['w']}x{rv['h']} "
              f"info={info_score:.1f} obs={n_observed} "
              f"[{result['queries_used']}/{result['queries_max']}]")

        if query_log_fn:
            query_log_fn(total_q, si, vp, result)

        if result["queries_used"] >= result["queries_max"]:
            break

        # Periodic submission
        if submit_fn and total_q % submit_every == 0:
            submit_fn(total_q)

        if delay > 0:
            import time
            time.sleep(delay)

    return total_q


# ── Simulator prior ────────────────────────────────────────────────────

def compute_simulator_prior(init_grid, init_settlements, W, H, n_sims=100, params=None):
    """Run our local simulator on the real initial state to get a per-cell prior.

    Returns an (H, W, 6) probability distribution — much better than hand-tuned
    domain_prior because it uses the actual map layout and settlement positions.
    """
    from astar_island_simulator.env import (
        AstarIslandSimulator, HiddenParams, Settlement as SimSettlement,
    )
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

    sim = AstarIslandSimulator.__new__(AstarIslandSimulator)
    sim.map_seed = 0
    sim.params = params
    sim.width = W
    sim.height = H
    sim.base_grid = grid_np
    sim.base_settlements = settlements_base

    counts = np.zeros((H, W, NUM_CLASSES))
    for i in range(n_sims):
        final_grid, _ = sim.run(sim_seed=i)
        for y in range(H):
            for x in range(W):
                cls = CODE_TO_CLASS.get(int(final_grid[y, x]), 0)
                counts[y, x, cls] += 1

    prior = counts / n_sims
    prior = np.maximum(prior, MIN_PROB)
    prior /= prior.sum(axis=2, keepdims=True)
    return prior


# ── Prediction building ───────────────────────────────────────────────

def build_prediction(seed_info_entry, obs_si, obs_n_si, model, H, W,
                     sim_prior=None):
    """Build the HxWx6 prediction tensor for one seed.

    If sim_prior (HxWx6) is provided, it replaces the hand-tuned domain_prior
    as the fallback for unobserved cells.
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
                # Use simulator prior if available, else hand-tuned
                if sim_prior is not None:
                    prior = sim_prior[y, x].copy()
                else:
                    prior = domain_prior(ic, db, co)
                model_pred = model.predict(fkey, prior)
                n = obs_n_si[y, x]

                if n >= 5:
                    emp = obs_si[y, x] / n
                    w = n / (n + 2)
                    p = w * emp + (1 - w) * model_pred
                elif n > 0:
                    emp = obs_si[y, x] / n
                    w = n / (n + 5)
                    p = w * emp + (1 - w) * model_pred
                else:
                    p = model_pred

            pred[y, x] = p

    # ── Bias correction: nudge toward empirical ground truth ──────────
    bias = _load_bias_correction()
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
                # Tuned on Round 7 ground truth: base_alpha=0.15, decay=0.5
                alpha = 0.15 / (1.0 + n * 0.5)
                pred[y, x] = (1.0 - alpha) * pred[y, x] + alpha * gt_target

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
