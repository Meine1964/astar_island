"""Data persistence for Astar Island.

Saves and loads round data, observations, predictions, and ground truth.
Directory structure:
    data/
      round_08/
        round_detail.json        # initial states, map size, seeds
        observations.npz         # obs & obs_n arrays per seed
        queries.jsonl            # append-only log of each query result
        predictions/
          seed_0.npy             # our submitted predictions
        analysis/                # populated after round completes
          seed_0_ground_truth.npy
          seed_0_score.json
"""
import json
import os
import numpy as np


DATA_DIR = "data"


def round_dir(round_number):
    """Get directory path for a round, e.g. data/round_08."""
    d = os.path.join(DATA_DIR, f"round_{round_number:02d}")
    os.makedirs(d, exist_ok=True)
    return d


def save_round_detail(round_number, detail):
    """Save the full round detail JSON (initial states, map size, etc.)."""
    d = round_dir(round_number)
    path = os.path.join(d, "round_detail.json")
    with open(path, "w") as f:
        json.dump(detail, f)
    print(f"  Saved round detail to {path}")


def load_round_detail(round_number):
    """Load saved round detail, or None if not found."""
    path = os.path.join(round_dir(round_number), "round_detail.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_observations(round_number, obs, obs_n, seeds):
    """Save observation arrays (obs and obs_n dicts) as .npz."""
    d = round_dir(round_number)
    path = os.path.join(d, "observations.npz")
    arrays = {}
    for si in range(seeds):
        arrays[f"obs_{si}"] = obs[si]
        arrays[f"obs_n_{si}"] = obs_n[si]
    np.savez_compressed(path, **arrays)
    total_obs = sum(int((obs_n[si] > 0).sum()) for si in range(seeds))
    print(f"  Saved observations to {path} ({total_obs} total observed cells)")


def load_observations(round_number, seeds, H, W, num_classes=6):
    """Load saved observations, or return empty arrays if not found."""
    path = os.path.join(round_dir(round_number), "observations.npz")
    obs = {i: np.zeros((H, W, num_classes)) for i in range(seeds)}
    obs_n = {i: np.zeros((H, W)) for i in range(seeds)}
    if not os.path.exists(path):
        return obs, obs_n, False
    data = np.load(path)
    for si in range(seeds):
        key_obs = f"obs_{si}"
        key_n = f"obs_n_{si}"
        if key_obs in data and key_n in data:
            obs[si] = data[key_obs]
            obs_n[si] = data[key_n]
    total_obs = sum(int((obs_n[si] > 0).sum()) for si in range(seeds))
    print(f"  Loaded observations from {path} ({total_obs} total observed cells)")
    return obs, obs_n, True


def append_query(round_number, query_num, seed_index, viewport, result):
    """Append a single query result to the JSONL log."""
    d = round_dir(round_number)
    path = os.path.join(d, "queries.jsonl")
    entry = {
        "query_num": query_num,
        "seed_index": seed_index,
        "viewport": viewport,
        "grid": result.get("grid"),
        "queries_used": result.get("queries_used"),
        "queries_max": result.get("queries_max"),
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_prediction(round_number, seed_index, prediction):
    """Save a submitted prediction array."""
    d = os.path.join(round_dir(round_number), "predictions")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"seed_{seed_index}.npy")
    np.save(path, prediction)


def load_prediction(round_number, seed_index):
    """Load a saved prediction, or None."""
    path = os.path.join(round_dir(round_number), "predictions",
                        f"seed_{seed_index}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def save_analysis(round_number, seed_index, ground_truth, score):
    """Save ground truth and score from a completed round."""
    d = os.path.join(round_dir(round_number), "analysis")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"seed_{seed_index}_ground_truth.npy"),
            ground_truth)
    with open(os.path.join(d, f"seed_{seed_index}_score.json"), "w") as f:
        json.dump({"seed_index": seed_index, "score": score}, f)


def load_analysis(round_number, seed_index):
    """Load saved ground truth and score, or (None, None)."""
    d = os.path.join(round_dir(round_number), "analysis")
    gt_path = os.path.join(d, f"seed_{seed_index}_ground_truth.npy")
    sc_path = os.path.join(d, f"seed_{seed_index}_score.json")
    gt = np.load(gt_path) if os.path.exists(gt_path) else None
    score = None
    if os.path.exists(sc_path):
        with open(sc_path) as f:
            score = json.load(f).get("score")
    return gt, score


def has_observations(round_number):
    """Check if we have saved observation data for a round."""
    path = os.path.join(round_dir(round_number), "observations.npz")
    return os.path.exists(path)
