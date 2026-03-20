"""Quick diagnostic: observed vs unobserved cell error in R9."""
import numpy as np, json, os
from strategy import compute_features, NUM_CLASSES

rdir = "data/round_09"
detail = json.load(open(os.path.join(rdir, "round_detail.json")))
W, H = detail["map_width"], detail["map_height"]
eps = 1e-10

obs_data = np.load(os.path.join(rdir, "observations.npz"))
# Keys are obs_0..obs_4, obs_n_0..obs_n_4
obs_n = np.stack([obs_data[f"obs_n_{si}"] for si in range(5)])

for si in range(5):
    gt = np.load(os.path.join(rdir, "analysis", f"seed_{si}_ground_truth.npy"))
    pred = np.load(os.path.join(rdir, "predictions", f"seed_{si}.npy"))
    entropy = -np.sum(gt * np.log(gt + eps), axis=2)
    kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)
    dynamic = entropy > 0.01

    obs_mask = obs_n[si] > 0
    unobs_mask = obs_n[si] == 0

    obs_dyn = dynamic & obs_mask
    unobs_dyn = dynamic & unobs_mask

    kl_obs = kl[obs_dyn].mean() if obs_dyn.any() else 0
    kl_unobs = kl[unobs_dyn].mean() if unobs_dyn.any() else 0
    total_kl_obs = kl[obs_dyn].sum() if obs_dyn.any() else 0
    total_kl_unobs = kl[unobs_dyn].sum() if unobs_dyn.any() else 0

    print(f"Seed {si}: obs={int(obs_dyn.sum())} kl_avg={kl_obs:.4f} kl_total={total_kl_obs:.1f} | "
          f"unobs={int(unobs_dyn.sum())} kl_avg={kl_unobs:.4f} kl_total={total_kl_unobs:.1f}")

print("\n--- Worst cells (seed 0) ---")
si = 0
gt = np.load(os.path.join(rdir, "analysis", "seed_0_ground_truth.npy"))
pred = np.load(os.path.join(rdir, "predictions", "seed_0.npy"))
entropy = -np.sum(gt * np.log(gt + eps), axis=2)
kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=2)

ys, xs = np.unravel_index(np.argsort(kl.ravel())[::-1][:10], kl.shape)
for y, x in zip(ys, xs):
    n_obs = obs_n[si, y, x]
    print(f"  ({x:2d},{y:2d}): kl={kl[y,x]:.4f} ent={entropy[y,x]:.3f} n_obs={int(n_obs)}")
    gt_str = " ".join(f"{v:.3f}" for v in gt[y, x])
    pr_str = " ".join(f"{v:.3f}" for v in pred[y, x])
    print(f"         GT:   [{gt_str}]")
    print(f"         Pred: [{pr_str}]")
