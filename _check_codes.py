"""Check terrain code meanings."""
import json, numpy as np

d = json.load(open("data/round_08/round_detail.json"))
grid = np.array(d["initial_states"][0]["grid"])
setts = d["initial_states"][0]["settlements"]

print("Terrain codes at settlement locations:")
for s in setts[:8]:
    code = grid[s["y"], s["x"]]
    print(f"  ({s['x']},{s['y']}): code={code}, alive={s.get('alive',True)}, port={s.get('has_port',False)}")

print("\nAll terrain code counts:")
codes, counts = np.unique(grid, return_counts=True)
for c, n in zip(codes, counts):
    print(f"  code {c:>2}: {n:>4} cells")

# Check ground truth class at settlement locations
gt = np.load("data/round_08/analysis/seed_0_ground_truth.npy")
print("\nGT class distribution at settlement locations:")
for s in setts[:5]:
    gt_cell = gt[s["y"], s["x"]]
    argmax = gt_cell.argmax()
    print(f"  ({s['x']},{s['y']}): argmax_class={argmax}, probs=[{', '.join(f'{v:.3f}' for v in gt_cell)}]")

# Check what class 1 and class 4 look like globally
print("\nCells where class 4 (settlement?) is most likely in GT:")
argmax_grid = gt.argmax(axis=2)
n4 = (argmax_grid == 4).sum()
n1 = (argmax_grid == 1).sum()
print(f"  Class 4 dominant: {n4} cells")
print(f"  Class 1 dominant: {n1} cells")

# Where does class 4 appear? Near settlements?
mask4 = argmax_grid == 4
ys4, xs4 = np.where(mask4)
if len(ys4) > 0:
    print(f"  Class 4 cells sample: {list(zip(xs4[:5].tolist(), ys4[:5].tolist()))}")
    # Check terrain codes at those spots
    for x, y in zip(xs4[:5], ys4[:5]):
        print(f"    ({x},{y}): initial_code={grid[y,x]}, gt=[{', '.join(f'{v:.2f}' for v in gt[y,x])}]")
