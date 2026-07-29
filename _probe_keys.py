import numpy as np
import glob

files = sorted(glob.glob("/home/playeee/projects/robosuite/logs/sac_lift_so101_realistic/rollouts/*.npz"))
f = files[-1]
print(f"Probing: {f}")
data = np.load(f, allow_pickle=True)
print(f"Keys: {list(data.keys())}")
for k in data.keys():
    arr = data[k]
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.ndim == 1 and arr.shape[0] < 30:
        print(f"    values: {arr}")
    elif arr.ndim >= 2 and arr.shape[0] < 5:
        print(f"    first rows: {arr[:3]}")
    else:
        print(f"    sample[:3]: {arr.flat[:3]}")
