import sys
import time

import numpy as np

print("importing jax...", flush=True)
import jax.numpy as jnp

print("loading npz...", flush=True)
data = np.load("/models/colabfold/params/params_model_1_multimer_v3.npz")
keys = list(data.keys())
print(f"{len(keys)} arrays to convert", flush=True)

for i, k in enumerate(keys):
    t0 = time.time()
    arr = data[k]
    y = jnp.array(arr)
    y.block_until_ready()
    dt = time.time() - t0
    print(f"[{i}] {k} shape={arr.shape} dtype={arr.dtype} took {dt:.4f}s", flush=True)
    if dt > 5:
        print(f"!!! SLOW ARRAY FOUND: {k} took {dt:.2f}s !!!", flush=True)
        sys.exit(1)

print("ALL ARRAYS CONVERTED SUCCESSFULLY", flush=True)
