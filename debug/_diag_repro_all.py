import sys
import time

import numpy as np
import jax.numpy as jnp

for model_num in [1, 2, 3, 4, 5]:
    path = f"/models/colabfold/params/params_model_{model_num}_multimer_v3.npz"
    print(f"=== model {model_num} ===", flush=True)
    t_start = time.time()
    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())
    print(f"model {model_num}: {len(keys)} arrays", flush=True)
    for i, k in enumerate(keys):
        t0 = time.time()
        arr = data[k]
        y = jnp.array(arr)
        y.block_until_ready()
        dt = time.time() - t0
        if dt > 2:
            print(f"!!! model {model_num} SLOW at [{i}] {k} shape={arr.shape} took {dt:.2f}s !!!", flush=True)
            sys.exit(1)
    print(f"model {model_num} done in {time.time() - t_start:.2f}s total", flush=True)

print("ALL MODELS OK", flush=True)
