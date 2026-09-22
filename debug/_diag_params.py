import time

import numpy as np

t0 = time.time()
data = np.load("/models/colabfold/params/params_model_1_multimer_v3.npz")
keys = list(data.keys())
print("num arrays:", len(keys))
total_bytes = 0
for k in keys:
    total_bytes += data[k].nbytes
print("total size (GB):", total_bytes / 1e9)
print("npz load (lazy) took", time.time() - t0, "s")

t1 = time.time()
n = 0
for k in keys[:50]:
    arr = data[k]
    n += arr.nbytes
print("eagerly reading first 50 arrays took", time.time() - t1, "s")
