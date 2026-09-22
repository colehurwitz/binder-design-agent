import time

t0 = time.time()
import jax

print("jax version:", jax.__version__)
print("devices:", jax.devices())
print("device_count:", jax.device_count())
print("local_device_count:", jax.local_device_count())
print("process_count:", jax.process_count())
print("import+detect took", time.time() - t0, "s")

import numpy as np

t1 = time.time()
x = np.random.rand(1000, 1000).astype("float32")
y = jax.device_put(x)
y.block_until_ready()
print("simple device_put took", time.time() - t1, "s")

t2 = time.time()
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

devices = jax.devices()
mesh = Mesh(devices, axis_names=("x",))
sharding = NamedSharding(mesh, P())
y2 = jax.device_put(x, sharding)
y2.block_until_ready()
print("pjit-style sharded device_put took", time.time() - t2, "s")
print("ALL DIAGNOSTICS PASSED")
