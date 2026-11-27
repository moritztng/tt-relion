"""Health probe for one chip (pin with TT_VISIBLE_DEVICES): open, one tiny matmul, close. Prints PROBE_OK."""
import numpy as np
import torch
import ttnn
from perf.rlx._tt import open_dev

dev, nodes = open_dev()
a = ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
s = float(ttnn.to_torch(ttnn.matmul(a, a)).sum())
assert s == 32 * 32 * 32, s
print("PROBE_OK", nodes, flush=True)
ttnn.close_device(dev)
