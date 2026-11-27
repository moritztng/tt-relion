"""fp32 precision of ttnn.sum / ttnn.min / ttnn.exp on chip (the significance + weights stage needs sums good to well
under the 1e-3 adaptive-fraction margin). x = exp(-delta) style data, [3328, 4096] and a [3, 256, 4096] view.
  TT_VISIBLE_DEVICES=16 python probe_reduce.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _tt import open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, _ = open_dev()
rng = np.random.default_rng(0)
d = (rng.random((3328, 4096)) * 40).astype(np.float32)
d[:, 0] = 0
x = np.exp(-d).astype(np.float32)
up = lambda a: ttnn.from_torch(torch.from_numpy(a), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
td, tx = up(d), up(x)
back = lambda t: ttnn.to_torch(t).double().numpy()  # noqa: E731
rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())  # noqa: E731
out = {}
out["sum_dim1"] = rel(back(ttnn.sum(tx, dim=1)).reshape(-1)[:3328], x.astype(np.float64).sum(1))
out["sum_dim0"] = rel(back(ttnn.sum(tx, dim=0)).reshape(-1)[:4096], x.astype(np.float64).sum(0))
out["min_dim1_abs"] = float(np.abs(back(ttnn.min(td, dim=1)).reshape(-1)[:3328] - d.min(1)).max())
out["max_dim1_abs"] = float(np.abs(back(ttnn.max(td, dim=1)).reshape(-1)[:3328] - d.max(1)).max())
out["exp_rel"] = rel(back(ttnn.exp(ttnn.neg(td))), np.exp(-d.astype(np.float64)))
out["exp_rel_noapprox"] = rel(back(ttnn.exp(ttnn.neg(td), fast_and_approximate_mode=False)), np.exp(-d.astype(np.float64)))
m = d < 20
out["le_mask_exact"] = bool((back(ttnn.le(td, 20.0)) == m).all())
out["mul_mask_sum_rel"] = rel(back(ttnn.sum(ttnn.multiply(tx, ttnn.le(td, 20.0)), dim=1)).reshape(-1)[:3328], (x * m).astype(np.float64).sum(1))
print(json.dumps(out), flush=True)
