"""tt-bio gather_tiled (row gather straight into tiles) on BH at the projection's shapes: one random 64-column bf16 row
(the 8-corner complex brick as hi/mid/lo pieces, padded 48 -> 64) per output pixel, table = a padded box-360 volume.
Reports rows/s = projected pixels/s for the gather half of a gather+lerp kernel, beside ttnn.embedding at the same shape.
  TT_VISIBLE_DEVICES=16 python gt_rate.py
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from _tt import Clock, open_dev  # noqa: E402

import ttnn  # noqa: E402
from tt_bio import gather_tiled as GT  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(0)
for N, C, R, W in ((8_000_000, 64, 1024, 1024), (8_000_000, 32, 1024, 1024), (8_000_000, 64, 4096, 1024)):
    tab = ttnn.from_torch(torch.randn(N, C, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    flat = torch.from_numpy(rng.integers(0, N, R * W).astype(np.int64))
    idx = ttnn.from_torch(GT.pad_index(flat, R, W), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    arms = {"gather_tiled": lambda: GT.gather_tiled(idx, tab, 1, W)}
    eidx = ttnn.from_torch(flat.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    arms["embedding_tile"] = lambda: ttnn.embedding(eidx, tab, layout=ttnn.TILE_LAYOUT)
    for name, f in arms.items():
        rec = dict(arm=name, N=N, C=C, rows=R * W)
        try:
            ttnn.deallocate(f()); ttnn.synchronize_device(dev)
            clk = Clock(nodes); t0 = time.perf_counter()
            for _ in range(5):
                ttnn.deallocate(f())
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / 5
            lo, med = clk.stop()
            rec.update(s=round(dt, 5), Mrows_per_s=round(R * W / dt / 1e6, 1), GBps=round(R * W * C * 2 / dt / 1e9, 1),
                       aiclk_min=lo, aiclk_med=med)
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)[:300]
        print(json.dumps(rec), flush=True)
    for z in (tab, idx, eidx):
        ttnn.deallocate(z)
