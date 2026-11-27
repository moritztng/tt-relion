"""Row-gather bandwidth on BH: can a shared projection table + per-particle row gathers replace per-particle interpolation?

A local search needs, per particle, ~155 projections out of a grid shared by all particles. If the half's projections at
the local order sit in a table on the chip, each particle's reference block is a gather of whole rows (no lerp).
Times ttnn.embedding (bf16 table, the fp32 table would be 2-3 bf16 pieces) at table and batch shapes of that use, and
reports gathered GB/s and the equivalent projected pixels/s (fp32 complex pixel = 8 B, as 4 bf16 pieces of 2 B).
  TT_VISIBLE_DEVICES=16 python gather_rate.py
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from _tt import Clock, open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(0)
for rows, width, nidx in ((65536, 2048, 155 * 64), (65536, 8192, 155 * 64), (16384, 32768, 155 * 16), (65536, 8192, 155 * 256)):
    tab = ttnn.from_torch(torch.randn(rows, width, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=dev)
    idx = ttnn.from_torch(torch.from_numpy(rng.integers(0, rows, (1, nidx)).astype(np.int32)), dtype=ttnn.uint32,
                          layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    for lay_name, lay in (("tile", ttnn.TILE_LAYOUT), ("rm", ttnn.ROW_MAJOR_LAYOUT)):
        try:
            f = lambda: ttnn.deallocate(ttnn.embedding(idx, tab, layout=lay))  # noqa: E731
            f(); ttnn.synchronize_device(dev)
            clk = Clock(nodes); t0 = time.perf_counter()
            for _ in range(5):
                f()
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / 5
            lo, med = clk.stop()
            gb = nidx * width * 2 / 1e9
            rec = dict(rows=rows, width=width, nidx=nidx, out_layout=lay_name, s=round(dt, 5), GBps=round(gb / dt, 1),
                       px_per_s_fp32complex=round(nidx * width * 2 / 8 / dt / 1e6, 1), unit="Mpx/s", aiclk_min=lo, aiclk_med=med)
        except Exception as e:  # noqa: BLE001
            rec = dict(rows=rows, width=width, nidx=nidx, out_layout=lay_name, error=str(e)[:200])
        print(json.dumps(rec), flush=True)
    ttnn.deallocate(tab); ttnn.deallocate(idx)
