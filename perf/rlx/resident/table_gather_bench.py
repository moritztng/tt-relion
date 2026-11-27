"""Local-pass projections as row gathers from a shared per-half DIGIT table: ttnn.embedding takes bf16 weights only
(fp32 is a TT_FATAL), so the per-half projection table is cut into n bf16 Ozaki digit planes once per iteration (each
row's exponent is the orientation's own, fixed), stored [O_tab * n, K] (row o * n + i), and each particle gathers
rows in (particle, digit, own orientation) order = oz_fused's q-batched [q * n, 160, K] layout, no permute.
Bit-exactness of the gathered rows vs the host table, and the rate (GB/s of gathered rows, us per particle).
  TT_VISIBLE_DEVICES=16 python table_gather_bench.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from _tt import Clock, open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(2)
N = 13
for (Otab, K, q, lay) in ((4096, 8160, 32, "tile"), (4096, 8160, 32, "rm"), (2048, 20640, 16, "tile"),
                          (8192, 2400, 64, "tile")):
    rec = dict(O_tab=Otab, n=N, K=K, q=q, rows=q * N * 160, layout=lay)
    try:
        T = rng.integers(-2, 3, (Otab * N, K)).astype(np.float32)   # digit values, exact in bf16
        tab = ttnn.from_torch(torch.from_numpy(T), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        own = rng.integers(0, Otab, (q, 160))
        idx_h = (own[:, None, :] * N + np.arange(N)[None, :, None]).reshape(1, -1).astype(np.uint32)
        idx = ttnn.from_torch(torch.from_numpy(idx_h.astype(np.int32)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=dev)
        L = ttnn.TILE_LAYOUT if lay == "tile" else ttnn.ROW_MAJOR_LAYOUT
        G = ttnn.embedding(idx, tab, layout=L)
        g = ttnn.to_torch(G).float().numpy().reshape(q * N * 160, K)
        rec["bit_exact"] = bool((g == T[idx_h[0]]).all())
        ttnn.deallocate(G)
        ttnn.synchronize_device(dev)
        clk = Clock(nodes); t0 = time.perf_counter()
        for _ in range(5):
            ttnn.deallocate(ttnn.embedding(idx, tab, layout=L))
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / 5
        amin, amed = clk.stop()
        rec.update(sec=round(dt, 5), GBs=round(q * N * 160 * K * 2 / dt / 1e9, 1), us_per_particle=round(dt / q * 1e6, 1),
                   aiclk_min=amin, aiclk_med=amed, load1=open("/proc/loadavg").read().split()[0])
        ttnn.deallocate(tab); ttnn.deallocate(idx)
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:400]
    print(json.dumps(rec), flush=True)
