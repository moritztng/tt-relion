"""oz_cut: exact digit decomposition check (float64 reconstruction) and rate on one BH chip.
  TT_VISIBLE_DEVICES=16 python oz_cut_bench.py
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
from tt_relion.oz_fused import oz_cut  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(3)
for (n, R, K, check) in ((14, 64, 64, True), (13, 512, 960, True), (14, 512, 960, True), (14, 3584, 2400, True)):
    X = (rng.standard_normal((R, K)) * np.exp(rng.uniform(-8, 8, (R, 1)))).astype(np.float32)
    X[:, -5:] = 0.0
    ea = np.ceil(np.log2(np.abs(X).max(1) + 1e-30)).astype(np.int32)
    tX = ttnn.from_torch(torch.from_numpy(X), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    tea = ttnn.from_torch(torch.from_numpy(ea.reshape(-1, 32)), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    rec = dict(n=n, R=R, K=K)
    try:
        D = ttnn.to_torch(oz_cut(tX, tea, n)).float().numpy().astype(np.float64)
        if check:
            rec_ = sum(D[i] * 2.0 ** -(2 * i + 1) for i in range(n)) * np.ldexp(1.0, ea)[:, None]
            bound = np.ldexp(1.0, ea)[:, None] * 2.0 ** -(2 * n + 1)
            rec.update(digit_range=[float(D.min()), float(D.max())], digits_integer=bool((D == np.round(D)).all()),
                       resid_over_bound=float((np.abs(rec_ - X) / bound).max()),
                       exact_when_n_covers=bool(n < 12 or (rec_ == X).all()))
            r = X.astype(np.float64) * np.ldexp(1.0, 1 - ea)[:, None]; bad = []
            for i in range(n):
                q = np.round(r); bad.append(int((np.abs(D[i] - q) > 0).sum())); r = 4 * (r - D[i])
            rec["mismatch_per_digit"] = bad
        ttnn.synchronize_device(dev)
        clk = Clock(nodes); t0 = time.perf_counter()
        for _ in range(3):
            ttnn.deallocate(oz_cut(tX, tea, n))
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / 3
        amin, amed = clk.stop()
        rec.update(sec=round(dt, 4), aiclk_min=amin, aiclk_med=amed, load1=open("/proc/loadavg").read().split()[0])
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:600]
    print(json.dumps(rec), flush=True)
