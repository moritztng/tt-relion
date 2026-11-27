"""oz_fused q-batched exactness and rate on one BH chip (local passes: many small products in one program).
  TT_VISIBLE_DEVICES=16 python oz_fused_batch_bench.py
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
from oz_fused import oz_fused  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(1)
up = lambda a: ttnn.from_torch(torch.from_numpy(a.astype(np.float32)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                               device=dev)
for (q, n, R, K, O, case, check) in ((3, 3, 512, 960, 256, "pos12", True), (4, 14, 512, 960, 256, "rand", True),
                                     (64, 14, 512, 2400, 256, "rand", False)):
    lo, hi = (1, 3) if case == "pos12" else (-2, 3)
    A = rng.integers(lo, hi, (q, n, R, K)).astype(np.float32)
    B = rng.integers(lo, hi, (q, n, O, K)).astype(np.float32)
    tA, tB = up(A.reshape(q * n, R, K)), up(B.reshape(q * n, O, K))
    rec = dict(q=q, n=n, R=R, K=K, O=O, case=case, bt=1)
    try:
        S = ttnn.to_torch(oz_fused(tA, tB, n, bt=True, q=q)).double().numpy().reshape(q, n, R, O)
        if check:
            Ad, Bd = A.astype(np.float64), B.astype(np.float64)
            ex = np.stack([np.stack([sum(Ad[j, i] @ Bd[j, k - i].T for i in range(k + 1)) for k in range(n)])
                           for j in range(q)])
            rec.update(exact=bool((S == ex).all()), max_abs_err=float(np.abs(S - ex).max()), max_val=float(np.abs(ex).max()))
        ttnn.synchronize_device(dev)
        clk = Clock(nodes); t0 = time.perf_counter()
        for _ in range(3):
            ttnn.deallocate(oz_fused(tA, tB, n, bt=True, q=q))
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / 3
        amin, amed = clk.stop()
        fl = q * n * (n + 1) / 2 * 2 * R * K * O
        rec.update(sec=round(dt, 4), TFLOPs=round(fl / dt / 1e12, 2), aiclk_min=amin, aiclk_med=amed,
                   load1=open("/proc/loadavg").read().split()[0])
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:400]
    print(json.dumps(rec), flush=True)
    ttnn.deallocate(tA); ttnn.deallocate(tB)
