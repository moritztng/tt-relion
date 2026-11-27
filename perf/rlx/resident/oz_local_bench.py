"""Local-pass shape for the q-batched Ozaki kernel: one product per particle, [T, K] x [O_own, K] (RELION local
search: ~37 translations, 155 own orientations), padded to [64, K] x [160, K] (K padded to a multiple of 480). Exactness on a small q, rate on q 128.
Reports the useful real-matmul rate (2 T O K per particle at the unpadded T 37, O 155) next to the padded digit rate.
  TT_VISIBLE_DEVICES=16 python oz_local_bench.py
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
from tt_relion.oz_fused import oz_fused  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(1)
up = lambda a: ttnn.from_torch(torch.from_numpy(a.astype(np.float32)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                               device=dev)
T_, O_ = 37, 155
for (q, n, R, K, O, rb, nb, check) in ((4, 13, 64, 2400, 160, 2, 5, True), (128, 13, 64, 2400, 160, 2, 5, False),
                                       (128, 13, 64, 2400, 160, 1, 5, False), (64, 13, 64, 8640, 160, 2, 5, False)):
    A = rng.integers(-2, 3, (q, n, R, K)).astype(np.float32)
    B = rng.integers(-2, 3, (q, n, O, K)).astype(np.float32)
    tA, tB = up(A.reshape(q * n, R, K)), up(B.reshape(q * n, O, K))
    rec = dict(q=q, n=n, R=R, K=K, O=O, rb=rb, nb=nb)
    try:
        S = ttnn.to_torch(oz_fused(tA, tB, n, rb=rb, nb=nb, bt=True, q=q)).double().numpy().reshape(q, n, R, O)
        if check:
            Ad, Bd = A.astype(np.float64), B.astype(np.float64)
            ex = np.stack([np.stack([sum(Ad[j, i] @ Bd[j, k - i].T for i in range(k + 1)) for k in range(n)])
                           for j in range(q)])
            rec.update(exact=bool((S == ex).all()))
        ttnn.synchronize_device(dev)
        clk = Clock(nodes); t0 = time.perf_counter()
        for _ in range(3):
            ttnn.deallocate(oz_fused(tA, tB, n, rb=rb, nb=nb, bt=True, q=q))
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / 3
        amin, amed = clk.stop()
        fl = q * n * (n + 1) / 2 * 2 * R * K * O
        rec.update(sec=round(dt, 4), ms_per_particle=round(dt / q * 1e3, 3), digit_TFLOPs=round(fl / dt / 1e12, 2),
                   useful_real_TFLOPs=round(q * 2 * T_ * O_ * K / dt / 1e12, 3), aiclk_min=amin, aiclk_med=amed,
                   load1=open("/proc/loadavg").read().split()[0])
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:400]
    print(json.dumps(rec), flush=True)
    ttnn.deallocate(tA); ttnn.deallocate(tB)
