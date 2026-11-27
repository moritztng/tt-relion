"""Per-kernel device time of the resident global pass at the production shape (s50 B256 T13: R3328 -> 3584 rows,
K2048 -> 2400, O4096), synthetic operands of the right ranges. TT_VISIBLE_DEVICES=16 python pass_profile.py"""
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
from tt_relion.oz_fused import oz_combine, oz_cut, oz_fused  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(5)
n, R, K, O, RN = 14, 3584, 2400, 4096, 512
put = lambda a, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(torch.from_numpy(a), dtype=dt, layout=lay, device=dev)  # noqa: E731
XA = put(rng.standard_normal((R, K)).astype(np.float32)); XB = put(rng.standard_normal((O, K)).astype(np.float32))
ea = put(np.full((R // 32, 32), 3, np.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
eb = put(np.full((O // 32, 32), 3, np.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
Ad, Bd = oz_cut(XA, ea, n), oz_cut(XB, eb, n)
A2 = put(rng.integers(-2, 3, (n, RN, K)).astype(np.float32), ttnn.bfloat16)
S0 = oz_fused(Ad, Bd, n, bt=True, prescale=True)


def tm(name, f, reps=3):
    ttnn.deallocate_all = None
    for o in (f(),):
        pass
    ttnn.synchronize_device(dev)
    clk = Clock(nodes); t0 = time.perf_counter()
    for _ in range(reps):
        f()
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / reps
    amin, amed = clk.stop()
    rec = dict(piece=name, ms=round(dt * 1e3, 2), aiclk_min=amin, aiclk_med=amed, load1=open("/proc/loadavg").read().split()[0])
    print(json.dumps(rec), flush=True)


tm("cut A 3584x2400", lambda: oz_cut(XA, ea, n))
tm("cut B 4096x2400", lambda: oz_cut(XB, eb, n))
tm("oz_fused C 3584x2400x4096", lambda: oz_fused(Ad, Bd, n, bt=True))
tm("oz_fused C prescale", lambda: oz_fused(Ad, Bd, n, bt=True, prescale=True))
tm("oz_fused N 512x2400x4096", lambda: oz_fused(A2, Bd, n, bt=True))
tm("combine C prescaled 3584x4096", lambda: oz_combine(S0, ea, eb, n, prescaled=True))
tm("combine C raw 3584x4096", lambda: oz_combine(S0, ea, eb, n, prescaled=False))
