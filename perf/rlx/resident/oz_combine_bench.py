"""oz_combine: bit-exact vs a numpy fp32 emulation of the same TwoSum order, error vs float64, and rate on one BH chip.
  TT_VISIBLE_DEVICES=16 python oz_combine_bench.py
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
from tt_relion.oz_fused import oz_combine  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(2)
f32 = np.float32


def emul(S, ea, eb, n):   # same op order as the kernel (Horner, Knuth TwoSum, lo += da, lo += db)
    H = S[0].astype(f32); L = np.zeros_like(H)
    for k in range(1, n):
        H = (H * f32(4)).astype(f32); L = (L * f32(4)).astype(f32); b = S[k]
        s = (H + b).astype(f32); ap = (s - b).astype(f32); da = (H - ap).astype(f32); bp = (s - ap).astype(f32)
        L = (L + da).astype(f32); L = (L + (b - bp).astype(f32)).astype(f32); H = s
    E = np.ldexp(f32(1.0), (ea.reshape(-1)[:, None] + eb.reshape(-1)[None, :] - 2 * n)).astype(f32)
    return (H * E).astype(f32), (L * E).astype(f32)


for (n, R, O, check) in ((3, 512, 256, True), (13, 1024, 512, True), (14, 1024, 512, True), (14, 3584, 4096, False)):
    S = rng.integers(-(1 << 22), 1 << 22, (n, R, O)).astype(f32)
    ea = rng.integers(-12, 12, (R // 32, 32)).astype(np.int32)
    eb = rng.integers(-12, 12, (O // 32, 32)).astype(np.int32)
    tS = ttnn.from_torch(torch.from_numpy(S), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    rm = lambda a: ttnn.from_torch(torch.from_numpy(a), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)  # noqa: E731
    tea, teb = rm(ea), rm(eb)
    rec = dict(n=n, R=R, O=O)
    try:
        h, l_ = oz_combine(tS, tea, teb, n)
        H, L = ttnn.to_torch(h).numpy(), ttnn.to_torch(l_).numpy()
        if check:
            eh, el = emul(S, ea, eb, n)
            V = sum(S[k].astype(np.float64) * 2.0 ** -(2 * k + 2) for k in range(n)) * np.ldexp(
                1.0, ea.reshape(-1)[:, None] + eb.reshape(-1)[None, :])
            got = H.astype(np.float64) + L
            rec.update(bitexact_hi=bool((H == eh).all()), bitexact_lo=bool((L == el).all()),
                       rel_err_vs_f64=float(np.abs(got - V).max() / np.abs(V).max()))
        ttnn.deallocate(h); ttnn.deallocate(l_)
        ttnn.synchronize_device(dev)
        clk = Clock(nodes); t0 = time.perf_counter()
        for _ in range(3):
            h, l_ = oz_combine(tS, tea, teb, n); ttnn.deallocate(h); ttnn.deallocate(l_)
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / 3
        amin, amed = clk.stop()
        rec.update(sec=round(dt, 4), GBps_S=round(n * R * O * 4 / dt / 1e9, 1), aiclk_min=amin, aiclk_med=amed,
                   load1=open("/proc/loadavg").read().split()[0])
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:600]
    print(json.dumps(rec), flush=True)
