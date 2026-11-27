"""proj_rot (full on-chip projection from rotations) parity and rate on one BH chip.
Parity: bit-exactness vs an fp32 numpy emulation of the same algorithm; error vs the bridge's float64 _project (RELION
AccProjectorKernel::project3Dmodel semantics) beside the fp32 _project's own error (RELION's precision class).
  TT_VISIBLE_DEVICES=16 python proj_rot_bench.py SIZE:O [SIZE:O ...]
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(HERE, "..", "..", "..")]
from _tt import Clock, open_dev  # noqa: E402

import ttnn  # noqa: E402
from tt_relion.proj_lerp import proj_rot  # noqa: E402
from tt_relion.relion import _project  # noqa: E402

PF = 2


def geometry(s):
    R = PF * (s // 2)
    X, Y = R + 2, 2 * R + 3
    return dict(R=R, X=X, Y=Y, Z=Y, initY=-(R + 1), initZ=-(R + 1), maxR2=R * R)


def brick_table(mdl, gm):
    """[nvox, 16] fp32, col 2 (4 dz + 2 dy + dx) + part."""
    nvox = mdl.shape[0]
    m = np.concatenate([mdl, np.zeros((gm["X"] * gm["Y"] + gm["X"] + 2, 2), np.float32)])
    T = np.empty((nvox, 16), np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                o = dx + dy * gm["X"] + dz * gm["X"] * gm["Y"]
                c = 4 * dz + 2 * dy + dx
                T[:, 2 * c:2 * c + 2] = m[o:o + nvox]
    return T


def pixels(s):
    imgX = s // 2 + 1
    pix = np.arange(imgX * s)
    yi = pix // imgX
    return (pix % imgX).astype(np.float64), np.where(yi > s // 2, yi - s, yi).astype(np.float64)


def eulers(O, g):
    q = g.standard_normal((O, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)


def emulate32(T, x, y, e32, gm):
    f = np.float32
    x, y = x.astype(f)[None], y.astype(f)[None]
    c = e32 * f(PF)
    p = [c[:, i:i + 1] * x + c[:, i + 1:i + 2] * y for i in (0, 3, 6)]
    r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
    inside = r2 < f(gm["maxR2"] + 1)
    fl = p[0] < 0
    q = [np.where(fl, -v, v) for v in p]
    q0 = [np.floor(v) for v in q]
    fr = [v - v0 for v, v0 in zip(q, q0)]
    xi = np.clip(q0[0].astype(np.int64), 0, gm["X"] - 2)
    yi = np.clip(q0[1].astype(np.int64) - gm["initY"], 0, gm["Y"] - 2)
    zi = np.clip(q0[2].astype(np.int64) - gm["initZ"], 0, gm["Z"] - 2)
    G = T[(zi * gm["Y"] + yi) * gm["X"] + xi]               # [O, P, 16]
    V = G.reshape(G.shape[:2] + (2, 2, 2, 2))               # dz, dy, dx, part
    a = V[:, :, :, :, 0] + (V[:, :, :, :, 1] - V[:, :, :, :, 0]) * fr[0][..., None, None, None]
    b = a[:, :, :, 0] + (a[:, :, :, 1] - a[:, :, :, 0]) * fr[1][..., None, None]
    r = b[:, :, 0] + (b[:, :, 1] - b[:, :, 0]) * fr[2][..., None]
    re = np.where(inside, r[..., 0], f(0))
    im = np.where(inside, np.where(fl, -r[..., 1], r[..., 1]), f(0))
    return re, im


dev, nodes = open_dev()
g = np.random.default_rng(0)
for spec in sys.argv[1:]:
    s, O = (int(v) for v in spec.split(":"))
    gm = geometry(s)
    nvox = gm["X"] * gm["Y"] * gm["Z"]
    mdl = g.standard_normal((nvox, 2), dtype=np.float32)
    T = brick_table(mdl, gm)
    x, y = pixels(s)
    P = x.size
    NC = -(-P // 1024)
    px = np.full(NC * 1024, 3000.0, np.float32); py = px.copy()
    px[:P], py[:P] = x, y
    e = eulers(O, g)
    e32 = e.astype(np.float32)
    rot = np.zeros((O, 8), np.float32)
    rot[:, :6] = e32[:, [0, 1, 3, 4, 6, 7]] * np.float32(PF)
    up = lambda a, dt=ttnn.float32: ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)), dtype=dt,  # noqa: E731
                                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    t_tab = up(T)
    t_pix = up(np.stack([px.reshape(NC, 1024), py.reshape(NC, 1024)], 1).reshape(2 * NC, 1024))
    t_rot = up(rot)
    r = ttnn.to_torch(proj_rot(t_tab, t_pix, t_rot, gm)).numpy().reshape(O, NC, 2, 1024)
    cre = r[:, :, 0].reshape(O, -1)[:, :P]
    cim = r[:, :, 1].reshape(O, -1)[:, :P]
    ere, eim = emulate32(T, x, y, e32, gm)
    tm = torch.from_numpy(mdl)
    def proj(dt, ee):
        return _project(torch, tm.to(dt), torch.from_numpy(x).to(dt), torch.from_numpy(y).to(dt),
                        torch.from_numpy(ee).to(dt), gm["X"], gm["Y"], gm["Z"], gm["initY"], gm["initZ"], gm["maxR2"], PF)
    r64, i64 = (v.numpy() for v in proj(torch.float64, e32.astype(np.float64)))
    r32, i32 = (v.numpy() for v in proj(torch.float32, e32))
    scale = max(np.abs(r64).max(), np.abs(i64).max())
    err = lambda a, b: float(max(np.abs(a - r64).max(), np.abs(b - i64).max()) / scale)  # noqa: E731
    rec = dict(s=s, O=O, P=P, NC=NC, px=O * P,
               bitexact_vs_fp32_emu=float(np.mean((cre == ere) & (cim == eim))),
               rel_chip_vs_f64=err(cre, cim), rel_emu_vs_f64=err(ere, eim), rel_relion_fp32_vs_f64=err(r32, i32))
    ttnn.synchronize_device(dev)
    clk = Clock(nodes); t0 = time.perf_counter()
    for _ in range(5):
        ttnn.deallocate(proj_rot(t_tab, t_pix, t_rot, gm))
    ttnn.synchronize_device(dev)
    dt_ = (time.perf_counter() - t0) / 5
    lo, med = clk.stop()
    rec.update(sec=round(dt_, 5), Mpx_per_s=round(O * P / dt_ / 1e6, 1), aiclk_min=lo, aiclk_med=med,
               load1=open("/proc/loadavg").read().split()[0])
    print(json.dumps(rec), flush=True)
    for t in (t_tab, t_pix, t_rot):
        ttnn.deallocate(t)
