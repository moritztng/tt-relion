"""TT_RELION_DEVPROJ parity on any host: the fine call with the chip projecting vs the fine call fed RELION's projection.

Drives the real bridge (tt_relion/relion.py, backend ttnn) against stub servers (tt_relion/stub_server.py with
RLX_STUB_MATMUL=1: the server loop, split and chunk sums are the real ones, the matmul is numpy fp32). A coarse call
hands each server the model; the fine call then runs twice on the same inputs, once given the host fp32 projection
(what RELION's projector hands over) and once with TT_RELION_DEVPROJ. Both are checked against float64.
    TT_RELION_PYTHON=<python with torch> python devproj_parity.py --s 50 --O 64 --T 21
--chip N runs both arms through the real device server on chip N instead (TT_RELION_PYTHON must have ttnn).
"""
import argparse
import os
import sys

os.environ["TT_RELION_BACKEND"] = "ttnn"
os.environ["RLX_STUB_MATMUL"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tt_relion import relion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--s", type=int, default=50)
ap.add_argument("--O", type=int, default=64)
ap.add_argument("--T", type=int, default=21)
ap.add_argument("--chip", default=None)
ap.add_argument("--reps", type=int, default=0, help="warm timed fine calls per arm (AICLK sampled with --chip)")
a = ap.parse_args()

PF = 2
g = np.random.default_rng(0)
R = PF * (a.s // 2)
X, Y = R + 2, 2 * R + 3
geo = dict(mdlX=X, mdlY=Y, mdlZ=Y, mdlInitY=-(R + 1), mdlInitZ=-(R + 1), maxR2_padded=R * R, padding_factor=PF)
mdl = g.standard_normal((X * Y * Y, 2)).astype(np.float32)
imgX, imgY, maxR = a.s // 2 + 1, a.s, a.s // 2
P = imgX * imgY
q = g.standard_normal((a.O, 4))
q /= np.linalg.norm(q, axis=1, keepdims=True)
w_, x_, y_, z_ = q.T
eul = np.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_),
                2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_),
                2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)], 1).astype(np.float32)
tx, ty = (g.uniform(-3, 3, a.T).astype(np.float32) for _ in range(2))
re, im = (g.standard_normal(P).astype(np.float32) for _ in range(2))
corr = g.uniform(0.5, 1.5, P).astype(np.float32)

if a.chip is None:
    relion._srv.extend(relion._Server(None, module="tt_relion.stub_server") for _ in range(2))
else:
    relion._srv.append(relion._Server(a.chip))
relion._pool()


def proj(dt):
    pix = torch.arange(P)
    x = (pix % imgX).to(dt)
    yi = pix // imgX
    y = torch.where(yi > maxR, yi - imgY, yi).to(dt)
    r, i = relion._project(torch, torch.from_numpy(mdl).to(dt), x, y, torch.from_numpy(eul).to(dt), X, Y, Y,
                           geo["mdlInitY"], geo["mdlInitZ"], geo["maxR2_padded"], PF)
    return torch.cat([r, i], 1)


def fine(ref):
    out = np.zeros(a.O * a.T, np.float32)
    rot = np.repeat(np.arange(a.O, dtype=np.uint64), a.T)
    tr = np.tile(np.arange(a.T, dtype=np.uint64), a.O)
    ok = relion.diff2_fine(memoryview(np.ascontiguousarray(ref, np.float32)), memoryview(eul), memoryview(tx),
                           memoryview(ty), memoryview(re), memoryview(im), memoryview(corr), memoryview(rot),
                           memoryview(tr), memoryview(out), maxR, imgX, imgY, 0.0, a.O, a.T, a.O * a.T, P, a.O)
    assert ok
    return out.astype(np.float64)


# float64 truth: the direct form, every term in float64
ref64 = proj(torch.float64).numpy()
pix = np.arange(P)
xx, yy = pix % imgX, np.where(pix // imgX > maxR, pix // imgX - imgY, pix // imgX)
ph = xx[None] * tx[:, None].astype(np.float64) + yy[None] * ty[:, None].astype(np.float64)
shr = np.cos(ph) * re - np.sin(ph) * im
shi = np.cos(ph) * im + np.sin(ph) * re
d64 = (((ref64[:, None, :P] - shr[None]) ** 2 + (ref64[:, None, P:] - shi[None]) ** 2) * corr * 0.5).sum(-1).reshape(-1)

# one coarse call so every server keeps the model and this thread remembers its geometry
relion._DEVPROJ = True
cout = np.zeros(a.O * a.T, np.float32)
assert relion.diff2_coarse(memoryview(mdl), memoryview(eul), memoryview(tx), memoryview(ty), memoryview(re),
                           memoryview(im), memoryview(corr), memoryview(cout), X, Y, Y, geo["mdlInitY"],
                           geo["mdlInitZ"], maxR, geo["maxR2_padded"], PF, imgX, imgY, a.O, a.T, P)
ref32 = proj(torch.float32).numpy()
relion._DEVPROJ = False
given = fine(ref32)
relion._DEVPROJ = True
n0 = relion._stats["devproj"]
dev = fine(ref32 * 0)          # a zeroed reference: any use of it instead of the chip's projection shows
assert relion._stats["devproj"] > n0, "fine call did not take the device-projection path"
scale = np.abs(d64).max()
print("chip=%s s=%d O=%d T=%d  d2 max %.4g | err vs f64: given %.3g  devproj %.3g  (rel %.2e / %.2e) | devproj vs given %.3g"
      % (a.chip, a.s, a.O, a.T, scale, np.abs(given - d64).max(), np.abs(dev - d64).max(),
         np.abs(given - d64).max() / scale, np.abs(dev - d64).max() / scale, np.abs(dev - given).max()))

if a.reps:
    import threading
    import time
    clk, run = [], [True]

    def sample():
        while run[0]:
            try:
                clk.append(int(open("/sys/class/tenstorrent/tenstorrent!%s/tt_aiclk" % a.chip).read().split()[0]))
            except (OSError, ValueError, TypeError):
                pass
            time.sleep(0.1)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    wall = {}
    for arm, on, r in (("given", False, ref32), ("devproj", True, ref32 * 0)):
        relion._DEVPROJ = on
        fine(r)
        t0 = time.time()
        for _ in range(a.reps):
            fine(r)
        wall[arm] = (time.time() - t0) / a.reps
    run[0] = False
    th.join()
    print("timing s=%d O=%d T=%d: fine call wall given %.4f s, devproj %.4f s; AICLK min/median %s/%s MHz"
          % (a.s, a.O, a.T, wall["given"], wall["devproj"], min(clk) if clk else None,
             sorted(clk)[len(clk) // 2] if clk else None))
