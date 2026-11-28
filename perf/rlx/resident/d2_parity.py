"""tt_relion.resident.Resident.d2 on one chip against float64, on a synthetic model in RELION's geometry.

Reference: the bridge's _project (RELION's trilinear projector) in float64, then the direct sum w |P - S X|^2.
Graded: relative error of d2 near each particle's minimum, argmin agreement, and that a held orientation set
(ref_key, second call) returns what a one-off call returns.
  TT_VISIBLE_DEVICES=16 python d2_parity.py --size 50 --O 4608 --N 8 --T 21 --out r.jsonl
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
sys.path.insert(0, __file__.rsplit("/", 4)[0])
ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=50)
ap.add_argument("--O", type=int, default=4608)
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--T", type=int, default=21)
ap.add_argument("--pf", type=int, default=2)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from _tt import Clock, open_dev  # noqa: E402
from tt_relion.relion import _project  # noqa: E402
from tt_relion.resident import Resident  # noqa: E402

rng = np.random.default_rng(0)
s, pf = a.size, a.pf
maxR = s // 2 - 1
imgX, imgY = s // 2 + 1, s
P = imgX * imgY
pix = np.arange(P)
x = (pix % imgX).astype(np.float32)
yi = pix // imgX
y = np.where(yi > maxR, yi - imgY, yi).astype(np.float32)
rp = pf * maxR
X, Y, Z, initY, initZ, maxR2 = rp + 3, 2 * rp + 5, 2 * rp + 5, -(rp + 2), -(rp + 2), rp * rp
zz, yy, xx = np.meshgrid(np.arange(Z) + initZ, np.arange(Y) + initY, np.arange(X), indexing="ij")
amp = 1.0 / (1.0 + (xx * xx + yy * yy + zz * zz) / (0.1 * rp) ** 2)          # a map-like radial falloff
mdl = (rng.normal(size=(Z, Y, X, 2)) * amp[..., None]).reshape(-1, 2).astype(np.float32)


def rand_rot(n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w_, i, j, k = q.T
    return np.stack([1 - 2 * (j * j + k * k), 2 * (i * j - k * w_), 2 * (i * k + j * w_),
                     2 * (i * j + k * w_), 1 - 2 * (i * i + k * k), 2 * (j * k - i * w_),
                     2 * (i * k - j * w_), 2 * (j * k + i * w_), 1 - 2 * (i * i + j * j)], 1).astype(np.float32)


eul = rand_rot(a.O)
t64 = lambda v: torch.from_numpy(np.asarray(v, np.float64))  # noqa: E731
pr, pi = _project(torch, t64(mdl), t64(x), t64(y), t64(eul), X, Y, Z, initY, initZ, maxR2, pf)
pr, pi = pr.numpy(), pi.numpy()                                                # [O, P] float64
inside = (np.hypot(x, y) <= maxR)
w = (inside * rng.uniform(0.5, 2.0, (a.N, P)) * (1 + np.hypot(x, y)) ** 1.5).astype(np.float32)   # 1/sigma2 grows with k
truth = rng.integers(0, a.O, a.N)
noise = rng.normal(size=(2, a.N, P)) / np.sqrt(np.maximum(w, 1e-3))
xr = (pr[truth] + noise[0]).astype(np.float32)
xi = (pi[truth] + noise[1]).astype(np.float32)
tx = np.tile((rng.uniform(-2, 2, a.T) * 2 * np.pi / s).astype(np.float32), (a.N, 1))
ty = np.tile((rng.uniform(-2, 2, a.T) * 2 * np.pi / s).astype(np.float32), (a.N, 1))
tx[:, 0] = ty[:, 0] = 0

ph = tx[:, :, None].astype(np.float64) * x + ty[:, :, None].astype(np.float64) * y
sx = (xr[:, None] + 1j * xi[:, None]) * np.exp(1j * ph)                        # [N, T, P]
ref = pr + 1j * pi
d64 = np.empty((a.N, a.O, a.T))
for n in range(a.N):
    wn = w[n].astype(np.float64)
    cross = (ref.conj() * wn) @ sx[n].T                                        # [O, T]
    d64[n] = (wn * np.abs(sx[n]) ** 2).sum(1)[None] + (wn * np.abs(ref) ** 2).sum(1)[:, None] - 2 * cross.real

dev, nodes = open_dev()
res = Resident(dev)
res.set_model("m", mdl, X, Y, Z, initY, initZ, maxR2, pf)
res.set_pixels(P, x, y)
rec = dict(size=s, O=a.O, N=a.N, T=a.T, P=P)


def grade(name, d):
    near = d64 < d64.min((1, 2), keepdims=True) + 40
    rec[name] = dict(err_near=float(np.abs(d - d64)[near].max()), rel_all=float(np.abs(d - d64).max() / np.abs(d64).max()),
                     argmin=int((d.reshape(a.N, -1).argmin(1) == d64.reshape(a.N, -1).argmin(1)).sum()),
                     d2_at_min=float(d64.min()))


grade("once", res.d2(eul, xr, xi, w, tx, ty, x, y))            # first call: includes the kernel builds
res.t = {}
t0 = time.perf_counter()
res.d2(eul, xr, xi, w, tx, ty, x, y)
rec["once_s"] = time.perf_counter() - t0
rec["once_phases"], res.t = {k: round(v, 4) for k, v in res.t.items()}, {}
vary = []                                       # unheld calls on set sizes the chip has not seen (a fine pass)
for O, T in ((a.O - 300, a.T), (a.O - 700, a.T - 4), (a.O - 301, a.T - 5), (a.O - 699, a.T - 1)):
    t0 = time.perf_counter()
    d = res.d2(eul[:O], xr, xi, w, tx[:, :T], ty[:, :T], x, y)
    vary.append(round(time.perf_counter() - t0, 4))
    assert (d.reshape(a.N, -1).argmin(1) == d64[:, :O, :T].reshape(a.N, -1).argmin(1)).all()
rec["vary_s"] = vary
held = res.d2(eul, xr, xi, w, tx, ty, x, y, ref_key="it")
grade("held_first", held)
res.t = {}
clk = Clock(nodes)
ts = []
for _ in range(a.reps):
    t0 = time.perf_counter()
    again = res.d2(eul, xr, xi, w, tx, ty, x, y, ref_key="it")
    ts.append(time.perf_counter() - t0)
rec["aiclk_min_med"] = clk.stop()
rec["held_phases"] = {k: round(v / a.reps, 4) for k, v in res.t.items()}
rec["load1"] = float(open("/proc/loadavg").read().split()[0])
grade("held_again", again)
rec["held_same"] = bool((again == held).all())
rec["batch_s"] = ts
print(json.dumps(rec))
open(a.out, "a").write(json.dumps(rec) + "\n")
