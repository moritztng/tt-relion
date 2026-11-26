"""Unit parity of the ttnn diff2 path against a float64 direct-form reference, at a coarse-pass shape.

    TT_VISIBLE_DEVICES=<chip> python3 parity_cross.py [--O 4608] [--T 21] [--box 64]

The image is a projection of the model plus unit-variance noise and w is 1/(2 sigma^2), so d2 sits at
the scale RELION sees (about the pixel count) and the quantity that enters the posterior, d2 - min(d2),
is order one near the best match. Reports the error of both the fp32 torch path (RELION's own
precision class) and the device path against float64, absolute and in that posterior-relevant form.
"""
import argparse
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--O", type=int, default=4608)
ap.add_argument("--T", type=int, default=21)
ap.add_argument("--box", type=int, default=64)
a = ap.parse_args()

os.environ["TT_RELION_BACKEND"] = "torch"
import tt_relion.relion as R  # noqa: E402

g = np.random.default_rng(0)
box, pf = a.box, 2
imgX, imgY, maxR = box // 2 + 1, box, box // 2
P = imgX * imgY
mdlX, mdlY, mdlZ = pf * box // 2 + 1 + 2, pf * box + 3, pf * box + 3
mdlInitY = mdlInitZ = -(pf * box // 2) - 1
maxR2p = (maxR * pf) ** 2
mdl = (g.standard_normal((mdlZ * mdlY * mdlX, 2)) * 3).astype(np.float32)


def rot(n):
    q = g.standard_normal((n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z),
                     2*(y*z-x*w), 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], 1).astype(np.float32)


eul = rot(a.O)
tx = (g.integers(-5, 6, a.T) * 2 * np.pi / box).astype(np.float32)
ty = (g.integers(-5, 6, a.T) * 2 * np.pi / box).astype(np.float32)
t = torch
pix = t.arange(P); x = (pix % imgX).double(); yi = pix // imgX
y = t.where(yi > maxR, yi - imgY, yi).double()
ref0_r, ref0_i = R._project(t, t.from_numpy(mdl).double(), x, y, t.from_numpy(eul[:1]).double(),
                            mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR2p, pf)
img_r = (ref0_r[0] + t.from_numpy(g.standard_normal(P))).float().numpy()
img_i = (ref0_i[0] + t.from_numpy(g.standard_normal(P))).float().numpy()
corr = np.ones(P, np.float32)  # w = corr / 2 = 1 / (2 sigma^2), sigma = 1

args = [memoryview(v) for v in (mdl, eul, tx, ty, img_r, img_i, corr)]
geo = (mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2p, pf, imgX, imgY, a.O, a.T, P, "C")

# float64 direct form
md, ed = t.from_numpy(mdl).double(), t.from_numpy(eul).double()
ir, ii, w = t.from_numpy(img_r).double(), t.from_numpy(img_i).double(), t.from_numpy(corr).double() * 0.5
ph = x.unsqueeze(0) * t.from_numpy(tx).double().unsqueeze(1) + y.unsqueeze(0) * t.from_numpy(ty).double().unsqueeze(1)
s, c = t.sin(ph), t.cos(ph); sr, si = c * ir - s * ii, c * ii + s * ir
ref64 = np.empty(a.O * a.T)
for o0 in range(0, a.O, 256):
    rr, ri = R._project(t, md, x, y, ed[o0:o0 + 256], mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR2p, pf)
    d = ((rr.unsqueeze(1) - sr) ** 2 + (ri.unsqueeze(1) - si) ** 2) * w
    ref64[o0 * a.T:(o0 + rr.shape[0]) * a.T] = d.sum(-1).reshape(-1).numpy()


def report(name, v):
    e = np.abs(v.astype(np.float64) - ref64)
    post = np.abs((v - v.min()) - (ref64 - ref64.min()))
    near = (ref64 - ref64.min()) < 30          # entries whose posterior weight is above exp(-30)
    print("%-6s max|d2-ref|=%.3e  max rel=%.3e  max|dpost| near best=%.3e  (d2 min %.1f, %d near)"
          % (name, e.max(), (e / np.abs(ref64)).max(), post[near].max(), ref64.min(), near.sum()))


t0 = time.time(); v32 = R._dense_diff2(t, *args, *geo); t1 = time.time()
report("torch", v32)
R._BACKEND = "ttnn"
vtt = R._dense_diff2(t, *args, *geo); t2 = time.time()
vtt = R._dense_diff2(t, *args, *geo); t3 = time.time()
report("ttnn", vtt)
print("wall torch %.3f s, ttnn first %.3f s, ttnn warm %.3f s" % (t1 - t0, t2 - t1, t3 - t2))
