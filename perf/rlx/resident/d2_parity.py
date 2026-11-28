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
ap.add_argument("--digits", type=int, default=13, help="base-4 digits, the device server's default")
ap.add_argument("--out", required=True)
ap.add_argument("--cand", type=float, default=0, help="also grade the candidate form: samples within this window of each particle's best")
ap.add_argument("--own", type=int, default=0, help="also grade d2_own: particle n scores its own orientations n*own .. (n+1)*own (needs O >= N*own)")
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
if a.own:                                       # every particle's pose lies in its own set
    assert a.O >= a.N * a.own
    truth = np.arange(a.N) * a.own + rng.integers(0, a.own, a.N)
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
res = Resident(dev, a.digits)
res.set_model("m", mdl, X, Y, Z, initY, initZ, maxR2, pf)
res.set_pixels(P, x, y)
rec = dict(size=s, O=a.O, N=a.N, T=a.T, P=P, digits=a.digits)


def grade(name, d):
    near = d64 < d64.min((1, 2), keepdims=True) + 40
    rec[name] = dict(err_near=float(np.abs(d - d64)[near].max()), rel_all=float(np.abs(d - d64).max() / np.abs(d64).max()),
                     argmin=int((d.reshape(a.N, -1).argmin(1) == d64.reshape(a.N, -1).argmin(1)).sum()),
                     d2_at_min=float(d64.min()))


def d2(eul, tx, ty, **k):                         # the op takes flat (particle, translation) rows
    N, T = tx.shape
    return res.d2(eul, xr, xi, w, tx.ravel(), ty.ravel(), np.repeat(np.arange(N), T), x, y, **k).reshape(N, T, -1).transpose(0, 2, 1)


grade("once", d2(eul, tx, ty))            # first call: includes the kernel builds
phases = lambda c0, n=1: {k: round((v - c0[k]) / n, 4) for k, v in res.c.items() if k.endswith("_s")}  # noqa: E731
c0 = dict(res.c)
t0 = time.perf_counter()
d2(eul, tx, ty)
rec["once_s"] = time.perf_counter() - t0
rec["once_phases"] = phases(c0)
vary = []                                       # unheld calls on set sizes the chip has not seen (a fine pass)
for O, T in ((a.O - a.O // 15, a.T), (a.O - a.O // 7, max(1, a.T - 4)), (a.O - a.O // 15 - 1, max(1, a.T - 5)), (a.O - a.O // 7 + 1, a.T - 1)):
    t0 = time.perf_counter()
    d = d2(eul[:O], tx[:, :T], ty[:, :T])
    vary.append(round(time.perf_counter() - t0, 4))
    assert (d.reshape(a.N, -1).argmin(1) == d64[:, :O, :T].reshape(a.N, -1).argmin(1)).all()
rec["vary_s"] = vary
held = d2(eul, tx, ty, ref_key="it")
grade("held_first", held)
c0 = dict(res.c)
clk = Clock(nodes)
ts = []
for _ in range(a.reps):
    t0 = time.perf_counter()
    again = d2(eul, tx, ty, ref_key="it")
    ts.append(time.perf_counter() - t0)
rec["aiclk_min_med"] = clk.stop()
rec["held_phases"] = phases(c0, a.reps)
rec["load1"] = float(open("/proc/loadavg").read().split()[0])
grade("held_again", again)
rec["held_same"] = bool((again == held).all())
rec["batch_s"] = ts
if a.cand:      # the candidate form against float64 with the same priors and window, and RELION's rule on both
    lo = rng.uniform(0, 4, a.O)
    lo[rng.random(a.O) < 0.03] = np.inf                                        # orientations the prior excludes
    lt = rng.uniform(0, 6, (a.N, a.T))
    pnr = np.repeat(np.arange(a.N), a.T)
    E = d64.transpose(0, 2, 1) + lo[None, None, :] + lt[:, :, None]            # [N, T, O]

    def rule(flat, delta):                      # the largest weights that hold 99.9 % of the weight sum
        o = np.argsort(delta, kind="stable")
        cs = np.cumsum(np.exp(-delta[o]))
        return set(flat[o[:int(np.searchsorted(cs, 0.999 * cs[-1]) + 1)]].tolist())

    c0, ts = dict(res.c), []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        rr, oo, dl, best = res.d2(eul, xr, xi, w, tx.ravel(), ty.ravel(), pnr, x, y, ref_key="it",
                                  prior=(lo, lt.ravel(), a.cand))
        ts.append(time.perf_counter() - t0)
    same = missed = 0
    err = errb = 0.0
    for n in range(a.N):
        refd = (E[n] - E[n].min()).ravel()
        k = rr // a.T == n
        fl, dv = (rr[k].astype(np.int64) % a.T) * a.O + oo[k], dl[k].astype(np.float64)
        got = np.full(refd.size, np.nan)
        got[fl] = dv
        want = np.flatnonzero(refd < a.cand - 1.0)                              # a unit inside the window: all must be there
        missed += int(np.isnan(got[want]).sum())
        err = max(err, float(np.nanmax(np.abs(got[want] - refd[want]))))
        errb = max(errb, abs(float(best[n] - E[n].min())))
        inw = np.flatnonzero(refd < a.cand)
        same += rule(fl, dv) == rule(inw, refd[inw])
    rec["cand"] = dict(window=a.cand, n=int(len(rr)), per_particle=len(rr) / a.N, missed=missed, same_sig=f"{same}/{a.N}",
                       err_delta=err, err_best=errb, batch_s=ts, phases=phases(c0, a.reps),
                       counters={k: res.c.get(k) for k in ("cand", "cand_dense")})
if a.own:      # the per-particle form against float64, and against the batch form of the same sets (the union)
    pnr = np.repeat(np.arange(a.N), a.T)
    eo, on = eul[:a.N * a.own], np.full(a.N, a.own)
    ref = np.stack([d64[n, n * a.own:(n + 1) * a.own].T for n in range(a.N)])   # [N, T, own]
    got = res.d2_own(eo, on, xr, xi, w, tx.ravel(), ty.ravel(), pnr, x, y)      # the first call builds the kernels
    c0, ts, clk = dict(res.c), [], Clock(nodes)
    for _ in range(a.reps):
        t0 = time.perf_counter()
        got = res.d2_own(eo, on, xr, xi, w, tx.ravel(), ty.ravel(), pnr, x, y)
        ts.append(time.perf_counter() - t0)
    d = np.stack(got).astype(np.float64)
    near = ref < ref.min((1, 2), keepdims=True) + 40
    bat = np.stack([again[n, n * a.own:(n + 1) * a.own].T for n in range(a.N)])
    rec["own"] = dict(own=a.own, err_near=float(np.abs(d - ref)[near].max()), rel_all=float(np.abs(d - ref).max() / np.abs(ref).max()),
                      argmin=int((d.reshape(a.N, -1).argmin(1) == ref.reshape(a.N, -1).argmin(1)).sum()),
                      vs_batch_near=float(np.abs(d - bat)[near].max()), batch_s=ts, ms_per_particle=1e3 * min(ts) / a.N,
                      phases=phases(c0, a.reps), aiclk_min_med=clk.stop(), load1=float(open("/proc/loadavg").read().split()[0]))
    # sets and translation lists of different lengths in one call
    on2 = np.array([max(1, a.own - 5 * n) for n in range(a.N)])
    nt2 = np.array([max(1, a.T - n % 3) for n in range(a.N)])
    eo2 = np.concatenate([eul[n * a.own:n * a.own + on2[n]] for n in range(a.N)])
    cat = lambda v: np.concatenate([v[n, :nt2[n]] for n in range(a.N)])  # noqa: E731
    got2 = res.d2_own(eo2, on2, xr, xi, w, cat(tx), cat(ty), np.repeat(np.arange(a.N), nt2), x, y)
    err2, am2 = 0.0, 0
    for n in range(a.N):
        r_ = ref[n, :nt2[n], :on2[n]]
        assert got2[n].shape == r_.shape
        err2 = max(err2, float(np.abs(got2[n] - r_)[r_ < r_.min() + 40].max()))
        am2 += int(got2[n].argmin() == r_.argmin())
    rec["own_ragged"] = dict(err_near=err2, argmin=am2)
print(json.dumps(rec))
open(a.out, "a").write(json.dumps(rec) + "\n")
