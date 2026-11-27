"""storeWeightedSums' dense core on Blackhole in true fp32: no FPU sum anywhere.

proto_wavg_fft.py (g1) put wavg on the FPU matmul: wavg_rel 6.2e-4 and wdiff2_rel 0.018 at box 360 against host fp32
7.2e-7 / 1.8e-6, because a BH matmul accumulates in ~11 significant bits (probe_int, D14) and ttnn.sum does too.
fp32 eltwise ops ARE exact IEEE fp32 on BH (probe_add), so here every sum is an eltwise add:
    wavg[o, :]  = sum_t W[o, t] sh[t, :]           one broadcast multiply-add of [O, 1] x [1, N] per translation t
    wdiff2[:]   = sum_o Wo[o] ref[o]^2 + sum_t Wt[t] sh[t]^2 - 2 sum_o ref[o] wavg[o]
the three [O|T, N] -> [N] reductions are pairwise fp32 add trees over rows on the chip, and the host combines the
three [N] vectors in float64. Same operands, posterior and grading as proto_wavg_fft.py (W tempered so ~nsig
samples are above 1e-6 of the best, as a real fine search spreads it).
    TT_VISIBLE_DEVICES=<chip> python3 wavg_fp32.py --out res.jsonl
"""
import argparse
import json
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--wavg", default="256x84x360,1024x37x100")   # O x T x image size
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--nsig", type=int, default=200)
ap.add_argument("--arms", default="bcast,addcmul")
a = ap.parse_args()

import ttnn  # noqa: E402

from _tt import Clock, open_dev  # noqa: E402

torch.set_num_threads(16)
dev, nodes = open_dev()
fh = open(a.out, "a")


def emit(rec):
    print(json.dumps(rec), flush=True)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


def up(x):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev)


def rows_sum(X):
    """fp32 [R, N] -> [1, N] by a pairwise tree of eltwise adds (exact fp32 rounding per add, no FPU sum)."""
    while X.shape[0] > 1:
        R, N = X.shape[0], X.shape[1]
        h = R // 2
        S = ttnn.add(ttnn.slice(X, [0, 0], [h, N]), ttnn.slice(X, [h, 0], [2 * h, N]))
        if R % 2:
            S = ttnn.concat([S, ttnn.slice(X, [2 * h, 0], [R, N])], 0)
        ttnn.deallocate(X)
        X = S
    return X


def wavg_dev(Wd, Sd, O, T, arm):
    N = Sd.shape[1]
    acc = None
    for t in range(T):
        w = ttnn.slice(Wd, [0, t], [O, t + 1])        # [O, 1]
        s = ttnn.slice(Sd, [t, 0], [t + 1, N])        # [1, N]
        if acc is None:
            acc = ttnn.multiply(w, s)
        elif arm == "addcmul":
            acc2 = ttnn.addcmul(acc, w, s, value=1.0)
            ttnn.deallocate(acc)
            acc = acc2
        else:
            p = ttnn.multiply(w, s)
            acc2 = ttnn.add(acc, p)
            ttnn.deallocate(acc), ttnn.deallocate(p)
            acc = acc2
        ttnn.deallocate(w), ttnn.deallocate(s)
    return acc


for spec in a.wavg.split(","):
    O, T, s = map(int, spec.split("x"))
    g = np.random.default_rng(1)
    imgX = s // 2 + 1
    P = imgX * s
    pix = np.arange(P)
    x, yi = pix % imgX, pix // imgX
    y = np.where(yi > s // 2, yi - s, yi)
    inside = np.tile((x * x + y * y) <= (s // 2) ** 2, 2)
    ref = (g.standard_normal((O, 2 * P)) * 3 * inside).astype(np.float32)
    sh = (ref[np.arange(T) % O] + g.standard_normal((T, 2 * P))).astype(np.float32)
    r64, s64 = ref.astype(np.float64), sh.astype(np.float64)
    d2 = 0.5 * ((r64 ** 2).sum(1)[:, None] + (s64 ** 2).sum(1)[None] - 2 * r64 @ s64.T)
    dd = (d2 - d2.min()).ravel()
    tau = np.partition(dd, a.nsig)[a.nsig] / np.log(1e6)
    W = np.exp(-(d2 - d2.min()) / tau)
    W = (W / W.sum()).astype(np.float32)
    W64 = W.astype(np.float64)
    wavg64 = W64 @ s64
    wd64 = sum((W64[o, :, None] * (r64[o] - s64) ** 2).sum(0) for o in range(O))
    scale, wscale = np.abs(wavg64).max(), np.abs(wd64).max()
    base = {"proto": "wavg_fp32", "O": O, "T": T, "s": s, "P": P, "n_W_above_1e-6": int((W64 > 1e-6 * W64.max()).sum())}
    wv32 = torch.from_numpy(W) @ torch.from_numpy(sh)
    wd32 = torch.zeros(2 * P)
    for o in range(O):
        wd32 += (torch.from_numpy(W[o])[:, None] * (torch.from_numpy(ref[o]) - torch.from_numpy(sh)) ** 2).sum(0)
    emit(dict(base, cfg="torch fp32 (host)", wavg_rel=float(np.abs(wv32.double().numpy() - wavg64).max() / scale),
              wdiff2_rel=float(np.abs(wd32.double().numpy() - wd64).max() / wscale)))
    Wo, Wt = W.sum(1).astype(np.float32), W.sum(0).astype(np.float32)   # O(OT) row/col sums, host
    for arm in a.arms.split(","):
        def run():
            Wd, Sd, Rd = up(W), up(sh), up(ref)
            Wod, Wtd = up(Wo[:, None]), up(Wt[:, None])
            wv = wavg_dev(Wd, Sd, O, T, arm)                                   # [O, N]
            ttnn.synchronize_device(dev)
            t_wv = time.time()   # wavg_s includes the three operand uploads
            cs = rows_sum(ttnn.multiply(Rd, wv))                               # sum_o ref wavg
            r2 = rows_sum(ttnn.multiply(Wod, ttnn.multiply(Rd, Rd)))           # sum_o Wo ref^2
            s2 = rows_sum(ttnn.multiply(Wtd, ttnn.multiply(Sd, Sd)))           # sum_t Wt sh^2
            ttnn.synchronize_device(dev)
            t_red = time.time()
            out = [ttnn.to_torch(z).double().numpy().reshape(-1) for z in (cs, r2, s2)]
            wvh = ttnn.to_torch(wv).double().numpy()
            for z in (Wd, Sd, Rd, Wod, Wtd, wv, cs, r2, s2):
                ttnn.deallocate(z)
            return wvh, out, t_wv, t_red

        clk = Clock(nodes)
        tim = []
        try:
            for rep in range(a.reps + 1):
                ttnn.synchronize_device(dev)
                t0 = time.time()
                wvh, (cs, r2, s2), t_wv, t_red = run()
                if rep:
                    tim.append((t_wv - t0, t_red - t_wv))
        except Exception as ex:  # noqa: BLE001
            clk.stop()
            emit(dict(base, cfg=arm, error=repr(ex)[:400]))
            continue
        cmin, cmed = clk.stop()
        wd = r2 + s2 - 2 * cs
        tm = np.median(np.array(tim), 0)
        emit(dict(base, cfg="fp32 eltwise " + arm, wavg_rel=float(np.abs(wvh - wavg64).max() / scale),
                  wdiff2_rel=float(np.abs(wd - wd64).max() / wscale), wavg_s=round(tm[0], 4),
                  reduce_s=round(tm[1], 4), s_per_call=round(tm.sum(), 4), aiclk_min=cmin, aiclk_med=cmed))
ttnn.close_device(dev)
print("WAVG-DONE", flush=True)
