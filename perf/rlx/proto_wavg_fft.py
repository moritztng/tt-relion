"""Two ttnn prototypes at box 360, each checked against float64: storeWeightedSums' dense core and the particle FFT.

    TT_VISIBLE_DEVICES=<chip> python3 proto_wavg_fft.py --out res.jsonl

wavg (RELION's kernel_wavg, the dense part of storeWeightedSums): for every orientation o of a particle,
    wavg[o, p]  = sum_t W[o, t] sh[t, p]                       (the weighted image, then back-projected)
    wdiff2[p]   = sum_{o,t} W[o, t] |ref[o, p] - sh[t, p]|^2   (the noise-spectrum sums)
              = sum_o Wo[o] ref[o,p]^2 + sum_t Wt[t] sh[t,p]^2 - 2 sum_o ref[o, p] wavg[o, p]
The chip does the matmul and the last contraction; the two weight-sum terms are O(P) on the host in float64.
W is a real posterior: softmax of -d2 over O x T, so mostly near zero, as RELION hands it over.

fft (the particle FFT, real 360 x 360 -> 360 x 181 complex, RELION's r2c): two DFT matmuls,
    Z = X . [C | -S]   (x axis, rfft half),   Y^T = Z^T . [C | S]   (y axis), Y = (C - iS) X (C - iS)^T restricted.
Both operands go up fp32 and are split into bf16 parts ON the chip (hi = bf16(a), a -= hi, ...), so the
intermediate Z never leaves the chip; parts i + j < split are multiplied (split 3 = the diff2 bridge's scheme).

Each prototype reports the error against float64 and against torch fp32 (RELION's own precision class), and
warm s/call (median of --reps after one warm-up), with AICLK sampled during the calls.
"""
import argparse
import json
import os
import threading
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--splits", default="3,2")
ap.add_argument("--fids", default="HiFi4,HiFi2")
ap.add_argument("--fft_batch", default="8,64")
ap.add_argument("--wavg", default="256x84x360,1024x37x100")   # O x T x image size
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

torch.set_num_threads(8)
dev = ttnn.open_device(device_id=0)


def tt_nodes():
    n = set()
    for f in os.listdir("/proc/self/fd"):
        try:
            t = os.readlink("/proc/self/fd/" + f)
        except OSError:
            continue
        if t.startswith("/dev/tenstorrent/"):
            n.add(int(t.rsplit("/", 1)[1]))
    return sorted(n)


NODES = tt_nodes()
print("device nodes opened:", NODES, flush=True)


class Clock:
    def __init__(self):
        self.paths = ["/sys/class/tenstorrent/tenstorrent!%d/tt_aiclk" % n for n in NODES]
        self.v, self.run = [], True
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.th.start()

    def loop(self):
        while self.run:
            for p in self.paths:
                try:
                    self.v.append(int(open(p).read().split()[0]))
                except (OSError, ValueError):
                    pass
            time.sleep(0.25)

    def stop(self):
        self.run = False
        self.th.join()
        return (min(self.v), int(np.median(self.v))) if self.v else (None, None)


def up(x, dt=ttnn.float32):
    return ttnn.from_torch(x.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)


def split_dev(x, k):
    """x (fp32 on chip) -> k bf16 tensors whose sum is x to ~2^-8k relative."""
    parts = []
    for i in range(k):
        h = ttnn.typecast(x, ttnn.bfloat16)
        parts.append(h)
        if i < k - 1:
            x2 = ttnn.subtract(x, ttnn.typecast(h, ttnn.float32))
            if i:
                ttnn.deallocate(x)
            x = x2
    if k > 1:
        ttnn.deallocate(x)
    return parts


def mm_split(A, B, k, cfg):
    """A @ B, both fp32 on chip, via bf16 parts: sum over i + j < k of A_i @ B_j, fp32 accumulate."""
    ap_, bp = split_dev(A, k), split_dev(B, k)
    acc = None
    for i in range(k):
        for j in range(k - i):
            c = ttnn.matmul(ap_[i], bp[j], compute_kernel_config=cfg, dtype=ttnn.float32)
            if acc is None:
                acc = c
            else:
                acc2 = ttnn.add(acc, c)
                ttnn.deallocate(acc)
                ttnn.deallocate(c)
                acc = acc2
    for x in ap_ + bp:
        ttnn.deallocate(x)
    return acc


def timed(fn, reps):
    clk = Clock()
    ts = []
    out = None
    for r in range(reps + 1):
        t0 = time.time()
        out = fn()
        ttnn.synchronize_device(dev)
        if r:
            ts.append(time.time() - t0)
    cmin, cmed = clk.stop()
    return out, float(np.median(ts)), cmin, cmed


fh = open(a.out, "a")


def emit(rec):
    print(json.dumps(rec), flush=True)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


def cfg_of(fid):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)


# ---------------- wavg ----------------
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
    W64 = np.exp(-(d2 - d2.min()))
    W64 /= W64.sum()
    W = W64.astype(np.float32)
    W64 = W.astype(np.float64)
    wavg64 = W64 @ s64
    wd64 = (W64.sum(1) @ r64 ** 2) + (W64.sum(0) @ s64 ** 2) - 2 * (r64 * wavg64).sum(0)
    wd64_direct = sum((W64[o, :, None] * (r64[o] - s64) ** 2).sum(0) for o in range(O))
    scale = np.abs(wavg64).max()
    wv32 = torch.from_numpy(W) @ torch.from_numpy(sh)
    wd32 = torch.zeros(2 * P)
    for o in range(O):
        wd32 += (torch.from_numpy(W[o])[:, None] * (torch.from_numpy(ref[o]) - torch.from_numpy(sh)) ** 2).sum(0)
    base = {"proto": "wavg", "O": O, "T": T, "s": s, "P": P, "n_W_above_1e-6": int((W64 > 1e-6).sum()),
            "decomp_vs_direct_f64": float(np.abs(wd64 - wd64_direct).max() / np.abs(wd64_direct).max())}
    emit(dict(base, cfg="torch fp32 (host)",
              wavg_rel=float(np.abs(wv32.double().numpy() - wavg64).max() / scale),
              wdiff2_rel=float(np.abs(wd32.double().numpy() - wd64_direct).max() / np.abs(wd64_direct).max())))
    Wt, St, Rt = torch.from_numpy(W), torch.from_numpy(sh), torch.from_numpy(ref)
    for k in map(int, a.splits.split(",")):
        for fid in a.fids.split(","):
            cfg = cfg_of(fid)

            def run():
                Wd, Sd, Rd = up(Wt), up(St), up(Rt)
                wv = mm_split(Wd, Sd, k, cfg)                       # [O, 2P]
                pr = ttnn.multiply(Rd, wv)
                cs = ttnn.sum(pr, dim=0, keepdim=True, compute_kernel_config=cfg)   # [1, 2P]
                res = ttnn.to_torch(wv).double().numpy(), ttnn.to_torch(cs).double().numpy().reshape(-1)
                for X in (Wd, Sd, Rd, wv, pr, cs):
                    ttnn.deallocate(X)
                return res

            try:
                (wv, cs), sec, cmin, cmed = timed(run, a.reps)
            except Exception as ex:  # noqa: BLE001
                emit(dict(base, cfg="split%d %s" % (k, fid), error=repr(ex)[:300]))
                continue
            wd = (W64.sum(1) @ r64 ** 2) + (W64.sum(0) @ s64 ** 2) - 2 * cs
            emit(dict(base, cfg="split%d %s" % (k, fid), split=k, fid=fid,
                      wavg_rel=float(np.abs(wv - wavg64).max() / scale),
                      wdiff2_rel=float(np.abs(wd - wd64_direct).max() / np.abs(wd64_direct).max()),
                      s_per_call=round(sec, 4), aiclk_min=cmin, aiclk_med=cmed))

# ---------------- fft ----------------
N = 360
kk = np.arange(N)
ang = 2 * np.pi * np.outer(kk, kk) / N
C, S = np.cos(ang), np.sin(ang)
Bx = torch.from_numpy(np.concatenate([C[:, :N // 2 + 1], -S[:, :N // 2 + 1]], 1).astype(np.float32))  # [360, 362]
By = torch.from_numpy(np.concatenate([C, S], 1).astype(np.float32))                                   # [360, 720]
H = N // 2 + 1
for nb in map(int, a.fft_batch.split(",")):
    g = np.random.default_rng(2)
    X = g.standard_normal((nb, N, N)).astype(np.float32)     # a normalised particle: unit-variance noise
    X[:, 120:240, 120:240] += 3 * g.standard_normal((120, 120)).astype(np.float32)
    Y64 = np.fft.rfft2(X.astype(np.float64))                  # [nb, 360, 181]
    sc = np.abs(Y64).max()
    Y32 = torch.fft.rfft2(torch.from_numpy(X)).numpy()
    base = {"proto": "fft", "box": N, "batch": nb}
    emit(dict(base, cfg="torch fp32 rfft2 (host)", rel=float(np.abs(Y32 - Y64).max() / sc)))
    Xt = torch.from_numpy(X)
    for k in map(int, a.splits.split(",")):
        for fid in a.fids.split(","):
            cfg = cfg_of(fid)

            def run():
                Xd, Bxd, Byd = up(Xt), up(Bx), up(By)
                Z = mm_split(Xd, Bxd, k, cfg)                               # [nb, 360(y), 362]
                Zt = ttnn.transpose(Z, -2, -1)                              # [nb, 362, 360(y)]
                Q = mm_split(Zt, Byd, k, cfg)                               # [nb, 362, 720]
                q = ttnn.to_torch(Q).double().reshape(nb, 2, H, 2, N)       # [img, Zr|Zi, kx, C|S, ky]
                for T_ in (Xd, Bxd, Byd, Z, Zt, Q):
                    ttnn.deallocate(T_)
                yr = q[:, 0, :, 0] + q[:, 1, :, 1]                          # C Zr + S Zi
                yi = q[:, 1, :, 0] - q[:, 0, :, 1]                          # C Zi - S Zr
                return (yr + 1j * yi).transpose(1, 2).numpy()

            try:
                Yd, sec, cmin, cmed = timed(run, a.reps)
            except Exception as ex:  # noqa: BLE001
                emit(dict(base, cfg="split%d %s" % (k, fid), error=repr(ex)[:300]))
                continue
            emit(dict(base, cfg="split%d %s" % (k, fid), split=k, fid=fid,
                      rel=float(np.abs(Yd - Y64).max() / sc), s_per_call=round(sec, 4),
                      s_per_image=round(sec / nb, 5), aiclk_min=cmin, aiclk_med=cmed))
ttnn.close_device(dev)
print("PROTO-DONE", flush=True)
