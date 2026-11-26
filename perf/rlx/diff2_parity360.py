"""diff2 cross-term accuracy and speed at the dev cell's real sizes (box 360, pad 2), one chip.

    TT_VISIBLE_DEVICES=<chip> python3 diff2_parity360.py --out res.jsonl [--cases c26,c50,...]

The bridge (tt_relion/relion.py::_device_cross) sends only C = ref . shw^T to the chip; the norms stay on the
host in float64, so the d2 error is exactly 2 |C_dev - C_64|. This script times and checks that matmul alone,
with the same operand construction, over the accumulation choices that cost something:

  split   bf16 parts per operand (3 = the bridge; 2 drops a1.b1 too; 0 = plain fp32 operands)
  dtype   how the parts travel: fp32 (the bridge) or bf16 (they are bf16-exact, so lossless, half the bytes)
  fid     math fidelity
  kc      K run length; each run is its own batch entry (0 = one run)
  sum     where the runs are added: host float64 (the bridge) or on the chip in fp32 (one [m, n] comes back)

Operands are whitened Fourier data (w = 1/2, unit noise), which is the scale RELION's weighted d2 lives at:
ref is signal with amplitude 3 inside the projector's r_max disc and 0 outside; every translation row is a
reference row plus unit noise. The float64 reference uses the same fp32-rounded operands, so the error is
the device's alone. "fp32 direct" is RELION's own precision class (sum over K of w (ref - sh)^2 in fp32).
Case sizes are RELION's: coarse image sizes 26/50/100 px at healpix 2/3/4 (ml_optimiser.cpp, keepsafe 1.2),
fine at 146 px and the full 360 px; O and T as the bridge issues them (_DEV_CHUNK 1024 coarse, rlx-host's
fine shapes). box64 reproduces the floor row's unit-parity cell for calibration.
AICLK of every chip this process opened is sampled every 0.25 s during each config.
"""
import argparse
import json
import os
import threading
import time

import numpy as np
import torch

CASES = {  # name: (image size s, O, T)
    "box64": (64, 1024, 21),
    "c26": (26, 1024, 21),
    "c50": (50, 1024, 37),
    "c100": (100, 1024, 37),
    "f146": (146, 256, 84),
    "f360": (360, 256, 84),
}

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cases", default=",".join(CASES))
ap.add_argument("--splits", default="3,2,0")
ap.add_argument("--dtypes", default="bf16,fp32")
ap.add_argument("--fids", default="HiFi4,HiFi3,HiFi2")
ap.add_argument("--kcs", default="0,1024,256,64,32")
ap.add_argument("--sums", default="host,dev")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402  (a lone p300 chip needs the 1x1 MGD)
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

torch.set_num_threads(8)


def operands(s, O, T, g):
    imgX = s // 2 + 1
    P = imgX * s
    pix = np.arange(P)
    x = pix % imgX
    yi = pix // imgX
    y = np.where(yi > s // 2, yi - s, yi)
    inside = (x * x + y * y) <= (s // 2) ** 2
    ref = g.standard_normal((O, 2 * P)) * 3 * np.tile(inside, 2)
    sh = ref[np.arange(T) % O] + g.standard_normal((T, 2 * P))
    w = 0.5
    return ref.astype(np.float32), sh.astype(np.float32), w


def bf16_split(a, k):
    parts = []
    for _ in range(k - 1):
        hi = (a.view(torch.int32) & ~0xFFFF).view(torch.float32)
        parts.append(hi)
        a = a - hi
    return parts + [a]


def build(ref, shw, k, kc):
    """The bridge's operands: lhs [nc, O, kc], rhs [nc, kc, k*T] (k=0: plain fp32, one part)."""
    ref, shw = torch.from_numpy(ref), torch.from_numpy(shw)
    if k == 0:
        lhs, rhs, nb = ref, shw.T.contiguous(), 1
    else:
        kk, n = ref.shape[1], shw.shape[0]
        lhs = torch.cat(bf16_split(ref, k), 1)
        bs = bf16_split(shw, k)
        rhs = torch.zeros(k * kk, k * n, dtype=torch.float32)
        for i in range(k):
            for j in range(k - i):
                rhs[i * kk:(i + 1) * kk, j * n:(j + 1) * n] = bs[j].T
        nb = k
    K = lhs.shape[1]
    kc = kc if kc > 0 else K
    nc = -(-K // kc)
    pad = nc * kc - K
    if pad:
        lhs = torch.nn.functional.pad(lhs, (0, pad))
        rhs = torch.nn.functional.pad(rhs, (0, 0, 0, pad))
    lhs = lhs.reshape(lhs.shape[0], nc, kc).permute(1, 0, 2).contiguous()
    rhs = rhs.reshape(nc, kc, rhs.shape[1])
    return lhs, rhs, nb


class Clock:
    """Samples tt_aiclk of the given /dev/tenstorrent nodes until stopped."""

    def __init__(self, nodes):
        self.paths = ["/sys/class/tenstorrent/tenstorrent!%d/tt_aiclk" % n for n in nodes]
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


dev = ttnn.open_device(device_id=0)


def tt_nodes():
    """/dev/tenstorrent/N nodes this process holds open (UMD may open a p300 board's partner chip too)."""
    n = set()
    for f in os.listdir("/proc/self/fd"):
        try:
            t = os.readlink("/proc/self/fd/" + f)
        except OSError:
            continue
        if t.startswith("/dev/tenstorrent/"):
            n.add(int(t.rsplit("/", 1)[1]))
    return sorted(n)


nodes = tt_nodes()
print("device nodes opened:", nodes, flush=True)
DT = {"fp32": (torch.float32, ttnn.float32), "bf16": (torch.bfloat16, ttnn.bfloat16)}


def on_chip(lhs, rhs, dtype, cfg, dsum):
    tdt, ndt = DT[dtype]
    t0 = time.time()
    A = ttnn.from_torch(lhs.to(tdt), dtype=ndt, layout=ttnn.TILE_LAYOUT, device=dev)
    B = ttnn.from_torch(rhs.to(tdt), dtype=ndt, layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.synchronize_device(dev)
    t1 = time.time()
    C = ttnn.matmul(A, B, compute_kernel_config=cfg, dtype=ttnn.float32)
    if dsum:
        S = ttnn.sum(C, dim=0, keepdim=True, compute_kernel_config=cfg)
        ttnn.deallocate(C)
        C = S
    ttnn.synchronize_device(dev)
    t2 = time.time()
    out = ttnn.to_torch(C).float()
    t3 = time.time()
    for X in (A, B, C):
        ttnn.deallocate(X)
    return out, (t1 - t0, t2 - t1, t3 - t2)


fh = open(a.out, "a")
for case in a.cases.split(","):
    s, O, T = CASES[case]
    g = np.random.default_rng(0)
    ref, sh, w = operands(s, O, T, g)
    shw = (sh * np.float32(w)).astype(np.float32)
    r64, sh64 = ref.astype(np.float64), sh.astype(np.float64)
    C64 = r64 @ (shw.astype(np.float64)).T
    ref2 = (r64 ** 2).sum(1) * w
    sh2 = (sh64 ** 2).sum(1) * w
    d64 = ref2[:, None] + sh2[None, :] - 2 * C64
    near = (d64 - d64.min()) < 30
    # RELION's own precision class: the direct form in fp32
    rt, st = torch.from_numpy(ref), torch.from_numpy(sh)
    d32 = torch.empty(O, T, dtype=torch.float32)
    for o0 in range(0, O, 8):
        d32[o0:o0 + 8] = (((rt[o0:o0 + 8, None] - st[None]) ** 2) * w).sum(-1)
    e32 = np.abs(d32.double().numpy() - d64)
    base = {"case": case, "s": s, "O": O, "T": T, "K": ref.shape[1], "d2_min": float(d64.min()),
            "n_near": int(near.sum())}
    rec = dict(base, cfg="fp32 direct (host)", err_max=float(e32.max()), err_near=float(e32[near].max()))
    print(json.dumps(rec), flush=True)
    fh.write(json.dumps(rec) + "\n"); fh.flush()
    for k in map(int, a.splits.split(",")):
        for kc in map(int, a.kcs.split(",")):
            if kc >= max(k, 1) * ref.shape[1]:
                continue   # one run: same as kc 0
            th = time.time()
            lhs, rhs, nb = build(ref, shw, k, kc)
            host_build = time.time() - th
            nc = lhs.shape[0]
            for dtype in a.dtypes.split(","):
                if k == 0 and dtype == "bf16":
                    continue   # plain fp32 operands are not bf16-exact
                for fid in a.fids.split(","):
                    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                           math_approx_mode=False, fp32_dest_acc_en=True,
                                                           packer_l1_acc=True)
                    for sm in a.sums.split(","):
                        if sm == "dev" and nc == 1:
                            continue
                        name = "split%d %s %s kc%d sum-%s" % (k, dtype, fid, kc, sm)
                        try:
                            clk = Clock(nodes)
                            times = []
                            for r in range(a.reps + 1):
                                out, tt = on_chip(lhs, rhs, dtype, cfg, sm == "dev")
                                th = time.time()
                                C = out.double().sum(0).reshape(O, nb, T).sum(1).numpy()
                                hs = time.time() - th
                                if r:
                                    times.append(tt + (hs,))
                            cmin, cmed = clk.stop()
                        except Exception as ex:  # noqa: BLE001  (an OOM or unsupported shape is a result)
                            clk.stop()
                            rec = dict(base, cfg=name, error=repr(ex)[:300])
                            print(json.dumps(rec), flush=True)
                            fh.write(json.dumps(rec) + "\n"); fh.flush()
                            continue
                        e = np.abs((ref2[:, None] + sh2[None, :] - 2 * C) - d64)
                        tm = np.median(np.array(times), 0)
                        rec = dict(base, cfg=name, split=k, dtype=dtype, fid=fid, kc=kc, sum=sm, nc=nc,
                                   err_max=float(e.max()), err_near=float(e[near].max()),
                                   up_s=round(tm[0], 4), mm_s=round(tm[1], 4), down_s=round(tm[2], 4),
                                   hostsum_s=round(tm[3], 4), chip_s=round(tm[0] + tm[1] + tm[2], 4),
                                   host_build_s=round(host_build, 4),
                                   up_MB=round((lhs.numel() + rhs.numel()) * (2 if dtype == "bf16" else 4) / 1e6, 1),
                                   down_MB=round(out.numel() * 4 / 1e6, 1), aiclk_min=cmin, aiclk_med=cmed)
                        print(json.dumps(rec), flush=True)
                        fh.write(json.dumps(rec) + "\n"); fh.flush()
            del lhs, rhs
ttnn.close_device(dev)
print("PARITY360-DONE", flush=True)
