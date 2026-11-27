"""Exact-integer (Ozaki) diff2 cross term on Blackhole: accuracy vs float64 and warm s/call at the dev cell sizes.

probe_int.py: a BH matmul keeps only ~11 significant bits in its accumulation, in every config (fp32 dest, packer
L1 acc on or off, fp32 or bf16 operands), but it is EXACT while every partial sum stays below 2^11. So:
  1. each operand row is scaled by a power of two to |r| < 1 (shared exponent per row; exact),
  2. cut into n balanced base-4 digits q_i in [-2, 2], r = sum_i q_i 2^-(2i+1) + O(2^-(2n+1)) (exact fp32 SFPU ops
     on the chip: round-to-nearest by the 1.5*2^23 trick),
  3. every digit pair (i, j) with i + j < n is one LoFi matmul over K runs of KC = 480: |partial| <= 480*4 < 2^11,
     so every product and partial sum is exact,
  4. runs are added on the chip in fp32 (ttnn.add, exact: integers < 2^24), pairs of equal i + j first; the n
     per-(i+j) integer sums [O, T] come back and the host combines them (2^-(2s+2), row exponents) in float64.
     Combining on the chip in fp32 floors the error at the fp32 ulp of C (f146: 0.0076 near the minimum).
Graded like diff2_parity360.py: d2 = ref2 + sh2 - 2C with the norms in float64, error vs float64 beside RELION's own
fp32 direct form. Operands are passed to the chip already in [runs, rows, KC] layout (the bridge can write them so).
    TT_VISIBLE_DEVICES=<chip> python3 ozaki_proto.py --out res.jsonl [--cases f146,f360] [--ns 8,9,10]
"""
import argparse
import json
import time

import numpy as np
import torch

CASES = {"box64": (64, 1024, 21), "c100": (100, 1024, 37), "f146": (146, 256, 84), "f360": (360, 256, 84)}
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cases", default="f146,f360")
ap.add_argument("--ns", default="8,9,10")
ap.add_argument("--kc", type=int, default=480)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
assert a.kc % 32 == 0 and a.kc * 4 < 2048

import ttnn  # noqa: E402

from _tt import Clock, open_dev  # noqa: E402

torch.set_num_threads(16)
dev, nodes = open_dev()
MAGIC = 12582912.0  # 1.5 * 2^23: (x + M) - M rounds |x| < 2^22 to the nearest integer in fp32
CFG = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)


def operands(s, O, T, g):  # same construction as diff2_parity360.py
    imgX = s // 2 + 1
    P = imgX * s
    pix = np.arange(P)
    x = pix % imgX
    yi = pix // imgX
    y = np.where(yi > s // 2, yi - s, yi)
    inside = (x * x + y * y) <= (s // 2) ** 2
    ref = g.standard_normal((O, 2 * P)) * 3 * np.tile(inside, 2)
    sh = ref[np.arange(T) % O] + g.standard_normal((T, 2 * P))
    return ref.astype(np.float32), sh.astype(np.float32), 0.5


def row_exp(x):
    """Power-of-two exponent per row so that |x / 2^e| < 1."""
    m = np.abs(x).max(1).astype(np.float64)
    return np.where(m > 0, np.floor(np.log2(np.maximum(m, 1e-300))) + 1, 0).astype(np.int32)


def runs(x, kc):
    """[R, K] -> [nc, R, kc] (zero-padded K)."""
    R, K = x.shape
    nc = -(-K // kc)
    x = np.pad(x, ((0, 0), (0, nc * kc - K)))
    return np.ascontiguousarray(x.reshape(R, nc, kc).transpose(1, 0, 2))


def digits(X, n):
    """X: [..] fp32 on chip, |X| < 1 -> n bf16 tensors of balanced base-4 digits."""
    out, r = [], X
    for i in range(n):
        t = ttnn.multiply(r, float(2 ** (2 * i + 1)))
        q = ttnn.subtract(ttnn.add(t, MAGIC), MAGIC)
        out.append(ttnn.typecast(q, ttnn.bfloat16))
        r2 = ttnn.subtract(r, ttnn.multiply(q, float(2.0 ** -(2 * i + 1))))
        for z in (t, q):
            ttnn.deallocate(z)
        if r is not X:
            ttnn.deallocate(r)
        r = r2
    ttnn.deallocate(r)
    return out


def tree_sum(C):
    """fp32 [nc, O, T] -> [1, O, T], pairwise ttnn.add (SFPU fp32, exact for the integer sums here)."""
    while C.shape[0] > 1:
        n, h = C.shape[0], C.shape[0] // 2
        lo = ttnn.slice(C, [0, 0, 0], [h, C.shape[1], C.shape[2]])
        hi = ttnn.slice(C, [h, 0, 0], [2 * h, C.shape[1], C.shape[2]])
        S = ttnn.add(lo, hi)
        if n % 2:
            S = ttnn.concat([S, ttnn.slice(C, [2 * h, 0, 0], [n, C.shape[1], C.shape[2]])], 0)
        for z in (lo, hi, C):
            ttnn.deallocate(z)
        C = S
    return C


def cross(Ad, Bd, n):
    """Digit tensors -> n integer tensors [1, O, T] (sum over i + j = s), exact; the host adds them in float64."""
    out = []
    for s in range(n):
        acc = None
        for i in range(s + 1):
            C = ttnn.matmul(Ad[i], Bd[s - i], compute_kernel_config=CFG, dtype=ttnn.float32)
            if acc is None:
                acc = C
            else:
                acc2 = ttnn.add(acc, C)
                ttnn.deallocate(acc), ttnn.deallocate(C)
                acc = acc2
        out.append(tree_sum(acc))
    return out


fh = open(a.out, "a")


def emit(rec):
    print(json.dumps(rec), flush=True)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


for case in a.cases.split(","):
    s, O, T = CASES[case]
    ref, sh, w = operands(s, O, T, np.random.default_rng(0))
    shw = (sh * np.float32(w)).astype(np.float32)
    r64, sh64 = ref.astype(np.float64), sh.astype(np.float64)
    C64 = r64 @ shw.astype(np.float64).T
    ref2, sh2 = (r64 ** 2).sum(1) * w, (sh64 ** 2).sum(1) * w
    d64 = ref2[:, None] + sh2[None, :] - 2 * C64
    near = (d64 - d64.min()) < 30
    rt, st = torch.from_numpy(ref), torch.from_numpy(sh)
    d32 = torch.empty(O, T, dtype=torch.float32)
    for o0 in range(0, O, 8):
        d32[o0:o0 + 8] = (((rt[o0:o0 + 8, None] - st[None]) ** 2) * w).sum(-1)
    e32 = np.abs(d32.double().numpy() - d64)
    base = dict(case=case, s=s, O=O, T=T, K=ref.shape[1], kc=a.kc)
    emit(dict(base, cfg="fp32 direct (host)", err_max=float(e32.max()), err_near=float(e32[near].max())))
    ea, eb = row_exp(ref), row_exp(shw)
    An = runs(np.ldexp(ref.astype(np.float64), -ea[:, None]).astype(np.float32), a.kc)   # exact scaling
    Bn = runs(np.ldexp(shw.astype(np.float64), -eb[:, None]).astype(np.float32), a.kc).transpose(0, 2, 1).copy()
    nc = An.shape[0]
    for n in map(int, a.ns.split(",")):
        clk = Clock(nodes)
        tim = []
        for rep in range(a.reps + 1):
            t0 = time.time()
            A = ttnn.from_torch(torch.from_numpy(An), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            B = ttnn.from_torch(torch.from_numpy(Bn), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            ttnn.synchronize_device(dev)
            t1 = time.time()
            Ad, Bd = digits(A, n), digits(B, n)
            ttnn.synchronize_device(dev)
            t2 = time.time()
            Cs = cross(Ad, Bd, n)
            ttnn.synchronize_device(dev)
            t3 = time.time()
            C = sum(ttnn.to_torch(x).double().numpy().reshape(O, T) * 2.0 ** -(2 * k + 2) for k, x in enumerate(Cs))
            t4 = time.time()
            if rep == 0:   # the on-chip digits must be the exact host digits
                ok = True
                r = np.ldexp(ref.astype(np.float64), -ea[:, None])
                for i in range(min(n, 3)):
                    q = np.round(r * 2.0 ** (2 * i + 1))   # numpy rounds half to even, as the magic add does
                    ok &= np.array_equal(runs(q, a.kc), ttnn.to_torch(Ad[i]).double().numpy())
                    r = r - q * 2.0 ** -(2 * i + 1)
            for z in Ad + Bd + [A, B] + Cs:
                ttnn.deallocate(z)
            if rep:
                tim.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        cmin, cmed = clk.stop()
        Cf = C * np.ldexp(1.0, ea)[:, None] * np.ldexp(1.0, eb)[None, :]
        e = np.abs(ref2[:, None] + sh2[None, :] - 2 * Cf - d64)
        tm = np.median(np.array(tim), 0)
        emit(dict(base, cfg="ozaki n%d" % n, n=n, matmuls=n * (n + 1) // 2, runs=nc, digits_exact=bool(ok),
                  err_max=float(e.max()), err_near=float(e[near].max()), up_s=round(tm[0], 4),
                  digits_s=round(tm[1], 4), cross_s=round(tm[2], 4), down_s=round(tm[3], 4),
                  chip_s=round(tm[1] + tm[2], 4), aiclk_min=cmin, aiclk_med=cmed))
ttnn.close_device(dev)
print("OZAKI-DONE", flush=True)
