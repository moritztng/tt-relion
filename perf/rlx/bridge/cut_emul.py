"""Where the resident op's d2 error on real data comes from: the digit cut of tt_relion/resident.py emulated on the CPU.

For recorded calls (tt_relion/record.py) the two products of d2 = sum w|X|^2 + N - 2 C are formed as the chip forms
them: float32 operands, a power-of-two balance along K from the column maxima, a row exponent, base-4 digits by
rounding (oz_cut) and the digit pairs i + j < n (oz_fused). The error of the cross term C = A B^T and of the norm
N = w (B^2)^T against the exact products of the same float32 operands is printed per digit count, at the call's
best sample. --dead zeroes the image columns where every reference of the set is zero (outside the projection's
radius): they add nothing to either product, but unbalanced they can set the row maximum.

    python perf/rlx/bridge/cut_emul.py --rec rec/rec.1234.pkl --calls 80:84 [--dead] [--omax 256]
"""
import argparse
import json

import numpy as np

from ref64 import Call
from tt_relion import record

ap = argparse.ArgumentParser()
ap.add_argument("--rec", required=True)
ap.add_argument("--calls", default="0:4", help="first:last of the calls in file order")
ap.add_argument("--kind", default="CF")
ap.add_argument("--omax", type=int, default=256, help="orientations per call (the first ones)")
ap.add_argument("--digits", default="13,14,15,16,17,18,20")
ap.add_argument("--dead", action="store_true")
ap.add_argument("--out", default="")
a = ap.parse_args()
c0, c1 = (int(v) for v in a.calls.split(":"))
ns = [int(v) for v in a.digits.split(",")]
models, euls, calls = record.load(a.rec, c1)
ex = lambda m: np.where(m > 0, np.floor(np.log2(np.maximum(m, 1e-300))) + 1, 0)  # noqa: E731


def cut(X, n):
    """Row exponents and the n digit planes of X [R, K]: X 2^-e = sum_i d_i 2^-(2i + 1) + residual."""
    e = ex(np.abs(X).max(1))
    r, d = X * 2.0 ** (1 - e)[:, None], []
    for _ in range(n):
        q = np.rint(r)
        d.append(q)
        r = (r - q) * 4
    return e, d


def product(A, B, nmax):
    """[len(ns)] truncated A B^T, pairs i + j < n, and the exact product."""
    ea, da = cut(A, nmax)
    eb, db = cut(B, nmax)
    S = [sum(da[i] @ db[k - i].T for i in range(k + 1)) for k in range(nmax)]
    sc = 2.0 ** (ea[:, None] + eb[None, :])
    return [sum(S[k] * 2.0 ** -(2 * k + 2) for k in range(n)) * sc for n in ns], A @ B.T


for r in calls[c0:c1]:
    if r["k"] not in a.kind:
        continue
    c = Call(models, euls, r)
    O = min(c.O, a.omax)
    B = c.refs(0, O).numpy().astype(np.float32)
    w = c.w.numpy().astype(np.float32)
    A = (c.sh * c.w).numpy().astype(np.float32)             # [T, K]
    B2 = (B * B).astype(np.float32)                         # the chip squares in float32
    A, B, B2, w2 = A.astype(np.float64), B.astype(np.float64), B2.astype(np.float64), w.astype(np.float64)[None]
    mb, mb2 = np.abs(B).max(0), B2.max(0)
    row = dict(kind=r["k"], O=O, T=len(A), K=A.shape[1], dead_cols=int((mb == 0).sum()), dead=a.dead)
    for name, X, Y, mY in (("cross", A, B, mb), ("norm", w2, B2, mb2)):
        mX = np.abs(X).max(0)
        ok = (mX > 0) & (mY > 0)
        e = np.where(ok, np.rint(0.5 * (np.log2(np.where(ok, mY, 1)) - np.log2(np.where(ok, mX, 1)))), 0)
        Xb, Yb = X * 2.0 ** e, Y * 2.0 ** -e
        row[name + "_rowmax_over_live"] = float(np.abs(Xb).max() / max(np.abs(Xb[:, mY > 0]).max(), 1e-300))
        if a.dead:
            Xb = Xb * (mY > 0)
        tr, exact = product(Xb, Yb, max(ns))
        row[name] = float(np.abs(exact).max())
        row[name + "_err"] = {n: [float((v - exact).mean()), float(np.abs(v - exact).max())] for n, v in zip(ns, tr)}
    # d2 = nx + N - 2 C: what the two errors do to it, mean and worst over the block
    row["d2_err"] = {n: [row["norm_err"][n][0] - 2 * row["cross_err"][n][0],
                         row["norm_err"][n][1] + 2 * row["cross_err"][n][1]] for n in ns}
    print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
