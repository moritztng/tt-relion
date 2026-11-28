"""Host seconds the resident op spends building one batch's particle operand, at recorded shapes. No chip.

Resident._operands runs in the device server before anything reaches the chip, so it is chip idle time. This times it
on the first --n coarse and fine calls of a record, for the dense fine translation set and the asked-for one.

    python perf/rlx/bridge/operand_cost.py rec/g16/rec.123.pkl --n 8
"""
import argparse
import json
import time

import numpy as np

from tt_relion import record, resident       # imports ttnn, opens no device

ap = argparse.ArgumentParser()
ap.add_argument("rec")
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()
models, euls, calls = record.load(a.rec, 40 * a.n)
f32 = lambda b: np.frombuffer(b, np.float32)  # noqa: E731
for kind, compact in (("C", False), ("F", False), ("F", True)):
    c = [r for r in calls if r["k"] == kind][:a.n]
    tx, ty = [f32(r["mv"][0]) for r in c], [f32(r["mv"][1]) for r in c]
    if compact:
        keep = [np.unique(np.frombuffer(r["mv"][6], np.uint64)) for r in c]
        tx, ty = [v[k] for v, k in zip(tx, keep)], [v[k] for v, k in zip(ty, keep)]
    T = max(len(v) for v in tx)
    pad = lambda vs: np.stack([np.pad(v, (0, T - len(v))) for v in vs])  # noqa: E731
    xr, xi, w = (np.stack([f32(r["mv"][j]) for r in c]) for j in (2, 3, 4))
    P = xr.shape[1]
    imgX = c[0]["sc"][8]
    pix = np.arange(P)
    x, y = (pix % imgX).astype(np.float32), (pix // imgX).astype(np.float32)
    R = resident.Resident.__new__(resident.Resident)
    R.P, R.NC = P, -(-P // 1024)
    R.K = R.NC * 2048
    t0 = time.time()
    A, A2, nx = R._operands(xr, xi, w * 0.5, pad(tx), pad(ty), x, y)
    t1 = time.time()
    e = np.zeros(R.K, int)
    A, A2 = np.ldexp(A, e), np.ldexp(A2, e)
    ea, ea2 = resident._exp(np.abs(A).max(1)), resident._exp(np.abs(A2).max(1))
    A32, A232 = np.ascontiguousarray(A, np.float32), np.ascontiguousarray(A2, np.float32)
    t2 = time.time()
    rows = len(c) * T
    print(json.dumps(dict(kind=kind, asked_only=compact, N=len(c), T=T, P=P, rows=rows, rows_padded=-(-rows // 512) * 512,
                          K=R.K, O=[r["sc"][-3] if kind == "C" else r["sc"][-5] for r in c][:3],
                          operands_s=round(t1 - t0, 3), scale_cast_s=round(t2 - t1, 3),
                          upload_mb=round((A32.nbytes + A232.nbytes) / 2**20, 1))))
