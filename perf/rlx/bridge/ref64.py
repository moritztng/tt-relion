"""Float64 reference digest of a recorded bridge call stream (tt_relion/record.py), for replay.py --ref.

Every recorded diff2_coarse / diff2_fine call is recomputed on the CPU in float64 from its float32 inputs: the
projection (trilinear, RELION's rule), the phase shifts and the weighted squared difference. The radius test of the
projection stays RELION's float32 decision. The CPU torch arm of the bridge is float32 like RELION itself, so it
cannot say whether a chip result that differs from it is worse or better; this can.

    python perf/rlx/bridge/ref64.py --rec rec/rec.1234.pkl --out ref64.npz [--limit 160] [--procs 16]
"""
import argparse
import multiprocessing as mp

import numpy as np
import torch as t

from tt_relion import record
from tt_relion.relion import _project

ap = argparse.ArgumentParser()
ap.add_argument("--rec", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--procs", type=int, default=8)
a = ap.parse_args()
models, euls, calls = record.load(a.rec, a.limit)
f32 = lambda b: t.from_numpy(np.frombuffer(b, dtype=np.float32).copy())  # noqa: E731


def d2(r):
    """The call's result as the bridge returns it: [O * T] (coarse) or one value per asked sample (fine)."""
    t.set_num_threads(1)
    fine = r["k"] == "F"
    tx, ty, re, im, corr = (f32(b) for b in r["mv"][:5])
    sc = r["sc"]
    mdlX, mdlY, mdlZ, initY, initZ, maxR, maxR2, pf, imgX, imgY = sc[:10]
    O, P = (sc[11], sc[14]) if fine else (sc[10], sc[12])
    mdl, eul = f32(models[r["mdl"]]).reshape(-1, 2), f32(euls[r["eul"]]).reshape(O, 9)
    if fine:
        rot, tr = (np.frombuffer(b, dtype=np.uint64) for b in r["mv"][5:7])
        trans, tr = np.unique(tr, return_inverse=True)
        tx, ty = tx[trans], ty[trans]
    pix = t.arange(P)
    x, yi = (pix % imgX).float(), pix // imgX
    y = t.where(yi > maxR, yi - imgY, yi).float()
    ph = x.double() * tx.double()[:, None] + y.double() * ty.double()[:, None]
    s, c = t.sin(ph), t.cos(ph)
    w = t.cat([corr, corr]).double() * 0.5
    sh = t.cat([c * re - s * im, c * im + s * re], 1)                       # [T, 2P]
    sh2 = (sh * sh * w).sum(1)
    out = t.empty(O, len(tx), dtype=t.float64)
    step = max(1, 2_000_000 // P)
    for o in range(0, O, step):
        e = eul[o:o + step]
        xp, yp, zp = ((e[:, i:i + 1] * x + e[:, i + 1:i + 2] * y) * pf for i in (0, 3, 6))
        inside = (xp * xp + yp * yp + zp * zp).trunc().to(t.int64) <= maxR2     # float32, as RELION decides it
        ref = t.cat(_project(t, mdl.double(), x.double(), y.double(), e.double(), mdlX, mdlY, mdlZ, initY, initZ,
                             maxR2, pf, inside), 1)
        out[o:o + step] = (ref * ref * w).sum(1)[:, None] + sh2 - 2.0 * (ref * w) @ sh.T
    out = out.numpy()
    return r["i"], out[rot.astype(np.intp), tr] + sc[10] if fine else out.reshape(-1)


if __name__ == "__main__":
    for i, r in enumerate(calls):
        r["i"] = i
    dg = record.Digests(len(calls))
    with mp.Pool(a.procs) as pool:
        for i, v in pool.imap_unordered(d2, calls):
            dg.put(i, v)
    dg.save(a.out)
    print("calls", len(calls), "coarse", sum(r["k"] == "C" for r in calls), "->", a.out)
