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

f32 = lambda b: t.from_numpy(np.frombuffer(b, dtype=np.float32).copy())  # noqa: E731


class Call:
    """One recorded call in float64: w [2P] (the weight on both parts), sh [T, 2P] the shifted image (real parts, then
    imaginary), refs(o0, o1) [o1 - o0, 2P] the projections. d2[o, t] = sum w (ref[o] - sh[t])^2."""

    def __init__(self, models, euls, r):
        self.fine = r["k"] == "F"
        tx, ty, re, im, corr = (f32(b) for b in r["mv"][:5])
        sc = r["sc"]
        self.geom, self.maxR, self.imgX, self.imgY = sc[:5], sc[5], sc[8], sc[9]
        self.maxR2, self.pf = sc[6], sc[7]
        self.O, P = (sc[11], sc[14]) if self.fine else (sc[10], sc[12])
        self.mdl, self.eul = f32(models[r["mdl"]]).reshape(-1, 2), f32(euls[r["eul"]]).reshape(self.O, 9)
        if self.fine:       # the asked samples: (orientation, index into the distinct translations) and the constant
            self.rot, tr = (np.frombuffer(b, dtype=np.uint64) for b in r["mv"][5:7])
            trans, self.tr = np.unique(tr, return_inverse=True)
            tx, ty, self.sum_init = tx[trans], ty[trans], sc[10]
        pix = t.arange(P)
        self.x, yi = (pix % self.imgX).float(), pix // self.imgX
        self.y = t.where(yi > self.maxR, yi - self.imgY, yi).float()
        ph = self.x.double() * tx.double()[:, None] + self.y.double() * ty.double()[:, None]
        s, c = t.sin(ph), t.cos(ph)
        self.w = t.cat([corr, corr]).double() * 0.5
        self.sh = t.cat([c * re - s * im, c * im + s * re], 1)

    def refs(self, o0, o1):
        e, x, y = self.eul[o0:o1], self.x, self.y
        xp, yp, zp = ((e[:, i:i + 1] * x + e[:, i + 1:i + 2] * y) * self.pf for i in (0, 3, 6))
        inside = (xp * xp + yp * yp + zp * zp).trunc().to(t.int64) <= self.maxR2    # float32, as RELION decides it
        return t.cat(_project(t, self.mdl.double(), x.double(), y.double(), e.double(), *self.geom, self.maxR2, self.pf,
                              inside), 1)

    def d2(self):
        """The call's result as the bridge returns it: [O * T] (coarse) or one value per asked sample (fine)."""
        sh2 = (self.sh * self.sh * self.w).sum(1)
        out = t.empty(self.O, len(self.sh), dtype=t.float64)
        step = max(1, 2_000_000 // len(self.x))
        for o in range(0, self.O, step):
            ref = self.refs(o, o + step)
            out[o:o + step] = (ref * ref * self.w).sum(1)[:, None] + sh2 - 2.0 * (ref * self.w) @ self.sh.T
        out = out.numpy()
        return out[self.rot.astype(np.intp), self.tr] + self.sum_init if self.fine else out.reshape(-1)


def _one(r):
    t.set_num_threads(1)
    return r["i"], Call(models, euls, r).d2()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    models, euls, calls = record.load(a.rec, a.limit)
    for i, r in enumerate(calls):
        r["i"] = i
    dg = record.Digests(len(calls))
    with mp.Pool(a.procs) as pool:
        for i, v in pool.imap_unordered(_one, calls):
            dg.put(i, v)
    dg.save(a.out)
    print("calls", len(calls), "coarse", sum(r["k"] == "C" for r in calls), "->", a.out)
