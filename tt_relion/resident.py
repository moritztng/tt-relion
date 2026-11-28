"""diff2 on the chip from raw operands: the device-resident form of RELION's coarse and fine kernels.

RELION's accelerated kernels compute, per particle, d2(o, t) = sum_p w_p |P_o(p) - S_t(p) X(p)|^2 with
w = corr / 2, P_o the reference projected at orientation o, and S_t a translation phase. Expanded,
    d2 = sum w |X|^2  +  sum w |P_o|^2  -  2 sum w Re(conj(P_o) S_t X),
the last two are matrix products with the projections [O, K] on one side for every particle of a batch. Here
the projections are made on the chip from the model (proj_rot), both products run as exact Ozaki digit
products (oz_cut, oz_fused, oz_combine) and the three terms are combined on the chip in double-fp32. The host
sends images, weights, translations and Euler matrices and reads back d2.

One instance per chip, used by tt_relion.device_server. The steps are the ones perf/rlx/resident/global_pass.py
measured against float64 on real particles; that script stays the precision reference.
"""
import time

import numpy as np
import torch
import ttnn

from .oz_fused import oz_combine, oz_cut, oz_fused
from .proj_lerp import proj_rot

KC = 480        # K run of the digit product (oz_fused KR = 15 tiles)
RBLK = 512      # oz_fused row block
OBLK = 256      # oz_fused column block
OCH = 4096      # orientations per product
RUP = 64        # uploaded rows come in whole steps of this: a fine pass has a new row count on every call, and
                # each new shape costs seconds of program builds
FAR = 3000.0    # pixel coordinate of a pad pixel: outside every radius, so it projects to zero


def _exp(m):
    """Exponent e with |x| <= 2^e for the maxima m (0 for an all-zero row)."""
    return np.where(m > 0, np.floor(np.log2(np.maximum(m, 1e-300))) + 1, 0).astype(np.int32)


class Resident:
    def __init__(self, dev, digits=13):
        self.dev, self.n = dev, digits
        self.tab = None
        self.pixsets = {}       # key -> (pixel coordinates on the chip, P, NC)
        self.model_key = self.ref_key = None
        self.refs = []          # per orientation chunk: dict(O, B, B2) before the first batch, digits after
        self.bal = None         # (e, e2): the K-diagonal balance exponents of the held reference set
        # where a batch's wall goes: operand build on the host, then the chip (dev_s, split into its phases with
        # the device synchronised at each mark), and how full the row tile was (rows used of rows computed)
        self.c = dict(host_s=0.0, dev_s=0.0, proj_s=0.0, refdig_s=0.0, imgdig_s=0.0, prod_s=0.0, read_s=0.0,
                      batches=0, rows=0, rows_padded=0)

    def _mark(self, name, t0):
        ttnn.synchronize_device(self.dev)
        t1 = time.time()
        self.c[name] += t1 - t0
        return t1

    # ---- uploads -------------------------------------------------------------------------------------------
    def _put(self, x, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), dtype=ttnn.float32,
                               layout=layout, device=self.dev)

    def _ti(self, e, m):
        e = np.pad(np.asarray(e, np.int32), (0, m - len(e))).reshape(-1, 32)
        return ttnn.from_torch(torch.from_numpy(e), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)

    def set_model(self, key, mdl, X, Y, Z, initY, initZ, maxR2, pf):
        """mdl float32 [Z*Y*X, 2], RELION's padded half volume. Held until the key changes (once per iteration)."""
        if key == self.model_key:
            return
        assert maxR2 < FAR * FAR
        self._drop_refs()
        for pix, _, _ in self.pixsets.values():
            ttnn.deallocate(pix)
        self.pixsets = {}
        if self.tab is not None:
            ttnn.deallocate(self.tab)
        nvox = mdl.shape[0]
        mp = np.concatenate([mdl, np.zeros((X * Y + X + 2, 2), np.float32)])
        tb = np.empty((nvox, 16), np.float32)          # the 8 trilinear corners of every base voxel, one row
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    o = dx + dy * X + dz * X * Y
                    c = 4 * dz + 2 * dy + dx
                    tb[:, 2 * c:2 * c + 2] = mp[o:o + nvox]
        self.tab = self._put(tb, ttnn.ROW_MAJOR_LAYOUT)
        self.geom = dict(X=X, Y=Y, Z=Z, initY=initY, initZ=initZ, maxR2=maxR2)
        self.pf, self.model_key = pf, key

    def set_radius(self, maxR2):
        """The projection's radius cut for the calls that follow. It belongs to the pass (the coarse pass cuts the same
        model at a smaller radius than the fine one), so it is set per call; a held set keeps the cut it was projected
        with, which is why the caller's ref_key carries the radius."""
        assert maxR2 < FAR * FAR
        if maxR2 != self.geom["maxR2"]:
            self.geom = dict(self.geom, maxR2=maxR2)

    def set_pixels(self, key, x, y):
        """x, y float32 [P]: the image's Fourier pixel coordinates, in RELION's pixel order. Coarse and fine calls
        alternate between two image sizes, so every set of the current model stays on the chip."""
        if key not in self.pixsets:
            P = len(x)
            NC = -(-P // 1024)
            pxy = np.full((2, NC * 1024), FAR, np.float32)
            pxy[0, :P], pxy[1, :P] = x, y
            pix = self._put(pxy.reshape(2, NC, 1024).transpose(1, 0, 2).reshape(2 * NC, 1024), ttnn.ROW_MAJOR_LAYOUT)
            self.pixsets[key] = (pix, P, NC)
        self.pix, self.P, self.NC = self.pixsets[key]
        self.K = self.NC * 2048
        self.Kp = -(-self.K // KC) * KC

    def _drop_refs(self):
        for r in self.refs:
            for v in r.values():
                if isinstance(v, ttnn.Tensor):
                    ttnn.deallocate(v)
        self.refs, self.bal, self.ref_key = [], None, None

    # ---- reference side ------------------------------------------------------------------------------------
    def _kchip(self, re, im):
        """[n, P] real and imaginary parts -> [n, K] in the chip's K order: per 1024-pixel chunk, re then im."""
        n = re.shape[0]
        z = np.zeros((2, n, self.NC * 1024), np.float32)
        z[0, :, :self.P], z[1, :, :self.P] = re, im
        return z.reshape(2, n, self.NC, 1024).transpose(1, 2, 0, 3).reshape(n, self.K)

    def _colmax(self, t):
        return ttnn.to_torch(ttnn.max(ttnn.abs(t), dim=0)).double().numpy().reshape(-1)[:self.K]

    def _rowmax(self, t):
        return ttnn.to_torch(ttnn.max(ttnn.abs(t), dim=-1)).double().numpy().reshape(-1)[:int(t.shape[0])]

    def _project(self, eul):
        """Euler matrices [O, 9] -> per chunk the projections B [O, K] and B^2, fp32 on the chip."""
        out = []
        for o0 in range(0, len(eul), OCH):
            e = eul[o0:o0 + OCH]
            Op = -(-len(e) // OBLK) * OBLK      # whole column blocks: every set size of a block count is one program
            rot = np.zeros((Op, 8), np.float32)
            rot[:len(e), :6] = e[:, [0, 1, 3, 4, 6, 7]] * self.pf
            rot[len(e):] = rot[0]               # pad with a real orientation, so the maxima are the set's own
            t_rot = self._put(rot, ttnn.ROW_MAJOR_LAYOUT)
            pr = proj_rot(self.tab, self.pix, t_rot, self.geom)
            B = ttnn.to_layout(ttnn.reshape(pr, (Op, self.K)), ttnn.TILE_LAYOUT)
            ttnn.deallocate(t_rot)
            out.append(dict(O=len(e), B=B, B2=ttnn.multiply(B, B)))
        return out

    def _digits(self, X, e, rows):
        pr, pk = rows - int(X.shape[0]), self.Kp - int(X.shape[1])
        Xp = ttnn.pad(X, [(0, pr), (0, pk)], 0.0) if pr or pk else X
        d = oz_cut(Xp, self._ti(e, rows), self.n)
        if Xp is not X:
            ttnn.deallocate(Xp)
        return d

    def _balance(self, refs, ma, ma2):
        """Exact power-of-two rebalance along K, A diag(d) and diag(1/d) B: the weights peak at high frequency and
        the reference at low, so without it the digits are spent on the mismatch. Then the reference digits."""
        def ex(mb, m):
            ok = (m > 0) & (mb > 0)
            return np.where(ok, np.round(0.5 * (np.log2(np.where(ok, mb, 1)) - np.log2(np.where(ok, m, 1)))), 0).astype(int)
        e = ex(np.max([self._colmax(r["B"]) for r in refs], 0), ma)
        e2 = ex(np.max([self._colmax(r["B2"]) for r in refs], 0), ma2)
        for r in refs:
            Op = -(-r["O"] // OBLK) * OBLK
            for name, ee in (("B", e), ("B2", e2)):
                s = self._put(np.ldexp(1.0, -ee).reshape(1, -1))
                Xb = ttnn.multiply(r[name], s)
                eb = _exp(self._rowmax(Xb))
                r[name + "d"], r[name + "e"] = self._digits(Xb, eb, Op), eb
                for t in (s, Xb, r[name]):
                    ttnn.deallocate(t)
                del r[name]
        return e, e2

    # ---- one batch -----------------------------------------------------------------------------------------
    def _product(self, Ad, ea, rows, r, name):
        Op = -(-r["O"] // OBLK) * OBLK
        S = oz_fused(Ad, r[name + "d"], self.n, bt=True)
        eb = r[name + "e"]
        assert int(ea.min()) + int(eb.min()) - 2 * self.n > -126 and int(ea.max()) + int(eb.max()) < 127
        hi, lo = oz_combine(S, self._ti(ea, rows), self._ti(eb, Op), self.n)
        ttnn.deallocate(S)
        return hi, lo

    def _operands(self, xr, xi, w, tx, ty, x, y):
        """Particle side in float64: A [N T, K] = w S_t X (rows n T + t), A2 = w on both parts, nx = sum w |X|^2."""
        T = tx.shape[1]
        w = w.astype(np.float64)
        ph = tx[:, :, None].astype(np.float64) * x + ty[:, :, None].astype(np.float64) * y      # [N, T, P]
        c, s = np.cos(ph), np.sin(ph)
        ar = (c * xr[:, None] - s * xi[:, None]) * w[:, None]
        ai = (c * xi[:, None] + s * xr[:, None]) * w[:, None]
        A = self._kchip(ar.reshape(-1, self.P), ai.reshape(-1, self.P)).astype(np.float64)
        A2 = self._kchip(np.repeat(w, T, 0), np.repeat(w, T, 0)).astype(np.float64)
        nx = np.repeat((w * (xr.astype(np.float64) ** 2 + xi.astype(np.float64) ** 2)).sum(1), T)
        return A, A2, nx

    def d2(self, eul, xr, xi, w, tx, ty, x, y, ref_key=None):
        """float32 [N, O, T]. eul [O, 9]; xr, xi, w [N, P]; tx, ty [N, T] (phase = x tx + y ty, RELION's units).

        ref_key names an orientation set shared by many batches (a global pass): its projections and digits stay on
        the chip until another key or model arrives, so the key must also identify the pixel set. Without it the set
        is used once.
        """
        N, T = tx.shape
        R = N * T
        rows = -(-R // RBLK) * RBLK
        t0 = time.time()
        A, A2, nx = self._operands(xr, xi, w, tx, ty, x, y)
        t0 = t1 = self._mark("host_s", t0)

        held = ref_key is not None and ref_key == self.ref_key
        if held:
            refs, (e, e2) = self.refs, self.bal
        else:
            refs = self._project(np.asarray(eul, np.float32).reshape(-1, 9))
            t0 = self._mark("proj_s", t0)
            e, e2 = self._balance(refs, np.abs(A).max(0), np.abs(A2).max(0))
            t0 = self._mark("refdig_s", t0)
            if ref_key is not None:
                self._drop_refs()
                self.refs, self.bal, self.ref_key = refs, (e, e2), ref_key
        A, A2 = np.ldexp(A, e), np.ldexp(A2, e2)
        ea, ea2 = _exp(np.abs(A).max(1)), _exp(np.abs(A2).max(1))
        up = lambda v: np.pad(v, ((0, -R % RUP), (0, 0)))  # noqa: E731
        tA, tA2 = self._put(up(A)), self._put(up(A2))
        Ad, A2d = self._digits(tA, ea, rows), self._digits(tA2, ea2, rows)
        nxh = (nx / 2).astype(np.float32)
        nxl = (nx / 2 - nxh).astype(np.float32)
        pad = lambda v: np.pad(v, (0, rows - R))[:, None]  # noqa: E731
        t_nxh, t_nxl = self._put(pad(nxh)), self._put(pad(nxl))
        t0 = self._mark("imgdig_s", t0)

        def tsum(a, b):     # TwoSum in true-fp32 eltwise ops: a + b = s + err exactly
            s_ = ttnn.add(a, b)
            bp = ttnn.subtract(s_, a)
            return s_, ttnn.add(ttnn.subtract(a, ttnn.subtract(s_, bp)), ttnn.subtract(b, bp))

        out = np.empty((R, len(eul)), np.float32)
        o0 = 0
        for r in refs:
            Ch, Cl = self._product(Ad, ea, rows, r, "B")
            Nh, Nl = self._product(A2d, ea2, rows, r, "B2")
            # d2 / 2 = (sum w|X|^2 + sum w|P|^2) / 2 - cross, rounded once at the end
            sh, se = tsum(ttnn.multiply(Nh, 0.5), ttnn.add(ttnn.multiply(Nh, 0.0), t_nxh))
            lo = ttnn.add(ttnn.add(se, ttnn.multiply(Nl, 0.5)), t_nxl)
            dh, de = tsum(sh, ttnn.neg(Ch))
            D = ttnn.add(dh, ttnn.subtract(ttnn.add(de, lo), Cl))
            t0 = self._mark("prod_s", t0)
            out[:, o0:o0 + r["O"]] = ttnn.to_torch(D).numpy()[:R, :r["O"]]
            t0 = self._mark("read_s", t0)
            o0 += r["O"]
            for t in (Ch, Cl, Nh, Nl, D):
                ttnn.deallocate(t)
        for t in (tA, tA2, Ad, A2d, t_nxh, t_nxl):
            ttnn.deallocate(t)
        if not held and ref_key is None:
            for r in refs:
                for v in r.values():
                    if isinstance(v, ttnn.Tensor):
                        ttnn.deallocate(v)
        c = self.c
        c["dev_s"] += time.time() - t1
        c["batches"], c["rows"], c["rows_padded"] = c["batches"] + 1, c["rows"] + R, c["rows_padded"] + rows
        return (2.0 * out).reshape(N, T, -1).transpose(0, 2, 1)
