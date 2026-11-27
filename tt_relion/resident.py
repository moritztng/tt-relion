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
import numpy as np
import torch
import ttnn

from .oz_fused import oz_combine, oz_cut, oz_fused
from .proj_lerp import proj_rot

KC = 480        # K run of the digit product (oz_fused KR = 15 tiles)
RBLK = 512      # oz_fused row block
OBLK = 256      # oz_fused column block
OCH = 4096      # orientations per product
FAR = 3000.0    # pixel coordinate of a pad pixel: outside every radius, so it projects to zero


def _exp(m):
    """Exponent e with |x| <= 2^e for the maxima m (0 for an all-zero row)."""
    return np.where(m > 0, np.floor(np.log2(np.maximum(m, 1e-300))) + 1, 0).astype(np.int32)


class Resident:
    def __init__(self, dev, digits=13):
        self.dev, self.n = dev, digits
        self.tab = self.pix = None
        self.model_key = self.pix_key = self.ref_key = None
        self.refs = []          # per orientation chunk: dict(O, B, B2) before the first batch, digits after
        self.bal = None         # (e, e2): the K-diagonal balance exponents of the held reference set

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

    def set_pixels(self, key, x, y):
        """x, y float32 [P]: the image's Fourier pixel coordinates, in RELION's pixel order."""
        if key == self.pix_key:
            return
        self._drop_refs()
        if self.pix is not None:
            ttnn.deallocate(self.pix)
        self.P = P = len(x)
        self.NC = NC = -(-P // 1024)
        self.K = NC * 2048
        self.Kp = -(-self.K // KC) * KC
        pxy = np.full((2, NC * 1024), FAR, np.float32)
        pxy[0, :P], pxy[1, :P] = x, y
        self.pix = self._put(pxy.reshape(2, NC, 1024).transpose(1, 0, 2).reshape(2 * NC, 1024), ttnn.ROW_MAJOR_LAYOUT)
        self.pix_key = key

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
            rot = np.zeros((len(e), 8), np.float32)
            rot[:, :6] = e[:, [0, 1, 3, 4, 6, 7]] * self.pf
            t_rot = self._put(rot, ttnn.ROW_MAJOR_LAYOUT)
            pr = proj_rot(self.tab, self.pix, t_rot, self.geom)
            B = ttnn.to_layout(ttnn.reshape(pr, (len(e), self.K)), ttnn.TILE_LAYOUT)
            ttnn.deallocate(t_rot)
            out.append(dict(O=len(e), B=B, B2=ttnn.multiply(B, B)))
        return out

    def _digits(self, X, e, rows):
        Xp = ttnn.pad(X, [(0, rows - int(X.shape[0])), (0, self.Kp - self.K)], 0.0)
        d = oz_cut(Xp, self._ti(e, rows), self.n)
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
        the chip until the key, the model or the pixel set changes. Without it the set is used once.
        """
        N, T = tx.shape
        R = N * T
        rows = -(-R // RBLK) * RBLK
        A, A2, nx = self._operands(xr, xi, w, tx, ty, x, y)

        held = ref_key is not None and ref_key == self.ref_key
        if held:
            refs, (e, e2) = self.refs, self.bal
        else:
            refs = self._project(np.asarray(eul, np.float32).reshape(-1, 9))
            e, e2 = self._balance(refs, np.abs(A).max(0), np.abs(A2).max(0))
            if ref_key is not None:
                self._drop_refs()
                self.refs, self.bal, self.ref_key = refs, (e, e2), ref_key
        A, A2 = np.ldexp(A, e), np.ldexp(A2, e2)
        ea, ea2 = _exp(np.abs(A).max(1)), _exp(np.abs(A2).max(1))
        tA, tA2 = self._put(A), self._put(A2)
        Ad, A2d = self._digits(tA, ea, rows), self._digits(tA2, ea2, rows)
        nxh = (nx / 2).astype(np.float32)
        nxl = (nx / 2 - nxh).astype(np.float32)
        pad = lambda v: np.pad(v, (0, rows - R))[:, None]  # noqa: E731
        t_nxh, t_nxl = self._put(pad(nxh)), self._put(pad(nxl))

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
            out[:, o0:o0 + r["O"]] = ttnn.to_torch(D).numpy()[:R, :r["O"]]
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
        return (2.0 * out).reshape(N, T, -1).transpose(0, 2, 1)
