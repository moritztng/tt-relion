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
import contextlib
import fcntl
import os
import tempfile
import time

import numpy as np
import torch
import ttnn

from .oz_fused import cand_compact, oz_combine, oz_cut, oz_fused
from .proj_lerp import proj_rot

KC = 480        # K run of the digit product (oz_fused KR = 15 tiles)
RBLK = 512      # oz_fused row block
OBLK = 256      # oz_fused column block
OCH = max(OBLK, int(os.environ.get("TT_RELION_OCH", "4096")) // OBLK * OBLK)     # orientations per product. Each chunk is
                # its own run of programs, so a small batch (one 512-row tile) may be cheaper on fewer, wider chunks
RUP = 64        # uploaded rows come in whole steps of this: a fine pass has a new row count on every call, and
                # each new shape costs seconds of program builds
TBLK = 1 << 14  # voxels per block of the table build (1 MB of table)
TURNS = int(os.environ.get("TT_RELION_TABLE_TURNS", "4"))   # chips of a host that upload a large table at once
SUB = os.environ.get("TT_RELION_SUBPHASE", "") not in ("", "0")   # split prod_s (a device sync per step: a profile, not a rate)
FAR = 3000.0    # pixel coordinate of a pad pixel: outside every radius, so it projects to zero
FARP = 1e30     # -log prior / 2 of an excluded orientation and of a pad column: never inside a window


def _amax(a, axis):
    """max |a| along an axis, without the |a| temporary."""
    return np.maximum(a.max(axis), -a.min(axis)).astype(np.float64)


def _exp(m):
    """Exponent e with |x| <= 2^e for the maxima m (0 for an all-zero row)."""
    return np.where(m > 0, np.floor(np.log2(np.maximum(m, 1e-300))) + 1, 0).astype(np.int32)


def _rep(x, k):
    """x [m, O] (whole tiles) k times along the rows, by tile copies; x is consumed. ttnn.repeat_interleave rounds
    fp32 through bf16, and one concat takes at most 50 tensors."""
    if k == 1:
        return x
    y = ttnn.concat([x] * min(k, 16), dim=0)
    if k > 16:
        y, y16 = ttnn.concat([y] * (k // 16) + [x] * (k % 16), dim=0), y
        ttnn.deallocate(y16)
    ttnn.deallocate(x)
    return y


def _corner_table(mdl, X, Y):
    """mdl float32 [nvox, 2] -> [nvox, 16]: the 8 trilinear corners of every base voxel on one row, corner
    4 dz + 2 dy + dx in columns 2c and 2c + 1, zero past the last voxel. Built a block at a time: at full image
    size the table is 12 GB, and a corner per pass over all of it is eight times the memory traffic."""
    nvox = len(mdl)
    tb = np.empty((nvox, 8, 2), np.float32)
    offs = [dx + dy * X + dz * X * Y for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    for a in range(0, nvox, TBLK):
        for c, o in enumerate(offs):
            src = mdl[a + o:a + o + TBLK]
            tb[a:a + len(src), c] = src
            tb[a + len(src):a + TBLK, c] = 0
    return tb.reshape(nvox, 16)


@contextlib.contextmanager
def _turn():
    """One of TURNS places on this host, waited for. The table of a full-size model is 12 GB and passes through host
    memory twice on its way to the chip; every chip of the host wants it in the same second of the last iteration."""
    d = os.path.join(tempfile.gettempdir(), "tt-relion-turn")
    os.makedirs(d, exist_ok=True)
    fd = None
    while fd is None:
        for i in range(TURNS):
            f = os.open(os.path.join(d, str(i)), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fd = f
                break
            except BlockingIOError:
                os.close(f)
        else:
            time.sleep(0.1)
    try:
        yield
    finally:
        os.close(fd)


class Resident:
    def __init__(self, dev, digits=13, norm_digits=None):
        # norm_digits: digits of the sum w |P|^2 product alone. Its element products are all positive, so every one
        # the digit pairs i + j < n cut off lowers the sum; it runs on one row per particle, so more digits there
        # cost a few percent of the cross product.
        self.dev, self.n, self.n2 = dev, digits, norm_digits or digits
        self.tab = None
        self.pixsets = {}       # key -> (pixel coordinates on the chip, P, NC)
        self.model_key = self.ref_key = None
        self.refs = []          # per orientation chunk: dict(O, B, B2) before the first batch, digits after
        self.bal = None         # the K-diagonal balance of the held reference set, image side, per product
        # where a batch's wall goes: operand build on the host, then the chip (dev_s, split into its phases with
        # the device synchronised at each mark), and how full the row tile was (rows used of rows computed)
        self.c = dict(host_s=0.0, dev_s=0.0, proj_s=0.0, refdig_s=0.0, imgdig_s=0.0, prod_s=0.0, read_s=0.0,
                      px_oz_s=0.0, px_comb_s=0.0, pn_oz_s=0.0, pn_comb_s=0.0, p_asm_s=0.0,
                      batches=0, rows=0, rows_padded=0)

    def _mark(self, name, t0):
        ttnn.synchronize_device(self.dev)
        t1 = time.time()
        self.c[name] += t1 - t0
        return t1

    def _sub(self, name, t0):
        return self._mark(name, t0) if SUB else t0

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
        with _turn() if len(mdl) * 64 > 1 << 30 else contextlib.nullcontext():
            self.tab = self._put(_corner_table(mdl, X, Y), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.synchronize_device(self.dev)
        v = ttnn.get_memory_view(self.dev, ttnn.BufferType.DRAM)
        self.free = v.num_banks * v.total_bytes_free_per_bank       # what the table leaves of the chip's memory
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

    def _digits(self, X, e, rows, n=None):
        pr, pk = rows - int(X.shape[0]), self.Kp - int(X.shape[1])
        Xp = ttnn.pad(X, [(0, pr), (0, pk)], 0.0) if pr or pk else X
        d = oz_cut(Xp, self._ti(e, rows), n or self.n)
        if Xp is not X:
            ttnn.deallocate(Xp)
        return d

    def _balance(self, refs, ma, ma2):
        """Exact power-of-two rebalance along K, A diag(d) and diag(1/d) B: the weights peak at high frequency and
        the reference at low, so without it the digits are spent on the mismatch. Cuts the reference digits and
        returns diag(d) for both products."""
        def ex(mb, m):
            ok = (m > 0) & (mb > 0)
            return np.where(ok, np.round(0.5 * (np.log2(np.where(ok, mb, 1)) - np.log2(np.where(ok, m, 1)))), 0).astype(int)
        mb, mb2 = (np.max([self._colmax(r[k]) for r in refs], 0) for k in ("B", "B2"))
        e, e2 = ex(mb, ma), ex(mb2, ma2)
        for r in refs:
            Op = -(-r["O"] // OBLK) * OBLK
            for name, ee in (("B", e), ("B2", e2)):
                s = self._put(np.ldexp(1.0, -ee).reshape(1, -1))
                Xb = ttnn.multiply(r[name], s)
                eb = _exp(self._rowmax(Xb))
                r[name + "d"], r[name + "e"] = self._digits(Xb, eb, Op, self.n2 if name == "B2" else self.n), eb
                for t in (s, Xb, r[name]):
                    ttnn.deallocate(t)
                del r[name]
        # the image side of the balance. A pixel no reference of the set reaches (outside the projection's radius)
        # adds nothing to either product, but left in, unbalanced, it sets the row's exponent: zero it
        return [(np.ldexp(1.0, ee) * (m > 0)).astype(np.float32) for ee, m in ((e, mb), (e2, mb2))]

    # ---- one batch -----------------------------------------------------------------------------------------
    def _product(self, Ad, ea, rows, r, name):
        Op = -(-r["O"] // OBLK) * OBLK
        n = self.n2 if name == "B2" else self.n
        ts = time.time()
        S = oz_fused(Ad, r[name + "d"], n, rb=min(rows, RBLK) // 32, bt=True)
        ts = self._sub("px_oz_s" if name == "B" else "pn_oz_s", ts)
        eb = r[name + "e"]
        assert int(ea.min()) + int(eb.min()) - 2 * n > -126 and int(ea.max()) + int(eb.max()) < 127
        hi, lo = oz_combine(S, self._ti(ea, rows), self._ti(eb, Op), n)
        ttnn.deallocate(S)
        self._sub("px_comb_s" if name == "B" else "pn_comb_s", ts)
        return hi, lo

    def _candidates(self, kept, rmin, plt, window, dr, pn, N, rows):
        """The samples within window of each particle's best, from the matrices kept on the chip (value = d2 / 2 +
        orientation prior / 2). rmin [R]: each row's minimum as the chip reduced it, which is not exact (measured
        bf16-class in the cell), so the chip window is wider by that much and the exact window is cut here."""
        best = np.full(N, np.inf)
        np.minimum.at(best, pn, 2.0 * rmin + plt)
        marg = 1.0 + float(np.abs(rmin[np.abs(rmin) < FARP / 2]).max()) * 2.0 ** -6
        thr = np.full(rows, -np.inf)
        thr[dr] = (best[pn] + window - plt) / 2 + marg
        t_thr = ttnn.from_torch(torch.from_numpy(thr.astype(np.float32).reshape(rows // 32, 32)), dtype=ttnn.float32,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)
        row_of = np.full(rows, -1, np.int64)
        row_of[dr] = np.arange(len(dr))
        rr, oo, vv = [], [], []
        for D, o0, O in kept:
            res = cand_compact(D, t_thr, rows, O)
            if res is None:                     # more candidates than the kernel's buffers: this chunk comes back whole
                d = ttnn.to_torch(D).numpy()[dr, :O]
                r_, c_ = np.nonzero(d < thr[dr].astype(np.float32)[:, None])
                v_ = d[r_, c_]
                self.c["cand_dense"] = self.c.get("cand_dense", 0) + 1
            else:
                r_, c_ = np.divmod(res[0], O)
                r_, v_ = row_of[r_], res[1]
            ttnn.deallocate(D)
            rr.append(r_), oo.append(c_ + o0), vv.append(2.0 * v_.astype(np.float64) + plt[r_])
        ttnn.deallocate(t_thr)
        rr, oo, vv = np.concatenate(rr), np.concatenate(oo), np.concatenate(vv)
        assert (rr >= 0).all()
        best = np.full(N, np.inf)
        np.minimum.at(best, pn[rr], vv)
        assert np.isfinite(best).all(), "a particle's window missed its own best sample"
        dl = vv - best[pn[rr]]
        k = np.flatnonzero(dl < window)
        k = k[np.lexsort((oo[k], rr[k]))]
        self.c["cand"] = self.c.get("cand", 0) + len(k)
        return rr[k].astype(np.int32), oo[k].astype(np.int32), dl[k].astype(np.float32), best

    @staticmethod
    def _layout(pn, N):
        """Where the rows go on the chip: (device row of each row, particle of each norm row, norm rows, rows).

        sum w |P_o|^2 depends on the particle alone, so its product runs on one row per particle and whole-tile
        copies spread the result over the translations: device row = slot * nb + particle, with nb particle rows
        (whole tiles) and one slot per translation of the longest list. A few particles with many translations
        would pay more rows that way than with a norm row of their own per row; those keep their order."""
        R = len(pn)
        assert (np.diff(pn) >= 0).all()
        blk = lambda v: -(-v // RBLK) * RBLK  # noqa: E731
        slot = np.arange(R) - np.searchsorted(pn, pn)       # a row's place among the rows of its particle
        nb = blk(N) if N > RBLK else 32 << int(np.ceil(np.log2(max(N, 32) / 32)))    # a row block oz_fused has
        rows = blk((int(slot.max()) + 1) * nb)
        if rows < 2 * blk(R):                   # both sides whole 512-row blocks; the norm block is a fraction of one
            return slot * nb + pn, np.arange(N), nb, rows
        return np.arange(R), pn, blk(R), blk(R)

    def _operands(self, xr, xi, w, tx, ty, pn, x, y, dr=None, sel=None, n=None):
        """Particle side, fp32 in the chip's K order. Row r is particle pn[r] at translation (tx[r], ty[r]) and sits
        on device row dr[r] of n: A [n, K] = w S_t X. A2 = w on both parts and nx = sum w |X|^2 (float64), one row
        per particle in sel. The products are float64 and rounded once; the phase is separable in x and y, so the
        sines are taken per distinct coordinate, and a 1024-pixel chunk at a time keeps the temporaries in cache."""
        dr, sel = np.arange(len(pn)) if dr is None else dr, pn if sel is None else sel
        n = int(dr.max()) + 1 if n is None else n
        w = w.astype(np.float64)
        xw = (xr + 1j * xi) * w                                                              # [N, P] complex128
        (xu, xj), (yu, yj) = np.unique(x, return_inverse=True), np.unique(y, return_inverse=True)
        ex = np.exp(1j * tx[:, None].astype(np.float64) * xu)                                # [R, distinct x]
        ey = np.exp(1j * ty[:, None].astype(np.float64) * yu)
        A = np.zeros((n, self.NC, 2, 1024), np.float32)
        for c in range(self.NC):
            p = slice(c * 1024, min(c * 1024 + 1024, self.P))
            z = ex[:, xj[p]] * ey[:, yj[p]] * xw[pn, p]
            A[dr, c, 0, :p.stop - p.start], A[dr, c, 1, :p.stop - p.start] = z.real, z.imag
        nx = (w * (xr.astype(np.float64) ** 2 + xi.astype(np.float64) ** 2)).sum(1)
        return A.reshape(n, self.K), self._kchip(w, w)[sel], nx[sel]

    def d2(self, eul, xr, xi, w, tx, ty, pn, x, y, ref_key=None, out=None, prior=None):
        """float32 [R, O]. eul [O, 9]; xr, xi, w [N, P]; tx, ty, pn [R]: row r scores particle pn[r] at translation
        (tx[r], ty[r]) (phase = x tx + y ty, RELION's units), so particles may bring different numbers of translations.

        ref_key names an orientation set shared by many batches (a global pass): its projections and digits stay on
        the chip until another key or model arrives, so the key must also identify the pixel set. Without it the set
        is used once. out: where to write the result (the server passes the shared segment).

        prior = (lo [O], lt [R], window), float64: -log of RELION's orientation prior per orientation (inf = excluded)
        and of its offset prior per row. Then the dense matrix stays on the chip and the call returns the candidates
        instead, (row [n] int32, orientation [n] int32, delta [n] float32, best [N] float64): every sample whose
        d2 + lo + lt lies within window of its particle's best, delta = that distance, in row order.
        """
        R = len(tx)
        t0 = time.time()
        dr, sel, nb, rows = self._layout(pn, len(xr))
        up = min(-(-(int(dr.max()) + 1) // RUP) * RUP, rows)
        A0, A20, nx = self._operands(xr, xi, w, tx, ty, pn, x, y, dr, sel, up)
        nxh = (nx / 2).astype(np.float32)
        nxl = (nx / 2 - nxh).astype(np.float32)
        pad = lambda v: np.pad(v, (0, nb - len(v)))[:, None]    # noqa: E731
        eul = np.asarray(eul, np.float32).reshape(-1, 9)
        if prior is None:
            out = np.empty((R, len(eul)), np.float32) if out is None else out
        else:
            plo, plt, window = np.asarray(prior[0], np.float64) / 2, np.asarray(prior[1], np.float64), float(prior[2])
            plo = np.where(np.isfinite(plo), plo, FARP)
            kept, rmin = [], []
        t0 = t1 = self._mark("host_s", t0)
        t_nxh, t_nxl = self._put(pad(nxh)), self._put(pad(nxl))

        def tsum(a, b):     # TwoSum in true-fp32 eltwise ops: a + b = s + err exactly
            s_ = ttnn.add(a, b)
            bp = ttnn.subtract(s_, a)
            return s_, ttnn.add(ttnn.subtract(a, ttnn.subtract(s_, bp)), ttnn.subtract(b, bp))

        # The digit planes of a reference set (two products, bf16 planes) get a third of the memory the model's
        # table left. At full image size that is about 1,000 orientations and a call brings more: it is then
        # projected, scored and dropped a piece at a time, each piece with its own balance, and never held.
        omax = max(OBLK, self.free // 3 // (2 * (self.n + self.n2) * self.Kp) // OBLK * OBLK)
        if len(eul) > omax:
            ref_key = None
        held = ref_key is not None and ref_key == self.ref_key
        ma, ma2 = (None, None) if held else (_amax(A0, 0), _amax(A20, 0))
        for o0 in range(0, len(eul), omax):
            if held:
                refs, (s, s2) = self.refs, self.bal
            else:
                refs = self._project(eul[o0:o0 + omax])
                t0 = self._mark("proj_s", t0)
                s, s2 = self._balance(refs, ma, ma2)
                t0 = self._mark("refdig_s", t0)
                if ref_key is not None:
                    self._drop_refs()
                    self.refs, self.bal, self.ref_key = refs, (s, s2), ref_key
            last = o0 + omax >= len(eul)
            A, A2 = np.multiply(A0, s, out=A0 if last else None), A20 * s2      # powers of two: exact in fp32
            ea, ea2 = _exp(_amax(A, 1)), _exp(_amax(A2, 1))
            tA, tA2 = self._put(A), self._put(np.pad(A2, ((0, min(-len(A2) % RUP, nb - len(A2))), (0, 0))))
            Ad, A2d = self._digits(tA, ea, rows), self._digits(tA2, ea2, nb, self.n2)
            t0 = self._mark("imgdig_s", t0)
            oo = o0
            for r in refs:
                Ch, Cl = self._product(Ad, ea, rows, r, "B")
                Nh, Nl = self._product(A2d, ea2, nb, r, "B2")
                ts = time.time()
                # d2 / 2 = (sum w|X|^2 + sum w|P|^2) / 2 - cross, rounded once at the end. The first term is the
                # particle's alone: summed on the norm rows, then copied over the particle's translation slots.
                sh, se = tsum(ttnn.multiply(Nh, 0.5), ttnn.add(ttnn.multiply(Nh, 0.0), t_nxh))
                lo = ttnn.add(ttnn.add(se, ttnn.multiply(Nl, 0.5)), t_nxl)
                for t in (Nh, Nl):
                    ttnn.deallocate(t)
                if prior is not None:           # the orientation prior belongs to the column: one more TwoSum here
                    Op = int(sh.shape[1])
                    ph = np.full(Op, FARP, np.float32)
                    ph[:r["O"]] = plo[oo:oo + r["O"]]
                    pl = np.zeros(Op, np.float32)
                    pl[:r["O"]] = plo[oo:oo + r["O"]] - ph[:r["O"]].astype(np.float64)
                    t_ph, t_pl = self._put(ph[None]), self._put(pl[None])
                    sh, sp = tsum(sh, ttnn.add(ttnn.multiply(sh, 0.0), t_ph))
                    lo = ttnn.add(ttnn.add(lo, sp), t_pl)
                    for t in (t_ph, t_pl):
                        ttnn.deallocate(t)
                sh, lo = _rep(sh, rows // nb), _rep(lo, rows // nb)
                dh, de = tsum(sh, ttnn.neg(Ch))
                D = ttnn.add(dh, ttnn.subtract(ttnn.add(de, lo), Cl))
                self._sub("p_asm_s", ts)
                t0 = self._mark("prod_s", t0)
                if prior is None:
                    np.multiply(ttnn.to_torch(D).numpy()[dr, :r["O"]], 2.0, out=out[:, oo:oo + r["O"]])     # the chip holds d2 / 2
                else:                           # the row minima come back; the matrix waits for its window
                    m = ttnn.min(D, dim=-1, keepdim=True)
                    rmin.append(ttnn.to_torch(m).double().numpy().reshape(-1)[dr])
                    ttnn.deallocate(m)
                    kept.append((D, oo, r["O"]))
                t0 = self._mark("read_s", t0)
                oo += r["O"]
                for t in (Ch, Cl, sh, lo) + ((D,) if prior is None else ()):
                    ttnn.deallocate(t)
            for t in (tA, tA2, Ad, A2d):
                ttnn.deallocate(t)
            if ref_key is None:
                for r in refs:
                    for v in r.values():
                        if isinstance(v, ttnn.Tensor):
                            ttnn.deallocate(v)
        if prior is not None:
            out = self._candidates(kept, np.min(rmin, 0), plt, window, dr, pn, len(xr), rows)
            self._mark("read_s", t0)
        for t in (t_nxh, t_nxl):
            ttnn.deallocate(t)
        c = self.c
        c["dev_s"] += time.time() - t1
        c["batches"], c["rows"], c["rows_padded"] = c["batches"] + 1, c["rows"] + R, c["rows_padded"] + rows
        return out
