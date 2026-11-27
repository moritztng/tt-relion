"""The chip, in a process of its own.

RELION's ranks run on the system Open MPI, and ttnn ships a second Open MPI that tt-metal always
initialises. Both in one address space abort the rank ("MPI_Init invoked more than once"). So each rank's
bridge starts one of these servers with mpirun's variables removed; it owns the rank's chip (the inherited
TT_VISIBLE_DEVICES) and computes the cross term. Protocol and numerics: tt_relion/cross.py. Kept left
operands stay on the chip as device tensors, so a coarse call ships only the shifted image up.
"""
import json
import os
import sys
import time

import numpy as np

# TT_RELION_PHASES=1: sync the chip after each phase of an Ozaki call and log, at exit, the median per call of
# issue_s (host time to enqueue the phase) and wait_s (then until the chip is done). Instrument only: the syncs
# remove the overlap a normal call has, so read the split, not the sum, against an unsynced warm call.
PHASES = os.environ.get("TT_RELION_PHASES") == "1"
# TT_RELION_OZ_CUT=batch: cut the digits from R_i = round(x 2^(2i+1)) for all i at once (_digits_batch), ~n + 8 ops
# per operand instead of 6n. Default "seq" (the per-digit loop) until the chip check passes.
OZ_CUT = os.environ.get("TT_RELION_OZ_CUT", "seq")
OZ_CUT_MB = 256   # fp32 bytes of one stacked group of digits; bounds the batch cut's peak device memory


class _Ttnn:
    def __init__(self, fidelity):
        import torch
        import ttnn
        self.torch, self.ttnn = torch, ttnn
        self.dev = ttnn.open_device(device_id=0)
        self.cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                                                    math_approx_mode=False, fp32_dest_acc_en=True,
                                                    packer_l1_acc=True)

    def _mark(self, name):
        if PHASES:
            t = time.perf_counter()
            self.ttnn.synchronize_device(self.dev)
            t2 = time.perf_counter()
            self._call[name] = (t - self._t, t2 - t)
            self._t = t2

    def _start(self):
        if PHASES and not getattr(self, "_open", False):
            self._call, self._open, self._t = {}, True, time.perf_counter()

    def _put(self, a):
        return self.ttnn.from_torch(self.torch.from_numpy(a), dtype=self.ttnn.float32,
                                    layout=self.ttnn.TILE_LAYOUT, device=self.dev)

    def upload(self, lhs):
        return (self._put(lhs), lhs.shape[0], lhs.shape[2])

    def shape(self, a):
        return a[1], a[2]

    def mm(self, a, rhs):
        ttnn = self.ttnn
        b = self._put(rhs)
        c = ttnn.matmul(a[0], b, compute_kernel_config=self.cfg)
        out = ttnn.to_torch(c).numpy()
        ttnn.deallocate(b)
        ttnn.deallocate(c)
        return out

    def free(self, a):
        for x in (a if isinstance(a, list) else a[:1]):
            self.ttnn.deallocate(x)

    # ---- exact-integer cross term (cross.py TT_RELION_OZAKI) ----
    _MAGIC = 12582912.0   # 1.5 * 2^23: (x + M) - M rounds to the nearest integer in fp32

    def _digits(self, runs, n, tag):
        """runs fp32 numpy, |x| < 1 -> n bf16 device tensors of balanced base-4 digits (exact fp32 ops on the chip)."""
        ttnn, out = self.ttnn, []
        x = r = self._put(runs)
        self._mark(tag + "_put")
        for i in range(n):
            t = ttnn.multiply(r, float(2 ** (2 * i + 1)))
            q = ttnn.subtract(ttnn.add(t, self._MAGIC), self._MAGIC)
            out.append(ttnn.typecast(q, ttnn.bfloat16))
            r2 = ttnn.subtract(r, ttnn.multiply(q, float(2.0 ** -(2 * i + 1))))
            ttnn.deallocate(t), ttnn.deallocate(q)
            if r is not x:
                ttnn.deallocate(r)
            r = r2
        ttnn.deallocate(r), ttnn.deallocate(x)
        self._mark(tag + "_digits")
        return out

    def _digits_batch(self, runs, n, tag, dim):
        """As _digits, stacked along dim (bf16, n blocks of the runs' shape). With R_i = round(x 2^(2i+1)) (half
        to even, R_-1 = 0), the loop's digit i is R_i - 4 R_{i-1}: shifting by the even integer 4 R_{i-1}
        commutes with half-to-even rounding. Every R_i and 4 R_{i-1} is an fp32 integer (|x 2^27| < 2^27 is
        exact; above 2^23 it is already whole) and their difference is |q| <= 2, so each step is exact."""
        ttnn = self.ttnn
        x = self._put(runs)
        self._mark(tag + "_put")
        g = max(1, min(n, (OZ_CUT_MB << 20) // max(1, runs.size * 4)))
        out = []
        for a in range(0, n, g):
            b = min(n, a + g)
            ys = [ttnn.multiply(x, 0.0 if i < 0 else float(2 ** (2 * i + 1))) for i in range(a - 1, b)]
            y = ttnn.concat(ys, dim)
            self.free(ys)
            r = ttnn.round(y, decimals=0)
            ttnn.deallocate(y)
            sh = list(r.shape)
            blk = sh[dim] // (b - a + 1)
            end = list(sh)
            end[dim] = blk * (b - a)
            lo = ttnn.slice(r, [0] * len(sh), end)
            start = [0] * len(sh)
            start[dim] = blk
            hi = ttnn.slice(r, start, sh)
            lo4 = ttnn.multiply(lo, 4.0)
            q = ttnn.subtract(hi, lo4)
            for z in (r, lo, hi, lo4):
                ttnn.deallocate(z)
            out.append(ttnn.typecast(q, ttnn.bfloat16))
            ttnn.deallocate(q)
        ttnn.deallocate(x)
        c = out[0] if len(out) == 1 else ttnn.concat(out, dim)
        if len(out) > 1:
            self.free(out)
        self._mark(tag + "_digits")
        return c

    def oz_upload(self, runs, n):
        """runs [nc, O, 480] -> (A, O, Op): the n lhs digits stacked by rows, A [nc, n * Op, 480] bf16, row block i
        = digit i, O padded to a whole tile so every block starts on a tile row."""
        o = runs.shape[1]
        op = -(-o // 32) * 32
        if op > o:
            runs = np.pad(runs, ((0, 0), (0, op - o), (0, 0)))
        self._start()
        if OZ_CUT == "batch":
            a = self._digits_batch(runs, n, "lhs", 1)
        else:
            d = self._digits(runs, n, "lhs")
            a = self.ttnn.concat(d, 1)
            self.free(d)
            self._mark("lhs_stack")
        return a, o, op

    def _tree_sum(self, c):
        """fp32 [nc, O, T] -> [1, O, T] by pairwise elementwise adds (exact: integer sums < 2^24)."""
        ttnn = self.ttnn
        while c.shape[0] > 1:
            m, h, o, t = c.shape[0], c.shape[0] // 2, c.shape[1], c.shape[2]
            lo, hi = ttnn.slice(c, [0, 0, 0], [h, o, t]), ttnn.slice(c, [h, 0, 0], [2 * h, o, t])
            s = ttnn.add(lo, hi)
            if m % 2:
                s = ttnn.concat([s, ttnn.slice(c, [2 * h, 0, 0], [m, o, t])], 0)
            for z in (lo, hi, c):
                ttnn.deallocate(z)
            c = s
        return c

    def oz_mm(self, a, rhs, n):
        """a = oz_upload(...); rhs runs [nc, 480, T] fp32 -> n float64 [O, T], sum s = sum_{i+j=s} a_i . b_j.

        One matmul per rhs digit j gives a_r . b_j for every r in row block r (the blocks with r + j >= n are
        thrown away: n^2 small products instead of n(n+1)/2, in n dispatches instead of n(n+1)/2). The diagonal
        sums are Horner shifts: acc_j = M_j + (acc_{j+1} moved down one block), so block s of acc_0 is sum s.
        Every value stays an integer below 2^24 (the old per-pair path's bound), so the result is the same."""
        ttnn = self.ttnn
        if not hasattr(self, "lofi"):
            self.lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                                         fp32_dest_acc_en=True, packer_l1_acc=True)
        A, o, op = a
        self._start()
        if OZ_CUT == "batch":
            bs, sh = self._digits_batch(rhs, n, "rhs", 0), rhs.shape
            b = [ttnn.slice(bs, [j * sh[0], 0, 0], [(j + 1) * sh[0], sh[1], sh[2]]) for j in range(n)]
            ttnn.deallocate(bs)
            self._mark("rhs_slices")
        else:
            b = self._digits(rhs, n, "rhs")
        nc, t = rhs.shape[0], rhs.shape[2]
        z = acc = None
        for j in range(n - 1, -1, -1):
            m = ttnn.matmul(A, b[j], compute_kernel_config=self.lofi, dtype=ttnn.float32)
            if acc is not None:
                if z is None:
                    z = ttnn.zeros([nc, op, t], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.dev)
                top = ttnn.slice(acc, [0, 0, 0], [nc, (n - 1) * op, t])
                sh = ttnn.concat([z, top], 1)
                m2 = ttnn.add(m, sh)
                for x in (m, acc, top, sh):
                    ttnn.deallocate(x)
                m = m2
            acc = m
        if z is not None:
            ttnn.deallocate(z)
        self.free(b)
        self._mark("mm_chain")
        acc = self._tree_sum(acc)
        self._mark("tree")
        out = ttnn.to_torch(acc).double().numpy().reshape(n, op, t)[:, :o]
        ttnn.deallocate(acc)
        self._mark("download")
        if PHASES:
            self._open = False
            self.calls_ph = getattr(self, "calls_ph", []) + [(o, t, self._call)]
        return list(out)

    # ---- projection on the chip (cross.py "proj" requests) ----
    # The model is kept as a corner table: row v holds the 2x2x2 neighbours of voxel v (16 fp32, column
    # dx*8 + dy*4 + dz*2 + real/imag), so one gather row per pixel feeds the trilinear interpolation.
    # ttnn.embedding gathers bf16 only, so each fp32 entry is stored as three bf16 pieces whose fp32 sum
    # is the entry exactly. Arithmetic is RELION's project3Dmodel in fp32 (perf/rlx/proj_proto.py on
    # wk/rlx-kernels checks it against float64: same error as torch fp32 at s50..s360).

    def set_model(self, mkey, mdl, geom):
        torch, ttnn = self.torch, self.ttnn
        self.models = getattr(self, "models", {})
        X, XY = geom["mdlX"], geom["mdlX"] * geom["mdlY"]
        m = torch.from_numpy(mdl)
        n = m.shape[0]
        m = torch.cat([m, torch.zeros(XY + X + 2, 2)])
        T = torch.empty(n, 16)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    o, k = dx + dy * X + dz * XY, dx * 8 + dy * 4 + dz * 2
                    T[:, k:k + 2] = m[o:o + n]
        t = T.double()
        hi = t.to(torch.bfloat16)
        mid = (t - hi.double()).to(torch.bfloat16)
        lo = (t - hi.double() - mid.double()).to(torch.bfloat16)
        tab = ttnn.from_torch(torch.cat([hi, mid, lo], 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=self.dev)
        self.models[mkey] = tab
        while len(self.models) > 2:
            ttnn.deallocate(self.models.pop(next(iter(self.models))))

    def _runs(self, g, kc):
        """Pixel coordinates as device [1, kc] runs; padding pixels sit far outside the radius (projected to 0)."""
        from tt_relion.cross import pixels
        key = (g["P"], g["imgX"], g["imgY"], g["maxR"], kc)
        if getattr(self, "_runs_key", None) != key:
            x, y = pixels(g)
            npc = -(-g["P"] // kc)
            xs, ys = (np.full(npc * kc, 1e4, np.float32) for _ in range(2))
            xs[:g["P"]], ys[:g["P"]] = x, y
            put = lambda v: self.ttnn.from_torch(self.torch.from_numpy(v.reshape(1, kc).copy()),  # noqa: E731
                                                 dtype=self.ttnn.float32, layout=self.ttnn.TILE_LAYOUT, device=self.dev)
            self._runs_xy = [(put(xs[c * kc:(c + 1) * kc]), put(ys[c * kc:(c + 1) * kc])) for c in range(npc)]
            self._runs_key = key
        return self._runs_xy

    def _project(self, tab, ecols, Xd, Yd, g):
        ttnn = self.ttnn
        e0, e1, e3, e4, e6, e7 = ecols
        pf = float(g["pf"])
        xp = ttnn.multiply(ttnn.add(ttnn.multiply(e0, Xd), ttnn.multiply(e1, Yd)), pf)
        yp = ttnn.multiply(ttnn.add(ttnn.multiply(e3, Xd), ttnn.multiply(e4, Yd)), pf)
        zp = ttnn.multiply(ttnn.add(ttnn.multiply(e6, Xd), ttnn.multiply(e7, Yd)), pf)
        r2 = ttnn.add(ttnn.add(ttnn.multiply(xp, xp), ttnn.multiply(yp, yp)), ttnn.multiply(zp, zp))
        inside = ttnn.lt(r2, float(g["maxR2"] + 1))          # RELION: int(r2) <= maxR2
        sg = ttnn.add(ttnn.multiply(ttnn.ltz(xp), -2.0), 1.0)  # Hermitian flip for x < 0
        xp, yp, zp = ttnn.multiply(xp, sg), ttnn.multiply(yp, sg), ttnn.multiply(zp, sg)
        x0, y0, z0 = ttnn.floor(xp), ttnn.floor(yp), ttnn.floor(zp)
        fx, fy, fz = ttnn.subtract(xp, x0), ttnn.subtract(yp, y0), ttnn.subtract(zp, z0)
        X, Y, Z = g["mdlX"], g["mdlY"], g["mdlZ"]
        xi = ttnn.typecast(ttnn.clamp(x0, 0.0, float(X - 2)), ttnn.int32)
        yi = ttnn.typecast(ttnn.clamp(ttnn.subtract(y0, float(g["mdlInitY"])), 0.0, float(Y - 2)), ttnn.int32)
        zi = ttnn.typecast(ttnn.clamp(ttnn.subtract(z0, float(g["mdlInitZ"])), 0.0, float(Z - 2)), ttnn.int32)
        idx = ttnn.add(ttnn.add(ttnn.multiply(zi, X * Y), ttnn.multiply(yi, X)), xi)
        o, p = idx.shape[0], idx.shape[1]
        idx = ttnn.to_layout(ttnn.typecast(idx, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        G = ttnn.embedding(idx, tab, layout=ttnn.ROW_MAJOR_LAYOUT)
        G = ttnn.to_layout(ttnn.permute(ttnn.reshape(G, (o, p, 48)), (2, 0, 1)), ttnn.TILE_LAYOUT)
        G = ttnn.typecast(G, ttnn.float32)
        G = ttnn.add(ttnn.add(G[:16], G[16:32]), G[32:])

        def lerp(lo, hi, f):
            return ttnn.add(lo, ttnn.multiply(ttnn.subtract(hi, lo), f))

        D = lerp(G[:8], G[8:], fx)
        E = lerp(D[:4], D[4:], fy)
        F = lerp(E[:2], E[2:], fz)
        return ttnn.multiply(F[0], inside), ttnn.multiply(ttnn.multiply(F[1], sg), inside)

    def _split(self, x):
        """x fp32 -> SPLIT fp32 tensors with bf16-exact values summing to x (cross.bf16_split, on the chip)."""
        from tt_relion.cross import SPLIT
        ttnn, parts = self.ttnn, []
        for _ in range(SPLIT - 1):
            hi = ttnn.typecast(ttnn.typecast(x, ttnn.bfloat16), ttnn.float32)
            parts.append(hi)
            x = ttnn.subtract(x, hi)
        return parts + [x]

    def project(self, mkey, eul, g, w):
        """-> (the cross.proj_lhs layout of ref, [nc, O, kc], built on the chip; ref2 [O] float64).

        ref2 = sum_p w[p] |ref[o, p]|^2 is accumulated with fp32 elementwise adds over the pixel runs (exact
        IEEE ops on the chip) and only the last [O, kc] is summed on the host in float64: a chip reduction,
        matmul or ttnn.sum, misses this all-positive sum by ~2.5e-4 relative on Blackhole."""
        from tt_relion.cross import KCHUNK, SPLIT
        ttnn, tab = self.ttnn, self.models[mkey]
        kc = KCHUNK
        ecols = [self._put(np.ascontiguousarray(eul[:, c:c + 1])) for c in (0, 1, 3, 4, 6, 7)]
        o = eul.shape[0]
        wp = np.zeros(len(self._runs(g, kc)) * kc, np.float32)
        wp[:g["P"]] = w
        lhs = [[[] for _ in (0, 1)] for _ in range(SPLIT)]
        acc = None
        for c, (Xd, Yd) in enumerate(self._runs(g, kc)):
            wd = self._put(wp[c * kc:(c + 1) * kc].reshape(1, kc).copy())
            for h, v in enumerate(self._project(tab, ecols, Xd, Yd, g)):
                for i, part in enumerate(self._split(v)):
                    lhs[i][h].append(ttnn.reshape(part, (1, o, kc)))
                vv = ttnn.multiply(ttnn.multiply(v, v), wd)
                acc = vv if acc is None else ttnn.add(acc, vv)
        a = ttnn.concat([r for i in lhs for h in i for r in h], 0)
        ref2 = self.torch.sum(ttnn.to_torch(acc).double(), 1).numpy()
        return (a, a.shape[0], kc), ref2

    def close(self):
        if PHASES and getattr(self, "calls_ph", None):
            by = {}
            for o, t, c in self.calls_ph[1:] or self.calls_ph:   # the first call compiles
                by.setdefault("O%d_T%d_%s" % (o, t, "fine" if "lhs_put" in c else "kept"), []).append(c)
            rep = {k: {p: [round(float(np.median([c[p][j] for c in cs if p in c])), 5) for j in (0, 1)]
                       for p in cs[-1]} | {"n": len(cs)} for k, cs in by.items()}
            print("TTPHASES " + json.dumps(rep, separators=(",", ":")), file=sys.stderr, flush=True)
        self.ttnn.close_device(self.dev)


def main():
    reply = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)                       # library chatter goes to stderr, never into the protocol
    from tt_relion.cross import serve
    serve(_Ttnn(sys.argv[1]), reply)


if __name__ == "__main__":
    main()
