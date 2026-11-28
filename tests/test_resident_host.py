"""Host side of tt_relion.resident without a chip: the operands it would upload, multiplied in float64, give
RELION's direct sum w |P - S X|^2. The chip side (projection, digit products, combine) is graded on a device."""
import sys
import types

import numpy as np

sys.modules.setdefault("ttnn", types.SimpleNamespace(TILE_LAYOUT=0, ROW_MAJOR_LAYOUT=1, Tensor=object))
for m in ("tt_relion.oz_fused", "tt_relion.proj_lerp"):
    sys.modules.setdefault(m, types.SimpleNamespace(cand_compact=None, oz_combine=None, oz_cut=None, oz_fused=None, proj_rot=None))
from tt_relion.resident import Resident  # noqa: E402


def test_operands_match_direct_form():
    rng = np.random.default_rng(0)
    N, T, P, O = 3, 5, 1500, 7
    xr, xi = rng.normal(size=(2, N, P)).astype(np.float32)
    w = rng.uniform(0, 1, (N, P)).astype(np.float32)
    tx, ty = (rng.normal(size=(2, N, T)) * 0.05).astype(np.float32)
    x, y = rng.integers(0, 30, P).astype(np.float32), rng.integers(-30, 30, P).astype(np.float32)
    pr, pi = rng.normal(size=(2, O, P)).astype(np.float32)
    r = Resident(None)
    r.P, r.NC = P, -(-P // 1024)
    r.K = r.NC * 2048
    A, A2, nx = r._operands(xr, xi, w, tx.ravel(), ty.ravel(), np.repeat(np.arange(N), T), x, y)
    B = r._kchip(pr, pi).astype(np.float64)
    d2 = (nx[:, None] + A2 @ (B * B).T - 2 * A @ B.T).reshape(N, T, O).transpose(0, 2, 1)
    ph = tx[:, :, None].astype(np.float64) * x + ty[:, :, None].astype(np.float64) * y
    sx = (xr[:, None] + 1j * xi[:, None]) * np.exp(1j * ph)                       # [N, T, P]
    ref = (pr + 1j * pi)[None, :, None]                                             # [1, O, 1, P]
    direct = (w[:, None, None].astype(np.float64) * np.abs(ref - sx[:, None]) ** 2).sum(-1)
    err = np.abs(d2 - direct).max() / np.abs(direct).max()
    assert err < 1e-6, err          # the operands are rounded to fp32 for upload


def test_layout_spreads_the_particle_norm_over_its_rows():
    """The chip runs sum w |P|^2 on the norm rows and copies it over rows // nb slots. Done here in float64, every
    layout gives the direct sum: many particles (one norm row each), ragged translation lists, and a few particles
    with many translations (a norm row per row)."""
    rng = np.random.default_rng(1)
    P, O = 300, 5
    x, y = rng.integers(0, 30, P).astype(np.float32), rng.integers(-30, 30, P).astype(np.float32)
    pr, pi = rng.normal(size=(2, O, P)).astype(np.float32)
    r = Resident(None)
    r.P, r.NC = P, 1
    r.K = 2048
    B = r._kchip(pr, pi).astype(np.float64)
    for counts, per_particle in (([37] * 24, True), ([4, 16, 8, 1, 9] * 4, True), ([37] * 3, False), ([2] * 40, True)):
        N = len(counts)
        pn = np.repeat(np.arange(N), counts)
        xr, xi = rng.normal(size=(2, N, P)).astype(np.float32)
        w = rng.uniform(0, 1, (N, P)).astype(np.float32)
        tx, ty = (rng.normal(size=(2, len(pn))) * 0.05).astype(np.float32)
        dr, sel, nb, rows = r._layout(pn, N)
        assert (len(sel) == N) == per_particle and rows % nb == 0 and rows % 512 == 0 and len(set(dr)) == len(pn)
        assert nb in (32, 64, 128, 256) or nb % 512 == 0
        A, A2, nx = r._operands(xr, xi, w, tx, ty, pn, x, y, dr, sel)
        norm = np.zeros((nb, O))
        norm[:len(A2)] = nx[:, None] + A2 @ (B * B).T
        d2 = (np.tile(norm, (rows // nb, 1))[:len(A)] - 2 * A @ B.T)[dr]
        ph = tx[:, None].astype(np.float64) * x + ty[:, None].astype(np.float64) * y
        sx = (xr[pn] + 1j * xi[pn]) * np.exp(1j * ph)                               # [R, P]
        direct = (w[pn][:, None].astype(np.float64) * np.abs((pr + 1j * pi)[None] - sx[:, None]) ** 2).sum(-1)
        err = np.abs(d2 - direct).max() / np.abs(direct).max()
        assert err < 1e-6, (counts[:3], err)


def test_candidates_are_the_window_of_each_particle():
    """The candidate form on stand-in chip pieces: an inexact row minimum, a compaction that emits exactly the
    elements under each row's threshold, device rows permuted. Every sample within the window of its particle's best
    comes back with its distance, and nothing else."""
    import tt_relion.resident as rs
    rng = np.random.default_rng(3)
    rs.ttnn.from_torch = lambda t, **k: t.numpy()
    rs.ttnn.deallocate = lambda t: None
    rs.ttnn.float32 = None

    def compact(D, thr, rows, cols):
        r, c = np.nonzero(D[:rows, :cols] < thr.reshape(-1)[:rows, None])
        return r * cols + c, D[r, c]
    rs.cand_compact = compact
    r = Resident(None)
    r.c = {}
    for counts in ([37] * 24, [4, 16, 8, 1, 9] * 4, [37] * 3):
        N = len(counts)
        pn = np.repeat(np.arange(N), counts)
        dr, sel, nb, rows = r._layout(pn, N)
        plt = rng.uniform(0, 6, len(pn))
        chunks, kept, rmin, o0 = [], [], [], 0
        for O, Op in ((512, 512), (300, 512)):
            half = (rng.gamma(2.0, 30.0, (len(pn), O)) + 500).astype(np.float32)     # d2 / 2 + orientation prior / 2
            D = np.zeros((rows, Op), np.float32)
            D[dr, :O] = half
            D[:, O:] = 1e30
            kept.append((D, o0, O))
            chunks.append(half)
            rmin.append(half.min(1).astype(np.float64) * (1 + rng.uniform(-1, 1, len(pn)) * 2.0 ** -8))
            o0 += O
        rr, oo, dl, best = r._candidates(kept, np.min(rmin, 0), plt, 40.0, dr, pn, N, rows)
        E = 2.0 * np.concatenate(chunks, 1).astype(np.float64) + plt[:, None]
        b = np.full(N, np.inf)
        np.minimum.at(b, pn, E.min(1))
        want_r, want_o = np.nonzero(E - b[pn][:, None] < 40.0)
        assert np.array_equal(rr, want_r) and np.array_equal(oo, want_o) and np.array_equal(best, b)
        assert np.abs(dl - (E - b[pn][:, None])[want_r, want_o]).max() < 1e-4


def test_corner_table_matches_the_direct_form():
    from tt_relion.resident import TBLK, _corner_table
    X, Y, Z = 20, 37, 37
    nvox = X * Y * Z
    assert nvox > TBLK and nvox % TBLK
    mdl = np.random.default_rng(3).standard_normal((nvox, 2)).astype(np.float32)
    mp = np.concatenate([mdl, np.zeros((X * Y + X + 2, 2), np.float32)])
    ref = np.empty((nvox, 16), np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                o, c = dx + dy * X + dz * X * Y, 4 * dz + 2 * dy + dx
                ref[:, 2 * c:2 * c + 2] = mp[o:o + nvox]
    assert np.array_equal(_corner_table(mdl, X, Y), ref)


def test_table_turns_are_host_wide_and_counted(monkeypatch, tmp_path):
    import threading

    from tt_relion import resident
    monkeypatch.setattr(resident, "TURNS", 2)
    monkeypatch.setattr(resident.tempfile, "gettempdir", lambda: str(tmp_path))
    got = threading.Event()

    def third():
        with resident._turn():
            got.set()
    with resident._turn():
        with resident._turn():
            t = threading.Thread(target=third)
            t.start()
            assert not got.wait(0.5)        # both places taken
        assert got.wait(5)                  # one given back
    t.join()


def test_own_form_scores_each_particle_on_its_own_set():
    """d2_own on stand-in chip pieces (float64 products, one 'digit'): every particle's block is the direct sum
    w |P - S X|^2 over its own orientations and translations, whatever the set sizes and list lengths, and a call
    that needs several programs or a padded one returns the same."""
    import torch

    import tt_relion.resident as rs

    class T:
        def __init__(self, a):
            self.a, self.shape = a, a.shape

        def is_allocated(self):
            return True
    tt = rs.ttnn
    tt.float32, tt.int32 = "f", "i"
    tt.from_torch = lambda t, **k: T(t.numpy().astype(np.float64 if k.get("dtype") == "f" else np.int64))
    tt.to_torch = lambda x: torch.from_numpy(np.ascontiguousarray(x.a))
    tt.to_layout = lambda x, layout: x
    tt.reshape = lambda x, shape: T(x.a.reshape(tuple(shape)))
    tt.multiply = lambda a, b: T(a.a * (b.a if isinstance(b, T) else b))
    tt.abs = lambda x: T(np.abs(x.a))
    tt.max = lambda x, dim: T(x.a.max(dim))
    tt.pad = lambda x, p, v: T(np.pad(x.a, p))
    tt.slice = lambda x, b, e: T(x.a[tuple(slice(i, j) for i, j in zip(b, e))])
    tt.deallocate = lambda x: None
    tt.synchronize_device = lambda d: None

    def proj(e, pix):                           # a stand-in projection: smooth in the matrix, zero on pad pixels
        ph = e[:, 0:1] * (pix + 1) * 0.37 + e[:, 4:5] * 1.3
        amp = 1.0 / (1.0 + 0.05 * pix)
        return np.cos(ph) * amp, np.sin(ph + e[:, 7:8]) * amp

    def proj_rot(tab, pix, rot, geom):          # rot [O, 8] = pf * (e0, e1, e3, e4, e6, e7, 0, 0) -> [2 O NC, 1024]
        e = np.zeros((len(rot.a), 9))
        e[:, [0, 1, 3, 4, 6, 7]] = rot.a[:, :6] / r.pf
        NC = pix.a.shape[0] // 2
        out = np.zeros((2, len(e), NC * 1024))
        out[0, :, :P], out[1, :, :P] = proj(e, np.arange(P))
        return T(out.reshape(2, len(e), NC, 1024).transpose(1, 2, 0, 3).reshape(-1, 1024))
    rs.proj_rot = proj_rot
    rs.oz_cut = lambda X, e, n: T(np.concatenate([(X.a * np.ldexp(1.0, -e.a.reshape(-1)[:len(X.a)])[:, None])[None],
                                                  np.zeros((n - 1,) + X.a.shape)]))
    seen = []

    def oz_fused(Ad, Bd, n, rb, nb, bt, q, adm, bdm):
        assert bt and adm and bdm and Ad.shape[1] == 32 * rb and Bd.shape[1] % (32 * nb) == 0
        seen.append(q)
        S = np.zeros((q * n, Ad.shape[1], Bd.shape[1]))
        for j in range(q):                      # digit-major operands: plane i of batch j is i * q + j
            S[j * n] = Ad.a[j] @ Bd.a[j].T
        return T(S)
    rs.oz_fused = oz_fused

    def oz_combine(S, ea, eb, n, q):
        R, O = S.shape[1], S.shape[2]
        hi = S.a.reshape(q, n, R, O)[:, 0] * np.ldexp(1.0, ea.a.reshape(q, R, 1) + eb.a.reshape(q, 1, O))
        return T(hi.reshape(q * R, O)), T(np.zeros((q * R, O)))
    rs.oz_combine = oz_combine

    rng = np.random.default_rng(5)
    P = 1500
    x, y = (np.arange(P) % 40).astype(np.float32), (np.arange(P) // 40 - 18).astype(np.float32)
    for N, free in ((5, 1 << 40), (5, 3 * 4 * 13 * 4320 * 96 * 2), (1, 1 << 40)):
        r = Resident(None)
        r.free, r.pf, r.geom, r.tab = free, 2, None, None
        r.set_pixels("k", x, y)
        on = rng.integers(20, 90, N)
        nt = rng.integers(1, 12, N)
        eul = rng.uniform(-1, 1, (int(on.sum()), 9)).astype(np.float32)
        xr, xi = rng.standard_normal((2, N, P)).astype(np.float32)
        w = rng.uniform(0.5, 2, (N, P)).astype(np.float32)
        pn = np.repeat(np.arange(N), nt)
        tx, ty = (rng.uniform(-0.2, 0.2, (2, len(pn)))).astype(np.float32)
        seen.clear()
        got = r.d2_own(eul, on, xr, xi, w, tx, ty, pn, x, y)
        assert len(got) == N and (len(seen) > 2) == (free < 1 << 40 and N > 1)
        e0, r0 = np.concatenate([[0], np.cumsum(on)]), np.concatenate([[0], np.cumsum(nt)])
        for i in range(N):
            pr, pi = proj(eul[e0[i]:e0[i + 1]].astype(np.float64), np.arange(P))
            ph = tx[r0[i]:r0[i + 1], None].astype(np.float64) * x + ty[r0[i]:r0[i + 1], None].astype(np.float64) * y
            sx = (xr[i] + 1j * xi[i]) * np.exp(1j * ph)                                 # [T, P]
            d = (w[i] * np.abs(sx[:, None] - (pr + 1j * pi)[None]) ** 2).sum(-1)        # [T, on]
            assert got[i].shape == d.shape
            assert np.abs(got[i] - d).max() < 2e-6 * np.abs(d).max(), (N, i, np.abs(got[i] - d).max())
