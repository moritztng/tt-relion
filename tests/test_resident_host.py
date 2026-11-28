"""Host side of tt_relion.resident without a chip: the operands it would upload, multiplied in float64, give
RELION's direct sum w |P - S X|^2. The chip side (projection, digit products, combine) is graded on a device."""
import sys
import types

import numpy as np

sys.modules.setdefault("ttnn", types.SimpleNamespace(TILE_LAYOUT=0, ROW_MAJOR_LAYOUT=1, Tensor=object))
for m in ("tt_relion.oz_fused", "tt_relion.proj_lerp"):
    sys.modules.setdefault(m, types.SimpleNamespace(oz_combine=None, oz_cut=None, oz_fused=None, proj_rot=None))
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
        norm[:len(A2)] = A2 @ (B * B).T
        d2 = (nx[:, None] + np.tile(norm, (rows // nb, 1))[:len(A)] - 2 * A @ B.T)[dr]
        ph = tx[:, None].astype(np.float64) * x + ty[:, None].astype(np.float64) * y
        sx = (xr[pn] + 1j * xi[pn]) * np.exp(1j * ph)                               # [R, P]
        direct = (w[pn][:, None].astype(np.float64) * np.abs((pr + 1j * pi)[None] - sx[:, None]) ** 2).sum(-1)
        err = np.abs(d2 - direct).max() / np.abs(direct).max()
        assert err < 1e-6, (counts[:3], err)
