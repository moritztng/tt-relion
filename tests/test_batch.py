"""diff2_coarse_batch equals N per-particle diff2_coarse calls (torch backend, CPU)."""
import os
import numpy as np

os.environ.setdefault("TT_RELION_BACKEND", "torch")
from tt_relion import relion  # noqa: E402


def _case(seed=0, n=5, O=40, T=7, img=16, box=24):
    rng = np.random.default_rng(seed)
    pad = 2
    mdlX = box * pad // 2 + 1
    mdlY = mdlZ = box * pad + 1
    mdl = rng.standard_normal((mdlZ, mdlY, mdlX, 2)).astype(np.float32)
    R = []
    for _ in range(O):
        q, _r = np.linalg.qr(rng.standard_normal((3, 3)))
        R.append(q.astype(np.float32).ravel())
    eul = np.array(R)
    imgX, imgY = img // 2 + 1, img
    P = imgX * imgY
    geom = dict(mdlX=mdlX, mdlY=mdlY, mdlZ=mdlZ, mdlInitY=-(mdlY // 2), mdlInitZ=-(mdlZ // 2),
                maxR=img // 2 - 1, maxR2_padded=(img // 2 * pad) ** 2, padding_factor=float(pad),
                imgX=imgX, imgY=imgY)
    tx = rng.uniform(-0.3, 0.3, (n, T)).astype(np.float32)
    ty = rng.uniform(-0.3, 0.3, (n, T)).astype(np.float32)
    ir = rng.standard_normal((n, P)).astype(np.float32)
    ii = rng.standard_normal((n, P)).astype(np.float32)
    corr = rng.uniform(0.1, 1.0, (n, P)).astype(np.float32)
    return mdl.ravel(), eul, tx, ty, ir, ii, corr, geom, P


def test_batch_matches_per_particle():
    mdl, eul, tx, ty, ir, ii, corr, g, P = _case()
    n, T, O = tx.shape[0], tx.shape[1], eul.shape[0]
    got = relion.diff2_coarse_batch(mdl, eul, tx, ty, ir, ii, corr, **g)
    for p in range(n):
        out = np.zeros(O * T, np.float32)
        assert relion.diff2_coarse(memoryview(mdl), memoryview(eul.ravel()), memoryview(tx[p]),
                                   memoryview(ty[p]), memoryview(ir[p]), memoryview(ii[p]),
                                   memoryview(corr[p]), memoryview(out),
                                   g["mdlX"], g["mdlY"], g["mdlZ"], g["mdlInitY"], g["mdlInitZ"],
                                   g["maxR"], g["maxR2_padded"], g["padding_factor"],
                                   g["imgX"], g["imgY"], O, T, P)
        ref = out.reshape(O, T)
        rel = np.abs(got[p] - ref).max() / np.abs(ref).max()
        assert rel < 1e-5, (p, rel)
        assert (got[p].argmin() == ref.argmin())


def test_threads_collect_into_batches():
    """Concurrent diff2_coarse calls (RELION's --j threads) are grouped and stay exact."""
    import threading
    mdl, eul, tx, ty, ir, ii, corr, g, P = _case(seed=1, n=11)
    n, T, O = tx.shape[0], tx.shape[1], eul.shape[0]
    ref = relion.diff2_coarse_batch(mdl, eul, tx, ty, ir, ii, corr, **g)
    relion._BATCH, relion._BATCH_WAIT = 4, 0.2
    outs = [np.zeros(O * T, np.float32) for _ in range(n)]
    ok = [False] * n

    def run(p):
        ok[p] = relion.diff2_coarse(memoryview(mdl), memoryview(eul.ravel()), memoryview(tx[p]),
                                    memoryview(ty[p]), memoryview(ir[p]), memoryview(ii[p]),
                                    memoryview(corr[p]), memoryview(outs[p]),
                                    g["mdlX"], g["mdlY"], g["mdlZ"], g["mdlInitY"], g["mdlInitZ"],
                                    g["maxR"], g["maxR2_padded"], g["padding_factor"],
                                    g["imgX"], g["imgY"], O, T, P)
    b0 = relion._stats.get("batches", 0)
    th = [threading.Thread(target=run, args=(p,)) for p in range(n)]
    [x.start() for x in th]
    [x.join(60) for x in th]
    relion._BATCH = 0
    assert all(ok)
    for p in range(n):
        assert np.array_equal(outs[p].reshape(O, T), ref[p])
    assert relion._stats["batches"] - b0 <= 6      # 11 particles, groups of up to 4
