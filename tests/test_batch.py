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
    got = relion.diff2_coarse_batch(mdl, eul, tx.ravel(), ty.ravel(), np.repeat(np.arange(n), T), ir, ii, corr, **g)
    got = got.reshape(n, T, O).transpose(0, 2, 1)
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
    ref = relion.diff2_coarse_batch(mdl, eul, tx.ravel(), ty.ravel(), np.repeat(np.arange(n), T), ir, ii, corr, **g)
    ref = ref.reshape(n, T, O).transpose(0, 2, 1)
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


def test_threads_with_own_orientation_sets():
    """Coarse and fine calls whose orientation and translation counts differ per particle (local searches, the
    fine pass) are batched over the union of the sets and each gets its own block back."""
    import threading
    mdl, eul, tx, ty, ir, ii, corr, g, P = _case(seed=2, n=9, O=60)
    n = tx.shape[0]
    geo = (g["mdlX"], g["mdlY"], g["mdlZ"], g["mdlInitY"], g["mdlInitZ"], g["maxR"], g["maxR2_padded"],
           g["padding_factor"], g["imgX"], g["imgY"])
    own = [eul[3 * p:3 * p + 5 + p].copy() for p in range(n)]          # 5..13 orientations each, overlapping
    nt = [3 + p % 4 for p in range(n)]                                  # 3..6 translations each

    def call(p, out):
        O, T = len(own[p]), nt[p]
        if p % 2:                                                       # odd particles through the fine entry
            rot = np.repeat(np.arange(O, dtype=np.uint64), T)
            tr = np.tile(np.arange(T, dtype=np.uint64), O)
            return relion.diff2_fine(memoryview(mdl), memoryview(own[p].ravel()), memoryview(tx[p, :T].copy()),
                                     memoryview(ty[p, :T].copy()), memoryview(ir[p]), memoryview(ii[p]),
                                     memoryview(corr[p]), memoryview(rot), memoryview(tr), memoryview(out),
                                     *geo, 0.0, O, T, O * T, P, O)
        return relion.diff2_coarse(memoryview(mdl), memoryview(own[p].ravel()), memoryview(tx[p, :T].copy()),
                                   memoryview(ty[p, :T].copy()), memoryview(ir[p]), memoryview(ii[p]),
                                   memoryview(corr[p]), memoryview(out), *geo, O, T, P)

    ref = [np.zeros(len(own[p]) * nt[p], np.float32) for p in range(n)]
    assert all(call(p, ref[p]) for p in range(n))
    relion._BATCH, relion._BATCH_WAIT = 4, 0.2
    outs = [np.zeros_like(r) for r in ref]
    ok = [False] * n
    b0 = relion._stats.get("batches", 0)

    def run(p):
        ok[p] = call(p, outs[p])
    th = [threading.Thread(target=run, args=(p,)) for p in range(n)]
    [x.start() for x in th]
    [x.join(60) for x in th]
    relion._BATCH = 0
    assert all(ok)
    assert 0 < relion._stats["batches"] - b0 <= 5
    for p in range(n):
        rel = np.abs(outs[p] - ref[p]).max() / np.abs(ref[p]).max()
        assert rel < 1e-5, (p, rel)
        assert outs[p].argmin() == ref[p].argmin()


def test_fine_scores_only_the_asked_translations():
    """diff2_fine equals the requested entries of the dense matrix, alone and collected into batches."""
    import threading
    import torch
    mdl, eul, tx, ty, ir, ii, corr, g, P = _case(seed=2, n=6, T=24)
    n, T, O = tx.shape[0], tx.shape[1], eul.shape[0]
    sc = (g["mdlX"], g["mdlY"], g["mdlZ"], g["mdlInitY"], g["mdlInitZ"], g["maxR"], g["maxR2_padded"],
          g["padding_factor"], g["imgX"], g["imgY"])
    rng = np.random.default_rng(5)
    rot = [np.sort(rng.integers(0, O, 30)).astype(np.uint64) for _ in range(n)]
    trn = [rng.choice(rng.choice(T, 2 + p, replace=False), 30).astype(np.uint64) for p in range(n)]
    mv = lambda p: [memoryview(v) for v in (mdl, eul.ravel(), tx[p], ty[p], ir[p], ii[p], corr[p])]  # noqa: E731
    ref = [relion._dense_diff2(torch, *mv(p), *sc, O, T, P, "F")[rot[p] * np.uint64(T) + trn[p]] + np.float32(0.5)
           for p in range(n)]
    outs = [np.zeros(30, np.float32) for _ in range(n)]

    def run(p):
        assert relion.diff2_fine(*mv(p), memoryview(rot[p]), memoryview(trn[p]), memoryview(outs[p]), *sc, 0.5,
                                 O, T, 30, P, 30)
    for p in range(n):
        run(p)
        assert np.array_equal(outs[p], ref[p])
        outs[p][:] = 0
    relion._BATCH, relion._BATCH_WAIT = 4, 0.2
    th = [threading.Thread(target=run, args=(p,)) for p in range(n)]
    [x.start() for x in th]
    [x.join(60) for x in th]
    relion._BATCH = 0
    for p in range(n):
        assert np.abs(outs[p] - ref[p]).max() / np.abs(ref[p]).max() < 1e-5


def test_batch_is_cut_by_rows():
    """TT_RELION_BATCH_ROWS cuts a batch by its (particle, translation) rows, whatever each particle brings, and
    every particle still gets exactly its own block."""
    import threading
    mdl, eul, tx, ty, ir, ii, corr, g, P = _case(seed=3, n=10)
    n, O = tx.shape[0], eul.shape[0]
    nt = [2 + p % 6 for p in range(n)]                                  # 2..7 translations each, 41 rows in all
    geo = (g["mdlX"], g["mdlY"], g["mdlZ"], g["mdlInitY"], g["mdlInitZ"], g["maxR"], g["maxR2_padded"],
           g["padding_factor"], g["imgX"], g["imgY"])

    def run(p, out):
        return relion.diff2_coarse(memoryview(mdl), memoryview(eul.ravel()), memoryview(tx[p, :nt[p]].copy()),
                                   memoryview(ty[p, :nt[p]].copy()), memoryview(ir[p]), memoryview(ii[p]),
                                   memoryview(corr[p]), memoryview(out), *geo, O, nt[p], P)
    refs = [np.zeros(O * nt[p], np.float32) for p in range(n)]
    assert all(run(p, refs[p]) for p in range(n))
    outs = [np.zeros(O * nt[p], np.float32) for p in range(n)]
    rows = []
    inner = relion.diff2_coarse_batch
    relion.diff2_coarse_batch = lambda mdl_, eul_, tx_, *a, **k: (rows.append(len(tx_)), inner(mdl_, eul_, tx_, *a, **k))[1]
    relion._BATCH, relion._BATCH_ROWS, relion._BATCH_WAIT = 64, 16, 0.5
    try:
        th = [threading.Thread(target=run, args=(p, outs[p])) for p in range(n)]
        [x.start() for x in th]
        [x.join(60) for x in th]
    finally:
        relion._BATCH, relion._BATCH_ROWS, relion.diff2_coarse_batch = 0, 0, inner
    assert sum(rows) == sum(nt) and max(rows) <= 16 and len(rows) >= 3, rows
    for p in range(n):
        rel = np.abs(outs[p] - refs[p]).max() / np.abs(refs[p]).max()
        assert rel < 1e-5, (p, rel)
        assert outs[p].argmin() == refs[p].argmin()
