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


def test_window_keeps_the_samples_near_each_particles_best():
    """_window on a dense matrix: every sample within the window of its particle's best, priors included, an
    excluded orientation (infinite prior) never."""
    rng = np.random.default_rng(5)
    pn = np.repeat(np.arange(4), [3, 5, 1, 4])
    d = (rng.gamma(2.0, 20.0, (len(pn), 60)) + 900).astype(np.float32)
    lo, lt = rng.uniform(0, 4, 60), rng.uniform(0, 6, len(pn))
    lo[7] = np.inf
    r, o, dl, best = relion._window(d, lo, lt, pn, 4, 30.0)
    e = d.astype(np.float64) + lo[None] + lt[:, None]
    for p in range(4):
        rows = np.flatnonzero(pn == p)
        assert best[p] == e[rows].min()
        want = {(int(a), int(b)) for a in rows for b in np.flatnonzero(e[a] - best[p] < 30.0)}
        got = {(int(a), int(b)) for a, b in zip(r, o) if pn[a] == p}
        assert got == want and 7 not in {b for _, b in got}
    assert np.abs(dl - (e[r, o] - best[pn[r]])).max() < 1e-5 and (np.diff(r) >= 0).all()


def test_batch_candidate_form_is_the_window_of_the_dense_matrix():
    rng = np.random.default_rng(7)
    mdl, eul, tx, ty, ir, ii, corr, g, P = _case()
    n, T = tx.shape
    d = relion.diff2_coarse_batch(mdl, eul, tx.ravel(), ty.ravel(), np.repeat(np.arange(n), T), ir, ii, corr, **g)
    lo, lt = rng.uniform(0, 4, d.shape[1]), rng.uniform(0, 6, d.shape[0])
    lo[3] = np.inf
    got = relion.diff2_coarse_batch(mdl, eul, tx.ravel(), ty.ravel(), np.repeat(np.arange(n), T), ir, ii, corr, **g,
                                    prior=(lo, lt, 25.0))
    want = relion._window(d, lo, lt, np.repeat(np.arange(n), T), n, 25.0)
    assert all(np.array_equal(a, b) for a, b in zip(got, want)) and len(got[0]) > n and 3 not in got[1]


def test_threads_collect_the_candidate_form():
    """The candidate form through the collector: each thread gets its own particle's window, whatever batch it
    rode in, equal to the window of its own dense rows."""
    import threading
    mdl, eul, tx, ty, ir, ii, corr, g, P = _case()
    n, T, O = tx.shape[0], tx.shape[1], eul.shape[0]
    rng = np.random.default_rng(9)
    lo, lt = rng.uniform(0, 4, O).astype(np.float32), rng.uniform(0, 6, (n, T)).astype(np.float32)
    d = relion.diff2_coarse_batch(mdl, eul, tx.ravel(), ty.ravel(), np.repeat(np.arange(n), T), ir, ii, corr, **g)
    geom = tuple(g[k] for k in ("mdlX", "mdlY", "mdlZ", "mdlInitY", "mdlInitZ", "maxR", "maxR2_padded",
                                "padding_factor", "imgX", "imgY"))
    outs = [None] * n
    relion._BATCH, relion._BATCH_WAIT = 4, 0.2

    def run(p):
        outs[p] = relion._collect(memoryview(mdl), memoryview(eul.ravel()), memoryview(tx[p].copy()),
                                  memoryview(ty[p].copy()), memoryview(ir[p].copy()), memoryview(ii[p].copy()),
                                  memoryview(corr[p].copy()), geom, O, prior=(lo, lt[p], 25.0))
    try:
        th = [threading.Thread(target=run, args=(p,)) for p in range(n)]
        [t.start() for t in th]
        [t.join() for t in th]
    finally:
        relion._BATCH = 0
    for p in range(n):
        r, o, dl, best = relion._window(d[p * T:(p + 1) * T], lo, lt[p], np.zeros(T, np.int64), 1, 25.0)
        assert np.array_equal(outs[p][0], o) and np.array_equal(outs[p][1], r)
        assert np.abs(outs[p][2] - dl).max() < 1e-4 and abs(outs[p][3] - best[0]) < 1e-6 * abs(best[0])


def test_relion_entry_writes_the_candidates_as_flat_indices():
    """diff2_coarse_cand, the entry RELION calls: idx = orientation * T + translation in ascending order, val the
    distance to the particle's best, and a decline when the window is off or idx is too short. Particles with their
    own orientation prior ride one batch and still get their own window."""
    import threading
    mdl, eul, tx, ty, ir, ii, corr, g, P = _case()
    n, T, O = tx.shape[0], tx.shape[1], eul.shape[0]
    rng = np.random.default_rng(11)
    lo = rng.uniform(0, 4, (n, O)).astype(np.float32)
    lo[:, 3] = np.inf                                   # an orientation the prior excludes
    lo[: n // 2] = lo[0]                                # half share one prior, the rest carry their own
    lt = rng.uniform(0, 6, (n, T)).astype(np.float32)
    d = relion.diff2_coarse_batch(mdl, eul, tx.ravel(), ty.ravel(), np.repeat(np.arange(n), T), ir, ii, corr, **g)
    geom = [g[k] for k in ("mdlX", "mdlY", "mdlZ", "mdlInitY", "mdlInitZ", "maxR", "maxR2_padded",
                           "padding_factor", "imgX", "imgY")]
    idx, val, ret = np.zeros((n, O * T), np.uint64), np.zeros((n, O * T), np.float32), [None] * n

    def run(p, cap=O * T):
        ret[p] = relion.diff2_coarse_cand(memoryview(mdl), memoryview(eul.ravel()), memoryview(tx[p].copy()),
                                          memoryview(ty[p].copy()), memoryview(ir[p].copy()),
                                          memoryview(ii[p].copy()), memoryview(corr[p].copy()),
                                          memoryview(lo[p].copy()), memoryview(lt[p].copy()),
                                          memoryview(idx[p])[:cap], memoryview(val[p])[:cap],
                                          *geom, O, T, ir.shape[1])
    assert relion._CAND_WINDOW == 0
    relion._BATCH, relion._BATCH_WAIT = 4, 0.2
    try:
        run(0)
        assert ret[0] is None                           # off: RELION runs the dense pass
        relion._CAND_WINDOW = 25.0
        th = [threading.Thread(target=run, args=(p,)) for p in range(n)]
        [t.start() for t in th]
        [t.join() for t in th]
        for p in range(n):
            e = d[p * T:(p + 1) * T].T.astype(np.float64) + lo[p].astype(np.float64)[:, None] + lt[p][None, :]
            want = np.flatnonzero((e - e.min() < 25.0).ravel())             # [O, T] flat, RELION's block order
            k, best = ret[p]
            assert k == len(want) and np.array_equal(idx[p, :k], want)
            assert np.abs(val[p, :k] - (e.ravel()[want] - e.min())).max() < 1e-4
            assert abs(best - e.min()) < 1e-6 * abs(e.min())
            assert not np.any(idx[p, :k] // T == 3)
        k = ret[0][0]
        run(0, cap=k - 1)
        assert ret[0] is None                           # more candidates than idx holds
    finally:
        relion._BATCH, relion._CAND_WINDOW = 0, 0.0


def test_rows_cut_does_not_open_a_tile_for_a_remainder():
    """With a rows cut, a batch whose last particles would open a mostly empty 512-row tile is trimmed to whole tiles."""
    q = lambda n, t: [{"tx": np.zeros(t, np.float32)} for _ in range(n)]     # noqa: E731
    keep = relion._BATCH, relion._BATCH_ROWS
    try:
        relion._BATCH, relion._BATCH_ROWS = 32, 1024
        assert relion._take(q(14, 37)) == (13, True)        # 518 rows: 13 fill one tile (481), one particle waits
        assert relion._take(q(13, 37)) == (13, False)       # 481 rows: one tile, still collecting
        assert relion._take(q(23, 37)) == (23, False)       # 851 of 1024: full enough to keep
        assert relion._take(q(40, 37)) == (27, True)        # the rows cut itself: 999 of 1024
        assert relion._take(q(17, 37)) == (13, True)        # 629 rows of 1024 would be 61 %
        assert relion._take(q(1, 600)) == (1, False)        # one particle over a tile is taken as it is
        assert relion._take(q(5, 9)) == (5, False)
        relion._BATCH_ROWS = 0
        assert relion._take(q(14, 37)) == (14, False)       # no rows cut, no trim
        assert relion._take(q(40, 37)) == (32, True)
    finally:
        relion._BATCH, relion._BATCH_ROWS = keep


def test_own_orientation_sets_take_their_own_rows_cut():
    """TT_RELION_OWN_ROWS cuts a batch of particles with different orientation sets; a shared set keeps the batch cut."""
    q = lambda n, t, own: [{"tx": np.zeros(t, np.float32), "eh": i if own else 0} for i in range(n)]     # noqa: E731
    keep = relion._BATCH, relion._BATCH_ROWS, relion._OWN_ROWS
    try:
        relion._BATCH, relion._BATCH_ROWS, relion._OWN_ROWS = 96, 3328, 512
        assert relion._take(q(96, 9, True)) == (56, True)       # 504 rows: one tile
        assert relion._take(q(40, 9, True)) == (40, False)
        assert relion._take(q(96, 37, False)) == (89, True)     # the shared set fills 3,293 of 3,328 rows
        relion._OWN_ROWS = 0
        assert relion._take(q(96, 9, True)) == (96, True)
    finally:
        relion._BATCH, relion._BATCH_ROWS, relion._OWN_ROWS = keep


def test_small_orientation_sets_stay_with_relion():
    """TT_RELION_MIN_ORI: a coarse call under the threshold is declined untouched, and so is the fine call after it
    on the same thread; a call at or over the threshold is handled, and its fine call too."""
    mdl, eul, tx, ty, ir, ii, corr, g, P = _case(seed=2, n=2, T=24)
    T, O = tx.shape[1], eul.shape[0]
    sc = (g["mdlX"], g["mdlY"], g["mdlZ"], g["mdlInitY"], g["mdlInitZ"], g["maxR"], g["maxR2_padded"],
          g["padding_factor"], g["imgX"], g["imgY"])
    mv = lambda p: [memoryview(v) for v in (mdl, eul.ravel(), tx[p], ty[p], ir[p], ii[p], corr[p])]  # noqa: E731
    rot, trn = np.arange(8, dtype=np.uint64), np.arange(8, dtype=np.uint64)
    fine = lambda out: relion.diff2_fine(*mv(0), memoryview(rot), memoryview(trn), memoryview(out), *sc, 0.5,  # noqa: E731
                                         O, T, 8, P, 8)
    keep, n0 = relion._MIN_ORI, relion._stats.get("declined_small", 0)
    try:
        relion._MIN_ORI = O + 1
        out, fo = np.zeros(O * T, np.float32), np.zeros(8, np.float32)
        assert relion.diff2_coarse(*mv(0), memoryview(out), *sc, O, T, P) is False and not out.any()
        assert fine(fo) is False and not fo.any()
        assert relion._stats["declined_small"] == n0 + 2
        relion._MIN_ORI = O
        assert relion.diff2_coarse(*mv(0), memoryview(out), *sc, O, T, P) is True and out.any()
        assert fine(fo) is True and fo.any()
        assert relion._stats["declined_small"] == n0 + 2
    finally:
        relion._MIN_ORI = keep
        relion._tl.__dict__.pop("small", None)
