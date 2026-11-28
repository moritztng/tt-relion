"""Staged hand-over (TT_RELION_STAGE): two requests are with one device server at once, each in its own segment, and
every caller gets its own batch back. The server here is a stand-in that speaks the d2 protocol without a chip."""
import subprocess
import sys
import threading
import time

import numpy as np

SERVER = """
import json, sys, time
from multiprocessing import shared_memory
import numpy as np
segs = {}
for line in sys.stdin:
    req = json.loads(line)
    name = req["shm"].lstrip("/")
    if name not in segs:
        assert req.get("staged")
        segs[name] = shared_memory.SharedMemory(name=req["shm"])
        segs[name].unlink()
    for d in req.get("drop", ()):
        segs.pop(d.lstrip("/")).close()
    assert len(segs) <= 2, len(segs)
    N, R, O, P = req["dims"]
    o = (4 if req["model_sent"] else 0) + O * 9 + 2 * P + 3 * N * P + 3 * R
    buf = np.ndarray((o + R * O,), np.float32, segs[name].buf)
    tx = buf[o - 3 * R:o - 2 * R].copy()
    time.sleep(0.15)
    buf[o:] = (tx[:, None] * 1000 + np.arange(O)[None]).reshape(-1)
    del buf
    print("ok", flush=True)
"""


def test_two_batches_are_with_one_server_and_each_gets_its_own(monkeypatch):
    from tt_relion import relion
    monkeypatch.setattr(relion, "_STAGE", True)
    s = object.__new__(relion._Server)
    s.p = subprocess.Popen([sys.executable, "-c", SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    s.lock, s.shm, s.model_key = threading.Lock(), None, None
    s.c = {"calls": 0, "bytes": 0, "rt_s": 0.0, "copy_s": 0.0}
    s.segs, s.busy, s.drop, s.sent, s.got = {}, [False, False], [], 0, 0
    s.turn = threading.Condition(threading.Lock())
    mdl = np.zeros(4, np.float32)
    got, most = {}, [0]

    def call(tag, N, R, O, P):
        rng = np.random.default_rng(tag)
        tx = np.full(R, tag, np.float32)
        a = lambda *sh: rng.standard_normal(sh).astype(np.float32)  # noqa: E731
        got[tag] = (s.d2(mdl, ["k", 4], a(O, 9), a(P), a(P), a(N, P), a(N, P), a(N, P), tx, a(R), a(R), None), R, O)

    for rnd, (N, R, O, P) in enumerate([(2, 6, 5, 40), (3, 9, 7, 400), (2, 4, 3, 30), (4, 12, 40, 3000)]):
        ths = [threading.Thread(target=call, args=(10 * rnd + j + 1, N, R + j, O, P)) for j in range(2)]
        t0 = time.time()
        for t in ths:
            t.start()
        while any(t.is_alive() for t in ths):
            most[0] = max(most[0], s.sent - s.got)
            time.sleep(0.005)
        for t in ths:
            t.join()
        assert time.time() - t0 < 0.6
        for j in range(2):
            out, R_, O_ = got[10 * rnd + j + 1]
            assert out.shape == (R_, O_) and np.array_equal(out, (10 * rnd + j + 1) * 1000.0 + np.tile(np.arange(O_), (R_, 1)))
    assert most[0] == 2                         # the second request was written before the first was answered
    assert s.sent == s.got == 8 and s.busy == [False, False] and s.c["calls"] == 8 and not s.lock.locked()
    s.p.stdin.close()
    assert s.p.wait(10) == 0


def test_a_full_batch_queues_behind_a_running_one_and_a_part_batch_waits_for_an_idle_chip(monkeypatch):
    from tt_relion import relion

    class S:
        def __init__(self, n):
            self.inflight = n
    for load, full, want in (((0, 1), False, 0), ((1, 0), True, 1), ((1, 1), False, None), ((1, 1), True, 0),
                             ((2, 1), True, 1), ((2, 1), False, None), ((2, 2), True, None)):
        srv = [S(n) for n in load]
        monkeypatch.setattr(relion, "_srv", srv)
        s = relion._idle(full)
        assert (None if s is None else srv.index(s)) == want, (load, full)


def test_threads_stage_full_batches_only(monkeypatch):
    """The collector with staging on, one chip: with many threads a full batch is cut and handed to the chip while
    the one before it still runs (two at the chip, never three); with fewer threads than a batch holds, a batch waits
    for the idle chip as without staging. Every particle gets its own rows either way."""
    import queue

    from tt_relion import relion

    class S:
        inflight, c = 0, {}
    geom = (8, 8, 5, 0, 0, 3, 9, 2, 5, 8)
    P, rounds = 40, 3
    mdl = np.zeros(8 * 8 * 5 * 2, np.float32)
    eul = np.zeros((6, 9), np.float32)
    lock = threading.Lock()

    def stub(mdl_, eul_, tx, ty, pn, img_r, img_i, corr, *g, shared=True, prior=None, own=None):
        s = relion._server()                    # the chip the leader reserved
        with lock:
            at[0] += 1
            most[0] = max(most[0], at[0], s.inflight)
        time.sleep(0.02)
        with lock:
            at[0] -= 1
        relion._done(s)
        return np.repeat(img_r[np.asarray(pn), :1], len(eul_), axis=1) + np.asarray(tx)[:, None]

    monkeypatch.setattr(relion, "diff2_coarse_batch", stub)
    monkeypatch.setattr(relion, "_BACKEND", "ttnn")
    monkeypatch.setattr(relion, "_STAGE", True)
    monkeypatch.setattr(relion, "_YIELD", 0.0)
    monkeypatch.setattr(relion, "_BATCH", 8)
    monkeypatch.setattr(relion, "_BATCH_WAIT", 0.03)
    monkeypatch.setattr(relion, "_free", queue.Queue())
    for n, want in ((48, 2), (5, 1)):
        srv, at, most, bad, good = S(), [0], [0], [], []
        monkeypatch.setattr(relion, "_srv", [srv])

        def run(p):
            for r in range(rounds):
                tx = np.arange(1 + p % 3, dtype=np.float32)
                img = np.full(P, 100.0 * p + r, np.float32)
                out = relion._collect(memoryview(mdl), memoryview(eul.ravel()), memoryview(tx), memoryview(tx.copy()),
                                      memoryview(img), memoryview(img.copy()), memoryview(np.ones(P, np.float32)), geom, 6)
                if out.shape != (tx.size, 6) or not np.array_equal(out, 100.0 * p + r + np.repeat(tx[:, None], 6, 1)):
                    bad.append((p, r))
                good.append(p)

        t0 = time.time()
        th = [threading.Thread(target=run, args=(p,)) for p in range(n)]
        [t.start() for t in th]
        [t.join() for t in th]
        assert not bad and len(good) == n * rounds and most[0] == want and srv.inflight == 0 and time.time() - t0 < 1.5, (n, most, srv.inflight)
        assert not relion._groups
