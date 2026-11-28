# Host only, no chip. A batch leader stacks and "sends" a batch (numpy calls that release the interpreter lock) while the
# 57 threads of the batch before take their results. Wall and CPU of the leader per batch, TT_RELION_YIELD_MS off / on.
import os
import sys
import threading
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from tt_relion import relion

def run(yield_ms, rounds=30, n=57):
    relion._YIELD = yield_ms / 1e3
    relion._staging[0] = 0
    rows = [np.random.rand(3, 9922).astype(np.float32) for _ in range(n)]
    go = [threading.Event() for _ in range(n)]
    stop = []
    def worker(i):
        while True:
            go[i].wait(); go[i].clear()
            if stop: return
            relion._stand_back()
            t = time.thread_time()
            while time.thread_time() - t < 0.002:          # ~2 ms of result handling, mostly holding the lock
                sum(range(2000)); np.argsort(rows[i][0][:2000])
    th = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in th: t.start()
    wall = cpu = 0.0
    for _ in range(rounds):
        relion._tl.staging = True; relion._stage(1) if yield_ms else None
        for e in go: e.set()                                 # the batch before comes back
        t0, c0 = time.time(), time.thread_time()
        for _k in range(10):                                 # the leader: ~30 calls that let go of the lock
            a = np.stack([r[0] for r in rows]); b = np.stack([r[1] for r in rows]); c = np.stack([r[2] for r in rows]) * 0.5
        wall += time.time() - t0; cpu += time.thread_time() - c0
        if yield_ms: relion._staged()
        else: relion._tl.__dict__.pop("staging", None)
        time.sleep(0.25)
    stop.append(1)
    for e in go: e.set()
    for t in th: t.join()
    return wall / rounds * 1e3, cpu / rounds * 1e3

for y in (0, 250, 0, 250):
    w, c = run(y)
    print("yield_ms %3d: leader wall %.1f ms, leader cpu %.1f ms a batch" % (y, w, c))
