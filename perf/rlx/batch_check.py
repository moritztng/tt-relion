"""TT_RELION_BATCH through the bridge's own _cross: J threads each make M coarse calls on one kept reference
chunk (one server, as a BH rank with one chip has). Every result must equal, bit for bit, the same call made
alone; the wall of the J x M calls is printed per batch size. Stub server (numpy) by default, --chip N = device.
    TT_RELION_PYTHON=<python> TT_RELION_OZAKI=14 python batch_check.py --s 146 --O 1024 --T 37 [--chip N]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

os.environ["RLX_STUB_MATMUL"] = "1"
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))

import numpy as np  # noqa: E402

from tt_relion import relion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--s", type=int, default=50)
ap.add_argument("--O", type=int, default=256)
ap.add_argument("--T", type=int, default=37)
ap.add_argument("--J", type=int, default=7)
ap.add_argument("--M", type=int, default=6)
ap.add_argument("--batches", default="1,4,8")
ap.add_argument("--chip", default=None)
ap.add_argument("--reps", type=int, default=2, help="timed passes per batch size; the first pays JIT for new T shapes")
a = ap.parse_args()

g = np.random.default_rng(0)
imgX = a.s // 2 + 1
k = 2 * imgX * a.s
ref = (g.standard_normal((a.O, k)) * 3).astype(np.float32)
shws = [(g.standard_normal((a.T, k)) * (1 + j % 3)).astype(np.float32) for j in range(a.J * a.M)]

srv = relion._Server(a.chip, "tt_relion.device_server" if a.chip is not None else "tt_relion.stub_server")
relion._srv.append(srv)
key = "m-e-0"
alone = [srv.cross(lambda: ref, a.O, x, key, relion._pc["gen"], 0) for x in shws]   # first call keeps ref

clk = None
if a.chip is not None:
    sys.path.insert(0, here)
    from _tt import Clock   # perf/rlx/_tt.py on wk/rlx-kernels
    clk = Clock([int(subprocess.check_output([os.path.join(here, "ttvd_nodes.sh"), str(a.chip)], text=True))])
for b, rep in [(b, r) for b in map(int, a.batches.split(",")) for r in range(a.reps)]:
    relion._BATCH = b
    got = [None] * len(shws)

    def work(j):
        for m in range(a.M):
            i = j * a.M + m
            got[i] = relion._cross(lambda: ref, a.O, shws[i], key, 0)
    r0, c0 = relion._stats["batch_req"], relion._stats["batch_calls"]
    t0 = time.time()
    th = [threading.Thread(target=work, args=(j,)) for j in range(a.J)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    same = all(np.array_equal(x, y) for x, y in zip(got, alone))
    print(json.dumps(dict(s=a.s, O=a.O, T=a.T, J=a.J, M=a.M, batch=b, rep=rep, ozaki=relion._OZAKI, chip=a.chip,
                          bit_identical=same, wall_s=round(wall, 3), per_call_s=round(wall / len(shws), 4),
                          requests=relion._stats["batch_req"] - r0 if b > 1 else len(shws),
                          calls=relion._stats["batch_calls"] - c0 if b > 1 else len(shws))), flush=True)
if clk:
    print(json.dumps(dict(aiclk_min_med=clk.stop())), flush=True)
srv.p.stdin.close()
srv.p.wait()
