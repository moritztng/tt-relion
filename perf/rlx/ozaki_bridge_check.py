"""The bridge's cross term with TT_RELION_OZAKI on vs off: d2 error vs float64 and warm s/call, through the real
server loop (tt_relion/cross.py). Stub server (numpy, exact digit sums) by default; --chip N uses the device server.
    TT_RELION_PYTHON=<python> python ozaki_bridge_check.py --s 146 --O 256 --T 84 [--chip N] [--ns 0,14]
sha (of C) compares two runs bitwise.
TT_RELION_PHASES=1 in the environment adds the device server's per-phase split (TTPHASES line on stderr).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ["RLX_STUB_MATMUL"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402

from tt_relion import relion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--s", type=int, default=50)
ap.add_argument("--O", type=int, default=64)
ap.add_argument("--T", type=int, default=21)
ap.add_argument("--chip", default=None)
ap.add_argument("--ns", default="0,14")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

# whitened Fourier data as perf/rlx/diff2_parity360.py (wk/rlx-kernels) builds it
g = np.random.default_rng(0)
imgX = a.s // 2 + 1
pix = np.arange(imgX * a.s)
x, yi = pix % imgX, pix // imgX
y = np.where(yi > a.s // 2, yi - a.s, yi)
inside = (x * x + y * y) <= (a.s // 2) ** 2
ref = (g.standard_normal((a.O, 2 * len(pix))) * 3 * np.tile(inside, 2)).astype(np.float32)
sh = (ref[np.arange(a.T) % a.O] + g.standard_normal((a.T, 2 * len(pix)))).astype(np.float32)
w = 0.5
shw = (sh * np.float32(w)).astype(np.float32)
C64 = ref.astype(np.float64) @ shw.astype(np.float64).T
ref2, sh2 = (ref.astype(np.float64) ** 2).sum(1) * w, (sh.astype(np.float64) ** 2).sum(1) * w
d64 = ref2[:, None] + sh2[None, :] - 2 * C64
near = (d64 - d64.min()) < 30

clk = None
if a.chip is not None:
    from _tt import Clock   # perf/rlx/_tt.py on wk/rlx-kernels; optional
for n in map(int, a.ns.split(",")):
    os.environ["TT_RELION_OZAKI"] = str(n)
    srv = relion._Server(a.chip, "tt_relion.device_server" if a.chip is not None else "tt_relion.stub_server")
    tim = []
    for rep in range(a.reps + 1):
        if rep == 1 and a.chip is not None:
            node = subprocess.check_output([os.path.join(os.path.dirname(__file__), "ttvd_nodes.sh"), str(a.chip)], text=True)
            clk = Clock([int(node)])   # the chip id is a UMD (PCI-order) id, the clock lives on its device node
        t0 = time.time()
        C = srv.cross(lambda: ref, a.O, shw, "k", 1, 0)     # rep 0 ships and keeps the reference
        tim.append(time.time() - t0)
    cl = clk.stop() if clk else (None, None)
    srv.p.stdin.close()
    srv.p.wait()
    srv.shm.close(), srv.shm.unlink()
    e = np.abs(ref2[:, None] + sh2[None, :] - 2 * C - d64)
    print(json.dumps(dict(s=a.s, O=a.O, T=a.T, ozaki=n, chip=a.chip, err_max=float(e.max()),
                          err_near=float(e[near].max()), sha=hashlib.sha1(np.ascontiguousarray(C).tobytes()).hexdigest()[:12],
                          first_s=round(tim[0], 4),
                          warm_s=round(float(np.median(tim[1:])), 4), aiclk_min=cl[0], aiclk_med=cl[1])), flush=True)
