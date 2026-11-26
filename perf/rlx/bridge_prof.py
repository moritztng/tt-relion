"""Split one bridge diff2 call into its host terms, with no device: where the dev_ttnn E-step goes.

Runs the real tt_relion/relion.py _dense_diff2 (TT_RELION_BACKEND=ttnn path) against
tt_relion/stub_server.py, which speaks device_server.py's protocol and does no matmul, so every
second measured here is host work the chip waits on. Per call it books:

  project  _project: the reference projection (gather + trilinear), torch on the host
  norms    |ref|^2_w in float64
  lhs      _lhs_operand: bf16 split + K-chunk permute of the projected references
  split    _device_cross minus the transport: rhs build, float64 chunk sum
  xfer     _Server.matmul: copy into shm, pipe round trip, copy out (stub only replies)
  rest     the call wall minus the above: shift stack, norms, d2 assembly, numpy marshal

Every thread shares one reference and one orientation set (a global search: every particle's coarse
pass asks for the same set) and has its own image. --cache-mb sets TT_RELION_PROJ_CACHE_MB (0 = the
uncached bridge). --check runs each case with and without the cache against a stub that does the
matmul in fp32 and asserts the outputs are bit-identical.

--threads N runs N calls at once in one process (a rank's --j threads in RELION's embedded
interpreter), sharing --chips stub servers, and reports calls/s against 1 thread: the GIL and
memory-bandwidth contention the 9x8 layout (8 threads, 4 chips per rank) pays.

    python perf/rlx/bridge_prof.py [--threads 1,8] [--chips 4] [--calls 6] [--cache-mb 0] [--check] [--json out.json]
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict

os.environ["TT_RELION_BACKEND"] = "ttnn"
os.environ.setdefault("TT_RELION_TORCH_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tt_relion import relion  # noqa: E402

CASES = [  # name, O per call, T, size (Fourier px = size*(size/2+1)); as perf/rlx/ipc_bench.py
    ("coarse h2 s26", 1024, 21, 26),
    ("coarse h3 s50", 1024, 37, 50),
    ("fine O256 s146", 256, 84, 146),
]
PF = 2
MDL = (100, 199, 199)   # 3.96 M complex voxels = the 31.68 MB model the bridge sees at box 360

_acc = defaultdict(float)
_alock = threading.Lock()


def _timed(name, fn):
    def w(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        with _alock:
            _acc[name] += time.perf_counter() - t0
        return r
    return w


relion._project = _timed("project", relion._project)
_cross = relion._device_cross
relion._device_cross = _timed("cross", _cross)
relion._matmul = _timed("xfer", relion._matmul)
relion._lhs_operand = _timed("lhs", relion._lhs_operand)


def shared(o, rng):
    """The reference and the orientation set, one per case."""
    mdl = rng.standard_normal(MDL[0] * MDL[1] * MDL[2] * 2).astype(np.float32)
    # Random rotation matrices so the gather pattern is a real one, not a stride.
    q = rng.standard_normal((o, 4)).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    eul = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).astype(np.float32)
    return memoryview(mdl), memoryview(eul.reshape(-1))


def inputs(sh, t, s, rng):
    """One particle: translations, image, weights."""
    isz = s * (s // 2 + 1)
    f = lambda n: rng.standard_normal(n).astype(np.float32)  # noqa: E731
    return sh + (memoryview(f(t)), memoryview(f(t)), memoryview(f(isz)), memoryview(f(isz)),
                 memoryview(np.abs(f(isz))), isz)


def one(args, s):
    mdl, eul, tx, ty, re, im, corr, isz = args
    maxR = s // 2
    return relion._dense_diff2(relion._torch(), mdl, eul, tx, ty, re, im, corr,
                               MDL[0], MDL[1], MDL[2], -(MDL[1] // 2), -(MDL[2] // 2), maxR,
                               (maxR * PF) ** 2, PF, s // 2 + 1, s,
                               len(eul) // 9, len(tx), isz, "C")


def run(case, threads, calls, rng):
    name, o, t, s = case
    sh = shared(o, rng)
    data = [inputs(sh, t, s, rng) for _ in range(threads)]
    relion._pc.update(model=None)                    # cold cache per case
    one(data[0], s)                                  # warm: server spawn, shm, allocator
    one(data[0], s)                                  # second sighting: the cache (if on) fills here
    _acc.clear()
    walls = []

    def worker(i):
        for _ in range(calls):
            t0 = time.perf_counter()
            one(data[i], s)
            dt = time.perf_counter() - t0
            with _alock:
                walls.append(dt)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.perf_counter()
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    span = time.perf_counter() - t0
    n = threads * calls
    wall = sum(walls)
    per = {k: 1e3 * v / n for k, v in _acc.items()}
    r = dict(case=name, threads=threads, calls=n, span_s=round(span, 3),
             calls_per_s=round(n / span, 2), wall_ms=round(1e3 * wall / n, 1),
             project_ms=round(per.get("project", 0), 1), lhs_ms=round(per.get("lhs", 0), 1),
             xfer_ms=round(per.get("xfer", 0), 1),
             split_ms=round(per.get("cross", 0) - per.get("xfer", 0), 1),
             cache_mb=relion._PCACHE_MB, cached_mb=round(relion._pc["bytes"] / 2 ** 20))
    r["rest_ms"] = round(r["wall_ms"] - r["project_ms"] - r["lhs_ms"] - r["split_ms"] - r["xfer_ms"], 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--chips", type=int, default=4)
    ap.add_argument("--calls", type=int, default=6)
    ap.add_argument("--cache-mb", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    relion._PCACHE_MB = a.cache_mb
    if a.check:
        os.environ["RLX_STUB_MATMUL"] = "1"
    relion._srv.extend(relion._Server(None, module="tt_relion.stub_server") for _ in range(a.chips))
    import queue
    relion._free = queue.Queue()
    for sv in relion._srv:
        relion._free.put(sv)
    rng = np.random.default_rng(0)
    if a.check:
        for name, o, t, s in CASES:
            sh = shared(o, rng)
            d = [inputs(sh, t, s, rng) for _ in range(3)]
            res = {}
            for mb in (0, 4096):
                relion._PCACHE_MB = mb
                relion._pc.update(model=None)
                res[mb] = [one(x, s) for x in d]         # 3 particles: miss, fill, hit
            same = all(np.array_equal(u, v) for u, v in zip(res[0], res[4096]))
            print("check", name, "bit-identical" if same else "DIFFERS", relion.stats(), flush=True)
            assert same, name
        return
    out = []
    print("load", os.getloadavg(), "cpus", os.cpu_count(), "torch threads", relion._torch().get_num_threads(),
          flush=True)
    for case in CASES:
        for th in [int(x) for x in a.threads.split(",")]:
            r = run(case, th, a.calls, rng)
            out.append(r)
            print(json.dumps(r), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
