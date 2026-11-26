"""Price the bridge <-> device-server round trip with no device: the IPC floor for the WALLMAP.

The bridge (tt_relion/relion.py) talks to one server process per chip over a pipe (one JSON line each way)
and a shared-memory segment holding lhs [nc, O, kc], rhs [nc, kc, 3T] and out [nc, O, 3T], float32. This
runs the real bridge code (_Server.matmul and _device_cross) against tt_relion/stub_server.py, which speaks
the same protocol and does no matmul, and reports per call:

  rt     pipe round trip with a 32x32 payload: latency floor of one call
  xfer   _Server.matmul at the real shape: copy into shm + round trip + copy out (stub only replies)
  xferc  as xfer, stub also reads every input and writes every output once (the host traffic the real
         server's from_torch / to_torch cannot avoid)
  cross  the whole _device_cross: bf16 split, operand build, K-chunk permute, xfer, float64 chunk sum

Shapes follow box 360 at the dev cell's operating points (dev_ref refine.log): coarse O = 1024 per call
(the bridge's TT_RELION_DEV_CHUNK), T = 21 at healpix 2 / 37 at healpix 3; fine T = 84 (oversampling 1),
O = the significant orientations, which only a run can tell, so a range. image_size = Fourier pixels,
size x (size/2 + 1).

    python perf/rlx/ipc_bench.py [--reps 5] [--pairs 1] [--json out.json]
--pairs N runs N bridge/server pairs at once (threads in one process, as a rank's --j threads do), to see
the host memory bandwidth the chips would share.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tt_relion import relion  # noqa: E402


def sizes(n):
    return n * (n // 2 + 1)


CASES = [  # name, O per call, T, image size (Fourier pixels)
    ("coarse h2 s64", 1024, 21, sizes(64)),
    ("coarse h2 s146", 1024, 21, sizes(146)),
    ("coarse h3 s146", 1024, 37, sizes(146)),
    ("coarse h3 s256", 1024, 37, sizes(256)),
    ("fine O64 s146", 64, 84, sizes(146)),
    ("fine O256 s146", 256, 84, sizes(146)),
    ("fine O256 s256", 256, 84, sizes(256)),
    ("fine O256 s360", 256, 84, sizes(360)),
]


def shapes(o, t, isz):
    k = relion._SPLIT
    kk = k * 2 * isz
    kc = relion._KCHUNK
    nc = -(-kk // kc)
    return nc, o, kc, k * t


def timed(f, reps):
    f()                                           # warm: shm create, page faults
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def run(pairs, reps, copy, max_mb):
    os.environ["RLX_STUB_COPY"] = "1" if copy else "0"
    srv = [relion._Server(module="tt_relion.stub_server") for _ in range(pairs)]
    out = {}
    try:
        def par(fn):
            res = [None] * pairs

            def go(i):
                res[i] = fn(i)
            th = [threading.Thread(target=go, args=(i,)) for i in range(pairs)]
            [x.start() for x in th]
            [x.join() for x in th]
            return statistics.median(res)

        a = np.zeros((1, 32, 32), np.float32)
        out["rt"] = par(lambda i: timed(lambda: srv[i].matmul(a, a), reps * 200))
        import torch
        g = torch.Generator().manual_seed(0)
        for name, o, t, isz in CASES:
            nc, m, kc, n = shapes(o, t, isz)
            if 4 * (nc * m * kc + nc * kc * n + nc * m * n) / 1e6 > max_mb:
                continue
            lhs = np.random.default_rng(0).standard_normal((nc, m, kc), np.float32)
            rhs = np.random.default_rng(1).standard_normal((nc, kc, n), np.float32)
            mb = 4 * (lhs.size + rhs.size + nc * m * n) / 1e6
            xfer = par(lambda i: timed(lambda: srv[i].matmul(lhs, rhs), reps))
            del lhs, rhs
            ref = torch.randn(o, 2 * isz, generator=g)
            shw = torch.randn(t, 2 * isz, generator=g)

            # _device_cross goes through relion._matmul's queue; point it at this thread's server.
            def cross(i):
                relion._free = _One(srv[i])
                return relion._device_cross(torch, ref, shw)
            if pairs == 1:
                xc = timed(lambda: cross(0), reps)
            else:
                xc = float("nan")                 # _free is module-global; the pair arm prices xfer only
            out[name] = {"O": o, "T": t, "image_size": isz, "shape": [nc, m, kc, n], "MB": round(mb, 1),
                         "xfer_s": xfer, "cross_s": xc}
    finally:
        for s in srv:
            s.p.stdin.close()
            s.p.wait()
            if s.shm is not None:
                s.shm.close()
                s.shm.unlink()
        relion._free = None
    return out


class _One:
    def __init__(self, s):
        self.s = s

    def get(self):
        return self.s

    def put(self, s):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--json")
    ap.add_argument("--max-mb", type=float, default=1e9, help="skip cases whose shm segment is larger")
    a = ap.parse_args()
    os.environ.setdefault("TT_RELION_PYTHON", sys.executable)
    relion._PYTHON = os.environ["TT_RELION_PYTHON"]
    big = np.ones(64 << 20, np.float32)           # 256 MB, plain single-thread copy: the bandwidth to scale by
    dst = np.empty_like(big)
    bw = 2 * big.nbytes / 1e9 / timed(lambda: np.copyto(dst, big), a.reps)   # read + write
    del big, dst
    res = {"host": {"nproc": os.cpu_count(), "load1": os.getloadavg()[0], "memcpy_GBps": bw}, "pairs": a.pairs}
    for copy in (False, True):
        res["copy" if copy else "nocopy"] = run(a.pairs, a.reps, copy, a.max_mb)
    r0, r1 = res["nocopy"], res["copy"]
    print("pairs=%d nproc=%d load1=%.1f memcpy %.1f GB/s (r+w)  rt (32x32) %.1f us"
          % (a.pairs, os.cpu_count(), os.getloadavg()[0], bw, 1e6 * r0["rt"]))
    print("%-16s %18s %8s %9s %9s %9s %8s" % ("case", "nc,O,kc,n", "MB", "xfer ms", "xferc ms", "cross ms",
                                               "GB/s"))
    for name, *_ in CASES:
        if name not in r0:
            continue
        c, d = r0[name], r1[name]
        print("%-16s %18s %8.1f %9.2f %9.2f %9.2f %8.2f" % (name, ",".join(map(str, c["shape"])), c["MB"],
              1e3 * c["xfer_s"], 1e3 * d["xfer_s"], 1e3 * c["cross_s"], c["MB"] / 1e3 / c["xfer_s"]))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
