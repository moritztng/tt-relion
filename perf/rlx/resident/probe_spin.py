"""What does waiting on the chip cost the host? A device server showed one thread at ~47 % of a core, user time only,
through a RELION arm. Here: n matmuls queued back to back (about 0.6 s of chip work), then
  wait:  synchronize at once (what Resident._mark does),
  sleep: sleep 90 % of the chip time first, then synchronize,
  read:  to_torch of the last product at once,  sleep_read: sleep first, then to_torch.
Per variant: wall, CPU seconds of the whole process, of the main thread, and of the busiest other thread."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
sys.path.insert(0, __file__.rsplit("/", 4)[0])
from _tt import open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, nodes = open_dev()
TICK = os.sysconf("SC_CLK_TCK")


def threads():
    out = {}
    for t in os.listdir("/proc/self/task"):
        try:
            f = open("/proc/self/task/%s/stat" % t).read().rsplit(") ", 1)[1].split()
            out[t] = (int(f[11]) + int(f[12])) / TICK
        except OSError:
            pass
    return out


a = ttnn.from_torch(torch.randn(2048, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
b = ttnn.from_torch(torch.randn(4096, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def queue(n):
    return [ttnn.matmul(a, b) for _ in range(n)]


def free(cs):
    for c in cs:
        ttnn.deallocate(c)


free(queue(2))
ttnn.synchronize_device(dev)
t0 = time.perf_counter()
cs = queue(8)
ttnn.synchronize_device(dev)
one = (time.perf_counter() - t0) / 8
free(cs)
n = max(2, min(400, int(round(0.6 / one))))
out = dict(load1=float(open("/proc/loadavg").read().split()[0]), s_per_matmul=round(one, 5), n=n)
chip = None
for name in ("wait", "sleep", "read", "sleep_read", "wait", "sleep"):
    recs = []
    for _ in range(5):
        th0, p0, m0, t0 = threads(), time.process_time(), time.thread_time(), time.perf_counter()
        cs = queue(n)
        tq = time.perf_counter() - t0
        if name.startswith("sleep") and chip is not None:
            time.sleep(max(0.0, 0.9 * chip - tq))
        if name.endswith("read"):
            ttnn.to_torch(cs[-1])
        else:
            ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        th1 = threads()
        me = str(os.getpid())
        other = max((th1[k] - th0.get(k, 0.0) for k in th1 if k != me), default=0.0)
        recs.append(dict(wall=wall, queue=tq, cpu=time.process_time() - p0, main=time.thread_time() - m0, other=other))
        free(cs)
    if name == "wait" and chip is None:
        chip = min(r["wall"] for r in recs)
    key = name if name not in out else name + "_again"
    out[key] = {k: round(sorted(r[k] for r in recs)[2], 4) for k in recs[0]}        # medians of 5
print(json.dumps(out), flush=True)
if len(sys.argv) > 1:
    open(sys.argv[1], "a").write(json.dumps(out) + "\n")
ttnn.close_device(dev)
