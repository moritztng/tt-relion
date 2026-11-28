"""Does the main thread hold the GIL while it waits on the chip? It decides how the device server can build batch
k + 1's image operand while the chip runs batch k: on a thread (if ttnn lets go of the GIL in its blocking calls) or
only in a second process (if it does not). A pure-Python counter thread and a numpy thread (complex products, as
Resident._operands does) run while the main thread sleeps, loops matmul + synchronize, loops to_torch, loops
from_torch. Counts per second of each; a phase whose counts stay near the sleeping phase's lets the threads run."""
import json
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
sys.path.insert(0, __file__.rsplit("/", 4)[0])
from _tt import open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, nodes = open_dev()
stop, n = [False], [0, 0]


def spin():
    while not stop[0]:
        n[0] += 1


def npw():
    z = (np.random.default_rng(0).standard_normal((128, 16384)) + 1j).astype(np.complex128)
    while not stop[0]:
        (z * z).real.astype(np.float32)
        n[1] += 1


a = ttnn.from_torch(torch.randn(2048, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
b = ttnn.from_torch(torch.randn(4096, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
c0 = ttnn.matmul(a, b)
h = torch.randn(1024, 16384)


def mm():
    c = ttnn.matmul(a, b)
    ttnn.synchronize_device(dev)
    ttnn.deallocate(c)


def up():
    t = ttnn.from_torch(h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.synchronize_device(dev)
    ttnn.deallocate(t)


phases = (("sleep", lambda: time.sleep(0.05)), ("matmul_sync", mm), ("to_torch", lambda: ttnn.to_torch(c0)),
          ("from_torch", up), ("sleep_again", lambda: time.sleep(0.05)))
for _, f in phases:
    f()                                 # programs
out = dict(load1=float(open("/proc/loadavg").read().split()[0]))
for who, target in (("py", spin), ("np", npw)):     # one helper thread at a time: they would hold each other up
    stop[0] = False
    th = threading.Thread(target=target, daemon=True)
    th.start()
    for name, f in phases:
        k0, t0, calls = list(n), time.perf_counter(), 0
        while time.perf_counter() - t0 < 3.0:
            f()
            calls += 1
        dt = time.perf_counter() - t0
        out.setdefault(name, {}).update({"calls": calls, "s_per_call": round(dt / calls, 5),
                                         who + "_per_s": round((n[who == "np"] - k0[who == "np"]) / dt, 1)})
    stop[0] = True
    th.join()
for name, _ in phases[1:4]:
    for who in ("py", "np"):
        out[name][who + "_vs_sleep"] = round(out[name][who + "_per_s"] / max(out["sleep"][who + "_per_s"], 1e-9), 3)
print(json.dumps(out), flush=True)
if len(sys.argv) > 1:
    open(sys.argv[1], "a").write(json.dumps(out) + "\n")
ttnn.close_device(dev)
