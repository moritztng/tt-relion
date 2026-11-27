"""d2 readback at the prod shape [3328, 4096] fp32: host untilize (to_torch of TILE) vs device untilize then a
row-major read; bit-exactness of the device untilize. TT_VISIBLE_DEVICES=16 python readback_probe.py"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from _tt import Clock, open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(7)
D = (rng.standard_normal((3328, 4096)) * 1000 + 900).astype(np.float32)
t = ttnn.from_torch(torch.from_numpy(D), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
out = {}
for name, f in (("host_untilize", lambda: ttnn.to_torch(t)),
                ("device_untilize", lambda: ttnn.to_torch(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT))),
                ("device_untilize_only", lambda: (ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), ttnn.synchronize_device(dev)))):
    r = f()
    if name != "device_untilize_only":
        out[name + "_bitexact"] = bool((r.numpy() == D).all())
    clk = Clock(nodes); t0 = time.perf_counter()
    for _ in range(3):
        f()
    out[name + "_ms"] = round((time.perf_counter() - t0) / 3 * 1e3, 1)
    out[name + "_aiclk"] = clk.stop()
out["load1"] = open("/proc/loadavg").read().split()[0]
print(json.dumps(out), flush=True)
