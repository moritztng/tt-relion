"""What one chip's model table costs the host: seconds and resident memory of Resident.set_model at an image size.

At full image size the corner table is 12 GB and every chip of the host wants it at the start of the last iteration,
so its build time and its host peak set how many chips may upload at once (TT_RELION_TABLE_TURNS).
  TT_VISIBLE_DEVICES=16 python table_probe.py --size 360 --out t.jsonl
"""
import argparse
import json
import resource
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
sys.path.insert(0, __file__.rsplit("/", 4)[0])
ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=360)
ap.add_argument("--pf", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from _tt import Clock, open_dev  # noqa: E402
from tt_relion import resident  # noqa: E402


def rss():
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("VmRSS")) / 1048576


rp = a.pf * (a.size // 2 - 1)
X, Y, Z = rp + 3, 2 * rp + 5, 2 * rp + 5
mdl = np.random.default_rng(0).normal(size=(Z * Y * X, 2)).astype(np.float32)
dev, nodes = open_dev()
res = resident.Resident(dev)
rec = dict(size=a.size, nvox=len(mdl), table_gb=len(mdl) * 64 / 2**30, rss_model_gb=rss(), load1=float(open("/proc/loadavg").read().split()[0]))
t0 = time.time()
tb = resident._corner_table(mdl, X, Y)
rec["build_s"] = time.time() - t0
del tb
clock = Clock(nodes)
for key in ("first", "next"):               # the second call is an iteration's model replacing the one before
    t0 = time.time()
    res.set_model(key, mdl, X, Y, Z, -(rp + 2), -(rp + 2), rp * rp, a.pf)
    rec[key + "_s"] = time.time() - t0
    rec[key + "_rss_after_gb"] = rss()
rec["aiclk_min"], rec["aiclk_med"] = clock.stop()
rec["rss_peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
rec["chip_free_gb"] = res.free / 2**30
with open(a.out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(json.dumps(rec, indent=1))
