"""Replay one rank's recorded bridge call stream (tt_relion/record.py) into the bridge, without RELION.

One process stands for one MPI rank: it imports tt_relion.relion exactly as RELION's embedded interpreter does
(same environment: TT_RELION_BACKEND, TT_RELION_BATCH, TT_RELION_RESIDENT, RLX_RANK_CHIPS) and calls diff2_coarse
and diff2_fine from one Python thread per recorded RELION thread. Between two calls a thread sleeps the recorded
gap times --think (RELION's host work happens outside the GIL, and so does a sleep); --think 0 asks the bridge
for everything at once and gives the rate the bridge and the chip can take with RELION's host work removed.

Reports calls/s, batch fill, the chip's busy share of the wall (TTDEV/TTRES lines of the device servers), AICLK
sampled during the replay and load1. --digest writes per call the argmin, the minimum and the significant set
(the samples that carry 99.9 % of exp(-(d2 - min))); --ref compares against the digest of another arm.

    RLX_RANK_CHIPS=18 TT_RELION_BACKEND=ttnn TT_RELION_RESIDENT=1 TT_RELION_BATCH=8 \
        python perf/rlx/bridge/replay.py --rec rec/rec.1234.pkl --out replay.jsonl --digest t.npz --ref cpu.npz
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rec", required=True)
ap.add_argument("--think", type=float, default=1.0, help="scale of the recorded gap between a thread's calls")
ap.add_argument("--threads", type=int, default=0, help="0: as recorded; else deal the particles over this many")
ap.add_argument("--limit", type=int, default=0, help="first N calls only")
ap.add_argument("--out", default="replay.jsonl")
ap.add_argument("--digest", default="")
ap.add_argument("--ref", default="")
ap.add_argument("--tag", default="")
a = ap.parse_args()

err = a.out + ".%d.err" % os.getpid()               # the device servers inherit fd 2: their exit lines land here
sys.stderr.flush()
os.dup2(os.open(err, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), 2)

from tt_relion import record, relion  # noqa: E402

models, euls, calls = record.load(a.rec, a.limit)
for i, r in enumerate(calls):
    r["i"] = i

# One recorded thread's calls in order; the gap before a call is the host work RELION did since its last one.
by_tid = {}
for r in calls:
    q = by_tid.setdefault(r["tid"], [])
    r["gap"] = max(0.0, r["t0"] - q[-1]["t1"]) if q else 0.0
    q.append(r)
lanes = list(by_tid.values())
if a.threads:                                       # a particle starts at a coarse call and stays on one thread
    units = []
    for q in lanes:
        for r in q:
            if r["k"] == "C" or not units or units[-1][0]["tid"] != r["tid"]:
                units.append([])
            units[-1].append(r)
    units.sort(key=lambda u: u[0]["t0"])
    lanes = [[r for u in units[j::a.threads] for r in u] for j in range(a.threads)]

n = len(calls)
arg_min, d_min = np.zeros(n, np.int64), np.zeros(n, np.float64)
n_sig, h_sig = np.zeros(n, np.int64), np.zeros(n, "S16")
handled = np.zeros(n, bool)


def digest(i, d2):
    d = d2.astype(np.float64)
    arg_min[i], d_min[i] = int(d.argmin()), d.min()
    w = np.exp(d.min() - d)
    o = np.argsort(-w, kind="stable")
    k = int(np.searchsorted(np.cumsum(w[o]), 0.999 * w.sum())) + 1
    n_sig[i] = k
    h_sig[i] = hashlib.sha256(np.sort(o[:k]).tobytes()).digest()[:16]


def lane(q):
    for r in q:
        if a.think and r["gap"]:
            time.sleep(r["gap"] * a.think)
        args, out = record.args(models, euls, r)
        fn = relion.diff2_coarse if r["k"] == "C" else relion.diff2_fine
        handled[r["i"]] = bool(fn(*args))
        if a.digest and handled[r["i"]]:
            digest(r["i"], out)


class Clock(threading.Thread):
    """AICLK of the chips this rank's device servers hold, every 0.25 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.v, self.nodes, self.on = [], set(), True

    def run(self):
        while self.on:
            for s in relion._srv:
                try:
                    for f in os.listdir("/proc/%d/fd" % s.p.pid):
                        t = os.readlink("/proc/%d/fd/%s" % (s.p.pid, f))
                        if t.startswith("/dev/tenstorrent/"):
                            self.nodes.add(int(t.rsplit("/", 1)[1]))
                except OSError:
                    pass
            for node in self.nodes:
                try:
                    self.v.append(int(open("/sys/class/tenstorrent/tenstorrent!%d/tt_aiclk" % node).read().split()[0]))
                except (OSError, ValueError):
                    pass
            time.sleep(0.25)


load0 = open("/proc/loadavg").read().split()[0]
clk = Clock()
clk.start()
th = [threading.Thread(target=lane, args=(q,)) for q in lanes]
t0 = time.time()
for t in th:
    t.start()
for t in th:
    t.join()
wall = time.time() - t0
clk.on = False
clk.join()
for s in relion._srv:                               # EOF: the server closes its chip and prints its totals
    s.p.stdin.close()
for s in relion._srv:
    s.p.wait()

log = open(err).read()
chips = []
for m in re.finditer(r"^TTDEV chip=(\S+) calls=(\d+) busy_s=([0-9.]+) up_s=([0-9.]+) mm_s=([0-9.]+) "
                     r"down_s=([0-9.]+) life_s=([0-9.]+) client=(\{.*?\})$", log, re.M):
    c = dict(chip=m.group(1), calls=int(m.group(2)), busy_s=float(m.group(3)), **json.loads(m.group(8)))
    r = re.search(r"^TTRES chip=%s (\{.*\})$" % re.escape(c["chip"]), log, re.M)
    if r:                                           # resident op: the chip's own share, host operand build apart
        c.update(json.loads(r.group(1)))
        c["fill"] = round(c["rows"] / max(1, c["rows_padded"]), 3)
    c["busy_frac"] = round(c.get("dev_s", c["busy_s"]) / wall, 3)
    chips.append(c)

st = relion.stats()
res = dict(tag=a.tag, rec=os.path.basename(a.rec), backend=os.environ.get("TT_RELION_BACKEND", "torch"),
           resident=os.environ.get("TT_RELION_RESIDENT", ""), batch=os.environ.get("TT_RELION_BATCH", ""),
           rank_chips=os.environ.get("RLX_RANK_CHIPS", ""), threads=len(lanes), think=a.think, calls=n,
           coarse=sum(r["k"] == "C" for r in calls), handled=int(handled.sum()), wall_s=round(wall, 3),
           calls_per_s=round(n / wall, 2), recorded_span_s=round(max(r["t1"] for r in calls) - min(r["t0"] for r in calls), 3),
           recorded_gap_s=round(sum(r["gap"] for r in calls), 3), batches=st.get("batches", 0),
           mean_batch=round(st.get("batched", 0) / max(1, st.get("batches", 0)), 2), chips=chips,
           busy_frac=round(float(np.mean([c["busy_frac"] for c in chips])), 3) if chips else None,
           aiclk_min=min(clk.v) if clk.v else None, aiclk_med=int(np.median(clk.v)) if clk.v else None,
           nodes=sorted(clk.nodes), load1=[load0, open("/proc/loadavg").read().split()[0]])
if a.digest:
    np.savez(a.digest, arg_min=arg_min, d_min=d_min, n_sig=n_sig, h_sig=h_sig, handled=handled)
if a.ref:
    ref = np.load(a.ref)
    ok = handled & ref["handled"]
    res.update(ref=os.path.basename(a.ref), compared=int(ok.sum()),
               same_argmin=int((arg_min == ref["arg_min"])[ok].sum()),
               same_set=int((h_sig == ref["h_sig"])[ok].sum()),
               max_dmin_err=float(np.abs(d_min - ref["d_min"])[ok].max()) if ok.any() else None)
with open(a.out, "a") as fh:
    fh.write(json.dumps(res) + "\n")
print(json.dumps(res))
