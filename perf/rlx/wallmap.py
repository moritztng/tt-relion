#!/usr/bin/env python3
"""Whole-pipeline wall map of one run.sh run directory (state/rlx/WALLMAP.md is built from this).

    python3 wallmap.py <rundir> [--chips N]

Sources, all written by the run itself:
  start / end              wall clock of mpirun (UTC, from run.sh)
  run_itNNN_optimiser.star mtime = end of iteration NNN (RELION writes it last), so diffs are iteration walls
  refine.log               E-step progress "a/b min" (the leader's clock, last value before "Maximization...");
                           TTPROF tables (one per follower at exit, thread-summed CPU seconds, shares only);
                           TTDEV lines (one per device server at exit: chip up/mm/down seconds + bridge counters)
  aiclk.tsv                every chip's AICLK at 1 Hz during the run

Per follower rank, inside the E-step:
  device busy   = mm_s (matmul, synchronised) + up_s + down_s (host<->chip transfers)
  IPC           = bridge round trip rt_s - server busy (pipe + wakeups) ; shm copies copy_s
  bridge host   = C_s + F_s - rt_s - copy_s (projection, bf16 split, float64 sums; summed over threads)
  RELION host   = E-step wall - the above (C++ outside the hook, incl. threads waiting on the chip lock)
"""
import argparse
import datetime as dt
import json
import re
import statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run")
a = ap.parse_args()
d = Path(a.run)
utc = lambda s: dt.datetime.strptime(s.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
t0 = utc((d / "start").read_text())
t1 = utc((d / "end").read_text()) if (d / "end").exists() else None
its = sorted((int(m.group(1)), p.stat().st_mtime) for p in d.glob("run_it*_optimiser.star")
             if (m := re.match(r"run_it(\d+)_optimiser", p.name)))
log = (d / "refine.log").read_text(errors="replace")

# E-step minutes per iteration: the last progress value before each "Maximization..."
estep = {}
for chunk in log.split("Maximization...")[:-1]:
    m = re.search(r"Expectation iteration (\d+)(.*)\Z", chunk, re.S)
    if m:
        v = re.findall(r"([0-9.]+)/\s*[0-9.]+ min", m.group(2))
        estep[int(m.group(1))] = float(v[-1]) * 60 if v else float("nan")

dev = []
for m in re.finditer(r"^TTDEV chip=(\S+) calls=(\d+) busy_s=([0-9.]+) up_s=([0-9.]+) mm_s=([0-9.]+) "
                     r"down_s=([0-9.]+) life_s=([0-9.]+) client=(\{.*?\})$", log, re.M):
    c = json.loads(m.group(8))
    dev.append(dict(chip=m.group(1), calls=int(m.group(2)), busy=float(m.group(3)), up=float(m.group(4)),
                    mm=float(m.group(5)), down=float(m.group(6)), life=float(m.group(7)), **c))

clk = []
for line in (d / "aiclk.tsv").read_text().splitlines():
    f = line.split("\t")
    if len(f) > 1:
        clk.append([int(x) for x in f[1:]])
busy_clk = [x for row in clk for x in row if x > 600]   # chips that were computing (idle WH sits at 500)

print(f"# WALLMAP {d}")
print(f"mpirun wall: {t1 - t0:.0f} s" if t1 else "mpirun wall: (running)")
if its:
    print(f"setup to it{its[0][0]:03d} written: {its[0][1] - t0:.0f} s")
    walls = [(i, b - a_) for (_, a_), (i, b) in zip(its, its[1:])]
    print("iter  wall_s  estep_s  mstep+io_s")
    for i, w in walls:
        e = estep.get(i, float("nan"))
        print(f"{i:4d}  {w:7.1f}  {e:7.1f}  {w - e:9.1f}")
    if t1:
        print(f"after it{its[-1][0]:03d} to exit: {t1 - its[-1][1]:.0f} s")
    E = sum(x for x in estep.values() if x == x)
    print(f"sum E-step {E:.0f} s of {its[-1][1] - its[0][1]:.0f} s iterating")
if dev:
    n = len(dev)
    mean = lambda k: sum(x.get(k, 0.0) for x in dev) / n
    rt, cp, bus = mean("rt_s"), mean("copy_s"), mean("busy")
    br = mean("C_s") + mean("F_s") - rt - cp
    print(f"\nper chip, mean over {n} device servers (seconds over the whole run):")
    print(f"  matmul {mean('mm'):.1f}  upload {mean('up'):.1f}  readback {mean('down'):.1f}  -> busy {bus:.1f}")
    print(f"  IPC (round trip - busy) {rt - bus:.1f}  shm copies {cp:.1f}  bytes {mean('bytes') / 1e9:.2f} GB"
          f"  calls {mean('calls'):.0f}")
    print(f"  bridge host (projection, split, sums; thread-summed) {br:.1f}")
    print(f"  server life {mean('life'):.1f}; busy fraction of life: min {min(x['busy'] / x['life'] for x in dev):.3f}"
          f" median {st.median(x['busy'] / x['life'] for x in dev):.3f} max {max(x['busy'] / x['life'] for x in dev):.3f}")
    print(f"  matmul fraction of life: median {st.median(x['mm'] / x['life'] for x in dev):.3f}")
if busy_clk:
    print(f"\nAICLK while computing: median {st.median(busy_clk)} MHz, min {min(busy_clk)}, max {max(busy_clk)}"
          f" ({len(busy_clk)} chip-samples > 600 MHz)")
