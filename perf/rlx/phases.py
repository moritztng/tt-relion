#!/usr/bin/env python3
"""Split a timestamped RELION auto-refine log (tsline.py) into E-step and host time per iteration.

E-step = the "Expectation iteration N" line to the end of its progress bar. Everything else between two E-steps
(maximisation, reconstruction, convergence checks, writing, the next iteration's setup) is host time, and so is the
start-up before the first E-step (reading images). wall is first line to last line.
  phases.py <run.out> [gpus]     prints per-iteration rows, totals, and wall/gpus beside E/gpus + host
"""
import re, sys

rows = []
for l in open(sys.argv[1], errors="replace"):
    m = re.match(r"(\d+\.\d+) (\d+\.\d+) (.*)", l)
    if m:
        rows.append((float(m[1]), float(m[2]), m[3]))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
t0, t1 = rows[0][0], rows[-1][1]
es = []  # (iteration, start, end)
for i, (a, b, t) in enumerate(rows):
    m = re.search(r"Expectation iteration (\d+)", t)
    if m:  # its progress bar is the next line carrying RELION's fish; that line ends when the E-step does
        bar = next((r for r in rows[i + 1:i + 6] if '~~(,_,">' in r[2]), None)
        es.append((int(m[1]), a, bar[1] if bar else b))
E = 0.0
print("iter  estep_s  host_before_s")
prev = t0
for it, a, b in es:
    print(f"{it:4d} {b - a:8.1f} {a - prev:8.1f}")
    E += b - a
    prev = b
wall = t1 - t0
print(f"tail (after the last E-step) {t1 - prev:.1f} s")
print(f"WALL_S {wall:.1f}  ESTEP_S {E:.1f}  RESIDUAL_S {wall - E:.1f}  iterations {len(es)}")
print(f"SCREEN_IDEAL_S {wall / n:.1f}  SCREEN_REAL_S {E / n + wall - E:.1f}  (gpus {n})")
