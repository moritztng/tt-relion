"""Where the chips' time goes in a RELION run on the resident op, from the device servers' per-batch lines
(TT_RELION_BATCHLOG=1: "TTB start wall host_s dev_s N R O P", one per chip call, all chips in one log; R = the
batch's (particle, translation) rows). Prints, per image size and pass (shared orientation set = coarse, own sets =
fine), the batches, mean particles, rows, how full the 512-row tiles were and seconds, then the busy share of the
chips over the logged span, and the same per E-step: the span also holds the M-steps between iterations and whatever
of the next iteration ran before the job was stopped, so a window of more than 2 s with no chip in a batch ends an
E-step here. idle_ms_per_batch is the chips' idle time in the middle of the E-step (5 % to 90 % of it) over its
batches there; tail_idle_s is the idle chip time of its last tenth, where the ranks run out of particles.
slow_batches counts the batches of 2 s or more that also took over three times the median of their pass (a program
built for a new shape: the first run of a newly placed source tree pays 60 to 130 chip seconds of them, its second
run about 5), and slow_excess_s is their time over that median. Compare the E-step walls of two arms only when both
are small, or take slow_excess_s / chips off each wall first.
usage: ttb.py refine.log <chips> [T]     T: a log from before the flat rows, whose sixth field is translations per particle"""
import sys
from collections import defaultdict

chips = int(sys.argv[2])
rows = [ln.split()[1:] for ln in open(sys.argv[1], errors="replace") if ln.startswith("TTB ")]
rows = [(float(r[0]), float(r[1]), int(r[4]), int(r[5]), int(r[6]), int(r[7])) for r in rows if len(r) == 8]
if not rows:
    sys.exit("no TTB lines")
g = defaultdict(list)
old = len(sys.argv) > 3 and sys.argv[3] == "T"
for t0, wall, N, R, O, P in rows:
    g[(P, "coarse" if O > 64 * N else "fine")].append((wall, N, N * R if old else R, O))
span = max(r[0] + r[1] for r in rows) - min(r[0] for r in rows)
busy = sum(r[1] for r in rows)
for (P, kind), v in sorted(g.items()):
    n = len(v)
    fill = sum(x[2] for x in v) / sum(-(-x[2] // 512) * 512 for x in v)
    print(f"P {P} {kind}: batches {n} particles/batch {sum(x[1] for x in v) / n:.1f} rows/batch {sum(x[2] for x in v) / n:.0f} "
          f"tile_fill {fill:.2f} "
          f"O {sum(x[3] for x in v) / n:.0f} s/batch {sum(x[0] for x in v) / n:.3f} total_s {sum(x[0] for x in v):.1f} "
          f"particles/s/chip-busy {sum(x[1] for x in v) / sum(x[0] for x in v):.1f}")
print(f"span_s {span:.1f} chips {chips} busy_s {busy:.1f} busy_share {busy / (span * chips):.3f}")
# per E-step
ev = sorted([(r[0], 1) for r in rows] + [(r[0] + r[1], -1) for r in rows])
cuts, act, last = [ev[0][0]], 0, ev[0][0]
for t, d in ev:
    if act == 0 and t - last > 2.0:
        cuts += [last, t]
    act, last = act + d, t
cuts.append(ev[-1][0])
for n, (a, b) in enumerate(zip(cuts[0::2], cuts[1::2]), 1):
    seg = [r for r in rows if a - 0.01 <= r[0] < b]
    if not seg:
        continue
    wall, sbusy = b - a, sum(r[1] for r in seg)
    act, last, mid, tail = 0, a, 0.0, 0.0
    for t, d in sorted([(r[0], 1) for r in seg] + [(r[0] + r[1], -1) for r in seg]):
        f = (last - a) / wall
        if 0.05 <= f < 0.9:
            mid += (t - last) * (chips - act)
        elif f >= 0.9:
            tail += (t - last) * (chips - act)
        act, last = act + d, t
    nmid = sum(1 for r in seg if 0.05 <= (r[0] - a) / wall < 0.9)
    med = {}
    for k in (True, False):
        w = sorted(r[1] for r in seg if (r[4] > 64 * r[2]) == k)
        med[k] = w[len(w) // 2] if w else 0.0
    slow = [r[1] - med[r[4] > 64 * r[2]] for r in seg if r[1] >= 2.0 and r[1] > 3 * med[r[4] > 64 * r[2]]]
    print(f"estep {n}: wall_s {wall:.1f} batches {len(seg)} busy_s {sbusy:.1f} busy_share {sbusy / (wall * chips):.3f} "
          f"idle_ms_per_batch {1e3 * mid / max(nmid, 1):.0f} tail_idle_s {tail:.1f} "
          f"slow_batches {len(slow)} slow_excess_s {sum(slow):.1f}")
