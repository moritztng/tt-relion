"""Where the chips' time goes in a RELION run on the resident op, from the device servers' per-batch lines
(TT_RELION_BATCHLOG=1: "TTB start wall host_s dev_s N R O P", one per chip call, all chips in one log; R = the
batch's (particle, translation) rows). Prints, per image size and pass (shared orientation set = coarse, own sets =
fine), the batches, mean particles, rows, how full the 512-row tiles were and seconds, then the busy share of the
chips over the logged span.
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
