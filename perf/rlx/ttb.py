"""Where the chips' time goes in a RELION run on the resident op, from the device servers' per-batch lines
(TT_RELION_BATCHLOG=1: "TTB start wall host_s dev_s N T O P", one per chip call, all chips in one log).
Prints, per image size and pass (shared orientation set = coarse, own sets = fine), the batches, mean particles,
mean rows and seconds, then the busy share of the chips over the logged span.
usage: ttb.py refine.log <chips>"""
import sys
from collections import defaultdict

chips = int(sys.argv[2])
rows = [ln.split()[1:] for ln in open(sys.argv[1], errors="replace") if ln.startswith("TTB ")]
rows = [(float(r[0]), float(r[1]), int(r[4]), int(r[5]), int(r[6]), int(r[7])) for r in rows if len(r) == 8]
if not rows:
    sys.exit("no TTB lines")
g = defaultdict(list)
for t0, wall, N, T, O, P in rows:
    g[(P, "coarse" if O > 64 * N else "fine")].append((wall, N, T, O))
span = max(r[0] + r[1] for r in rows) - min(r[0] for r in rows)
busy = sum(r[1] for r in rows)
for (P, kind), v in sorted(g.items()):
    n = len(v)
    print(f"P {P} {kind}: batches {n} particles/batch {sum(x[1] for x in v) / n:.1f} T {sum(x[2] for x in v) / n:.1f} "
          f"O {sum(x[3] for x in v) / n:.0f} s/batch {sum(x[0] for x in v) / n:.3f} total_s {sum(x[0] for x in v):.1f} "
          f"particles/s/chip-busy {sum(x[1] for x in v) / sum(x[0] for x in v):.1f}")
print(f"span_s {span:.1f} chips {chips} busy_s {busy:.1f} busy_share {busy / (span * chips):.3f}")
