"""Seconds of the expectation and the maximization step of every iteration in a RELION refine log, read from the
progress bars that closed (RELION prints them in sec, min or hrs; a min bar resolves about a second).
  python iters.py refine.log
"""
import re
import sys

UNIT = {"sec": 1, "min": 60, "hrs": 3600}
it, stage, rows = None, None, {}
for ln in open(sys.argv[1], errors="replace").read().replace("\r", "\n").split("\n"):
    m = re.search(r"Expectation iteration (\d+)", ln)
    if m:
        it, stage = int(m.group(1)), "E"
    elif "Maximization" in ln:
        stage = "M"
    m = re.match(r"\s*([\d.]+)/\s*([\d.]+) (sec|min|hrs)", ln)
    if m and it is not None and float(m.group(1)) == float(m.group(2)):
        rows.setdefault(it, {})[stage] = float(m.group(1)) * UNIT[m.group(3)]
for i in sorted(rows):
    print(f"it{i:03d} E {rows[i].get('E', float('nan')):7.0f} s  M {rows[i].get('M', float('nan')):6.0f} s")
e, m_ = (sum(r.get(k, 0) for r in rows.values()) for k in "EM")
print(f"iterations {len(rows)}  E {e:.0f} s  M {m_:.0f} s")
