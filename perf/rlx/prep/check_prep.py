#!/usr/bin/env python3
"""check_prep.py <record dir> [<second record dir>]

Relative L2 error of each recorded prep output against the float64 reference (tt_relion/prep.py), per particle
and worst over the directory. With a second directory (the other arm of an N/T pair) also the difference
between the two arms on the particles both recorded.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from tt_relion.prep import read_record, reference

KEYS = ["Fimg_nomask", "Fimg", "power", "highres_Xi2"]


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.linalg.norm((a - b).ravel())
    return float(d / max(np.linalg.norm(b.ravel()), 1e-300)) if b.size else 0.0


def load(d):
    return {os.path.basename(p).split(".")[2]: read_record(p) for p in sorted(glob.glob(os.path.join(d, "prep.*.bin")))}


arms = [load(d) for d in sys.argv[1:3]]
for d, recs in zip(sys.argv[1:], arms):
    worst = dict.fromkeys(KEYS, 0.0)
    for part, r in recs.items():
        if not r.get("zero_mask"):
            print(f"{d} particle {part}: noise mask, no reference"); continue
        ref = reference(r)
        err = {k: rel(r[k], ref[k]) for k in KEYS}
        worst = {k: max(worst[k], err[k]) for k in KEYS}
        print(f"{d} particle {part} size {int(r['current_size'])} shift {int(r['dx'])},{int(r['dy'])}: "
              + " ".join(f"{k} {err[k]:.2e}" for k in KEYS))
    print(f"{d} worst of {len(recs)} vs float64: " + " ".join(f"{k} {worst[k]:.2e}" for k in KEYS))
if len(arms) == 2:
    both = sorted(set(arms[0]) & set(arms[1]))
    worst = {k: max([rel(arms[0][p][k], arms[1][p][k]) for p in both] or [0.0]) for k in KEYS + ["Fctf"]}
    print(f"arm vs arm on {len(both)} shared particles: " + " ".join(f"{k} {v:.2e}" for k, v in worst.items()))
