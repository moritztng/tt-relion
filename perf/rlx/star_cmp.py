"""Compare two RELION data.star files row by row: per column, rows that differ and the max |diff|.
usage: star_cmp.py a.star b.star"""
import sys

import numpy as np

COLS = ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi", "_rlnOriginXAngst", "_rlnOriginYAngst",
        "_rlnMaxValueProbDistribution", "_rlnLogLikeliContribution"]


def rows(path):
    lines = open(path).read().split("data_particles")[1].splitlines()
    cols = [ln.split()[0] for ln in lines if ln.startswith("_rln")]
    body = [ln.split() for ln in lines if ln.strip() and not ln.startswith(("_", "loop_"))]
    return cols, [r for r in body if len(r) == len(cols)]


ca, a = rows(sys.argv[1])
cb, b = rows(sys.argv[2])
print(f"rows {len(a)} vs {len(b)}")
for k in COLS:
    x = np.array([float(r[ca.index(k)]) for r in a])
    y = np.array([float(r[cb.index(k)]) for r in b])
    print(f"{k} differ {int((x != y).sum())} max {float(np.abs(x - y).max()):.6g}")
