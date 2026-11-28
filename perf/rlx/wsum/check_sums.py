#!/usr/bin/env python3
"""check_sums.py <record dir> [--json]: the float64 reference of storeWeightedSums against what RELION recorded.

Per particle: relative L2 error of each of the three per-pixel sums and of the particle's own backprojection
(data and weight), and how sparse its sums are. RELION sums in float32, so 1e-5..1e-4 is its own rounding;
anything above 1e-3 is a different transform.
"""
import json
import sys

import numpy as np

from tt_relion import sums


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main():
    rows = []
    for path in sums.records(sys.argv[1]):
        r = sums.load(path)
        row = dict(particle=r["part_id"], **sums.stats(r))
        got3 = sums.wavg(r)
        for name, got in zip(("parts", "XA", "AA"), got3):
            row[name] = rel(got, r["wdiff2s_" + name].astype(np.float64))
        if "Mresol_fine" in r:
            shells, _, xa, aa = sums.post(r, *got3)
            n = len(shells)
            row["shells"] = rel(shells, r["wsum_sigma2_noise"][:n])
            row["scale_XA"] = abs(xa - r["wsum_scale_XA"]) / max(abs(r["wsum_scale_XA"]), 1e-300)
            row["scale_AA"] = abs(aa - r["wsum_scale_AA"]) / max(abs(r["wsum_scale_AA"]), 1e-300)
            # the same four from the per-shell form against the per-pixel one
            a, b = sums.post_shells(r, *sums.wavg_shells(r)), (shells, shells.sum(), xa, aa)
            row["shell_form"] = max(rel(np.atleast_1d(u), np.atleast_1d(v)) for u, v in zip(a, b))
        if "bp_index" in r:
            ref, rec = sums.backproject(r), sums.recorded_backprojection(r)
            row["bp_data"] = rel(ref[0] + 1j * ref[1], rec[0] + 1j * rec[1])
            row["bp_weight"] = rel(ref[2], rec[2])
            row["bp_voxels"] = [int((ref[2] != 0).sum()), int(len(r["bp_index"]))]
        rows.append(row)
    if "--json" in sys.argv:
        print(json.dumps(rows))
        return
    for row in rows:
        print(" ".join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    for k in ("parts", "XA", "AA", "shells", "scale_XA", "scale_AA", "shell_form", "bp_data", "bp_weight"):
        v = [row[k] for row in rows if k in row]
        if v:
            print(f"worst {k} {max(v):.2e} over {len(v)} particles")


if __name__ == "__main__":
    main()
