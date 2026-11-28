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
        for name, got in zip(("parts", "XA", "AA"), sums.wavg(r)):
            row[name] = rel(got, r["wdiff2s_" + name].astype(np.float64))
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
    for k in ("parts", "XA", "AA", "bp_data", "bp_weight"):
        v = [row[k] for row in rows if k in row]
        if v:
            print(f"worst {k} {max(v):.2e} over {len(v)} particles")


if __name__ == "__main__":
    main()
