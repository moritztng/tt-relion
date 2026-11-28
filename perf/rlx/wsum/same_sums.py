#!/usr/bin/env python3
"""same_sums.py <record dir A> <record dir B>: are the recorded sums of the particles both runs recorded the same
to the bit? For a kernel change that only skips work (relion5-sums-sparse) nothing may differ."""
import os
import sys

import numpy as np

from tt_relion import sums

OUT = ("wdiff2s_parts", "wdiff2s_XA", "wdiff2s_AA", "bp_index", "bp_real", "bp_imag", "bp_weight", "wsum_sigma2_noise",
       "wsum_norm_correction", "wsum_scale_XA", "wsum_scale_AA")


def by_particle(d):
    return {os.path.basename(p).split(".")[2]: p for p in sums.records(d)}


def main():
    a, b = by_particle(sys.argv[1]), by_particle(sys.argv[2])
    common = sorted(set(a) & set(b), key=int)
    bad = 0
    for k in common:
        ra, rb = sums.read(a[k]), sums.read(b[k])
        diff = [n for n in OUT if not np.array_equal(ra[n], rb[n])]
        same_in = all(np.array_equal(ra[n], rb[n]) for n in ("weights", "eulers", "Fimg_real", "trans_x"))
        bad += bool(diff)
        print(f"particle {k}: inputs {'same' if same_in else 'DIFFER'}, outputs {'same to the bit' if not diff else 'DIFFER in ' + ' '.join(diff)}")
    print(f"{len(common)} particles in both, {bad} differ")
    return 1 if bad or not common else 0


if __name__ == "__main__":
    sys.exit(main())
