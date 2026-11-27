"""Compare one iteration's output across --continue arms: resolution and half-map correlation against a reference arm.

  python d32_cmp.py IT REF_DIR DIR...     (IT = 19 -> run_it019_*)
Prints per arm: current resolution (model.star), changes in orientations/offsets (optimiser.star), and the Pearson
correlation of each half map with the reference arm's, plus 1 - corr so small differences stay readable.
"""
import re
import sys
from pathlib import Path

import numpy as np


def mrc(p):
    b = Path(p).read_bytes()
    nx, ny, nz, mode = np.frombuffer(b[:16], np.int32)
    next_ = int(np.frombuffer(b[92:96], np.int32)[0])
    assert mode == 2, mode
    return np.frombuffer(b[1024 + next_:], np.float32, nx * ny * nz).astype(np.float64)


def star(p, key):
    m = re.search(r"^%s\s+(\S+)" % key, Path(p).read_text(), re.M)
    return m.group(1) if m else None


def main():
    it, ref, arms = "%03d" % int(sys.argv[1]), sys.argv[2], sys.argv[3:]
    rmaps = [mrc("%s/run_it%s_half%d_class001.mrc" % (ref, it, h)) for h in (1, 2)]
    for d in [ref] + arms:
        f = "%s/run_it%s_" % (d, it)
        res = star(f + "half1_model.star", "_rlnCurrentResolution")
        ch = [star(f + "optimiser.star", k) for k in ("_rlnChangesOptimalOrientations", "_rlnChangesOptimalOffsets")]
        cs = []
        for h in (1, 2):
            c = np.corrcoef(rmaps[h - 1], mrc(f + "half%d_class001.mrc" % h))[0, 1]
            cs.append("%.6f (1-c %.2e)" % (c, 1 - c))
        print("%-40s res %s  dOri %s dOff %s  corr h1 %s  h2 %s" % (Path(d).name, res, ch[0], ch[1], cs[0], cs[1]))


if __name__ == "__main__":
    main()
