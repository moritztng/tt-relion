"""Shape and timing census of a recorded call stream (tt_relion/record.py): what one rank asks the bridge for.

Per call kind (C coarse, F fine): orientations O, translations T (= chip rows per particle), pixels P, entries asked
for, the time RELION spent between a thread's calls (gap) and inside the call. The row counts say how many particles
fill the chip's 512-row tile; the gaps say how many threads it takes to keep one chip fed.

    python perf/rlx/bridge/shapes.py rec/g16/rec.*.pkl
"""
import json
import sys

import numpy as np

from tt_relion import record

q = lambda v: [round(float(x), 4) for x in np.percentile(v, [5, 50, 95])] if len(v) else None  # noqa: E731
for path in sys.argv[1:]:
    models, euls, calls = record.load(path)
    last = {}
    for r in calls:
        r["gap"] = r["t0"] - last[r["tid"]] if r["tid"] in last else np.nan
        last[r["tid"]] = r["t1"]
    out = dict(rec=path, calls=len(calls), threads=len(last), span_s=round(calls[-1]["t1"] - calls[0]["t0"], 2),
               model_mb=[round(len(m) / 2**20, 1) for m in models.values()], euler_sets=len(euls))
    for k in "CF":
        c = [r for r in calls if r["k"] == k]
        if not c:
            continue
        sc = np.array([r["sc"][-3:] if k == "C" else [r["sc"][-5], r["sc"][-4], r["sc"][-2]] for r in c])
        gap = np.array([r["gap"] for r in c])
        out[k] = dict(n=len(c), O=q(sc[:, 0]), T=q(sc[:, 1]), P=q(sc[:, 2]), entries=q([r["out"] // 4 for r in c]),
                      distinct_euler_sets=len({r["eul"] for r in c}), gap_s=q(gap[~np.isnan(gap)]),
                      gap_sum_s=round(float(np.nansum(gap)), 2), in_call_s=q([r["t1"] - r["t0"] for r in c]),
                      in_call_sum_s=round(sum(r["t1"] - r["t0"] for r in c), 2))
        if k == "F":        # the translations and orientations the fine pass actually asks for
            out[k]["T_asked"] = q([len(np.unique(np.frombuffer(r["mv"][6], np.uint64))) for r in c])
            out[k]["O_asked"] = q([len(np.unique(np.frombuffer(r["mv"][5], np.uint64))) for r in c])
    print(json.dumps(out))
