#!/bin/bash
# Parity gate for the RELION-called path: replay a recorded rank (TT_RELION_RECORD, tt_relion/record.py) into the
# bridge on one chip and compare every call with the CPU digest of the same record, coarse and fine pass apart.
# Synthetic float64 parity of the chip op is not enough: it passed while the fine pass of a real iteration was wrong
# twice over (the radius cut of the coarse pass kept for fine references; too few digits for a real spectrum).
# usage: replay_gate.sh <record.pkl> <reference digest.npz> <out dir> [KEY=VAL ...]
#   CHIP (default 0), LIMIT (first N calls, the reference's length), TO (s), any TT_RELION_* (BATCH, BATCH_ROWS, DIGITS, ..)
#   the reference: replay.py --digest with TT_RELION_BACKEND=torch on the same record and limit
# Prints PASS or FAIL and the counts; exit 0 only if every compared call has the CPU arm's argmin and significant set.
set -u
rec=$1 ref=$2 o=$3; shift 3
for kv in "$@"; do export "$kv"; done
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../../.." && pwd)
PY=${TT_RELION_PYTHON:-python3}; CHIP=${CHIP:-0}
mkdir -p "$o"; rm -f "$o/replay.jsonl"
export PYTHONPATH=$root${PYTHONPATH:+:$PYTHONPATH} TT_METAL_INSPECTOR=0 TT_RELION_TORCH_THREADS=1
export TT_RELION_BACKEND=${TT_RELION_BACKEND:-ttnn} TT_RELION_RESIDENT=${TT_RELION_RESIDENT-1} TT_RELION_BATCH=${TT_RELION_BATCH:-24}
export TT_RELION_BATCH_WAIT_MS=${TT_RELION_BATCH_WAIT_MS:-50} RLX_RANK_CHIPS=$CHIP TT_VISIBLE_DEVICES=$CHIP
timeout -k 20 -s INT "${TO:-600}" "$PY" "$here/replay.py" --rec "$rec" --think 0 --out "$o/replay.jsonl" --digest "$o/d.npz" \
  --ref "$ref" --tag "$(basename "$o")" ${LIMIT:+--limit "$LIMIT"} > "$o/log" 2>&1
"$PY" - "$o/replay.jsonl" <<'PYEOF'
import json, sys
try:
    r = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
    ps = {k: r["pass_" + k] for k in "CF" if r.get("pass_" + k, {}).get("compared")}
    ok = bool(ps) and all(p["same_argmin"] == p["compared"] == p["same_set"] for p in ps.values())
    print("PASS" if ok else "FAIL", " ".join("%s %d/%d argmin %d/%d sets err %.4g" % (
        k, p["same_argmin"], p["compared"], p["same_set"], p["compared"], p["max_dmin_err"]) for k, p in ps.items()),
        "aiclk_min", r.get("aiclk_min"), "load1", r.get("load1"))
    sys.exit(0 if ok else 1)
except Exception as e:
    print("FAIL", repr(e)[:200]); sys.exit(2)
PYEOF
