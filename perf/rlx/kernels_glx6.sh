#!/bin/bash
# Pass g6: Ozaki diff2 cross term at the fine sizes, one chip per case.
set -u
cd "$(dirname "$0")/../.."
R=${1:?run dir}; C=${2:?first chip}; PY=${PY:-$HOME/fleet/env/bin/python}
mkdir -p "$R"; export PYTHONPATH=$PWD/perf/rlx
[ -e ~/fleet/HANDBACK ] && { echo "HANDBACK present, not launching"; exit 3; }
i=0
for case in box64 c100 f146 f360; do
    ( TT_VISIBLE_DEVICES=$((C+i)) timeout 2400 "$PY" perf/rlx/ozaki_proto.py --cases $case --ns 10,11,12,13 \
        --out "$R/oz_$case.jsonl" > "$R/oz_$case.log" 2>&1; echo "$case rc=$?" >> "$R/status" ) &
    i=$((i+1))
done
wait; echo GLX-DONE >> "$R/status"
