#!/bin/bash
# Pass g7: Ozaki n14 at the coarse sizes; grouped (one matmul per digit diagonal) vs per-pair at the fine sizes.
set -u
cd "$(dirname "$0")/../.."
R=${1:?run dir}; C=${2:?first chip}; PY=${PY:-$HOME/fleet/env/bin/python}
mkdir -p "$R"; export PYTHONPATH=$PWD/perf/rlx
[ -e ~/fleet/HANDBACK ] && { echo "HANDBACK present, not launching"; exit 3; }
run() { ( TT_VISIBLE_DEVICES=$1 timeout 1500 "$PY" perf/rlx/ozaki_proto.py "${@:3}" --out "$R/$2.jsonl" > "$R/$2.log" 2>&1
          echo "$2 rc=$?" >> "$R/status" ) & }
run $C coarse --cases box64,c100 --ns 13,14
run $((C+1)) f146g --cases f146 --ns 13 --grouped 1
run $((C+2)) f360g --cases f360 --ns 13 --grouped 1
run $((C+3)) f360p --cases f360 --ns 13 --reps 6
wait; echo GLX-DONE >> "$R/status"
