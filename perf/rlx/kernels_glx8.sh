#!/bin/bash
# Pass g8: storeWeightedSums' core in true fp32 eltwise (wavg_fp32.py), two arms on two chips.
set -u
cd "$(dirname "$0")/../.."
R=${1:?run dir}; C=${2:?first chip}; PY=${PY:-$HOME/fleet/env/bin/python}
mkdir -p "$R"; export PYTHONPATH=$PWD/perf/rlx
[ -e ~/fleet/HANDBACK ] && { echo "HANDBACK present, not launching"; exit 3; }
run() { ( TT_VISIBLE_DEVICES=$1 timeout 1500 "$PY" perf/rlx/wavg_fp32.py "${@:3}" --out "$R/$2.jsonl" > "$R/$2.log" 2>&1
          echo "$2 rc=$?" >> "$R/status" ) & }
run $C bcast --arms bcast
run $((C+1)) addcmul --arms addcmul
wait; echo GLX-DONE >> "$R/status"
