#!/bin/bash
# Pass g2 on the rlx Galaxy range: piece-width probe, then the fixed projection prototype at all sizes.
set -u
cd "$(dirname "$0")/../.."
R=${1:?run dir}; C=${2:?first chip}; PY=${PY:-$HOME/fleet/env/bin/python}
mkdir -p "$R"; export PYTHONPATH=$PWD/perf/rlx
[ -e ~/fleet/HANDBACK ] && { echo "HANDBACK present, not launching"; exit 3; }
run() {
    local chip=$1 name=$2 t=$3; shift 3
    ( TT_VISIBLE_DEVICES=$chip timeout "$t" "$PY" "$@" > "$R/$name.log" 2>&1; echo "$name rc=$? chip=$chip" >> "$R/status" ) &
}
run $((C+0)) pieces  2400 perf/rlx/probe_pieces.py "$R/pieces.jsonl"
run $((C+1)) proj    2400 perf/rlx/proj_proto.py --sizes 50,100,146 --out "$R/proj.jsonl"
run $((C+2)) proj360 2400 perf/rlx/proj_proto.py --sizes 360 --oc 32 --out "$R/proj360.jsonl"
wait; echo GLX-DONE >> "$R/status"
