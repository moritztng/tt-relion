#!/bin/bash
# Pass g4: is the projection prototype dispatch-bound? s146 O1024 at four chunk shapes, one chip each.
set -u
cd "$(dirname "$0")/../.."
R=${1:?run dir}; C=${2:?first chip}; PY=${PY:-$HOME/fleet/env/bin/python}
mkdir -p "$R"; export PYTHONPATH=$PWD/perf/rlx
[ -e ~/fleet/HANDBACK ] && { echo "HANDBACK present, not launching"; exit 3; }
i=0
for shape in "64 1024" "256 1024" "512 512" "1024 256"; do
    set -- $shape
    ( TT_VISIBLE_DEVICES=$((C+i)) timeout 1800 "$PY" perf/rlx/proj_proto.py --sizes 146 --O 1024 --oc $1 --pc $2 \
        --out "$R/oc$1_pc$2.jsonl" > "$R/oc$1_pc$2.log" 2>&1; echo "oc$1_pc$2 rc=$?" >> "$R/status" ) &
    i=$((i+1))
done
wait; echo GLX-DONE >> "$R/status"
