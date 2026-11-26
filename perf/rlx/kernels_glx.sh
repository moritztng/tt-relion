#!/bin/bash
# Kernel prototypes on a BH Galaxy chip range, one job per chip, all in parallel.
# Usage: kernels_glx.sh RUN_DIR FIRST_CHIP   (uses FIRST..FIRST+4). Stops before launching if ~/fleet/HANDBACK exists.
set -u
cd "$(dirname "$0")/../.."
R=${1:?run dir}; C=${2:?first chip}; PY=${PY:-$HOME/fleet/env/bin/python}
mkdir -p "$R"; export PYTHONPATH=$PWD/perf/rlx
[ -e ~/fleet/HANDBACK ] && { echo "HANDBACK present, not launching"; exit 3; }
run() {  # chip name timeout cmd...
    local chip=$1 name=$2 t=$3; shift 3
    ( TT_VISIBLE_DEVICES=$chip timeout "$t" "$PY" "$@" > "$R/$name.log" 2>&1; echo "$name rc=$? chip=$chip" >> "$R/status" ) &
}
run $((C+0)) probe   1200 perf/rlx/probe_acc.py "$R/acc.jsonl"
run $((C+1)) proj    2400 perf/rlx/proj_proto.py --sizes 50,100,146 --out "$R/proj.jsonl"
run $((C+2)) proj360 2400 perf/rlx/proj_proto.py --sizes 360 --oc 32 --out "$R/proj360.jsonl"
run $((C+3)) cmp     1200 perf/rlx/diff2_parity360.py --cases box64,f360 --splits 3 --dtypes bf16 --fids HiFi4 --kcs 64 --sums host --out "$R/cmp.jsonl"
run $((C+4)) proto   2400 perf/rlx/proto_wavg_fft.py --out "$R/proto.jsonl" --fft_batch 8
wait; echo GLX-DONE >> "$R/status"
