#!/usr/bin/env bash
# run.sh <outdir> <ranks> <threads> <relion_refine_mpi args...>
# Timed RELION run: /usr/bin/time -v around mpirun, AICLK of every chip sampled at 1 Hz into
# <outdir>/aiclk.tsv for the whole run. Env: RELION_BIN, TT_RELION_BACKEND, PYTHONPATH, RLX_*.
set -u
out=$1 n=$2 j=$3; shift 3
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
( while :; do printf '%s' "$(date +%s.%N)"; for f in /sys/class/tenstorrent/*/tt_aiclk; do
      printf '\t%s' "$(cat "$f")"; done; echo; sleep 1; done ) > "$out/aiclk.tsv" &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT
date -u +%FT%TZ > "$out/start"
/usr/bin/time -v -o "$out/time.txt" mpirun -n "$n" --oversubscribe --bind-to none \
    -x PYTHONPATH -x TT_RELION_BACKEND -x RLX_CHIPS -x RLX_LEASE_DIR -x RLX_LEASE_HOST \
    -x TT_RELION_KCHUNK -x TT_RELION_SPLIT -x TT_RELION_FIDELITY -x TT_RELION_DEV_CHUNK -x LD_LIBRARY_PATH \
    "$here/rank.sh" "$RELION_BIN" "$@" --j "$j" --cpu > "$out/refine.log" 2>&1
rc=$?
date -u +%FT%TZ > "$out/end"
echo "rc=$rc" > "$out/rc"
exit $rc
