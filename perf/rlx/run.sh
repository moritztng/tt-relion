#!/usr/bin/env bash
# run.sh <outdir> <ranks> <threads> <relion_refine_mpi args...>
# Timed RELION run: /usr/bin/time -v around mpirun, AICLK of every chip sampled at 1 Hz into
# <outdir>/aiclk.tsv for the whole run. Env: RELION_BIN, TT_RELION_BACKEND, PYTHONPATH, RLX_*.
set -u
out=$1 n=$2 j=$3; shift 3
mkdir -p "$out"
# One torch thread per bridge: RELION already runs ranks x --j threads on every core.
export TT_RELION_TORCH_THREADS=${TT_RELION_TORCH_THREADS:-1}
# A one-iteration arm is stopped from outside and never prints the profiler's exit table: keep it in a file per rank.
[ -n "${TT_RELION_PROFILE:-}" ] && export TT_RELION_PROFILE_FILE=${TT_RELION_PROFILE_FILE:-$out/prof}
here=$(cd "$(dirname "$0")" && pwd)
# Columns follow the glob (lexicographic: 0 1 10 11 ..), so the first line names the node of each column.
( printf 't'; for f in /sys/class/tenstorrent/*/tt_aiclk; do d=${f%/tt_aiclk}; printf '\t%s' "${d##*[!0-9]}"; done; echo
  while :; do printf '%s' "$(date +%s.%N)"; for f in /sys/class/tenstorrent/*/tt_aiclk; do
      printf '\t%s' "$(cat "$f")"; done; echo; sleep 1; done ) > "$out/aiclk.tsv" &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT
date -u +%FT%TZ > "$out/start"
/usr/bin/time -v -o "$out/time.txt" mpirun -n "$n" --oversubscribe --bind-to none \
    -x PYTHONPATH -x TT_RELION_BACKEND -x RLX_CHIPS -x RLX_CHIP_LIST -x RLX_CHIP_BASE -x RLX_CHIPS_PER_RANK -x RLX_LEASE_DIR -x RLX_LEASE_HOST \
    -x TT_RELION_KCHUNK -x TT_RELION_SPLIT -x TT_RELION_FIDELITY -x TT_RELION_DEV_CHUNK -x TT_RELION_PROFILE -x TT_RELION_PROFILE_FILE -x TT_RELION_TORCH_THREADS -x TT_RELION_BATCH -x TT_RELION_BATCH_ROWS -x TT_RELION_OWN_ROWS -x TT_RELION_OZ_ZIG -x TT_RELION_BATCH_WAIT_MS -x TT_RELION_RESIDENT -x TT_RELION_RECORD -x TT_RELION_RECORD_N -x TT_RELION_RECORD_SUMS -x TT_RELION_RECORD_SUMS_N -x TT_RELION_BATCHLOG -x TT_RELION_DIGITS -x TT_RELION_NORM_DIGITS -x TT_RELION_CAND_WINDOW -x TT_RELION_CAND_CAP -x TT_RELION_OCH -x TT_RELION_MIN_ORI -x LD_LIBRARY_PATH -x TT_RELION_TABLE_TURNS \
    "$here/rank.sh" "$RELION_BIN" "$@" --j "$j" --cpu > "$out/refine.log" 2>&1
rc=$?
date -u +%FT%TZ > "$out/end"
echo "rc=$rc" > "$out/rc"
exit $rc
