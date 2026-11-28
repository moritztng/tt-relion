#!/bin/bash
# record.sh <tag> <optimiser.star to continue from> <iteration>
# Runs that one iteration on RELION's own CPU kernels with the bridge call stream recorded (tt_relion/record.py),
# RANKS-1 follower ranks x J threads, and stops RELION once every follower has N calls. No chip is opened.
# Records land in ~/rlx-bridge/rec/<tag>/rec.<pid>.pkl, one per follower rank. Env: RANKS (3), J (8), N (1200).
tag=${1:?tag}; from=${2:?optimiser.star}; it=${3:?iteration}
R=$HOME/rlx-bridge; here=$(cd "$(dirname "$0")" && pwd); src=$(cd "$here/../../.." && pwd)
ranks=${RANKS:-3}; o=$R/runs/rec_$tag; rec=$R/rec/$tag
export RELION_BIN=${RELION_BIN:-$HOME/rlx-e2e/relion/build/bin/relion_refine_mpi}
export PYTHONPATH=$src:$HOME/fleet/env/lib/python3.10/site-packages
export TT_RELION_BACKEND=nullbase TT_RELION_RECORD=$rec TT_RELION_RECORD_N=${N:-1200}
unset TT_RELION_BATCH TT_RELION_RESIDENT
rm -rf "$o" "$rec"; mkdir -p "$o" "$rec"
echo "start $(date -u +%FT%TZ) load: $(cat /proc/loadavg)" > $o/status
cd $HOME/rlx/data
setsid timeout -s INT 2700 $src/perf/rlx/run.sh $o $ranks ${J:-8} --continue $from --iter $it \
    --dont_combine_weights_via_disc --preread_images --pool 3 --o $o/run &
p=$!
while kill -0 $p 2>/dev/null; do
  sleep 5
  [ "$(ls $rec/*.full 2>/dev/null | wc -l)" -ge $((ranks - 1)) ] && { kill -INT -- -$p 2>/dev/null; break; }
done
wait $p
echo "done rc=$? $(date -u +%FT%TZ) load1 $(cut -d' ' -f1 /proc/loadavg) files: $(ls $rec | wc -l) $(du -sh $rec | cut -f1)" >> $o/status
