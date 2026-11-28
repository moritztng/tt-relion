#!/bin/bash
# prof_arm.sh <tag> <optimiser.star to continue from> <iteration>
# One iteration on RELION's own CPU kernels (no chip) with the stage profiler on, stopped once every follower rank
# has N particles in its table. Output: $R/runs/<tag>/prof.<pid> per rank, read by split.py.
# Env: R (work dir, default ~/rlx-wsum), RELION_BIN, RANKS (3), J (8), N (1500), BACKEND (nullbase), NICE (19).
# REC=<k> also records storeWeightedSums of each rank's first k particles into <tag>/rec (tt_relion/sums.py reads
# them); the table of such an arm carries the recorder's cost in those particles.
tag=${1:?tag}; from=${2:?optimiser.star}; it=${3:?iteration}
R=${R:-$HOME/rlx-wsum}; here=$(cd "$(dirname "$0")" && pwd); src=$(cd "$here/../../.." && pwd)
ranks=${RANKS:-3}; n=${N:-1500}; o=$R/runs/$tag
export RELION_BIN=${RELION_BIN:-$R/relion/build/bin/relion_refine_mpi}
export PYTHONPATH=$src:$HOME/fleet/env/lib/python3.10/site-packages
export TT_RELION_BACKEND=${BACKEND:-nullbase} TT_RELION_PROFILE=1 TT_RELION_PROFILE_FILE=$o/prof
unset TT_RELION_BATCH TT_RELION_RESIDENT TT_RELION_RECORD TT_RELION_RECORD_SUMS
rm -rf "$o"; mkdir -p "$o"
[ -n "$REC" ] && { mkdir $o/rec; export TT_RELION_RECORD_SUMS=$o/rec TT_RELION_RECORD_SUMS_N=$REC; }
echo "start $(date -u +%FT%TZ) load: $(cat /proc/loadavg) ranks $ranks j ${J:-8} n $n bin $RELION_BIN" > $o/status
cd $HOME/rlx/data
setsid nice -n ${NICE:-19} timeout -s INT ${TO:-2400} $src/perf/rlx/run.sh $o $ranks ${J:-8} --continue $from --iter $it \
    --dont_combine_weights_via_disc --preread_images --pool 3 --o $o/run &
p=$!
full() { [ "$(awk -v n=$n '$2=="oneParticle" && $5>=n' $o/prof.* 2>/dev/null | wc -l)" -ge $((ranks - 1)) ]; }
while kill -0 $p 2>/dev/null; do
  sleep 5
  full && break
  # the iteration is over (N was more than it holds): give the ranks time to finish writing their half maps
  [ -e $o/run_it$(printf %03d $it)_data.star ] && { sleep 30; break; }
done
# Stop: mpirun aborts on INT and forces it on a second one within 5 s; whatever is left after 20 s is killed by its
# output prefix (one arm ran two more iterations after a single INT).
kill -INT -- -$p 2>/dev/null; sleep 2; kill -INT -- -$p 2>/dev/null
for i in $(seq 20); do kill -0 $p 2>/dev/null || break; sleep 1; done
kill -0 $p 2>/dev/null && pkill -TERM -f -- "--o $o/run"
wait $p
echo "done rc=$? $(date -u +%FT%TZ) load1 $(cut -d' ' -f1 /proc/loadavg) tables: $(ls $o/prof.* 2>/dev/null | wc -l)" >> $o/status
