#!/bin/bash
# One heavy late iteration of the dev cell (it016: healpix 3, global finer pass), continued from the CPU reference
# run's it015 state. Arm N: RELION CPU kernels (nullbase). Arm T: ttnn hook + TT_RELION_BATCH=8. Chips 16-23, 9 ranks x --j 8.
# Env: SRC (source tree under $R, required: a default would silently run an old tree), TAG (run name, default dc2), ARMS, TT_RELION_RESIDENT=1 for arm T,
# RANKS (default 9), RLX_CHIP_LIST (explicit chips), J (threads per rank, default 8) and BATCH (TT_RELION_BATCH of arm T, default 8).
# FROM (optimiser.star to continue, default the CPU reference run's it015) and IT (the iteration to run, default 16),
# RLX_CHIPS_PER_RANK (default 1). R (run root holding $SRC and runs/, default ~/rlx-e2e).
# END_BY (epoch): the arm is stopped then, for a booked window that opens after it (default 45 min from its start).
R=${R:-$HOME/rlx-e2e}; : ${SRC:?name the source tree under $R}; TAG=${TAG:-dc2}
FROM=${FROM:-$HOME/rlx/runs/dev_ref_bh/run_it015_optimiser.star}; IT=${IT:-16}; it=$(printf %03d $IT)
export RELION_BIN=${RELION_BIN:-$R/relion/build/bin/relion_refine_mpi}
export PYTHONPATH=$R/$SRC:$HOME/fleet/env/lib/python3.10/site-packages TT_RELION_PYTHON=$HOME/fleet/env/bin/python
export TT_METAL_INSPECTOR=0 RLX_CHIP_BASE=16 RLX_CHIPS=8 RLX_CHIPS_PER_RANK=${RLX_CHIPS_PER_RANK:-1}
export RLX_LEASE_DIR=/tmp/tt-bio-device-leases RLX_LEASE_HOST=$(hostname)
S=$R/runs/$TAG.status
# start gate: not before NOT_BEFORE (epoch), and no process holding more than 8 chips (a whole-box mesh), 3 checks running
while [ $(date +%s) -lt ${NOT_BEFORE:-0} ]; do sleep 20; done
ok=0; while [ $ok -lt 3 ]; do
  big=$(fuser /dev/tenstorrent/* 2>&1 | grep -o '[0-9]\+m\?$\| [0-9]\+m' | tr -d ' m' | sort | uniq -c | awk '$1>8' | wc -l)
  [ "$big" = 0 ] && ok=$((ok+1)) || ok=0; sleep 20
done
echo "start $(date -u +%FT%TZ) load: $(cat /proc/loadavg)" >> $S
cd $HOME/rlx/data
for arm in ${ARMS:-T N}; do
  [ -e $HOME/fleet/HANDBACK ] && { echo HANDBACK >> $S; exit 0; }
  if [ $arm = N ]; then export TT_RELION_BACKEND=nullbase; unset TT_RELION_BATCH TT_RELION_RESIDENT
  else export TT_RELION_BACKEND=ttnn TT_RELION_BATCH=${BATCH:-8} TT_RELION_BATCH_WAIT_MS=50; fi
  o=$R/runs/${TAG}_$arm; rm -rf $o; mkdir -p $o
  setsid timeout -s INT $(( ${END_BY:-$(date +%s) + 2700} - $(date +%s) )) $R/$SRC/perf/rlx/run.sh $o ${RANKS:-9} ${J:-8} --continue $FROM --iter $IT --dont_combine_weights_via_disc --preread_images --pool 3 --o $o/run &
  p=$!
  ( while kill -0 $p 2>/dev/null; do sleep 15; [ -e $o/run_it${it}_data.star ] && { sleep 90; kill -INT -- -$p 2>/dev/null; break; }; done ) &
  wait $p; rc=$?; grep -q "TTBridge: .* raised" $o/refine.log && rc=FALLBACK
  echo "$arm rc=$rc $(date -u +%FT%TZ) load1 $(cut -d" " -f1 /proc/loadavg) wall $(grep Elapsed $o/time.txt | awk "{print \$NF}") it$it=$([ -e $o/run_it${it}_data.star ] && echo yes || echo no)" >> $S
done
echo "done $(date -u +%FT%TZ)" >> $S
