#!/bin/bash
# One late iteration of the dev cell through RELION with the coarse pass in candidate form (arm C,
# TT_RELION_CAND_WINDOW) and/or as the dense block (arm D), same binary, same chips, resident path.
# Needs a RELION built with relion5-tt.patch + relion5-tt-candidates.patch (RELION_BIN).
# Env: SRC (tree under ~/rlx-bridge, default src-d), TAG (default ca1), ARMS (default "C"), WINDOW (default 40),
# RLX_CHIP_LIST (default 12,13), KPR (chips per follower rank, default 1), J (threads per rank, default 40),
# BATCH (default 32),
# FROM / IT (state to continue and iteration to run, default the CPU reference's it015 -> 16),
# CPU_STAR (the CPU arm's data.star of that iteration, for the pose grade), NOT_BEFORE / GIVE_UP / END_BY (epochs).
# ~/rlx-bridge/STOP and ~/fleet/HANDBACK veto. Status: ~/rlx-bridge/runs/$TAG.status.
R=$HOME/rlx-bridge; SRC=${SRC:-src-d}; TAG=${TAG:-ca1}; PY=$HOME/fleet/env/bin/python
FROM=${FROM:-$HOME/rlx/runs/dev_ref_bh/run_it015_optimiser.star}; IT=${IT:-16}; it=$(printf %03d $IT)
CPU_STAR=${CPU_STAR:-$HOME/rlx-e2e/runs/dc2_N/run_it${it}_data.star}
export RELION_BIN=${RELION_BIN:-$R/relion/build/bin/relion_refine_mpi}
export PYTHONPATH=$R/$SRC:$HOME/fleet/env/lib/python3.10/site-packages TT_RELION_PYTHON=$PY
export TT_METAL_INSPECTOR=0 RLX_CHIP_LIST=${RLX_CHIP_LIST:-12,13} RLX_CHIPS_PER_RANK=${KPR:-1}
export RLX_LEASE_DIR=/tmp/tt-bio-device-leases RLX_LEASE_HOST=$(hostname)
export TT_RELION_BACKEND=ttnn TT_RELION_RESIDENT=1 TT_RELION_BATCH=${BATCH:-32} TT_RELION_BATCH_ROWS=${ROWS:-1024} TT_RELION_BATCH_WAIT_MS=50
export TT_RELION_BATCHLOG=1 TT_RELION_PROFILE=1
IFS=, read -ra CH <<< "$RLX_CHIP_LIST"; NC=${#CH[@]}
S=$R/runs/$TAG.status; mkdir -p $R/runs
veto() { [ -e $R/STOP ] || [ -e $HOME/fleet/HANDBACK ]; }
leased() { for c in "${CH[@]}"; do ( exec {fd}>"$RLX_LEASE_DIR/${RLX_LEASE_HOST}-card$c.json"; flock -n $fd ) || return 0; done; return 1; }
while [ $(date +%s) -lt ${NOT_BEFORE:-0} ]; do veto && exit 0; sleep 20; done
ok=0; while [ $ok -lt 3 ]; do     # no whole-box mesh running and my chips unleased, three checks in a row
  veto && { echo "veto $(date -u +%FT%TZ)" >> $S; exit 0; }
  [ $(date +%s) -gt ${GIVE_UP:-9999999999} ] && { echo "GAVE-UP chips busy $(date -u +%FT%TZ)" >> $S; exit 3; }
  big=$(fuser /dev/tenstorrent/* 2>&1 | grep -o '[0-9]\+m\?$\| [0-9]\+m' | tr -d ' m' | sort | uniq -c | awk '$1>8' | wc -l)
  if [ "$big" = 0 ] && ! leased; then ok=$((ok+1)); else ok=0; fi; sleep 20
done
echo "start $(date -u +%FT%TZ) chips $RLX_CHIP_LIST src $SRC load: $(cat /proc/loadavg)" >> $S
cd $HOME/rlx/data
for arm in ${ARMS:-C}; do
  veto && { echo "veto $(date -u +%FT%TZ)" >> $S; exit 0; }
  if [ $arm = C ]; then export TT_RELION_CAND_WINDOW=${WINDOW:-40}; else unset TT_RELION_CAND_WINDOW; fi
  o=$R/runs/${TAG}_$arm; rm -rf $o; mkdir -p $o
  setsid timeout -s INT $(( ${END_BY:-$(date +%s) + 3000} - $(date +%s) )) $R/$SRC/perf/rlx/run.sh $o $((NC / ${KPR:-1} + 1)) ${J:-40} \
    --continue $FROM --iter $IT --dont_combine_weights_via_disc --preread_images --pool ${POOL:-3} --o $o/run &
  p=$!
  ( while kill -0 $p 2>/dev/null; do sleep 15; [ -e $o/run_it${it}_data.star ] && { sleep 90; kill -INT -- -$p 2>/dev/null; break; }; done ) &
  wait $p; rc=$?; grep -q "TTBridge: .* raised" $o/refine.log && rc=FALLBACK
  echo "$arm rc=$rc $(date -u +%FT%TZ) load1 $(cut -d" " -f1 /proc/loadavg) wall $(grep Elapsed $o/time.txt | awk "{print \$NF}") it$it=$([ -e $o/run_it${it}_data.star ] && echo yes || echo no)" >> $S
  grep "TTBridge: " $o/refine.log | sort | uniq -c | sed "s/^/$arm /" >> $S
  [ -e $o/run_it${it}_data.star ] || continue
  [ -e $CPU_STAR ] && $PY $R/$SRC/perf/rlx/star_cmp.py $CPU_STAR $o/run_it${it}_data.star 2>&1 | sed "s/^/$arm vs CPU: /" >> $S
  $PY $R/$SRC/perf/rlx/ttb.py $o/refine.log $NC 2>&1 | sed "s/^/$arm ttb: /" >> $S
done
[ -e $R/runs/${TAG}_C/run_it${it}_data.star ] && [ -e $R/runs/${TAG}_D/run_it${it}_data.star ] && \
  $PY $R/$SRC/perf/rlx/star_cmp.py $R/runs/${TAG}_D/run_it${it}_data.star $R/runs/${TAG}_C/run_it${it}_data.star 2>&1 | sed "s/^/C vs D: /" >> $S
echo "done $(date -u +%FT%TZ)" >> $S
