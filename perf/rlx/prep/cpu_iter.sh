#!/bin/bash
# cpu_iter.sh <tag> <prep 0|1> [iteration 16]: one whole iteration on RELION's CPU kernels (nullbase, no chip opened),
# niced, from the same optimiser state as rlx-e2e's one-iteration arms. Two of these with prep off and on are the
# pose / log-likelihood / half-map grade of TTPrep through RELION with nothing else in the path; prep off against
# an older CPU arm says whether this binary is still RELION when the switch is off.
# Env: R (~/rlx-prep), SRC (src-f), RELION_BIN, RANKS (5), J (12), FROM. Status: $R/runs/cpu.status
R=${R:-$HOME/rlx-prep}; SRC=${SRC:-src-f}; tag=$1; it=${3:-16}; itz=$(printf %03d $it)
FROM=${FROM:-$HOME/rlx/runs/dev_ref_bh/run_it015_optimiser.star}
[ $it = 24 ] && FROM=$HOME/rlx/runs/d32_l_n14/run_it023_optimiser.star
export RELION_BIN=${RELION_BIN:-$R/relion-f/build/bin/relion_refine_mpi} TT_RELION_PREP=$2 TT_RELION_BACKEND=nullbase
export PYTHONPATH=$R/$SRC:$HOME/fleet/env/lib/python3.10/site-packages TT_RELION_PYTHON=$HOME/fleet/env/bin/python
unset TT_RELION_BATCH TT_RELION_RESIDENT TT_RELION_PREP_WISDOM TT_RELION_PROFILE
o=$R/runs/${tag}_N; rm -rf $o; mkdir -p $o; cd $HOME/rlx/data
setsid nice -n 19 timeout -s INT ${LIMIT:-5400} $R/$SRC/perf/rlx/run.sh $o ${RANKS:-5} ${J:-12} --continue $FROM --iter $it \
  --dont_combine_weights_via_disc --preread_images --pool 3 --o $o/run &
p=$!
( while kill -0 $p 2>/dev/null; do sleep 15
    [ -e $o/run_it${itz}_data.star ] && { sleep 45; kill -INT -- -$p 2>/dev/null; sleep 20; kill -INT -- -$p 2>/dev/null; break; }
  done ) &
wait $p
echo "$(date -u +%FT%TZ) $tag prep=$2 it$itz=$([ -e $o/run_it${itz}_data.star ] && echo yes || echo no) wall $(grep Elapsed $o/time.txt | awk '{print $NF}') load1 $(cut -d' ' -f1 /proc/loadavg)" >> $R/runs/cpu.status
