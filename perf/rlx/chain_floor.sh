#!/usr/bin/env bash
# The RLX floor chain on one Galaxy host, one stage at a time so no two timed runs share the CPUs.
#   1. tutorial cell, ttnn, continue from RELION's own job019 it012 (accuracy: 3.79378 A is the bar)
#   2. dev cell (1/8 EMPIAR-10028), CPU reference: RELION's own kernels, same ranks and seed
#   3. dev cell, ttnn on all 32 chips
# Each stage writes <runs>/<stage>/{start,end,rc,time.txt,aiclk.tsv,refine.log}; a stage whose rc
# file exists is skipped, so the chain resumes after a kill.
set -u
R=${RLX_ROOT:?}; runs=$R/runs; data=$R/data; mkdir -p "$runs"
export RELION_BIN=$R/relion/build/bin/relion_refine_mpi
export PYTHONPATH=$R/tt-relion-wk:${RLX_SITE:?}
export RLX_LEASE_DIR RLX_LEASE_HOST
wait_for() { for _ in $(seq 1 360); do grep -q "$2" "$1" 2>/dev/null && return 0; sleep 60; done; return 1; }
stage() { [ -f "$runs/$1/rc" ] && return 0; case " ${RLX_STAGES:-tut_ttnn dev_ref dev_ttnn} " in *" $1 "*) ;; *) return 0;; esac; echo "$(date -u +%T) stage $1"; shift; "$@"; }

if [[ " ${RLX_STAGES:-tut_ttnn} " == *" tut_ttnn "* ]]; then
wait_for "$data/tut.log" TUT-OK || { echo "tutorial never arrived"; exit 1; }
tut=$(dirname "$(find "$data" -path '*Refine3D/job019/run_it012_optimiser.star' | head -1)")/../..
cd "$tut"
TT_RELION_BACKEND=ttnn RLX_CHIPS=4 stage tut_ttnn "$R/tt-relion-wk/perf/rlx/run.sh" "$runs/tut_ttnn" 5 6 \
    --continue Refine3D/job019/run_it012_optimiser.star --o "$runs/tut_ttnn/run" \
    --dont_combine_weights_via_disc --preread_images --pool 6
fi

wait_for "$data/fetch.log" FETCH-OK || { echo "dev cell never arrived"; exit 1; }
cd "$data"
DEV=(--i dev.star --ref emd_2660.map --ini_high 60 --firstiter_cc --auto_refine --split_random_halves
     --particle_diameter 360 --ctf --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2
     --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --norm --scale --pad 2 --sym C1
     --low_resol_join_halves 40 --dont_combine_weights_via_disc --preread_images --pool 3 --random_seed 0)
# 32 ranks OOM'd the host at box 360, pad 2 (~15.5 GB a follower, 2026-11-26), so 16 followers with
# 2 chips and 4 threads each: same 64 cores, same 32 chips, half the RELION host memory.
TT_RELION_BACKEND=nullbase stage dev_ref "$R/tt-relion-wk/perf/rlx/run.sh" "$runs/dev_ref" 17 4 \
    "${DEV[@]}" --o "$runs/dev_ref/run"
TT_RELION_BACKEND=ttnn RLX_CHIPS=32 RLX_CHIPS_PER_RANK=2 stage dev_ttnn "$R/tt-relion-wk/perf/rlx/run.sh" "$runs/dev_ttnn" 17 4 \
    "${DEV[@]}" --o "$runs/dev_ttnn/run"
echo "$(date -u +%T) CHAIN-DONE"
