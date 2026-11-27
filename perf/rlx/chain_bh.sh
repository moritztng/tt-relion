#!/usr/bin/env bash
# BH wall map on one 8-chip range of a Blackhole Galaxy (RLX D17): the dev cell hooked (Ozaki diff2) to the
# end, then RELION's own CPU kernels in the identical rank layout up to iteration RLX_REF_ITERS, both on the same
# host so the per-iteration walls compare like for like. Stages land in <runs>/<stage>/ (run.sh layout); a stage
# whose rc file exists is skipped. Waits for RLX_WAIT_UNTIL (epoch) first; stops if ~/fleet/HANDBACK appears.
#   RLX_ROOT=~/rlx RLX_SITE=<site-packages> RLX_CHIP_BASE=16 RLX_CHIPS=8 chain_bh.sh
set -u
R=${RLX_ROOT:?}; runs=$R/runs; mkdir -p "$runs"; here=$(cd "$(dirname "$0")" && pwd)
export RELION_BIN=$R/relion/build/bin/relion_refine_mpi PYTHONPATH=$(dirname "$(dirname "$here")"):${RLX_SITE:?}
export TT_RELION_PYTHON=${TT_RELION_PYTHON:-python3}
export RLX_CHIPS=${RLX_CHIPS:-8} RLX_CHIP_BASE=${RLX_CHIP_BASE:-16} RLX_CHIPS_PER_RANK=1 TT_RELION_PROFILE=1
n=$((RLX_CHIPS + 1)) j=${RLX_J:-7}
while [ "$(date +%s)" -lt "${RLX_WAIT_UNTIL:-0}" ]; do sleep 30; done
[ -e "$HOME/fleet/HANDBACK" ] && { echo "HANDBACK present, not starting"; exit 1; }
# Never share a chip: wait until no process holds any chip of the range (a co-tenant may borrow an idle range).
# Chip ids are UMD (PCI-order) ids; fuser needs the device node each one opens.
nodes=$("$here/ttvd_nodes.sh" $(seq "$RLX_CHIP_BASE" $((RLX_CHIP_BASE + RLX_CHIPS - 1)))) || exit 1
busy() { for c in $nodes; do fuser -s "/dev/tenstorrent/$c" && return 0; done; return 1; }
while busy; do echo "$(date -u +%T) range held, waiting"; sleep 60; done
cd "$R/data"
DEV=(--i dev.star --ref emd_2660.map --ini_high 60 --firstiter_cc --auto_refine --split_random_halves
     --particle_diameter 360 --ctf --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2
     --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --norm --scale --pad 2 --sym C1
     --low_resol_join_halves 40 --dont_combine_weights_via_disc --preread_images --pool 3 --random_seed 0)
stage() { local s=$1; shift; [ -f "$runs/$s/rc" ] && return 0; echo "$(date -u +%T) stage $s"; "$@"; }
# Stops a stage once run_it<N>_optimiser.star exists (the CPU reference only needs the early walls).
cap() { local s=$1 it=$2; shift 2; "$@" & local p=$!
  while kill -0 $p 2>/dev/null; do
    [ -f "$runs/$s/run_it$(printf %03d "$it")_optimiser.star" ] && { sleep 5; pkill -TERM -f -- "--o $runs/$s/run"; sleep 60; }
    sleep 20; done; wait $p; }
TT_RELION_BACKEND=ttnn TT_RELION_OZAKI=${RLX_OZAKI:-14} TT_RELION_PROJ_CACHE_MB=4096 \
  stage dev_ttnn_bh "$here/run.sh" "$runs/dev_ttnn_bh" $n $j "${DEV[@]}" --o "$runs/dev_ttnn_bh/run"
TT_RELION_BACKEND=nullbase stage dev_ref_bh cap dev_ref_bh "${RLX_REF_ITERS:-5}" \
  "$here/run.sh" "$runs/dev_ref_bh" $n $j "${DEV[@]}" --o "$runs/dev_ref_bh/run"
echo "$(date -u +%T) CHAIN-DONE"
