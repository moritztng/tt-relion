#!/bin/bash
# RLX frozen job (state PROTOCOL.md): RELION 3D auto-refine of EMPIAR-10028, RELION's own GUI recipe for
# Refine3D plus the 3dem benchmark settings. Only ranks, --j, --pool, --gpu and the seed vary between runs.
#   refine_cmd.sh <relion prefix> <data dir> <star> <out name> <ranks> <j> <pool> <seed> [gpu map] [extra]
set -uo pipefail
R=$1 D=$2 STAR=$3 O=$4 N=$5 J=$6 P=$7 S=$8 G=${9:-} X=${10:-}
mkdir -p "$PWD/$O"; OUT=$PWD/$O
cd "$D"; export LD_LIBRARY_PATH=$R/lib:${LD_LIBRARY_PATH:-}
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "WARN page cache not dropped"
/usr/bin/time -v -o "$OUT/time-v.txt" mpirun --allow-run-as-root --bind-to none -x LD_LIBRARY_PATH -n "$N" \
  "$R/bin/relion_refine_mpi" --o "$OUT/run" --auto_refine --split_random_halves --i "$STAR" \
  --ref emd_2660.map:mrc --firstiter_cc --ini_high 60 --dont_combine_weights_via_disc --preread_images \
  --pool "$P" --pad 2 --ctf --particle_diameter 360 --flatten_solvent --zero_mask --oversampling 1 \
  --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym C1 \
  --low_resol_join_halves 40 --norm --scale --random_seed "$S" --j "$J" --gpu "$G" $X
