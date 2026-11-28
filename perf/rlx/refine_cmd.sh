#!/bin/bash
# RLX frozen job (state PROTOCOL.md): RELION 3D auto-refine of EMPIAR-10028, RELION's own GUI recipe for
# Refine3D plus the 3dem benchmark settings. Only ranks, --j, --pool, --gpu and the seed vary between runs.
#   refine_cmd.sh <relion prefix> <data dir> <star> <out name> <ranks> <j> <pool> <seed> [gpu map] [extra] [mpirun opts] [images: auto|ram|cache]
set -uo pipefail
R=$1 D=$2 STAR=$3 O=$4 N=$5 J=$6 P=$7 S=$8 G=${9:-} X=${10:-} M=${11:-} IM=${12:-auto}
mkdir -p "$PWD/$O"; OUT=$PWD/$O
RD=--preread_images; case "$X" in *--scratch_dir*) RD=;; esac   # scratch replaces RAM only when RAM is short
case "$IM" in cache) RD=;; ram|auto) ;; *) echo "bad images mode $IM"; exit 2;; esac
# Every follower holds ALL images with --preread_images, so it is on only while (ranks-1) x set fits 60 % of RAM;
# otherwise images come from the page cache (the box has the RAM for one copy).
[ -n "$RD" ] && [ "$IM" = auto ] && python3 - "$D" "$STAR" "$N" <<'PY' || RD=
import sys, os, glob
d, star, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = sum(1 for l in open(os.path.join(d, star)) if '.mrcs' in l)
stacks = glob.glob(os.path.join(d, '**', '*.mrcs'), recursive=True); per = sum(os.path.getsize(f) for f in stacks) / 105247
mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
need = (n - 1) * rows * per; print(f"preread need {need/1e9:.0f} GB of {mem/1e9:.0f} GB RAM"); sys.exit(0 if need < 0.6 * mem else 1)
PY
echo "images: ${RD:-page cache}"
# Host levers (PROTOCOL "Host environment"): huge pages for large mallocs on every rank. stdout goes through tsline.py
# when it sits beside this script, so each line carries its wall-clock start and end (the E-step/host split).
HERE=$(dirname "$(realpath "$0")"); TS=cat; [ -s "$HERE/tsline.py" ] && TS="python3 -u $HERE/tsline.py"
export GLIBC_TUNABLES=${GLIBC_TUNABLES-glibc.malloc.hugetlb=1}; echo "GLIBC_TUNABLES=$GLIBC_TUNABLES THP=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
cd "$D"; export LD_LIBRARY_PATH=$R/lib:${LD_LIBRARY_PATH:-}
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "WARN page cache not dropped"
/usr/bin/time -v -o "$OUT/time-v.txt" mpirun --allow-run-as-root --bind-to none -x LD_LIBRARY_PATH -x GLIBC_TUNABLES $M -n "$N" \
  "$R/bin/relion_refine_mpi" --o "$OUT/run" --auto_refine --split_random_halves --i "$STAR" \
  --ref emd_2660.map:mrc --firstiter_cc --ini_high 60 --dont_combine_weights_via_disc $RD \
  --pool "$P" --pad 2 --ctf --particle_diameter 360 --flatten_solvent --zero_mask --oversampling 1 \
  --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym C1 \
  --low_resol_join_halves 40 --norm --scale --random_seed "$S" --j "$J" --gpu "$G" $X | $TS; exit ${PIPESTATUS[0]}
