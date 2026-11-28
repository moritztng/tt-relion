#!/bin/bash
# ab_chain.sh: prep A/B on RELION's CPU kernels (no chip), one binary, three modes per arm:
#   n  TT_RELION_PREP=0, RELION's own code
#   e  TTPrep with estimated FFT plans (same bits as RELION's transform)
#   w  TTPrep with the plans of $R/wisdom.<box> (measured once here, the same for every rank and run)
# First the record arms of both iteration types, whose first particles go to check_prep.py (float64
# reference), then per iteration type the profile arms n w e w n (thread CPU per particle by stage, split.py).
# Status: $R/runs/$P.status. Env: R (~/rlx-prep), RELION_BIN, P (tag prefix, ab), BOX (360),
# IT16 / IT24 (optimiser.star to continue from), N16 / N24 (particles per rank before an arm stops).
R=${R:-$HOME/rlx-prep}; here=$(cd "$(dirname "$0")" && pwd); src=$(cd "$here/../../.." && pwd)
P=${P:-ab}; S=$R/runs/$P.status; W=$R/wisdom.${BOX:-360}
export R RELION_BIN=${RELION_BIN:-$R/relion/build/bin/relion_refine_mpi}
py=${PY:-$HOME/fleet/env/bin/python}
IT16=${IT16:-$HOME/rlx/runs/dev_ref_bh/run_it015_optimiser.star}; IT24=${IT24:-$HOME/rlx/runs/d32_l_n14/run_it023_optimiser.star}
mode() { # n|e|w -> the two switches
  case $1 in n) export TT_RELION_PREP=0; unset TT_RELION_PREP_WISDOM;; e) export TT_RELION_PREP=1; unset TT_RELION_PREP_WISDOM;;
             w) export TT_RELION_PREP=1 TT_RELION_PREP_WISDOM=$W;; esac
}
arm() { # arm <tag> <mode> <optimiser> <it> <n>
  mode $2; N=$5 "$here/../wsum/prof_arm.sh" ${P}_$1 $3 $4
  $py "$here/../wsum/split.py" $R/runs/${P}_$1 > $R/runs/${P}_$1/split.txt 2>&1
  echo "$1 mode=$2 $(tail -1 $R/runs/${P}_$1/status) | $(grep -m1 fft_ctf $R/runs/${P}_$1/split.txt) | $(grep -m1 ttPrep $R/runs/${P}_$1/split.txt | xargs)" >> $S
}
rec() { # rec <tag> <mode> <optimiser> <it>
  mkdir -p $R/runs/${P}_rec_$1; mode $2
  TT_RELION_RECORD_PREP=$R/runs/${P}_rec_$1 TT_RELION_RECORD_PREP_N=16 RANKS=2 N=256 "$here/../wsum/prof_arm.sh" ${P}_$1 $3 $4
  echo "$1 mode=$2 records $(ls $R/runs/${P}_rec_$1 | wc -l) wisdom: $(grep -m1 -h TTPrep: $R/runs/${P}_$1/*.log $R/runs/${P}_$1/run.* 2>/dev/null | head -1)" >> $S
}
check() { # check <it tag>: every mode against float64, and e and w against n
  for m in n e w; do
    echo "== $1 $m vs float64, and vs n" >> $R/runs/$P.check
    $py "$here/check_prep.py" $R/runs/${P}_rec_r$m$1 $R/runs/${P}_rec_rn$1 >> $R/runs/$P.check 2>&1
  done
}
echo "start $(date -u +%FT%TZ) bin $RELION_BIN" >> $S
if [ ! -s $W ]; then
  g++ -O3 -fopenmp -I "$src/integrations/relion/src_acc_tt" "$here/bench_prep.cpp" "$src/integrations/relion/src_acc_tt/tt_prep.cpp" \
      -lfftw3f -ltbb -o $R/bench_prep.$P && $R/bench_prep.$P wisdom ${BOX:-360} $W
  echo "wisdom $W $(wc -c < $W) bytes $(date -u +%FT%TZ)" >> $S
fi
for m in n e w; do rec r${m}16 $m $IT16 16; done; check 16
for m in n e w; do rec r${m}24 $m $IT24 24; done; check 24
echo "records checked $(date -u +%FT%TZ)" >> $S
arm n16a n $IT16 16 ${N16:-768}; arm w16a w $IT16 16 ${N16:-768}; arm e16a e $IT16 16 ${N16:-768}
arm w16b w $IT16 16 ${N16:-768}; arm n16b n $IT16 16 ${N16:-768}
arm n24a n $IT24 24 ${N24:-2048}; arm w24a w $IT24 24 ${N24:-2048}; arm e24a e $IT24 24 ${N24:-2048}
arm w24b w $IT24 24 ${N24:-2048}; arm n24b n $IT24 24 ${N24:-2048}
echo "done $(date -u +%FT%TZ)" >> $S
