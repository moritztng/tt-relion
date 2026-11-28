#!/bin/bash
# ab_chain.sh: prep A/B on RELION's CPU kernels (no chip), one binary, TT_RELION_PREP=0 (N) vs default (T).
# Per iteration type: profile arms N T T N (thread CPU per particle by stage, split.py), then two record arms
# whose first particles go to check_prep.py. Status: $R/runs/ab.status. Env: R (~/rlx-prep), RELION_BIN,
# IT16 / IT24 (optimiser.star to continue from), N16 / N24 (particles per rank before an arm stops).
R=${R:-$HOME/rlx-prep}; here=$(cd "$(dirname "$0")" && pwd); S=$R/runs/ab.status
export R RELION_BIN=${RELION_BIN:-$R/relion-b/build/bin/relion_refine_mpi}
py=${PY:-$HOME/fleet/env/bin/python}
IT16=${IT16:-$HOME/rlx/runs/dev_ref_bh/run_it015_optimiser.star}; IT24=${IT24:-$HOME/rlx/runs/d32_l_n14/run_it023_optimiser.star}
arm() { # arm <tag> <prep 0|1> <optimiser> <it> <n>
  TT_RELION_PREP=$2 N=$5 "$here/../wsum/prof_arm.sh" $1 $3 $4
  $py "$here/../wsum/split.py" $R/runs/$1 > $R/runs/$1/split.txt 2>&1
  echo "$1 prep=$2 $(tail -1 $R/runs/$1/status) | $(grep -m1 fft_ctf $R/runs/$1/split.txt)" >> $S
}
rec() { # rec <tag> <prep 0|1> <optimiser> <it>
  mkdir -p $R/runs/rec_$1
  TT_RELION_RECORD_PREP=$R/runs/rec_$1 TT_RELION_RECORD_PREP_N=16 TT_RELION_PREP=$2 RANKS=2 N=256 "$here/../wsum/prof_arm.sh" $1 $3 $4
  echo "$1 prep=$2 records $(ls $R/runs/rec_$1 | wc -l)" >> $S
}
echo "start $(date -u +%FT%TZ) bin $RELION_BIN" >> $S
arm n16a 0 $IT16 16 ${N16:-768}; arm t16a 1 $IT16 16 ${N16:-768}; arm t16b 1 $IT16 16 ${N16:-768}; arm n16b 0 $IT16 16 ${N16:-768}
arm n24a 0 $IT24 24 ${N24:-2048}; arm t24a 1 $IT24 24 ${N24:-2048}; arm t24b 1 $IT24 24 ${N24:-2048}; arm n24b 0 $IT24 24 ${N24:-2048}
rec rn16 0 $IT16 16; rec rt16 1 $IT16 16; rec rn24 0 $IT24 24; rec rt24 1 $IT24 24
echo "done $(date -u +%FT%TZ)" >> $S
