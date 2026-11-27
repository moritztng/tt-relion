#!/bin/bash
# Dev-cell measured wall: one resident E-step process per chip, all chips at once, each on its own particle slice
# (--seed = chip). Per chip: the global finer pass shape (B256 x T13 x O4096 per batch, s50, on-chip operand +
# candidate compaction) for NG batches, then the local pass (B128 particles x own 160 of a 4096 table, s50) for NL
# rounds, then the final-pass route (per-particle lerp, s352 stand-in for s360, B8 x own 160) for NF rounds.
# Wall per chip and for the cell; AICLK and load1 sampled inside each process. budget_model.py --from-dc <run dir>
# turns the worst chip's rates into the full-job per-chip budget.
#   devcell.sh "17 18 19" NG NL   (run from the run dir; out_<chip>.jsonl, <chip>_{g,l}.log, wall.tsv)
CHIPS=${1:?chips}; NG=${2:-9}; NL=${3:-4}; NF=${4:-2}
# host threads per chip process: 64 cores / 32 processes; unset, every process spawns 64 BLAS/OMP threads
HT=${HT:-2}
export OMP_NUM_THREADS=$HT MKL_NUM_THREADS=$HT OPENBLAS_NUM_THREADS=$HT
export PATH=~/fleet/env/bin:$PATH PYTHONPATH=${SRC:-~/rlx-resident/src} GLIBC_TUNABLES=glibc.malloc.hugetlb=1
P=${SRC:-~/rlx-resident/src}/perf/rlx/resident/global_pass.py
D="--run $HOME/rlx/runs/dev_ref_bh --it 18 --data $HOME/rlx/data --fast"
[ -e ~/fleet/HANDBACK ] && { echo HANDBACK; exit 3; }
echo "start $(date -u +%FT%TZ) chips=[$CHIPS] NG=$NG NL=$NL load: $(cat /proc/loadavg)"
t0=$(date +%s.%N)
for c in $CHIPS; do
  ( s=$(date +%s.%N)
    TT_VISIBLE_DEVICES=$c timeout -k 30 -s INT ${TO:-900} python $P $D --seed $c --size 50 --B 256 --O 4096 --pchip 1 --cand 1 --reps $NG --out out_$c.jsonl > ${c}_g.log 2>&1; rg=$?
    m=$(date +%s.%N)
    TT_VISIBLE_DEVICES=$c timeout -k 30 -s INT ${TO:-900} python $P $D --seed $c --size 50 --B 128 --O 4096 --local 1 --reps $NL --out out_$c.jsonl > ${c}_l.log 2>&1; rl=$?
    e=$(date +%s.%N)
    TT_VISIBLE_DEVICES=$c timeout -k 30 -s INT ${TO:-900} python $P $D --seed $c --size 352 --B 8 --O 256 --local 1 --lerp 1 --reps $NF --out out_f_$c.jsonl > ${c}_f.log 2>&1; rf=$?
    f=$(date +%s.%N)
    echo -e "$c\t$rg\t$rl\t$rf\t$(echo "$m - $s" | bc)\t$(echo "$e - $m" | bc)\t$(echo "$f - $e" | bc)" >> wall.tsv ) &
done
wait
echo "end $(date -u +%FT%TZ) cell_wall_s=$(echo "$(date +%s.%N) - $t0" | bc)"
