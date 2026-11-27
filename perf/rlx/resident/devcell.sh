#!/bin/bash
# Dev-cell measured wall: one resident E-step process per chip, all chips at once, each on its own particle slice
# (--seed = chip). Per chip: the global finer pass shape (B256 x T13 x O4096 per batch, s50, on-chip operand +
# candidate compaction) for NG batches, then the local pass (B128 particles x own 160 of a 4096 table, s50) for NL
# rounds. Wall per chip and for the cell; AICLK and load1 sampled inside each process.
#   devcell.sh "17 18 19" NG NL   (run from the run dir; out_<chip>.jsonl, <chip>_{g,l}.log, wall.tsv)
CHIPS=${1:?chips}; NG=${2:-9}; NL=${3:-4}
export PATH=~/fleet/env/bin:$PATH PYTHONPATH=~/rlx-resident/src GLIBC_TUNABLES=glibc.malloc.hugetlb=1
P=~/rlx-resident/src/perf/rlx/resident/global_pass.py
D="--run $HOME/rlx/runs/dev_ref_bh --it 18 --data $HOME/rlx/data --fast"
[ -e ~/fleet/HANDBACK ] && { echo HANDBACK; exit 3; }
echo "start $(date -u +%FT%TZ) chips=[$CHIPS] NG=$NG NL=$NL load: $(cat /proc/loadavg)"
t0=$(date +%s.%N)
for c in $CHIPS; do
  ( s=$(date +%s.%N)
    TT_VISIBLE_DEVICES=$c timeout 900 python $P $D --seed $c --size 50 --B 256 --O 4096 --pchip 1 --cand 1 --reps $NG --out out_$c.jsonl > ${c}_g.log 2>&1; rg=$?
    m=$(date +%s.%N)
    TT_VISIBLE_DEVICES=$c timeout 900 python $P $D --seed $c --size 50 --B 128 --O 4096 --local 1 --reps $NL --out out_$c.jsonl > ${c}_l.log 2>&1; rl=$?
    e=$(date +%s.%N)
    echo -e "$c\t$rg\t$rl\t$(echo "$m - $s" | bc)\t$(echo "$e - $m" | bc)" >> wall.tsv ) &
done
wait
echo "end $(date -u +%FT%TZ) cell_wall_s=$(echo "$(date +%s.%N) - $t0" | bc)"
