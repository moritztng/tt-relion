#!/bin/bash
# replay_chips.sh <record tag> <arm tag> "<chips>" [replay.py args...]
# One replay process per chip, each a rank of its own on that chip (record files of the tag in order, reused
# round-robin), all at once, each pinned to CORES host cores. The arm is the environment: TT_RELION_BACKEND,
# TT_RELION_RESIDENT, TT_RELION_BATCH, TT_RELION_BATCH_WAIT_MS. Every chip is taken under its tt-bio lease
# flock, so a chip someone holds fails the arm instead of sharing it. Results: ~/rlx-bridge/runs/<arm tag>/.
# Refuses to start with ~/rlx-bridge/STOP or ~/fleet/HANDBACK present, or after NOT_AFTER (epoch).
rtag=${1:?record tag}; tag=${2:?arm tag}; chips=${3:?chips}; shift 3
R=$HOME/rlx-bridge; here=$(cd "$(dirname "$0")" && pwd); src=$(cd "$here/../../.." && pwd)
o=$R/runs/$tag; mkdir -p $o; S=$R/runs/replay.status
[ -e $R/STOP ] || [ -e $HOME/fleet/HANDBACK ] && { echo "$tag STOP/HANDBACK" >> $S; exit 3; }
[ -n "${NOT_AFTER:-}" ] && [ $(date +%s) -gt $NOT_AFTER ] && { echo "$tag past NOT_AFTER" >> $S; exit 3; }
export PYTHONPATH=$src:$HOME/fleet/env/lib/python3.10/site-packages TT_RELION_PYTHON=$HOME/fleet/env/bin/python
export TT_METAL_INSPECTOR=0 TT_RELION_TORCH_THREADS=1 OMP_NUM_THREADS=${HT:-2} MKL_NUM_THREADS=${HT:-2} OPENBLAS_NUM_THREADS=${HT:-2}
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
recs=($R/rec/$rtag/rec.*.pkl); cores=${CORES:-8}; base=${CORE_BASE:-0}; i=0
echo "$tag start $(date -u +%FT%TZ) chips=[$chips] load: $(cat /proc/loadavg) env: $(env | grep '^TT_RELION_\(BACKEND\|RESIDENT\|BATCH\)' | tr '\n' ' ') args: $*" >> $S
for c in $chips; do
  rec=${recs[$((i % ${#recs[@]}))]}; lo=$((base + i * cores))
  ( exec {fd}>"/tmp/tt-bio-device-leases/$(hostname)-card$c.json"
    flock -n $fd || { echo "$tag chip $c is leased by someone else" >> $S; exit 3; }
    printf '{"host":"%s","card":"%s","holder":"worker:rlx-bridge","pid":%d,"acquired":%d,"released":null}\n' "$(hostname)" "$c" $$ "$(date +%s)" >&$fd
    RLX_RANK_CHIPS=$c TT_VISIBLE_DEVICES=$c timeout -k 20 -s INT ${TO:-900} taskset -c $lo-$((lo + cores - 1)) \
      $TT_RELION_PYTHON $src/perf/rlx/bridge/replay.py --rec $rec --out $o/replay.jsonl --digest $o/d.$c.npz \
      --ref $R/runs/ref/$rtag.$(basename $rec .pkl).npz --tag $tag.$c "$@" > $o/$c.log 2>&1
    echo "$tag chip $c rc=$? $(date -u +%FT%TZ) load1 $(cut -d' ' -f1 /proc/loadavg)" >> $S ) &
  i=$((i + 1))
done
wait
