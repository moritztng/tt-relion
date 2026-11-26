#!/bin/bash
# Timed RLX runs on a rented 8-GPU box. Runs as root in /work, detached; progress in /work/box.log, DONE/FAILED marker.
# Inputs in /work: relion-builds.tgz (from box_build_dev.sh), shiny_2sets.star, subset.py, stage.sh, refine_cmd.sh.
#   box_timed.sh <a100|b200> "<ranks j pool;...>" <seeds> [stock]
# Each "ranks j pool" config runs on the dev cell; the fastest runs the full set for every seed; "stock" adds one
# full stock-RELION run with that config at seed 0.
set -uo pipefail
KIND=$1 CONFIGS=$2 SEEDS=$3 STOCK=${4:-}
W=/work; cd $W; exec > >(tee -a $W/box.log) 2>&1
st(){ echo "$(date -u +%FT%TZ) $*"; }
fail(){ st "FAIL $*"; touch $W/FAILED; exit 1; }
st start $KIND; nproc; free -g | head -2; nvidia-smi -L
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq libopenmpi-dev openmpi-bin libfftw3-dev libtiff-dev libpng-dev curl aria2 time python3 >/dev/null || fail apt
tar -C /opt -xzf relion-builds.tgz || fail untar
FAIR=/opt/relion-fair; [ $KIND = b200 ] && [ -d /opt/relion-fair100 ] && FAIR=/opt/relion-fair100
python3 subset.py shiny_2sets.star dev.star > dev.txt && python3 subset.py shiny_2sets.star full.star --n 0 > full.txt || fail subset
bash stage.sh data full.txt || fail stage; cp dev.star full.star data/
smi(){ nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 1 > smi-$1.csv 2>&1 & echo $!; }
run(){ local name=$1 relion=$2 star=$3 n=$4 j=$5 p=$6 s=$7 pid; pid=$(smi $name); st "run $name start ($n ranks --j $j --pool $p seed $s)"
  bash refine_cmd.sh $relion data $star $name $n $j $p $s > $name.out 2>&1; local rc=$?; kill $pid
  st "run $name rc=$rc wall=$(awk -F': ' '/Elapsed/{print $2}' $name/time-v.txt) $(grep -h 'Auto-refine: Refinement has converged\|Final resolution' $name.out | tail -1)"; return $rc; }
best= bt=
IFS=';' read -ra CS <<< "$CONFIGS"
for c in "${CS[@]}"; do set -- $c; name=dev-n$1-j$2-p$3
  run $name $FAIR dev.star $1 $2 $3 0 || continue
  t=$(python3 -c "import sys;h,m,s=(['0']+sys.argv[1].split(':'))[-3:];print(int(h)*3600+int(m)*60+float(s))" "$(awk -F': ' '/Elapsed/{print $2}' $name/time-v.txt)")
  st "config $c dev wall ${t}s"; if [ -z "$bt" ] || python3 -c "import sys;sys.exit(0 if $t < $bt else 1)"; then best=$c bt=$t; fi
done
[ -n "$best" ] || fail "no config ran"; st "BEST $best (${bt}s on the dev cell)"; set -- $best
for s in $SEEDS; do run full-s$s $FAIR full.star $1 $2 $3 $s; done
[ -n "$STOCK" ] && run full-stock /opt/relion-stock full.star $1 $2 $3 0
touch $W/DONE; st done
