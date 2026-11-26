#!/bin/bash
# Timed RLX runs on a rented 8-GPU box. Runs as root in /work, detached; progress in /work/box.log, DONE/FAILED marker.
# Inputs in /work: subset.py, stage.sh, refine_cmd.sh, and either relion-builds.tgz or build.sh
# plus the two patches (then it builds while the data stages).
#   box_timed.sh <a100|b200> "<ranks j pool;...>|auto" <seeds> [stock]
# auto = 1, 2, 3 ranks per GPU plus a leader, --j min(6, cores/ranks), --pool 100 (NVIDIA's run_relion.sh), and
# 2 ranks per GPU at --pool 30. MPS is on when the daemon starts. A full run that fails with images in RAM is
# repeated with --scratch_dir on local disk.
# Each "ranks j pool" config runs on the dev cell; the fastest runs the full set for every seed; "stock" adds one
# full stock-RELION run with that config at seed 0.
set -uo pipefail
KIND=$1 CONFIGS=$2 SEEDS=$3 STOCK=${4:-}
W=/work; cd $W; exec > >(tee -a $W/box.log) 2>&1
st(){ echo "$(date -u +%FT%TZ) $*"; }
fail(){ st "FAIL $*"; touch $W/FAILED; exit 1; }
st start $KIND; nproc; free -g | head -2; nvidia-smi -L
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential libopenmpi-dev openmpi-bin libfftw3-dev libtiff-dev libpng-dev curl aria2 time python3 >/dev/null || fail apt
[ -s shiny_2sets.star ] || curl -sf https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/shiny_2sets.star -o shiny_2sets.star || fail star
python3 subset.py shiny_2sets.star dev.star > dev.txt && python3 subset.py shiny_2sets.star full.star --n 0 > full.txt || fail subset
if [ -s relion-builds.tgz ]; then tar -C /opt -xzf relion-builds.tgz || fail untar; BP=
else ARCH=80; [ $KIND = b200 ] && ARCH=100; bash build.sh fair:$ARCH stock:$ARCH:stock > build.log 2>&1 & BP=$!; fi
# The dev cell (1/8) stages first and the sweep runs on it while nothing else is paid for, so a config that cannot run
# costs minutes, not the 2 h full staging (11-26: 51 GB staged, then every config aborted at iteration 2).
bash stage.sh data dev.txt || fail stage; cp dev.star full.star data/
[ -z "$BP" ] || { wait $BP || { tail -30 build.log; fail build; }; cat build.log; st built; }
FAIR=/opt/relion-fair; [ $KIND = b200 ] && [ -d /opt/relion-fair100 ] && FAIR=/opt/relion-fair100
nvidia-cuda-mps-control -d && st "MPS on" || st "MPS unavailable"
NG=$(nvidia-smi -L | wc -l) NC=$(nproc)
[ "$CONFIGS" = auto ] && CONFIGS=$(for r in 1 2 3; do n=$((NG*r+1)); j=$((NC/n)); [ $j -gt 6 ] && j=6; [ $j -lt 1 ] && j=1
  printf "%s %s 100;" $n $j; done; n=$((NG*2+1)); j=$((NC/n)); [ $j -gt 6 ] && j=6; echo "$n $j 30")
smi(){ nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 1 > smi-$1.csv 2>&1 & echo $!; }
run(){ local name=$1 relion=$2 star=$3 n=$4 j=$5 p=$6 s=$7 x=${8:-} pid; rm -rf $name; pid=$(smi $name); st "run $name start ($n ranks --j $j --pool $p seed $s)"
  bash refine_cmd.sh $relion data $star $name $n $j $p $s "" "$x" > $name.out 2>&1; local rc=$?; kill $pid
  st "run $name rc=$rc wall=$(awk -F': ' '/Elapsed/{print $2}' $name/time-v.txt) $(grep -h 'Auto-refine: Refinement has converged\|Final resolution' $name.out | tail -1)"; return $rc; }
best= bt=
IFS=';' read -ra CS <<< "$CONFIGS"
for c in "${CS[@]}"; do set -- $c; name=dev-n$1-j$2-p$3
  run $name $FAIR dev.star $1 $2 $3 0 || continue
  t=$(python3 -c "import sys;h,m,s=(['0']+sys.argv[1].split(':'))[-3:];print(int(h)*3600+int(m)*60+float(s))" "$(awk -F': ' '/Elapsed/{print $2}' $name/time-v.txt)")
  st "config $c dev wall ${t}s"; if [ -z "$bt" ] || python3 -c "import sys;sys.exit(0 if $t < $bt else 1)"; then best=$c bt=$t; fi
done
[ -n "$best" ] || fail "no config ran"; st "BEST $best (${bt}s on the dev cell)"; set -- $best
bash stage.sh data full.txt || fail stage
X=
full(){ run $1 $2 full.star $3 $4 $5 $6 "$X" && return; [ -n "$X" ] && return 1
  X="--scratch_dir $W/scratch"; mkdir -p $W/scratch; st "$1 failed with images in RAM, repeating with $X"; run $1 $2 full.star $3 $4 $5 $6 "$X"; }
for s in $SEEDS; do full full-s$s $FAIR $1 $2 $3 $s; done
[ -n "$STOCK" ] && full full-stock /opt/relion-stock $1 $2 $3 0
touch $W/DONE; st done
