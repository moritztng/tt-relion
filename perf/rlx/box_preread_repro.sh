#!/bin/bash
# The 1-GPU MPI repro failed every variant at iteration 2 ("min_diff2= 3.4e38"), 3 ranks included, so MPI is not the
# cause. The only passing run (A5000, 3 ranks, --j 4) read images from disk; every failing run used --preread_images.
# This checks every staged stack, then runs the dev cell from page cache and from RAM at 3 ranks, then 5 ranks cached.
# A run passes when it reaches iteration 4. Runs as root in /work, detached; /work/box.log, DONE marker.
# Inputs in /work: build.sh, the two patches, stage.sh, subset.py, refine_cmd.sh, check_stacks.py
set -uo pipefail
W=/work; cd $W; exec > >(tee -a $W/box.log) 2>&1
st(){ echo "$(date -u +%FT%TZ) $*"; }
st start; nproc; free -g | head -2; nvidia-smi -L
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential libopenmpi-dev openmpi-bin libfftw3-dev libtiff-dev libpng-dev curl aria2 time python3 >/dev/null || { st FAIL apt; touch FAILED; exit 1; }
curl -sf https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/shiny_2sets.star -o shiny_2sets.star && python3 subset.py shiny_2sets.star dev.star > dev.txt || { st FAIL star; touch FAILED; exit 1; }
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
bash build.sh fair:$ARCH > build.log 2>&1 & BP=$!
bash stage.sh data dev.txt || { st FAIL stage; touch FAILED; exit 1; }; cp dev.star data/
python3 check_stacks.py data | tail -20; st "stack check rc=${PIPESTATUS[0]}"
wait $BP || { tail -20 build.log; st FAIL build; touch FAILED; exit 1; }; tail -2 build.log
t(){ local name=$1 n=$2 j=$3 im=$4; st "run $name ($n ranks, --j $j, images $im)"
  timeout 900 bash refine_cmd.sh /opt/relion-fair data dev.star $name $n $j 30 0 "" "" "" $im > $name.out 2>&1
  if tr '\r' '\n' < $name.out | grep -q 'Iteration= 4'; then st "$name PASS (reached iteration 4)"; return 0; fi
  st "$name FAIL: $(tr '\r' '\n' < $name.out | grep -m1 -E 'min_diff2|ERROR|rror' )"; return 1; }
t P-n3-j4-cache 3 4 cache
t Q-n3-j4-ram 3 4 ram
t R-n3-j2-cache 3 2 cache
t S-n5-j2-cache 5 2 cache
touch $W/DONE; st done
