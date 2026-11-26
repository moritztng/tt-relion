#!/bin/bash
# Why every 8-GPU dev run died at iteration 2 ("min_diff2= 3.4e38", acc_ml_optimiser_impl.h:2255) while the 1-GPU,
# 2-follower dev run passed: test whether the cross-follower MPI sum of each half's reconstruction is the cause.
# One GPU, dev cell, 5 ranks (2 followers per half) with default Open MPI vs ob1/vader, then MPS, then 3 ranks.
# A run passes when it reaches iteration 4. Runs as root in /work, detached; /work/box.log, DONE marker.
# Inputs in /work: build.sh, the two patches, stage.sh, subset.py, refine_cmd.sh
set -uo pipefail
W=/work; cd $W; exec > >(tee -a $W/box.log) 2>&1
st(){ echo "$(date -u +%FT%TZ) $*"; }
st start; nproc; free -g | head -2; nvidia-smi -L
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential libopenmpi-dev openmpi-bin libfftw3-dev libtiff-dev libpng-dev curl aria2 time python3 >/dev/null || { st FAIL apt; touch FAILED; exit 1; }
ompi_info | grep -E 'Open MPI:|MCA pml|MCA btl' ; df -h /dev/shm
curl -sf https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/Particles/shiny_2sets.star -o shiny_2sets.star && python3 subset.py shiny_2sets.star dev.star > dev.txt || { st FAIL star; touch FAILED; exit 1; }
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
bash build.sh fair:$ARCH > build.log 2>&1 & BP=$!
bash stage.sh data dev.txt || { st FAIL stage; touch FAILED; exit 1; }; cp dev.star data/
wait $BP || { tail -20 build.log; st FAIL build; touch FAILED; exit 1; }; cat build.log
t(){ local name=$1 n=$2 m=$3; st "run $name ($n ranks, mpirun '$m')"
  timeout 900 bash refine_cmd.sh /opt/relion-fair data dev.star $name $n 2 30 0 "" "" "$m" > $name.out 2>&1
  if tr '\r' '\n' < $name.out | grep -q 'Iteration= 4'; then st "$name PASS (reached iteration 4)"; return 0; fi
  st "$name FAIL: $(tr '\r' '\n' < $name.out | grep -m1 -E 'min_diff2|ERROR|rror' )"; return 1; }
t A-n5-default 5 ""
t B-n5-ob1 5 "--mca pml ob1 --mca btl self,vader,tcp"
t E-n5-ucx-nomemcache 5 "-x UCX_MEMTYPE_CACHE=n"
t D-n3-default 3 ""
nvidia-cuda-mps-control -d && st "MPS on" && t C-n5-default-mps 5 ""
touch $W/DONE; st done
