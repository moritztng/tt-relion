#!/bin/bash
# Runs as root on a rented NVIDIA box. Builds pinned RELION 5.0.x twice (our two CPU fixes, and stock) as fat
# CUDA binaries (sm_80 + sm_90 + sm_100), stages the RLX dev cell, and runs one dev-cell auto-refine on GPU 0.
# Inputs in /work: relion5-coarse-blocking.patch relion5-nonzeroprob.patch dev.star files.txt refine_cmd.sh
set -uo pipefail
SHA=7f45a711105bc6c15976c2e47d087233d643f568
W=/work; cd $W; exec > >(tee -a $W/box.log) 2>&1
st(){ echo "$(date -u +%FT%TZ) $*"; }
st start; nproc; free -g | head -2; nvidia-smi -L
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential libopenmpi-dev openmpi-bin libfftw3-dev \
  libtiff-dev libpng-dev curl aria2 time python3 >/dev/null || { st FAIL apt; touch $W/FAILED; exit 1; }
st apt done
git clone -q https://github.com/3dem/relion src && git -C src checkout -q $SHA || { st FAIL clone; touch FAILED; exit 1; }
build(){ local name=$1 arch=$2 extra=$3; shift 3; rm -rf b-$name; cp -a src b-$name
  for p in "$@"; do git -C b-$name apply $W/$p || return 1; done
  mkdir b-$name/build && cd b-$name/build
  cmake .. -DCUDA=ON -DCUDA_ARCH=$arch -DEXTRA_NVCC_FLAGS="$extra" -DGUI=OFF -DFETCH_WEIGHTS=OFF \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/relion-$name > $W/cmake-$name.log 2>&1 &&
  make -j"$(nproc)" install > $W/make-$name.log 2>&1; local rc=$?; cd $W; return $rc; }
FAT="-gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100"
build fair 80 "$FAT" relion5-coarse-blocking.patch relion5-nonzeroprob.patch || { st FAIL build fair; tail -30 make-fair.log; touch FAILED; exit 1; }
st built fair
cuobjdump --list-elf /opt/relion-fair/bin/relion_refine_mpi 2>/dev/null | grep -o 'sm_[0-9]*' | sort | uniq -c | tee archs-fair.txt
if ! grep -q sm_100 archs-fair.txt; then st "no sm_100 in fat build; building sm_100 separately"
  build fair100 100 "" relion5-coarse-blocking.patch relion5-nonzeroprob.patch && st built fair100 || st FAIL fair100; fi
build stock 80 "$FAT" && st built stock || st FAIL build stock
/opt/relion-fair/bin/relion_refine --version | head -3
tar -C /opt -czf relion-builds.tgz $(cd /opt; ls -d relion-*); ls -la relion-builds.tgz; st tarred
# dev cell data, timed so the 50 GB staging plan uses a measured rate
U=https://ftp.ebi.ac.uk/empiar/world_availability/10028/data
mkdir -p data && cd data && cp ../dev.star .
python3 - <<'P'
L=[l.strip() for l in open('/work/files.txt') if l.strip()]
open('/work/aria.txt','w').write(''.join(f"https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/{p}\n  out={p}\n" for p in L))
P
t0=$(date +%s); aria2c -q -j 16 -x 4 -i ../aria.txt --auto-file-renaming=false
curl -s https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2660/map/emd_2660.map.gz | gunzip > emd_2660.map
t1=$(date +%s); du -sb Particles | awk -v s=$((t1-t0)) '{printf "DOWNLOAD %d bytes in %d s = %.1f MB/s\n",$1,s,$1/s/1e6}' | tee ../download.txt
cd $W
st dev run start
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 1 > smi-dev.csv 2>&1 & SMI=$!
bash refine_cmd.sh /opt/relion-fair data dev.star dev-fair 3 4 30 0 > dev-fair.out 2>&1; echo "rc=$?" >> dev-fair.out
kill $SMI; st dev run done; tail -5 dev-fair.out
touch $W/DONE
