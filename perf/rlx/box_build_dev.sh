#!/bin/bash
# Runs as root on a rented NVIDIA box. Builds pinned RELION 5.0.x with build.sh (sm_80 fair + stock, sm_100 fair), stages the RLX dev cell, and runs one dev-cell auto-refine on GPU 0.
# Inputs in /work: build.sh relion5-coarse-blocking.patch relion5-nonzeroprob.patch dev.star files.txt refine_cmd.sh
set -uo pipefail
W=/work; cd $W; exec > >(tee -a $W/box.log) 2>&1
st(){ echo "$(date -u +%FT%TZ) $*"; }
st start; nproc; free -g | head -2; nvidia-smi -L
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential libopenmpi-dev openmpi-bin libfftw3-dev \
  libtiff-dev libpng-dev curl aria2 time python3 >/dev/null || { st FAIL apt; touch $W/FAILED; exit 1; }
st apt done
bash build.sh fair:80 stock:80:stock fair100:100 || { st FAIL build; touch FAILED; exit 1; }
/opt/relion-fair/bin/relion_refine --version | head -3; ls -la relion-builds.tgz; st built
# dev cell data, timed so the 50 GB staging plan uses a measured rate
U=https://ftp.ebi.ac.uk/empiar/world_availability/10028/data
mkdir -p data && cd data && cp ../dev.star .
python3 - <<'P'
L=[l.strip() for l in open('/work/files.txt') if l.strip()]
open('/work/aria.txt','w').write(''.join(f"https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/{p}\n  out={p}\n" for p in L))
P
t0=$(date +%s); aria2c -q -j 16 -x 4 -i ../aria.txt --auto-file-renaming=false
curl -s https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2660/map/emd_2660.map.gz | gunzip > emd_2660.map
t1=$(date +%s); du -sb Particles | awk -v s=$((t1-t0)) '{printf "DOWNLOAD %.0f bytes in %d s = %.1f MB/s\n",$1,s,$1/s/1e6}' | tee ../download.txt
cd $W
st dev run start
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 1 > smi-dev.csv 2>&1 & SMI=$!
bash refine_cmd.sh /opt/relion-fair data dev.star dev-fair 3 4 30 0 > dev-fair.out 2>&1; echo "rc=$?" >> dev-fair.out
kill $SMI; st dev run done; tail -5 dev-fair.out
touch $W/DONE
