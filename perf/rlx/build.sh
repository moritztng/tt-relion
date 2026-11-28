#!/bin/bash
# Build pinned RELION 5.0.x into /opt/relion-<name> on a rented NVIDIA box and tar them to /work/relion-builds.tgz.
# fair = our CPU fixes (tt-relion integrations/relion/*.patch), stock = upstream at the same commit. The fair build
# carries relion5-host-threads.patch when it is in /work; that one needs RELION's own FFTW (built with threads).
# CUDA_ARCH takes one arch, so each card generation gets its own build.
#   build.sh <name:arch[:stock]> ...      e.g. build.sh fair:80 stock:80:stock fair100:100
# Inputs in /work: relion5-coarse-blocking.patch relion5-nonzeroprob.patch
set -uo pipefail
SHA=7f45a711105bc6c15976c2e47d087233d643f568
W=/work; cd $W
[ -d src ] || { git clone -q https://github.com/3dem/relion src && git -C src checkout -q $SHA; } || exit 1
for spec in "$@"; do IFS=: read -r name arch stock <<< "$spec"; rm -rf b-$name; cp -a src b-$name
  [ -n "$stock" ] || for p in relion5-coarse-blocking.patch relion5-nonzeroprob.patch; do git -C b-$name apply $W/$p || exit 1; done
  own=; [ -z "$stock" ] && [ -s $W/relion5-host-threads.patch ] && { git -C b-$name apply $W/relion5-host-threads.patch || exit 1; own=-DFORCE_OWN_FFTW=ON; }
  mkdir b-$name/build && cd b-$name/build
  cmake .. -DCUDA=ON -DCUDA_ARCH=$arch -DGUI=OFF -DFETCH_WEIGHTS=OFF -DCMAKE_BUILD_TYPE=Release $own \
    -DCMAKE_INSTALL_PREFIX=/opt/relion-$name > $W/cmake-$name.log 2>&1 && make -j"$(nproc)" install > $W/make-$name.log 2>&1 ||
    { tail -30 $W/make-$name.log; exit 1; }
  cd $W; [ -z "$own" ] || ldd /opt/relion-$name/bin/relion_refine_mpi | grep -q fftw3_threads || { echo "$name: no threaded FFTW linked"; exit 1; }
  echo "built $name sm_$arch $(cuobjdump --list-elf /opt/relion-$name/bin/relion_refine_mpi 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
done
tar -C /opt -czf relion-builds.tgz $(cd /opt; ls -d relion-*)
