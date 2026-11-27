#!/bin/bash
# Build relion_refine_mpi from a clean RELION checkout plus this branch's patches and bridge only,
# so a run never picks up a binary built from another branch's bridge (a signature mismatch makes
# the bridge fall back to CPU while the job still exits 0).
# usage: build_relion.sh <relion git repo> <commit> <out dir> [cmake]
set -euo pipefail
SRC=$1 REV=$2 OUT=$3 CMAKE=${4:-cmake}
HERE=$(cd "$(dirname "$0")/../.." && pwd)/integrations/relion
rm -rf "$OUT"
git clone -q --shared --no-checkout "$SRC" "$OUT"
cd "$OUT"
git checkout -q "$REV"
for p in relion5-tt relion5-coarse-blocking relion5-nonzeroprob relion5-estep-profile; do
  git apply "$HERE/$p.patch"
done
mkdir -p src/acc/tt build
cp "$HERE"/src_acc_tt/* src/acc/tt/
cd build
"$CMAKE" .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DALTCPU=ON -DCUDA=OFF \
  -DTT=ON -DGUI=OFF -DFETCH_WEIGHTS=OFF
make -j"$(( $(nproc) / 4 ))" refine_mpi
