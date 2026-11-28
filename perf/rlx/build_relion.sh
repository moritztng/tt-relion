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
# relion5-host-threads (rlx-mstep3): the host side between E-steps on --j threads (slab FFTs, threaded map loops);
# HOST_THREADS=0 leaves it out, for a binary to compare against. JOBS = make -j (default a quarter of the cores).
# relion5-sums-sparse (rlx-wsum): the sums kernels skip samples below significance; SUMS_SPARSE=0 leaves it out.
# relion5-prep (rlx-prep): particle prep on tables and its own FFT plans; TT_RELION_PREP=0 at run time takes RELION's code.
# relion5-tt-candidates (rlx-bridge): the coarse pass may hand back candidates instead of the dense matrix (used only
# with TT_RELION_CAND_WINDOW). relion5-particle-arena: --j may exceed the core count. CANDIDATES=0 / ARENA=0 leave them out.
for p in relion5-tt $([ "${CANDIDATES:-1}" = 1 ] && echo relion5-tt-candidates) relion5-coarse-blocking relion5-nonzeroprob relion5-estep-profile relion5-prep relion5-sums-record $([ "${SUMS_SPARSE:-1}" = 1 ] && echo relion5-sums-sparse) $([ "${HOST_THREADS:-1}" = 1 ] && echo relion5-host-threads) $([ "${ARENA:-1}" = 1 ] && echo relion5-particle-arena); do
  git apply "$HERE/$p.patch"
done
mkdir -p src/acc/tt build
cp "$HERE"/src_acc_tt/* src/acc/tt/
cd build
"$CMAKE" .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DALTCPU=ON -DCUDA=OFF \
  -DTT=ON -DGUI=OFF -DFETCH_WEIGHTS=OFF
make -j"${JOBS:-$(( $(nproc) / 4 ))}" refine_mpi
