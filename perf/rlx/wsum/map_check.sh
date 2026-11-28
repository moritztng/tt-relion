#!/bin/bash
# map_check.sh <tag> <optimiser.star to continue from> <iteration> <relion_refine_mpi A> <relion_refine_mpi B>
# One full iteration (E-step and M-step) with two RELION builds over a fixed subset of the particles, then a
# byte compare of both half maps and of data.star. It is the gate for a lever that changes a data layout: the
# per-particle records (same_sums.py) cover the kernels, but not the shared volume or the code that reads it.
# A layout change that passed the records gave half maps with correlation 0.04 here, because the MPI program
# carries its own copy of the loop that reads the backprojected volume.
# Each arm runs 3 ranks x 1 thread, so each half set is summed by one thread in a fixed order and builds that do
# the same adds give maps equal to the bit. The subset is every EVERY-th particle plus the first of each group.
# Env: R (work dir, default ~/rlx-wsum), EVERY (16), PYTHON (python3 with numpy). Output: $R/runs/<tag>.status
tag=${1:?tag}; from=${2:?optimiser.star}; it=${3:?iteration}; A=${4:?build A}; B=${5:?build B}
R=${R:-$HOME/rlx-wsum}; here=$(cd "$(dirname "$0")" && pwd); PY=${PYTHON:-python3}
S=$R/runs/$tag.status; T=$R/runs/${tag}_src; D=$(dirname "$from"); stem=$(basename "$from" _optimiser.star)
rm -rf "$T"; mkdir -p "$T"; cp "$D/$stem"_* "$T"/
$PY - "$T/${stem}_data.star" ${EVERY:-16} > "$S" <<'PY'
import sys
p, every = sys.argv[1], int(sys.argv[2]); out = []; part = False; cols = {}; i = 0; seen = set(); kept = 0
for line in open(p):
    s = line.split()
    if s and s[0].startswith("data_"):
        part = s[0] == "data_particles"
    if part and s and s[0].startswith("_rln"):
        cols[s[0]] = int(s[1].lstrip("#")) - 1
    if part and s and not s[0].startswith(("data_", "loop_", "_", "#")):
        g = s[cols["_rlnGroupName"]]
        keep = i % every == 0 or g not in seen
        seen.add(g); i += 1
        if not keep:
            continue
        kept += 1
    out.append(line)
open(p, "w").writelines(out)
print("subset: %d of %d particles, %d groups" % (kept, i, len(seen)))
PY
sed -i "s#$D/#$T/#g" "$T/${stem}_optimiser.star"
for b in a b; do
  bin=$A; [ $b = b ] && bin=$B
  ( RELION_BIN=$bin RANKS=3 J=1 N=999999999 TO=${TO:-2400} "$here/prof_arm.sh" ${tag}_$b "$T/${stem}_optimiser.star" $it
    echo "${tag}_$b $bin: $(tail -1 $R/runs/${tag}_$b/status) $(grep -i -m1 "error\|segmentation" $R/runs/${tag}_$b/refine.log)" >> "$S" ) &
done
wait
n=$(printf %03d $it); rc=0
for h in 1 2; do
  f=run_it${n}_half${h}_class001.mrc; a=$R/runs/${tag}_a/$f; b=$R/runs/${tag}_b/$f
  # the first 1,024 bytes are the MRC header, which carries the time the file was written
  if [ ! -s $a ] || [ ! -s $b ]; then echo "half$h map: MISSING" >> "$S"; rc=1
  elif cmp -s -i 1024 $a $b; then echo "half$h map: same to the bit ($(( $(stat -c %s $b) - 1024 )) bytes of voxels)" >> "$S"
  else rc=1; echo "half$h map: DIFFER $($PY -c "
import numpy as np, sys
a, b = (np.fromfile(p, np.float32, offset=1024).astype(np.float64) for p in sys.argv[1:])
print('relative L2 %.3e corr %.12f' % (np.linalg.norm(a - b) / np.linalg.norm(a), np.corrcoef(a, b)[0, 1]))" $a $b 2>&1 | tail -1)" >> "$S"; fi
done
a=$R/runs/${tag}_a/run_it${n}_data.star; b=$R/runs/${tag}_b/run_it${n}_data.star
if [ -s $a ] && cmp -s $a $b; then echo "data.star: same" >> "$S"; else rc=1; echo "data.star: differ or missing" >> "$S"; fi
echo "$tag $([ $rc = 0 ] && echo PASS || echo FAIL) $(date -u +%FT%TZ)" >> "$S"
exit $rc
