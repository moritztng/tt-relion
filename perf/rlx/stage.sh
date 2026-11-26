#!/bin/bash
# Stage the EMPIAR-10028 stacks named in <list> (one Particles/... path per line) plus EMD-2660 into <dir>.
# aria2 drops files silently under load, so it repeats until every stack is non-empty or fails loudly.
#   stage.sh <dir> <list>
set -uo pipefail
D=$1 LIST=$(realpath "$2"); U=https://ftp.ebi.ac.uk/empiar/world_availability/10028/data
mkdir -p "$D" && cd "$D" || exit 1; t0=$(date +%s)
[ -s emd_2660.map ] || curl -sf https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2660/map/emd_2660.map.gz | gunzip > emd_2660.map
for t in 1 2 3 4 5 6; do
  m=$(while read -r p; do [ -s "$p" ] || echo "$p"; done < "$LIST"); [ -z "$m" ] && break
  echo "stage pass $t: $(echo "$m" | wc -l) stacks missing"
  echo "$m" | awk -v u=$U '{print u"/"$1"\n  out="$1}' > aria.txt
  aria2c -q -j 16 -x 4 --max-tries=10 --retry-wait=5 -i aria.txt --auto-file-renaming=false --allow-overwrite=true
done
[ -z "$m" ] && [ -s emd_2660.map ] || { echo "STAGE FAILED: $(echo "$m" | wc -l) stacks missing"; exit 1; }
du -sb Particles | awk -v s=$(( $(date +%s) - t0 )) '{printf "STAGED %.0f bytes in %d s = %.1f MB/s\n",$1,s,$1/s/1e6}'
