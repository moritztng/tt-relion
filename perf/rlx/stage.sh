#!/bin/bash
# Stage the EMPIAR-10028 stacks named in <list> (one Particles/... path per line) plus EMD-2660 into <dir>.
# aria2 drops files silently under load and preallocates, so a stack counts as staged only when it is non-empty and
# has no .aria2 control file left and check_stacks.py passes it (EBI can splice an error body into a 200 response);
# downloads resume (-c). EBI caps each host's rate, so it opens many connections.
#   stage.sh <dir> <list>
set -uo pipefail
D=$1 LIST=$(realpath "$2") CHK=$(dirname "$(realpath "$0")")/check_stacks.py; U=https://ftp.ebi.ac.uk/empiar/world_availability/10028/data
mkdir -p "$D" && cd "$D" || exit 1; t0=$(date +%s)
# One slow stream beside the stacks. A cut-off .gz fails gunzip -t and is fetched again, so it never looks complete.
[ -s emd_2660.map ] || for t in 1 2 3 4 5 6; do curl -sf --retry 5 -o map.gz \
  https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2660/map/emd_2660.map.gz && gunzip -t map.gz &&
  gunzip -c map.gz > map.part && mv map.part emd_2660.map && rm map.gz && break; sleep 30; done &
for t in 1 2 3 4 5 6 7 8; do
  m=$(while read -r p; do { [ -s "$p" ] && [ ! -e "$p.aria2" ]; } || echo "$p"; done < "$LIST")
  if [ -z "$m" ]; then   # all present: a stack holding EBI's error body is re-fetched whole
    c=$(python3 "$CHK" Particles); echo "$c" | tail -1
    echo "$c" | grep -q '^stacks checked' || { echo "STAGE FAILED: check_stacks.py did not run"; exit 1; }
    m=$(echo "$c" | awk '/^BAD /{print $2}'); [ -z "$m" ] && break
    echo "$m" | xargs rm -f; fi
  echo "stage pass $t: $(echo "$m" | wc -l) stacks missing or bad"
  echo "$m" | awk -v u=$U '{print u"/"$1"\n  out="$1}' > aria.txt
  aria2c -q -c -j 48 -x 8 --max-tries=20 --retry-wait=5 -i aria.txt --auto-file-renaming=false
done
wait; [ -z "$m" ] && [ -s emd_2660.map ] || { echo "STAGE FAILED: $(echo "$m" | wc -l) stacks missing"; exit 1; }
du -sb Particles | awk -v s=$(( $(date +%s) - t0 )) '{printf "STAGED %.0f bytes in %d s = %.1f MB/s\n",$1,s,$1/s/1e6}'
