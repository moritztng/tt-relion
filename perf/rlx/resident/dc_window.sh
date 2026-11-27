#!/bin/bash
# Booked whole-box dev cell, launched detached ahead of the window: sleep until START (UTC, HH:MM), require the
# box to be free (nothing holding /dev/tenstorrent, waits up to GRACE min), probe all chips in parallel, then
# devcell.sh on every chip that passed. Per-phase timeout TO keeps the cell inside the booking.
#   dc_window.sh 20:00 10 NG NL NF   (run from the run dir)
START=${1:?HH:MM}; GRACE=${2:-10}; NG=${3:-9}; NL=${4:-4}; NF=${5:-2}
SRC=${SRC:-~/rlx-resident/src}; export SRC
export PATH=~/fleet/env/bin:$PATH PYTHONPATH=$SRC
R=$SRC/perf/rlx/resident
t=$(date -u -d "today $START" +%s); now=$(date +%s); [ $t -gt $now ] && { echo "sleep until ${START}Z"; sleep $((t-now)); }
[ -e ~/fleet/HANDBACK ] && { echo HANDBACK; exit 3; }
end=$(( $(date +%s) + GRACE*60 ))
while h=$(fuser /dev/tenstorrent/* 2>/dev/null | tr -s ' '); [ -n "$h" ]; do
  echo "$(date -u +%TZ) box busy: pids $h"; ps -o pid,etime,args -p $(echo $h | tr ' ' ',') | cut -c1-160
  [ $(date +%s) -ge $end ] && { echo "still held past grace: probing anyway, a held chip fails its probe and is left out"; break; }; sleep 30
done
echo "$(date -u +%TZ) box free; probing 0..31"
for c in $(seq 0 31); do (TT_VISIBLE_DEVICES=$c timeout -k 30 -s INT 120 python $R/probe_chip.py > probe_$c.log 2>&1) & done; wait
OK=$(for c in $(seq 0 31); do grep -q PROBE_OK probe_$c.log && echo -n "$c "; done)
echo "healthy: [$OK] ($(echo $OK | wc -w) chips)"
[ -z "$OK" ] && exit 5
TO=${TO:-400} bash $R/devcell.sh "$OK" $NG $NL $NF
echo "window done $(date -u +%FT%TZ)"
