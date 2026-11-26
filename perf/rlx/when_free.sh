#!/bin/bash
# when_free.sh CARD LOG -- CMD...: wait until no other lease holder has CARD on this host, then run CMD
# pinned to it under this row's lease identity. Polls every 60 s; gives up after 12 h.
card=$1; log=$2; shift 3
for i in $(seq 720); do
  python3 ~/.coworker/scripts/vrt_yield.py --free local "$card" >/dev/null && break
  sleep 60
done
python3 ~/.coworker/scripts/vrt_yield.py --free local "$card" || { echo "GAVE-UP card $card busy" >>"$log"; exit 3; }
echo "START $(date -u +%FT%TZ) card $card" >>"$log"
TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:rlx-kernels "$@" >>"$log" 2>&1
echo "END $(date -u +%FT%TZ) rc=$?" >>"$log"
