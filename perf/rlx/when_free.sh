#!/bin/bash
# when_free.sh CARDS LOG -- CMD...: wait until one of CARDS (comma list, e.g. 0,1) has no other lease holder
# on this host, then run CMD pinned to it under this row's lease identity. Polls every 60 s; gives up after 12 h.
cards=$1; log=$2; shift 3
pick() { for c in ${cards//,/ }; do python3 ~/.coworker/scripts/vrt_yield.py --free local "$c" >/dev/null && { echo "$c"; return 0; }; done; return 1; }
for i in $(seq 720); do card=$(pick) && break; sleep 60; done
[ -n "$card" ] || { echo "GAVE-UP cards $cards busy" >>"$log"; exit 3; }
echo "START $(date -u +%FT%TZ) card $card" >>"$log"
TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:rlx-kernels "$@" >>"$log" 2>&1
echo "END $(date -u +%FT%TZ) rc=$?" >>"$log"
