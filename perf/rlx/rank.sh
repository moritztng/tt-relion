#!/usr/bin/env bash
# One MPI rank. Rank 0 (RELION's leader) runs no expectation step and opens no chip; follower rank r
# gets chip r-1 (UMD logical id), pinned with TT_VISIBLE_DEVICES, and holds that chip's tt-bio lease
# flock for its whole life so co-tenants see it taken.
r=${OMPI_COMM_WORLD_RANK:?not under mpirun}
if [ "$r" -gt 0 ] && [ "${TT_RELION_BACKEND:-}" = ttnn ]; then
    chip=$(( (r - 1) % ${RLX_CHIPS:-32} ))
    export TT_VISIBLE_DEVICES=$chip
    if [ -n "${RLX_LEASE_DIR:-}" ]; then
        exec 9>"$RLX_LEASE_DIR/${RLX_LEASE_HOST}-card$chip.json"
        flock -n 9 || { echo "rank $r: chip $chip is leased by someone else" >&2; exit 3; }
        printf '{"host":"%s","card":"%s","holder":"worker:rlx-galaxy-floor","pid":%d,"acquired":%d,"released":null}\n' \
            "$RLX_LEASE_HOST" "$chip" $$ "$(date +%s)" >&9
    fi
fi
exec "$@"
