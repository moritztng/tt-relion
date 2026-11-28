#!/usr/bin/env bash
# One MPI rank. Rank 0 (RELION's leader) runs no expectation step and opens no chip. Follower rank r
# gets RLX_CHIPS_PER_RANK consecutive chips (UMD logical ids) starting at RLX_CHIP_BASE + (r-1)*K mod RLX_CHIPS, one
# device server each (RLX_RANK_CHIPS), and holds every one of their tt-bio lease flocks for its whole
# life so co-tenants see them taken. K > 1 keeps the rank count, and so RELION's host memory, down.
r=${OMPI_COMM_WORLD_RANK:?not under mpirun}
if [ "$r" -gt 0 ] && [ "${TT_RELION_BACKEND:-}" = ttnn ]; then
    k=${RLX_CHIPS_PER_RANK:-1} chips=()
    if [ -n "${RLX_CHIP_LIST:-}" ]; then    # an explicit, possibly non-contiguous set: "16,17,20"
        IFS=, read -ra all <<< "$RLX_CHIP_LIST"
        for i in $(seq 0 $((k - 1))); do chips+=(${all[$(( ((r - 1) * k + i) % ${#all[@]} ))]}); done
    else
        for i in $(seq 0 $((k - 1))); do chips+=($(( ${RLX_CHIP_BASE:-0} + ((r - 1) * k + i) % ${RLX_CHIPS:-32} ))); done
    fi
    export RLX_RANK_CHIPS=$(IFS=,; echo "${chips[*]}") TT_VISIBLE_DEVICES=${chips[0]}
    if [ -n "${RLX_LEASE_DIR:-}" ]; then
        for chip in "${chips[@]}"; do
            lease="$RLX_LEASE_DIR/${RLX_LEASE_HOST}-card$chip.json"
            exec {fd}<>"$lease"     # not ">": that empties the file of a holder before the lock is even tried
            flock -n "$fd" || { echo "rank $r: chip $chip is leased by someone else" >&2; exit 3; }
            : > "$lease"
            printf '{"host":"%s","card":"%s","holder":"worker:rlx-galaxy-floor","pid":%d,"acquired":%d,"released":null}\n' \
                "$RLX_LEASE_HOST" "$chip" $$ "$(date +%s)" >&"$fd"
        done
    fi
fi
exec "$@"
