#!/usr/bin/env bash
# ttvd_nodes.sh <id>... : the /dev/tenstorrent node each TT_VISIBLE_DEVICES id opens, one per line.
# UMD numbers chips in PCI address order, not by node number (on one BH Galaxy id 16 is node 24 and id 24 is node 16).
mapfile -t by_pci < <(for d in /sys/class/tenstorrent/tenstorrent!*; do
    echo "$(basename "$(readlink -f "$d/device")") ${d##*!}"; done | sort | cut -d' ' -f2)
for i in "$@"; do echo "${by_pci[$i]:?no chip $i}"; done
