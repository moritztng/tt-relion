#!/usr/bin/env python3
"""Every MRC stack in <dir> must be exactly header + nx*ny*nz*mode bytes and its last image must not be all zero
(aria2 preallocates, so a cut-off download can have the right size); prints the bad ones. Usage: check_stacks.py <dir>"""
import sys, glob, os, struct
bpp = {0: 1, 1: 2, 2: 4, 6: 2, 12: 2}
bad = 0; files = glob.glob(os.path.join(sys.argv[1], '**', '*.mrcs'), recursive=True)
for f in files:
    with open(f, 'rb') as h: hd = h.read(1024)
    nx, ny, nz, mode = struct.unpack('<4i', hd[:16]); ext = struct.unpack('<i', hd[92:96])[0]
    want = 1024 + ext + nx * ny * nz * bpp.get(mode, 4); got = os.path.getsize(f)
    if got != want: bad += 1; print(f"BAD {f} size {got} want {want} mode {mode}"); continue
    img = nx * ny * bpp.get(mode, 4)
    with open(f, 'rb') as h: h.seek(got - img); tail = h.read(img)
    if not tail.strip(b'\0'): bad += 1; print(f"BAD {f} last image all zero")
print(f"stacks checked {len(files)} bad {bad}"); sys.exit(1 if bad else 0)
