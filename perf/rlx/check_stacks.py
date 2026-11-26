#!/usr/bin/env python3
"""Check staged MRC stacks; print the bad ones, one 'BAD <path> <why>' line each, and exit 1 if any.

    check_stacks.py <dir>

A stack is bad when its size is not header + nx*ny*nz*mode bytes, when its last image is all zero (aria2
preallocates, so a cut-off download can have the right size), or when it contains EBI's error body. EBI's
object store sometimes drops a connection mid-response and streams '<Error><Code>ConnectionClosedException...'
as part of the 200 body; aria2 writes those bytes into the stack as pixel values, the file keeps its exact
size, and RELION later aborts at iteration 2 because the image's norm correction is inf. The marker scan
reads every byte, at disk speed.
"""
import glob
import os
import struct
import sys

BPP = {0: 1, 1: 2, 2: 4, 6: 2, 12: 2}
MARK = b"<Error><Code>"
CHUNK = 64 << 20


def why_bad(f):
    with open(f, "rb") as h:
        hd = h.read(1024)
    nx, ny, nz, mode = struct.unpack("<4i", hd[:16])
    ext = struct.unpack("<i", hd[92:96])[0]
    want, got = 1024 + ext + nx * ny * nz * BPP.get(mode, 4), os.path.getsize(f)
    if got != want:
        return f"size {got} want {want} mode {mode}"
    img = nx * ny * BPP.get(mode, 4)
    with open(f, "rb") as h:
        h.seek(got - img)
        if not h.read(img).strip(b"\0"):
            return "last image all zero"
        h.seek(0)
        prev = b""
        while True:
            b = h.read(CHUNK)
            if not b:
                return None
            i = (prev + b).find(MARK)
            if i >= 0:
                return f"EBI error body at byte {h.tell() - len(b) - len(prev) + i}"
            prev = b[-len(MARK):]


files = glob.glob(os.path.join(sys.argv[1], "**", "*.mrcs"), recursive=True)
bad = 0
for f in files:
    w = why_bad(f)
    if w:
        bad += 1
        print(f"BAD {f} {w}")
print(f"stacks checked {len(files)} bad {bad}")
sys.exit(1 if bad else 0)
