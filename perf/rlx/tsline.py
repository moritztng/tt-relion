#!/usr/bin/env python3
"""Timestamp a program's stdout per line: `<start> <end> <text>`, epoch seconds.

start is when the line's first byte arrived, end when its newline did, so a RELION progress bar (redrawn with \\r
until it completes) gives the duration of the step it tracks. Only the last \\r segment of a line is kept.
"""
import os, sys, time

buf, start = b"", None
out = sys.stdout.buffer
while True:
    chunk = os.read(0, 65536)
    now = time.time()
    if not chunk:
        break
    for part in chunk.splitlines(keepends=True):
        if start is None:
            start = now
        buf += part
        if part.endswith(b"\n"):
            out.write(b"%.2f %.2f %s\n" % (start, now, buf.rstrip(b"\r\n").split(b"\r")[-1]))
            buf, start = b"", None
    out.flush()
if buf:
    out.write(b"%.2f %.2f %s\n" % (start, time.time(), buf.split(b"\r")[-1]))
