"""Record the bridge call stream of a RELION run, and read it back.

TT_RELION_RECORD=<dir> wraps diff2_coarse and diff2_fine: every call is appended to <dir>/rec.<pid>.pkl with
its thread, its entry and exit time and its inputs, whatever the backend then does with it. One file per rank.
perf/rlx/bridge/replay.py feeds a file back into the bridge from the same number of threads, so the rate of the
RELION-called path (batch fill, chip busy share) can be measured in minutes without RELION.

The model volume and the orientation sets recur, so they are written once per process and referenced by key.
Under a declining backend (nullbase) the gap between a thread's calls includes RELION's own kernel; under a
handling backend it is RELION's host work alone.
"""
import hashlib
import os
import pickle
import threading
import time

import numpy as np

_lock = threading.Lock()
_files = {}             # directory -> [file, keys already written, calls written]
_MAX = int(os.environ.get("TT_RELION_RECORD_N", "0"))      # calls per process, 0 = all


def _key(b):
    return hashlib.sha256(b).hexdigest()[:24]


def wrap(kind, fn, directory):
    """fn with every call recorded. kind "C": out is argument 7; "F": rot_idx, trans_idx are 7, 8 and out is 9."""
    n_mv = 8 if kind == "C" else 10

    def call(*a):
        t0 = time.time()
        try:
            return fn(*a)
        finally:
            _write(directory, kind, a, n_mv, t0, time.time())
    return call


def _write(directory, kind, a, n_mv, t0, t1):
    from .relion import _model_key
    mk = _key(repr(_model_key(a[0])).encode())
    eul = bytes(a[1])
    ek = _key(eul)
    rec = {"k": kind, "tid": threading.get_ident(), "t0": t0, "t1": t1, "mdl": mk, "eul": ek,
           "mv": [bytes(m) for m in a[2:n_mv - 1]], "out": a[n_mv - 1].nbytes, "sc": list(a[n_mv:])}
    with _lock:
        if directory not in _files:
            os.makedirs(directory, exist_ok=True)
            _files[directory] = [open(os.path.join(directory, "rec.%d.pkl" % os.getpid()), "ab"), set(), 0]
        f = _files[directory]
        if _MAX and f[2] >= _MAX:
            if f[2] == _MAX:            # tell perf/rlx/bridge/record.sh this rank has what it needs
                open(f[0].name + ".full", "w").close()
                f[2] += 1
            return
        for tag, key, mv in (("M", mk, a[0]), ("E", ek, eul)):
            if key not in f[1]:
                f[1].add(key)
                pickle.dump({"k": tag, "key": key, "data": bytes(mv)}, f[0], protocol=5)
        pickle.dump(rec, f[0], protocol=5)
        f[0].flush()
        f[2] += 1


def load(path, limit=0):
    """-> (models {key: bytes}, euler sets {key: bytes}, calls in file order)."""
    models, euls, calls = {}, {}, []
    with open(path, "rb") as fh:
        while True:
            try:
                r = pickle.load(fh)
            except EOFError:
                break
            if r["k"] == "M":
                models[r["key"]] = r["data"]
            elif r["k"] == "E":
                euls[r["key"]] = r["data"]
            else:
                calls.append(r)
                if limit and len(calls) >= limit:
                    break
    return models, euls, calls


def args(models, euls, r):
    """The argument tuple of the recorded call and its zeroed output buffer (float32 view)."""
    out = np.zeros(r["out"] // 4, np.float32)
    mv = [memoryview(models[r["mdl"]]), memoryview(euls[r["eul"]])] + [memoryview(b) for b in r["mv"]]
    return tuple(mv) + (memoryview(out).cast("B"),) + tuple(r["sc"]), out
