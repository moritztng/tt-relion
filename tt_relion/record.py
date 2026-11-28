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


def digest(d2):
    """What RELION takes from one call's squared differences: (argmin, minimum, the significant samples in index
    order, their values in float64). Significant = the samples that carry 99.9 % of sum exp(min - d2)."""
    d = np.asarray(d2, np.float64).reshape(-1)
    w = np.exp(d.min() - d)
    o = np.argsort(-w, kind="stable")
    sig = np.sort(o[:int(np.searchsorted(np.cumsum(w[o]), 0.999 * w.sum())) + 1])
    return int(d.argmin()), float(d.min()), sig, d[sig]


class Digests:
    """The digests of a call stream, kept per call and written as one npz. h_sig is a hash of the significant set;
    sig_idx / sig_val (flat, call i at sig_off[i]:sig_off[i + 1]) let a later arm be graded on the reference's own
    significant samples, where an error in d2 moves a weight."""

    def __init__(self, n):
        self.arg_min, self.d_min = np.zeros(n, np.int64), np.zeros(n, np.float64)
        self.n_sig, self.h_sig, self.handled = np.zeros(n, np.int64), np.zeros(n, "S16"), np.zeros(n, bool)
        self.sig = [(np.zeros(0, np.int64), np.zeros(0))] * n

    def put(self, i, d2):
        self.arg_min[i], self.d_min[i], idx, val = digest(d2)
        self.n_sig[i], self.h_sig[i], self.sig[i] = len(idx), hashlib.sha256(idx.tobytes()).digest()[:16], (idx, val)
        self.handled[i] = True

    def save(self, path):
        np.savez(path, arg_min=self.arg_min, d_min=self.d_min, n_sig=self.n_sig, h_sig=self.h_sig, handled=self.handled,
                 sig_off=np.concatenate([[0], np.cumsum(self.n_sig)]), sig_idx=np.concatenate([s[0] for s in self.sig]),
                 sig_val=np.concatenate([s[1] for s in self.sig]))
