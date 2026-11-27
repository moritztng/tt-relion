"""The kernels RELION calls, on the RELION side of the bridge.

RELION hands these functions raw memoryviews over its own buffers, so nothing is copied on the way
in. Every entry point returns True when it handled the call and False to decline it, in which case
RELION runs its own kernel: an unsupported shape, a missing device or a disagreeing environment must
degrade to the CPU path, never fail the refinement.

The numerics are RELION's, transcribed from src/acc/cpu/cpu_kernels/diff2.h, cpu_utils.h and
src/acc/acc_projectorkernel_impl.h, for the 3D-reference / 2D-data case that 3D auto-refine runs.
One deliberate difference: RELION accumulates per 256-pixel block, this reduces the whole image at
once. That changes the last bits and nothing else, and TT_RELION_CHECK=1 measures the residual
against RELION's own value rather than asserting it.

Both of RELION's difference passes are the same computation. The coarse pass wants the whole
orientation x translation matrix; the fine pass wants a sparse subset of the entries of the same
matrix, over a different (significant) orientation set, plus a per-image constant. So there is one
core, _dense_diff2, and two entry points.

Backend selected by TT_RELION_BACKEND:
  torch     torch on the host. Exact, slow, and the arm that proves the plumbing.
  ttnn      the device path: projection on the host (the torch code below), the orientation x
            translation cross term on the chip. See tt_relion/cross.py.
  nullbase  decline the instant the hook is entered. Computes nothing, touches no buffer.
  null      do the buffer traffic the device path will do, then decline.

nullbase and null exist to price the bridge itself. A device kernel cannot beat RELION's own coarse
kernel if merely entering Python costs more than the kernel does, and neither the reference wall nor
the shape arm answers that: the reference never crosses into Python, and the shape arm crosses AND
writes a log line with an np.unique per fine call, so its wall carries an instrument the real path
does not have. Three walls -- reference, nullbase, null -- separate the crossing from the marshalling
from the compute.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time

import numpy as np

_BACKEND = os.environ.get("TT_RELION_BACKEND", "torch")
_ORI_CHUNK = int(os.environ.get("TT_RELION_ORI_CHUNK", "32"))
_CHECK = os.environ.get("TT_RELION_CHECK", "") not in ("", "0")
# TT_RELION_DEV_CHUNK: orientations per device matmul. Larger amortises the launch, smaller bounds
# the host copy of the projected references.
_DEV_CHUNK = int(os.environ.get("TT_RELION_DEV_CHUNK", "1024"))
# TT_RELION_FINE_CALL_MB: host bytes one fine chunk may hold while it builds its operand. A fine chunk
# costs ~40 bytes per orientation and pixel (the float64 copy, |ref|^2, its weighted product, the shared
# memory copy), and the final iteration runs the fine pass at Nyquist: at box 360 a 1024-orientation
# chunk held ~2.7 GB per --j thread and OOM-killed the dev cell's last iteration on a 32-chip Galaxy.
# The coarse pass is never near this size, so its chunks (and its cache keys) stay _DEV_CHUNK.
_FINE_CALL_MB = int(os.environ.get("TT_RELION_FINE_CALL_MB", "1024"))
_FIDELITY = os.environ.get("TT_RELION_FIDELITY", "HiFi3")
_KCHUNK = int(os.environ.get("TT_RELION_KCHUNK", "256"))
# TT_RELION_OZAKI=<n>: exact-integer cross term with n base-4 digits (Blackhole; 14 meets the accuracy bar
# at every size). 0 (default): the bf16 split, which meets it on Wormhole. tt_relion/cross.py.
_OZAKI = int(os.environ.get("TT_RELION_OZAKI", "0"))
# TT_RELION_TRACE=<path>: append one line per call with the shape, a hash of the euler set and a
# hash of the reference. Answers "is the coarse orientation set shared across particles" from a
# real run rather than from reading RELION's sampling code.
_TRACE = os.environ.get("TT_RELION_TRACE", "")
# TT_RELION_SHAPE=<path>: append one line per FINE call with the shape only, and keep declining.
# The fine pass runs on a data-dependent significant subset, so its orientation count, its job
# count and its density are properties of the refinement rather than of the code. This logs them
# from the reference arm, where every call declines and the run costs what RELION's own run costs.
_SHAPE = os.environ.get("TT_RELION_SHAPE", "")
# TT_RELION_DUMP=<path>: write the first call's raw inputs to <path>.<pid>.npz and keep going.
# The separable-interpolant study needs RELION's own padded model and its own orientation set at
# the real operating point; reconstructing either from RELION's source is guesswork.
_DUMP = os.environ.get("TT_RELION_DUMP", "")
_DUMP_N = int(os.environ.get("TT_RELION_DUMP_N", "1"))
_dumped = [0]
_stats = {"handled": 0, "declined": 0, "fine_handled": 0, "fine_declined": 0,
          "proj_hit": 0, "proj_miss": 0, "ref_asked": 0, "devproj": 0, "once_cpu": 0}
# TT_RELION_PROJ_CACHE_MB: cap on kept coarse references (device path), per rank on the host and per
# chip on the device. In a global search every particle's coarse pass asks for the same orientation set
# against the same reference, so the projection and its left operand are per-iteration constants:
# _project was 66-86 % of a coarse call's host wall (perf/rlx/bridge_prof.py). The host keeps the
# projection, each chip keeps the operand, so a kept coarse call ships only the shifted image. An entry
# is kept only on its second sighting, so local searches (per-particle orientations) keep nothing.
# 0 disables.
_PCACHE_MB = int(os.environ.get("TT_RELION_PROJ_CACHE_MB", "4096"))
# TT_RELION_FINE_PROJECT: who projects the fine pass's reference. "relion" (default): RELION's own
# AccProjectorKernel in the calling --j thread, handed over as ref_mv. "bridge": RELION hands over
# the model and geometry instead and the bridge projects (today on the host in torch, the pre-L4 path;
# the on-device projection op plugs in here). Read by the C++ bridge at init, so RELION skips its
# projection loop when the bridge asks for the model.
_FINE_PROJECT = os.environ.get("TT_RELION_FINE_PROJECT", "relion")
if _FINE_PROJECT not in ("relion", "bridge"):
    raise ValueError("TT_RELION_FINE_PROJECT must be relion or bridge, got %r" % _FINE_PROJECT)
_pc = {"model": None, "gen": 0, "seen": set(), "hit": {}, "bytes": 0}
# TT_RELION_COARSE_ONCE: where a coarse orientation set seen for the first time under the current
# reference goes. "chip" (default): RELION projects it in C++ and the chip computes it. "cpu": the
# call declines and RELION's own CPU kernel computes it; from the second sighting on the set goes to
# the chip and is kept. A local search asks every particle for its own set, so under "cpu" its coarse
# pass stays on the host, where one call costs less than one chip round trip with a fresh operand.
_COARSE_ONCE = os.environ.get("TT_RELION_COARSE_ONCE", "chip")
if _COARSE_ONCE not in ("chip", "cpu"):
    raise ValueError("TT_RELION_COARSE_ONCE must be chip or cpu, got %r" % _COARSE_ONCE)
# TT_RELION_DEVPROJ=1: the fine pass projects on the chip. Each server keeps the model it last saw in a
# coarse call (the same projector the particle's fine call uses), and a fine call ships only its Euler
# matrices instead of the [O, 2P] reference. Off by default until the dev-cell A/B.
_DEVPROJ = os.environ.get("TT_RELION_DEVPROJ", "") not in ("", "0")
_pc_lock = threading.Lock()


# TT_RELION_TORCH_THREADS=<n>: torch's intra-op thread count inside the bridge. RELION already
# parallelises over --j threads per rank and runs several ranks per box, so torch's default (one
# thread per core, per process) multiplies that by the core count. Unset leaves torch alone.
_TORCH_THREADS = int(os.environ.get("TT_RELION_TORCH_THREADS", "0"))
_torch_ready = [False]


def _torch():
    import torch
    # set_grad_enabled is thread-local and RELION calls this from every one of its --j threads, so
    # it stays per call. set_num_threads is process-wide, so it is set once.
    torch.set_grad_enabled(False)
    if _TORCH_THREADS > 0 and not _torch_ready[0]:
        torch.set_num_threads(_TORCH_THREADS)
        _torch_ready[0] = True
    return torch


def _project(t, mdl, x, y, e, mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ,
             maxR2_padded, padding_factor):
    """AccProjectorKernel::project3Dmodel, the 2D-data overload, PROJECTOR_NO_TEXTURES.

    e is [C, 9], x and y are [P]. Returns ref_real, ref_imag as [C, P].
    """
    pf = padding_factor
    xp = (e[:, 0:1] * x + e[:, 1:2] * y) * pf
    yp = (e[:, 3:4] * x + e[:, 4:5] * y) * pf
    zp = (e[:, 6:7] * x + e[:, 7:8] * y) * pf

    # RELION truncates the radius test to int, so 4.9 becomes 4. trunc, not round and not floor.
    inside = (xp * xp + yp * yp + zp * zp).trunc().to(t.int64) <= maxR2_padded

    # Hermitian pair for the negative-x half space.
    invers = xp < 0
    xp = t.where(invers, -xp, xp)
    yp = t.where(invers, -yp, yp)
    zp = t.where(invers, -zp, zp)

    x0 = xp.floor()
    fx = xp - x0
    y0 = yp.floor()
    fy = yp - y0
    z0 = zp.floor()
    fz = zp - z0

    mdlXY = mdlX * mdlY
    base = ((z0.to(t.int64) - mdlInitZ) * mdlXY
            + (y0.to(t.int64) - mdlInitY) * mdlX
            + x0.to(t.int64))
    # Outside the radius RELION never evaluates the interpolation; here the gather is
    # unconditional, so clamp it in bounds and mask the result afterwards.
    nvox = mdlXY * mdlZ
    base = base.clamp(0, nvox - mdlXY - mdlX - 2)

    off = (0, 1, mdlX, mdlX + 1, mdlXY, mdlXY + 1, mdlXY + mdlX, mdlXY + mdlX + 1)
    c = [mdl.index_select(0, (base + o).reshape(-1)).reshape(base.shape + (2,)) for o in off]

    def lerp(a, b, f):
        return a + (b - a) * f.unsqueeze(-1)

    dx00 = lerp(c[0], c[1], fx)
    dx10 = lerp(c[2], c[3], fx)
    dx01 = lerp(c[4], c[5], fx)
    dx11 = lerp(c[6], c[7], fx)
    dxy0 = lerp(dx00, dx10, fy)
    dxy1 = lerp(dx01, dx11, fy)
    ref = lerp(dxy0, dxy1, fz)

    ref_r = ref[..., 0]
    ref_i = t.where(invers, -ref[..., 1], ref[..., 1])
    zero = t.zeros((), dtype=ref_r.dtype)
    return t.where(inside, ref_r, zero), t.where(inside, ref_i, zero)


_srv = []            # this rank's device servers, one per chip
_free = None         # queue of idle servers; a --j thread takes one for one matmul
_lock = threading.Lock()
_tl = threading.local()   # the server this thread used last (its bridge seconds are booked to it)
# The python the device server runs on. RELION embeds CPython, so sys.executable is RELION itself.
_PYTHON = os.environ.get("TT_RELION_PYTHON", "python3")
_MPI_ENV = ("OMPI_", "PMIX_", "OPAL_", "ORTE_", "PRTE_")
_MALLOC_ENV = {"MALLOC_MMAP_THRESHOLD_": "2147483647", "MALLOC_TRIM_THRESHOLD_": "68719476736",
               "MALLOC_TOP_PAD_": "1073741824"}


class _Server:
    """One chip's device server (tt_relion/device_server.py) and the shared memory it reads."""

    def __init__(self, chip=None, module="tt_relion.device_server"):
        import subprocess
        env = {k: v for k, v in os.environ.items() if not k.startswith(_MPI_ENV)}
        if chip is not None:
            env["TT_VISIBLE_DEVICES"] = chip
        # glibc hands every allocation over 32 MB back to the kernel on free, so each call's operands
        # are page-faulted in afresh: 640 vs 28 ms for one fine-size operand build on pc. Keep them.
        for k, v in _MALLOC_ENV.items():
            env.setdefault(k, v)
        # tt-metal's inspector logs every workload to ./generated/inspector (~75 MB/min per chip). Each new server
        # deletes the previous one's files while that one keeps writing: 32 chips filled 25 GB of disk in 15 min.
        env.setdefault("TT_METAL_INSPECTOR", "0")
        # torch defaults to one OpenMP worker per core. On a 64-thread host the first large torch op in each of 32
        # servers started 31 spinning workers: 785 runnable threads, RELION's own threads starved from it13 on.
        env.setdefault("OMP_NUM_THREADS", "1")
        self.p = subprocess.Popen([_PYTHON, "-m", module, _FIDELITY],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True)
        self.lock = threading.Lock()
        self.shm = None
        self.c = {"calls": 0, "bytes": 0, "rt_s": 0.0, "copy_s": 0.0}   # echoed into the server's exit line
        self.gen, self.kept, self.kept_bytes = None, set(), 0            # operands this chip keeps
        self.models, self.declined = [], set()                           # model keys this chip keeps / refused

    def _seg(self, need):
        from multiprocessing import shared_memory
        if self.shm is None or self.shm.size < need:
            if self.shm is not None:
                self.shm.close()
                self.shm.unlink()
            self.shm = shared_memory.SharedMemory(create=True, size=need)

    def _ask(self, req):
        self.p.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
        self.p.stdin.flush()
        r = self.p.stdout.readline().strip()
        if r not in ("ok", "nomodel"):
            raise RuntimeError("device server died, rc=%s" % self.p.poll())
        return r

    def model(self, mkey, mdl, geom):
        """Make this chip keep the model mkey (fp32 [nvox, 2], box geom); a no-op if it already does."""
        with self.lock:
            if mkey in self.models or mkey in self.declined:
                return
            self._seg(mdl.nbytes)
            np.ndarray(mdl.shape, np.float32, self.shm.buf)[...] = mdl
            if self._ask({"shm": self.shm.name, "model": mkey, "nvox": int(mdl.shape[0]), "geom": geom}) == "ok":
                self.models = (self.models + [mkey])[-2:]
            else:                                   # the chip cannot hold it: never re-ship it
                self.declined.add(mkey)

    def cross_proj(self, mkey, g, eul, shw, w):
        """(C [O, T], ref2 [O]) float64 with the references projected on the chip; None if it lost the model."""
        o, (t, k) = eul.shape[0], shw.shape
        with self.lock:
            if mkey not in self.models:
                return None
            off = 4 * (9 * o + t * k + k // 2)
            self._seg(off + 8 * (o * t + o))
            f32 = np.ndarray((off // 4,), np.float32, self.shm.buf)
            f32[:9 * o] = eul.reshape(-1)
            f32[9 * o:9 * o + t * k] = shw.reshape(-1)
            f32[9 * o + t * k:] = w
            del f32
            t1 = time.time()
            r = self._ask({"shm": self.shm.name, "O": o, "T": t, "K": k, "gen": self.gen,
                           "proj": {"mkey": mkey, "geom": g}, "client": self.c})
            self.c["rt_s"] = round(self.c["rt_s"] + time.time() - t1, 4)
            self.c["calls"] += 1
            if r == "nomodel":
                self.models = [m for m in self.models if m != mkey]
                return None
            c = np.ndarray((o, t), np.float64, self.shm.buf, offset=off).copy()
            ref2 = np.ndarray((o,), np.float64, self.shm.buf, offset=off + 8 * o * t).copy()
        return c, ref2

    def cross(self, ref, o, shw, key, gen, kbytes):
        """C [O, T] float64 = ref [O, K] . shw [T, K]^T on the chip (tt_relion/cross.py).

        ref is a callable giving the fp32 [O, K] references. It is only called, and the references
        only shipped, when this server does not already keep key (key None: one-off). kbytes is what
        keeping the operand costs the chip, capped per server by TT_RELION_PROJ_CACHE_MB."""
        from multiprocessing import shared_memory
        with self.lock:
            if self.gen != gen:
                self.gen, self.kept, self.kept_bytes = gen, set(), 0
            have = key is not None and key in self.kept
            r = None if have else np.ascontiguousarray(ref(), np.float32)
            t, k = shw.shape
            nref = 0 if have else o * k
            need = 4 * (nref + t * k) + 8 * o * t
            if self.shm is None or self.shm.size < need:
                if self.shm is not None:
                    self.shm.close()
                    self.shm.unlink()
                self.shm = shared_memory.SharedMemory(create=True, size=need)
            keep = None
            if have:
                keep = key
            elif key is not None and self.kept_bytes + kbytes <= _PCACHE_MB << 20:
                keep = key
                self.kept.add(key)
                self.kept_bytes += kbytes
            t0 = time.time()
            f32 = np.ndarray((nref + t * k,), np.float32, self.shm.buf)
            if not have:
                f32[:nref] = r.reshape(-1)
            f32[nref:] = shw.reshape(-1)
            t1 = time.time()
            self.p.stdin.write(json.dumps({"shm": self.shm.name, "O": o, "T": t, "K": k, "ref": not have,
                                           "key": keep, "gen": gen, "client": self.c},
                                          separators=(",", ":")) + "\n")
            self.p.stdin.flush()
            if self.p.stdout.readline().strip() != "ok":
                raise RuntimeError("device server died, rc=%s" % self.p.poll())
            t2 = time.time()
            out = np.ndarray((o, t), np.float64, self.shm.buf, offset=4 * (nref + t * k)).copy()
            del f32
            c = self.c
            c["calls"] += 1
            c["bytes"] += need
            c["rt_s"] = round(c["rt_s"] + t2 - t1, 4)
            c["copy_s"] = round(c["copy_s"] + t1 - t0 + time.time() - t2, 4)
        return out


def _pool():
    """This rank's idle-server queue, started on first use. RLX_RANK_CHIPS="a,b,..." gives a rank several chips
    (one server each), so fewer ranks, and so less host memory, can still drive every chip. Servers a caller
    registered in _srv beforehand (the stub benches) are used as they are."""
    global _free
    if _free is None:
        with _lock:
            if _free is None:
                if not _srv:
                    chips = [c for c in os.environ.get("RLX_RANK_CHIPS", "").split(",") if c] or [None]
                    _srv.extend(_Server(c) for c in chips)
                q = queue.Queue()
                for s in _srv:
                    q.put(s)
                _free = q
    return _free


def _cross(ref, o, shw, key, kbytes):
    """C [O, T] float64 on whichever of this rank's chips is idle."""
    free = _pool()
    s = free.get()
    try:
        _tl.srv = s
        return s.cross(ref, o, shw, key, _pc["gen"], kbytes)
    finally:
        free.put(s)


def _cross_proj(eul, shw, w, g):
    """(C, ref2) with the chip projecting from the model this thread's last coarse call used; None if no chip
    keeps it (the caller then uses RELION's projection)."""
    dp = getattr(_tl, "dp", None)
    if dp is None:
        return None
    free = _pool()
    s = free.get()
    try:
        _tl.srv = s
        return s.cross_proj(dp[0], dict(dp[1], **g), eul, shw, w)
    finally:
        free.put(s)


def _keep_model(mdl_mv, geom):
    """TT_RELION_DEVPROJ: every chip of this rank keeps the model of this coarse call, for the fine calls after it."""
    import hashlib
    mkey = hashlib.sha1(repr(_model_key(mdl_mv)).encode()).hexdigest()[:16]
    _pool()
    mdl = np.frombuffer(mdl_mv, dtype=np.float32).reshape(-1, 2)
    for s in _srv:
        s.model(mkey, mdl, geom)
    _tl.dp = (mkey, geom)


def _progress():
    """One cumulative line per reference change: hit rates and where this rank's bridge seconds went.
    C_s/F_s are bridge wall per coarse/fine call summed over threads; rt_s is the part spent waiting on
    the device servers (chip + transport), copy_s the shared-memory copies around it."""
    tot = {}
    for s in _srv:
        for k, v in list(s.c.items()):     # copy: other threads insert keys
            tot[k] = tot.get(k, 0) + v
    keys = ("handled", "fine_handled", "proj_hit", "proj_miss", "ref_asked", "once_cpu")
    return (f"tt_relion pid {os.getpid()} " + " ".join(f"{k} {_stats[k]}" for k in keys)
            + f" cached_mb {_pc['bytes'] >> 20} "
            + " ".join(f"{k} {tot.get(k, 0):.1f}" for k in ("C_s", "F_s", "rt_s", "copy_s", "calls")))


def _book(key, dt):
    s = getattr(_tl, "srv", None)                   # None on the torch backend: no servers to book to
    if s is None:
        return
    c = s.c
    c[key] = round(c.get(key, 0.0) + dt, 4)


def _coarse_refs(t, key, o0, project):
    """(ref, rr, key) for orientations [o0, o1): ref the fp32 [O, 2P] operand (real | imag), rr the
    float64 |ref|^2 per pixel [O, P], key the name the chips may keep the operand under (None: they
    must not). From the host cache when this orientation set was asked for before against the same
    reference; an entry is kept on its second sighting. key is None to bypass."""
    def build():
        ref = project()
        r, i = ref.double().chunk(2, 1)
        return ref.numpy(), r ** 2 + i ** 2
    if key is None:
        return build() + (None,)
    mkey, ekey = key
    k = (ekey, o0)
    with _pc_lock:
        _pc_reference(mkey)
        got = _pc["hit"].get(k)
        keep = k in _pc["seen"]
        _pc["seen"].add(k)
    if got is not None:
        _stats["proj_hit"] += 1
        return got
    _stats["proj_miss"] += 1
    ref, rr = build()
    nb = ref.nbytes + rr.numel() * 8
    if keep:
        with _pc_lock:
            if _pc["model"] == mkey:
                if k in _pc["hit"]:                 # another thread built it meanwhile: one entry, one chip key
                    return _pc["hit"][k]
                if _pc["bytes"] + nb <= _PCACHE_MB << 20:
                    # A copy: on the L10 path ref views the caller's thread_local projection buffer,
                    # which RELION overwrites (or reallocates) on that thread's next call.
                    got = _pc["hit"][k] = (ref.copy(), rr, str(len(_pc["hit"])))
                    _pc["bytes"] += nb
                    return got
    return ref, rr, None


def _pc_reference(mkey):
    """Under _pc_lock: start a fresh cache when the reference changed (new iteration or half)."""
    if _pc["model"] != mkey:
        if _pc["model"] is not None:                # one line per reference, so a run log shows hit rates
            print(_progress(), file=sys.stderr, flush=True)
        _pc.update(model=mkey, gen=_pc["gen"] + 1, seen=set(), hit={}, bytes=0)


def _coarse_first_sighting(key, orientation_num):
    """True, and the set is recorded as seen, when no chunk of this coarse call was seen before
    under the current reference."""
    mkey, ekey = key
    ks = [(ekey, o0) for o0 in range(0, orientation_num, _DEV_CHUNK)]
    with _pc_lock:
        _pc_reference(mkey)
        first = not any(k in _pc["seen"] for k in ks)
        if first:
            _pc["seen"].update(ks)
    return first


def _coarse_key(mdl_mv, eul_mv, geom):
    """The host cache's name for a coarse call's projections, None where nothing is cached."""
    if _BACKEND != "ttnn" or _PCACHE_MB <= 0:
        return None
    import hashlib
    return (_model_key(mdl_mv), (hashlib.sha1(eul_mv).digest(),) + tuple(geom) + (_DEV_CHUNK,))


def _coarse_cached(key, orientation_num):
    """True when every chunk of this coarse call is in the host cache under the current reference."""
    if key is None:
        return False
    mkey, ekey = key
    with _pc_lock:
        return _pc["model"] == mkey and all((ekey, o0) in _pc["hit"]
                                            for o0 in range(0, orientation_num, _DEV_CHUNK))


def _dense_diff2(t, mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv,
                 mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
                 padding_factor, imgX, imgY,
                 orientation_num, translation_num, image_size, tag, ref_mv=None):
    """The whole orientation x translation squared-difference matrix, flat [O*T] float32.

    This is the body of RELION's diff2_coarse_2D and, entry for entry, of diff2_fine_2D as well:
    the fine kernel differs only in which entries it is asked for and in the constant it adds.
    """
    # ref_mv: the fine pass hands over RELION's own projection, [O, 2P] real | imag, and no model.
    mdl = None if mdl_mv is None else t.from_numpy(
        np.frombuffer(mdl_mv, dtype=np.float32).reshape(-1, 2).copy())
    given = None if ref_mv is None else t.from_numpy(
        np.frombuffer(ref_mv, dtype=np.float32).reshape(orientation_num, 2 * image_size))
    eul = t.from_numpy(np.frombuffer(eul_mv, dtype=np.float32)
                       .reshape(orientation_num, 9).copy())
    tx = t.from_numpy(np.frombuffer(tx_mv, dtype=np.float32).copy())
    ty = t.from_numpy(np.frombuffer(ty_mv, dtype=np.float32).copy())
    img_r = t.from_numpy(np.frombuffer(re_mv, dtype=np.float32).copy())
    img_i = t.from_numpy(np.frombuffer(im_mv, dtype=np.float32).copy())
    w = t.from_numpy(np.frombuffer(corr_mv, dtype=np.float32).copy()) * 0.5

    pix = t.arange(image_size, dtype=t.int64)
    x = (pix % imgX).to(t.float32)
    yi = pix // imgX
    y = t.where(yi > maxR, yi - imgY, yi).to(t.float32)

    # The shift stack. A phase shift preserves magnitude, so this is the only place the
    # translations enter and it does not depend on the orientation.
    ph = x.unsqueeze(0) * tx.unsqueeze(1) + y.unsqueeze(0) * ty.unsqueeze(1)
    s, c = t.sin(ph), t.cos(ph)
    sh_r = c * img_r - s * img_i
    sh_i = c * img_i + s * img_r

    acc = np.empty(orientation_num * translation_num, dtype=np.float32)
    device = _BACKEND == "ttnn"
    if device:
        # |ref - sh|^2_w = |ref|^2_w + |sh|^2_w - 2 ref.(w sh). The two norms are cheap and stay
        # on the host in float64; only the O x T cross term goes to the chip.
        shw_np = t.cat([sh_r * w, sh_i * w], 1).numpy()
        sh2 = ((sh_r.double() ** 2 + sh_i.double() ** 2) * w.double()).sum(-1)
    chunk = _DEV_CHUNK if device else _ORI_CHUNK
    if device and tag == "F":
        chunk = max(1, min(chunk, (_FINE_CALL_MB << 20) // (40 * image_size)))
    ckey = _coarse_key(mdl_mv, eul_mv, (mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR2_padded,
                                        padding_factor, imgX, imgY, image_size)) if tag == "C" else None
    for o0 in range(0, orientation_num, chunk):
        o1 = min(o0 + chunk, orientation_num)
        if given is not None:
            project = lambda: given[o0:o1]          # noqa: E731
        else:
            project = lambda: t.cat(_project(t, mdl, x, y, eul[o0:o1], mdlX, mdlY,  # noqa: E731
                                             mdlZ, mdlInitY, mdlInitZ, maxR2_padded,
                                             padding_factor), 1)
        got = None
        if device and tag == "F" and _DEVPROJ:
            got = _cross_proj(eul[o0:o1].numpy(), shw_np, w.numpy(),
                              dict(imgX=imgX, imgY=imgY, maxR=maxR, P=image_size))
        if got is not None:
            _stats["devproj"] += 1
            d2 = t.from_numpy(got[1]).unsqueeze(1) + sh2.unsqueeze(0) - 2.0 * t.from_numpy(got[0])
        elif device:
            ref, rr, key = _coarse_refs(t, ckey, o0, project)
            ref2 = (rr * w.double()).sum(-1)
            # Kept on the chip: 3 bf16 parts (fp32 on device) padded to _KCHUNK, or OZAKI bf16 digit tensors.
            kbytes = (_OZAKI * ref.nbytes // 2 + (o1 - o0) * 480 * 2 * _OZAKI if _OZAKI
                      else 3 * ref.nbytes + (o1 - o0) * _KCHUNK * 4)
            cross = t.from_numpy(_cross(lambda: ref, o1 - o0, shw_np, key, kbytes))
            d2 = ref2.unsqueeze(1) + sh2.unsqueeze(0) - 2.0 * cross
        else:
            ref = project()
            ref_r, ref_i = ref[:, :image_size], ref[:, image_size:]
            dr = ref_r.unsqueeze(1) - sh_r.unsqueeze(0)
            di = ref_i.unsqueeze(1) - sh_i.unsqueeze(0)
            d2 = ((dr * dr + di * di) * w).sum(-1)
        acc[o0 * translation_num:o1 * translation_num] = d2.reshape(-1).float().numpy()

    if _DUMP and _dumped[0] < _DUMP_N:
        _dumped[0] += 1
        np.savez("%s.%s.%d.%d.npz" % (_DUMP, tag, os.getpid(), _dumped[0]),
                 mdl=(mdl if mdl is not None else given).numpy(), eul=eul.numpy(), tx=tx.numpy(), ty=ty.numpy(),
                 img_r=img_r.numpy(), img_i=img_i.numpy(), w=w.numpy(),
                 diff2=acc,
                 geom=np.array([mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
                                padding_factor, imgX, imgY, orientation_num,
                                translation_num, image_size], dtype=np.int64))
    if _TRACE:
        import hashlib
        eh = hashlib.sha256(np.frombuffer(eul_mv, dtype=np.float32).tobytes()).hexdigest()[:16]
        mh = hashlib.sha256(mdl_mv if mdl_mv is not None else ref_mv).hexdigest()[:16]
        with open("%s.%d" % (_TRACE, os.getpid()), "a") as fh:
            fh.write("%s %d %d %d %s %s\n"
                     % (tag, orientation_num, translation_num, image_size, eh, mh))
    return acc


def _model_key(mdl_mv):
    """Cheap identity for the model buffer, for the device path's upload cache.

    The model is 31.68 MB and byte-identical across every call in a process, so a per-call sha256
    (~50 ms) would cost more than the kernel it feeds. 1024 evenly spaced samples plus the length
    take ~10 us and a miss only costs a re-upload.
    """
    a = np.frombuffer(mdl_mv, dtype=np.float32)
    return (len(a), a[::max(1, len(a) // 1024)][:1024].tobytes())


def _marshal(mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv, orientation_num):
    """The buffer traffic the device path does before any compute, and nothing else.

    The model is keyed, not copied: §2.1 of the design measured it byte-identical across every call
    in a process, so it uploads once per iteration and the per-call cost is the key.
    """
    return (_model_key(mdl_mv),
            np.frombuffer(eul_mv, dtype=np.float32).reshape(orientation_num, 9).copy(),
            np.frombuffer(tx_mv, dtype=np.float32).copy(),
            np.frombuffer(ty_mv, dtype=np.float32).copy(),
            np.frombuffer(re_mv, dtype=np.float32).copy(),
            np.frombuffer(im_mv, dtype=np.float32).copy(),
            np.frombuffer(corr_mv, dtype=np.float32).copy())


def diff2_coarse(mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv, out_mv,
                 mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
                 padding_factor, imgX, imgY,
                 orientation_num, translation_num, image_size, ref_mv=None, can_ref=False):
    """runDiff2KernelCoarse, 3D reference and 2D data. Accumulates onto out_mv.

    ref_mv is RELION's own projection of these orientations, laid out as for diff2_fine, or None.
    A caller that can project (can_ref) calls with None first and gets 2 back, out_mv untouched,
    unless the host cache holds every projection already: a global search asks every particle
    for the same orientation set and hits, a local search asks for its own set and projects in
    C++ instead of in torch under this rank's interpreter.
    """
    if _BACKEND in ("nullbase", "null"):
        if _BACKEND == "null":
            _marshal(mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv, orientation_num)
        _stats["declined"] += 1
        return False
    try:
        t0 = time.time()                            # bridge wall per call, summed over threads
        key = _coarse_key(mdl_mv, eul_mv, (mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR2_padded,
                                           padding_factor, imgX, imgY, image_size))
        if can_ref and ref_mv is None and not _coarse_cached(key, orientation_num):
            if _COARSE_ONCE == "cpu" and key is not None and _coarse_first_sighting(key, orientation_num):
                _stats["once_cpu"] += 1
                _stats["declined"] += 1
                _book("C_s", time.time() - t0)
                return False
            _stats["ref_asked"] += 1
            _book("C_s", time.time() - t0)
            return 2
        t = _torch()
        if _DEVPROJ and _BACKEND == "ttnn":
            _keep_model(mdl_mv, dict(mdlX=mdlX, mdlY=mdlY, mdlZ=mdlZ, mdlInitY=mdlInitY, mdlInitZ=mdlInitZ,
                                     maxR2=maxR2_padded, pf=padding_factor))
        acc = _dense_diff2(t, mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv,
                           mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
                           padding_factor, imgX, imgY,
                           orientation_num, translation_num, image_size, "C", ref_mv=ref_mv)
        _book("C_s", time.time() - t0)
        out = np.frombuffer(out_mv, dtype=np.float32)
        out += acc
        _stats["handled"] += 1
        return True
    except Exception as exc:                        # declining beats killing the refinement
        import traceback
        traceback.print_exc()
        print("tt_relion.relion: diff2_coarse declined: %r" % (exc,))
        _stats["declined"] += 1
        return False


def diff2_fine(ref_mv, mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv,
               rot_idx_mv, trans_idx_mv, out_mv,
               mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
               padding_factor, imgX, imgY, sum_init,
               orientation_num, translation_num, significant_num, image_size,
               job_num_count):
    """runDiff2KernelFine, 3D reference and 2D data. Accumulates onto out_mv.

    Exactly one of ref_mv and mdl_mv is given (TT_RELION_FINE_PROJECT). ref_mv is the reference
    already projected by RELION's own AccProjectorKernel in the calling thread, [orientation_num,
    2 * image_size] float32, real | imag: in torch that projection was 65-80 % of a fine call's host
    wall and held this rank's interpreter while it ran. mdl_mv is the padded model with the same
    geometry diff2_coarse takes, for a bridge that projects itself.

    RELION's fine kernel is written as a list of jobs, each job one orientation and a run of at
    most D2F_CHUNK_REF3D consecutive translations, and it re-projects the reference once per job.
    The jobs tile [0, significant_num) exactly (makeJobsForDiff2Fine, acc_helper_functions_impl.h),
    so the job list is CUDA block geometry and carries no information the flat rot_idx/trans_idx
    pair does not. Projecting each distinct orientation once and reading the requested entries out
    of the dense matrix is therefore the same answer with strictly less projection work.
    """
    if _SHAPE:
        rot_idx = np.frombuffer(rot_idx_mv, dtype=np.uint64)
        with open("%s.%d" % (_SHAPE, os.getpid()), "a") as fh:
            fh.write("F %d %d %d %d %d %d\n"
                     % (orientation_num, translation_num, significant_num, job_num_count,
                        image_size, len(np.unique(rot_idx))))
    if _BACKEND in ("nullbase", "null"):
        # Only has to cost what an entry-and-decline costs, so that the coarse arms differ in one term.
        _stats["fine_declined"] += 1
        return False
    try:
        t0 = time.time()                            # bridge wall per call, summed over threads
        t = _torch()
        acc = _dense_diff2(t, mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv,
                           mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
                           padding_factor, imgX, imgY,
                           orientation_num, translation_num, image_size, "F", ref_mv=ref_mv)
        _book("F_s", time.time() - t0)
        rot_idx = np.frombuffer(rot_idx_mv, dtype=np.uint64)
        trans_idx = np.frombuffer(trans_idx_mv, dtype=np.uint64)
        take = rot_idx * np.uint64(translation_num) + trans_idx
        out = np.frombuffer(out_mv, dtype=np.float32)
        out += acc[take] + np.float32(sum_init)
        _stats["fine_handled"] += 1
        return True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("tt_relion.relion: diff2_fine declined: %r" % (exc,))
        _stats["fine_declined"] += 1
        return False


def stats():
    return dict(_stats)
