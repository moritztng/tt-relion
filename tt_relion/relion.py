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
            translation cross term on the chip. See _device_cross.
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
import sys
import queue
import threading
import time

import numpy as np

_BACKEND = os.environ.get("TT_RELION_BACKEND", "torch")
_ORI_CHUNK = int(os.environ.get("TT_RELION_ORI_CHUNK", "32"))
_CHECK = os.environ.get("TT_RELION_CHECK", "") not in ("", "0")
# TT_RELION_DEV_CHUNK: orientations per device matmul. Larger amortises the launch, smaller bounds
# the host copy of the projected references.
_DEV_CHUNK = int(os.environ.get("TT_RELION_DEV_CHUNK", "1024"))
_FIDELITY = os.environ.get("TT_RELION_FIDELITY", "HiFi3")
_SPLIT = int(os.environ.get("TT_RELION_SPLIT", "3"))
_KCHUNK = int(os.environ.get("TT_RELION_KCHUNK", "256"))
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
_stats = {"handled": 0, "declined": 0, "fine_handled": 0, "fine_declined": 0}


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


class _Server:
    """One chip's device server (tt_relion/device_server.py) and the shared memory it reads."""

    def __init__(self, chip=None):
        import subprocess
        env = {k: v for k, v in os.environ.items() if not k.startswith(_MPI_ENV)}
        if chip is not None:
            env["TT_VISIBLE_DEVICES"] = chip
        self.p = subprocess.Popen([_PYTHON, "-m", "tt_relion.device_server", _FIDELITY],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True)
        self.lock = threading.Lock()
        self.shm = None
        self.c = {"calls": 0, "bytes": 0, "rt_s": 0.0, "copy_s": 0.0}   # echoed into the server's exit line

    def matmul(self, lhs, rhs):
        """lhs [nc, m, kc] @ rhs [nc, kc, n] -> [nc, m, n], float32, on the chip."""
        from multiprocessing import shared_memory
        nc, m, kc = lhs.shape
        n = rhs.shape[2]
        need = 4 * (lhs.size + rhs.size + nc * m * n)
        with self.lock:
            if self.shm is None or self.shm.size < need:
                if self.shm is not None:
                    self.shm.close()
                    self.shm.unlink()
                self.shm = shared_memory.SharedMemory(create=True, size=need)
            t0 = time.time()
            buf = np.ndarray((need // 4,), np.float32, self.shm.buf)
            buf[:lhs.size] = lhs.reshape(-1)
            buf[lhs.size:lhs.size + rhs.size] = rhs.reshape(-1)
            t1 = time.time()
            self.p.stdin.write(json.dumps({"shm": self.shm.name, "shape": [nc, m, kc, n], "client": self.c},
                                          separators=(",", ":")) + "\n")
            self.p.stdin.flush()
            if self.p.stdout.readline().strip() != "ok":
                raise RuntimeError("device server died, rc=%s" % self.p.poll())
            t2 = time.time()
            out = buf[lhs.size + rhs.size:].reshape(nc, m, n).copy()
            del buf
            c = self.c
            c["calls"] += 1
            c["bytes"] += need
            c["rt_s"] = round(c["rt_s"] + t2 - t1, 4)
            c["copy_s"] = round(c["copy_s"] + t1 - t0 + time.time() - t2, 4)
        return out


def _matmul(lhs, rhs):
    """On whichever of this rank's chips is idle. RLX_RANK_CHIPS="a,b,..." gives a rank several chips
    (one server each), so fewer ranks, and so less host memory, can still drive every chip."""
    global _free
    if _free is None:
        with _lock:
            if _free is None:
                chips = [c for c in os.environ.get("RLX_RANK_CHIPS", "").split(",") if c] or [None]
                _srv.extend(_Server(c) for c in chips)
                q = queue.Queue()
                for s in _srv:
                    q.put(s)
                _free = q
    s = _free.get()
    try:
        _tl.srv = s
        return s.matmul(lhs, rhs)
    finally:
        _free.put(s)


def _book(key, dt):
    srv = getattr(_tl, "srv", None)              # the torch backend has no server to book to
    if srv is None:
        return
    c = srv.c
    c[key] = round(c.get(key, 0.0) + dt, 4)


def _bf16_split(t, a, k):
    """a = a[0] + ... + a[k-1], every part exactly representable in bf16.

    Measured on Wormhole (perf/rlx/parity_cross.py): fp32 operands that are exact in tf32 still come
    out of the matrix unit with a fidelity-dependent error (HiFi4 9e-5 relative, HiFi2 1e-3), so it
    does not multiply tf32 exactly. bf16 parts are exact in whatever format the unit reads.
    """
    parts = []
    for _ in range(k - 1):
        hi = (a.view(t.int32) & ~0xFFFF).view(t.float32)
        parts.append(hi)
        a = a - hi
    return parts + [a]


def _device_cross(t, ref, shw):
    """C[o, t] = sum_p ref[o, p] * shw[t, p] on the chip, fp32 in, fp32 accumulate.

    ref [O, K] and shw [T, K] are real (real and imaginary parts concatenated along K). Both are
    split into _SPLIT bf16 parts and C = sum over i + j < _SPLIT of a_i . b_j, which drops terms of
    relative size 2^-24 at _SPLIT=3. All of it is ONE matmul: the left operand is [a_0 | a_1 | ...]
    and output column block j collects b_j against every a_i with i + j < _SPLIT.
    """
    k, n, kk = _SPLIT, shw.shape[0], ref.shape[1]
    lhs = t.cat(_bf16_split(t, ref, k), 1)
    bs = _bf16_split(t, shw, k)
    rhs = t.zeros(k * kk, k * n, dtype=t.float32)
    for i in range(k):
        for j in range(k - i):
            rhs[i * kk:(i + 1) * kk, j * n:(j + 1) * n] = bs[j].T
    # With exact products the chip still loses bits in accumulation, so K is cut into _KCHUNK runs,
    # each its own batch entry, summed on the host in float64. Measured at box 64 (parity_cross.py):
    # max |d2 error| 0.54 unchunked, 0.35 at 256, 0.25 at 64; fp32 torch is 0.005.
    kc = _KCHUNK if _KCHUNK > 0 else lhs.shape[1]
    nc = -(-lhs.shape[1] // kc)
    pad = nc * kc - lhs.shape[1]
    if pad:
        lhs = t.nn.functional.pad(lhs, (0, pad))
        rhs = t.nn.functional.pad(rhs, (0, 0, 0, pad))
    lhs = lhs.reshape(lhs.shape[0], nc, kc).permute(1, 0, 2).contiguous()
    rhs = rhs.reshape(nc, kc, rhs.shape[1])
    out = _matmul(lhs.numpy(), rhs.numpy())
    return t.from_numpy(out).double().sum(0).reshape(-1, k, n).sum(1)


def _dense_diff2(t, mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv,
                 mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
                 padding_factor, imgX, imgY,
                 orientation_num, translation_num, image_size, tag):
    """The whole orientation x translation squared-difference matrix, flat [O*T] float32.

    This is the body of RELION's diff2_coarse_2D and, entry for entry, of diff2_fine_2D as well:
    the fine kernel differs only in which entries it is asked for and in the constant it adds.
    """
    mdl = t.from_numpy(np.frombuffer(mdl_mv, dtype=np.float32).reshape(-1, 2).copy())
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
        shw = t.cat([sh_r * w, sh_i * w], 1)
        sh2 = ((sh_r.double() ** 2 + sh_i.double() ** 2) * w.double()).sum(-1)
    chunk = _DEV_CHUNK if device else _ORI_CHUNK
    for o0 in range(0, orientation_num, chunk):
        o1 = min(o0 + chunk, orientation_num)
        ref_r, ref_i = _project(t, mdl, x, y, eul[o0:o1], mdlX, mdlY, mdlZ,
                                mdlInitY, mdlInitZ, maxR2_padded, padding_factor)
        if device:
            ref2 = ((ref_r.double() ** 2 + ref_i.double() ** 2) * w.double()).sum(-1)
            cross = _device_cross(t, t.cat([ref_r, ref_i], 1), shw).double()
            d2 = ref2.unsqueeze(1) + sh2.unsqueeze(0) - 2.0 * cross
        else:
            dr = ref_r.unsqueeze(1) - sh_r.unsqueeze(0)
            di = ref_i.unsqueeze(1) - sh_i.unsqueeze(0)
            d2 = ((dr * dr + di * di) * w).sum(-1)
        acc[o0 * translation_num:o1 * translation_num] = d2.reshape(-1).float().numpy()

    if _DUMP and _dumped[0] < _DUMP_N:
        _dumped[0] += 1
        np.savez("%s.%s.%d.%d.npz" % (_DUMP, tag, os.getpid(), _dumped[0]),
                 mdl=mdl.numpy(), eul=eul.numpy(), tx=tx.numpy(), ty=ty.numpy(),
                 img_r=img_r.numpy(), img_i=img_i.numpy(), w=w.numpy(),
                 diff2=acc,
                 geom=np.array([mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
                                padding_factor, imgX, imgY, orientation_num,
                                translation_num, image_size], dtype=np.int64))
    if _TRACE:
        import hashlib
        eh = hashlib.sha256(np.frombuffer(eul_mv, dtype=np.float32).tobytes()).hexdigest()[:16]
        mh = hashlib.sha256(np.frombuffer(mdl_mv, dtype=np.float32).tobytes()).hexdigest()[:16]
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
                 orientation_num, translation_num, image_size):
    """runDiff2KernelCoarse, 3D reference and 2D data. Accumulates onto out_mv."""
    if _BACKEND in ("nullbase", "null"):
        if _BACKEND == "null":
            _marshal(mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv, orientation_num)
        _stats["declined"] += 1
        return False
    try:
        t0 = time.time()                            # bridge wall per call, summed over threads
        t = _torch()
        geom = (mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded, padding_factor, imgX, imgY)
        if _BATCH > 1:
            acc = _collect(mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv, geom,
                           orientation_num).reshape(-1)
        else:
            acc = _dense_diff2(t, mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv,
                               *geom, orientation_num, translation_num, image_size, "C")
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


def diff2_fine(mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv,
               rot_idx_mv, trans_idx_mv, out_mv,
               mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
               padding_factor, imgX, imgY, sum_init,
               orientation_num, translation_num, significant_num, image_size,
               job_num_count):
    """runDiff2KernelFine, 3D reference and 2D data. Accumulates onto out_mv.

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
                           orientation_num, translation_num, image_size, "F")
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


# ---- batch entry: one call per particle batch (the resident E-step's C++ call) ----

_refcache = {}       # (model key, euler hash) -> (ref_r, ref_i) for the shared coarse set


def _shared_refs(t, mdl_mv, eul, x, y, geom):
    """Projections of the coarse orientation set, built once per iteration and reused by every batch.

    In a global pass every particle scores the same orientations against the same reference, so the
    projection is a property of the iteration, not of the particle.
    """
    import hashlib
    key = (_model_key(mdl_mv), hashlib.sha256(eul.numpy().tobytes()).digest(), x.shape[0])
    hit = _refcache.get(key)
    if hit is None:
        _refcache.clear()                           # one iteration's set at a time
        mdl = t.from_numpy(np.frombuffer(mdl_mv, dtype=np.float32).reshape(-1, 2).copy())
        rr, ri = [], []
        for o0 in range(0, eul.shape[0], _DEV_CHUNK):
            a, b = _project(t, mdl, x, y, eul[o0:o0 + _DEV_CHUNK], *geom)
            rr.append(a), ri.append(b)
        hit = _refcache[key] = (t.cat(rr), t.cat(ri))
    return hit


def diff2_coarse_batch(mdl, eul, tx, ty, img_r, img_i, corr,
                       mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2_padded,
                       padding_factor, imgX, imgY):
    """diff2 for N particles over one shared orientation set: float32 [N, O, T].

    img_r/img_i/corr are [N, image_size], tx/ty are [N, T] (each particle's own offsets), eul is
    [O, 9] and shared. The cross term is one product of [N*T, 2P] by [2P, O]; |ref|^2_w is one
    product of [N, P] by [P, O]; |shift|^2_w is a row norm. Entry for entry this equals N calls of
    diff2_coarse, which tests/test_batch.py checks.
    """
    t = _torch()
    n, image_size = img_r.shape
    T = tx.shape[1]
    eul = t.from_numpy(np.ascontiguousarray(eul, dtype=np.float32).reshape(-1, 9))
    pix = t.arange(image_size, dtype=t.int64)
    x = (pix % imgX).to(t.float32)
    yi = pix // imgX
    y = t.where(yi > maxR, yi - imgY, yi).to(t.float32)
    ref_r, ref_i = _shared_refs(t, mdl, eul, x, y,
                                (mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR2_padded, padding_factor))
    w = t.from_numpy(np.asarray(corr, np.float32)) * 0.5                      # [N, P]
    ir, ii = t.from_numpy(np.asarray(img_r, np.float32)), t.from_numpy(np.asarray(img_i, np.float32))
    ph = (x[None, None] * t.from_numpy(np.asarray(tx, np.float32))[..., None]
          + y[None, None] * t.from_numpy(np.asarray(ty, np.float32))[..., None])   # [N, T, P]
    s, c = t.sin(ph), t.cos(ph)
    sh_r = c * ir[:, None] - s * ii[:, None]
    sh_i = c * ii[:, None] + s * ir[:, None]
    wd = w.double()
    sh2 = ((sh_r.double() ** 2 + sh_i.double() ** 2) * wd[:, None]).sum(-1)          # [N, T]
    ref2 = wd @ (ref_r.double() ** 2 + ref_i.double() ** 2).T                         # [N, O]
    shw = t.cat([sh_r * w[:, None], sh_i * w[:, None]], -1).reshape(n * T, -1)       # [N*T, 2P]
    refc = t.cat([ref_r, ref_i], 1)                                                  # [O, 2P]
    if _BACKEND == "ttnn":
        cross = _device_cross(t, refc, shw).double()                                  # [O, N*T]
    else:
        cross = (refc.double() @ shw.double().T)
    cross = cross.reshape(-1, n, T).permute(1, 0, 2)                                  # [N, O, T]
    return (ref2[:, :, None] + sh2[:, None, :] - 2.0 * cross).float().numpy()


# TT_RELION_BATCH=<n>: coarse calls from RELION's --j threads that share an orientation set are
# grouped, up to n at a time, into one diff2_coarse_batch. A thread that finds no partner within
# TT_RELION_BATCH_WAIT_MS runs the batch it has, so a lone particle (the last of a rank's range)
# never stalls.
_BATCH = int(os.environ.get("TT_RELION_BATCH", "0"))
_BATCH_WAIT = float(os.environ.get("TT_RELION_BATCH_WAIT_MS", "20")) / 1e3
_groups = {}
_gcv = threading.Condition()


def _collect(mdl_mv, eul_mv, tx_mv, ty_mv, re_mv, im_mv, corr_mv, geom, orientation_num):
    """Queue this particle with its group; the first waiter with no active leader leads a batch."""
    import hashlib
    eul = np.frombuffer(eul_mv, dtype=np.float32)
    tx = np.frombuffer(tx_mv, dtype=np.float32)
    key = (_model_key(mdl_mv), hashlib.sha256(eul.tobytes()).digest(), tx.size, geom)
    item = {"tx": tx.copy(), "ty": np.frombuffer(ty_mv, dtype=np.float32).copy(),
            "ir": np.frombuffer(re_mv, dtype=np.float32).copy(),
            "ii": np.frombuffer(im_mv, dtype=np.float32).copy(),
            "w": np.frombuffer(corr_mv, dtype=np.float32).copy(), "out": None, "err": None}
    with _gcv:
        g = _groups.setdefault(key, {"q": [], "led": False})
        g["q"].append(item)
        _gcv.notify_all()
        while True:
            if item["out"] is not None or item["err"] is not None:
                break
            if not g["led"]:
                g["led"] = True
                deadline = time.time() + _BATCH_WAIT
                while len(g["q"]) < _BATCH and time.time() < deadline:
                    _gcv.wait(deadline - time.time())
                batch = g["q"][:_BATCH]
                del g["q"][:_BATCH]
                _gcv.release()
                try:
                    st = lambda k: np.stack([b[k] for b in batch])
                    res = diff2_coarse_batch(mdl_mv, eul.reshape(orientation_num, 9), st("tx"),
                                             st("ty"), st("ir"), st("ii"), st("w"), *geom)
                    for b, r in zip(batch, res):
                        b["out"] = r
                    _stats["batches"] = _stats.get("batches", 0) + 1
                    _stats["batched"] = _stats.get("batched", 0) + len(batch)
                    if _stats["batches"] % 50 == 0:
                        print(f"tt_relion: coarse batches={_stats['batches']} particles="
                              f"{_stats['batched']} mean={_stats['batched'] / _stats['batches']:.2f}",
                              file=sys.stderr, flush=True)
                except Exception as exc:            # every particle of the batch declines
                    for b in batch:
                        b["err"] = exc
                finally:
                    _gcv.acquire()
                    g["led"] = False
                    if not g["q"]:
                        _groups.pop(key, None)
                    _gcv.notify_all()
            else:
                _gcv.wait()
    if item["err"] is not None:
        raise item["err"]
    return item["out"]
