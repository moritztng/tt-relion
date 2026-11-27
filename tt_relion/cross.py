"""The O x T cross term C[o, t] = sum_k ref[o, k] * shw[t, k], served by one chip's server process.

Both device_server.py (ttnn) and stub_server.py (numpy, no device) run serve() below; they differ only
in the backend's upload() and mm(). The bridge (relion.py) ships fp32 ref [O, K] and shw [T, K] and gets
back float64 C [O, T]; the bf16 split, the block-triangular right operand, the K-chunk cut and the
float64 chunk sum all happen here, in the server's own process, not under the bridge's GIL.

Exactness: the chip's matrix unit is exact only on bf16 operands (perf/rlx/parity_cross.py), so both
sides are split into SPLIT bf16 parts and C = sum over i + j < SPLIT of a_i . b_j (terms of relative
size 2^-24 dropped at SPLIT=3). One matmul does it: the left operand is [a_0 | a_1 | ...] and output
column block j collects b_j against every a_i with i + j < SPLIT. The chip still loses bits in
accumulation, so K is cut into KCHUNK runs, each its own batch entry, summed here in float64.

Resident references. A coarse pass in a global search asks every particle for the same orientation set
against the same reference, so the left operand is a per-iteration constant. A request may name a key:
with "ref" present the operand is built and kept under that key, without it the kept one is used.
"gen" changes with the reference; a new gen drops every kept operand.

Protocol, one JSON line per request on stdin, one line back ("ok"):
    {"shm": name, "O": o, "T": t, "K": k, "ref": bool, "key": str|null, "gen": int, "client": {...}}
Segment: [ref fp32 O*K, only if "ref"] [shw fp32 T*K] [out float64 O*T].

Projected on the chip. The server keeps up to two 3D models (the two half-set references), each sent
once:  {"shm", "model": mkey, "nvox": n, "geom": {mdlX mdlY}}  with segment [model fp32 n*2] (RELION's voxels).
A request with "proj" then ships no reference; the server projects the O orientations itself:
    "proj": {"mkey": str, "geom": {mdlX mdlY mdlZ mdlInitY mdlInitZ maxR2 pf imgX imgY maxR P}}
Segment: [eul fp32 O*9] [shw fp32 T*K] [w fp32 K/2] -> [out float64 O*T] [ref2 float64 O],
ref2[o] = sum_p w[p] |ref[o, p]|^2 (the bridge needs it for d2), summed by the backend without a matmul:
an all-positive sum is where the chip's accumulation error shows most. The reply is "nomodel" if mkey is not kept (evicted).
EOF on stdin closes the backend and exits; the exit line is the TTDEV line perf/rlx/wallmap.py reads.
"""
import json
import os
import sys
import time

import numpy as np

SPLIT = int(os.environ.get("TT_RELION_SPLIT", "3"))
KCHUNK = int(os.environ.get("TT_RELION_KCHUNK", "256"))


def bf16_split(a, k=SPLIT):
    """a = a[0] + ... + a[k-1], every part exactly representable in bf16."""
    parts = []
    for _ in range(k - 1):
        hi = (a.view(np.int32) & np.int32(~0xFFFF)).view(np.float32)
        parts.append(hi)
        a = a - hi
    return parts + [a]


def lhs_operand(ref):
    """ref [O, K] fp32 -> [nc, O, kc]: the SPLIT parts side by side, cut into KCHUNK runs."""
    o = ref.shape[0]
    lhs = np.concatenate(bf16_split(ref), 1)
    kc = KCHUNK if KCHUNK > 0 else lhs.shape[1]
    nc = -(-lhs.shape[1] // kc)
    if nc * kc > lhs.shape[1]:
        lhs = np.pad(lhs, ((0, 0), (0, nc * kc - lhs.shape[1])))
    return np.ascontiguousarray(lhs.reshape(o, nc, kc).transpose(1, 0, 2))


def rhs_operand(shw, nc, kc):
    """shw [T, K] fp32 -> [nc, kc, SPLIT*T], block-triangular (see the module docstring)."""
    k, n, kk = SPLIT, shw.shape[0], shw.shape[1]
    bs = bf16_split(shw)
    rhs = np.zeros((nc * kc, k * n), np.float32)
    for i in range(k):
        for j in range(k - i):
            rhs[i * kk:(i + 1) * kk, j * n:(j + 1) * n] = bs[j].T
    return rhs.reshape(nc, kc, k * n)


def reduce(out, n):
    """[nc, O, SPLIT*T] fp32 chunk partials -> [O, T] float64."""
    return out.sum(0, dtype=np.float64).reshape(-1, SPLIT, n).sum(1)


def proj_lhs(ref, p, kc):
    """ref [O, 2P] fp32 -> [nc, O, kc] for a "proj" request: chunk order (bf16 part, real/imag, pixel run of kc),
    each half zero-padded to whole runs, so a chip that projects in pixel runs of kc can build it in place."""
    o, npc = ref.shape[0], -(-p // kc)
    out = []
    for part in bf16_split(np.ascontiguousarray(ref, np.float32)):
        for h in (0, 1):
            seg = np.zeros((o, npc * kc), np.float32)
            seg[:, :p] = part[:, h * p:(h + 1) * p]
            out.append(seg.reshape(o, npc, kc).transpose(1, 0, 2))
    return np.ascontiguousarray(np.concatenate(out))


def proj_rhs(shw, p, kc):
    """shw [T, 2P] -> [nc, kc, SPLIT*T] matching proj_lhs: chunk (i, h, c) carries b_j for every i + j < SPLIT."""
    n, npc = shw.shape[0], -(-p // kc)
    bs = bf16_split(np.ascontiguousarray(shw, np.float32))
    rhs = np.zeros((SPLIT, 2, npc, kc, SPLIT * n), np.float32)
    for j in range(SPLIT):
        for h in (0, 1):
            seg = np.zeros((n, npc * kc), np.float32)
            seg[:, :p] = bs[j][:, h * p:(h + 1) * p]
            seg = seg.reshape(n, npc, kc).transpose(1, 2, 0)
            for i in range(SPLIT - j):
                rhs[i, h, :, :, j * n:(j + 1) * n] = seg
    return rhs.reshape(-1, kc, SPLIT * n)


def pixels(g):
    """The Fourier pixels of one image, as the bridge's _dense_diff2 builds them: x, y float32 [P]."""
    pix = np.arange(g["P"])
    yi = pix // g["imgX"]
    return (pix % g["imgX"]).astype(np.float32), np.where(yi > g["maxR"], yi - g["imgY"], yi).astype(np.float32)


def _projected(backend, req, buf, o, t, k):
    """One "proj" request: the chip projects, then the cross term and ref2. Returns (C, ref2) float64."""
    pj, g = req["proj"], req["proj"]["geom"]
    off = 0
    eul = np.ndarray((o, 9), np.float32, buf, offset=off)
    shw = np.ndarray((t, k), np.float32, buf, offset=off + 36 * o)
    w = np.ndarray((k // 2,), np.float32, buf, offset=off + 36 * o + 4 * t * k)
    a, ref2 = backend.project(pj["mkey"], eul, g, w)  # proj_lhs layout of ref, on the chip; ref2 float64
    c = reduce(backend.mm(a, proj_rhs(shw, g["P"], backend.shape(a)[1])), t)
    backend.free(a)
    return c, ref2, off + 36 * o + 4 * t * k + 2 * k


def serve(backend, reply):
    from multiprocessing import resource_tracker, shared_memory
    shm, gen, kept = None, None, {}
    t_open, calls, client = time.time(), 0, {}
    up = mm = down = 0.0
    for line in sys.stdin:
        req = json.loads(line)
        t0 = time.time()
        if shm is None or shm.name != req["shm"].lstrip("/"):
            if shm is not None:
                shm.close()
            shm = shared_memory.SharedMemory(name=req["shm"])
            resource_tracker.unregister(shm._name, "shared_memory")   # the bridge owns it
        if req.get("gen", gen) != gen:
            for a in kept.values():
                backend.free(a)
            gen, kept = req["gen"], {}
        if "model" in req:
            try:   # a model the chip cannot hold (memory) is declined; the bridge then uses RELION's projection
                backend.set_model(req["model"], np.ndarray((req["nvox"], 2), np.float32, shm.buf), req["geom"])
                r = "ok"
            except Exception as exc:  # noqa: BLE001
                print("tt_relion.cross: model %s declined: %r" % (req["model"], exc), file=sys.stderr, flush=True)
                r = "nomodel"
            reply.write(r + "\n")
            reply.flush()
            continue
        o, t, k = req["O"], req["T"], req["K"]
        if req.get("proj"):
            try:
                c, ref2, off = _projected(backend, req, shm.buf, o, t, k)
            except Exception as exc:  # noqa: BLE001  (KeyError: model not kept; anything else: declined)
                if not isinstance(exc, KeyError):
                    print("tt_relion.cross: proj declined: %r" % (exc,), file=sys.stderr, flush=True)
                reply.write("nomodel\n")
                reply.flush()
                continue
            np.ndarray((o, t), np.float64, shm.buf, offset=off)[...] = c
            np.ndarray((o,), np.float64, shm.buf, offset=off + 8 * o * t)[...] = ref2
            t3 = time.time()
            mm += t3 - t0
            client = req.get("client", client)
            calls += 1
            reply.write("ok\n")
            reply.flush()
            continue
        nref = o * k if req["ref"] else 0
        f32 = np.ndarray((nref + t * k,), np.float32, shm.buf)
        if req["ref"]:
            a = backend.upload(lhs_operand(f32[:nref].reshape(o, k)))
            if req["key"] is not None:
                kept[req["key"]] = a
        else:
            a = kept[req["key"]]
        nc, kc = backend.shape(a)
        rhs = rhs_operand(f32[nref:].reshape(t, k), nc, kc)
        t1 = time.time()
        part = backend.mm(a, rhs)
        t2 = time.time()
        if req["ref"] and req["key"] is None:
            backend.free(a)
        off = 4 * (nref + t * k)
        np.ndarray((o, t), np.float64, shm.buf, offset=off)[...] = reduce(part, t)
        del f32
        t3 = time.time()
        up, mm, down = up + t1 - t0, mm + t2 - t1, down + t3 - t2
        client = req.get("client", client)
        calls += 1
        reply.write("ok\n")
        reply.flush()
    if shm is not None:
        shm.close()
    backend.close()
    # busy_s = up + mm + down. up_s: segment map, split, operand build (and the reference upload when
    # shipped); mm_s: rhs upload + matmul + result download; down_s: the float64 chunk sum. client = the bridge's counters.
    print("TTDEV chip=%s calls=%d busy_s=%.3f up_s=%.3f mm_s=%.3f down_s=%.3f life_s=%.3f client=%s"
          % (os.environ.get("TT_VISIBLE_DEVICES", "?"), calls, up + mm + down, up, mm, down,
             time.time() - t_open, json.dumps(client, separators=(",", ":"))), file=sys.stderr, flush=True)
