"""Where does a Blackhole fp32-out matmul lose bits? One tile of K, exact bf16 operands, float64 truth.

The box-360 sweep found every split (bf16x2, bf16x3, plain fp32) at the same ~2^-14 relative error on the
cross term, growing linearly in K even when each output is a single 32-term tile, and HiFi4 worse than HiFi3.
Hypothesis: each fidelity phase's partial product is truncated into dest. This probe reads the signed error
(bias) per fidelity, with and without fp32 dest, for K = 32 and K = 4096, and tries the Ozaki form: operands
cut into pieces narrow enough (5 bits for A, 7 for B) that one LoFi phase multiplies them exactly.
Prints one JSON line per config.
"""
import json
import sys
import time

import numpy as np
import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

dev = ttnn.open_device(device_id=0)
out = open(sys.argv[1], "a") if len(sys.argv) > 1 else sys.stdout


def bits_split(a, nbits, npieces):
    """a (float64) -> pieces each with <= nbits significant bits (incl. the leading one), exact in bf16."""
    parts, r = [], a.copy()
    for _ in range(npieces):
        e = np.floor(np.log2(np.abs(r) + 1e-300))
        q = np.ldexp(np.trunc(np.ldexp(r, -(e - nbits + 1).astype(int))), (e - nbits + 1).astype(int))
        q[np.abs(r) < 1e-300] = 0
        parts.append(q)
        r = r - q
    return parts


def mm(A, B, fid, dest32, l1acc=True):
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                           fp32_dest_acc_en=dest32, packer_l1_acc=l1acc)
    a = ttnn.from_torch(torch.from_numpy(A).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev)
    b = ttnn.from_torch(torch.from_numpy(B).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev)
    c = ttnn.matmul(a, b, compute_kernel_config=cfg, dtype=ttnn.float32)
    r = ttnn.to_torch(c).double().numpy()
    for x in (a, b, c):
        ttnn.deallocate(x)
    return r


def emit(**k):
    out.write(json.dumps(k) + "\n")
    out.flush()


g = np.random.default_rng(1)
for K in (32, 4096):
    for sign in ("pos", "mixed"):
        A = g.uniform(0.5, 1.0, (64, K)) if sign == "pos" else g.standard_normal((64, K))
        B = g.uniform(0.5, 1.0, (K, 64)) if sign == "pos" else g.standard_normal((K, 64))
        A = torch.from_numpy(A).to(torch.bfloat16).double().numpy()   # exact bf16 values
        B = torch.from_numpy(B).to(torch.bfloat16).double().numpy()
        C = A @ B
        scale = np.abs(A) @ np.abs(B)
        for fid in ("LoFi", "HiFi2", "HiFi3", "HiFi4"):
            for dest32 in (True, False):
                t0 = time.time()
                D = mm(A, B, fid, dest32)
                e = (D - C) / scale
                emit(probe="bf16x1", K=K, sign=sign, fid=fid, dest32=dest32, rel_max=float(np.abs(e).max()),
                     rel_bias=float(e.mean()), log2_rel=float(np.log2(np.abs(e).max() + 1e-30)),
                     s=round(time.time() - t0, 3))
        # Ozaki: A in 5-bit pieces, B in 7-bit pieces, every piece pair one LoFi matmul (exact products)
        Af = g.standard_normal((64, K))
        Bf = g.standard_normal((K, 64))
        Af32 = Af.astype(np.float32).astype(np.float64)
        Bf32 = Bf.astype(np.float32).astype(np.float64)
        C = Af32 @ Bf32
        scale = np.abs(Af32) @ np.abs(Bf32)
        for na, nb, fid in ((5, 4, "LoFi"), (3, 2, "HiFi4"), (3, 2, "HiFi3")):
            if fid == "LoFi":
                pa, pb = bits_split(Af32, 5, na), bits_split(Bf32, 7, nb)
            else:
                pa, pb = bits_split(Af32, 8, na), bits_split(Bf32, 8, nb)
            for dest32 in (True,):
                t0 = time.time()
                acc = np.zeros_like(C)
                for i, x in enumerate(pa):
                    for j, y in enumerate(pb):
                        acc += mm(x, y, fid, dest32)
                e = (acc - C) / scale
                emit(probe="pieces", K=K, sign="mixed", fid=fid, na=na, nb=nb, dest32=dest32,
                     rel_max=float(np.abs(e).max()), rel_bias=float(e.mean()),
                     log2_rel=float(np.log2(np.abs(e).max() + 1e-30)), s=round(time.time() - t0, 3))
ttnn.close_device(dev)
print("PROBE-DONE", flush=True)
