"""Which operand piece width makes a Blackhole fp32-dest matmul exact? Float64 truth, one JSON line per config.

probe_acc.py found exact bf16 operands still miss by ~2^-13 relative at HiFi4 (unbiased noise, so products are
rounded, not a dropped fidelity phase) and 5x7-bit pieces at LoFi miss by 2^-5. Here both fp32 operands are cut
into w-bit pieces (w = significant bits incl. the leading one), every pair with i + j < n is one matmul, partial
sums are added in float64 on the host. The width at which the error drops to the fp32-accumulation floor is the
widest product the FPU keeps exactly; that sets the cost of an exact split (number of matmuls).
    TT_VISIBLE_DEVICES=<chip> python3 probe_pieces.py out.jsonl
"""
import json
import sys
import time

import numpy as np
import torch
import ttnn

from _tt import open_dev
from probe_acc_util import bits_split

dev, _ = open_dev()
out = open(sys.argv[1], "a")


def mm(A, B, fid):
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    a, b = (ttnn.from_torch(torch.from_numpy(x).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev) for x in (A, B))
    c = ttnn.matmul(a, b, compute_kernel_config=cfg, dtype=ttnn.float32)
    r = ttnn.to_torch(c).double().numpy()
    for x in (a, b, c):
        ttnn.deallocate(x)
    return r


g = np.random.default_rng(2)
for K in (32, 4096):
    A = g.standard_normal((64, K)).astype(np.float32).astype(np.float64)
    B = g.standard_normal((K, 64)).astype(np.float32).astype(np.float64)
    C, scale = A @ B, np.abs(A) @ np.abs(B)
    e32 = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float64) - C
    out.write(json.dumps(dict(K=K, cfg="fp32 host", rel_max=float(np.abs(e32 / scale).max()))) + "\n")
    for w in (2, 3, 4, 5, 6, 8):
        n = -(-24 // w)
        pa, pb = bits_split(A, w, n), bits_split(B, w, n)
        assert np.array_equal(sum(pa), A) and np.array_equal(sum(pb), B)
        for fid in ("LoFi", "HiFi2", "HiFi4"):
            t0, acc, nmm = time.time(), np.zeros_like(C), 0
            for i in range(n):
                for j in range(n - i):
                    acc += mm(pa[i], pb[j], fid)
                    nmm += 1
            e = (acc - C) / scale
            out.write(json.dumps(dict(K=K, w=w, pieces=n, matmuls=nmm, fid=fid, rel_max=float(np.abs(e).max()),
                                      log2_rel=float(np.log2(np.abs(e).max() + 1e-30)), rel_bias=float(e.mean()),
                                      s=round(time.time() - t0, 2))) + "\n")
            out.flush()
ttnn.close_device(dev)
print("PIECES-DONE", flush=True)
