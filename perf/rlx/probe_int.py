"""How many bits does the Blackhole in-FPU sum keep exactly? Integer operands, exact int64 truth.

probe_pieces.py showed the BH matmul error lives in the in-tile accumulation (~2^-13 of the output), not in the
products. If that sum is exact for integers up to some width, an Ozaki split (shared exponent per row/column, integer
pieces) gives an exact cross term. Operands are uniform integers in [-2^(b-1), 2^(b-1)); one JSON line per
(K, b, fidelity) with the max abs error in units of 1 and the bit width of the largest exact sum.
    TT_VISIBLE_DEVICES=<chip> python3 probe_int.py out.jsonl
"""
import json
import sys

import numpy as np
import torch
import ttnn

from _tt import open_dev

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


g = np.random.default_rng(3)
for K in (32, 64, 1024, 8192):
    for b in (2, 3, 4, 5, 6, 7, 8):
        h = 2 ** (b - 1)
        A = g.integers(-h, h, (64, K)).astype(np.float64)
        B = g.integers(-h, h, (K, 64)).astype(np.float64)
        C = (A.astype(np.int64) @ B.astype(np.int64)).astype(np.float64)
        for fid in ("LoFi", "HiFi2", "HiFi4"):
            e = np.abs(mm(A, B, fid) - C)
            out.write(json.dumps(dict(K=K, b=b, fid=fid, err_max=float(e.max()), exact=bool(e.max() == 0),
                                      frac_wrong=float((e > 0).mean()),
                                      sum_bits=float(np.log2(np.abs(C).max() + 1)))) + "\n")
            out.flush()
ttnn.close_device(dev)
print("INT-DONE", flush=True)
