"""Is any matmul config's accumulation wider than the ~11 bits probe_int.py found? Same integer test, varied configs.
    TT_VISIBLE_DEVICES=<chip> python3 probe_int2.py out.jsonl
"""
import json
import sys

import numpy as np
import torch
import ttnn

from _tt import open_dev

dev, _ = open_dev()
out = open(sys.argv[1], "a")


def mm(A, B, fid, l1acc, dt):
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=l1acc)
    a, b = (ttnn.from_torch(torch.from_numpy(x).to(torch.float32), dtype=dt, layout=ttnn.TILE_LAYOUT,
                            device=dev) for x in (A, B))
    c = ttnn.matmul(a, b, compute_kernel_config=cfg, dtype=ttnn.float32)
    r = ttnn.to_torch(c).double().numpy()
    for x in (a, b, c):
        ttnn.deallocate(x)
    return r


g = np.random.default_rng(4)
for K, b in ((32, 7), (64, 6), (8192, 2), (8192, 4)):
    h = 2 ** (b - 1)
    A = g.integers(-h, h, (64, K)).astype(np.float64)
    B = g.integers(-h, h, (K, 64)).astype(np.float64)
    C = (A.astype(np.int64) @ B.astype(np.int64)).astype(np.float64)
    # all-positive variant: no cancellation, sum grows monotonically
    Ap, Bp = np.abs(A), np.abs(B)
    Cp = (Ap.astype(np.int64) @ Bp.astype(np.int64)).astype(np.float64)
    for l1acc in (True, False):
        for dtn in ("bfloat16", "float32"):
            for tag, (x, y, z) in (("signed", (A, B, C)), ("pos", (Ap, Bp, Cp))):
                e = np.abs(mm(x, y, "HiFi4", l1acc, getattr(ttnn, dtn)) - z)
                out.write(json.dumps(dict(K=K, b=b, l1acc=l1acc, dt=dtn, sign=tag, err_max=float(e.max()),
                                          frac_wrong=round(float((e > 0).mean()), 3),
                                          sum_bits=round(float(np.log2(np.abs(z).max() + 1)), 1))) + "\n")
                out.flush()
ttnn.close_device(dev)
print("INT2-DONE", flush=True)
