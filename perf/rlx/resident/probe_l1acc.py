"""Is a long-K bf16 digit matmul EXACT when every K block is <= 480 wide and the blocks are accumulated by the packer in
L1 (packer_l1_acc) at fp32? probe_int found ~11 accumulation bits with ttnn's default configs; if the loss is inside
a K block only, an explicit in0_block_w <= 15 tiles makes one matmul per Ozaki diagonal exact (no 480-wide run outputs,
no fp32 run adds). Digits in [-2, 2]; adversarial all-+2 case and random case; compares to exact int64.
  TT_VISIBLE_DEVICES=16 python probe_l1acc.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _tt import open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, nodes = open_dev()
g = dev.compute_with_storage_grid_size()
rng = np.random.default_rng(0)
M, N = 256, 512
for K in (2048, 18432):
    for case in ("pos12", "pos012"):
        # positive digits: partial sums grow ~K with many significant bits (the case that exposes a short accumulator)
        if case == "pos12":
            A = rng.integers(1, 3, (M, K)); B = rng.integers(1, 3, (K, N))
        else:
            A = rng.integers(0, 3, (M, K)); B = rng.integers(1, 3, (K, N))
        ex = A.astype(np.int64) @ B.astype(np.int64)
        ta = ttnn.from_torch(torch.from_numpy(A.astype(np.float32)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tb = ttnn.from_torch(torch.from_numpy(B.astype(np.float32)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for l1acc in (True, False):
            for w in (None, 1, 4, 8, 15, 16):
                cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                                       fp32_dest_acc_en=True, packer_l1_acc=l1acc)
                kw = {}
                if w is not None:
                    if (K // 32) % w:
                        continue
                    kw["program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(8, 8), in0_block_w=w, out_subblock_h=1, out_subblock_w=2,
                        out_block_h=1, out_block_w=2, per_core_M=M // 32 // 8, per_core_N=N // 32 // 8,
                        transpose_mcast=False, fused_activation=None)
                rec = dict(K=K, case=case, l1acc=l1acc, in0_block_w=w)
                try:
                    c = ttnn.to_torch(ttnn.matmul(ta, tb, compute_kernel_config=cfg, dtype=ttnn.float32, **kw)).double().numpy()
                    rec.update(exact=bool((c == ex).all()), max_abs_err=float(np.abs(c - ex).max()), max_val=float(np.abs(ex).max()))
                except Exception as e:  # noqa: BLE001
                    rec["error"] = str(e).splitlines()[0][:200]
                print(json.dumps(rec), flush=True)
