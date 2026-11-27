"""q-batched oz_combine == per-batch q=1 calls (the q=1 kernel is bit-exact vs its fp32 emulation,
results/oz_combine_bh_chip16.jsonl), bit for bit, at the local-pass shape [q 13, 64, 160].
  TT_VISIBLE_DEVICES=16 python oz_combine_q_test.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from _tt import open_dev  # noqa: E402

import ttnn  # noqa: E402
from oz_fused import oz_combine  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(3)
q, n, R, O = 5, 13, 64, 160
S = rng.integers(-200000, 200000, (q, n, R, O)).astype(np.float32)
ea = rng.integers(-20, 20, (q, R)).astype(np.int32)
eb = rng.integers(-20, 20, (q, O)).astype(np.int32)
put = lambda x, dt, lay: ttnn.from_torch(torch.from_numpy(x), dtype=dt, layout=lay, device=dev)  # noqa: E731
hq, lq = oz_combine(put(S.reshape(q * n, R, O), ttnn.float32, ttnn.TILE_LAYOUT),
                    put(ea.reshape(q * R // 32, 32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    put(eb.reshape(q * O // 32, 32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), n, q=q)
hq, lq = ttnn.to_torch(hq).numpy().reshape(q, R, O), ttnn.to_torch(lq).numpy().reshape(q, R, O)
same = True
for j in range(q):
    h1, l1 = oz_combine(put(S[j], ttnn.float32, ttnn.TILE_LAYOUT),
                        put(ea[j].reshape(R // 32, 32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                        put(eb[j].reshape(O // 32, 32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), n)
    h1, l1 = ttnn.to_torch(h1).numpy(), ttnn.to_torch(l1).numpy()
    same &= bool((h1.view(np.uint32) == hq[j].view(np.uint32)).all() and (l1.view(np.uint32) == lq[j].view(np.uint32)).all())
ref = np.einsum("qkro,k->qro", S.astype(np.float64), 2.0 ** (-2 * np.arange(n) - 2.0)) * 2.0 ** (ea[:, :, None] + eb[:, None, :])
rel = float(np.abs(hq.astype(np.float64) + lq - ref).max() / np.abs(ref).max())
print(json.dumps(dict(q=q, n=n, R=R, O=O, bit_identical_to_q1=same, hi_lo_vs_f64_rel=rel)), flush=True)
