"""Final-pass projections from a shared fp32 table stored as THREE bf16 pieces (x = hi + mid + lo exactly: each
round-to-nearest bf16 step takes >= 8 more significant bits, 3 x 8 >= fp32's 24), gathered with ttnn.embedding
(bf16-only) in K chunks <= 8160 columns, reassembled on chip as hi + (mid + lo) in fp32. Checks bit-exactness vs the
fp32 table rows and times gather + reassembly per particle (own 160 rows) at the s352 K.
  TT_VISIBLE_DEVICES=19 python table_split_bench.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from _tt import Clock, open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(5)
for (Otab, K, q, KCH) in ((512, 98304, 8, 6144), (512, 24576, 32, 6144)):
    rec = dict(O_tab=Otab, K=K, q=q, own=160, kchunk=KCH)
    try:
        X = (rng.standard_normal((Otab, K)) * np.exp(rng.uniform(-8, 8, (Otab, 1)))).astype(np.float32)
        xt = torch.from_numpy(X)
        hi = xt.bfloat16(); r1 = xt - hi.float(); mid = r1.bfloat16(); lo = (r1 - mid.float()).bfloat16()
        assert torch.equal(hi.float() + (mid.float() + lo.float()), xt)
        nck = K // KCH
        tabs = [[ttnn.from_torch(pc[:, c * KCH:(c + 1) * KCH].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=dev) for c in range(nck)] for pc in (hi, mid, lo)]
        own = rng.integers(0, Otab, (q, 160))
        idx = ttnn.from_torch(torch.from_numpy(own.reshape(1, -1).astype(np.int32)), dtype=ttnn.uint32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

        def gather():
            parts = []
            for c in range(nck):
                h, m, l_ = (ttnn.typecast(ttnn.embedding(idx, tb[c], layout=ttnn.TILE_LAYOUT), ttnn.float32) for tb in tabs)
                parts.append(ttnn.add(h, ttnn.add(m, l_)))
            return ttnn.concat(parts, dim=-1) if nck > 1 else parts[0]
        G = ttnn.to_torch(gather()).numpy().reshape(q * 160, K)
        rec["bit_exact"] = bool((G.view(np.uint32) == X[own.reshape(-1)].view(np.uint32)).all())
        ttnn.synchronize_device(dev)
        clk = Clock(nodes); t0 = time.perf_counter()
        for _ in range(3):
            ttnn.deallocate(gather())
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / 3
        amin, amed = clk.stop()
        rec.update(sec=round(dt, 4), ms_per_particle=round(dt / q * 1e3, 3), aiclk_min=amin, aiclk_med=amed,
                   load1=open("/proc/loadavg").read().split()[0])
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:300]
    print(json.dumps(rec), flush=True)
