"""oz_fused adm (A digit-major [n q, R, K]) == batch-major A, bit for bit (q-batched, bt, local shape).
  TT_VISIBLE_DEVICES=17 python oz_adm_test.py
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
from tt_relion.oz_fused import oz_fused  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(4)
up = lambda a: ttnn.from_torch(torch.from_numpy(a.astype(np.float32)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
for (q, n, R, K, O, rb, nb) in ((3, 13, 64, 960, 160, 2, 5), (2, 4, 512, 480, 256, 16, 8)):
    A = rng.integers(-2, 3, (q, n, R, K)).astype(np.float32)
    B = rng.integers(-2, 3, (q, n, O, K)).astype(np.float32)
    S0 = ttnn.to_torch(oz_fused(up(A.reshape(q * n, R, K)), up(B.reshape(q * n, O, K)), n, rb=rb, nb=nb, bt=True, q=q)).numpy()
    S1 = ttnn.to_torch(oz_fused(up(A.transpose(1, 0, 2, 3).reshape(n * q, R, K)), up(B.reshape(q * n, O, K)), n, rb=rb,
                                nb=nb, bt=True, q=q, adm=True)).numpy()
    ex = np.stack([np.stack([sum(A[j, i].astype(np.float64) @ B[j, k - i].T for i in range(k + 1)) for k in range(n)])
                   for j in range(q)]).reshape(q * n, R, O)
    S2 = ttnn.to_torch(oz_fused(up(A.transpose(1, 0, 2, 3).reshape(n * q, R, K)), up(B.transpose(1, 0, 2, 3).reshape(n * q, O, K)),
                                n, rb=rb, nb=nb, bt=True, q=q, adm=True, bdm=True)).numpy()
    print(json.dumps(dict(q=q, n=n, R=R, K=K, O=O, rb=rb, nb=nb, adm_bit_identical=bool((S0 == S1).all()),
                          exact=bool((S1 == ex).all()), adm_bdm_bit_identical=bool((S0 == S2).all()))), flush=True)
