"""Which small blockings of oz_fused are exact (q-batched, bt)."""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
from _tt import open_dev  # noqa
import ttnn  # noqa
from oz_fused import oz_fused  # noqa
dev, nodes = open_dev()
rng = np.random.default_rng(1)
up = lambda a: ttnn.from_torch(torch.from_numpy(a.astype(np.float32)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa
for (q, n, R, K, O, rb, nb) in ((1, 1, 64, 480, 160, 2, 5), (1, 2, 64, 480, 160, 2, 5), (1, 1, 512, 480, 256, 16, 8),
                                (1, 1, 64, 480, 256, 2, 8), (1, 1, 512, 480, 160, 16, 5), (2, 1, 64, 480, 160, 2, 5),
                                (1, 1, 64, 2400, 160, 2, 5), (1, 1, 128, 480, 160, 4, 5), (1, 1, 64, 480, 128, 2, 4)):
    A = rng.integers(-2, 3, (q, n, R, K)).astype(np.float32); B = rng.integers(-2, 3, (q, n, O, K)).astype(np.float32)
    S = ttnn.to_torch(oz_fused(up(A.reshape(q * n, R, K)), up(B.reshape(q * n, O, K)), n, rb=rb, nb=nb, bt=True, q=q)).double().numpy().reshape(q, n, R, O)
    ex = np.stack([np.stack([sum(A[j, i].astype(np.float64) @ B[j, k - i].T for i in range(k + 1)) for k in range(n)]) for j in range(q)])
    bad = np.argwhere(S != ex)
    print(json.dumps(dict(q=q, n=n, R=R, K=K, O=O, rb=rb, nb=nb, nbad=len(bad), first=bad[:3].tolist(),
                          rows_bad=sorted(set((bad[:, 2] // 32).tolist()))[:8], cols_bad=sorted(set((bad[:, 3] // 32).tolist()))[:8])), flush=True)
