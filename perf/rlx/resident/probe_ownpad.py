"""Where the image side of d2_own goes at full size, and whether the rows can be placed on the chip instead of
uploaded as zeros: q particles x keep rows x K fp32. Route H: host builds [q*32, K] with zeros, TILE upload (what
d2_own does now). Route D: ROW_MAJOR upload of the used rows [q*keep, K], reshape, pad to [q, 32, Kp] and tilize on
the chip. Equality of the two tensors (read back), seconds of each, load1."""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
sys.path.insert(0, __file__.rsplit("/", 4)[0])
from _tt import open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(0)
for q, keep, K, Kp in ((4, 9, 4096, 4320), (4, 9, 131072, 131520), (4, 1, 131072, 131520), (16, 9, 4096, 4320)):
    A = rng.standard_normal((q, keep, K)).astype(np.float32)
    rec = dict(q=q, keep=keep, K=K, load1=float(open("/proc/loadavg").read().split()[0]))
    try:
        def host():
            Z = np.zeros((q, 32, K), np.float32)
            Z[:, :keep] = A
            t = ttnn.from_torch(torch.from_numpy(Z.reshape(q * 32, K)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            return ttnn.pad(t, [(0, 0), (0, Kp - K)], 0.0)

        def chip():
            t = ttnn.from_torch(torch.from_numpy(A.reshape(q * keep, K)), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            t = ttnn.pad(ttnn.reshape(t, (q, keep, K)), [(0, 0), (0, 32 - keep), (0, Kp - K)], 0.0)
            return ttnn.to_layout(ttnn.reshape(t, (q * 32, Kp)), ttnn.TILE_LAYOUT)
        for name, f in (("host", host), ("chip", chip)):
            t = f()                         # first call: programs
            ttnn.synchronize_device(dev)
            ts = []
            for _ in range(3):
                ttnn.deallocate(t)
                t0 = time.perf_counter()
                t = f()
                ttnn.synchronize_device(dev)
                ts.append(round(time.perf_counter() - t0, 4))
            rec[name + "_s"] = ts
            rec[name] = ttnn.to_torch(t).numpy()
            ttnn.deallocate(t)
        rec["equal"] = bool(rec["host"].shape == rec["chip"].shape and (rec.pop("host") == rec.pop("chip")).all())
    except Exception as e:  # noqa: BLE001
        rec.pop("host", None), rec.pop("chip", None)
        rec["error"] = repr(e)[:500]
    print(json.dumps(rec), flush=True)
