"""proj_lerp parity (vs float64 and vs an fp32 IEEE emulation of RELION's lerp order) and rate on one BH chip.
  TT_VISIBLE_DEVICES=16 python proj_lerp_bench.py [units] [table rows]
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _tt import Clock, open_dev  # noqa: E402

import ttnn  # noqa: E402
from tt_relion.proj_lerp import proj_lerp  # noqa: E402

dev, nodes = open_dev()
rng = np.random.default_rng(0)
out = []
NBIG = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
for N, U in ((2_000_000, 64), (NBIG, int(sys.argv[1]) if len(sys.argv) > 1 else 4096)):
    tab = rng.standard_normal((N, 16)).astype(np.float32)
    idx = rng.integers(0, N, (U, 1024)).astype(np.int64)
    frc = rng.random((3 * U, 1024)).astype(np.float32)
    t_tab = ttnn.from_torch(torch.from_numpy(tab), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    t_idx = ttnn.from_torch(torch.from_numpy(idx.astype(np.int32)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    t_frc = ttnn.from_torch(torch.from_numpy(frc), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    r = ttnn.to_torch(proj_lerp(t_tab, t_idx, t_frc)).numpy().reshape(U, 2, 1024)
    b = tab[idx]                       # [U, 1024, 16]
    f = frc.reshape(U, 3, 1024)
    for dt in (np.float32, np.float64):
        v = b.astype(dt).reshape(U, 1024, 2, 2, 2, 2)    # dz, dy, dx, part
        fx, fy, fz = (f[:, i].astype(dt)[..., None] for i in range(3))
        x = v[:, :, :, :, 0] + fx[..., None, None] * (v[:, :, :, :, 1] - v[:, :, :, :, 0])   # [U,1024,dz,dy,p]
        y = x[:, :, :, 0] + fy[..., None] * (x[:, :, :, 1] - x[:, :, :, 0])
        z = y[:, :, 0] + fz * (y[:, :, 1] - y[:, :, 0])                                       # [U,1024,p]
        ref = z.transpose(0, 2, 1)
        if dt is np.float32:
            emu = ref
        else:
            f64 = ref
    rec = dict(N=N, U=U, px=U * 1024, bitexact_vs_fp32_emu=float(np.mean(r == emu)),
               max_abs_vs_f64=float(np.abs(r - f64).max()), max_abs_emu_vs_f64=float(np.abs(emu - f64).max()))
    ttnn.synchronize_device(dev)
    clk = Clock(nodes); t0 = time.perf_counter()
    for _ in range(5):
        ttnn.deallocate(proj_lerp(t_tab, t_idx, t_frc))
    ttnn.synchronize_device(dev)
    dt_ = (time.perf_counter() - t0) / 5
    lo, med = clk.stop()
    rec.update(s=round(dt_, 5), Mpx_per_s=round(U * 1024 / dt_ / 1e6, 1), aiclk_min=lo, aiclk_med=med,
               load1=open("/proc/loadavg").read().split()[0])
    print(json.dumps(rec), flush=True)
    out.append(rec)
    for t in (t_tab, t_idx, t_frc):
        ttnn.deallocate(t)
