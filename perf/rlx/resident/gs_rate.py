"""ttnn.grid_sample (bilinear, NHWC) on BH: throughput and precision vs float64, as a projection building block.

Trilinear projection = two bilinear samples on z-stacked slices + one lerp, or (psi route) a bilinear 2-D rotation of a
direction's projection with C = 2 x directions channels per stick. Throughput depends on the stick size C, so this sweeps
C. Precision: max relative error vs torch float64 grid_sample on the same fp32 input and grid (align_corners=False).
  TT_VISIBLE_DEVICES=16 python gs_rate.py
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from _tt import Clock, open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, nodes = open_dev()
g = torch.Generator().manual_seed(0)
for H, W, C, Ho, Wo in ((360, 360, 32, 256, 256), (360, 360, 64, 256, 256), (720, 361, 32, 512, 512),
                        (200, 200, 256, 128, 128), (100, 100, 1024, 64, 64)):
    x = torch.randn(1, H, W, C, generator=g, dtype=torch.float64)
    grid = (torch.rand(1, Ho, Wo, 2, generator=g, dtype=torch.float64) * 1.9 - 0.95)
    ref = torch.nn.functional.grid_sample(x.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros",
                                          align_corners=False).permute(0, 2, 3, 1)
    for in_dt_name, in_dt in (("f32", ttnn.float32), ("bf16", ttnn.bfloat16)):
        rec = dict(H=H, W=W, C=C, Ho=Ho, Wo=Wo, input=in_dt_name)
        try:
            xi = ttnn.from_torch(x.float(), dtype=in_dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            gi = ttnn.from_torch(grid.float(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            f = lambda: ttnn.grid_sample(xi, gi, mode="bilinear", padding_mode="zeros", align_corners=False)  # noqa: E731
            y = f(); ttnn.synchronize_device(dev)
            yt = ttnn.to_torch(y).double().reshape(ref.shape)
            ttnn.deallocate(y)
            # error relative to the input fp32 rounding: compare with float64 sampling of the fp32-rounded input
            ref32 = torch.nn.functional.grid_sample(x.float().double().permute(0, 3, 1, 2), grid.float().double(), mode="bilinear",
                                                    padding_mode="zeros", align_corners=False).permute(0, 2, 3, 1)
            rec["rel_err"] = float((yt - ref32).abs().max() / ref32.abs().max())
            clk = Clock(nodes); t0 = time.perf_counter()
            for _ in range(5):
                ttnn.deallocate(f())
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / 5
            lo, med = clk.stop()
            rec.update(s=round(dt, 5), samples_per_s_M=round(Ho * Wo / dt / 1e6, 1),
                       chan_samples_per_s_M=round(Ho * Wo * C / dt / 1e6, 1), aiclk_min=lo, aiclk_med=med)
            ttnn.deallocate(xi); ttnn.deallocate(gi)
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)[:300]
        print(json.dumps(rec), flush=True)
