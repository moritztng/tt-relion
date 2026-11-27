"""Are ttnn.grid_sample's bilinear weights fp32? Input values exactly representable in bf16, so any error vs float64
is the weights / accumulation, not the input rounding. Also the 3-piece split: fp32 x = hi + mid + lo (each bf16-exact)
as 3 channel groups, summed after sampling; if the weights are fp32 this reaches fp32-class error.
  TT_VISIBLE_DEVICES=16 python gs_weights.py
"""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from _tt import open_dev  # noqa: E402

import ttnn  # noqa: E402

dev, nodes = open_dev()
g = torch.Generator().manual_seed(1)
H = W = 360; C = 32; Ho = Wo = 256
x = torch.randn(1, H, W, C, generator=g, dtype=torch.float64)
grid = (torch.rand(1, Ho, Wo, 2, generator=g, dtype=torch.float64) * 1.9 - 0.95).float()


def gs_ref(v):
    return torch.nn.functional.grid_sample(v.permute(0, 3, 1, 2), grid.double(), mode="bilinear", padding_mode="zeros",
                                           align_corners=False).permute(0, 2, 3, 1)


def gs_tt(v):
    xi = ttnn.from_torch(v.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    gi = ttnn.from_torch(grid, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    y = ttnn.to_torch(ttnn.grid_sample(xi, gi, mode="bilinear", padding_mode="zeros", align_corners=False))
    return y.double().reshape(1, Ho, Wo, -1)


xb = x.to(torch.bfloat16).double()       # bf16-exact input
r = gs_ref(xb)
e = (gs_tt(xb) - r).abs().max() / r.abs().max()
print(json.dumps(dict(test="bf16-exact input", rel_err=float(e))), flush=True)
x32 = x.float().double()
hi = x32.to(torch.bfloat16).double(); mid = (x32 - hi).to(torch.bfloat16).double(); lo = (x32 - hi - mid).to(torch.bfloat16).double()
y = gs_tt(torch.cat([hi, mid, lo], -1))
ys = y[..., :C] + y[..., C:2 * C] + y[..., 2 * C:]
r32 = gs_ref(x32)
print(json.dumps(dict(test="fp32 as 3 bf16 pieces", rel_err=float((ys - r32).abs().max() / r32.abs().max()),
                      fp32_host_rel=float((gs_ref(x32).float().double() - r32).abs().max() / r32.abs().max()))), flush=True)
