"""Is ttnn.add / ttnn.multiply on fp32 tensors exact (true fp32) on Blackhole? Integer and random operands.
    TT_VISIBLE_DEVICES=<chip> python3 probe_add.py
"""
import numpy as np
import torch
import ttnn

from _tt import open_dev

dev, _ = open_dev()
g = np.random.default_rng(5)
up = lambda x: ttnn.from_torch(torch.from_numpy(x), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
down = lambda x: ttnn.to_torch(x).double().numpy()
for shape in ((9, 1024, 32), (1, 256, 96)):
    a = g.integers(-2 ** 20, 2 ** 20, shape).astype(np.float32)
    b = g.integers(-2 ** 20, 2 ** 20, shape).astype(np.float32)
    print(shape, "int add err", np.abs(down(ttnn.add(up(a), up(b))) - (a.astype(np.float64) + b)).max())
    x = g.standard_normal(shape).astype(np.float32)
    y = g.standard_normal(shape).astype(np.float32)
    ref = (x + y).astype(np.float32).astype(np.float64)   # fp32 RNE
    print(shape, "rand add max ulp-ish", np.abs(down(ttnn.add(up(x), up(y))) - ref).max(),
          "sub", np.abs(down(ttnn.subtract(up(x), up(y))) - (x - y).astype(np.float32)).max(),
          "mul", np.abs(down(ttnn.multiply(up(x), up(y))) - (x * y).astype(np.float32)).max(),
          "mul scalar 2^-3", np.abs(down(ttnn.multiply(up(x), 0.125)) - x / 8).max())
# matmul of digits with K runs of 480, |d| <= 2
A = g.integers(-2, 3, (9, 1024, 480)).astype(np.float32)
B = g.integers(-2, 3, (9, 480, 21)).astype(np.float32)
cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
ta = ttnn.typecast(up(A), ttnn.bfloat16); tb = ttnn.typecast(up(B), ttnn.bfloat16)
C = ttnn.matmul(ta, tb, compute_kernel_config=cfg, dtype=ttnn.float32)
print("digit matmul err", np.abs(down(C) - A.astype(np.float64) @ B).max())
ttnn.close_device(dev)
