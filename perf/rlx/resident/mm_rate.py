"""BH matmul rate at the resident E-step's shapes (rlx-resident CEILING step).

Times ttnn.matmul on resident bf16 operands at the shapes a particle-batched diff2 cross term produces:
  [M = batch*T, K = 2*Pc] x [K, N = O] for the global passes, and batched [B, T, K] x [B, K, O_local] for local
searches (each particle has its own orientation set). Square 4096/8192 are the ttnn reference. Each shape runs with
the default program config and with an explicit full compute grid; fp32 and bf16 outputs; LoFi and HiFi2.
AICLK of the held node sampled during every timed loop. One JSON line per arm.
  TT_VISIBLE_DEVICES=16 python mm_rate.py
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from _tt import Clock, open_dev  # noqa: E402

SHAPES = [  # name, B, M, K, N
    ("sq4096", 1, 4096, 4096, 4096),
    ("sq8192", 1, 8192, 8192, 8192),
    ("gc_h2_b256", 1, 256 * 21, 576, 4608),          # coarse global, s26 (Pc 285), O 4608, T 21
    ("gf_h3_b256_n8k", 1, 256 * 13, 2048, 8192),     # coarse global finer, s50 (Pc 1021), T 13, O chunk 8192
    ("gf_h3_b256_n36k", 1, 256 * 13, 2048, 36864),   # same, all 36864 orientations in one call
    ("gf_h3_b1024_n8k", 1, 1024 * 13, 2048, 8192),
    ("oz_run480", 1, 256 * 13, 480, 8192),           # Ozaki K-run shape at the same M, N
    ("loc_b256", 256, 32, 8032, 160),                # local: own 155 orientations per particle, s100 (Pc 4003), T 25->32
    ("fin_b32", 32, 32, 102400, 160),                # final: s360 coarse (Pc 51177), T 9->32, O 155->160
]


def main():
    import ttnn

    reps = 5
    dev, nodes = open_dev()
    g = dev.compute_with_storage_grid_size()
    cores = g.x * g.y
    rng = np.random.default_rng(0)
    cfgs = {
        "lofi": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True),
        "hifi2": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True),
    }

    def put(shape):
        x = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    for name, b, m, k, n in SHAPES:
        sa = (b, m, k) if b > 1 else (m, k)
        sb = (b, k, n) if b > 1 else (k, n)
        try:
            x, y = put(sa), put(sb)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"shape": name, "error": str(e)[:200]}), flush=True)
            continue
        for cname, cfg in cfgs.items():
            for odt_name, odt in (("f32", ttnn.float32), ("bf16", ttnn.bfloat16)):
                for grid in ("auto", "full"):
                    kw = dict(compute_kernel_config=cfg, dtype=odt)
                    if grid == "full":
                        kw["core_grid"] = ttnn.CoreGrid(y=g.y, x=g.x)

                    def f():
                        ttnn.deallocate(ttnn.matmul(x, y, **kw))
                    try:
                        f()
                        ttnn.synchronize_device(dev)
                        clk = Clock(nodes)
                        t0 = time.perf_counter()
                        for _ in range(reps):
                            f()
                        ttnn.synchronize_device(dev)
                        dt = (time.perf_counter() - t0) / reps
                        lo, med = clk.stop()
                    except Exception as e:  # noqa: BLE001
                        print(json.dumps({"shape": name, "cfg": cname, "out": odt_name, "grid": grid,
                                          "error": str(e)[:200]}), flush=True)
                        continue
                    fl = 2.0 * b * m * k * n
                    peak = 4096 * cores * (med or 0) * 1e6 / 1e12
                    print(json.dumps({"shape": name, "B": b, "M": m, "K": k, "N": n, "cfg": cname, "out": odt_name,
                                      "grid": grid, "s": round(dt, 5), "tflops": round(fl / dt / 1e12, 2),
                                      "lofi_peak": round(peak, 1), "aiclk_min": lo, "aiclk_med": med,
                                      "cores": cores}), flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(y)


if __name__ == "__main__":
    main()
