"""Achieved matmul throughput of the bridge's Ozaki pair matmuls (D32 item 2).

Times only the device matmuls of DeviceServer.oz_mm: for lhs digit i, [nc, Op, 480] x [nc, 480, (n - i) * Tp] bf16 LoFi
into fp32, i = n-1 .. 0, digit tensors already resident (random integers in [-2, 2], as the cut produces). FLOP counted on
the padded shapes the chip computes (2 * nc * Op * 480 * Tp * n(n+1)/2) and on the useful ones (O, T). A plain square
bf16 LoFi matmul is timed beside it as the ttnn ceiling on this chip. Peak = 4096 FLOP/cycle/core (LoFi, BH and WH
spec: p150 774 TFLOP/s = 140 cores x 1.35 GHz x 4096) x cores in the compute grid x AICLK sampled during.

  TT_VISIBLE_DEVICES=16 python oz_tflops.py --shape 1024,22,296 --shape 256,272,84 --ns 14,10,8
shape = O,nc,T (nc = K / 480 runs).
"""
import argparse
import json
import time

import numpy as np

from _tt import Clock, open_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", required=True)
    ap.add_argument("--ns", default="14,10,8")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    import ttnn

    dev, nodes = open_dev()
    g = dev.compute_with_storage_grid_size()
    cores = g.x * g.y
    lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=True)
    rng = np.random.default_rng(0)

    def put(x):
        return ttnn.from_torch(__import__("torch").from_numpy(x), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def timed(f):
        f()
        ttnn.synchronize_device(dev)
        clk = Clock(nodes)
        t0 = time.perf_counter()
        for _ in range(a.reps):
            f()
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / a.reps
        return dt, clk.stop()

    def rec(**kw):
        mhz = kw["aiclk_med"] or 0
        kw["peak_tflops"] = round(4096 * cores * mhz * 1e6 / 1e12, 1)
        print(json.dumps(kw), flush=True)

    # ttnn ceiling: square bf16 LoFi matmul
    m = 4096
    x, y = put(rng.integers(-2, 3, (m, m)).astype(np.float32)), put(rng.integers(-2, 3, (m, m)).astype(np.float32))
    dt, (lo, med) = timed(lambda: ttnn.deallocate(ttnn.matmul(x, y, compute_kernel_config=lofi, dtype=ttnn.float32)))
    rec(kind="square", m=m, s=dt, tflops=round(2 * m ** 3 / dt / 1e12, 2), aiclk_min=lo, aiclk_med=med, cores=cores)
    for z in (x, y):
        ttnn.deallocate(z)

    for sh in a.shape:
        o, nc, t = map(int, sh.split(","))
        op, tp = -(-o // 32) * 32, -(-t // 32) * 32
        for n in map(int, a.ns.split(",")):
            A = [put(rng.integers(-2, 3, (nc, op, 480)).astype(np.float32)) for _ in range(n)]
            B = put(rng.integers(-2, 3, (nc, 480, n * tp)).astype(np.float32))
            bs = [B] + [ttnn.slice(B, [0, 0, 0], [nc, 480, (n - i) * tp]) for i in range(1, n)]

            def chain():
                for i in range(n - 1, -1, -1):
                    ttnn.deallocate(ttnn.matmul(A[i], bs[i], compute_kernel_config=lofi, dtype=ttnn.float32))

            dt, (lo, med) = timed(chain)
            pairs = n * (n + 1) // 2
            fp, fu = 2 * nc * op * 480 * tp * pairs, 2 * nc * o * 480 * t * pairs
            rec(kind="ozaki", O=o, nc=nc, T=t, n=n, s=dt, tflops=round(fp / dt / 1e12, 2),
                useful_tflops=round(fu / dt / 1e12, 2), aiclk_min=lo, aiclk_med=med, cores=cores)
            for z in A + bs:
                ttnn.deallocate(z)


if __name__ == "__main__":
    main()
