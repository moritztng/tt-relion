"""Fused Ozaki digit-pair matmul: S_k = sum_{i<=k} A_i @ B_{k-i} for k < n, exact integers, one Tensix program.

A: bf16 TILE [n, R, K] digits in [-2, 2]; B: bf16 TILE [n, K, O]; S: fp32 TILE [n, R, O]. R a multiple of 32 RB, O of
32 NB, K of 32 KR (KR = 15 tiles = the 480-wide run whose FPU partials stay below 2^11, exact). The run partials are
added on the SFPU into an fp32 accumulator in L1, so nothing but S goes back to DRAM (ttnn.matmul + ttnn.add wrote and
re-read an fp32 [R, O] partial per run and pair: DRAM-bound at 8.4 TFLOP/s, results/global_pass_s50_prod_shape.jsonl).
"""
from __future__ import annotations

from pathlib import Path

import ttnn
from tt_bio import core_split

KDIR = Path(__file__).resolve().parent / "kernels" / "oz_fused"
RB, NB, KR = 16, 8, 15


def oz_fused(A, B, n, rb=RB, nb=NB, bt=False):
    """bt: B is [n, O, K] (row-major digits of the reference, no transpose needed); the unpacker transposes each tile."""
    dev = A.device()
    R, K = int(A.shape[-2]), int(A.shape[-1])
    O = int(B.shape[-2]) if bt else int(B.shape[-1])
    assert R % (32 * rb) == 0 and O % (32 * nb) == 0 and K % (32 * KR) == 0 and int(B.shape[-1 if bt else -2]) == K
    RT, OT, KT = R // 32, O // 32, K // 32
    NRB, NCB, NRUN = RT // rb, OT // nb, KT // KR
    S = ttnn.allocate_tensor_on_device(ttnn.Shape([n, R, O]), ttnn.float32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    units = n * NRB * NCB
    g = dev.compute_with_storage_grid_size()
    _, grid, cg1, cg2, w1, w2 = core_split.split_work_to_cores(g, units, 0)

    def cb(i, dt, page, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(0, ttnn.bfloat16, 2048, 2 * KR), cb(1, ttnn.bfloat16, 2048, 2 * KR * nb),
           cb(24, ttnn.float32, 4096, rb * nb), cb(25, ttnn.float32, 32, 1), cb(26, ttnn.float32, 32, 1),
           cb(16, ttnn.float32, 4096, 2)]
    rt = ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, w1), (cg2, w2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rt[cx][cy] = [first, per]
                    first += per
    assert first == units
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=str(KDIR / "reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=acc(A) + acc(B) + [rb, nb, KR, NRUN, RT, OT, NRB, NCB, int(bt)],
                                   runtime_args=rt, common_runtime_args=[A.buffer_address(), B.buffer_address()],
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(KDIR / "writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=acc(S) + [rb, nb, RT, OT, NRB, NCB], runtime_args=rt,
                                   common_runtime_args=[S.buffer_address()], config=ttnn.WriterConfigDescriptor())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[24] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(kernel_source=str(KDIR / "compute.cpp"), source_type=path, core_ranges=grid,
                                    compile_time_args=[rb, nb, KR, NRUN, NRB, NCB, int(bt)], runtime_args=rt, config=cfg)
    ttnn.generic_op([A, B, S], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs))
    return S
