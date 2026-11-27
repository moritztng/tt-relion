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


def oz_fused(A, B, n, rb=RB, nb=NB, bt=False, q=1):
    """bt: B is [n, O, K] (row-major digits of the reference, no transpose needed); the unpacker transposes each tile.
    q: independent batches in one program (local passes: one small product per pooled orientation set). A is
    [q*n, R, K], B [q*n, K, O] (or [q*n, O, K]), S [q*n, R, O]; batch j owns digit planes j*n .. j*n+n-1."""
    dev = A.device()
    R, K = int(A.shape[-2]), int(A.shape[-1])
    O = int(B.shape[-2]) if bt else int(B.shape[-1])
    assert R % (32 * rb) == 0 and O % (32 * nb) == 0 and K % (32 * KR) == 0 and int(B.shape[-1 if bt else -2]) == K
    RT, OT, KT = R // 32, O // 32, K // 32
    NRB, NCB, NRUN = RT // rb, OT // nb, KT // KR
    assert int(A.shape[0]) == q * n and int(B.shape[0]) == q * n
    S = ttnn.allocate_tensor_on_device(ttnn.Shape([q * n, R, O]), ttnn.float32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    units = q * n * NRB * NCB
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
                                   compile_time_args=acc(A) + acc(B) + [rb, nb, KR, NRUN, RT, OT, NRB, NCB, int(bt), n],
                                   runtime_args=rt, common_runtime_args=[A.buffer_address(), B.buffer_address()],
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(KDIR / "writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=acc(S) + [rb, nb, RT, OT, NRB, NCB, n], runtime_args=rt,
                                   common_runtime_args=[S.buffer_address()], config=ttnn.WriterConfigDescriptor())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[24] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(kernel_source=str(KDIR / "compute.cpp"), source_type=path, core_ranges=grid,
                                    compile_time_args=[rb, nb, KR, NRUN, NRB, NCB, int(bt), n], runtime_args=rt, config=cfg)
    ttnn.generic_op([A, B, S], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs))
    return S


CDIR = Path(__file__).resolve().parent / "kernels" / "oz_combine"


def oz_combine(S, ea, eb, n, q=1):
    """S fp32 TILE [n, R, O] from oz_fused; ea int32 RM [R/32, 32] row exponents, eb int32 RM [O/32, 32] column
    exponents. q batches (oz_fused q-batched): S [q n, R, O], ea [q R/32, 32], eb [q O/32, 32], out [q R, O].
    Returns (hi, lo) fp32 TILE [R, O], the df32 of sum_k S_k 2^-(2k+2) 2^(ea[r] + eb[c]): Horner
    (H <- 4 H + S_k, exact x4) with TwoSum on the SFPU in one pass over S, then one exact 2^(ea + eb - 2n) scale
    (the ttnn chain cost ~6 [R, O] DRAM passes per digit). Needs ea + eb - 2n > -126."""
    dev = S.device()
    R, O = int(S.shape[-2]), int(S.shape[-1])
    RT, OT = R // 32, O // 32
    assert int(S.shape[0]) == q * n and int(ea.shape[0]) == q * RT and int(eb.shape[0]) == q * OT
    hi = ttnn.allocate_tensor_on_device(ttnn.Shape([q * R, O]), ttnn.float32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    lo = ttnn.allocate_tensor_on_device(ttnn.Shape([q * R, O]), ttnn.float32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    units = q * RT * OT
    g = dev.compute_with_storage_grid_size()
    _, grid, cg1, cg2, w1, w2 = core_split.split_work_to_cores(g, units, 0)

    def cb(i, dt, page, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(0, ttnn.float32, 4096, 2 * n), cb(2, ttnn.float32, 4096, 2), cb(3, ttnn.int32, 128, 2),
           cb(16, ttnn.float32, 4096, 2), cb(17, ttnn.float32, 4096, 2)]
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
    reader = ttnn.KernelDescriptor(kernel_source=str(CDIR / "reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=acc(S) + acc(ea) + acc(eb) + [n, RT, OT], runtime_args=rt,
                                   common_runtime_args=[S.buffer_address(), ea.buffer_address(), eb.buffer_address()],
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(CDIR / "writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=acc(hi) + acc(lo), runtime_args=rt,
                                   common_runtime_args=[hi.buffer_address(), lo.buffer_address()],
                                   config=ttnn.WriterConfigDescriptor())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (0, 2):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(kernel_source=str(CDIR / "compute.cpp"), source_type=path, core_ranges=grid,
                                    compile_time_args=[n], runtime_args=rt, config=cfg)
    ttnn.generic_op([S, ea, eb, hi, lo], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs))
    return hi, lo


XDIR = Path(__file__).resolve().parent / "kernels" / "oz_cut"


def oz_cut(X, ea, n, grp=2):
    """X fp32 TILE [R, K] (padded), ea int32 RM [R/32, 32] row exponents (|X[r]| <= 2^ea[r]). Returns bf16 TILE
    [n, R, K] base-4 digits in [-2, 2] with X 2^-ea = sum_i d_i 2^-(2i+1) + residual (< 2^-(2n+1)), in one pass
    (the ttnn chain was ~4 [R, K] DRAM passes per digit)."""
    dev = X.device()
    R, K = int(X.shape[-2]), int(X.shape[-1])
    RT, KT = R // 32, K // 32
    D = ttnn.allocate_tensor_on_device(ttnn.Shape([n, R, K]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    units = RT * KT
    g = dev.compute_with_storage_grid_size()
    _, grid, cg1, cg2, w1, w2 = core_split.split_work_to_cores(g, units, 0)

    def cb(i, dt, page, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(0, ttnn.float32, 4096, 2), cb(1, ttnn.float32, 4096, 2), cb(2, ttnn.int32, 128, 1),
           cb(16, ttnn.bfloat16, 2048, 12)]
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
    reader = ttnn.KernelDescriptor(kernel_source=str(XDIR / "reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=acc(X) + acc(ea) + [KT], runtime_args=rt,
                                   common_runtime_args=[X.buffer_address(), ea.buffer_address()],
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(XDIR / "writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=acc(D) + [n, units], runtime_args=rt,
                                   common_runtime_args=[D.buffer_address()], config=ttnn.WriterConfigDescriptor())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (0, 1):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(kernel_source=str(XDIR / "compute.cpp"), source_type=path, core_ranges=grid,
                                    compile_time_args=[n, grp], runtime_args=rt, config=cfg)
    ttnn.generic_op([X, ea, D], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs))
    return D


KCDIR = Path(__file__).resolve().parent / "kernels" / "cand"


def cand_compact(D, thr, rows, cols, cap=4096):
    assert cap % 256 == 0  # flush offsets stay 2 KiB aligned and the last rounded write stays inside the row
    """Candidate compaction of d2 on chip. D fp32 TILE [R, O] (R, O multiples of 32), thr fp32 RM [R/32, 32] per-row
    thresholds. Every element D[r, c] < thr[r] with r < rows, c < cols is emitted as (r * cols + c, value) exactly.
    Returns (idx int64, val float32), or None when a core overflowed cap (caller reads D back in full)."""
    import numpy as np
    dev = D.device()
    R, O = int(D.padded_shape[-2]), int(D.padded_shape[-1])
    RT, OT = R // 32, O // 32
    units = RT * OT
    g = dev.compute_with_storage_grid_size()
    ncores, grid, cg1, cg2, w1, w2 = core_split.split_work_to_cores(g, units, 0)
    OUT = ttnn.allocate_tensor_on_device(ttnn.Shape([ncores, 2 * cap]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)
    CNT = ttnn.allocate_tensor_on_device(ttnn.Shape([ncores, 16]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)

    def cb(i, dt, page):
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)
        return ttnn.CBDescriptor(total_size=page, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(0, ttnn.float32, 4096), cb(1, ttnn.float32, 128), cb(2, ttnn.uint32, 2048), cb(3, ttnn.uint32, 64)]
    rt = ttnn.RuntimeArgs()
    first = core = 0
    for group, per in ((cg1, w1), (cg2, w2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rt[cx][cy] = [first, per, core]
                    first += per
                    core += 1
    assert first == units and core == ncores
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    k = ttnn.KernelDescriptor(kernel_source=str(KCDIR / "reader.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=grid, compile_time_args=acc(D) + acc(thr) + acc(OUT) + acc(CNT) + [OT, cap, cols, rows],
                              runtime_args=rt, common_runtime_args=[D.buffer_address(), thr.buffer_address(),
                                                                    OUT.buffer_address(), CNT.buffer_address()],
                              config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op([D, thr, OUT, CNT], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs))
    cnt = ttnn.to_torch(CNT).numpy().astype(np.int64)[:, 0]
    if cnt.max() > cap:
        return None
    o = np.ascontiguousarray(ttnn.to_torch(OUT).numpy()).astype(np.int64) & 0xffffffff
    idx = np.concatenate([o[j, 0:2 * cnt[j]:2] for j in range(ncores)])
    val = np.concatenate([o[j, 1:2 * cnt[j]:2] for j in range(ncores)]).astype(np.uint32).view(np.float32)
    return idx, val
