"""Projection gather + trilinear lerp as one Tensix program (step 3a of docs/resident-estep.md).

table: fp32 ROW_MAJOR [Nvox, 16], row v = the 2x2x2 corner brick at base voxel v, corner c = 4 dz + 2 dy + dx, value
(re, im) at 2 c, 2 c + 1. idx: uint32 ROW_MAJOR [U, 1024] base voxel per pixel; frc: fp32 ROW_MAJOR [3 U, 1024] rows
3u..3u+2 = fx, fy, fz of unit u. Out: fp32 ROW_MAJOR [2 U, 1024], rows 2u, 2u + 1 = re, im. Each lerp a + f (b - a) is
fp32 on the SFPU in RELION's order (x, y, z), so the result is fp32-exact against an IEEE emulation of the same order.
"""
from __future__ import annotations

from pathlib import Path

import ttnn
from tt_bio import core_split

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "proj_lerp"
CB_V, CB_F, CB_IDX, CB_STAGE, CB_X, CB_Y, CB_OUT = 0, 1, 2, 3, 4, 5, 16
_PAGE = 4096


def proj_lerp(table, idx, frc):
    dev = idx.device()
    U = int(idx.shape[0])
    assert int(idx.shape[1]) == 1024 and int(table.shape[1]) == 16 and int(frc.shape[0]) == 3 * U
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([2 * U, 1024]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)
    g = dev.compute_with_storage_grid_size()
    _, grid, cg1, cg2, w1, w2 = core_split.split_work_to_cores(g, U, 0)

    def cb(i, dt, page, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=grid, format_descriptors=[fmt])

    f32 = ttnn.float32
    cbs = [cb(CB_V, f32, _PAGE, 32), cb(CB_F, f32, _PAGE, 6), cb(CB_IDX, ttnn.uint32, _PAGE, 1),
           cb(CB_STAGE, f32, 64 * 1024, 1), cb(CB_X, f32, _PAGE, 8), cb(CB_Y, f32, _PAGE, 4), cb(CB_OUT, f32, _PAGE, 4)]
    rt = ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, w1), (cg2, w2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rt[cx][cy] = [first, per]
                    first += per
    assert first == U
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    src = lambda n: str(KERNEL_DIR / n)  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=src("reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=acc(table) + acc(idx) + acc(frc), runtime_args=rt,
                                   common_runtime_args=[table.buffer_address(), idx.buffer_address(), frc.buffer_address()],
                                   config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=src("writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=acc(out), runtime_args=rt,
                                   common_runtime_args=[out.buffer_address()], config=ttnn.WriterConfigDescriptor())
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (CB_V, CB_F, CB_X, CB_Y):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(kernel_source=src("compute.cpp"), source_type=path, core_ranges=grid,
                                    compile_time_args=[], runtime_args=rt, config=cfg)
    ttnn.generic_op([table, idx, frc, out], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs))
    return out
