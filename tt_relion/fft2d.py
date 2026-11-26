"""A fused 2D complex FFT for Tenstorrent Blackhole.

The transform is computed as Y = F . X . F with F the dense DFT matrix, blocked into 32x32 tiles.
The image and F both stay in a core's L1 for the whole transform, so DRAM sees one read and one
write per image instead of the four a two-pass GPU FFT needs. Each core transforms its own images;
nothing is exchanged between cores.

F is built in float64 on the host and cast once, so nothing trigonometric runs on the device.
Because F is dense and O(N^2), it only fits in L1 at box 256 in bf16 today.

    import ttnn
    from tt_relion.fft2d import FFT2D

    dev = ttnn.open_device(device_id=0)
    fft = FFT2D(dev, box=256)
    spectra = fft(images)          # complex [n, 256, 256] in, complex128 out, = np.fft.fft2(images)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

KERNELS = Path(__file__).resolve().parent / "kernels" / "fft2d"
CB_F, CB_X, CB_T, CB_O = 0, 1, 2, 16
TILE = 32
SUPPORTED = {(256, "bf16")}   # what fits in L1 today; other shapes are rejected, not slowed down


def dft_blocks(n: int, dtype=torch.bfloat16) -> torch.Tensor:
    """F, F.imag and -F.imag as an (n/32)^2 grid of 32x32 tiles, interleaved per block.

    The compute kernel can only accumulate (DST += A.B), so -Fi is shipped as a third block set and
    the real part Xr.Fr - Xi.Fi becomes Xr.Fr + Xi.(-Fi): every term is an accumulate.
    """
    k = np.arange(n)
    f = np.exp(-2j * np.pi * np.outer(k, k) / n)
    nt = n // TILE
    b = f.reshape(nt, TILE, nt, TILE).transpose(0, 2, 1, 3).reshape(nt * nt, TILE, TILE)
    out = np.stack([b.real, b.imag, -b.imag], axis=1).reshape(-1, TILE, TILE)
    return torch.from_numpy(np.ascontiguousarray(out)).to(dtype)


def to_tiles(img: np.ndarray, dtype=torch.bfloat16) -> torch.Tensor:
    """Complex [B, N, N] -> tile pages, real/imag interleaved, row-major over (i, j) tile blocks."""
    bsz, n, _ = img.shape
    nt = n // TILE
    b = img.reshape(bsz, nt, TILE, nt, TILE).transpose(0, 1, 3, 2, 4).reshape(bsz, nt * nt, TILE, TILE)
    out = np.stack([b.real, b.imag], axis=2).reshape(-1, TILE, TILE)
    return torch.from_numpy(np.ascontiguousarray(out)).to(dtype)


def from_tiles(t: torch.Tensor, bsz: int, n: int) -> np.ndarray:
    """Inverse of to_tiles, returning complex128 [B, N, N]."""
    nt = n // TILE
    a = t.to(torch.float64).numpy().reshape(bsz, nt * nt, 2, TILE, TILE)
    c = a[:, :, 0] + 1j * a[:, :, 1]
    return c.reshape(bsz, nt, nt, TILE, TILE).transpose(0, 1, 3, 2, 4).reshape(bsz, n, n)


def _pages(t: torch.Tensor) -> torch.Tensor:
    """[P, 32, 32] -> the [1, 1, 32P, 32] page layout a TensorAccessor indexes."""
    return t.reshape(1, 1, -1, TILE)


class FFT2D:
    """Forward 2D FFT of a batch of complex images, one program launch per `images_per_core * cores`.

    Arguments are the tuning that produced the published number: `x_depth` two images deep lets the
    reader run one image ahead so the DRAM read hides behind the matmuls; `chunk` is how many
    output tiles the writer drains behind one barrier (must divide the tiles of one image).
    """

    def __init__(self, device, box: int = 256, dtype: str = "bf16", images_per_core: int = 16,
                 x_depth: int = 2, out_depth: int = 32, chunk: int = 16, fidelity: str = "HiFi2"):
        import ttnn

        if (box, dtype) not in SUPPORTED:
            raise ValueError(f"box {box} {dtype} is not supported; supported: {sorted(SUPPORTED)}")
        self.ttnn, self.dev, self.n, self.nimg = ttnn, device, box, images_per_core
        self.tdt, self.tor, self.nbytes = {"bf16": (ttnn.bfloat16, torch.bfloat16, 2)}[dtype]
        self.x_depth, self.out_depth, self.chunk = x_depth, out_depth, chunk
        self.fidelity = getattr(ttnn.MathFidelity, fidelity)
        g = device.compute_with_storage_grid_size()
        self.grid = g
        self.cores = g.x * g.y
        self.batch = self.cores * images_per_core
        self.f = ttnn.from_torch(_pages(dft_blocks(box, self.tor)), dtype=self.tdt,
                                 layout=ttnn.TILE_LAYOUT, device=device)

    def _program(self, x, o):
        ttnn, g = self.ttnn, self.grid
        nt = self.n // TILE
        tb = TILE * TILE * self.nbytes
        nf, ntile = 3 * nt * nt, 2 * nt * nt
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

        def cb(i, depth):
            fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=self.tdt, page_size=tb)
            return ttnn.CBDescriptor(total_size=depth * tb, core_ranges=cores, format_descriptors=[fmt])

        rct = [CB_F, CB_X, tb, nf, ntile, self.nimg]
        rct += list(ttnn.TensorAccessorArgs(self.f).get_compile_time_args())
        rct += list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        wct = [CB_O, tb, ntile, self.nimg, self.chunk] + list(ttnn.TensorAccessorArgs(o).get_compile_time_args())

        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        c = 0
        for cy in range(g.y):
            for cx in range(g.x):
                rrt[cx][cy] = [self.f.buffer_address(), x.buffer_address(), c * self.nimg * ntile]
                crt[cx][cy] = [0]
                wrt[cx][cy] = [o.buffer_address(), c * self.nimg * ntile]
                c += 1

        def kernel(src, ct, rt, cfg):
            return ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)

        return ttnn.ProgramDescriptor(kernels=[
            kernel("reader_fft2d.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
            kernel("writer_fft2d.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
            kernel("compute_fft2d.cpp", [CB_F, CB_X, CB_T, CB_O, nt, self.nimg], crt,
                   ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True)),
        ], semaphores=[], cbs=[cb(CB_F, nf), cb(CB_X, self.x_depth * ntile), cb(CB_T, ntile),
                               cb(CB_O, self.out_depth)])

    def upload(self, images: np.ndarray):
        """Stage one full batch (exactly `self.batch` images) on device; returns (x, out, program)."""
        ttnn = self.ttnn
        assert images.shape == (self.batch, self.n, self.n), images.shape
        x = ttnn.from_torch(_pages(to_tiles(images, self.tor)), dtype=self.tdt,
                            layout=ttnn.TILE_LAYOUT, device=self.dev)
        o = ttnn.from_torch(_pages(torch.zeros(self.batch * 2 * (self.n // TILE) ** 2, TILE, TILE,
                                               dtype=self.tor)),
                            dtype=self.tdt, layout=ttnn.TILE_LAYOUT, device=self.dev)
        return x, o, self._program(x, o)

    def run(self, staged) -> None:
        x, o, prog = staged
        self.ttnn.generic_op([self.f, x, o], prog)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """np.fft.fft2 over the last two axes of a complex [B, N, N] batch, any B."""
        images = np.asarray(images)
        bsz = images.shape[0]
        out = np.empty((bsz, self.n, self.n), dtype=np.complex128)
        for s in range(0, bsz, self.batch):
            part = images[s:s + self.batch]
            pad = np.zeros((self.batch, self.n, self.n), dtype=np.complex128)
            pad[:len(part)] = part
            staged = self.upload(pad)
            self.run(staged)
            got = self.ttnn.to_torch(staged[1]).reshape(-1, TILE, TILE)
            out[s:s + len(part)] = from_tiles(got, self.batch, self.n)[:len(part)]
        return out

    def benchmark(self, reps: int = 5, seed: int = 0) -> dict:
        """Device time for one full batch, data already resident (the published number's protocol)."""
        rng = np.random.default_rng(seed)
        img = (rng.standard_normal((self.batch, self.n, self.n))
               + 1j * rng.standard_normal((self.batch, self.n, self.n))) / self.n
        staged = self.upload(img)
        self.run(staged)
        self.ttnn.synchronize_device(self.dev)
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            self.run(staged)
            self.ttnn.synchronize_device(self.dev)
            best = min(best, time.perf_counter() - t0)
        got = from_tiles(self.ttnn.to_torch(staged[1]).reshape(-1, TILE, TILE), self.batch, self.n)
        ref = np.fft.fft2(img, axes=(-2, -1))
        return {"box": self.n, "images": self.batch, "seconds": best, "images_per_s": self.batch / best,
                "rel_l2": float(np.linalg.norm(got - ref) / np.linalg.norm(ref))}
