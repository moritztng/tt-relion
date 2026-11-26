"""FFT kernel tests against numpy.fft.

The host-side tests run anywhere. The device test needs ttnn and a Blackhole card:
    pytest tests/test_fft2d.py
"""
import numpy as np
import pytest
import torch

from tt_relion.fft2d import TILE, dft_blocks, from_tiles, to_tiles

N = 256


def images(b, n=N, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((b, n, n)) + 1j * rng.standard_normal((b, n, n))) / n


def test_tile_roundtrip():
    x = images(3)
    assert np.array_equal(from_tiles(to_tiles(x, torch.float64), 3, N), x)


def test_blocked_transform_is_fft2():
    """The exact tile-level arithmetic the compute kernel performs, in float64, equals np.fft.fft2."""
    nt = N // TILE
    f = dft_blocks(N, torch.float64).numpy().reshape(nt, nt, 3, TILE, TILE)
    fr, fi, fneg = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    x = images(1)
    xt = to_tiles(x, torch.float64).numpy().reshape(nt, nt, 2, TILE, TILE)
    xr, xi = xt[:, :, 0], xt[:, :, 1]
    # Pass 1: T = X . F, accumulate-only complex product (real part uses the -Fi block set).
    tr = np.einsum("ikab,kjbc->ijac", xr, fr) + np.einsum("ikab,kjbc->ijac", xi, fneg)
    ti = np.einsum("ikab,kjbc->ijac", xr, fi) + np.einsum("ikab,kjbc->ijac", xi, fr)
    # Pass 2: Y = F . T.
    yr = np.einsum("ikab,kjbc->ijac", fr, tr) + np.einsum("ikab,kjbc->ijac", fneg, ti)
    yi = np.einsum("ikab,kjbc->ijac", fr, ti) + np.einsum("ikab,kjbc->ijac", fi, tr)
    y = np.stack([yr, yi], axis=2).reshape(-1, TILE, TILE)
    got = from_tiles(torch.from_numpy(y), 1, N)
    np.testing.assert_allclose(got, np.fft.fft2(x), rtol=0, atol=1e-9)


def test_device_matches_numpy():
    ttnn = pytest.importorskip("ttnn")
    from tt_relion.fft2d import FFT2D
    try:
        dev = ttnn.open_device(device_id=0)
    except Exception as e:  # no card visible
        pytest.skip(f"no Tenstorrent device: {e}")
    try:
        fft = FFT2D(dev, box=N)
        x = images(fft.batch + 5, seed=1)         # not a multiple of the batch: exercises padding
        got = fft(x)
        ref = np.fft.fft2(x)
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        assert rel < 1e-2, rel                    # bf16 storage; measured 7.0e-3
    finally:
        ttnn.close_device(dev)
