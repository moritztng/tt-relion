# tt-relion

RELION cryo-EM single-particle refinement on Tenstorrent hardware, plus a fast 2D FFT for Blackhole
and two speed fixes for RELION's own CPU code. Research code, not a product.

## Status

| | result |
|---|---|
| Accuracy | A full RELION-5 3D auto-refinement of the tutorial dataset (4,452 particles) runs with its expectation step on Tenstorrent and reaches **3.79 Å**, RELION's own published resolution to the digit. Per-particle orientations match the CPU reference more closely than RELION's CUDA backend does. |
| Speed | At best roughly **1x per dollar** against one H200, a projection from measured parts, and 15-36x worse on energy. The refinement stays host-bound, so more chips buy little. |
| RELION CPU fixes | Two patches make RELION itself **2.2x faster** end to end on CPU, with a bit-identical final map. No Tenstorrent hardware needed. |
| 2D FFT | **616,078 images/s** at box 256 in bf16 on one Blackhole chip, 16x the stock ttnn path, within 7e-3 of `numpy.fft`. Box 256 only. |

The FFT and the CPU fixes are the parts most likely to be useful to someone else today.

## 2D FFT on Blackhole

`tt_relion/fft2d.py` computes `np.fft.fft2` of a batch of complex 256x256 images on one Blackhole
chip. Each core holds an image and the DFT matrix in its own SRAM and computes the transform as two
blocked matrix products, so every image is read from DRAM once and written once.

```python
import numpy as np
import ttnn
from tt_relion.fft2d import FFT2D

dev = ttnn.open_device(device_id=0)
fft = FFT2D(dev, box=256)
spectra = fft(images)        # complex [n, 256, 256] -> complex128, same as np.fft.fft2(images)
ttnn.close_device(dev)
```

- Accuracy: relative L2 error 7.0e-3 against `numpy.fft` (bf16 storage, fp32 accumulation). In a
  simulated gold-standard reconstruction this precision changed the FSC 0.143 resolution by 0.0000 Å.
- Throughput: 616,078 images/s with the data already on device, 77 % of the chip's measured DRAM
  bandwidth limit. The same transform built from stock ttnn matmuls runs 16.2x slower.
- Limits: box 256 in bf16 only. The dense DFT matrix grows as N² and larger boxes or fp32 do not
  fit in on-chip memory. Forward transform only.

Details and the measurement scripts: [`perf/fft2d/`](perf/fft2d/).

## RELION integration

[`integrations/relion/`](integrations/relion/) holds patches for RELION 5 (`ver5.0`):

- `relion5-tt.patch` plus `src_acc_tt/` add a Tenstorrent backend for the expectation step. RELION
  hands its coarse and fine squared-difference kernels to `tt_relion/relion.py`; any shape it does not
  handle, or any error, falls back to RELION's own CPU code.
- `relion5-coarse-blocking.patch` and `relion5-nonzeroprob.patch` are the two CPU fixes. Apply them
  to any RELION 5 build.
- `relion5-prep.patch` cuts the CPU cost of preparing a particle (shift, mask, two Fourier transforms,
  CTF) from 5 to 6 ms to about 1 ms on a 360 pixel box, with the same outputs. See
  [`docs/particle-prep.md`](docs/particle-prep.md).

Build and run instructions are in that directory's README.

## Running the tests

```sh
pip install numpy torch pytest     # plus ttnn for the device test
pytest tests/
```

The host-side tests check the tile layout and the blocked transform against `numpy.fft` in float64
and run anywhere. The device test needs ttnn and a Blackhole card and is skipped otherwise.

## Measurements

[`perf/`](perf/) keeps the raw records behind every number above, one directory per experiment.
The scripts there were written for our own machines and are not turnkey; read them for method.

## License

MIT for this repository, except [`integrations/relion/`](integrations/relion/), which patches RELION
and is GPL-2.0 like RELION itself. The device kernels under `tt_relion/kernels/` carry Apache-2.0
headers.
