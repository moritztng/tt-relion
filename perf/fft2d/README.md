# 2D FFT measurements

| file | what |
|---|---|
| `bench.py` | throughput and error of `tt_relion.fft2d.FFT2D` at box 256 bf16, data resident on device |
| `fft_ab.py`, `fft2d_ab.json` | A/B against the same transform built from stock ttnn matmuls, plus an A/A pair for noise |
| `fft2d_result.json` | the box/dtype sweep: box 256 bf16 runs, larger boxes and fp32 fail on on-chip memory |
| `e1_fsc.py`, `e1_fsc_box*.json` | simulated gold-standard reconstruction with bf16 FFT precision injected; FSC 0.143 moves by 0.0000 Å |

Published numbers (one Blackhole p150, August 2026): 616,078 images/s at box 256 bf16, 76.8 % of the
measured DRAM bandwidth floor, 16.2x the stock ttnn path (summed op time; 271x end to end including its dispatch overhead), relative L2 error 7.013e-3 against
`numpy.fft`. The chip clock (AICLK) was not recorded for these runs; throughput scales with it, so
record it next to any new number.

The measured runs used an earlier version of `tt_relion/fft2d.py` with the same kernels, tile layout
and default tuning (16 images per core, `x_depth=2`, `out_depth=32`, `chunk=16`, HiFi2). The tile layout of the current
module is checked bit-for-bit against that version in this repository's history.
