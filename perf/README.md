# Measurements

Raw records from the experiments behind the numbers in the top-level README, one directory per
experiment. Each holds the scripts that ran, their outputs, and a short write-up where one exists.
Paths such as `/path/to/tt-relion` and the `benchlock.sh` wrapper (a lock that waits until the card
is free) come from the machines the runs were made on; the scripts document method and are not
meant to run unmodified.

| directory | what it measured |
|---|---|
| `fft2d/` | the fused 2D FFT: throughput, accuracy, A/B against stock ttnn, FSC impact of bf16 |
| `relion-acc-backend/` | the RELION accelerator bridge and its per-call cost |
| `relion-end-to-end/` | full auto-refinement on Tenstorrent against CPU and CUDA references, including the H200 runs |
| `relion-kernel-coarse-projection/`, `relion-kernel-diff2-fine/` | the coarse and fine squared-difference kernels on device |
| `relion-kernel-particle-fft/` | where the per-particle FFT sits in the refinement, and the `nonZeroProb` CPU fix |
| `relion-kernel-storeweightedsums/`, `relion-device-ceiling/` | whether moving more of the expectation step to the device can pay off per dollar |
| `relion-full-device-deep-perf/` | host-side profile of the whole refinement and the coarse-blocking CPU fix |

The full history of the original development branches is kept as `archive/tt-bio/*` tags.
