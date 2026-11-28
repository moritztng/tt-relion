# Particle preparation on the CPU

Every iteration, before a particle is scored, RELION's `getFourierTransformsAndCtfs` prepares it: copy the image,
scale it by the normalisation correction, shift it by the rounded offset of the last iteration, take the Fourier
transform of the unmasked image, apply the soft mask, transform again, sum the power spectrum per shell and fill the
CTF image. With the scoring on chips this is the largest piece of CPU left in a rank.

`relion5-prep.patch` routes that function to `TTPrep` (`integrations/relion/src_acc_tt/tt_prep.cpp`), which does
the same steps with less overhead. It adds no device work.

## What changed

| step | RELION | TTPrep |
|---|---|---|
| mask weights, shell index per Fourier pixel | computed per particle, per pixel | one table per box size, mask radius and edge width |
| Fourier transforms | a new FFTW plan and new arrays per particle | plans made once; a row pass, then a column pass in blocks of 32 |
| unmasked transform | full image, then windowed | the column pass covers only the columns the window keeps |
| image copy | preread float image copied to a double array | read where it is |
| CTF image | one call per pixel | one row at a time, `sin` from the vector math library |

The whole-pixel shift stays a real-space shift with zero fill. RELION does not wrap the image around, so a phase
ramp in Fourier space would be different math and is not used.

The fast path covers 2D images in an even square box with `--zero_mask`, and for the CTF an optics group without
magnification matrix, even Zernike terms, dose weighting or padding. Any other case calls RELION's code for that
piece. `TT_RELION_PREP=0` calls RELION's code for all of it.

## Cost

Thread CPU per particle in `getFourierTransformsAndCtfs`, 360 pixel box, one binary with the switch off and on,
measured by the stage profiler (`TT_RELION_PROFILE=1`, `perf/rlx/wsum/split.py`).

CPU-only arms, 2 ranks of 8 threads on a 64-core host (load average 18 to 36):

| iteration | RELION's code | TTPrep, estimated plans | TTPrep (measured plans) |
|---|---|---|---|
| global search (current size 158) | 5.08 to 5.88 ms | 1.12 ms | 0.91 to 0.97 ms |
| local search (current size 176) | 4.92 to 4.96 ms | 1.19 ms | 0.96 to 1.00 ms |

Inside a chip run the threads are busier and the same code costs more. One iteration of the 13,156 particle
development set on 4 Blackhole chips, 5 ranks of 40 threads on the same host, same binary with the switch off and
on (chip clock at least 1343 MHz in every one-second sample of all four arms):

| iteration | RELION's code | TTPrep | load average |
|---|---|---|---|
| global search (current size 158) | 7.47 ms | 1.76 ms | 87 to 97, 63 to 85 |
| local search (current size 176) | 7.38 ms | 1.69 ms | 81 to 101, 81 to 132 |

All CPU of a rank fell from 36.0 to 31.3 ms per particle in global search and from 24.0 to 16.1 ms in local search.
Against the arm with the switch off, no particle changed its rotation, tilt or psi in either iteration type (one
particle's origin moved by one pixel in global search, none in local search), and the half maps correlate to within
1.1e-11. On the merged 2-chips-per-rank layout (3 ranks of 80 threads, chip clock at least 1337 MHz, load average
91 to 127), TTPrep measured 1.83 ms in global search and 1.79 ms in local search.

FFTW's estimated plan for a 360 point transform is slow on this build (FFTW 3.3.8); a measured plan is about a
third faster per transform. FFTW picks measured plans by timing, so two ranks that each measured would end up with
different plans and round differently. TTPrep therefore measures once per box size and keeps the plans in a file,
`~/.cache/tt-relion/fftw.<box>`: the first process that needs them takes about a second, and every rank and every
later run reads the same file. `TT_RELION_PREP_WISDOM=<prefix>` puts the files elsewhere (a directory all nodes
share, for a run across machines), and `TT_RELION_PREP_WISDOM=0` uses estimated plans as RELION does, which makes
the unmasked transform the same bits as RELION's. Delete the file after changing the FFTW build.

## Accuracy

`tt_relion/prep.py` is a float64 reference of the whole function (`tests/test_prep.py` checks each of its pieces
against RELION's definitions written out as loops). `TT_RELION_RECORD_PREP=<dir>` makes RELION write
the inputs and outputs of the first particles, and `perf/rlx/prep/check_prep.py <dir>` grades them. Worst relative
L2 error over 32 real particles per iteration type:

| output | RELION's code | TTPrep, estimated plans | TTPrep (measured plans) |
|---|---|---|---|
| unmasked transform | 1.1e-7 | 1.1e-7 | 1.1e-7 |
| masked transform | 1.7e-7 | 1.7e-7 | 1.6e-7 |
| CTF image | 4.5e-15 | 4.5e-15 | 4.5e-15 |
| power spectrum | 1.7e-7 | 5.0e-8 | 4.5e-8 |
| high-resolution sum | 1.1e-6 | 2.8e-8 | 2.6e-8 |

Both paths are single precision and sit at the same distance from float64. TTPrep is closer on the two sums because
it accumulates them in double. With estimated plans the unmasked transform has the same bits as RELION's.
