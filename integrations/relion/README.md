# RELION 5 on Tenstorrent

This directory holds everything needed to make RELION 5 call Tenstorrent kernels for its expectation step.
It is a fork overlay, not a vendored copy: `relion5-tt.patch` applies to a clean `3dem/relion` `ver5.0`
checkout and `src_acc_tt/` drops into `src/acc/tt/`.

Nothing here is merged into RELION upstream and nothing here is enabled by default.

## What it does

RELION's accelerator seam, `src/acc`, already carries four backends (`cuda`, `hip`, `sycl`, `cpu`).
The patch adds a fifth entry point at the layer where RELION passes whole tensors rather than launch
geometry: `runDiff2KernelCoarse` in `src/acc/acc_helper_functions_impl.h`. When the shape is a 3D
reference against 2D data, the call is handed to `TTBridge::diff2Coarse` and RELION's own kernel is
skipped. Every other shape, and any failure, falls through to the existing code, so a broken device
degrades to CPU instead of failing the refinement.

`src/acc/tt/tt_bridge.cpp` is the only file in RELION that knows about Python. It embeds one CPython
interpreter per process and calls `tt_relion.relion`, passing RELION's own buffers as memoryviews
so nothing is copied. The interface is plain C++ over plain arrays, so replacing the embedded
interpreter with a C++ libttnn implementation later needs no change anywhere else in RELION.

## Build

```sh
git clone --branch ver5.0 --depth 1 https://github.com/3dem/relion
cd relion && git apply /path/to/relion5-tt.patch
mkdir -p src/acc/tt && cp /path/to/src_acc_tt/* src/acc/tt/
mkdir build && cd build
cmake .. -DALTCPU=ON -DTT=ON -DCUDA=OFF -DGUI=OFF -DMKLFFT=OFF -DFETCH_WEIGHTS=OFF
make -j
```

`relion5-tt-candidates.patch` (optional, apply after `relion5-tt.patch`) hands the coarse pass over in
candidate form: RELION passes the log priors of its orientations and translations with the particle, and
gets back only the samples whose `diff2 - log(pdf_orientation) - log(pdf_offset)` lies within
`TT_RELION_CAND_WINDOW` of the particle's best (40 is the tested value; unset or 0 keeps the dense block).
RELION's own significance rule (exponent, sort, cumulative sum, `adaptive_fraction` threshold) then runs
on that list instead of on the dense orientation x translation block. It applies to a single-image
particle with a fine pass to follow (`--oversampling` above 0) and no first-iteration cross-correlation;
every other case, and any call the bridge declines, takes the dense path unchanged.

`relion5-particle-arena.patch` (optional) lets `--j` exceed the number of cores. RELION's `--cpu` path
runs its particle loop in TBB's default arena, which stops at the core count whatever `--j` says. With
the expectation step on a card a particle thread mostly waits for its batch, and the number of waiting
threads is what fills a batch, so the patch gives the loop an arena of `--j` slots.

`TT=ON` requires `ALTCPU=ON`: the kernels that are not offloaded yet compile from `src/acc/cpu`, and
`ALTCPU` is also what defines `PROJECTOR_NO_TEXTURES`, which makes the padded Fourier reference a plain
array instead of a texture object. `TT=ON` without `ALTCPU=ON` is a configure-time error.

Needs Python development headers (`python3-dev`).

## Run

```sh
export PYTHONPATH=/path/to/tt-relion
export TT_RELION_BACKEND=torch        # or ttnn
mpirun -n 3 -x PYTHONPATH -x TT_RELION_BACKEND \
    build/bin/relion_refine_mpi --cpu --j 6 ... 
```

`--cpu` is required. Without it RELION uses its plain scalar path and never enters `src/acc` at all,
so the bridge is never called and the run silently produces a pure-CPU answer.

One MPI rank per card. Gold-standard 3D auto-refine needs at least three ranks (a leader plus one
follower per half set), so it needs two cards. RELION's own error message suggests
`--debug_split_random_half` as a single-process escape; that option no longer parses in RELION 5.

Environment read by the Python side:

| variable | meaning |
|---|---|
| `TT_RELION_BACKEND` | `torch` (host, exact, slow) or `ttnn` (device) |
| `TT_RELION_ORI_CHUNK` | orientations per chunk, bounds peak host memory |
| `TT_RELION_CHECK` | keep the computed scores for a residual check |

## Two CPU fixes that need no Tenstorrent hardware

Both apply to a clean `ver5.0` checkout on their own and leave every output bit-identical. Together
they make the RELION-5 tutorial auto-refinement 2.2x faster on CPU (878.18 s to 399.22 s wall), and
the final map is the same to the last digit (4.033896 Å post-processed).

- `relion5-coarse-blocking.patch`: a one-line fix in `src/acc/acc_helper_functions_impl.h`. The
  remainder of the coarse orientation loop is taken modulo a pixel block size instead of the
  orientation block size, so for most jobs the blocked kernel never runs. RELION's coarse pass gets
  2.985x faster. The CUDA path has the same line.
- `relion5-nonzeroprob.patch`: caches the per-sampling constants in
  `HealpixSampling::selectOrientationsWithNonZeroPriorProbability`, 54x faster for that function.

```sh
cd relion && git apply /path/to/relion5-coarse-blocking.patch /path/to/relion5-nonzeroprob.patch
```

## Particle preparation

Before RELION scores a particle it shifts and normalises the image, applies the soft mask, takes the Fourier
transform of the masked and of the unmasked image, sums the power spectrum and fills the CTF image. On a 360 pixel
box that is 5 to 6 ms of CPU per particle in every iteration, which is most of what a rank's CPU does once the
scoring runs on chips.

`relion5-prep.patch` with `src_acc_tt/tt_prep.{h,cpp}` does the same work in about 1 ms. Mask weights, shell
indices and FFT plans are built once per box size instead of once per particle, the transforms run as a row pass
and a column pass that stops at the columns RELION keeps, the shift reads the preread single-precision image
directly, and the CTF is filled a row at a time. RELION's work is kept in full: soft mask with its background
fill, both transforms, power spectrum and the high-resolution sum.

It handles 2D images in an even square box with `--zero_mask`. Helices, tomography and anything else take
RELION's own code. Apply it after `relion5-tt.patch` and `relion5-estep-profile.patch`
(`perf/rlx/build_relion.sh` has the order).

| variable | meaning |
|---|---|
| `TT_RELION_PREP=0` | use RELION's own code |
| `TT_RELION_PREP_WISDOM=<prefix>` | where the measured FFT plans are kept, one file per box size (`<prefix>.<box>`, default `~/.cache/tt-relion/fftw`); `0` uses estimated plans like RELION, slower |
| `TT_RELION_RECORD_PREP=<dir>` | write inputs and outputs of the first particles for `perf/rlx/prep/check_prep.py` |

Numbers and the float64 check: [`docs/particle-prep.md`](../../docs/particle-prep.md).

## Weighted sums

After scoring, RELION adds each particle to the sums the reconstruction is made from (`storeWeightedSums`): it
projects the reference again for the orientations of the fine pass, forms the weighted image sums and splats them
into the 3D volume. Once the scoring runs on chips this is one of the largest CPU stages of a rank. Four patches
cut it without changing what is summed:

- `relion5-sums-sparse.patch`: the two kernels skip orientations and translations that have no weight above
  RELION's own significance threshold. RELION adds nothing for them either, it only does the arithmetic first.
  Outputs are the same bits.
- `relion5-sums-memory.patch`: the back-projected volume is one block with real part, imaginary part and weight of
  a voxel side by side, on 2 MB pages, instead of three arrays; the reference volume is on 2 MB pages too. Same
  adds in the same order, same bits.
- `relion5-sums-defer.patch` (needs `relion5-sums-memory.patch`): the back-projection appends its pixels to a log
  and applies the log in batches sorted by region of the volume. The adds reach a voxel in another order, so half
  maps agree to rounding (relative L2 3e-8, the same as two runs of unpatched RELION with more than one thread)
  and poses are identical. It takes about 200 MB per rank.
- `relion5-sums-prefetch.patch`: the projection asks for the reference lines of an image row before it
  interpolates the row. Same bits.

On the 360 pixel dev cell, global-search iteration, scoring on 4 chips with 5 ranks of 40 threads, the stage costs
7.05 ms of CPU per particle with the first patch alone, 3.99 ms with the first three (back-projection 4.23 to
1.09 ms) and 3.00 ms with all four (projection and image sums 2.44 to 1.63 ms); poses equal the CPU run's in all
13,156 particles every time. Without any of them the same stage read 20.0 ms in an 8-chip run. On RELION's CPU
path, two arms side by side at one load: 12.9 ms without, 4.3 ms with the first two.

`relion5-plan-cache.patch` is a fifth, outside that stage: in a search without orientation priors RELION builds
the same coarse orientation list and its Euler matrices (36,864 orientations at healpix order 3) for every
particle. The patch keeps the last few lists together with every input they were built from and copies one when
a call brings the same inputs. Same bits (a check mode builds every reused list again and compares), and the
step costs 0.16 ms of CPU per particle on the dev cell where it cost 2.0 ms.

| variable | meaning |
|---|---|
| `TT_RELION_SPLAT_DEFER=0` | RELION's direct splat instead of the log |
| `TT_RELION_SPLAT_BATCH=<pixels>` | pixels that wait in the log before a batch is applied (default 4,194,304) |
| `TT_RELION_PLAN_CACHE=0` | build the orientation list for every particle, as RELION does |
| `TT_RELION_PLAN_CACHE=2` | build every reused list again and stop if it is not the same bits |

`perf/rlx/build_relion.sh` applies all of them in order; `SUMS_SPARSE=0`, `SUMS_MEMORY=0`, `SUMS_DEFER=0`,
`SUMS_PREFETCH=0` and `PLAN_CACHE=0` leave one out. `relion5-sums-record.patch` with
`src_acc_tt/tt_sums_record.{h,cpp}` records the stage's inputs and outputs for the float64 reference in
`tt_relion/sums.py` (`perf/rlx/wsum/check_sums.py`) and is not needed to run.

`relion5-estep-profile.patch` adds the timers used to measure the expectation step and is not needed
to run.

## License

Everything in this directory modifies or links into RELION and is GPL-2.0, like RELION itself. See
`LICENSE` here. The rest of the repository is MIT.

