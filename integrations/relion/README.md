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

`relion5-estep-profile.patch` adds the timers used to measure the expectation step and is not needed
to run.

## Huge pages for the reconstruction

Outside the expectation step, every iteration RELION allocates and fills several 720^3 double
grids (the padded reference transform on each follower, the back-projection on the two half-set
ranks). On a 32-core EPYC host most of that time is page faults, not arithmetic. Letting glibc back
those allocations with transparent huge pages takes no code change:

```sh
mpirun -n 3 -x GLIBC_TUNABLES=glibc.malloc.hugetlb=1 ... relion_refine_mpi ...
```

On the dev cell this cuts the host time per refinement iteration from 20.2 s to 11.4 s, and the
final iteration from 85 s to 42 s. It only changes where memory comes from, so the maps are the same
within RELION's own run-to-run variation. The kernel needs THP set to `madvise` or `always`.

## Threads for the host side of each iteration

Between expectation steps RELION works on whole 720^3 grids with one thread: the padded reference
transform, the reconstruction, and the accuracy estimate before each expectation step.
`relion5-host-threads.patch` gives that work the `--j` threads:

- 3D transforms of at least 2^22 elements run as OpenMP slabs (2D transforms per slice, then 1D
  along the third axis). The output is bit-identical to FFTW's own single-threaded 3D plan, and it
  needs no threaded FFTW build.
- The element-wise loops over those grids (padding, centring, gridding correction, weighting, soft
  mask) run in parallel. Their results do not depend on the thread count.
- The orientation accuracy estimate runs over particles in parallel and prints the same values.
- Zeroing a big grid is split over the threads, so the page faults of its first touch are too.

Per-particle 2D transforms are untouched. Apply it on top of the other patches and run with huge
pages as above:

```sh
cd relion && git apply /path/to/relion5-host-threads.patch
```

On the dev cell (3 MPI ranks, `--j 24`) the host time per refinement iteration drops from 11.4 s
with huge pages alone to 2.6 s, and the final iteration from 42 s to 18 s. The final map matches
the unpatched run to a correlation of 0.99999, well inside RELION's own run-to-run variation. The
patch only touches RELION's CPU code, so CPU and GPU builds both pick it up.

## License

Everything in this directory modifies or links into RELION and is GPL-2.0, like RELION itself. See
`LICENSE` here. The rest of the repository is MIT.

