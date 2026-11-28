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
| `TT_RELION_NICE` | `<n>`: RELION's `--j` threads lower their own priority by n after their first batch, so the device server gets a core when it asks. Default 0: off. |
| `TT_RELION_SWITCH_MS` | `<ms>`: how long a thread may keep the interpreter lock while another asks for it (`sys.setswitchinterval`). Default 0: CPython's 5 ms. Not measured through RELION yet. |
| `TT_RELION_YIELD_MS` | `<ms>`: threads whose result is in, or that bring a new particle, wait at most this long while a batch leader with a chip reserved sends its batch. Default 250; 0 turns it off. |
| `TT_RELION_STAGE` | `1`: a full batch is written to a chip that still runs the batch before it (its own shared segment, the request waiting in the server's pipe), so the server starts it without a thread being woken in between. A batch that is not full waits for an idle chip as before. Default off: through RELION it cut the chip's wait from 30 ms to 9 ms a batch with the same orientations, but showed no gain in the E-step (`docs/resident-estep.md`). |
| `TT_RELION_SYNC_SLEEP` | `<f>`: the device server sleeps f times a phase's shortest time so far before it waits for the card, instead of holding a host thread busy through the wait. Default 0: off. |
| `TT_RELION_PIX_DROP` | `0` hands every pixel to the card. Default: only the pixels RELION weights. RELION passes the whole coarse or fine image and gives the pixels past its resolution limit a weight of exactly zero (21 % of them in the recorded 3D refinement: 1300 points of which 1021 weighted on a 50-pixel coarse pass); they add nothing to any sum |
| `TT_RELION_OWN_FORM` | `1` scores a batch of particles that bring their own orientation sets (fine pass, local searches) one particle against its own set, instead of every row against the union of the sets. Off by default |
| `TT_RELION_OP_THREADS` | host threads that build a batch's image operand in the device server (default 4) |

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

## License

Everything in this directory modifies or links into RELION and is GPL-2.0, like RELION itself. See
`LICENSE` here. The rest of the repository is MIT.

