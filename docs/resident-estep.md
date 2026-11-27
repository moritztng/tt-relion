# Device-resident E-step: design

The per-call hook shipped every particle's references and images to a chip for each small diff2 product and lost to
the host CPU. This design keeps particles and references on the chips for the whole run, so the host only does the
M-step and I/O.

## Where the work is (full EMPIAR-10028 job, 105,247 particles, 28 E-steps)

Counted from RELION's own sampling (`perf/rlx/resident/count_full.py` on the GPU run's log):

| part | amount | at today's measured BH rate, 32 chips |
|---|---|---|
| diff2 cross term, exact (Ozaki n14) | 1.33e15 FLOP x 105 bf16 pair products | 152 s (43 TFLOP/s per chip on the 480-wide runs) |
| projections shared by all particles (global passes) | 4e8 pixels | under 1 s |
| projections per particle (local searches, final, fine pass) | 3.6e12 interpolated pixels | 2,050 s at 55 M px/s/chip (stock ttnn gather + lerp, AICLK 1350) |

So the matmuls are cheap, and per-particle projection is the cost to engineer: at today's rate it alone is twice the
1,022 s bar. A gather + trilinear kernel that reads the
8 corners straight into tiles is DRAM-bound at about 64 B per pixel. At 1e9 px/s per chip the projections take 114 s,
which is the target for that kernel.

## Shared projection tables for local searches

RELION draws every particle's local orientations from one global grid per iteration, so a half's projections at the local
order can be built once and each particle's ~155 references fetched by row gathers. ttnn.embedding gathers rows at
160-190 GB/s on one BH chip, about 400x the interpolation rate, so the per-particle step becomes nearly free. The table
build is interpolation over the shared grid: 4.2e11 pixels over the job instead of 2.0e12 per particle. At order 5 a half's
table is ~260 GB, so it is spread over that half's 16 chips (~16 GB each) and particles are routed to the chip holding their
prior direction. The final iteration and the fine pass stay per particle (1.6e12 pixels) and need the faster kernel.
At today's interpolation rate this route brings projection from 2,050 s to ~1,150 s. With a 5x faster kernel it drops to
~230 s.

## Layout

- **Sharding by particle and half-set.** Chips 0-15 hold half 1 and chips 16-31 hold half 2, about 3,300 particles each
  at 32 chips. Each chip therefore needs only its own half's reference. Particle Fourier images (full 360 half-transform,
  fp32, 0.5 MB each) plus CTF parameters are loaded once, about 1.7 GB per chip, and stay resident.
- **Reference per iteration.** The host uploads the half's Fourier volume (pad 2, fp32) once per iteration, about 1.5 GB
  per chip. For global passes the host (or one chip per half) projects the coarse set once, which is 36,864 x 1,021 pixels
  at s50, and every chip keeps that set as the shared reference operand.
- **diff2, global passes.** One matmul of [particle batch x translations, 2 Pc] by [2 Pc, orientations], with
  w * CTF * shifted X on the particle side so the reference side is shared by all particles. The cross term is exact
  through Ozaki base-4 digits over 480-wide K runs. |CTF P|^2 is a second product of the same form, once per particle
  rather than per translation, and |X|^2 is a row norm. The digit sums are combined on the chip in plain fp32, which
  matches the float64 combine on real data because the noise weighting keeps d2 near the pixel count.
- **diff2, local searches and final.** Each particle has its own ~155 coarse orientations. These are projected on the
  chip (the gather kernel above) into a per-particle reference block, followed by a batched product. Batched per-particle
  products run at only ~8 TFLOP/s, which costs ~36 s over the job and is acceptable.
- **Significance and weights** stay on the chip: row minimum, exp, a top-k for RELION's adaptive fraction, then the fine
  pass on the significant set (8 orientation x 4 translation children each, 1.5 % of the FLOP).
- **Back to host:** per particle, its best pose, its significant samples and their weights, plus the weighted 2-D image
  sums per significant orientation (about 0.8 MB per particle per iteration). The host backprojects them in RELION's own
  code, so the M-step and the reconstruction are unchanged.

## Precision

Measured on 16 real dev-cell particles x 13 translations x 1,024 orientations, with the iteration-18 map, CTF and noise
weighting (`perf/rlx/resident/prec_real.py`). Error of d2 near the minimum vs float64, at s50 / s146:

| scheme | s50 | s146 |
|---|---|---|
| RELION fp32 direct form | 1.6e-4 | 1.4e-3 |
| Ozaki n14, combined on chip | 1.8e-4 | 1.1e-3 |
| Ozaki n12 | 2.4e-3 | 1.3e-2 |
| Ozaki n10 | 3.6e-2 (best pose flips on 1 of 16) | 0.20 |

n14 matches RELION's own precision, so it is the default. Fewer digits are a lever to be judged end to end against the
seed floor, never per call.

## C++ side

RELION calls the device once per particle batch from `MlOptimiser::expectation` (not per particle, not per orientation
chunk). Each call passes particle ids, the orientation and translation sets and returns the weights and sums above. No
Python runs per call.
