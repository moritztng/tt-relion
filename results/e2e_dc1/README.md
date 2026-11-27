# Dev cell, one global iteration: RELION CPU kernels vs the chip hook with coarse batching (2026-11-27 23:07Z)

13,156 particles, healpix 2, 9 ranks x --j 8, chips 16-23 (one per rank), same binary both arms.
N = TT_RELION_BACKEND=nullbase (the bridge declines, RELION's CPU kernels run). T = ttnn + TT_RELION_BATCH=8.

- E-step: N 30 s, T 35 s. Job wall: N 68.3 s, T 73.3 s. The hook is slower than the CPU at this iteration.
- Each chip was busy 12.9-15.1 s of its 40.8 s life (upload 5.3-7.8, matmul 2.4-3.0, readback 4.5-4.9 s). The rest of
  the E-step is host work in the bridge: 37-42 thread-seconds of coarse operand building and 60-65 thread-seconds of
  per-particle fine calls per rank, in Python under one GIL per rank.
- it001 vs N: rot/tilt differ in 55 of 13,156 rows, psi 65, origin X 235, origin Y 223. Not graded against the seed
  floor: that is done on a finished refinement.
- AICLK: the eight busy chips reached 1350 MHz and averaged 1101-1109 MHz over the arm, because they idle at 800
  between calls. load1 30.5 (T), 24.9 (N).
- No fallback: every coarse and fine call was handled on chip (summary.txt).

Reading: the chips wait on the host. Batching the coarse product is not enough while the particle operand and the
fine pass are prepared per call on the host. The late iterations carry most of the job's work, so the same A/B on
one of them (perf/rlx/iter_ab.sh) is the next measurement.
