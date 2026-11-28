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

### Hook points (RELION 5, `src/acc/acc_ml_optimiser_impl.h`)

`accDoExpectationOneParticle` runs per particle and per thread in two passes. Pass 0 calls
`getAllSquaredDifferencesCoarse`, which fills the dense `Mweight` array
`[class][dir][psi][trans]` with diff2, then `convertAllSquaredDifferencesToWeights`. Pass 1 builds the fine
projection set from the coarse significant samples, calls `getAllSquaredDifferencesFine` into
`FinePassWeights`, converts again, and `storeWeightedSums` backprojects.

The batch call goes in `MlOptimiser::expectationSomeParticles`, before the threads start, over
`[my_first_part_id, my_last_part_id]`:

1. `TTBridge::estepBatch` sends the particle ids, each particle's direction and psi prior ranges and the translation
   grid. The chips already hold the particles and the half's reference for this iteration.
2. The chips return, per particle, its coarse significant set (sample index, diff2) and min diff2. This replaces
   pass 0's `getAllSquaredDifferencesCoarse` and the coarse half of `convertAllSquaredDifferencesToWeights`: the
   per-particle code copies the returned diff2 into `Mweight` (all other entries stay at `lowest()`, which RELION
   already treats as insignificant) and runs its own convert, so the fine set it builds is RELION's own.
3. In the next step the fine diff2 and weights come back in the same call and pass 1 is skipped the same way, leaving
   `storeWeightedSums` on the host. Moving it on card is the last step.

Each particle that the batch did not cover (declined shape, broken chip) runs RELION's own code unchanged, so a
failure degrades to the CPU path instead of failing the run.

### Parity of the RELION-called path

The chip op is checked against float64 on synthetic operands (`perf/rlx/resident/d2_parity.py`), and that is not
enough. Two defects passed it and were found only when RELION called the op on real data. The chip kept the radius cut
of the coarse pass for the fine pass's references (the model is uploaded once, the cut belongs to the pass). And the
balance along K left a column unscaled when every reference of the set is zero there, which is every pixel outside
the projection radius, about a fifth of the image. Those columns then set the row exponent, up to 1e5 above the
largest entry that multiplies anything, and half the digits were spent on them: the squared difference came out
0.2-0.4 too small at 16 digits and 6 at 13. Together the two changed the pose of one particle in eight in a single
iteration. The balance now gives such a column the scale zero, which is exact.

So the gate for this path is the recorded call stream: `TT_RELION_RECORD` writes what a rank asked the bridge, and
`perf/rlx/bridge/replay_gate.sh` replays it on a chip and compares every call with the CPU digest of the same record,
coarse and fine pass apart (a mixed count hides a broken pass: 122 of 160 looks like noise, 43 of 80 does not). The
CPU digest is float32, like RELION, so it cannot say which side is off when the two differ. `perf/rlx/bridge/ref64.py`
recomputes the record in float64 and `replay.py --ref` grades against that: the largest error on the samples that
carry 99.9 % of a particle's weight, its mean (a bias common to a call cancels in the weights), its spread within a
call (which does not), and the weight of the samples that are in one significant set and not the other.
`perf/rlx/bridge/cut_emul.py` forms the two products of a recorded call on the CPU the way the chip does and splits
the error into the cross and the norm term, per digit count.

Largest error against float64 on a recorded global finer iteration and a recorded local-search iteration (80 coarse
and 80 fine calls each, squared differences near 1,050 / 9,800 and 12,100):

| | global coarse | global fine | local coarse | local fine | significant sets |
|---|---|---|---|---|---|
| RELION's arithmetic (float32 CPU arm) | 0.0001 | 0.0019 | 0.002 | 0.002 | 320 / 320 |
| chip, 16 digits, before the balance fix | 0.0039 | 0.29 | 0.40 | 0.42 | 317 / 320 |
| chip, 16 digits | 0.00008 | 0.0009 | 0.0005 | 0.0010 | 320 / 320 |
| chip, 13 digits | 0.0010 | 0.0011 | 0.0007 | 0.0011 | 320 / 320 |
| chip, 11 digits | 0.0015 | 0.0051 | 0.0043 | 0.0085 | 320 / 320 |

At 13 digits the op is at the float32 floor of its operands, and through RELION it moves the same poses as 16 (table
below), so 13 is the default (`TT_RELION_DIGITS`). 16 costs 31 % more chip time in the fine product and 49 % more in
the coarse one. Run the gate after every change to the op, the bridge or
the batch form, then grade one iteration through RELION against the CPU arm continued from the same state
(`perf/rlx/iter_ab.sh`): `star_cmp.py` counts the particles whose pose differs and
`d32_cmp.py` correlates each half map with the CPU arm's.

One iteration of the 13,156-particle dev cell through RELION on 8 chips, against the CPU arm:

| | poses that differ (Rot) | half maps, 1 - correlation |
|---|---|---|
| global finer iteration, 16 digits, before the balance fix | 2 | 1.2e-8, 9.8e-9 |
| global finer iteration, 16 digits | 0 | 1.0e-10, 1.2e-10 |
| global finer iteration, 13 digits | 0 | 1.0e-10, 1.2e-10 |
| local-search iteration, 16 digits, before the balance fix | 24 | 6.3e-7, 4.5e-7 |
| local-search iteration, 16 digits | 1 | 2.7e-9, 1.5e-8 |
| local-search iteration, 13 digits | 1 | 2.0e-10, 1.5e-8 |

Two CPU runs of the same iteration that differ only in the random seed read 1.2e-3 to 1.4e-3 on the same scale, so
the chip path sits five to seven orders of magnitude inside the variation RELION has by itself.

The gate for a whole refinement is `perf/rlx/fsc.py REF.mrc X.mrc --floor SEED.mrc`: the map against the CPU
reference, shell by shell, beside the reference against a second CPU run with another seed. It passes when no shell
up to the seed pair's own resolution is more than 0.02 under the pair.

### Pixels RELION weights with zero

RELION hands over the whole (size / 2 + 1) x size transform and weights every pixel past its resolution limit with
exactly zero, so about pi / 4 of the pixels carry weight (iteration 16 of the 10028 job: 1,021 of 1,300 on the
coarse pass, 9,922 of 12,640 on the fine pass). A zero-weight pixel adds nothing to any of the three sums, so the
bridge does not hand it to the chip (`TT_RELION_PIX_DROP`, on by default). It keeps one set per pixel grid, the
union of the pixels any particle has weighted so far, so the chip sees one set per pass. The products run over K in
chunks of 1,024 pixels: the coarse pass goes from two chunks to one, the fine pass from 13 to 10.

Replay of 160 recorded calls on one chip against float64 (`perf/rlx/bridge/replay.py`, 13 digits, batch 32, AICLK
1,343 to 1,350): the best pose and the significant set agree for 160 of 160 calls with the drop and without, global
iteration and local iteration alike. The largest error with the drop is 1.0e-3 (global iteration without it:
0.9e-3). Chip seconds for the 160 calls: 5.2
with the drop against 11.0 without (global, iteration 16), 4.3 against 5.2 (local, iteration 24).

Called from RELION on the global finer iteration of the dev cell (four chips as two ranks with two chips, `--j 80`,
13 digits, `results/e2e_pix`): 0 of 13,156 poses differ from the CPU arm, and the E-step takes 2.55 min against
2.75 min without the drop (AICLK 1337 MHz or more in every sample, host load 66 to 95).

### Waking the threads

RELION runs one particle per `--j` thread, and the bridge parks each thread until its batch is back. A parked thread
is woken for its result or to collect the next batch, never because another particle arrived: with one condition for
all, each arrival woke every parked thread (190 at `--j 192`), and the thread that stacks the batch for the chip
waited behind them. On an idle 8-core host with 190 parked threads, stacking a 58-particle batch takes 1.2 ms alone
and 42 ms while 64 particles arrive (`tests/test_batch.py` counts the wake-ups).
Called from RELION the change keeps the poses (0 of 13,156 differ from the CPU arm on the dev cell's global finer
iteration, `results/e2e_wake`) and has shown no gain yet: the E-step took 2.62 min at host load 100 to 127 against
2.55 min at 66 to 95 before it, on the same four chips.

### Leaving small orientation sets to RELION

`TT_RELION_MIN_ORI=<n>` declines a coarse call that brings fewer than n orientations, and the fine call that follows
it on the same thread, so RELION's own kernel does that particle. A local search brings a few hundred orientations
per particle and cannot fill a chip tile. Off by default (0).

### The last iteration

An auto-refinement ends with one iteration at the full image size. For a 360-pixel box that is 65,160 pixels per
image, and the model's table on the chip is 12 GB of its 34 GB. Two things follow.

A reference set that would not fit beside the table is scored a piece at a time, about 1,000 orientations a piece at
this size, each piece with its own balance. The products stay exact. Sets that fit run as before. On one chip against
float64 at this size (`perf/rlx/resident/d2_parity.py --size 360`, `results/e2e_fs1`): 12 particles against 2,304
orientations, the best pose agrees for all 12, and the squared difference is off by 0.014 at a value of 100,355,
where float32 resolves 0.008. A batch takes 3.7 s, 0.31 s per particle.

The table passes through host memory on its way to the chip: measured at box 360 (`perf/rlx/resident/table_probe.py`,
`results/e2e_tp`, AICLK 1350 MHz), one server peaks at 24.5 GB during the 10.1 s build and 26.6 s first upload and
drops back to 2.14 GB steady RSS afterwards, leaving 20.7 GB free on the chip beside the 11.2 GB table. Every chip
wants its table at the start of that iteration, so `TT_RELION_TABLE_TURNS=<n>` (default 4) limits how many chips of a
host upload at once while the others wait their turn. `--preread_images` also counts here: RELION keeps the whole
particle set in every MPI rank, so with one rank per chip the host holds as many copies as it has chips. Without it
RELION reads each particle from its stack when it needs it, which costs nothing here if the stacks are in the page
cache or on tmpfs: the same dev-cell iteration without `--preread_images` gave 0 of 13,156 differing poses and an
E-step of 2.38 min (2.55 min with it; AICLK 1337 MHz or more, host load 103 to 123, `results/e2e_pre`). Combining
candidate window 40 and no `--preread_images` with `TT_RELION_PREP=0` (`results/e2e_stack`, four chips as two ranks
with two chips and `--j 80`, AICLK 1337 MHz or more) kept 0 of 13,156 differing poses and half-map 1 - correlation
1.04e-10 / 1.11e-10 (it016 E-step 1.70 min, it017 1.57 min); adding the weighted-sums memory, deferred back-projection
and prefetch patches kept the same grade (0 of 13,156 poses, 1.03e-10 / 1.12e-10), cut rank CPU from 25.4 to 18.0 ms
per particle, and took it017 from 1.57 to 1.38 min.

A rank can drive more than one chip. `RLX_CHIPS_PER_RANK=<k>` (read by `perf/rlx/rank.sh`) gives every follower rank
k chips and its `--j` threads feed all of them, so the host holds fewer copies of RELION for the same chips. On the
global finer iteration of the dev cell, four chips as two ranks with two chips and `--j 80` each gave the poses of
the CPU arm (0 of 13,156 differ, half maps 1.0e-10 and 1.1e-10) and finished the E-step in 2.75 min; one rank per
chip with `--j 40` took 3.28 min on the same chips with the same grade (AICLK 1343 MHz or more in every sample of
both runs, host load 54 to 95). Four chips per rank grade the same: eight chips as two ranks with four chips and
`--j 160` each gave 0 of 13,156 differing poses and an E-step of 1.23 min (host load 8 to 29, AICLK 1331 MHz or
more), and the same eight chips as four ranks with two chips and `--j 80` each took 1.18 min (host load 29 to 80,
`results/e2e_k4`). Split half-sets need two follower ranks, so k is at most half the chips.

`TT_RELION_NORM_DIGITS=<n>` gives the norm product alone more digits than the cross product. Off by default: with
the balance fix the norm term needs none.
