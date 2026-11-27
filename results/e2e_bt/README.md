# Coarse batching on ttnn, chips 16-23 (2026-11-27)

400-particle smoke, 1 iteration, 9 ranks x --j 8, RELION built by perf/rlx/build_relion.sh.
A: one device call per particle. B: TT_RELION_BATCH=8 (TT_RELION_BATCH_WAIT_MS=50).

- it001 poses, origins, MaxValueProbDistribution and LogLikeliContribution: identical in 400/400 rows.
- Batches formed: per rank 8 batches of 44-56 particles, mean 5.5-7.0 per batch.
- Wall 38.6 s (A) vs 34.5 s (B). The run is dominated by startup, so this is not a perf number.
- AICLK on chips 16-23 sampled during the run: up to 1350 MHz; load1 15.5 (A), 8.8 (B).
- Neither arm fell back to the CPU path (no "TTBridge: ... raised").
