# Coarse batching on ttnn, chips 16-23 (2026-11-27)

400-particle smoke, 1 iteration, 9 ranks x --j 8, RELION built by perf/rlx/build_relion.sh.
A: one device call per particle. B: TT_RELION_BATCH=8 (TT_RELION_BATCH_WAIT_MS=50).

- it001 poses, origins, MaxValueProbDistribution and LogLikeliContribution: identical in 400/400 rows.
- Batches formed: per rank 8 batches of 44-56 particles, mean 5.5-7.0 per batch.
- Wall 38.6 s (A) vs 34.5 s (B). The run is dominated by startup, so this is not a perf number.
- AICLK on chips 16-23 sampled during the run: up to 1350 MHz; load1 15.5 (A), 8.8 (B).
- Neither arm fell back to the CPU path (no "TTBridge: ... raised").

## Reference projections kept on chip (pass 10, 23:02Z)

The coarse batch path now uploads its split reference projections once per iteration (device_server
`lhs_key`) and ships only particle operands after that. B vs A: 0 of 400 data.star rows differ.
Per-chip upload time 0.52-0.66 s -> 0.26-0.30 s (TTDEV up_s). The chip is busy ~2 s per rank against
~8-9 s of host-side coarse work (C_s), and readback (down_s ~1.2-1.3 s) is the largest device term.
AICLK up to 1350 MHz during, load1 5.7.

## K-chunk partials added on chip (pass 10, 23:04Z), now the default

TT_RELION_SUM0=1 vs A (host float64 sum): poses and origins identical 400/400; MaxValueProbDistribution
differs in 280 rows by at most 2.1e-5, LogLikeliContribution in 354 rows by at most 6e-5. Per chip:
readback 1.2-1.3 s -> 0.10-0.15 s, chip busy ~1.9 s -> 0.57-0.77 s, host coarse time (C_s) ~8.7 s -> ~5.0 s.
load1 7.8. `perf/rlx/star_cmp.py` does the row comparison.
