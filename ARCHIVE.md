Archive of tt-bio branch wk/relion-precision-fsc at 1782459325687fcf2f9f002af50994b5e67a9073, forked from tt-bio main at 71cae6ce66a0a393c8b6776df049a3af2574b681.
Files are the ones this branch added or changed; the rest of tt-bio is unchanged.

commit 1782459325687fcf2f9f002af50994b5e67a9073
Author: moritztng
Date:   Fri Aug 14 09:04:31 2026 +0200

    precision(p3): bf16 costs 16 microangstrom at n=8, indistinguishable from RELION's own GPU path
    
    The seed campaign the verdict owed. Box 256, seeds 11 23 37 51 67 83 101 113, paired noise,
    control RELION axis-aligned trilinear in fp64.
    
      tri/bf16_dev        +0.000016 A  95% CI [-0.000001, +0.000033]  n=8   -> 3,000x inside the 0.1 A bar
      tri/bf16_dev_pess   +0.000014 A  95% CI [+0.000005, +0.000023]  n=8
      tri/tex8            +0.000018 A  95% CI [-0.000225, +0.000260]  n=8
      tri/fp32            +0.000000 A                                 n=3
      twopass/bf16_dev    +0.060216 A  95% CI [+0.049379, +0.071053]  n=8
    
    Paired per-seed, which is the form the acceptance clauses are written in:
    bf16_dev minus tex8 is -0.000002 A on a CI of +-0.000244, straddling zero. bf16 is not separable
    from the arithmetic RELION's own GPU port ships. The seed-11 point estimate the planning pass
    quoted (+0.000028) sat 0.6 sd above the n=8 mean, so the seeds narrowed a number rather than
    moving it.
    
    Three things the campaign found that one seed could not.
    
    bf16's seed-to-seed sd is 14x tighter than tex8's (2.0e-5 against 2.9e-4). p3_shells.py gives the
    mechanism from the stored FSC curves: bf16's deficit is systematic and low-frequency, 46.4% of it
    in the lowest octile and 4.9% in the top two, where the crossing is decided. tex8's is uniform and
    sign-alternating, i.e. seed noise, because the texture unit's 8-bit fraction is an error in the
    sampling position while bf16's is an error in the value.
    
    That also clears the stop condition section 5 set before the run -- no arm concentrates its deficit
    at high frequency -- and confirms the prediction at the curve and not just at the crossing:
    -7.1e-7 in FSC units predicted, +1.16e-6 measured in the crossing's octile.
    
    Cross-arm pairing buys no variance reduction here (measured 1.005, 0.920, 1.000 against an
    independent difference), because each delta is already differenced against a common per-seed
    control. p3_paired.py says so rather than implying otherwise. The absence of common mode is itself
    evidence the two error mechanisms are unrelated.
    
    Refilled seed 37's two arms, lost to the 04:03Z OOM. Paid for by dropping the second accumulator
    seed at 200/400 -- an unpaired headline arm is a defect in the primary acceptance, a second seed
    of an arm that moves no decision is not. Recorded in section 8.1, not left to be noticed.

commit 50acd90d1fde77cc5a4990097fadd8315bd898bd
Author: moritztng
Date:   Fri Aug 14 06:07:26 2026 +0200

    precision(p3): one OOM must cost one arm, not the queue
    
    The campaign was OOM-killed on seed 37 at 04:03Z because a cheap box-96 arm was run
    beside it. A box-256 arm holds 8.7 GB against ~9 GB available, so nothing may run
    concurrently -- the plan already said so and I did it anyway. `set -e` then turned that
    single kill into an abort of the remaining 22 arms, which is the part that actually cost
    something, so it is gone: a failed arm now logs `!!! ARM FAILED` and the queue continues.
    
    Folded p3_campaign2.sh's tex8 seeds into the one queue rather than run a second script
    that waits on the first; two scripts is how you end up with two arms alive at once. Arms
    are skipped when their JSON exists, so a relaunch resumes. Each log line now records
    MemAvailable at arm start.

commit b3e05908be7a84e1cb21dfbb8bb2126b55ebe80c
Author: moritztng
Date:   Fri Aug 14 06:03:46 2026 +0200

    precision(p3): 25x more accumulator roundings does not cost more resolution
    
    bf16acc at FLUSH=1 rounds the backprojection accumulator 400 times instead of 16 and
    lands at -0.000038 A, smaller in magnitude than FLUSH=25's -0.000131 A. Both sit under
    the estimator's own seed-to-seed spread, so the arm is null and the rounding count is
    not what it is limited by -- which is the honest form of the lower bound section 6
    asked for, and it costs 201 s at box 96 against ~30 s.
    
    First box-256 campaign seeds alongside.

commit 426e0536f2689c841ca1b8bb8b00c7c1652165d6
Author: moritztng
Date:   Fri Aug 14 05:57:07 2026 +0200

    precision(p3): tex8 needs the same seeds, or the comparison to quote is unpaired
    
    Section 8.2 names "bf16_dev no worse than tex8, in the same harness" as the secondary
    acceptance and the one to quote, but section 8.1's loop runs tex8 at seed 11 only. A
    paired n=8 against an unpaired n=1 is not that comparison. p3_campaign2.sh runs the
    seven missing tex8 seeds, chained on the first campaign's completion marker rather
    than beside it, because two box-256 arms do not fit in 9 GB.

commit 9b07292c72d5cac3d0ab7a8df264d29688bfa173
Author: moritztng
Date:   Fri Aug 14 05:55:05 2026 +0200

    precision(p3): the seed campaign harness, and FLUSH is an argument now
    
    The verdict owes a confidence interval, not another point: the predecessor got three
    different verdicts out of one experiment at one seed. p3_campaign.sh runs the box-256
    arms one at a time (an arm holds 8.7 GB RSS, MEASURED, against 9 GB available) and
    orders them so each seed lands a complete triple. p3_agg.py folds them into mean, sd
    and a t-interval with the sha256 of every file it read.
    
    FLUSH is argv[7] so the accumulator arm can be run at 1 rounding per orientation
    instead of 25 without editing the file; the default and the output filename are
    unchanged, so every result already on disk still reproduces.
    
    s2p2_fsc_box96_snr0.05_s11_pcrepro.json is the pc re-run of the predecessor's bh-host
    box-96 regression, kept beside it rather than over it. The paired delta agrees to
    1.67e-9 A.

commit e88d2045e5c0f54176ae001f0339f93139ffd63f
Author: moritztng
Date:   Fri Aug 14 05:49:33 2026 +0200

    precision(p3): bf16 costs 28 microangstrom at box 256, less than RELION's own GPU path
    
    Box 256, padded 512, SNR 0.05, 400 orientations, seed 11, paired noise, symmetric pipelines,
    control = RELION axis-aligned trilinear in fp64 crossing FSC 0.143 at 4.7518 A.
    
      tri/bf16_dev       +0.000028 A   the kernel's round-off at the card's measured magnitude
      tri/bf16_dev_pess  +0.000024 A   the same error with all of it static, so none can average
      tri/tex8           +0.000045 A   RELION's shipping CUDA path, 8 fixed-point fraction bits
      twopass/bf16_dev   +0.056486 A   the kernel as built, interpolant and round-off together
      tri/pert1e0        +0.024983 A   sensitivity control at 170x the kernel's error
    
    The interpolant arm reproduces the predecessor's fp64 measurement at the same box and seed
    (+0.056434 A), so the whole of the kernel's round-off is worth +0.000052 A on top of its
    interpolant, and the control resolution agrees to 4 decimal places across two harnesses on
    two hosts. Section 5's prediction, written before the run, was under 0.001 A.

commit aeead04b3f49b518c1544d35f99baeaf86a5a7e7
Author: moritztng
Date:   Fri Aug 14 05:36:36 2026 +0200

    precision(p3): compute the FSC one z-plane at a time, or box 256 dies on pc
    
    The first box-256 arm was OOM-killed AFTER its last orientation and before writing anything:
    H.fsc masks a P^3 complex128 array twice and the gridding divide allocates a third, 6.4 GB on
    top of an 8.5 GB working set on a 30 GB host with 16 GB free.
    
    fsc_planewise consumes the accumulators plane by plane and frees each arm as it goes. Same
    shells, same float32 radius, same rounding; the box-96 seed-11 tex8 delta agrees to 12
    significant figures (3.5e-15 A), the residual being float64 summation order.

commit 84d4b92c2829b7528e4c3ca5ea96ccf86947a8b5
Author: moritztng
Date:   Fri Aug 14 05:32:36 2026 +0200

    precision(p3): the bf16 round-off harness, and the error is flat across shells
    
    The projection spike answered the interpolant question in Angstroms and left the round-off
    question at an aggregate per-slice relative L2 on the device. This measures the round-off
    error per spatial-frequency shell, and prices it at gold-standard FSC 0.143.
    
    p3_precision.py, box 256, 64 orientations: bf16 2.18e-3 relative and flat across eight
    shells (max/min 1.05), of which 1.25e-3 is the static volume-storage term. RELION's own
    shipping CUDA path, whose texture unit quantises the interpolation fraction to 8 fractional
    bits, is 1.31e-3 on the same measurement -- so our bf16 sits within 1.7x of the error
    RELION's production GPU already accepts.
    
    p3_precision_fsc.py adds a precision model to the s2p2 FSC harness. tri/fp64 against the
    control is bit-identical, which is the A/A. Box 96, 200 orientations, seed 11: every
    candidate precision lands within 0.0002 A, and the injected ladder does not move the
    crossing until 1e0 (+0.0119 A) and 3e0 (+0.2677 A).
    
    s2p2_fsc.py only reads argv when run directly, so its weight functions import. Verified
    bit-identical on the box-96 seed-11 regression.

