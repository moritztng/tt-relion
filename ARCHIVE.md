Archive of the RELION FFT and Fourier-slice primitives that lived on tt-bio main from merge 36a406eb6 until revert 0bc00281e (2026-08-14). Snapshot at tt-bio bfac7008da71b5e4884e014eeb6d064e89144fdc.

commit 71cae6ce66a0a393c8b6776df049a3af2574b681
Author: moritztng
Date:   Fri Aug 14 03:12:18 2026 +0000

    spike(fslice): VERDICT -- GO with a caveat; 648.5 k slices/s projection, 176x grid_sample
    
    Consolidated verdict written to the state doc, superseding the running narrative. Sections 26-35 quoted
    throughput that section 36 withdrew, and section 6s prediction was withdrawn at section 13; section 40
    is now the current numbers.
    
    Blackhole p150, 130 cores, general shear, bf16, all measured:
      projection      648.5 k slices/s (box 256), 288.0 k (384), 159.8 k (512)
      backprojection  842.7 k slices/s, 0.77x projections cost per slice
      20.2% / 20.2% / 19.9% of the DRAM write floor, and 100.4% of the kernels own traffic limit
      176x ttnn.grid_sample like-for-like; 1.6e6 x ttnn.gather, which is what a wrong op adds
      Galaxy 32 processors: 20.8 M slices/s
      linear scaling: 1.03x per-core degradation across a 130x fanout
    
    Both primitives correct to bf16 round-off. Neither uses gather or scatter and neither needs them: the
    two ops the FFT spike called algorithmically wrong are routed around by the FORMULATION.
    
    GO WITH A CAVEAT. The caveat is the interpolant -- the two-pass misses section 8s accuracy gate by
    0.0089 A, a 1.2% relative resolution loss on a 4.75 A reconstruction. Every branch settleable by
    measurement is settled; the decision is not mine. The worst reachable outcome, falling back to exact
    trilinear at roughly 3.4x on stage 2, still leaves the primitive near 67x grid_sample. There is no
    reachable outcome in which RELION refinement is blocked on this primitive, which is the question
    section 12 said the task was asked to settle.

commit f65001f835d915de889c7526d826b9c2f1c5757a
Author: moritztng
Date:   Fri Aug 14 03:07:57 2026 +0000

    spike(fslice): padding and cubic taps both closed off -- every measurable interpolant branch is now dead
    
    Two levers tried, both aimed at fixing the two-pass WITHOUT giving up the tile-native property.
    
    padding_factor. It is a RELION parameter rather than a kernel property, and finer Fourier sampling
    helps every interpolant, so raising it looked free. It goes the wrong way: box 128, pad 3 gives +0.1056
    and +0.1126 A against pad 2s +0.0480 and +0.0429. Higher padding oversamples the volume but ALSO widens
    the slices sampling stride, since u3 = PAD * A[:,0], so the separable passes alias more. Recorded as a
    lever that is real and points backwards.
    
    Cubic (Catmull-Rom) taps in each 1D pass. Still exactly tile-native -- 4 taps has the same structure as
    2, just more r-independent selection matmuls -- and nearly free in wall-clock on a kernel measured at
    84.6% of its traffic limit with arithmetic at 13% of a tile. But it is NOT EVALUABLE in this harness.
    Catmull-Rom has negative lobes, and gridding divides by the accumulated weight: the signed sum cancels
    to near zero and the first run returned +11.7 A. Normalising by the sum of |w| instead -- which leaves
    every linear arm bit-identical, since linear weights are non-negative -- gives +0.1589 and +0.1823 A,
    still worse than linear, because |w| is not the correct sampling density for a signed kernel either and
    biases the reconstruction amplitude.
    
    So a higher-order kernel needs a different reconstruction normalisation, and choosing one is a
    methodology change rather than a measurement. Every branch I can settle by measuring is now settled.

commit 70c7bfa00f7a7e66fdc0f4a0e916a8ac8327e46b
Author: moritztng
Date:   Fri Aug 14 02:57:49 2026 +0000

    spike(fslice): the projection is bandwidth-saturated -- 100.4% of its traffic-implied limit
    
    Counting what the kernel actually moves, box 256, general shear, per slice:
      stage 1  28-tile band read + 8 replicated copies written, over 96 psi   393,216 B  (60%)
      stage 2  two 32x128 B assemblies + the slice write, x 25.13 tiles       257,360 B
      TOTAL                                                                   650,576 B = 4.96x the floor
    
    At the measured 420.2 GB/s roof that traffic allows 645.9 k slices/s. Measured: 648.5 k -- 100.4% of
    it, saturated within noise. And 1/4.96 = 20.2%, which is exactly the fraction of nominal floor
    reported, so the headline decomposes with nothing left over: the kernel is perfectly bandwidth-bound
    and the entire distance from the floor is the algorithms byte count.
    
    There is therefore no efficiency left to recover, only traffic, and 60% of the traffic is stage 1s
    28-tile band. Section 36 established that band is irreducible -- 1.000 reads per distinct volume tile,
    so nothing is re-read. The alternatives, all dead:
      sub-blocking the (X,Y) tile      reads are tile-granular, so 4 sub-blocks of band 14.6 cost 58 reads
                                       against 28 for the whole tile
      larger (X,Y) regions             the band is per 32x32 tile position, so nothing changes
      pre-shearing the volume per      3.92 s per iteration against 0.28 s as built -- 14.2x WORSE,
      direction so the band collapses  which confirms section 4.2s rejection with a number
    
    So the projection is at its ALGORITHMIC floor, not merely its implementation floor.

commit c10bb039a6fdfe86f0459a3ae28f970e137711f9
Author: moritztng
Date:   Fri Aug 14 02:55:46 2026 +0000

    spike(bproj): stage 1s adjoint built -- BOTH primitives now measured, backprojection is 0.77x projection
    
    The last piece. Forward contracts a band of z-planes into one W tile; the adjoint spreads one W tile
    back across the same band, applying the same masks. Transaction count identical, direction reversed.
    
      nplane  2  rel L2 4.27e-3   2353.0 ns per W tile ->    96.5 ns/slice/core
      nplane  8  rel L2 2.64e-3   7550.7              ->   309.8
      nplane 28  rel L2 2.79e-3  25195.4              ->  1033.7   (the S3 mean band)
      nplane 51  rel L2 2.79e-3  45357.0              ->  1860.8
    
    1.37x the forwards 755.3 ns/slice/core, the gap being that 28 DRAM writes cost more than 28 reads.
    
    BOTH PRIMITIVES, box 256, general shear, per slice per core:
      projection      stage2 786.8  stage1  755.3  total 1542.1 -> 648.5 k slices/s
      backprojection  stage2 153.0  stage1 1033.7  total 1186.7 -> 842.7 k slices/s
    
    Backprojection costs 0.77x projection per slice. Section 5 budgeted 1.3x -- right magnitude, wrong
    direction. It is CHEAPER because its stage-2 reads are bulk slice tiles where projections are per-row
    assemblies, and because matmul accumulation makes the write amortise across contributions.
    
    Stage 1 is now 49% of projection and 87% of backprojection, so the z-collapse and its adjoint are where
    this primitive pair actually spends its time -- not the in-plane resample the design spent most of its
    effort on.

commit 73575fd4598f1982a17a0da181fee9b9d205f047
Author: moritztng
Date:   Fri Aug 14 02:50:37 2026 +0000

    spike(fslice): section 26s 7.1x plane reuse does not exist -- corrected headline is 648 k slices/s
    
    Building stage 1s adjoint forced a look at the forward geometry, and the reuse I measured in section 26
    is not physically available.
    
    W(X,Y) = V(X, Y, a*X + b*Y). The (X,Y) index passes through unchanged, so volume tile (TX,TY,tz) is
    needed by the W tile at (TX,TY) and by no other. Consecutive output tiles have DIFFERENT (X,Y): they
    share a z-RANGE but not a single volume tile. Enumerated over a full direction at box 256: 512 W tiles,
    14336 volume-tile reads required, 14336 distinct volume tiles touched -- 1.000 reads per distinct tile.
    There is nothing to reuse.
    
    Section 26s sliding window slides through a LINEAR page space, which models fixing (X,Y) and sliding z.
    That is not the access pattern. The 7.1x was real as a measurement and irrelevant as a model.
    
    The correct configuration is shift = nplane, every tile read once, which also settles the L1-versus-DRAM
    band from section 34: with each tile read exactly once, L1 residency buys nothing and the volume must
    stream from DRAM. Measured, 8 replicated copies, DRAM: 18410.0 ns per W tile, 755.3 ns per slice per
    core.
    
    CORRECTED projection, general shear, box 256:
      stage 2  786.8 ns/slice/core   stage 1  755.3   total 1542.1  -> 648.5 k slices/s, 20.2% of floor
      boxes 384 and 512 hold the same fraction: 20.2%, 19.9%
      176x grid_sample; Galaxy 32 processors 20.8 M slices/s
    
    That is 1.33x below the 816-903 k / 25.0-28.2% reported in sections 26 through 35, and those figures are
    withdrawn. Stage 1 is now 49% of the projection rather than 13%.

commit b9d904eeb34f388a010796f3ad9027173ede221c
Author: moritztng
Date:   Fri Aug 14 02:47:08 2026 +0000

    spike(bproj): the per-row write scatter is FREE in the regime that matters
    
    Section 5 said the forwards per-row READ offset becomes a per-row WRITE offset in the adjoint. Built
    and measured: the bulk 2048 B page write becomes 32 scattered 128 B writes, with the same 16 B
    alignment quantum the read side has.
    
      fp32, 1 contribution:  bulk 2391.1 ns/volume-tile   scattered 5009.1   2.09x
      fp32, 8 contributions: bulk 6380.6                  scattered 6248.3   0.98x
    
    The scatter penalty is per WRITE, and matmul_tiles accumulating into DST means one write serves every
    contribution to that volume tile. So it costs 2.09x when a write serves one contribution and nothing at
    all when it serves eight. Real backprojection has enormous numbers of contributions per voxel, so the
    amortised column is the real one and the scatter is free there.
    
    Stage-2 adjoint stands at about 780-800 ns per slice tile scattered, or 153 ns per slice per core at box
    256 -- against projections 1108 ns per slice per core to produce one.
    
    NOT a backprojection rate yet: stage 1s adjoint is unbuilt. Section 5 describes it as spreading one
    value into 2 z-planes over the same ~28-plane window the forward contracts, so it should cost about what
    forward stage 1 costs (321-438 ns per slice per core), which would put backprojection near 0.43-0.53x
    projection rather than section 5s budgeted 1.3x. That comparison is partial until it is built.

commit 8f41aaabed15ae215ce8e38cc3c3700de5164587
Author: moritztng
Date:   Fri Aug 14 02:43:57 2026 +0000

    spike(fslice): boxes 384 and 512 reach the SAME fraction of floor, and the volume never fits L1
    
    Two things the brief asks for, and one correction to earlier numbers.
    
    Stage 2 is box-independent, measured rather than assumed. The general shear needs 8 copies of the
    padded W plane -- 8 MB at box 256, 19 MB at 384, 34 MB at 512 -- and the claim was that all three fit
    the 195 MB of chip L1 so the per-tile cost does not move. Measured at those source sizes: 4069.6 ns
    (1.0 MB), 4076.0 ns (19.9 MB), 4183.6 ns (35.7 MB). A 2.8% degradation across a 36x source, with
    correctness bit-identical at rel L2 4.8232e-03 in all three.
    
    Both stages then scale as N^2 in tiles per slice, and the floor scales as 1/N^2, so the FRACTION of
    floor is box-independent:
      box 256   902.7 k slices/s   floor 3206 k   28.2%
      box 384   400.8 k            floor 1425 k   28.1%
      box 512   221.3 k            floor  801 k   27.6%
    The FFT spike covered box 256 only and said so; this design reaches 384 and 512 at the same efficiency.
    
    THE CORRECTION. Every stage-1 figure so far used an L1-resident volume, and the volume does not fit L1
    at ANY real box -- 268 MB at 256, 906 MB at 384, 2.147 GB at 512, against 195 MB. What is L1-resident
    is the sliding plane window, 7.5 MB chip-wide, which is section 4.3s slab loop -- AND THE SLAB LOOP IS
    NOT IMPLEMENTED. Measuring the pessimistic bound instead, every entering plane straight from DRAM:
    shift 7 costs 2.07x an L1 source (2673.4 -> 5532.1 ns/tile, stage 1 109.7 -> 227.0 ns/slice/core).
    
    So the honest projection is a BAND, not a number: 25.0-28.2% of floor depending on whether the slab
    loop lands. Earlier sections reported the L1 end as if it were measured end to end.

commit dc9bb63b3f5de51f7f4721277c0b3c46612e8317
Author: moritztng
Date:   Fri Aug 14 02:40:37 2026 +0000

    spike(fslice): record the scale-balanced two-pass measurement

commit 08a66bd738208b77814ad4bed2885c4e8214d1cd
Author: moritztng
Date:   Fri Aug 14 02:40:00 2026 +0000

    spike(fslice): the two-pass deficit is intrinsic, not a bad choice of decomposition
    
    Hypothesis worth eliminating cheaply before designing anything: the two-pass picks among its four
    Catmull-Smith candidates by largest PIVOT, which is what keeps the 2x2 solve conditioned but is not what
    governs accuracy. A separable resample loses accuracy where a pass magnifies or minifies, so selecting
    the candidate whose pass SCALES sit closest to unity should have helped.
    
    It does not. Box 96 seed 11: pivot criterion +0.0615 A, scale-balanced criterion +0.0607 A. A 1.3%
    change against sep2ds +0.0484 A on the same seed and noise.
    
    So the two-passs 0.0156 A deficit is intrinsic to routing through an intermediate on the mixed
    (x_out, Y_source) lattice at all, not to which of the four ways of doing it you choose. There is no
    cheap fix inside the two-pass family, and the remaining candidates are the ones that change the
    decomposition rather than its parameters.

commit 722afcf2d1f11da3d6264b5cae93d1e42d5640d8
Author: moritztng
Date:   Fri Aug 14 02:37:30 2026 +0000

    spike(fslice): separable-2D PASSES the accuracy gate where the two-pass fails
    
    Box 256, SNR 0.05, 400 orientations, paired noise, symmetric pipelines:
    
      two-pass  n=12  mean +0.05894 A  sd 0.01132  95% CI [+0.0517, +0.0661]  -> FAILS, CI above 0.05
      sep2d     n=18  mean +0.04418 A  sd 0.01023  95% CI [+0.0391, +0.0493]  -> PASSES, CI below 0.05
      paired on the 12 shared seeds: sep2d better on 12 of 12, mean improvement +0.0156 A
    
    So the accuracy question has an answer inside the cheap family: the loss is not intrinsic to collapsing
    z and resampling in plane, it is intrinsic to the two-pass INTERMEDIATE. Both arms share the same
    z-collapse and the same axis permutation; the two-pass routes through the mixed (x_out, Y_source)
    lattice and sep2d keeps a single bilinear cell.
    
    What this does NOT establish, and it is the catch: sep2d may not be TILE-NATIVE. The two-pass
    decomposition exists precisely to make an affine 2D warp separable into two 1D passes, each of which is
    one matmul plus a broadcast plus a reader offset. A true 2D bilinear needs the four neighbours of a
    point whose BOTH coordinates vary with both output indices -- a per-element 2D gather, which is what
    section 4.2 rejected on measurement at 1849 core-cycles per point.
    
    The accuracy ordering therefore runs OPPOSITE to the implementability ordering, which is the real shape
    of the problem and was not visible before this measurement.

commit 7913bde5a87da603b4f20634d149e542e5439e0b
Author: moritztng
Date:   Fri Aug 14 02:11:36 2026 +0000

    spike(fslice): separable-2D beats the two-pass on 8 of 8 seeds -- 0.0451 A against 0.0602 A
    
    Paired at box 256, same seeds and the same noise field, so the two arms differ only in the interpolant:
    
      seed 11  two-pass +0.0564  sep2d +0.0398      seed 67  +0.0809  +0.0621
      seed 23           +0.0764        +0.0568      seed 83  +0.0495  +0.0387
      seed 37           +0.0492        +0.0364      seed 101 +0.0659  +0.0482
      seed 51           +0.0568        +0.0410      seed 113 +0.0465  +0.0373
    
    sep2d wins on EVERY seed, by 0.009 to 0.020 A, mean 25.2% better: +0.06020 -> +0.04505 A.
    
    Against the 0.05 A gate: sep2d n=8 mean +0.04505, sd 0.00967, 95% CI [+0.0370, +0.0531]. The MEAN is
    below the gate where the two-pass mean was above it, but the interval still straddles, so this is not
    yet a pass -- the margin is 0.005 A against an sd of 0.0097, which needs roughly 18 seeds to resolve.
    
    The difference is entirely the in-plane resample. Both arms share the same z-collapse and the same axis
    permutation; the two-pass routes through an intermediate on the mixed (x_out, Y_source) lattice and
    sep2d keeps a single bilinear cell. That intermediate is what section 15.3 identified as making the
    two-pass a third interpolant, and it is what costs the 0.015 A.

commit 99b4182c3d5b75187efb47e0408c9619d3f168ae
Author: moritztng
Date:   Fri Aug 14 01:50:48 2026 +0000

    spike(fslice): fix a path-precedence bug that silently discarded every sep2d result
    
    HERE / f"..." + tag + ".json" builds the Path FIRST and then attempts Path + str, which raises. The
    runs did their full five minutes of work, printed the answer to a stdout I had redirected to /dev/null,
    and then threw the result away at json.dump. Four box-256 seeds were lost that way before the empty
    output directory gave it away.
    
    Worth carrying: a result written only at the end of a long run, with stdout discarded, fails silently
    and looks exactly like a run still in progress. The tell was that the process was on seed 67 while zero
    output files existed.
    
    Fixed by parenthesising the string concatenation before the / operator. Verified: box 96 seed 11
    reproduces +0.0484 A and the file now lands.

commit 0e4a8c495ade1dbf236df112b33eb9b577f540d9
Author: moritztng
Date:   Fri Aug 14 01:39:38 2026 +0000

    spike(fslice): add the separable-2D variant to the FSC gate
    
    Section 31 left three options and two of them need measurements rather than a decision. This adds the
    one that can collapse the question: separable-2D, the same z-collapse followed by a TRUE 2D bilinear
    instead of two 1D passes. Section 15.3 measured it at 1.16-1.21x trilinears error against the two-pass
    1.23-1.27x but never costed it, treating it only as the familys floor.
    
    The pairing is exact. Swapping which separable arm runs does not change the rng consumption order, so
    the trilinear reference reproduces bit-identically: box 96 seed 11 gives tri 4.7100 A under both
    variants, with sep 4.7715 (two-pass, +0.0615) against 4.7580 (sep2d, +0.0484). A 21% smaller loss,
    which tracks the 15.3 error ratios closely -- predicted 0.74, measured 0.787.
    
    If that carries to box 256 the two-pass 0.0589 A becomes roughly 0.046 A and the gate is met by the
    cheaper family rather than by falling back to exact trilinear.

commit 813a694ba0f9fdca7db691d95f6a98cea72a32ea
Author: moritztng
Date:   Fri Aug 14 01:37:08 2026 +0000

    spike(fslice): the accuracy gate FAILS at box 256 -- separable costs 0.0589 A, gate is 0.05 A
    
    Section 8s kill gate, the only one that could turn the design down on accuracy: more than 0.05 A of
    lost resolution at box 256 is a NO-GO for the separable form. Measured at the stated box, n=12 seeds,
    SNR 0.05, 400 orientations, paired noise, symmetric pipelines:
    
      deltas +0.0465 +0.0483 +0.0492 +0.0495 +0.0506 +0.0564 +0.0568 +0.0625 +0.0642 +0.0659 +0.0764 +0.0809
      mean +0.05894 A   sd 0.01132   sem 0.00327   95% CI [+0.0517, +0.0661]
      8 of 12 seeds individually above the gate; the CI lies ENTIRELY above it
    
    GATE FAILED. The separable two-pass interpolant this kernel implements loses about 0.059 A of
    resolution against RELION trilinear at box 256, and the bar was 0.05 A.
    
    Getting here needed the sample size. n=4 gave mean +0.0597 with a CI straddling the gate, and n=8 gave
    +0.0602 with a lower bound of +0.0494 -- a hair under. Only at n=12 does the interval clear it. A
    single seed would have said FAIL at box 96 (+0.0795), PASS at box 128 (+0.0454) and FAIL at box 256
    (+0.0564), which is three different verdicts from the same experiment.
    
    What this does NOT settle is the consequence. Section 8s fallback is section 4.2s exact-trilinear
    diagonal-matmul variant, rejected there at 19x the compute -- a judgement made when the kernel was
    assumed compute-bound. Section 27.2 measured it at 84.6% of its TRAFFIC limit with the arithmetic
    nearly free, so that 19x would be substantially absorbed rather than paid. There is also an untested
    middle option: the separable-2D form of section 15.3 was 1.16-1.21x trilinears error against the
    two-pass 1.23-1.27x, and its cost was never measured. Re-pricing those is a design decision.

commit 936636568c4ecd87898e7ce976d03f520b0b53cf
Author: moritztng
Date:   Fri Aug 14 00:56:22 2026 +0000

    spike(fslice): batch the FSC scatter so box 256 is reachable
    
    np.bincount allocates and zeroes its output on every call, and the output is P^3 -- 1.07 GB at box 256,
    three times per arm per orientation, 400 orientations. That allocation, not the arithmetic, was the
    cost: box 96 ran at 0.5 s per orientation for only 174k scatter operations.
    
    Buffering the indices and values and flushing every 25 orientations amortises it 25-fold. Reproduces
    box 96 seed 11 EXACTLY -- tri 4.7100 A, sep 4.7715 A, delta +0.0615 A, identical to four digits before
    and after -- so this is a speed change and not a numerical one.
    
    The FSC shell array is also built in float32 rather than float64, which halves another P^3 allocation.

commit 6cb7d2789769121646d237c56fb7c8a182342ac7
Author: moritztng
Date:   Fri Aug 14 00:53:45 2026 +0000

    spike(fslice): FSC gate -- the separable interpolant costs 0.045-0.080 A, and the cost SHRINKS with box
    
    Section 8s kill gate, owed since section 15.3: more than 0.05 A of lost resolution at box 256 is a
    NO-GO for the separable form. Measured, paired noise, symmetric pipelines (trilinear projected AND
    backprojected against separable projected AND backprojected), SNR 0.05, 400 orientations:
    
      box  96  n=4  mean +0.0795 A  sd 0.0288   deltas +0.0615 +0.1205 +0.0576 +0.0782   above the gate
      box 128  n=2  mean +0.0454 A  sd 0.0036   deltas +0.0480 +0.0429                   below the gate
    
    The delta is consistently POSITIVE -- the separable form does lose resolution, as section 15.3s
    1.23-1.27x error ratio implied -- but it SHRINKS with box size, 0.0795 at 96 to 0.0454 at 128. My
    assumption going in was that it would be scale-invariant, because resolution in Angstrom is N*apix/shell
    and the padded lattice is a fixed factor 2 in every box. That was wrong, and it is exactly why section 8
    specified the gate at box 256 rather than at any convenient box.
    
    So the gate is NOT failed. At box 128 it already passes, and the trend runs the right way. But box 256
    is the stated gate and it is NOT measured, so the gate is not passed either.
    
    Also recorded: the first version of this experiment was noiseless and returned the Nyquist shell for
    both arms, discriminating nothing. FSC 0.143 marks where signal falls below noise; without noise there
    is no crossing to compare.

commit 1a71ce5b1aeadde2907a7083b5fd1a4919ac08c5
Author: moritztng
Date:   Fri Aug 14 00:23:39 2026 +0000

    spike(fslice): S2 part 2 harness -- FSC 0.143 resolution in Angstroms, projection and backprojection
    
    The accuracy gate owed since section 15.3. Builds a padded Fourier volume from Gaussian atoms, projects
    N orientations with each interpolant, backprojects with the SAME interpolant, reconstructs by dividing
    out the accumulated weights, and reports the FSC 0.143 crossing against the true volume in Angstroms.
    
    Two design points that are not garnish.
    
    Symmetric pipelines. RELION projects AND backprojects with trilinear, so its interpolation errors
    partially cancel; comparing a separable projection against a trilinear backprojection would be a
    different and unfair experiment. The arms are trilinear/trilinear against separable/separable.
    
    Noise is part of the experiment. FSC 0.143 marks where signal falls below noise. Run noiseless, both
    pipelines recover the volume essentially exactly and the FSC never crosses 0.143 -- the first run
    returned the Nyquist shell for both arms and discriminated nothing. Both arms get the SAME noise
    realisation, because independent draws would bury the difference between interpolants under the
    difference between noise fields.
