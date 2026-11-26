Archive of tt-bio branch wk/relion-kernel-particle-fft at 64f21d4c0ce79fd1cee212fb43caaa63feffa4c1, forked from tt-bio main at eb9757fc8dad965832d4e2ff0f0effb40706d9c4.
Files are the ones this branch added or changed; the rest of tt-bio is unchanged.

commit 64f21d4c0ce79fd1cee212fb43caaa63feffa4c1
Author: moritztng
Date:   Sat Aug 15 09:46:49 2026 +0000

    relion(nonZeroProb): bit-exact 54-56x direction loop, as a standalone RELION patch
    
    selectOrientationsWithNonZeroPriorProbability is 968.0 of the 1074.0 CPU-s inside
    getFourierTransformsAndCtfs (90.1%), ~42.7 s of a 922 s refinement. It scans 49,344
    grid directions to keep ~118, recomputing a loop-invariant Euler transform and heap
    allocating ~16 matrix temporaries per direction.
    
    integrations/relion/relion5-nonzeroprob.patch caches the grid unit vectors and the
    symmetry products M_j = L_j R_j^T once per sampling change and reduces the inner loop
    to nsym+1 dot products on doubles. 607.8 -> 10.8 ns/direction.
    
    Bit-identical rather than close. The fast path is gated on every M_j being an exact
    signed permutation matrix, which is what makes it the same sequence of IEEE operations
    instead of an algebraically equal rearrangement, and the dots accumulate from 0 in
    dotProduct() order. Any other symmetry group falls through to the original loop, as do
    isRelax, bimodal psi, the multi-body (0,90) prior and non-3D samplings.
    
    Verified elementwise against the UNPATCHED librelion_lib.a over 14.8M direction
    evaluations at orders 6 and 7: 0 index mismatches, 0 prior-value mismatches, max rel
    diff 0.000e+00. Patch compiles clean and git apply --check passes on pristine
    src/healpix_sampling.{h,cpp} at e4a4aad.
    
    Kept out of relion5-tt.patch deliberately: it speeds up every backend, so any benchmark
    applies it to both arms or neither. The paired-refinement grade is NOT yet run.

commit cadc9a2cf687fa47be9a5a7c97f89338eba216e4
Author: moritztng
Date:   Sat Aug 15 09:35:15 2026 +0000

    relion(fft): independent reproduction of the nonZeroProb screen -- 67.9-69.0x, still bit-exact
    
    Rebuilt nzp_screen.cpp from source against the same librelion_lib.a and re-ran both
    orders. 0 index mismatches, 0 prior-value mismatches, max rel diff 0.000e+00 over the
    same 14.8M direction evaluations. Speedup lands at 67.87x (order 6) and 68.98x
    (order 7), inside the 67-72x the first pass reported.
    
    Also re-derived the profile attribution straight from e2e_stages.json cpu_s rather
    than the summary table: nonZeroProb 968.01 of getFourierTransformsAndCtfs 1073.95
    CPU-s (90.14%), the two FFTs 34.59 (3.22%), nine children summing to 1069.78 with a
    0.39% residue. The NO-GO stands.

commit 179eea578f6a35d90ba537142216a3a0401bf7b7
Author: moritztng
Date:   Sat Aug 15 09:30:55 2026 +0000

    relion(fft): NO-GO -- the FFT is 1.53 s of 922 s, and 90% of the region it sits in is a host grid scan that is 67-72x too slow
    
    RELION own CTIC/CTOC tree, already collected by relion-end-to-end and re-read this
    pass, puts 968.01 of getFourierTransformsAndCtfs 1073.95 cumulative CPU-seconds in
    nonZeroProb. The two Fourier transforms are 34.59 CPU-s, 1.53 s of the refinement,
    68x below the 11.2% whole-refinement noise floor. The Python crossing alone costs
    3.0-6.1 s, so hooking the FFT makes the refinement slower.
    
    What owns the time is HealpixSampling::selectOrientationsWithNonZeroPriorProbability,
    which scans 12,384-49,344 grid directions per particle to keep ~118, recomputing a
    loop-invariant Euler transform and heap-allocating ~16 matrix temporaries per
    direction. nzp_screen links librelion_lib.a and calls the real function: 67-72x
    faster with an identical index list and bit-identical prior values, 0 mismatches
    over 14.8M direction evaluations. Projected saving 42.1 s, 4.6% of the refinement.
    
    Also corrects relion-end-to-end $5: the seam DOES exist (CUDA runs this FFT on the
    GPU via cufftExecR2C), and the 51.0 s floor is really 93.7 s because nonZeroProb was
    counted on the accelerable side.

commit 92feead671f8607e602937b14436da6a870107af
Author: moritztng
Date:   Sat Aug 15 09:04:31 2026 +0000

    relion(sws): the screen says NO-GO -- the ceiling is 1.095x and both halves lose
    
    storeWeightedSums is 80.1 s of a 922.19 s refinement, so a free one is worth
    1.095x. The screen prices both halves before building anything, and the model
    reproduces the coarse leg is own 50 s and 30 s per iteration to within 0.5%.
    
    wavg is a projection, not a scatter: it calls the same project3Dmodel the coarse
    pass does and writes densely. Priced with the coarse kernel is measured 420 ns/pair
    it is 79.1 s against RELION is 38.2 s CPU, 2.07x slower; the best point in the whole
    design space is parity at 38.5 s.
    
    backproject is a 24-way read-modify-write into a shared volume and never reads the
    model. Against this card is measured 128 B rmw roof of 80.1 GB/s it is 643.1 s
    against 41.9 s CPU, 15.3x slower; 114.5 s with the destination held in L1; 57.2 s
    destination-stationary. The no-scatter form from relion-backprojection does not
    transfer -- wrong interpolant (+0.0589 A) and the exact trilinear has 0.061 touches
    per voxel per orientation, so there is no reuse to fix a destination tile around.
    
    Settles end-to-end section 5 is open assumption: the device/CPU ratio for this term
    is 0.92 at best and 2.42 today, 13-35x its assumed 0.0688, and its 154.7 s / 5.96x
    row becomes 488-608 s / 1.52-1.89x. The fine pass is a wash at 1.04x too.
    
    Hands forward one uncomfortable open item: RELION is own CPU coarse kernel costs
    273.5 ns per (orientation,pixel) pair against its own wavg kernel is 63.1 ns for the
    same single trilinear projection, and ~2.6x of that 4.3x is unexplained. If it is a
    fixable CPU inefficiency the program is only win shrinks with it.
    
    Shape from 22,260 live calls, roofs from projprobe/b0_roofs.json, rates from
    relion-kernel-coarse-projection 8.15/8.17. Nothing built, nothing run on the card.

commit 2b739c454a3aab2be1c858a74d75d15252920eef
Author: moritztng
Date:   Sat Aug 15 08:42:17 2026 +0000

    relion(coarse): final artifacts and the carry-forward lessons, every arm re-verified bit-exact
    
    All five graded arms re-run on the final tree: p1_dense 160/160 tiles, p1_gather 8/8 slots, p1_blend,
    p2_compare (accumulator and folded diff2), p3_fused -- every one at max abs error 0.000000e+00 with
    its non-vacuity guard passing. The concluding claim rests on that run, not on results from earlier
    passes.
    
    §11 now lists the whole artifact set and marks compute_p3_pipe.cpp as the incomplete NaN-producing
    attempt so nobody picks it up by mistake. It also states plainly that the ttnn backend in
    tt_bio/cryoem/relion.py still declines -- the device path is not wired.
    
    §12 separates the generalisable findings from this leg's own outcome: only the SFPU holds fp32 and
    only with unpack_to_dest, per-CB; a NoC transfer needs 16 B at both ends and fails silently two ways,
    wrong data at the source end and a device hang at the destination end that tt-smi -r is needed to
    clear; a read costs 36.7-37.5 ns from 16 B to 256 B so the gather is issue-bound; a scratch CB's
    depth must equal exactly what one pass pushes; composing separately-measured parts underestimated the
    assembled kernel by 1.60x and the first diagnosis of the gap was wrong; and the fp32 primitive cost
    table.

commit 8b296732f9363faf6471b2f19be3d52c44bc92ae
Author: moritztng
Date:   Sat Aug 15 08:40:23 2026 +0000

    relion(coarse): NO-GO for leg 3 as scoped, and this leg's own refinement gate is explicitly not reached
    
    §9 delivers the leg-3 verdict the brief asks for. NO-GO for relion-kernel-storeweightedsums as
    currently scoped -- not because a device storeWeightedSums cannot be built, but because leg 3 is
    scoped against relion-end-to-end §5's extrapolated row and three measurements here say that row
    cannot be met.
    
    The alignment tax is structural and a scatter meets it in the worse direction. §8.9 found a NoC
    transfer needs 16 B alignment at both ends: a misaligned source silently returns the neighbouring
    block, a misaligned destination hangs the device. A gather's data-dependent address is its source, so
    this leg's failure mode was wrong data, which a parity check catches. A scatter's data-dependent
    address is its DESTINATION, so the same misalignment hangs the card, with no diagnostic, and killing
    the process does not clear it.
    
    The workaround is not cheap and §8.16 shows it is not removable: 153-170 ns/pair, ~36% of the whole
    kernel, purely to split one aligned 16 B transfer across two tiles. A scatter needs the identical
    split in reverse, and no model relayout removes it. On top of relion-backprojection-kernel-result's
    16-17%-of-floor prior, that does not produce §5's row.
    
    §9.1 states plainly that this leg's own deliverable is NOT complete. Done: the design, the
    kernel-level result at 0.000000e+00 on every stage, and a measured 420.0 ns/pair with full cost
    attribution. Not done: the full-refinement parity table, the measured device-vs-CPU ratio and
    refinement wall, and the ttnn hook -- relion.py still declines. P5's 8-13 s band is REFUTED on
    extrapolation; P4, P9 and the refinement gates are ungraded and no resolution digit exists.

commit 1fd030a5772d6cd8f03b41b34366b1520ef41de9
Author: moritztng
Date:   Sat Aug 15 08:38:56 2026 +0000

    relion(coarse): the relayout is worth 1.20x not 2.4x, and the pointer copies are structural
    
    I claimed last pass that an x-parity relayout would remove the 170.6 ns/pair of pointer copies as well
    as halving the reads. Re-deriving it before measuring caught the error: one 16 B read still spans
    corners x0 and x0+1, which belong to two different slot tiles 4 kB apart, and a NoC read writes
    contiguous bytes. The copies survive any relayout.
    
    Measured with an ablation issuing one aligned read per x-pair, which is what x-parity buys: 420.0
    ns/pair today, 350.5 with one read per x-pair, 197.0 with one read and no copies. So the relayout is
    1.20x for a doubling of model memory to 63.4 MB and ~488 kB/core, a far weaker case than I sketched,
    and it should be judged on that number.
    
    The pointer copies are structural rather than an implementation wart. They are the price of the 16 B
    alignment rule meeting a layout where an x-pair's two corners must land in different tiles. Removing
    them means interleaving corner pairs within one tile so a read lands contiguously, which destroys the
    element-alignment the blend depends on -- the whole reason §8.7's layout exists. At ~36% of the wall
    that is a hard floor for this design on this silicon.
    
    Scaling the single-core figure the way §8.7 did: ~50 s/iteration as verified today, ~42 with the
    relayout, ~30 with a second gather RISC. Against RELION's own coarse pass at 127 s/iteration that is
    roughly 2.5-4x, not the 13.5x the brief's 9.37 s target implies. P5's 8-13 s band is REFUTED on
    current evidence.
    
    Every one of those rows is an extrapolation from a single-core arm and none is a measured iteration
    wall. The measured number is 420.0 ns/pair, bit-exact, one core, one gather RISC.

commit e1404b4e6c4b67b15914ab09fa66419c6096112c
Author: moritztng
Date:   Sat Aug 15 08:36:49 2026 +0000

    relion(coarse): ablation shows the alignment workaround costs 170.6 ns/pair, not nothing -- and a 10.8% fix
    
    Instrumented instead of guessing. Compile-time ablation flags in the reader, each arm deliberately
    wrong, measuring cost attribution only: full 470.9 ns/pair, no zero-fill 435.4, no NoC reads 287.9,
    no parity-select copies 300.3, none of the three 81.3. So zero-fill 35.5, gather reads 183.0, pointer
    copies 170.6, everything else including all compute 81.3. Those sum to 470.4 against a measured
    470.9, i.e. additive -- nothing overlaps.
    
    Three corrections, two to my own earlier claims in this doc.
    
    §8.9 said the parity select was free, "four word moves". It is 170.6 ns/pair, 36% of the wall, nearly
    as much as the NoC reads. The alignment workaround, not the gather, is the second-biggest cost.
    
    §8.14 attributed ~145 ns/pair to compute/gather serialisation. All compute together is 81.3, so
    perfect overlap could save at most 17%, which is why the pipelining attempt returned 6% -- it was
    chasing headroom that does not exist.
    
    And the gather reads are CHEAPER than the composition predicted, 183.0 against 293.6. E4's per-read
    cost held; what blew the budget was the 206.1 ns/pair of workaround the alignment rule forced.
    
    One fix landed and verified. The per-block zero-fill only has to run while the CB slots are still
    uninitialised L1 -- after that a skipped pair's stale value is a finite number the radius mask
    multiplies out. Hoisting it out of the block loop: 470.7 -> 419.8 ns/pair, 10.8%, still bit-exact at
    0.000000e+00.
    
    It also changes the case for the model relayout. §4.7 argued it purely as a gather-rate win, but an
    x-parity relayout keeping x0 always even would make each 16 B read land directly in the slot tiles
    with no scratch and no copies, removing the 170.6 as well as halving the reads. On these numbers that
    is the largest lever in the kernel, and it is a different argument from the one §10 item 7 rejected.

commit d4f9431533c738319c4a295a0492e2f5954dcd04
Author: moritztng
Date:   Sat Aug 15 08:33:26 2026 +0000

    relion(coarse): the pipelining fix does NOT work -- 6% and wrong, so the stall diagnosis is unconfirmed
    
    §8.14 attributed 145 of the 177 ns/pair gap between the assembled kernel and its composed estimate to
    the compute and gather units ping-ponging, and named a one-block lookahead as the fix. Implemented:
    dense(b+1) before blend(b), cb_s1 at 70 tiles so two blocks stay live.
    
    It does not work. 442.8 ns/pair against the straightforward fusion's 470.7, i.e. 6%, and the output
    is NaN with a non-zero fraction of 0.914 against 1.000, so the variant is incorrect and its number is
    not a rate for anything.
    
    Both halves of that matter. Even wrong, it moved the wall by 6% where the hypothesis predicted
    recovering ~145 ns/pair, so the attribution of the gap to compute/gather serialisation is NOT
    confirmed and must not be repeated as if it were. Either the lookahead is not overlapping the units
    or the gap is somewhere else.
    
    The verified kernel remains compute_p3_fused.cpp at 470.7 ns/pair, bit-exact. compute_p3_pipe.cpp is
    committed as an unfinished attempt and the doc marks it unusable.
    
    Next step is instrumentation rather than another optimisation guess: find where the time actually
    goes among the per-block zero-fill of eight slot tiles, the parity-select pointer copies, the per-pair
    sentinel test and the overlap. Only the first two have been costed even roughly.

commit 33f5b76ffd2a0c35940b5434028bd99f3f6639d6
Author: moritztng
Date:   Sat Aug 15 08:30:11 2026 +0000

    relion(coarse): the assembled kernel measures 470.7 ns/pair, 1.60x what composing its parts predicts
    
    Every rate this leg has quoted for the fused pipeline was a composition of separately-measured parts:
    E4b's gather rate times a read count, plus an SFPU op budget. perf-method-floor-screen-predict-then-
    build exists because that is where predictions go wrong. Timing the real program -- the same code
    §8.13 graded bit-exact -- over 200 blocks and 102,400 pairs, best of five: 470.7 ns per pair against a
    composed 293.6, a ratio of 1.60x. The arm stays bit-exact while being timed, so this is the cost of
    the working kernel and not of a broken one.
    
    Caveats, all in the direction of being pessimistic about the production design: one core, one
    dataflow RISC issuing the gather where §4.2 uses both at a measured 1.97x, the model in the reading
    core's own L1 rather than a neighbour's shard, and no compare stage.
    
    Where the 177 ns goes. About 32 ns/pair is RISC-side non-NoC work the composition ignored: the
    per-block zero-fill of eight slot tiles and the parity-select pointer copies, 16 word-writes per pair
    each. The remaining ~145 is the compute stage not overlapping the gather -- the compute kernel pushes
    the address tile then immediately blocks on cb_slot, so the two units ping-pong per block.
    
    Deepening the CBs does not fix it: cb_addr 2->4 and cb_slot 8->16 moved the wall from 470.7 to 470.8,
    i.e. not at all. The serialisation is structural, since the compute kernel cannot produce block b+1's
    addresses until it has finished blending block b. Fixing it needs software pipelining with a one-block
    lookahead, dense(b+1) before blend(b), at the cost of a second block of dense scratch.
    
    §8.7's ~18.6 s/iteration assumed the composed 150.2 ns/pair. If the 1.60x gap survives to the
    dual-RISC configuration the figure is nearer ~30 s; if pipelining recovers the 145 it lands near the
    original. Both are extrapolations from a single-core arm and neither is a result.

commit d2e0cf348c2be296e8bce5f991302ed4ced02049
Author: moritztng
Date:   Sat Aug 15 08:27:49 2026 +0000

    relion(coarse): the three projection stages run fused in one program, bit-exact
    
    Phase 3. Coordinate stage, gather and blend as ONE program, max abs error 0.000000e+00, non-zero
    fraction 1.000 on both sides.
    
    What this arm proves is the dataflow direction rather than the arithmetic, which was already graded
    stage by stage. §4.4 requires the addresses to come from the compute unit rather than the reader --
    if the reader recomputed floor(xp) itself, one ulp of disagreement would fetch a different cell and
    blend it with the right weights. So within each block the program runs compute -> reader -> compute:
    the compute kernel packs an address tile into a CB, the reader waits on it, gathers eight corners
    from its L1-resident model, pushes them back, and the compute kernel blends them. That inverts the
    usual direction and it runs without deadlocking.
    
    Two structural points the fusion forced. Two scratch CBs rather than one, because the dense chain
    pushes 35 intermediates and the blend pushes 31 and §8.8's rule is that a scratch CB's depth must
    equal exactly what one pass pushes. And the reader both feeds and gathers, because generic_op gives
    three kernels and the inverted dataflow needs the gather on a dataflow RISC.
    
    Limits, stated rather than glossed. The geometry is reduced -- a 16x16x16 model, 32 kB, so it fits
    one core's L1 -- because a single core cannot hold RELION's real 31.68 MB model; §4.1's 130-core
    sharding is what makes that work in production. And because every pixel of the small lattice lies
    inside the radius, the sentinel and radius-mask paths are NOT exercised here. They are exercised in
    §8.9 on their own.

commit b9106aa89656a316e94292fa87f0fb4bdcb67abe
Author: moritztng
Date:   Sat Aug 15 08:24:05 2026 +0000

    relion(coarse): the final column fold is not a concession -- §4.1 already required an off-core step
    
    Closing out the one item that was blocking Phase 2 from being assembled. Folding the accumulator's 32
    columns into one number per orientation cannot be done by an element-wise op, and reduce_tile
    truncates, so it has to happen somewhere else. Two routes, and the choice is now clear.
    
    On the dataflow RISC the fold is structurally easy: in the face layout a logical row's 32 columns are
    two contiguous runs of 16 words, one in each of two faces, so it is 32 rows x 32 additions per tile
    with plain L1 pointer arithmetic and no alignment constraint. But whether that is cheap depends on
    whether the dataflow RISC has hardware floating point, which is NOT measured. If those additions are
    software-emulated the fold is far more expensive than the host route, so this option must not be
    chosen without measuring it first.
    
    On the host it is 54 tiles of 4 kB per call, 1.2 GB per rank per iteration over PCIe -- and this is
    not an extra concession, because §4.1 already requires a cross-core reduction: 130 cores each produce
    a partial diff2 over their own pixels and those partials have to be summed off the core anyway. The
    column fold rides along with a step the design always had.

commit bcfeca7e7242220e2fb18ee61c099c554e9484ee
Author: moritztng
Date:   Sat Aug 15 08:23:17 2026 +0000

    relion(coarse): the compare and the pixel accumulation are bit-exact, so the whole chain is on device
    
    Phase 2. RELION's diff2 body for one orientation block and one translation: accumulator over pixel
    blocks max abs 0.000000e+00, folded diff2 max abs 0.000000e+00, non-zero fraction 1.000 on both
    sides.
    
    The paired-column layout pays off a third time. The real and imaginary differences are the even and
    odd columns of one tile, so their squares are a single tile op, and summing over both columns IS the
    |.|^2 the formula asks for -- the component sum falls out of the layout instead of costing an op. The
    whole compare is four SFPU ops per pixel block: d = ref - sh, d2 = d * d, wd = d2 * w, acc = acc + wd.
    The accumulation across pixel blocks is element-wise, so it stays on the exact set.
    
    Every stage of the coarse kernel now runs on device bit-exact except the final column fold: the
    coordinate stage, the gather, the blend, the compare and the accumulation, each graded against numpy
    at exactly zero.
    
    What is left is folding 32 columns into one number per orientation. No element-wise op moves data
    across columns and reduce_tile truncates, but the quantity is 6 orientation blocks x 9 translations
    per call and there are two cheap ways to finish it -- on the host at 1.2 GB per rank per iteration,
    or on the dataflow RISC with plain L1 pointer arithmetic, which has no alignment constraint because
    it never touches the NoC. That makes it a choice, not a blocker.
    
    Corrected cost note: the compare runs once per translation, so at 4 ops per pixel block and 9
    translations it is ~11 ns per pair, not the ~4 §8.7 estimated. Total compute lands near ~45 ns per
    pair against the 150.2 ns gather, still gather-bound by better than 3x.

commit 8de5e428b249dce9dbcf10c7ba54f4e9958ef0aa
Author: moritztng
Date:   Sat Aug 15 08:20:33 2026 +0000

    relion(coarse): the read cost is flat to 256 B, which screens the compare-stage broadcast
    
    E4's own harness with the chunk sweep extended past its original 8-64 B range, and with 8 B dropped
    since §8.9 showed it cannot be used: 37.28 ns at 16 B, 37.47 at 32, 37.34 at 64, 37.45 at 128, 36.87
    at 256. Flat to within 1.6% across a 16x range in transfer size, so §8.5's issue-bound conclusion
    holds well past the range it was drawn from. The run also reconfirms E4b's dual-RISC 18.93 ns per
    read at 1.97x one RISC.
    
    That screens the first of §8.7's two unmeasured estimates. The compare-stage broadcast materialises a
    dense tile by 32 reads of the same 128 B row, and a 128 B read is now measured rather than
    extrapolated: 32 x 37.45 = 1.20 us per dense tile, and per core per call 9 translations x ~5 pixel
    blocks x 2 components = 90 tiles = 108 us, i.e. 0.60 s per iteration per rank against §8.7's ~0.6
    estimate. Confirmed.
    
    The second estimate needs no arm: the host-side horizontal finish is 54 tiles of 4 kB per call and
    1.2 GB per rank per iteration over PCIe, a fraction of a second at any plausible bandwidth. It is
    only needed at all because reduce_tile goes through SrcA and truncates like everything else there.
    
    The 256 B row also prices the §4.7 / §10-item-7 relayout: at 32 corners per read the gather would be
    1.14 s per iteration. That does not answer the capacity objection -- 253 MB against ~190 MB of usable
    L1, tighter now that §8.9 has made L1 residency a correctness requirement -- but it prices the prize.

commit 10cc9dcad3d3bfb0cb1d449c8b7949c94aa19cc8
Author: moritztng
Date:   Sat Aug 15 08:18:48 2026 +0000

    relion(coarse): the blend is bit-exact, and Phase 1 is complete
    
    RELION's seven-lerp cascade in its own association order -- a + (b - a) * f, not the algebraically
    equal a*(1-f) + b*f, because the two differ in the last bits and §6's gate is a residual against
    RELION's own answer. Max abs error 0.000000e+00, max relative 0.000000e+00, non-zero fraction 0.743
    on both sides against a 75% in-mask rate, so the comparison is not vacuous.
    
    The paired-column layout does the complex bookkeeping for free. Each pixel owns two adjacent columns,
    re and im, so the same weight multiplies both and the seven lerps are ordinary tile ops with no
    notion of complexity in them. Only the Friedel conjugate distinguishes the columns, and that is
    sgn_odd = 1 + (sgn - 1) * oddmask against a constant parity tile: three SFPU ops for the whole tile,
    not per element.
    
    So all three Phase 1 stages -- the coordinate stage, the gather and the blend -- are each bit-exact
    against numpy on the exact primitive set. 26 SFPU ops per 512-pair tile for the blend, in line with
    §8.7's ~7.4 ns per pair.
    
    One more self-inflicted trap. The blend arm first looked like it hung the device and it was the
    WRITER, not the compute kernel: it reused Phase 1(i)'s writer, which drains six tiles per block,
    while the blend emits one, so it waited forever on tiles that were never coming. A dataflow kernel
    whose per-block count disagrees with the compute kernel's output is indistinguishable from a kernel
    hang. writer_p1_out.cpp now takes the count as a compile-time arg so no stage needs its own.

commit b079bebec5a49e1baf93adffa42cf5f8ec77a6d7
Author: moritztng
Date:   Sat Aug 15 08:05:59 2026 +0000

    relion(coarse): the aligned gather is bit-exact on all eight corners, and the parity select is free
    
    Phase 1(ii) passes. Eight of eight corners at max abs error 0.000000e+00 against a numpy gather on a
    diagnostic model whose every voxel's value is its index, 512 pairs per block over 3 blocks, arbitrary
    index parity -- the case the 8 B version got wrong -- with 25% deliberately sentinel.
    
    Guarded against a vacuous pass: a gather that wrote nothing would match an all-zero expectation and
    report bit-exact, so both sides are asserted substantially non-zero first. 0.735 and 0.735, against a
    75% non-sentinel rate.
    
    The design came out cheaper than §8.9 predicted. Each x-pair reads the two aligned 16 B blocks that
    can contain (x0, x0+1) into a scratch, and the reader copies the wanted 8 B out with plain pointer
    writes, which have no alignment constraint because they never touch the NoC. The parity select
    therefore happens on the dataflow RISC for four word moves, and the compute side needs no parity mask
    and no select at all. §8.7's compute budget stands unchanged.
    
    Two things that cost this pass real time. A misaligned-destination hang WEDGES the card and killing
    the process does not clear it: after the stack-array version hung, every later run hung too,
    including a correct kernel, and two cycles went into concluding "the kernel still hangs" when the
    kernel was fine and the card was not. tt-smi -r 0 is required after any hang, not optional. And the
    scratch buffer has to be a CB rather than a stack array -- a stack array is 4 B aligned and sits
    wherever the compiler puts it, while a CB write pointer is a real tile-aligned L1 address.

commit c4a0f82b0b53b9fd4b24bfb7b1662a67fdab0e62
Author: moritztng
Date:   Sat Aug 15 07:54:18 2026 +0000

    relion(coarse): a misaligned NoC destination HANGS the device, where a misaligned source only lies
    
    Second half of the alignment finding. The rebuilt reader stages each aligned 16 B block into a
    scratch and copies the wanted 8 B out with plain pointer writes, which carry no alignment constraint
    because they never touch the NoC. Declaring that scratch as a plain stack array -- 4 B aligned --
    hung the device instead of returning wrong data.
    
    So a misaligned NoC transfer has two distinct failure modes and neither produces a diagnostic:
    silently wrong at the source end, hang at the destination end. Marking the scratch
    __attribute__((aligned(16))) is the fix.
    
    Status: the aligned reader still does not complete. After the alignment fix it neither returned nor
    produced output within 300 s, so something in it still hangs and it has NOT been graded. The stray
    process was killed by explicit pid and the card is clean. What is on the branch is this incomplete
    aligned reader, committed for continuity, not a working stage and not a source of any number.
    
    Prime suspect is the barrier-per-x-pair structure: 4,096 noc_async_read_barrier calls per block,
    against a barrier that waits on every outstanding read on the RISC. It is also the wrong shape for
    the assembled kernel. Batch the reads per pair, barrier once, re-grade.

commit 5486647721152988f638e366e1231538e348500e
Author: moritztng
Date:   Sat Aug 15 07:38:55 2026 +0000

    relion(coarse): a NoC read needs 16 B alignment at both ends and fails SILENTLY without it
    
    Root cause of §8.9's wrong gather, and it invalidates an inherited number.
    
    Bisected with a diagnostic model whose every voxel's value is its index, so a wrong result names the
    voxel actually fetched. Every error was exactly +/-1 voxel, which ruled out index recovery and
    face-order bookkeeping and pointed at the read. Forcing both ends to 16 B alignment -- even voxel
    index, even pair index -- makes the four even-offset corners (0, 8, 64, 72) bit-exact at 0.000000e+00
    while the four odd-offset corners (1, 9, 65, 73) stay off by exactly 1.
    
    So an 8 B noc_async_read at an address that is 8 B but not 16 B aligned silently returns the
    neighbouring aligned block. No error, no assert, no hang. The destination has the same requirement:
    in the first run, pairs at destination offset 8k with odd k were also off by one.
    
    That invalidates the 8 B row of §8.5. E4's chunk sweep timed a 32x8B arm and recorded nonzero=true
    but never checked WHICH bytes came back -- it was a timing instrument. Those reads were not fetching
    what they addressed, so 150.2 ns per pair for "8 corners x 8 B" prices an operation that cannot be
    performed, and the ~18.6 s/iteration built on it does not stand. The 16, 32 and 64 B rows are
    unaffected, being naturally aligned.
    
    The design survives, by coincidence. Getting corners x0 and x0+1 with 16 B alignment means fetching
    the two aligned blocks that can contain them and selecting on x0 & 1: 2 reads per x-pair, 4 x-pairs
    per cell, 8 reads per pair -- the same count §8.7 assumed, and at §8.5's flat 36.7 ns the same
    150.2 ns per pair. The cost moves to the compute side as a parity select per x-pair, which is not yet
    costed.
    
    The reader on this branch is the 8 B version and is known-wrong. It is committed for its diagnostic
    value, not as a working stage.

commit e2b8e8ae3f77a5e2d3256d6097a386ed788a0182
Author: moritztng
Date:   Sat Aug 15 07:33:28 2026 +0000

    relion(coarse): the gather reader runs but is not yet correct -- WIP, no number from it counts
    
    Phase 1(ii). The reader recovers the voxel index from Phase 1(i)'s biased float with one AND, skips
    the sentinel, and issues eight 8 B NoC reads at RELION's corner offsets into eight dense slot tiles.
    Graded against a numpy gather on a small synthetic model with a quarter of the pairs sentinel.
    
    It compiles, runs, and is wrong: values come back at the right magnitude, about 2.2 against model
    values of std 0.5, but at the wrong positions. So the gather works and the indexing between the
    address tile, the slot tiles and the host expectation does not line up. Nothing from this arm is
    quotable and the doc says so.
    
    Two real bugs found and fixed on the way. An INTERLEAVED L1 tensor is not addressable by raw byte
    offset -- passing buffer_address() and reading with get_noc_addr(local) returned garbage around 1e35,
    because an interleaved tensor is spread page-wise across many cores' L1 banks. The gather needs
    byte-granular random access, so the model must be staged into the reading core's own L1 first. That
    is §4.1's sharded-residency design reached from the other direction, and it makes model residency a
    correctness requirement rather than only a performance one. Second, a ROW_MAJOR page is one row and
    not one tile, so reading a (1,1,32,32) tensor with the tile's 4096 B page size moved 128 B and left
    the rest stale; shaping the model (1,1,1,1024) makes the page the whole array.
    
    Open: the residual position mismatch. The suspect is the face-order convention connecting the address
    tile, the linear offsets the reader writes at, and the host's faces_to_tile -- all three should
    cancel and one does not.

commit d5436cb4ab874ade0a23ecec8140bb87c071b858
Author: moritztng
Date:   Sat Aug 15 07:29:26 2026 +0000

    relion(coarse): a voxel is complex, so the gather layout needs paired columns -- still bit-exact
    
    Correction to my own §8.7 item 1, found while building the gather. The 8 B read fetches one complex
    voxel, re and im contiguous, not one number, and one NoC read cannot send those to two different
    dense slot tiles. As first written the design would have needed 4 B reads, 16 per pair, which at
    §8.5's flat 36.7 ns per read is about 300 ns per pair rather than 150.2 -- twice the quoted figure
    and about 37 s per iteration.
    
    The fix keeps the 8 B read and costs nothing: each pixel owns two adjacent columns, re and im, so one
    gather lands inside a single slot tile. Every per-pair quantity then comes out already replicated
    across the pair's two columns, which is what the blend wants -- the weights multiply re and im alike
    and only the Friedel sign distinguishes them, two SFPU ops against a constant parity mask. The
    compare gets the sum over the component axis for free, because summing both columns is |.|^2.
    
    The price is 16 pixels per tile instead of 32, so per-pair compute doubles to about 36 ns. Against a
    150.2 ns gather that is still gather-bound by 4x and §8.7's ~18.6 s per iteration is unchanged.
    
    Phase 1(i) re-run under the new layout: still 0.000000e+00 on all six outputs, 160/160 bit-exact
    tiles. The change is verified, not assumed.

commit 2239a4ae363ccb03646d34b6d82ef2848d373c25
Author: moritztng
Date:   Sat Aug 15 07:24:53 2026 +0000

    relion(coarse): RELION arithmetic runs on a Tenstorrent device, and every output is bit-exact
    
    Phase 1(i). The dense coordinate stage -- three rotated coordinates, the radius mask, the Friedel
    sign, three floors, three fractions and the packed voxel address -- on RELION's own dumped inputs,
    built entirely from the primitive set §8.4 measured exact. Graded per output against numpy fp32
    running RELION's arithmetic in the kernel's own order, over 5 orientation blocks x 32 pixel blocks =
    160 tiles, 163,840 pairs: max abs error 0.000000e+00 on all six outputs, 160/160 bit-exact tiles.
    
    Zero, not a residual. P3 pre-registered max relative <= 1e-6 for this stage and it clears by being
    exactly equal. That is what §8.4 buys: on the SFPU set under unpack_to_dest there is nothing to trade,
    so the only open question in the numeric path is whether the algebra is right.
    
    This is the first RELION arithmetic on a Tenstorrent device. Everything before it in this lineage was
    host torch behind the bridge or a different interpolant.
    
    Three things worth keeping. RELION's trunc(r2) <= maxR2_padded is exactly r2 < maxR2_padded + 1 for a
    non-negative sum, so one unary_lt_tile replaces trunc plus compare, saving 287 ns per tile and being
    bit-identical rather than close. The scratch CB's depth has to equal the chain length exactly: the
    block pushes 35 intermediates and pops none until it ends, and at depth 40 the first block was exact
    and every later one wrong, straddling a partial wrap. And §4.4 says "byte address" but means voxel
    index -- its own 2^23 bound only holds for the index, so the kernel emits that and the reader shifts.

commit 980de05240537a157727451e85d8b4ef7363f5d5
Author: moritztng
Date:   Sat Aug 15 07:17:51 2026 +0000

    relion(coarse): a design that is buildable from the exact primitive set, at ~18.6 s per iteration
    
    §8.7, proposed and not built. The observation that makes it work is that both broadcasts the kernel
    needs can be moved onto the reader, where a NoC read writes to whatever L1 address it likes and
    nothing passes through SrcA.
    
    The blend broadcast disappears into the layout: gather each of the 8 corners with its own 8 B read
    into its own dense [orientation, pixel] tile, and every blend operand is element-aligned with the
    dense coordinate stage. No transpose, no broadcast, no [pair, slot] layout. 49 SFPU binaries per 1024
    pairs, about 7.4 ns per pair.
    
    The compare broadcast the reader materialises: 32 reads of the same 128 B row fill one dense tile,
    2,880 reads per core per call at 36.7 ns, about 0.6 s per iteration per rank, reused across all 6
    orientation blocks.
    
    The pixel sum never touches the FPU: accumulate across pixel blocks with add_binary_tile, which is
    exact and element-wise, then pack the 6x9 [orientation, 32] partials out and finish the horizontal
    sum on the host, about 0.24 s per iteration over PCIe.
    
    Gather 150.2 ns per pair against about 18 ns of compute, so gather-bound by 8x, which is the
    comfortable direction. Roughly 18.6 s per iteration against P5's 8-13 band, so P5 would be REFUTED
    and the leg would still be worth about 7x on the term that is 78.2% of the particle loop. The cost
    against §4.3 is the 16 B read shape and its 2x gather advantage, which is what not having a
    precision-safe broadcast costs.
    
    Two numbers in it are estimates and are labelled as such: the reader-side row replication and the
    host-side horizontal finish.

commit 4a8b16fd8f9f6efa68425f886282de5506714c67
Author: moritztng
Date:   Sat Aug 15 07:16:27 2026 +0000

    relion(coarse): no precision-safe broadcast exists on this hardware, and the kernel's shape needs two
    
    bcast.h carries a unary broadcast whose source reads like the missing primitive -- it picks its own
    path per operand, with the comment "32bit formats are implemented using unpack to dest, since SrcB is
    only 19bits wide", which is §8.4's root cause in the vendor's own words. E8k measured it and it is
    not exact: 5.787e-4 for ROW and 7.197e-4 for COL with the knob off, and all-zero output with it on.
    The off column is small enough to confirm the semantics are the ones the design wants, so the probe
    is right and the values are simply truncated like everything else on that path.
    
    So the exact-fp32 primitive set is: SFPU DST-to-DST binaries at 155.8 ns, SFPU unaries, and
    copy_tile. All strictly element-wise on aligned tiles. Every layout-changing primitive -- the binary
    bcast family, unary_bcast, matmul_tiles, reduce_tile -- goes through SrcA/SrcB and truncates.
    
    The kernel needs a broadcast twice by its shape, not by its implementation: per-pair quantities
    against gathered corners in the blend, and the orientation x translation outer product in the
    compare. §8.5's dense-slot-tile layout removes the first one completely and brings the blend to about
    7.4 ns per pair, because dense work amortises over 1024 pairs instead of 32. It does nothing for the
    second.
    
    The reduce arms in e8l are a probe bug and are labelled as such in the doc, not quoted as a result:
    reduce_tile takes a scaler CB and the probe passed the data CB, so it reduced against arbitrary
    multipliers. Whether the fp32-enforced reduce is exact is still open and worth answering, since E8h
    priced it at 61.60 ns against the ordinary reduce's 61.53.

commit 21d422f6fd20bf912791a5ac01e11156252d4334
Author: moritztng
Date:   Sat Aug 15 07:11:50 2026 +0000

    relion(coarse): the gather costs 36.7 ns per read whatever its size, which prices every layout option
    
    E4 recorded a chunk sweep and no document in this lineage quoted it. A noc_async_read costs
    36.66-37.03 ns at 8, 16, 32 and 64 B alike: the gather is bound by the rate the RISC issues reads,
    not by bytes. So per-pair gather cost is set purely by corners per read -- 8 B one corner 150.2 ns
    per pair dual-RISC, 16 B two corners 75.8 (the §4.2 design), 32 B four corners 37.8 (§4.7's held-back
    lever), 64 B eight corners 18.8.
    
    That prices the escape from §8.4's broadcast problem. Give each of the 8 corners its own 8 B read
    into its own dense [orientation, pixel] slot tile and every blend operand is element-aligned with the
    dense stage: no transpose, no broadcast, no [pair, slot] layout, every op a plain SFPU binary on
    matching tiles. The gather doubles to 150.2 ns per pair and the kernel is gather-bound again at
    roughly 18.5 s per iteration.
    
    The same table says §4.7's lever is worth more than §4.7 credits, and that a third option sits above
    it at 18.8 ns per pair. §10 item 7's L1 arithmetic against that one is unchanged, but part of its
    reasoning was that the compute floor made the extra 2x worthless, and the compute floor has moved.

commit 816f928ded17dc031b0bb83a3bba6f5cdac304aa
Author: moritztng
Date:   Sat Aug 15 07:10:34 2026 +0000

    relion(coarse): fp32 is bit-exact on the SFPU and nowhere else, and that unbuilds §4.3's central primitive
    
    E8g sweeps unpack_to_dest_mode off and on. On, the SFPU copy_tile pass-through, mul_binary_tile and
    add_binary_tile all come back at exactly 0.000e+00 against torch fp32 -- zero, not small. §6's 1e-5
    gate is reachable. The FPU arms go to inf on the same setting, which is the same fact from the other
    side: the knob bypasses SrcA, and SrcA is the only thing an FPU binary op reads. Off, every op sits
    at 7e-4 to 2e-3 because the operand is truncated to about 11 mantissa bits on the way in.
    
    E8i shows the mode is per-CB: set on an index nothing reads, every row is unchanged. One kernel can
    run precision-critical math on unpack_to_dest CBs and keep FPU ops on the others.
    
    E8h prices the SFPU in the configuration it will run in. A binary op is 155.8 ns, 2.56 mul_tiles,
    not the 4.71 the unary rounding ops cost. reduce_tile with enforce_fp32_accumulation costs the same
    as the ordinary one, 61.60 against 61.53, so it is free if it is accurate -- not measured yet.
    mul_tiles re-measures at 60.90 ns, third independent time.
    
    Re-costing §4.3 on the SFPU: 118 ns per pair against a 75.8 ns gather budget, compute-bound by 1.6x
    instead of gather-bound with 1.9x of margin, so roughly 15 s per iteration against P5's 8-13 band
    and the brief's 9.37 target. Still far under RELION's 127 s.
    
    The harder half is structural. §4.3's per-gather-tile stage rests on mul_tiles_bcast_rows and its
    bcast_row_idx argument, which §4.3 itself calls the primitive that makes the design work. The bcast
    family is FPU-only, there is no SFPU row broadcast, and an FPU broadcast multiply's 2.05e-3 lands
    directly on the blended reference value and therefore on diff2. The layout reconciliation has to be
    re-architected before the kernel can be written.

commit d6891fd5446e8fbf648a091704f5ce2ab0781bef
Author: moritztng
Date:   Sat Aug 15 07:04:11 2026 +0000

    relion(coarse): P1 clears at 1.0036, and an fp32 tile op on this silicon carries 7e-4, not 1e-7
    
    Phase 0a closes. Three walls on one binary and one box: ref (no bridge) 911.74 s, null (crossing
    plus the device path's buffer traffic) 915.00 s, nullbase (crossing only) 917.80 s. P1 pre-registered
    null/ref <= 1.15 and it lands at 1.0036. null beating nullbase, which does strictly less work, is the
    honest reading: all three sit inside a 6.06 s spread, 0.66%, and neither the crossing nor the
    marshalling is resolvable above it. Counters checked on every arm before its number was used --
    handled=0 declined=5580 per rank. The bridge is not the wall.
    
    Phase 0b turns up something the design has to answer for. E8d asked whether eltwise binary
    accumulates into DST: mul_tiles does, add_tiles does not, matmul_tiles does. The two eltwise ops
    differing is not in any document in this lineage.
    
    E8d's residuals were also 2e-3 to 4e-3 against a 1e-5 gate, so E8e swept one op per fidelity. An
    fp32 add_tiles is 6.99e-4 at every fidelity, and a+a is exact arithmetic in any format, so the adder
    is not the problem -- the operand is truncated on the way in, at about 11 mantissa bits, the SrcA
    register width. MathFidelity buys back mantissa on multiplies only.
    
    E8f tried the SFPU and its pass-through control settles where the loss is: copy_tile in, pack out, no
    arithmetic, comes back at 6.99e-4 too. So the fp32 value never reaches DST under the config these
    probes ran, and the SFPU arithmetic numbers measure the same truncation.
    
    Not yet established: whether this is the silicon or the probe's compute config.
    ttnn.ComputeConfigDescriptor exposes unpack_to_dest_mode and none of E8d/E8e/E8f set it. That is the
    next measurement and the pass-through control is the instrument -- it has to go to zero. No
    feasibility conclusion before then.

commit d29d2ed45fdf38d30753efd498a1ef9ff3c767df
Author: moritztng
Date:   Sat Aug 15 06:42:38 2026 +0000

    relion(coarse): the bridge crossing costs 0.7%, and the shape arm's +59% was its own instrument
    
    nullbase enters Python on every coarse and every fine call and declines without touching a buffer:
    917.80 s against the bridge-free ref arm's 911.74 s. 44,352 crossings cost 6.06 s of wall, 0.66%,
    at or below what one pair of runs resolves. Counters clear the no-op trap in the direction this arm
    needs -- handled=0 declined=5544 on every rank, so every crossing happened and nothing computed
    behind it.
    
    That retires an inherited number. relion-kernel-diff2-fine reports the shape arm at 1,469 s against
    a 922.19 s reference and attributes the gap to 22,260 declined round-trips. The round-trips are
    worth 6 s; the other ~550 s was the shape instrument, a file append and an np.unique over the
    significant-index array per fine call. The fine doc says in words that the shape arm is not a timing
    arm. This is the measurement behind the words.
    
    The consequence for this leg: RELION spends 0.169 s per coarse call and the round-trip costs three
    orders of magnitude less, so a device kernel is free to be called per particle.

commit acc2ac7dbabc65722365161fa9994f97f8f4fc7b
Author: moritztng
Date:   Sat Aug 15 06:35:32 2026 +0000

    relion(coarse): the reference wall reproduces at 911.74 s, and the coarse pass is 78.2% of the particle loop
    
    The ref arm is the first bridge-free wall this lineage has measured on the box it is optimising
    rather than inherited from relion-end-to-end §4. It lands at 911.74 s against that document's
    922.19 s, 1.1% apart, on a fresh build directory and a different day. Every speedup this leg quotes
    now has a control measured under its own conditions.
    
    Bridge-free is verified, not assumed: with no PYTHONPATH, TTBridge::ensureReady fails its import
    once per rank and g_usable stays false, so RELION runs its own kernels with zero Python crossings.
    The log shows exactly four "cannot import" lines and no counter line.
    
    RELION's own Timer puts getAllSquaredDifferencesCoarse at 3,775.38 of 4,829.86 thread-seconds,
    78.2% of the particle loop, above the 68.8% the wall-share in §5 implies. storeWeightedSums 9.24%,
    maximization 9.19%, getAllSquaredDifferencesFine 7.05%.
    
    One arm-design correction found by running it: --iter 13 does not stop the run at iteration 13.
    Continuing from it012, every arm runs it13-it17 to convergence, so these are full-refinement walls
    at ~15 min per arm, not single-iteration walls at ~4.

commit 4716d52bf91cfffb5918e312ec9fc1df148b7082
Author: moritztng
Date:   Sat Aug 15 06:09:03 2026 +0000

    relion(coarse): every FPU primitive the op budget spends costs 0.41-1.58 mul_tiles, and a multiply costs 2.4x an add
    
    E8 priced one primitive. E8c prices the other nine the design leans on, in the fp32/HiFi4/fp32-acc
    configuration the parity gate forces. P2 pre-registered each at <= 3 mul_tiles and every FPU one
    lands well inside that: transpose_wh_tile 0.44, add_tiles_bcast_rows 0.41, matmul_tiles 0.82,
    reduce_sum_col 1.01, mul_tiles_bcast_rows 1.00, mul_tiles_bcast_cols 1.02, reduce_sum_row 1.58.
    
    Two things the budget did not know. A multiply is 61.1 ns and an add or a datacopy is 25-27, so
    counting mixed arithmetic in undifferentiated tile ops overstates it; and matmul_tiles does 32x the
    MACs of mul_tiles for 0.82x the cost. SFPU stays expensive at 4.7-5.1 mul_tiles, with trunc_tile
    (286.8 ns) 20% cheaper than the floor_tile E8 measured.
    
    Also adds the nullbase/null backends and the coarse campaign, so Phase 0a can separate the bridge
    crossing from its marshalling against a bridge-free reference wall on the same binary.

commit af995a17bf495b9ceda871ff5b6641ec95e90fda
Author: moritztng
Date:   Sat Aug 15 05:55:50 2026 +0000

    relion(coarse): the free tile-op budget is the same in fp32 as in bf16, and an SFPU floor costs 5.9 FPU ops
    
    E4c measured the blend's headroom with bfloat16 CBs at HiFi4 and concluded the arithmetic hides under
    the gather with 14.6x to spare. The coarse kernel cannot run in bfloat16 -- its gate is a 1e-5
    relative residual and bf16 carries 8 mantissa bits -- so the budget had to be restated in float32.
    
    fp32 HiFi4 mul_tiles: 61.10 ns per tile. bf16 HiFi4: 60.88 ns. 1.004x, so float32 costs nothing on
    the FPU. LoFi is 3.7x cheaper in bf16 and 2.5x in fp32, and is unusable here for the same mantissa
    reason.
    
    floor_tile on the SFPU: 357.9 ns, flat across both formats and both fidelities, 5.86x an FPU tile op.
    That is the number that decides the layout. Three floors and three fracs per coordinate block cost
    2.1 us, which at the 32 pixel-orientations per tile the gather layout naturally gives is 67 ns per
    pair against a 75.8 ns gather budget -- the coordinate math has to run at full tile utilisation
    instead.

commit 3648a09a213d25b3db328be164481768f893277a
Author: moritztng
Date:   Sat Aug 15 05:30:13 2026 +0000

    relion(fine): the tt arm passes every parity gate, and the thread-count fix a 5-process proxy priced at 34x made the real iteration slower
    
    Both hooks live for a whole refinement: 22,260/22,260 calls handled, 0 declined, 3.79378 A
    unmasked, 4.033896 A after relion_postprocess, and assignments one particle CLOSER to the
    reference than the arm that computes nothing of ours.
    
    Cost, from RELION's own Timer against the declining control: coarse 4.99x slower than its CPU
    kernel, fine 34.6x. An isolated five-process proxy blamed torch intra-op oversubscription and
    priced the fix at 33.9x, but the confirming in-refinement run did not finish it13 in 24:57
    against the tt arm's 20:34, so the proxy inverted the ranking it was built to decide. The
    34.6x stands and the proxy is recorded as a dead end.
    
    The whole-tensor form is also the wrong shape for the fine pass: it densifies to all 36
    translations, and compare is ~66% of our fine call rather than the 4.4% section 3.1 assumed by
    transferring the coarse pass's projection share.

commit 9427b6bfa3fceb11766ec0d4c75d15d73afbcacf
Author: moritztng
Date:   Sat Aug 15 02:30:02 2026 +0000

    relion(fine): the check arm is the coarse-only control the leg needed, and the no-op trap can be cleared mid-run
    
    The check arm ran the coarse hook live for a whole refinement (g_check only gates the fine path)
    and reached the reference resolution both unmasked and after relion_postprocess. Its assignments
    disagree with the reference on the same 24 of 4452 particles the shape arm does, and the shape arm
    computes nothing.
    
    harvest_counters.py reads TTBridge s counters out of a running process, so an arm that is silently
    declining is caught in minutes instead of hours. report() only prints at exit.

commit 32d0d33e826e276a8e0d41bc474aae4c98be61fa
Author: moritztng
Date:   Sat Aug 15 00:56:48 2026 +0000

    relion(fine): report() was never called, so every arm was throwing its own evidence away
    
    TTBridge::report() is declared in tt_bridge.h, defined in tt_bridge.cpp and invoked from nowhere.
    Two things went missing because of it. The no-op trap the parity gate depends on, "diff2Fine
    handled=N declined=0 with N > 0 before the numbers count", has never printed in any arm; the shape
    arm passed it on the TT_RELION_SHAPE dumps instead, which is luck, not the gate. And under
    TT_RELION_CHECK the whole residual accumulates into file-static counters that die with the process,
    so the check arm computes the kernel-level parity number for hours and then discards it.
    
    std::atexit at the end of a successful init, and the residual also flushes every 500k graded
    entries: a refinement is hours long, an aborting rank runs no atexit handler, and the number should
    be readable while the arm is still running rather than only after it.
    
    The number the running check arm had already accumulated was not lost. It was read out of the four
    live follower ranks through /proc/PID/mem against the PIE load base, harvest_residual.py, and it is
    in results/check_residual.json: 4,375,232 fine entries graded against RELION own kernel over
    it13-16, max relative residual 1.070e-06, median in the 1e-7 to 1e-6 decade, 13.196 percent
    bit-identical. That grades P10: the max clause passes with 9.3x of margin, the median clause is
    refuted by one decade, and one decade above float32 epsilon on a 19,404-term reduction is what a
    different summation order costs, not a defect.
    
    Also here: fine_disagree.py, which grades this leg arms with the e2e leg grader byte for byte by
    pointing its module global at relion-scratch/fine, so the two legs tables stay comparable without
    overwriting the e2e leg artifact. Its first result is the shape arm as a control, and it matters:
    with both hooks declining, RELION against itself scores 4428/4452 on _rlnAngleRot at p99 0.000000,
    against the e2e leg bridge arm 4427/4452. The two dozen disagreeing particles the e2e leg attributed
    to the bridge are RELION own run-to-run nondeterminism.

commit b3544dd96aa4ca89e2552a0ef6f96b8e79756ff7
Author: moritztng
Date:   Fri Aug 14 22:45:35 2026 +0000

    relion(fine): the shape arm measures what the plan had to assume -- the orientation count triples at it14 and the fine call is 3.3x smaller than the coarse one, not 15x
    
    22,260 fine calls, 4,452 particles x 5 iterations, none missed. translation_num is 36
    everywhere and image_size is 19,404 through it16 and 33,024 at Nyquist, both as predicted.
    The orientation count is not: median 16 at it13 and 48 from it14 on, so a scheme picked from
    the first iteration is wrong by 3x on the orientation axis.
    
    Three facts hold on every one of the 22,260 calls: job_num_count is exactly
    significant_num/4, so RELION's chunk limit of 7 never binds; distinct rot_idx equals
    orientation_num, so there are no dead orientations and the design gives up less than §3.1
    thought; and orientation_num is always a multiple of 8.
    
    The whole-tensor form projects 1.373x less than RELION's own fine kernel. P6 predicted the
    fine call was ~15x smaller than the coarse one and it is 3.3x, and on the compare axis it is
    1.19x larger, so the per-call fixed-cost argument keeps its sign and loses its margin.

commit 6be6d0842745aae2b9bef9da40f96aa51e5c5f81
Author: moritztng
Date:   Fri Aug 14 19:38:06 2026 +0000

    relion(fine): let the campaign wait longer than an hour for the box
    
    relion-production-scale-arm holds benchlock with a timed ladder and the box is at loadavg 21 from a
    third worker's boltz fold, so the hardcoded 3600 s would have exited 75 without running an arm.
    BENCHLOCK_WAIT_S now falls back to 3600 instead of overriding it.

commit a448ec115adc470d1d7060ea649856e81879ad41
Author: moritztng
Date:   Fri Aug 14 19:36:38 2026 +0000

    relion(fine): the fine hook compiles, and the campaign gets its own tt-bio clone
    
    Phase 1 of the exec work order. The two inserts land in runDiff2KernelFine: the bridge call at the
    top of the body, and the TT_RELION_CHECK grader at the end, both behind !do_CC && mdlZ != 0 &&
    !data_is_3D, both falling through to RELION's own kernel on a decline. relion5-tt.patch regenerated
    from the working tree, which is how it was made.
    
    Both changed translation units pass -fsyntax-only under the exact flags build-e2e uses -- tt_bridge.cpp
    and cpu_helper_functions.cpp, which is the TU that carries the hook. Worth the two minutes: the arms
    queue behind another worker's benchlock, so a typo would have surfaced hours later as BUILD_FAIL.
    
    The campaign points PYTHONPATH at a new clone, relion-scratch/tt-bio-fine, instead of the shared
    relion-scratch/tt-bio. That clone sits on wk/relion-end-to-end and relion-production-scale-arm is
    running against it right now; moving it to this branch would have changed the Python under a live
    timed ladder.

commit 3f320560cae22e81648d1f676c875b7a999c87fa
Author: moritztng
Date:   Fri Aug 14 19:28:50 2026 +0000

    relion(fine): the fine pass is the coarse pass plus a gather, and the job list carries no information
    
    RELION writes diff2_fine_2D as a loop over jobs -- one orientation, a run of at most 7 consecutive
    translations -- and re-projects the reference once per job. Transcribing makeJobsForDiff2Fine and
    driving it with 4000 random significance masks shows the jobs tile [0, significant_num) exactly, so
    rot_idx/trans_idx already name every entry once and the job arrays are CUDA block geometry. Project
    each distinct orientation once, read the named entries out of the dense matrix: same answer, and
    1.06-6.00x less projection work depending on how dense the coarse significance mask is.
    
    So tt_bio/cryoem/relion.py gets one _dense_diff2 core and two entry points. The coarse path's
    numerics move into the helper statement for statement, because it is parity-gated. diff2_fine is
    bit-identical to the coarse matrix indexed at rot_idx/trans_idx plus sum_init on identical inputs,
    max delta 0.0, and it accumulates rather than assigns.
    
    TTBridge gets diff2Fine on the same discipline as diff2Coarse, with two differences: a separate
    kill switch, so a fine-side raise cannot silently move the coarse arm back to CPU mid-refinement
    and make the parity result uninterpretable, and an optional resolve, so an older tt_bio without the
    entry point still runs the coarse hook. TT_RELION_CHECK=1 makes diff2Fine compute into a private
    buffer and decline, so RELION's own kernel writes diff2s and diff2FineCheck grades one against the
    other -- RELION's real kernel on RELION's real inputs, no transcription and no separate driver.
    
    Not compiled into RELION yet. The RELION-side hook, the three arms and the shape characterisation
    are in state/relion-kernel-diff2-fine.md.

