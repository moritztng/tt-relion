Archive of tt-bio branch wk/relion-kernel-diff2-fine at 3648a09a213d25b3db328be164481768f893277a, forked from tt-bio main at eb9757fc8dad965832d4e2ff0f0effb40706d9c4.
Files are the ones this branch added or changed; the rest of tt-bio is unchanged.

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

