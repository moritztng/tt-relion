Archive of tt-bio branch wk/relion-production-scale-arm at a25650d41801b73f6c8bed741c85cb8e9f80911e, forked from tt-bio main at eb9757fc8dad965832d4e2ff0f0effb40706d9c4.
Files are the ones this branch added or changed; the rest of tt-bio is unchanged.

commit a25650d41801b73f6c8bed741c85cb8e9f80911e
Author: moritztng
Date:   Fri Aug 14 22:02:47 2026 +0000

    relion production-scale arm: the x25 point lands, the coarse share is flat to 0.14 points over 25x
    
    ref25, 111,300 particles: wall 5538.57 s, expectation_6 5428.22 s, rc=0. Coarse share of
    oneParticle 90.78/90.92/90.72/90.90% across four follower ranks, mean 90.83%, against
    90.69% at 4,452. Twenty-five times the particles moves it by less than the rank-to-rank
    spread of the instrument itself.
    
    Composed one-iteration verdict at 111,300 on a Xeon-class host: Galaxy 32 chips coarse-only
    158.11 s against 548.21 s measured for 1x H200 (3.47x) and 310.65 s for 2x H200 (1.96x).
    Whole refinement, at the trajectory 78.2% coarse share: Galaxy 3381 s against 3626-5107 s
    for 1x H200, so the wall-clock verdict flips against one H200 and not against two.

commit e2a69512cd73dabb1aa8f92efed86417f796d613
Author: moritztng
Date:   Fri Aug 14 21:17:20 2026 +0000

    relion production-scale arm: bh-host stage-split ladder and the iteration-17 measurement
    
    The coarse pass share of RELION E-step is flat in particle count: 90.69% at 4,452,
    90.81% at 13,356, 90.81% at 35,616. relion-end-to-end §9 predicted it would rise
    toward 90.4% at production scale as the argument for a crossover; it was never a
    function of particle count.
    
    Iteration 17, the combined-halves pass, measured directly for the first time by
    continuing from ref_run_it016_optimiser.star: 76.16% coarse against iteration 13 90.7%.
    relion-end-to-end §4 could only estimate this iteration. It is what sets the
    trajectory 78.2%, and it is why the whole-refinement composition uses 78.2% not 90.7%.
    
    compose.py does the verdict arithmetic from measured terms only.

commit 58b37e5c5870bc3d07c8f5ef82cf7cbf29f0eeab
Author: moritztng
Date:   Fri Aug 14 19:16:22 2026 +0000

    relion(prod): hold benchlock once for the whole ladder, not once per point
    
    A slope is a difference between points, so acquiring the lock four times on a shared box bakes four
    different co-tenancy states into the one quantity that cannot survive them. One hold, ~90 min, every
    point still recording its own loadavg and CPU-seconds.

commit d1547b547e83b8f0723b438140758df694e64ef8
Author: moritztng
Date:   Fri Aug 14 19:13:53 2026 +0000

    relion(prod): the rented ladder is complete, and RELION on a Xeon is linear to 0.4 percent while the H200 is not
    
    Seven arms, four particle counts, all rc=0. The Xeon costs 12.079/12.232/12.125 ms per particle
    across 4,452 to 111,300; the H200 costs 3.954/4.734/4.871, rising 23 percent and not yet flat. End to
    end at 111,300 the H200 leads this Xeon 2.50x, not the 4.06x the 4,452-particle job advertised.
    Instance destroyed, 8.48 dollars, 0 instances running.

commit db5d40e06a363203c39c55872a069e035f1562b2
Author: moritztng
Date:   Fri Aug 14 18:41:49 2026 +0000

    relion(prod): the rented ladder, and the H200 box amortises with particle count too
    
    Four 1x H200 points from 4,452 to 111,300 particles, one 2x H200 point at the top, and the same box
    Xeon at the bottom two. 1 -> 2 GPUs is 1.765x at 111,300 against the 1.321x relion-end-to-end measured
    at 4,452, so the amortisation the crossover argument credits to the Galaxy fanout is not exclusive to
    it. Board power 130.8 W at 87 percent utilisation on the 111,300-particle run, unchanged from the
    132.0 W at 72 percent measured on the small job: still 19 percent of the 700 W datasheet TDP.

commit 843a0ed0dab8d38b7b67abe4821c2573c90e76bc
Author: moritztng
Date:   Fri Aug 14 18:13:01 2026 +0000

    relion(prod): scale the tutorial job to 111,300 particles and run the same iteration at four counts
    
    The crossover verdict in relion-end-to-end §9 rests on a measured ~7,060-particle threshold and a
    4,452-particle job. Deciding whether it survives production scale needs the scaling law, so this
    replicates RELION own it012 particle table 1x/3x/8x/25x and runs the SAME single iteration at each
    count on bh-host and on a rented 2x H200 box. One iteration, not a refinement: a refinement converges in
    a different number of iterations at each scale and its wall would confound more particles with more
    iterations.

