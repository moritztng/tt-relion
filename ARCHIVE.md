Archive of tt-bio branch wk/relion-intercard-scaling at 20d995f28434aba56af1371612cea7d23c247e75, forked from tt-bio main at d1a4b2d4355728ec2637c3c0b065fce0b364c391.
Files are the ones this branch added or changed; the rest of tt-bio is unchanged.

commit 20d995f28434aba56af1371612cea7d23c247e75
Author: moritztng
Date:   Fri Aug 14 11:33:43 2026 +0000

    spike(intercard): power re-measured on a bounded sampler -- 135 W p50 per chip, 1.38 TFLOP/s/W, and umd/telemetry streams forever
    
    The first power sampling was taken with a `for i in $(seq 1 70); do telemetry; sleep 1; done` wrapper.
    umd/telemetry streams continuously instead of printing one snapshot, so the loop never advanced past
    its first invocation and kill on the wrapper left the streaming grandchild alive. Two of them outlived
    their windows and wrote 7.5 GB onto a host at 97% full. Killed by explicit pid, logs deleted, disk back
    to 35 G free. Use timeout on the tool itself.
    
    Re-measured with a bounded 60 s stream sampled 1-in-200: active chip TDP p50 135 W, p90 139, peak 157,
    AICLK p50 1031 MHz, temp 52.6 C mean. Agrees with the first parse to 2 W on p50 and p90, and the
    sustained throughput reproduced to 0.5% (188.2-188.9 TFLOP/s vs 188.0-189.1). Per watt 1.38 measured
    TFLOP/s/W on 4.32 kW of chip power against the H200 1.41 datasheet and ~1.13 achieved. Still a tie.
    
    The sampled log strides against the 32-chip cycle so it covers one active chip, not eight, which is why
    its peak is 157 W where the all-chip parse saw 180. Both recorded. Summary artifact committed as
    projprobe/intercard/telemetry_summary.json.

commit 2b64296685be289767fae08382077358633974f0
Author: moritztng
Date:   Fri Aug 14 11:27:20 2026 +0000

    spike(intercard): the composed trajectory is 109.6 ms on 8 chips, 98% of its own floor, and per watt it is a tie
    
    The trajectory was priced as 15 model evaluations summed, which is the per-call-census projection this
    lineage keeps getting wrong. Measured directly instead: all 15 iterations issued back to back in one
    timed region, two identical arms per card count. 63.323/63.275 ms at N=2, 94.704/94.504 at N=4,
    110.010/109.551 at N=8 -- A/A spread 0.08-0.42%, sha256 of the last reduced volume identical across
    arms. The composed wall beats the per-call model by 4.3-7.6% because the 0.444 ms fixed cost pipelines
    instead of being paid 15 times; drop it and pure bandwidth predicts within 1.2-2.4%. So the trajectory
    runs at 98% of its bandwidth-optimal floor.
    
    Per server on the composed rate: 117.4 ms reduction + 74.1 ms compute = 191.5 ms, 12.38x on 32 chips,
    38.7% of linear, crossover ~7,060 particles. Only extrapolation left is 8 chips to 16.
    
    The 15 exact CurrentImageSize values read out of run_it0NN_half1_model.star (32 76 80 128 144 146 146
    150 154 154 160 160 196 196 196) give 2.660 GB against the 2.661 assumed, so the collapsed band was
    harmless this time -- worth recording because the previous four instances of that pattern were wrong by
    1.4-4.5x.
    
    E7 under a real load: sustained bf16 matmul on 8 chips gives 179-189 TFLOP/s each (mean 185.8) against
    212.7 in a burst, at p50 133 W and AICLK mean 978 MHz with temps at 52-61 C, so it is a power/clock
    envelope and not thermal -- a candidate for the 9% per-core residual against a p150 300 W card. Per
    watt: 1.40 measured TFLOP/s/W (5.95 PFLOP/s over 4.26 kW of chip power) against the H200 1.41
    datasheet and ~1.13 achieved. A tie, and neither side includes host, board or memory power.
    
    Probes and raw JSON under projprobe/intercard/. Analysis in state/relion-intercard-scaling.md.
    Not merged.

commit 2484357f8055b0d579bdc23ad3696398be812bd1
Author: moritztng
Date:   Fri Aug 14 11:17:48 2026 +0000

    spike(intercard): the PCIe arm dies at its floor, and 0.67 GB/s was not the PCIe roof
    
    E4 measured as a floor rather than built: any host-mediated reduce pays one D2H plus one H2D per card,
    so if that loses there is nothing to build. One pinned process per card, 821 MB fp32, all released at a
    common wall-clock instant. Solo card, TILE layout: D2H 1278.8 ms, H2D 722.8 ms. ROW_MAJOR: D2H 1165.1,
    H2D 264.3. Eight concurrent cards on one root complex: each within 3% of solo, so no fan-in knee at 8
    and the bottleneck is per-process host work, not the root complex.
    
    The roof correction matters more than the result. One card in the 8-way run did H2D at 21.6 GB/s while
    the other seven sat at an identical 264.3 ms, which is a fixed-rate software path and not contention --
    probably the one hugepage-backed DMA buffer with IOMMU disabled, hypothesis untested. So the PCIe roof
    is 21.6 GB/s measured and 0.67 GB/s must never be quoted as PCIe bandwidth.
    
    Verdict NO-GO. At the measured roof a 32-card host round trip is 76 ms against 39.2 ms predicted for
    the entire 32-way ethernet all-reduce, before the host sums 26 GB; as shipped it is 1.43 s per card,
    36x worse. Section 4 predicted 4-8x worse: right in direction, low in magnitude.
    
    Probes and per-card JSON under projprobe/intercard/. Analysis in state/relion-intercard-scaling.md.
    Not merged.

commit c63648aa558383ec74f4b5ca7fa8c9e204efd47e
Author: moritztng
Date:   Fri Aug 14 11:12:26 2026 +0000

    spike(intercard): both inherited roofs measured on UBB -- DRAM holds at 405 GB/s, matmul is 212.7 not 254.5 -- and RELION own collective is 707.9 ms where ours is 16.5
    
    E3. One chip, ttnn.open_device, chips 0 and 16 (the two idle-clock groups) within 0.2% of each other,
    so the 800/1350 MHz idle split biased nothing. DRAM copy fp32 821 MB read+write: 404.9 and 405.7 GB/s,
    which confirms the ~400 GB/s roof every byte-model sanity check in this task leans on. bf16 matmul
    m=7744 k=13312 n=2304 LoFi: 212.4 and 212.7 TFLOP/s against the 254.5 inherited from a p150. Not a
    timing artifact -- 20 back-to-back calls in one timed region give the same number as single calls
    best-of-9, so dispatch is not inflating it. Mechanism, partially: the UBB chip exposes a 12x10 compute
    grid against the p150 13x10, which is 7.7% of the 16.4% gap; the remaining 9% is per-core and stays
    unattributed. A server is 6.81 PFLOP/s, not 8.14, so a DGX H200 leads 1.16x on datasheet peak rather
    than trailing by 3%.
    
    E6. RELION collective measured without renting a DGX: MPI_Allreduce, MPI_DOUBLE, MPI_SUM, at the
    packed 24 B/voxel volumes, OpenMPI 4.1.2 on the galaxy host (64 cores, loadavg 1.1) because bh-host was at
    21 of 32. At cs=196, 741.4 MB: 463.4 ms on 2 ranks, 707.9 on 8, 1063.0 on 16, 1984-2041 on 32. The
    comparable pair is 8 ranks against 8 chips for the same iteration: 707.9 ms vs 16.5 ms, 43x, of which
    2x is fp64-vs-fp32 bytes and 21x is the path. Roof check kills the headline: a 1-rank allreduce is a
    copy at 29.8 GB/s of memory traffic and the 2-rank collective moves 1.68 GB/s of payload, under 6% of
    that, so RELION reduction on one node is bound by OpenMPI default large-message allreduce, not by
    memory or fabric. A tuned MPI would close most of the 21x. Claim shipped as "structurally cheaper",
    not 43x.
    
    Full sweep run twice; the 32-rank 741 MB point reproduces within 2.8%.
    
    Probes, raw JSON and the MPI log under projprobe/intercard/. Analysis in
    state/relion-intercard-scaling.md. Not merged.

commit f9a67f18e69d691f3d2f5c9cc1c455ba8695f16d
Author: moritztng
Date:   Fri Aug 14 11:02:27 2026 +0000

    spike(intercard): the second ethernet link was idle, so the reduction is 42.1 GB/s not 21.96 -- and E0 warm reset took a link down
    
    ttnn.all_reduce defaults to num_links=1 and nothing in the first pass passed the argument. Measured
    A/B on devices 0,1 at 821 MB: 1 link 40.116 ms / 21.46 GB/s, 2 links 20.435 ms / 42.13 GB/s, 1.96x.
    num_links=4 is refused by the fabric (2 eth channels per neighbour), so 42.1 GB/s is the per-chip
    ceiling. Re-took the fanout curve at num_links=2: bus bandwidth flat 41.8-42.5 GB/s from 2 to 8
    chips, model t = 0.444 ms + 2(N-1)/N * S / 43.01 GB/s fits all six rows within 2.2%. DRAM side rises
    to 20.1% of its roof, still not binding.
    
    Per server recomputed: 196.8 ms, 12.04x on 32 chips, 37.6% of linear, crossover ~7,400 particles
    (was 307.5 ms / 7.71x / 24% / 14,000). NO-GO on the byte levers stands and is stronger: 1.88x now,
    and the all-reduce is already bf16-grade -- max relative error 5.4e-3 against a host fp32 reference,
    constant in size, so it is the datapath and not accumulation length.
    
    Power under the reduction: 39.2 W mean per active chip against 36 W idle, peak 74 W. A ring
    all-reduce is ~3 W/chip above idle, so there is no per-watt number without the compute arm.
    
    E0 regression: warm_reset cleared device 20 stalled ERISC heartbeat but took the 19-27 ethernet
    connection down, 256 -> 250 links UP, and a second reset did not restore it. Every mesh open spanning
    two groups of 8 now throws IndexError: unordered_map::at from Cluster::generate_cluster_descriptor.
    N=16 and N=32 are unmeasurable until the link returns. Intra-group 8-chip work is unaffected.
    
    Probes and raw JSON per arm under projprobe/intercard/. Analysis in state/relion-intercard-scaling.md.
    Not merged.

commit 8257421b4df10d7646207f7810315fe49f14a16c
Author: moritztng
Date:   Fri Aug 14 10:37:08 2026 +0000

    spike(intercard): the volume reduction is 21.96 GB/s of ring bus bandwidth, and it is 4.5x smaller than we priced it
    
    Blackhole UBB galaxy, 32 chips, all free by sudo lsof. ttnn.all_reduce over FABRIC_1D at the real
    RELION volume sizes: bus bandwidth flat at 21.6-22.0 GB/s from 2 to 8 chips while algorithmic
    bandwidth falls exactly as 2(N-1)/N. Bandwidth-bound ring, ethernet-bound: the DRAM side is at 10.5%
    of its roof and the fixed cost is 0.41 ms of a 39-69 ms wall. Model t = 0.41 ms + 2(N-1)/N * S /
    21.96 GB/s fits every measured row to within 3.1%.
    
    Separately: the 808.5 MB per-iteration reduction volume was the box-256 figure and this job's coarse
    pass never reaches 256. RELION sizes the backprojector from current_size, projector.cpp:59, so the
    trajectory sum is 2.661 GB per card, not 12.1 GB.
    
    Topology read out of system_health: 4 groups of 8, each a 2x4 mesh, groups in a 4-ring. A 32-chip
    Hamiltonian cycle exists and its prefixes are the right TT_VISIBLE_DEVICES sets for a fanout curve.
    Device 20's ERISC heartbeat is stalled, which blocks a 32-way mesh open until a warm reset.
    
    Probes and raw JSON under projprobe/intercard/. Analysis in state/relion-intercard-scaling.md.
    Not merged.

