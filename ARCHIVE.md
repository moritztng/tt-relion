Archive of tt-bio branch wk/relion-projection-optimize at e6e71e1b5815531be3f90715b59c79e6cf2cc4d3, forked from tt-bio main at 8fa729612f483e7ff5eadc93fc16821d1b980b87.
Files are the ones this branch added or changed; the rest of tt-bio is unchanged.

commit e6e71e1b5815531be3f90715b59c79e6cf2cc4d3
Author: moritztng
Date:   Fri Aug 14 14:11:00 2026 +0200

    spike(fslice): the per-arm JSON for the twelve arms measured on pc card 0
    
    One file per arm, each carrying its own wall, sha256, parity residual and the mode/fid/dscale that
    produced it, so the box sweep and the fidelity ladder can be re-read without the logs.

commit 48fec0299017392fdfa31e5469c2e2dc7bffb166
Author: moritztng
Date:   Fri Aug 14 14:07:25 2026 +0200

    spike(fslice): lever B measured at 1.25x on every box, and the parity gate was blind to every lever
    
    The arm batch re-homed on pc card 0 (bh-host's benchlock never freed). Roofs re-measured here first:
    402.2 / 175.3 / 297.4 GB/s, reproducing bh-host's to 1.1%, so the 1.0145 M floor is this card's.
    
    Lever B (mode 14) lands 519.4 k slices/s at box 256 against a baseline of 416.0 k, 1.25x, and holds
    1.25x at 384 and 512 with the fraction of floor box-independent to 0.4% (51.20 / 51.23 / 51.32%).
    Assembly issue explains 419.7 of the 478.5 ns saved; the TRISC third returns only half its standalone
    cost because it was already partly hidden behind NCRISC, which is also why cbtil 8 measures zero.
    
    parity() never took the mode or the fidelity under test, so every --fid and --mode arm printed the
    mode-13/HiFi4 residual and four lever arms shared one byte-identical parity line. With it plumbed
    through, HiFi2 is 3.13x worse against fp64 and LoFi 10.4x, for 1.13x and 1.14x -- lever F is a
    perf win with an accuracy price, not a free one.
    
    Mode 14 has no one-tile parity arm and is no longer silently downgraded: the reader holds one
    bo[]/ri[] for every block, so mode 13 and mode 14 read the identical window and the sha gate could
    never have certified either.

commit 1f00565d5d21f6e0c694254bfd7d3f14b3f11126
Author: moritztng
Date:   Fri Aug 14 11:04:15 2026 +0000

    spike(fslice): the ten-arm batch, one benchlock acquisition
    
    bh-host benchlock has been held continuously by relion-acc-backend is RELION refine timing campaign, so
    ten separate lock acquisitions would have been ten separate 15-minute waits. One script takes the
    lock once and measures every arm, ordered safest-first because a generic_op deadlock costs a card
    reset: mode 14 changes the reader/compute handshake and goes last.

commit 062ce7591d9a5a64afedcd76a19c8c0ca0cc7867
Author: moritztng
Date:   Fri Aug 14 11:03:07 2026 +0000

    spike(fslice): mode 14 -- pass 1 once per intermediate, and the predicted landing before it runs
    
    Lever B built. Mode 13 runs pass 1 TWICE per output tile because pass 2 resamples the intermediate
    along its r axis and needs a 64-row window of it. Those windows overlap down a tile-column: the next
    output tile wants the window shifted by 32*D rows, and a 2-pass separable rotation forces A*D == 1,
    so D = 1/cos(theta) is in [1, 1.414] and the shift is 32 to 46 rows. One or two of the two window
    tiles are therefore already computed.
    
    Mode 14 keeps the window as the front of cb_int2. Pass 2 always reads tiles 0 and 1, tiles that fall
    out are popped, and only the shortfall is produced. The window start is a Q16 accumulator in mid-rows
    so its tile index is acc >> 21, and the host derives the reader's assembly count with the identical
    integer arithmetic -- the reader has to issue exactly that many or the program deadlocks.
    
    Nothing in mode 14 changes any arithmetic, so with the harness's fixed shear the intermediates are
    the values mode 13 computes and the output must be bit-identical. That is the gate: sha256 unchanged
    against arm 0, or the window is mis-indexed.
    
    At nb = 1508 output tiles/core, run 5, D = 1.1228: 1810 assemblies against mode 13's 3016, r = 0.600
    (0.700 at the D = 1.414 worst case).
    
    PREDICTED LANDING, written before the arm runs, from the resource model of the state doc section 3:
      assemblies 1050.2 * 0.6001 = 630.2 ns/slice
      TRISC      450.0 * 2.2003/3 = 330.0 ns/slice   (compute blocks per output tile 3 -> 2.200)
      stage 1    835.2 ns/slice, unchanged -- lever D is dead, see 435044ae
      total      1795.4 ns/slice = 557.0 k slices/s = 56.5% of the 985.7 ns floor
      bracket at D = 1.414: 735.0 + 360.0 + 835.2 = 1930.2 ns = 518.1 k

commit 435044ae5fd3544a3fd3f9f98d5ad75b4a7a7a58
Author: moritztng
Date:   Fri Aug 14 10:56:55 2026 +0000

    spike(fslice): lever D is dead on tile granularity and lever B clears its gate at r = 0.664
    
    The two host screens section 6 of the plan put before any build, run on bh-host, no device.
    
    LEVER D -- NO-GO. Stage 1 computes the whole padded plane and a slice only samples the inscribed
    disc, so the area argument says 0.785. Tile granularity does not let you have it: at box 256 a
    16x16 grid over a disc of radius 256 in a 512-wide plane touches 0.875 of the tiles, and once the
    reader window dilation is counted (64 elements right of the leftmost sample of a row, +-1 row for
    the pass-2 tap) it is 0.9219. The pre-registered drop threshold was 0.9. Bigger boxes barely help,
    0.8819 at 384 and 0.8711 at 512, because the tile is a fixed 32 and the boundary is O(g).
    
    LEVER B -- GO, and the assumed D was right. Over the 96 psi grid with theta reduced into
    [-45, 45] degrees, D = 1/cos(theta) has mean 1.1228 (the plan assumed 1.13) and the reuse ratio
    r(T = 5.03, D) has mean 0.6636 against the 0.78 kill gate. That is 1.507x on the assemblies.
    
    fslice_e2e.py gains --mode, --fid, --cbtil, --cbsrc and --split so the step-2 screens and levers O
    and F can be measured in the integrated harness rather than a separate one, and the reader issues
    the same 2*nb assemblies whatever the mode so the arms stay comparable. Mode 13 with the defaults
    is byte-identical to before. FLOOR_SLICES_S is replaced by the corrected floor: 985.7 ns/slice at
    box 256 from roofs measured on this card, scaling exactly as box^2.

commit eb59349f85f5a9ca151e66e14dd9bfe388e3c21f
Author: moritztng
Date:   Fri Aug 14 10:35:21 2026 +0000

    spike(fslice): the read/write roofs re-measured on card 0 -- the write roof is 173.7 GB/s and the projection floor was 2.24x too high
    
    b0_roofs.py and roof_rw.cpp come from wk/relion-backprojection unchanged; this is an independent
    re-run on bh-host card 0 under benchlock, 130 cores, and the write arm still reads back 100.0% exactly
    1.0 so the writes landed.
    
      read  406.6 GB/s  (be16, 2048 B)   backprojection measured 404.5
      write 173.7 GB/s  (be16, 2048 B)   backprojection measured 173.5
      rmw   297.3 GB/s  (be16, 2048 B)   backprojection measured 293.7
    
    Two consequences for the forward. The published floors of 3.206 / 1.425 / 0.801 M slices/s are
    420.2 GB/s (ttnn.add, two reads and one write) used as a write roof, and a slice write is a write.
    And state/relion-projection-complete.md section 4 prices the slice write at 685 ns per 2048 B, which
    is the DRAM READ number from s1c_pipe.json (dram/1x2048B/be8 = 684.7); the write costs 1532.8 ns of
    core time at 130 cores, 2.24x more.

