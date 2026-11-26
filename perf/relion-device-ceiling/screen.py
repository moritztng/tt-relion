"""Ceiling screen: put storeWeightedSums (and the fine pass) on the device.

Every input is a measurement from an earlier pass, cited inline. Nothing here ran on a card.
The question: can the best composition this program has ever written reach per-dollar parity with
RELION CUDA on one H200, if storeWeightedSums costs ZERO on the device?
"""
XEON = 1.749            # EPYC(bh-host)-to-Xeon host correction, relion-end-to-end s7.3
H200_HONEST = 105.5     # 1x H200 wall with the nonZeroProb fix on both sides, full-device-deep-perf s8
GALAXY_HONEST = 88.8    # Galaxy 32 chips, coarse+fine on device, 30.08x fanout, nzp both sides, same table
SWS_QB1 = 80.1          # storeWeightedSums s per refinement on bh-host, full-device-deep-perf s0 table
SWS_DEV_BEST = 73.8     # best device route for one p150 (kernel unbuilt), relion-kernel-storeweightedsums s0
SWS_DEV_TODAY = 193.6   # route that exists today, same doc
FANOUT = 30.08          # 32-chip fanout at 111,300 particles, relion-production-scale-arm s0
PRICE = 110_000 / (370_000 / 8)   # Galaxy list vs one H200's share of an 8-GPU box, e2e s7.2

sws_host = SWS_QB1 / XEON
rows = [("sws at zero cost (ceiling)", 0.0),
        ("best device route / fanout", SWS_DEV_BEST / XEON / FANOUT),
        ("today's route / fanout", SWS_DEV_TODAY / XEON / FANOUT)]
print(f"storeWeightedSums host term in the Galaxy wall: {sws_host:.1f} s of {GALAXY_HONEST} s")
print(f"price ratio Galaxy / one H200: {PRICE:.2f}x")
for name, dev in rows:
    wall = GALAXY_HONEST - sws_host + dev
    speed = H200_HONEST / wall
    print(f"{name:30s} Galaxy {wall:5.1f} s  vs 1x H200 {speed:.2f}x wall  {speed / PRICE:.2f}x per $")
