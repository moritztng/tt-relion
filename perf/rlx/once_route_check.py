"""TT_RELION_COARSE_ONCE=cpu routing: a particle whose coarse set is seen for the first time goes to
RELION's CPU kernel for its coarse AND its fine pass; from the second sighting on both go to the chip.
The flag is per --j thread. fp32 stub server, ttnn backend path."""
import os, sys, threading
os.environ["RLX_STUB_MATMUL"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import coarse_ref_check as cr                       # runs the L10 check (default routing), then reuse its helpers
R, bp = cr.R, cr.bp
R._COARSE_ONCE = "cpu"                              # as TT_RELION_COARSE_ONCE=cpu at import

def fine_routed():
    """Calls diff2_fine with empty inputs: only the routing decision is under test."""
    n = R._stats["once_cpu_fine"]
    R.diff2_fine(None, None, b"", b"", b"", b"", b"", b"", b"", b"", bytearray(), 0, 0, 0, 0, 0, 0, 0,
                 1.0, 0, 0, 0.0, 0, 0, 0, 0, 0)
    return R._stats["once_cpu_fine"] > n

ok = True
O, T, s = 300, 21, 26
rng = np.random.default_rng(3)
sh = bp.shared(O, rng)
parts = [bp.inputs(sh, T, s, rng) for _ in range(3)]
R._pc.update(model=None)
ref = cr.ref_for(sh[0], sh[1], s, parts[0][-1])
seen = []
for p in parts:
    r, _ = cr.call(p, s, can_ref=True)
    if r == 2:
        r, _ = cr.call(p, s, ref_mv=memoryview(np.array(ref, copy=True)), can_ref=True)
    seen.append((r, fine_routed()))
ok &= seen == [(False, True), (True, False), (True, False)]
print("sightings (coarse handled, fine to host):", seen)

# per thread: thread A's host particle must not send thread B's chip particle to the host
R._pc.update(model=None)
res = {}
ev = threading.Event()
def a():
    r, _ = cr.call(parts[0], s, can_ref=True); ev.set(); ev2.wait(); res["a"] = (r, fine_routed())
def b():
    ev.wait(); r, _ = cr.call(parts[1], s, can_ref=True)
    if r == 2:
        r, _ = cr.call(parts[1], s, ref_mv=memoryview(np.array(ref, copy=True)), can_ref=True)
    res["b"] = (r, fine_routed()); ev2.set()
ev2 = threading.Event()
th = [threading.Thread(target=f) for f in (a, b)]
[x.start() for x in th]; [x.join() for x in th]
ok &= res == {"a": (False, True), "b": (True, False)}
print("threads:", res)
print("ONCE-ROUTE", "PASS" if ok else "FAIL")
