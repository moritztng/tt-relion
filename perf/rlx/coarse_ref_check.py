"""Coarse calls with RELION's projection handed in (asked for on a cache miss) vs the bridge projecting
in torch. fp32 stub server, ttnn backend path, cache miss -> fill -> hit. The handed-in buffer is
overwritten after each call, as RELION reuses its thread_local projection buffer."""
import os, sys
os.environ["RLX_STUB_MATMUL"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bridge_prof as bp
R = bp.relion
R._srv.extend(R._Server(None, module="tt_relion.stub_server") for _ in range(2))
t = R._torch()

def ref_for(mdl, eul, s, isz):
    maxR, imgX = s // 2, s // 2 + 1
    P = np.arange(isz); yy = P // imgX
    O = len(eul) // 9
    r, i = R._project(t, t.from_numpy(np.frombuffer(mdl, np.float32).reshape(-1, 2).copy()),
                      t.from_numpy((P % imgX).astype(np.float32)),
                      t.from_numpy(np.where(yy > maxR, yy - s, yy).astype(np.float32)),
                      t.from_numpy(np.frombuffer(eul, np.float32).reshape(O, 9).copy()),
                      bp.MDL[0], bp.MDL[1], bp.MDL[2], -(bp.MDL[1] // 2), -(bp.MDL[2] // 2),
                      (maxR * bp.PF) ** 2, bp.PF)
    return memoryview(np.ascontiguousarray(t.cat([r, i], 1).numpy()).reshape(-1))

def call(args, s, **kw):
    mdl, eul, tx, ty, re, im, corr, isz = args
    out = np.zeros(len(eul) // 9 * len(tx), np.float32)
    maxR = s // 2
    r = R.diff2_coarse(mdl, eul, tx, ty, re, im, corr, memoryview(out), bp.MDL[0], bp.MDL[1], bp.MDL[2],
                       -(bp.MDL[1] // 2), -(bp.MDL[2] // 2), maxR, (maxR * bp.PF) ** 2, float(bp.PF),
                       s // 2 + 1, s, len(eul) // 9, len(tx), isz, **kw)
    return r, out

ok = True
for O, T, s in ((1024, 37, 50), (300, 21, 26), (2100, 9, 50)):
    rng = np.random.default_rng(7)
    sh = bp.shared(O, rng)
    parts = [bp.inputs(sh, T, s, rng) for _ in range(4)]
    R._pc.update(model=None)
    old = [call(p, s)[1] for p in parts]
    R._pc.update(model=None)
    ref = ref_for(sh[0], sh[1], s, parts[0][-1])
    asks, new = [], []
    for p in parts:
        r, out = call(p, s, can_ref=True)
        asks.append(r == 2)
        if r == 2:
            assert not out.any()
            buf = np.array(ref, copy=True)              # RELION's thread_local buffer: reused after the call
            r, out = call(p, s, ref_mv=memoryview(buf), can_ref=True)
            buf[:] = np.nan
        assert r is True
        new.append(out)
    same = all(np.array_equal(a, b) for a, b in zip(old, new))
    ok &= same and asks == [True, True, False, False]
    print("O%d T%d s%d bit-identical=%s asked=%s" % (O, T, s, same, asks), R.stats() if hasattr(R, "stats") else "")
print("L10-SCRIBBLE", "PASS" if ok else "FAIL")
