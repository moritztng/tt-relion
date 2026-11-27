"""On-device Fourier-slice projection from a resident 3D reference, checked against float64.

Semantics are the bridge's _project (RELION AccProjectorKernel::project3Dmodel, 2D data, no textures):
rotate the slice's half-plane pixels, trunc-radius mask, Hermitian flip for x < 0, trilinear interpolation.
Device path (ttnn ops only, a prototype to put a number on the structure, not the final kernel):
  coords on [O, P] tiles -> floor/frac/mask/flip -> int32 corner index -> ttnn.embedding from a resident
  table whose row v holds the 2x2x2 corners of voxel v (16 fp32) -> permute to [16, O, P] ->
  three lerps as slices of that batch dim -> ref_r, ref_i resident on chip.
  ttnn.embedding gathers bf16 only, so each fp32 corner is stored as three bf16 pieces (hi, mid, lo; their
  fp32 sum is the value exactly) in one 48-column row and summed after the gather. The pixel axis is cut
  into --pc chunks so no untilize row exceeds L1.
Table column order: col = dx*8 + dy*4 + dz*2 + c, so each lerp is "first half vs second half".
Reports error vs float64 for the chip and for torch fp32 (RELION's own class), the d2 error that error
causes against a noisy image, warm s/call per O orientations, and AICLK sampled during the timed calls.
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_relion.relion import _project  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="50,100,146,360")
ap.add_argument("--O", type=int, default=256)
ap.add_argument("--oc", type=int, default=64, help="orientations per device call chunk")
ap.add_argument("--pc", type=int, default=1024, help="pixels per device call chunk")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", default="proj.jsonl")
ap.add_argument("--cpu-only", action="store_true", help="check the reference and table layout, no device")
a = ap.parse_args()
PF = 2
OFF = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]


def geometry(s):
    R = PF * (s // 2)
    X, Y = R + 2, 2 * R + 3
    return dict(R=R, X=X, Y=Y, Z=Y, initY=-(R + 1), initZ=-(R + 1), maxR2=R * R)


def volume(gm, g):
    nvox = gm["X"] * gm["Y"] * gm["Z"]
    return torch.from_numpy(g.standard_normal((nvox, 2)).astype(np.float32))


def table(mdl, gm):
    """[nvox, 16] fp32, row v = corners of v in col order dx*8 + dy*4 + dz*2 + c (zero past the end)."""
    nvox = mdl.shape[0]
    pad = gm["X"] * gm["Y"] + gm["X"] + 2
    m = torch.cat([mdl, torch.zeros(pad, 2)])
    T = torch.empty(nvox, 16)
    for dx, dy, dz in OFF:
        o = dx + dy * gm["X"] + dz * gm["X"] * gm["Y"]
        k = dx * 8 + dy * 4 + dz * 2
        T[:, k:k + 2] = m[o:o + nvox]
    return T


def pieces3(T):
    """fp32 [n, 16] -> bf16 [n, 48]: hi | mid | lo, with hi + mid + lo == T exactly in fp32."""
    t = T.double()
    hi = t.to(torch.bfloat16)
    mid = (t - hi.double()).to(torch.bfloat16)
    lo = (t - hi.double() - mid.double()).to(torch.bfloat16)
    assert torch.equal(hi.float() + mid.float() + lo.float(), T.float())
    return torch.cat([hi, mid, lo], 1)


def pixels(s):
    imgX = s // 2 + 1
    pix = np.arange(imgX * s)
    x = pix % imgX
    yi = pix // imgX
    y = np.where(yi > s // 2, yi - s, yi)
    return torch.from_numpy(x.astype(np.float64)), torch.from_numpy(y.astype(np.float64))


def eulers(O, g):
    q = torch.from_numpy(g.standard_normal((O, 4)))
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.T
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)
    return R


def ref_project(dt, mdl, x, y, e, gm):
    return _project(torch, mdl.to(dt), x.to(dt), y.to(dt), e.to(dt), gm["X"], gm["Y"], gm["Z"], gm["initY"],
                    gm["initZ"], gm["maxR2"], PF)


def emulate(T, x, y, e, gm):
    """The device algorithm in float64 torch: proves the table layout and lerp order before any chip time."""
    xp = (e[:, 0:1] * x + e[:, 1:2] * y) * PF
    yp = (e[:, 3:4] * x + e[:, 4:5] * y) * PF
    zp = (e[:, 6:7] * x + e[:, 7:8] * y) * PF
    inside = (xp * xp + yp * yp + zp * zp) < gm["maxR2"] + 1
    sg = torch.where(xp < 0, -1.0, 1.0).double()
    xp, yp, zp = xp * sg, yp * sg, zp * sg
    x0, y0, z0 = xp.floor(), yp.floor(), zp.floor()
    fx, fy, fz = xp - x0, yp - y0, zp - z0
    xi = x0.clamp(0, gm["X"] - 2).long()
    yi = (y0 - gm["initY"]).clamp(0, gm["Y"] - 2).long()
    zi = (z0 - gm["initZ"]).clamp(0, gm["Z"] - 2).long()
    idx = zi * (gm["X"] * gm["Y"]) + yi * gm["X"] + xi
    G = T.double()[idx.reshape(-1)].reshape(idx.shape + (16,)).permute(2, 0, 1)
    D = G[:8] + (G[8:] - G[:8]) * fx
    E = D[:4] + (D[4:] - D[:4]) * fy
    F = E[:2] + (E[2:] - E[:2]) * fz
    zero = torch.zeros(())
    return torch.where(inside, F[0], zero), torch.where(inside, F[1] * sg, zero)




def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


fh = open(a.out, "a")


def emit(**k):
    fh.write(json.dumps(k) + "\n")
    fh.flush()
    print(json.dumps(k), flush=True)


if not a.cpu_only:
    import ttnn

    from _tt import Clock, open_dev

    dev, nodes = open_dev()

    def up(t, layout=None, dtype=None):
        return ttnn.from_torch(t, dtype=dtype or ttnn.float32, layout=layout or ttnn.TILE_LAYOUT, device=dev)

    def project_dev(Tdev, Xd, Yd, ecols, gm):
        """ecols: 6 device [Oc,1] tensors (e0 e1 e3 e4 e6 e7). Returns device ref_r, ref_i [Oc, P]."""
        e0, e1, e3, e4, e6, e7 = ecols
        xp = ttnn.multiply(ttnn.add(ttnn.multiply(e0, Xd), ttnn.multiply(e1, Yd)), float(PF))
        yp = ttnn.multiply(ttnn.add(ttnn.multiply(e3, Xd), ttnn.multiply(e4, Yd)), float(PF))
        zp = ttnn.multiply(ttnn.add(ttnn.multiply(e6, Xd), ttnn.multiply(e7, Yd)), float(PF))
        r2 = ttnn.add(ttnn.add(ttnn.multiply(xp, xp), ttnn.multiply(yp, yp)), ttnn.multiply(zp, zp))
        inside = ttnn.lt(r2, float(gm["maxR2"] + 1))
        sg = ttnn.add(ttnn.multiply(ttnn.ltz(xp), -2.0), 1.0)
        xp, yp, zp = ttnn.multiply(xp, sg), ttnn.multiply(yp, sg), ttnn.multiply(zp, sg)
        x0, y0, z0 = ttnn.floor(xp), ttnn.floor(yp), ttnn.floor(zp)
        fx, fy, fz = ttnn.subtract(xp, x0), ttnn.subtract(yp, y0), ttnn.subtract(zp, z0)
        xi = ttnn.typecast(ttnn.clamp(x0, 0.0, float(gm["X"] - 2)), ttnn.int32)
        yi = ttnn.typecast(ttnn.clamp(ttnn.subtract(y0, float(gm["initY"])), 0.0, float(gm["Y"] - 2)), ttnn.int32)
        zi = ttnn.typecast(ttnn.clamp(ttnn.subtract(z0, float(gm["initZ"])), 0.0, float(gm["Z"] - 2)), ttnn.int32)
        idx = ttnn.add(ttnn.add(ttnn.multiply(zi, gm["X"] * gm["Y"]), ttnn.multiply(yi, gm["X"])), xi)
        Oc, P = idx.shape[0], idx.shape[1]
        idx = ttnn.to_layout(ttnn.typecast(idx, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)   # [Oc, P]: one gather row per orientation
        G = ttnn.embedding(idx, Tdev, layout=ttnn.ROW_MAJOR_LAYOUT)
        G = ttnn.to_layout(ttnn.permute(ttnn.reshape(G, (Oc, P, 48)), (2, 0, 1)), ttnn.TILE_LAYOUT)
        G = ttnn.typecast(G, ttnn.float32)
        G = ttnn.add(ttnn.add(G[:16], G[16:32]), G[32:])

        def lerp(lo, hi, f):
            return ttnn.add(lo, ttnn.multiply(ttnn.subtract(hi, lo), f))

        D = lerp(G[:8], G[8:], fx)
        E = lerp(D[:4], D[4:], fy)
        F = lerp(E[:2], E[2:], fz)
        ref_r = ttnn.multiply(F[0], inside)
        ref_i = ttnn.multiply(ttnn.multiply(F[1], sg), inside)
        return ref_r, ref_i

g = np.random.default_rng(0)
for s in [int(v) for v in a.sizes.split(",")]:
    gm = geometry(s)
    mdl = volume(gm, g)
    T = table(mdl, gm)
    x, y = pixels(s)
    e = eulers(a.O, g)
    t0 = time.time()
    r64, i64 = ref_project(torch.float64, mdl, x, y, e, gm)
    host64_s = time.time() - t0
    t0 = time.time()
    r32, i32 = ref_project(torch.float32, mdl, x, y, e, gm)
    host32_s = time.time() - t0
    re_, ie_ = emulate(T, x, y, e, gm)
    img_r = r64 + torch.from_numpy(g.standard_normal(r64.shape))
    img_i = i64 + torch.from_numpy(g.standard_normal(r64.shape))

    def d2err(rr, ii):
        d = ((rr.double() - img_r) ** 2 + (ii.double() - img_i) ** 2).sum(1)
        d0 = ((r64 - img_r) ** 2 + (i64 - img_i) ** 2).sum(1)
        return float((d - d0).abs().max()), float(d0.mean())

    base = dict(s=s, O=a.O, P=int(x.numel()), nvox=int(mdl.shape[0]), table_MB=round(T.numel() * 6 / 2**20, 1))
    emit(**base, cfg="emulate f64 (device algorithm)", rel=max(rel(re_, r64), rel(ie_, i64)),
         d2_err=d2err(re_, ie_)[0])
    de, d2m = d2err(r32, i32)
    emit(**base, cfg="torch fp32 (host, RELION class)", rel=max(rel(r32, r64), rel(i32, i64)), d2_err=de,
         d2_mean=d2m, host32_s=round(host32_s, 3), host64_s=round(host64_s, 3))
    if a.cpu_only:
        continue
    try:
        t0 = time.time()
        Tdev = up(pieces3(T), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
        pcs = [(p, min(p + a.pc, x.numel())) for p in range(0, x.numel(), a.pc)]
        XY = [(up(x[p:q].float().reshape(1, -1)), up(y[p:q].float().reshape(1, -1))) for p, q in pcs]
        table_up_s = time.time() - t0
        cols = (0, 1, 3, 4, 6, 7)
        chunks = [[up(e[o:o + a.oc, c:c + 1].float()) for c in cols] for o in range(0, a.O, a.oc)]

        def run(keep):
            outs = []
            for ec in chunks:
                row = []
                for (p, q), (Xd, Yd) in zip(pcs, XY):
                    rr, ii = project_dev(Tdev, Xd, Yd, ec, gm)
                    if keep:
                        row.append((ttnn.to_torch(rr)[:, :q - p], ttnn.to_torch(ii)[:, :q - p]))
                    ttnn.deallocate(rr)
                    ttnn.deallocate(ii)
                if keep:
                    outs.append((torch.cat([r[0] for r in row], 1), torch.cat([r[1] for r in row], 1)))
            ttnn.synchronize_device(dev)
            return outs

        outs = run(True)
        rd = torch.cat([o[0] for o in outs]).double()
        id_ = torch.cat([o[1] for o in outs]).double()
        run(False)  # warm
        clk = Clock(nodes)
        t0 = time.time()
        for _ in range(a.reps):
            run(False)
        s_call = (time.time() - t0) / a.reps
        cmin, cmed = clk.stop()
        de, _ = d2err(rd, id_)
        emit(**base, cfg="ttnn fp32 embedding+lerp", oc=a.oc, pc=a.pc, rel=max(rel(rd, r64), rel(id_, i64)), d2_err=de,
             s_per_call=round(s_call, 4), table_up_s=round(table_up_s, 2), aiclk_min=cmin, aiclk_med=cmed)
        ttnn.deallocate(Tdev)
    except Exception as ex:  # report and move to the next size
        emit(**base, cfg="ttnn fp32 embedding+lerp", error=repr(ex)[:600])
if not a.cpu_only:
    ttnn.close_device(dev)
print("PROJ-DONE", flush=True)
