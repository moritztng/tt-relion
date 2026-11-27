"""Precision of the resident diff2 on REAL particles: Ozaki cross term with the digit sums combined ON THE CHIP.

The hook combined Ozaki's n per-diagonal integer sums on the host in float64. A resident E-step cannot ship n x [B*T, O]
back per batch, so the combine and d2 = |X|^2 + |CP|^2 - 2C must happen on the chip. d2 near the minimum is a small
difference of large terms, so a plain fp32 combine floors at ulp(|X|^2) (D14: 16x host fp32 at f146). This grades:
  host fp32 direct          RELION's own form, sum w |X - CP|^2 in fp32 (the class to match)
  oz n / host f64 combine   the hook's form
  oz n / chip fp32          sums combined on chip in plain fp32, d2 = N - 2C in fp32
  oz n / chip df32          same sums combined on chip as double-fp32 (TwoSum, true-fp32 eltwise ops), d2 rounded once
against float64 on the same operands. Operands are real: particles from the dev cell's run_it018_data.star (its stack,
CTF, pose, origin), reference = that iteration's half1 map projected in float64 (trilinear, pad 1), noise sigma2 per
shell estimated from the particles at their poses, so w = 1/(2 sigma2) is the real weighting. Orientations: the pose,
local perturbations and global random ones. Rows = particle x translation (the resident batch), columns = orientations
(shared reference operand). Also reports the posterior weights exp(-(d2 - min)) per particle: max |dw| and argmax.
  TT_VISIBLE_DEVICES=16 python prec_real.py --run ~/rlx/runs/dev_ref_bh --it 18 --data ~/rlx/data --size 50 --out r.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--it", type=int, default=18)
ap.add_argument("--data", required=True)
ap.add_argument("--size", type=int, default=50, help="coarse image size (s50 = the global finer passes)")
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--O", type=int, default=1024)
ap.add_argument("--ns", default="8,10,12,14")
ap.add_argument("--kc", type=int, default=480)
ap.add_argument("--out", required=True)
ap.add_argument("--nodev", action="store_true")
a = ap.parse_args()
APIX, ORI = 1.34, 360


def star(fn, block):
    cols, rows, cur, loop = [], [], None, False
    for l in open(fn):
        s = l.strip()
        if s.startswith("data_"):
            cur, cols, loop = s, [], False
            continue
        if cur != block:
            continue
        if s == "loop_":
            loop = True
        elif loop and s.startswith("_rln"):
            cols.append(s.split()[0][1:])
        elif loop and s and not s.startswith("#"):
            rows.append(dict(zip(cols, s.split())))
    return rows


def mrc(fn, idx=None):
    with open(fn, "rb") as f:
        h = np.frombuffer(f.read(1024), dtype=np.int32)
        nx, ny, nz, mode, next_ = h[0], h[1], h[2], h[3], h[23]
        assert mode == 2, mode
        f.seek(1024 + next_ + (0 if idx is None else idx * nx * ny * 4))
        n = nx * ny * (nz if idx is None else 1)
        v = np.frombuffer(f.read(n * 4), dtype=np.float32)
    return v.reshape((nz, ny, nx) if idx is None else (ny, nx)).astype(np.float64)


def euler(rot, tilt, psi):  # RELION Euler_angles2matrix (ZYZ)
    a, b, g = np.radians([rot, tilt, psi])
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return np.array([[cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
                     [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
                     [sc, ss, cb]])


# half-transform pixel set of the coarse size, RELION's Mresol rule
s = a.size
ky, kx = [], []
for i in range(s):
    ip = i if i < (s + 1) // 2 else i - s
    for jp in range(s // 2 + 1):
        if round(np.hypot(ip, jp)) < s // 2 + 1 and not (jp == 0 and ip < 0):
            ky.append(ip), kx.append(jp)
ky, kx = np.array(ky, float), np.array(kx, float)
P = len(kx)
shell = np.round(np.hypot(kx, ky)).astype(int)

optics = star(os.path.join(a.run, "run_it%03d_data.star" % a.it), "data_optics")[0]
parts = star(os.path.join(a.run, "run_it%03d_data.star" % a.it), "data_particles")
V, Cs, Q0 = float(optics["rlnVoltage"]), float(optics["rlnSphericalAberration"]), float(optics["rlnAmplitudeContrast"])
lam = 12.2643247 / np.sqrt(V * 1e3 * (1 + V * 1e3 * 0.978466e-6))
fx, fy = kx / (ORI * APIX), ky / (ORI * APIX)   # 1/A

# reference: that iteration's half1 map, float64 3-D FFT (pad 1), centred
vol = mrc(os.path.join(a.run, "run_it%03d_half1_class001.mrc" % a.it))
F3 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol)))   # index c = ORI//2 is k = 0
c = ORI // 2


def project(R):
    k = np.stack([kx, ky, np.zeros_like(kx)], 1) @ R   # 2-D frequency -> 3-D: k3 = R^T k2
    g = k + c
    i0 = np.floor(g).astype(int)
    f = g - i0
    out = np.zeros(P, complex)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                wgt = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out += wgt * F3[i0[:, 2] + dz, i0[:, 1] + dy, i0[:, 0] + dx]
    return out


rng = np.random.default_rng(0)
pick = rng.choice(len(parts), a.B, replace=False)
X, CTF, poses = [], [], []
for pi in pick:
    p = parts[pi]
    idx, fn = p["rlnImageName"].split("@")
    img = mrc(os.path.join(a.data, fn), int(idx) - 1)
    Fi = np.fft.rfft2(np.fft.ifftshift(img))
    X.append(Fi[ky.astype(int) % ORI, kx.astype(int)])
    du, dv, ang = float(p["rlnDefocusU"]), float(p["rlnDefocusV"]), np.radians(float(p["rlnDefocusAngle"]))
    th = np.arctan2(fy, fx)
    df = 0.5 * (du + dv + (du - dv) * np.cos(2 * (th - ang)))
    q2 = fx * fx + fy * fy
    chi = np.pi * lam * df * q2 - 0.5 * np.pi * Cs * 1e7 * lam ** 3 * q2 * q2
    CTF.append(-np.sqrt(1 - Q0 * Q0) * np.sin(chi) - Q0 * np.cos(chi))
    poses.append((float(p["rlnAngleRot"]), float(p["rlnAngleTilt"]), float(p["rlnAnglePsi"]),
                  float(p["rlnOriginXAngst"]) / APIX, float(p["rlnOriginYAngst"]) / APIX))
X, CTF = np.array(X), np.array(CTF)


def shiftp(dx, dy):   # image shifted by (dx, dy) px: X(k) exp(-2 pi i k.d / N)
    return np.exp(-2j * np.pi * (kx * dx + ky * dy) / ORI)


# scale + noise per shell at the pose (self-consistent, like RELION's sigma2_noise)
best = np.array([project(euler(*q[:3])) for q in poses])
Xc = np.array([X[b] * shiftp(poses[b][3], poses[b][4]) for b in range(a.B)])
sc = np.array([np.real(np.vdot(CTF[b] * best[b], Xc[b])) / np.vdot(CTF[b] * best[b], CTF[b] * best[b]).real for b in range(a.B)])
res = np.abs(Xc - sc[:, None] * CTF * best) ** 2
sig2 = np.array([res[:, shell == k].mean() for k in range(shell.max() + 1)]) / 2
w = 1.0 / (2 * sig2[shell])

# orientations: each particle's pose, 15 local perturbations each (sigma 3 deg), the rest global random
eul = [q[:3] for q in poses] + [tuple(np.array(poses[b][:3]) + rng.normal(0, 3, 3)) for b in range(a.B) for _ in range(15)]
while len(eul) < a.O:
    eul.append((rng.uniform(0, 360), np.degrees(np.arccos(rng.uniform(-1, 1))), rng.uniform(0, 360)))
eul = eul[:a.O]
Pref = np.array([project(euler(*e)) for e in eul])   # [O, P]
# translations: 13 = offsets within radius 2 px of the particle origin, step 1 px
tr = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if dx * dx + dy * dy <= 4]
T = len(tr)
# rows: particle side carries CTF, scale and w (the reference operand is then shared by all particles)
rows, rowpart = [], []
for b in range(a.B):
    for dx, dy in tr:
        xs = X[b] * shiftp(poses[b][3] + dx, poses[b][4] + dy)
        rows.append((xs, sc[b] * CTF[b]))
        rowpart.append(b)
rowpart = np.array(rowpart)
Ar = np.array([np.concatenate([(x * cf * w).real, (x * cf * w).imag]) for x, cf in rows])   # [R, 2P]: w*ctf*X
Br = np.concatenate([Pref.real, Pref.imag], 1)                                            # [O, 2P]
C64 = Ar @ Br.T                                                                          # sum w ctf Re(X conj P)
Nx = np.array([(w * np.abs(x) ** 2).sum() for x, cf in rows])
Ncp = np.array([[(w * (cf ** 2) * np.abs(Pref[o]) ** 2).sum() for o in range(a.O)] for x, cf in rows[::T]])[rowpart]
d64 = Nx[:, None] + Ncp - 2 * C64
fh = open(a.out, "a")


def grade(name, d, extra=None):
    e = np.abs(d - d64)
    rec = dict(arm=name, size=s, P=P, R=len(rows), O=a.O, T=T, err_max=float(e.max()))
    near = (d64 - d64.min(1, keepdims=True)) < 30
    rec["err_near"] = float(e[near].max())
    dw, am = [], 0
    for b in range(a.B):
        m = rowpart == b
        r64, r = d64[m].ravel(), d[m].ravel()
        p64 = np.exp(-(r64 - r64.min())); p64 /= p64.sum()
        pp = np.exp(-(r - r.min())); pp /= pp.sum()
        dw.append(np.abs(pp - p64).max()); am += int(np.argmax(pp) == np.argmax(p64))
    rec.update(w_maxdiff=float(max(dw)), argmax_agree=f"{am}/{a.B}", d2_min=float(d64.min()), d2_scale=float(np.median(Nx)))
    if extra:
        rec.update(extra)
    print(json.dumps(rec), flush=True)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


# RELION class: direct form in fp32, sum_p w |X - ctf P|^2
d32 = np.empty_like(d64)
Xr = torch.tensor(np.array([np.concatenate([x.real, x.imag]) for x, cf in rows]), dtype=torch.float32)
CF = torch.tensor(np.array([np.tile(cf, 2) for x, cf in rows]), dtype=torch.float32)
W2 = torch.tensor(np.tile(w, 2), dtype=torch.float32)
Pt = torch.tensor(Br, dtype=torch.float32)
for r0 in range(len(rows)):
    d32[r0] = ((Xr[r0][None] - CF[r0][None] * Pt) ** 2 * W2).sum(1).double().numpy()
grade("host fp32 direct", d32)

# exact scaling + digits, host emulation of the integer sums (exact) to separate combine error from chip arithmetic
def row_exp(x):
    m = np.abs(x).max(1)
    return np.where(m > 0, np.floor(np.log2(np.maximum(m, 1e-300))) + 1, 0).astype(np.int32)


ea, eb = row_exp(Ar), row_exp(Br)
An = np.ldexp(Ar.astype(np.float32).astype(np.float64), -ea[:, None])
Bn = np.ldexp(Br.astype(np.float32).astype(np.float64), -eb[:, None])
# fp32 operands are what the chip holds: grade the float64 truth on those same fp32 operands too
C64f = np.ldexp(np.ldexp(An @ Bn.T, ea[:, None]), eb[None, :])


def digits_np(x, n):
    out, r = [], x.copy()
    for i in range(n):
        q = np.round(r * 2.0 ** (2 * i + 1))
        out.append(q)
        r = r - q * 2.0 ** -(2 * i + 1)
    return out


def two_sum(x, y):  # float32 TwoSum, exact error term under round-to-nearest
    s_ = (x + y).astype(np.float32)
    bp = (s_ - x).astype(np.float32)
    ap_ = (s_ - bp).astype(np.float32)
    return s_, ((x - ap_).astype(np.float32) + (y - bp).astype(np.float32)).astype(np.float32)


Nf = (Nx[:, None] + Ncp) / 2          # float64 norms; on the chip they are a second Ozaki product (per particle, not per t)
Nhi = Nf.astype(np.float32); Nlo = (Nf - Nhi).astype(np.float32)
scale = np.ldexp(1.0, ea)[:, None] * np.ldexp(1.0, eb)[None, :]
for n in map(int, a.ns.split(",")):
    Ad, Bd = digits_np(An, n), digits_np(Bn, n)
    S = [sum(Ad[i] @ Bd[k - i].T for i in range(k + 1)) for k in range(n)]   # exact integers (< 2^24)
    # host float64 combine (the hook)
    Cf = sum(S[k] * 2.0 ** -(2 * k + 2) for k in range(n)) * scale
    grade(f"oz n{n} host f64 combine", Nx[:, None] + Ncp - 2 * Cf, dict(n=n))
    # plain fp32 combine (fp32 numpy = IEEE, as the chip's eltwise ops)
    acc = np.zeros(S[0].shape, np.float32)
    for k in range(n):
        acc = (acc + (S[k] * 2.0 ** -(2 * k + 2) * scale).astype(np.float32)).astype(np.float32)
    dd = (2 * ((Nf.astype(np.float32) - acc).astype(np.float32))).astype(np.float64)
    grade(f"oz n{n} fp32 combine (emul)", dd, dict(n=n))
    # double-fp32 combine: (hi, lo) += term with TwoSum; D = N/2 - C in df32, rounded once
    hi = np.zeros(S[0].shape, np.float32); lo = np.zeros_like(hi)
    for k in range(n):
        t = (S[k] * 2.0 ** -(2 * k + 2) * scale).astype(np.float32)   # exact: integer < 2^24 times a power of 2
        hi, e = two_sum(hi, t)
        lo = (lo + e).astype(np.float32)
    dh, de = two_sum(Nhi, -hi)
    D = (dh + ((de + Nlo).astype(np.float32) - lo).astype(np.float32)).astype(np.float32)
    grade(f"oz n{n} df32 combine (emul)", 2 * D.astype(np.float64), dict(n=n))

if not a.nodev:
    import ttnn
    from _tt import Clock, open_dev

    dev, nodes = open_dev()
    CFG = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)

    def runs(x, kc):
        R, K = x.shape
        nc = -(-K // kc)
        x = np.pad(x, ((0, 0), (0, nc * kc - K)))
        return np.ascontiguousarray(x.reshape(R, nc, kc).transpose(1, 0, 2))

    def put(x, dt=ttnn.float32):
        return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), dtype=dt,
                               layout=ttnn.TILE_LAYOUT, device=dev)

    def tsum(x, y):  # on-chip TwoSum
        s_ = ttnn.add(x, y); bp = ttnn.subtract(s_, x); ap_ = ttnn.subtract(s_, bp)
        return s_, ttnn.add(ttnn.subtract(x, ap_), ttnn.subtract(y, bp))

    clk = Clock(nodes)
    for n in map(int, a.ns.split(",")):
        Ad = [put(runs(q, a.kc), ttnn.bfloat16) for q in digits_np(An, n)]                     # [nc, R, kc]
        Bd = [put(runs(q, a.kc).transpose(0, 2, 1), ttnn.bfloat16) for q in digits_np(Bn, n)]  # [nc, kc, O]
        sc_t = put(scale)
        hi = lo = None
        t0 = time.perf_counter()
        for k in range(n):
            acc = None
            for i in range(k + 1):
                Cm = ttnn.matmul(Ad[i], Bd[k - i], compute_kernel_config=CFG, dtype=ttnn.float32)
                acc = Cm if acc is None else ttnn.add(acc, Cm)
            Sk = ttnn.sum(acc, 0)          # run tree: integers < 2^24, exact in fp32
            tk = ttnn.multiply(ttnn.multiply(ttnn.reshape(Sk, (Sk.shape[-2], Sk.shape[-1])), 2.0 ** -(2 * k + 2)), sc_t)
            if hi is None:
                hi, lo = tk, ttnn.multiply(tk, 0.0)
            else:
                hi, e = tsum(hi, tk)
                lo = ttnn.add(lo, e)
        dh, de = tsum(put(Nhi), ttnn.neg(hi))
        D = ttnn.add(dh, ttnn.subtract(ttnn.add(de, put(Nlo)), lo))
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        Dn = ttnn.to_torch(D).double().numpy()[:len(rows), :a.O]
        Hn = ttnn.to_torch(hi).double().numpy()[:len(rows), :a.O]
        grade(f"oz n{n} df32 combine CHIP", 2 * Dn, dict(n=n, s_cold=round(dt, 3)))
        grade(f"oz n{n} fp32 combine CHIP", 2 * (Nf.astype(np.float32) - Hn.astype(np.float32)).astype(np.float64), dict(n=n))
    lo_, med = clk.stop()
    print(json.dumps({"aiclk_min": lo_, "aiclk_med": med, "nodes": nodes}), flush=True)
    fh.write(json.dumps({"aiclk_min": lo_, "aiclk_med": med, "nodes": nodes}) + "\n")
