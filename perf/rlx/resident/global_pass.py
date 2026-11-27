"""Resident global pass on one chip, real particles: reference projected ON CHIP (proj_rot, fp32, RELION order), digit cut
ON CHIP, Ozaki cross term and |CTF P|^2 term on chip, d2 = |X|^2 + |CTF P|^2 - 2C combined on chip in df32.
Graded like prec_real (whose setup this reuses verbatim) against float64, beside RELION's own fp32 direct form computed
from the chip's fp32 projection (the class to match). prec_real's original docstring follows.

Precision of the resident diff2 on REAL particles: Ozaki cross term with the digit sums combined ON THE CHIP.

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
ap.add_argument("--ns-n", type=int, default=0, help="digits for the |CTF P|^2 product (0 = same as --ns)")
ap.add_argument("--kc", type=int, default=480)
ap.add_argument("--out", required=True)
ap.add_argument("--nodev", action="store_true")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--balance", type=int, default=1)
ap.add_argument("--fused", type=int, default=1, help="oz_fused kernel (1) or the composed ttnn matmul+add path (0)")
ap.add_argument("--nchip", type=int, default=13, help="Ozaki digits of the chip pass (13: below RELION fp32 error at s50 and s146, sets identical; runs/gp19)")
ap.add_argument("--pchip", type=int, default=0, help="build the particle operand on chip from the resident stack")
ap.add_argument("--fcut", type=int, default=1, help="fused SFPU digit-cut kernel (1) or the ttnn chain (0)")
ap.add_argument("--fcomb", type=int, default=1, help="fused SFPU combine kernel (1) or the ttnn TwoSum chain (0)")
ap.add_argument("--frac", type=float, default=0.999, help="RELION adaptive_fraction")
ap.add_argument("--sig-off", type=float, default=3.0, help="offset prior sigma, px")
ap.add_argument("--fast", action="store_true", help="skip the host fp32/Ozaki arms (timing at production row counts)")
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
for r0 in ([] if a.fast else range(len(rows))):
    d32[r0] = ((Xr[r0][None] - CF[r0][None] * Pt) ** 2 * W2).sum(1).double().numpy()
if not a.fast:
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
for n in ([] if a.fast else map(int, a.ns.split(","))):
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

def significance(d2, prior, frac):
    """RELION's coarse-pass significance per particle on d2 [T, B, O] (+ per-translation -log prior): weights
    exp(-(d2 + prior - min)), the smallest set of largest weights holding >= frac of the sum. Samples with delta > 25
    carry < 1.4e-11 each (< 1e-6 summed over 53k), so only candidates below that are sorted. Returns per particle
    (n_sig, sorted significant flat indices into [T, O], weight sum)."""
    Tn, Bn, On = d2.shape
    out = []
    for b in range(Bn):
        x = (d2[:, b, :] + prior[:, None]).ravel()
        dl = x - x.min()
        cand = np.flatnonzero(dl < 25.0)
        w = np.exp(-dl[cand])
        order = np.argsort(-w, kind="stable")
        cs = np.cumsum(w[order])
        S = w.sum() + np.exp(-dl[dl >= 25.0]).sum()
        k = int(np.searchsorted(cs, frac * S) + 1)
        out.append((k, np.sort(cand[order[:k]]), S))
    return out


if not a.nodev:
    import ttnn
    from _tt import Clock, open_dev
    from proj_lerp import proj_rot

    dev, nodes = open_dev()
    CFG = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    KC = a.kc
    rec_t = {}

    def put(x, dt=ttnn.float32, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), dtype=dt, layout=layout,
                               device=dev)

    def tsum(x, y):  # on-chip TwoSum (true-fp32 eltwise ops)
        s_ = ttnn.add(x, y); bp = ttnn.subtract(s_, x); ap_ = ttnn.subtract(s_, bp)
        return s_, ttnn.add(ttnn.subtract(x, ap_), ttnn.subtract(y, bp))

    # --- 1. resident reference: half-volume brick table from the map's FFT, projected on chip -------------------------
    Rr = s // 2 + 1
    gm = dict(X=Rr + 2, Y=2 * Rr + 3, Z=2 * Rr + 3, initY=-(Rr + 1), initZ=-(Rr + 1), maxR2=(Rr + 1) ** 2)
    sub = F3[c + gm["initZ"]:c + gm["initZ"] + gm["Z"], c + gm["initY"]:c + gm["initY"] + gm["Y"], c:c + gm["X"]]
    mdl = np.stack([sub.real.ravel(), sub.imag.ravel()], 1).astype(np.float32)
    nvox = mdl.shape[0]
    mp = np.concatenate([mdl, np.zeros((gm["X"] * gm["Y"] + gm["X"] + 2, 2), np.float32)])
    Tb = np.empty((nvox, 16), np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                o_ = dx + dy * gm["X"] + dz * gm["X"] * gm["Y"]
                cc = 4 * dz + 2 * dy + dx
                Tb[:, 2 * cc:2 * cc + 2] = mp[o_:o_ + nvox]
    NC = -(-P // 1024)
    K = NC * 2048
    pxy = np.full((2, NC * 1024), 3000.0, np.float32)
    pxy[0, :P], pxy[1, :P] = kx, ky
    pix = pxy.reshape(2, NC, 1024).transpose(1, 0, 2).reshape(2 * NC, 1024)
    rot = np.zeros((a.O, 8), np.float32)
    for o_, e_ in enumerate(eul):
        Rm = euler(*e_)
        rot[o_, :6] = [Rm[0, 0], Rm[1, 0], Rm[0, 1], Rm[1, 1], Rm[0, 2], Rm[1, 2]]
    RM = ttnn.ROW_MAJOR_LAYOUT
    t_tab, t_pix, t_rot = put(Tb, layout=RM), put(pix, layout=RM), put(rot, layout=RM)

    def kchip(re, im):  # [.., P] complex parts -> chip K order: chunk c = re(1024 px), im(1024 px); pads 0
        n_ = re.shape[0]
        z = np.zeros((n_, NC, 2, 1024), np.float32)
        r_ = np.zeros((n_, NC * 1024), np.float32); r_[:, :P] = re
        i_ = np.zeros((n_, NC * 1024), np.float32); i_[:, :P] = im
        z[:, :, 0] = r_.reshape(n_, NC, 1024); z[:, :, 1] = i_.reshape(n_, NC, 1024)
        return z.reshape(n_, K)

    # particle side (v0 built on host, uploaded once; resident version builds it on chip from the resident stack)
    Ah = kchip(Ar[:, :P], Ar[:, P:])                                          # [R, K] w ctf X
    A2h = kchip(np.array([w * cf ** 2 for x, cf in rows[::T]]), np.array([w * cf ** 2 for x, cf in rows[::T]]))  # [B, K]
    # device rows are translation-major (t B + b) so the per-particle |CTF P|^2 [B, O] expands by a pure-copy concat;
    # ttnn.repeat_interleave on fp32 measured 3e-3 relative error (bf16-class), so it is not used
    perm = np.array([b * T + t for t in range(T) for b in range(a.B)])
    prior_t = np.array([(dx * dx + dy * dy) / (2 * a.sig_off ** 2) for dx, dy in tr])   # -log pdf_offset
    Ah = Ah[perm]
    t_A, t_A2 = put(Ah), put(A2h)
    # resident particle side for --pchip: pre-centred images (pose offset applied once at upload, I/O-side), the
    # per-particle w*sc*CTF and w*(sc*CTF)^2 spectra, and the T translation phases; the [T*B, K] operand is then
    # built on chip each pass (rows translation-major, the same order as perm).
    Xs = np.array([X[b] * shiftp(poses[b][3], poses[b][4]) for b in range(a.B)])
    wcf = np.array([w * sc[b] * CTF[b] for b in range(a.B)])
    PH = np.array([shiftp(dx, dy) for dx, dy in tr])

    def half(v):
        z = np.zeros((v.shape[0], NC * 1024), np.float32); z[:, :P] = v
        return z
    t_xr, t_xi, t_wc = put(half(Xs.real)), put(half(Xs.imag)), put(half(wcf))
    t_wc2 = put(half(wcf * np.array([sc[b] * CTF[b] for b in range(a.B)])))
    t_pr, t_pi = put(half(PH.real)), put(half(PH.imag))

    def kjoin(re, im):   # [n, NC*1024] re and im -> [n, K] chip order (chunk c = re 1024 px, im 1024 px); tile-aligned copies
        parts = []
        for c in range(NC):
            parts += [re[:, c * 1024:(c + 1) * 1024], im[:, c * 1024:(c + 1) * 1024]]
        return ttnn.concat(parts, dim=1)

    def particle_operand():
        ar, ai = ttnn.multiply(t_xr, t_wc), ttnn.multiply(t_xi, t_wc)
        rr, ri = [], []
        for t in range(T):
            pr, pi_ = t_pr[t:t + 1], t_pi[t:t + 1]
            rr.append(ttnn.subtract(ttnn.multiply(ar, pr), ttnn.multiply(ai, pi_)))
            ri.append(ttnn.add(ttnn.multiply(ar, pi_), ttnn.multiply(ai, pr)))
        return kjoin(ttnn.concat(rr, dim=0), ttnn.concat(ri, dim=0)), kjoin(t_wc2, t_wc2)
    Kp = -(-K // KC) * KC
    nc = Kp // KC

    def runs_dev(X_, rows_first):   # [n, K] fp32 TILE -> [nc, n, KC] (rows_first) or [nc, KC, n]
        n_ = X_.shape[0]
        if Kp > K:
            X_ = ttnn.pad(X_, [(0, 0), (0, Kp - K)], 0.0)
        X_ = ttnn.reshape(X_, (n_, nc, KC))
        return ttnn.permute(X_, (1, 0, 2) if rows_first else (1, 2, 0))

    def row_exp_dev(X_):  # per-row exponent: max |x| on chip, exponent on host (n floats back)
        m = ttnn.to_torch(ttnn.max(ttnn.abs(X_), dim=-1)).double().numpy().reshape(-1)[:X_.shape[0]]
        return np.where(m > 0, np.floor(np.log2(np.maximum(m, 1e-300))) + 1, 0).astype(np.int32)

    def cut_dev(Xr, sc_b, n, kmajor=False):
        """Xr runs layout [nc, n, KC] fp32, sc_b [1, n, 1] = 2^-e; -> n bf16 digit tensors, exact. kmajor: each digit
        transposed to [nc, KC, n] AFTER the cut (an fp32 transpose on chip truncates to ~2^-11, measured: cutB 4.9e-4;
        the digits are small integers, exact in any format)."""
        r = ttnn.multiply(Xr, sc_b)
        out = []
        for i in range(n):
            q = ttnn.round(ttnn.multiply(r, 2.0 ** (2 * i + 1)))
            qb = ttnn.typecast(q, ttnn.bfloat16)
            out.append(ttnn.permute(qb, (0, 2, 1)) if kmajor else qb)
            if i + 1 < n:
                r = ttnn.subtract(r, ttnn.multiply(q, 2.0 ** -(2 * i + 1)))
        return out

    def oz_dev(Ad, Bd, sc_t, n):  # sum_k 2^-(2k+2) sum_i A_i B_(k-i), df32 (hi, lo) with power-of-2 scale sc_t
        hi = lo = None
        for k in range(n):
            acc = None
            for i in range(k + 1):
                Cm = ttnn.matmul(Ad[i], Bd[k - i], compute_kernel_config=CFG, dtype=ttnn.float32)
                acc = Cm if acc is None else ttnn.add(acc, Cm)
            Sk = ttnn.sum(acc, 0)
            tk = ttnn.multiply(ttnn.multiply(ttnn.reshape(Sk, (Sk.shape[-2], Sk.shape[-1])), 2.0 ** -(2 * k + 2)), sc_t)
            if hi is None:
                hi, lo = tk, ttnn.multiply(tk, 0.0)
            else:
                hi, e_ = tsum(hi, tk)
                lo = ttnn.add(lo, e_)
        return hi, lo

    def balance(XA, XB):
        """Exact K-diagonal rebalance A diag(d), diag(1/d) B with d = powers of 2: the weights (w CTF, w CTF^2) peak at high
        frequency and the reference (P, |P|^2) at low, so per-row Ozaki scaling wastes digits on the mismatch (measured:
        |CTF P|^2 at 8e-6 relative before). Column maxima come back (K floats per side); d goes up as one [1, K] row."""
        ma = ttnn.to_torch(ttnn.max(ttnn.abs(XA), dim=0)).double().numpy().reshape(-1)[:K]
        mb = ttnn.to_torch(ttnn.max(ttnn.abs(XB), dim=0)).double().numpy().reshape(-1)[:K]
        ok = (ma > 0) & (mb > 0)
        e = np.where(ok, np.round(0.5 * (np.log2(np.where(ok, mb, 1)) - np.log2(np.where(ok, ma, 1)))), 0)
        return ttnn.multiply(XA, put(np.ldexp(1.0, e.astype(int)).reshape(1, -1))), \
            ttnn.multiply(XB, put(np.ldexp(1.0, -e.astype(int)).reshape(1, -1)))

    from oz_fused import oz_combine, oz_cut, oz_fused

    def oz_fused_dev(Xa, Xb, n):
        """Xa [Ra, K], Xb [Ob, K] fp32 TILE -> (hi, lo) df32 of Xa Xb^T, Ozaki n through oz_fused. Row exponents per side;
        rows padded to oz_fused's blocks (zero rows cut to zero digits); B digits stay row-major (oz_fused bt)."""
        Ra, Ob = Xa.shape[0], Xb.shape[0]
        Rp, Op = -(-Ra // 512) * 512, -(-Ob // 256) * 256
        ea_, eb_ = row_exp_dev(Xa), row_exp_dev(Xb)

        ti = lambda e_, m: ttnn.from_torch(torch.from_numpy(np.pad(e_, (0, m - len(e_))).astype(np.int32).reshape(-1, 32)),  # noqa: E731
                                          dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

        def digits(X_, e_, npad, tr):
            Xp = ttnn.pad(X_, [(0, npad - X_.shape[0]), (0, Kp - K)], 0.0)
            if a.fcut:   # one SFPU kernel: exact scale, n round/subtract/x4 steps, bf16 digit planes
                return oz_cut(Xp, ti(e_, npad), n)
            r = ttnn.multiply(Xp, put(np.ldexp(1.0, -np.pad(e_, (0, npad - len(e_)))).reshape(-1, 1)))
            out = []
            for i in range(n):
                q = ttnn.round(ttnn.multiply(r, 2.0 ** (2 * i + 1)))
                qb = ttnn.typecast(q, ttnn.bfloat16)
                out.append(ttnn.reshape(qb, (1,) + tuple(qb.shape)))
                if i + 1 < n:
                    r = ttnn.subtract(r, ttnn.multiply(q, 2.0 ** -(2 * i + 1)))
            return ttnn.concat(out, dim=0)

        Ad, Bd = digits(Xa, ea_, Rp, False), digits(Xb, eb_, Op, False)
        S = oz_fused(Ad, Bd, n, bt=True)   # B digits stay [n, O, K]; the unpacker transposes each tile
        if a.fcomb:   # one SFPU kernel: TwoSum over k and the exact 2^(ea + eb) scale, a single pass over S
            assert int(np.min(ea_)) + int(np.min(eb_)) - 2 * n > -126 and int(np.max(ea_)) + int(np.max(eb_)) < 127
            hi, lo = oz_combine(S, ti(ea_, Rp), ti(eb_, Op), n)
            return hi[:Ra, :Ob], lo[:Ra, :Ob]
        cs = put(np.ldexp(1.0, np.pad(ea_, (0, Rp - Ra))).reshape(-1, 1))
        rs = put(np.ldexp(1.0, np.pad(eb_, (0, Op - Ob))).reshape(1, -1))
        hi = lo = None
        for k in range(n):
            tk = ttnn.multiply(ttnn.multiply(ttnn.multiply(ttnn.reshape(S[k:k + 1], (Rp, Op)), 2.0 ** -(2 * k + 2)), cs), rs)
            if hi is None:
                hi, lo = tk, ttnn.multiply(tk, 0.0)
            else:
                hi, e2 = tsum(hi, tk)
                lo = ttnn.add(lo, e2)
        return hi[:Ra, :Ob], lo[:Ra, :Ob]

    def chip_pass_fused(n, check=False):
        t = {}
        t0 = time.perf_counter()
        pr = proj_rot(t_tab, t_pix, t_rot, gm)
        B_ = ttnn.to_layout(ttnn.reshape(pr, (a.O, K)), ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(dev); t["proj"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        Braw = B_
        B2 = ttnn.multiply(B_, B_)
        if a.pchip:
            ttnn.synchronize_device(dev); t1 = time.perf_counter()
            A_, A2_ = particle_operand()
            ttnn.synchronize_device(dev); t["particle_op"] = time.perf_counter() - t1
            t2 = time.perf_counter()
            if check:   # vs the host float64-built operand rounded to fp32
                pa = ttnn.to_torch(A_).numpy()[:Ah.shape[0], :K]
                t["particle_op_maxrel"] = float(np.abs(pa - Ah).max() / np.abs(Ah).max())
            t0 += time.perf_counter() - t1   # cut_oz_combine excludes the operand build (timed on its own)
        else:
            A_, A2_ = t_A, t_A2
        tA, B_ = balance(A_, B_) if a.balance else (A_, B_)
        tA2, B2 = balance(A2_, B2) if a.balance else (A2_, B2)
        Ch, Cl = oz_fused_dev(tA, B_, n)
        Nh, Nl = oz_fused_dev(tA2, B2, a.ns_n or n)   # N = sum |CTF P|^2 w is all-positive: no cancellation inside it
        Nh = ttnn.concat([Nh] * T, dim=0)
        Nl = ttnn.concat([Nl] * T, dim=0)
        Nxp = Nx[perm]
        Nxh = Nxp.astype(np.float32); Nxl = (Nxp - Nxh).astype(np.float32)
        # |X|^2 / 2 per row as [R, 1] columns broadcast on chip (an [R, O] host upload per pass cost ~100 ms)
        nxh = ttnn.add(ttnn.multiply(Nh, 0.0), put(Nxh[:, None] / 2))
        sh, se = tsum(ttnn.multiply(Nh, 0.5), nxh)
        lo_ = ttnn.add(ttnn.add(se, ttnn.multiply(Nl, 0.5)), put(Nxl[:, None] / 2))
        dh, de = tsum(sh, ttnn.neg(Ch))
        D = ttnn.add(dh, ttnn.subtract(ttnn.add(de, lo_), Cl))
        ttnn.synchronize_device(dev); t["cut_oz_combine"] = time.perf_counter() - t0
        # weights + significance v0: d2 back to the host (fp32), RELION's rule in float64 there
        t0 = time.perf_counter()
        Dd = ttnn.to_torch(D).numpy()[:len(rows), :a.O]
        t["readback"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        sig = significance((2.0 * Dd.astype(np.float64)).reshape(T, a.B, a.O), prior_t, a.frac)
        t["host_sig"] = time.perf_counter() - t0
        out = dict(t=t, sig=sig)
        if check:
            out["B"] = ttnn.to_torch(Braw).double().numpy()
            Dd = ttnn.to_torch(D).double().numpy()[:len(rows), :a.O]
            out["D"] = np.empty_like(Dd)
            out["D"][perm] = Dd
            out["diag"] = {}
        return out

    def chip_pass(n, check=False):
        if a.fused:
            return chip_pass_fused(n, check)
        t = {}
        t0 = time.perf_counter()
        pr = proj_rot(t_tab, t_pix, t_rot, gm)                                    # [2 O NC, 1024] RM
        B_ = ttnn.to_layout(ttnn.reshape(pr, (a.O, K)), ttnn.TILE_LAYOUT)         # [O, K] reference operand
        ttnn.synchronize_device(dev); t["proj"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        B2 = ttnn.multiply(B_, B_)                                               # |P|^2 parts (fp32, RELION's rounding)
        B2raw, Braw = B2, B_
        tA, B_ = balance(t_A, B_) if a.balance else (t_A, B_)
        tA2, B2 = balance(t_A2, B2) if a.balance else (t_A2, B2)
        ea_, eb_ = row_exp_dev(tA), row_exp_dev(B_)
        ea2, eb2 = row_exp_dev(tA2), row_exp_dev(B2)
        Ad = cut_dev(runs_dev(tA, True), put(np.ldexp(1.0, -ea_).reshape(1, -1, 1)), n)
        Bd = cut_dev(runs_dev(B_, True), put(np.ldexp(1.0, -eb_).reshape(1, -1, 1)), n, kmajor=True)
        A2d = cut_dev(runs_dev(tA2, True), put(np.ldexp(1.0, -ea2).reshape(1, -1, 1)), n)
        B2d = cut_dev(runs_dev(B2, True), put(np.ldexp(1.0, -eb2).reshape(1, -1, 1)), n, kmajor=True)
        ttnn.synchronize_device(dev); t["cut"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ch, Cl = oz_dev(Ad, Bd, put(np.ldexp(1.0, ea_)[:, None] * np.ldexp(1.0, eb_)[None, :]), n)
        Nh, Nl = oz_dev(A2d, B2d, put(np.ldexp(1.0, ea2)[:, None] * np.ldexp(1.0, eb2)[None, :]), n)  # [B, O]
        Nh0, Nl0 = Nh, Nl
        # rows are b * T + t: expand the per-particle |CTF P|^2 over the T translations
        Nh = ttnn.concat([Nh] * T, dim=0)
        Nl = ttnn.concat([Nl] * T, dim=0)
        Nxp = Nx[perm]
        Nxh = Nxp.astype(np.float32); Nxl = (Nxp - Nxh).astype(np.float32)
        # D = (Nx + Ncp) / 2 - C in df32, rounded once
        sh, se = tsum(ttnn.multiply(Nh, 0.5), put(np.repeat(Nxh[:, None] / 2, a.O, 1)))
        lo_ = ttnn.add(ttnn.add(se, ttnn.multiply(Nl, 0.5)), put(np.repeat(Nxl[:, None] / 2, a.O, 1)))
        dh, de = tsum(sh, ttnn.neg(Ch))
        D = ttnn.add(dh, ttnn.subtract(ttnn.add(de, lo_), Cl))
        ttnn.synchronize_device(dev); t["oz_combine"] = time.perf_counter() - t0
        out = dict(t=t)
        if check:
            Bh = ttnn.to_torch(Braw).double().numpy()
            out["B"] = Bh
            Dd = ttnn.to_torch(D).double().numpy()[:len(rows), :a.O]
            out["D"] = np.empty_like(Dd)
            out["D"][perm] = Dd
            # component diagnostics against float64 on the chip's own fp32 operands
            th = lambda x_: ttnn.to_torch(x_).double().numpy()  # noqa: E731
            Bh = Bh[:a.O, :K]
            Cc = (th(Ch) + th(Cl))[:len(rows), :a.O]
            C_ = Ah.astype(np.float64) @ Bh.T
            B2h = (Bh.astype(np.float32) ** 2).astype(np.float64)
            Nc = (th(Nh) + th(Nl))[:len(rows), :a.O]
            N_ = (A2h.astype(np.float64) @ B2h.T)[np.arange(len(rows)) % a.B]
            def recon(dl, sc_e, rows_first):  # sum_i q_i 2^-(2i+1), back in [n, K] order, vs the scaled operand
                acc_ = None
                for i, q_ in enumerate(dl):
                    v = th(q_) * 2.0 ** -(2 * i + 1)
                    acc_ = v if acc_ is None else acc_ + v
                if rows_first:   # [nc, n, KC] -> [n, nc KC]
                    acc_ = acc_.transpose(1, 0, 2).reshape(acc_.shape[1], -1)
                else:            # [nc, KC, n] -> [n, nc KC]
                    acc_ = acc_.transpose(2, 0, 1).reshape(acc_.shape[2], -1)
                return acc_[:, :K]
            An_ = np.ldexp(th(tA)[:len(rows), :K], -ea_[:, None])
            Bn_ = np.ldexp(th(B_)[:a.O, :K], -eb_[:, None])
            out["diag"] = dict(
                C_rel=float(np.abs(Cc - C_).max() / np.abs(C_).max()),
                B2_rel=float(np.abs(th(B2raw)[:a.O, :K] - B2h).max() / B2h.max()),
                Ncp0_rel=float(np.abs((th(Nh0) + th(Nl0))[:a.B, :a.O] - N_[:a.B]).max() / np.abs(N_).max()),
                A2cut_err=float(np.abs(recon(A2d, ea2, True)[:a.B] - np.ldexp(th(tA2)[:a.B, :K], -ea2[:, None])).max()),
                Ncp_rel=float(np.abs(Nc - N_).max() / np.abs(N_).max()),
                cutA_err=float(np.abs(recon(Ad, ea_, True)[:len(rows)] - An_).max()),
                cutB_err=float(np.abs(recon(Bd, eb_, False)[:a.O] - Bn_).max()),
                digA_max=float(max(np.abs(th(q_)).max() for q_ in Ad)),
                digB_max=float(max(np.abs(th(q_)).max() for q_ in Bd)))
        return out

    n = a.nchip
    r0 = chip_pass(n, check=True)
    # chip projection vs float64 projection (same pixels, same orientations)
    Bh = r0["B"].reshape(a.O, NC, 2, 1024)
    Pre = Bh[:, :, 0].reshape(a.O, -1)[:, :P]; Pim = Bh[:, :, 1].reshape(a.O, -1)[:, :P]
    prel = float(max(np.abs(Pre - Pref.real).max(), np.abs(Pim - Pref.imag).max()) / np.abs(Pref).max())
    # RELION's own class on the chip's fp32 projection: direct form in fp32
    Pt2 = torch.tensor(np.concatenate([Pre, Pim], 1), dtype=torch.float32)
    d32c = np.empty_like(d64)
    for q_ in ([] if a.fast else range(len(rows))):
        d32c[q_] = ((Xr[q_][None] - CF[q_][None] * Pt2) ** 2 * W2).sum(1).double().numpy()
    if not a.fast:
        grade("host fp32 direct on CHIP projection", d32c, dict(proj_rel_vs_f64=prel))
    print(json.dumps({"diag": r0["diag"]}), flush=True)
    # significance parity: chip d2 vs float64 d2 (rows of d64 are b-major: b * T + t)
    ref_sig = significance(d64.reshape(a.B, T, a.O).transpose(1, 0, 2), prior_t, a.frac)
    same_set = sum(int(np.array_equal(c_[1], r_[1])) for c_, r_ in zip(r0["sig"], ref_sig))
    nd = [abs(c_[0] - r_[0]) for c_, r_ in zip(r0["sig"], ref_sig)]
    Srel = max(abs(c_[2] - r_[2]) / r_[2] for c_, r_ in zip(r0["sig"], ref_sig))
    rec_s = {"significance": {"same_set": f"{same_set}/{a.B}", "max_nsig_diff": int(max(nd)),
                              "nsig_ref_median": float(np.median([r_[0] for r_ in ref_sig])),
                              "nsig_ref_max": int(max(r_[0] for r_ in ref_sig)), "weight_sum_rel": float(Srel),
                              "frac": a.frac}}
    print(json.dumps(rec_s), flush=True)
    fh.write(json.dumps(rec_s) + "\n")
    grade(f"RESIDENT chip n{n} (proj+cut+oz+combine on chip)", 2 * r0["D"], dict(n=n, ns_n=a.ns_n or n, proj_rel_vs_f64=prel, **r0["diag"]))
    clk = Clock(nodes)
    ts = []
    for _ in range(a.reps):
        ts.append(chip_pass(n)["t"])
    lo_c, med = clk.stop()
    tw = {k_: round(min(t_[k_] for t_ in ts), 4) for k_ in ts[0]}
    fl = n * (n + 1) / 2 * 2 * (len(rows) + a.B) * a.O * Kp      # Ozaki pair matmuls, C and |CTF P|^2 terms
    rec = {"timing_warm": tw, "fused": a.fused,
           "oz_TFLOPs_incl_cut_combine" if a.fused else "oz_TFLOPs": round(fl / tw.get("oz_combine", tw.get("cut_oz_combine")) / 1e12, 1), "real_TFLOP_equiv_per_s":
           round(2 * len(rows) * a.O * K / sum(tw.values()) / 1e12, 3), "R": len(rows), "O": a.O, "K": K,
           "P": P, "aiclk_min": lo_c, "aiclk_med": med, "load1": open("/proc/loadavg").read().split()[0], "nodes": nodes}
    print(json.dumps(rec), flush=True)
    fh.write(json.dumps(rec) + "\n")
