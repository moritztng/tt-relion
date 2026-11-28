"""storeWeightedSums of one particle in float64: the reference the on-chip sums are graded against.

A record (TT_RELION_RECORD_SUMS=<dir>, integrations/relion/src_acc_tt/tt_sums_record.cpp) holds what RELION's two
kernels saw for one particle and what they wrote: sums.<pid>.<particle>.bin, the model once per rank in
model.<pid>.<k>.bin. `wavg` and `backproject` below recompute the outputs from the inputs alone.

With w'[o, t] = w[o, t] / sum_weight for the pairs at or above significant_weight (0 elsewhere), X the masked
image, N the unmasked one, R[o] the CTF-weighted projection and S[t] = exp(i (x tx[t] + y ty[t])):

    Z[o]   = sum_t w'[o, t] S[t]                       one matmul [O, T] x [T, P], shared by every sum below
    XA     = sum_o Re(R[o] conj(Z[o] X))
    AA     = sum_o (sum_t w'[o, t]) |R[o]|^2
    parts  = AA - 2 XA + |X|^2 sum_ot w'[o, t]          (= sum_ot w' |R[o] - S[t] X|^2, as |S| = 1)
    B[o]   = Minvsigma2 ctf Z[o] N                      the image back-projected along o
    F[o]   = (sum_t w'[o, t]) Minvsigma2 ctf^2          its weight, rank one

RELION compares and truncates in float32 (the significance test, the radius tests, floor of the model
coordinates); those are done in float32 here too, so the same samples and voxels take part. Everything summed is
float64.
"""
import glob
import os
import struct

import numpy as np

_TYPES = {b"f": np.float32, b"d": np.float64, b"i": np.int32, b"q": np.int64, b"Q": np.uint64}


def read(path):
    """One record file as {name: array, or a Python number for a single value}."""
    out = {}
    with open(path, "rb") as f:
        if f.read(8) != b"TTSUMS1\n":
            raise ValueError(f"{path}: not a sums record")
        while True:
            head = f.read(2)
            if not head:
                return out
            name = f.read(struct.unpack("<H", head)[0]).decode()
            dtype = _TYPES[f.read(1)]
            count = struct.unpack("<Q", f.read(8))[0]
            a = np.fromfile(f, dtype, count)
            if a.size != count:
                raise ValueError(f"{path}: {name} is cut short")
            out[name] = a[0].item() if count == 1 and not name.startswith(("fine_", "bp_", "wsum_", "Mresol", "data_vs")) else a


def load(path):
    """A particle's record with its rank's model volume as r["mdl"], [voxels, 2] float64."""
    r = read(path)
    pid = os.path.basename(path).split(".")[1]
    m = read(os.path.join(os.path.dirname(path), f"model.{pid}.{r['model']}.bin"))["mdlComplex"]
    r["mdl"] = m.astype(np.float64).reshape(-1, 2)
    return r


def records(directory):
    return sorted(glob.glob(os.path.join(directory, "sums.*.bin")))


def _weights(r):
    """w' as [orientations with a significant pair, T] float64, and those orientations' indices."""
    w = r["weights"].reshape(r["orientation_num"], r["translation_num"])
    w = np.where(w >= np.float32(r["significant_weight"]), w, 0).astype(np.float64) / r["weight_norm"]
    o = np.flatnonzero(w.any(1))
    return w[o], o


def _phases(r, x, y):
    """S as [T, P] complex128."""
    tx = r["trans_x"].astype(np.float64)[:, None]
    ty = r["trans_y"].astype(np.float64)[:, None]
    return np.exp(1j * (tx * x + ty * y))


def stats(r):
    """How sparse the particle's sums are: what the kernels walk against what carries weight."""
    w, o = _weights(r)
    return dict(orientations=r["orientation_num"], translations=r["translation_num"], pixels=r["image_size"],
                orientations_significant=len(o), pairs_significant=int((w > 0).sum()),
                translations_significant=int((w > 0).any(0).sum()))


def wavg(r):
    """wavg_ref3D: (wdiff2s_parts, wdiff2s_XA, wdiff2s_AA), each [image_size] float64."""
    import torch
    from .relion import _project
    nx, ny, maxR = r["imgX"], r["imgY"], r["maxR"]
    pix = np.arange(r["image_size"])
    x, iy = pix % nx, pix // nx
    wrapped = iy >= ny - maxR
    y = np.where((iy > maxR) & wrapped, iy - ny, iy)
    # rows between the two halves: RELION touches the one pixel at x = maxR there
    walked = (iy <= maxR) | wrapped | (x == maxR)

    w, o = _weights(r)
    e = torch.from_numpy(r["eulers"].reshape(-1, 9)[o])
    ref_r, ref_i = _project(torch, torch.from_numpy(r["mdl"]), torch.from_numpy(x.astype(np.float32)),
                            torch.from_numpy(y.astype(np.float32)), e, r["mdlX"], r["mdlY"], r["mdlZ"],
                            r["mdlInitY"], r["mdlInitZ"], r["maxR2_padded"], np.float32(r["padding_factor"]))
    ref = (ref_r.double() + 1j * ref_i.double()).numpy()
    ref *= r["ctfs"].astype(np.float64) if r["refs_are_ctf_corrected"] else r["part_scale"]

    X = r["Fimg_real"].astype(np.float64) + 1j * r["Fimg_imag"]
    Z = w @ _phases(r, x, y)
    sw = w.sum(1)[:, None]
    XA = (ref * np.conj(Z * X)).real.sum(0)
    AA = (sw * np.abs(ref) ** 2).sum(0)
    parts = AA - 2 * XA + np.abs(X) ** 2 * sw.sum()
    return parts * walked, XA * walked, AA * walked


def backproject(r):
    """backprojectRef3D of this particle alone into an empty half volume: (real, imag, weight), [voxels] float64."""
    nx, ny = r["imgX"], r["imgY"]
    pix = np.arange(r["image_size"])
    x, iy = pix % nx, pix // nx
    half = ny // 2
    y = np.where(iy > half, iy - ny, iy)
    keep = x < np.sqrt((half * half - y * y).astype(np.float32)).astype(np.int64)
    pix, x, y = pix[keep], x[keep], y[keep]

    w, o = _weights(r)
    ctf = r["ctfs"].astype(np.float64)[pix]
    m = r["Minvsigma2s"].astype(np.float64)[pix]
    N = (r["Fimg_nomask_real"].astype(np.float64) + 1j * r["Fimg_nomask_imag"])[pix]
    data_gain = m if r["ctf_premultiplied"] else m * ctf
    B = (w @ _phases(r, x, y)) * (data_gain * N)
    F = w.sum(1)[:, None] * (data_gain * ctf)

    f32 = np.float32
    e = r["eulers"].reshape(-1, 9)[o]
    pf = f32(r["bp_padding_factor"])
    xf, yf = x.astype(f32), y.astype(f32)
    xp = (e[:, 0:1] * xf + e[:, 1:2] * yf) * pf
    yp = (e[:, 3:4] * xf + e[:, 4:5] * yf) * pf
    zp = (e[:, 6:7] * xf + e[:, 7:8] * yf) * pf
    max_r2_vol = int(f32(r["bp_maxR2"]) * pf * pf)
    F = np.where(xp * xp + yp * yp + zp * zp > max_r2_vol, 0, F)
    neg = xp < 0
    xp, yp, zp = (np.where(neg, -c, c) for c in (xp, yp, zp))
    B = np.where(neg, np.conj(B), B)

    use = F > 0
    xp, yp, zp, B, F = xp[use], yp[use], zp[use], B[use], F[use]
    x0, y0, z0 = np.floor(xp), np.floor(yp), np.floor(zp)
    fx, fy, fz = ((c - c0).astype(np.float64) for c, c0 in ((xp, x0), (yp, y0), (zp, z0)))
    mx, my = r["bp_mdlX"], r["bp_mdlY"]
    base = ((z0.astype(np.int64) - r["bp_mdlInitZ"]) * my + y0.astype(np.int64) - r["bp_mdlInitY"]) * mx + x0.astype(np.int64)
    n = mx * my * r["bp_mdlZ"]
    real, imag, weight = np.zeros(n), np.zeros(n), np.zeros(n)
    for dz, gz in ((0, 1 - fz), (1, fz)):
        for dy, gy in ((0, 1 - fy), (1, fy)):
            for dx, gx in ((0, 1 - fx), (1, fx)):
                idx = base + (dz * my + dy) * mx + dx
                g = gz * gy * gx
                np.add.at(real, idx, g * B.real)
                np.add.at(imag, idx, g * B.imag)
                np.add.at(weight, idx, g * F)
    return real, imag, weight


def post(r, parts, XA, AA):
    """What RELION folds the per-pixel sums into: (sigma2_noise shells below the current size, their sum, scale XA,
    scale AA). The shells above the current size are the image's own power spectrum and never depend on the sums."""
    ires = r["Mresol_fine"].astype(np.int64)
    in_shell = ires > -1
    shells = np.bincount(ires[in_shell], parts[in_shell], r["image_current_size"] // 2 + 1)
    scaled = in_shell & (r["data_vs_prior"][np.maximum(ires, 0)] > 3)
    return shells, shells.sum(), XA[scaled].sum() / r["part_scale"], AA[scaled].sum() / r["part_scale"] ** 2


def recorded_backprojection(r):
    """The record's sparse private volume as three dense float64 arrays."""
    n = r["bp_mdlX"] * r["bp_mdlY"] * r["bp_mdlZ"]
    out = []
    for k in ("bp_real", "bp_imag", "bp_weight"):
        v = np.zeros(n)
        v[r["bp_index"].astype(np.int64)] = r[k]
        out.append(v)
    return out
