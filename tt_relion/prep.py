"""Float64 reference of RELION's particle prep (getFourierTransformsAndCtfs; 2D image, --zero_mask).

Computed from a record's inputs alone (tt_prep.h, TT_RELION_RECORD_PREP), so it grades RELION's own path and the
TTPrep path against the same numbers. The image is taken as the float32 values the kernels see; everything
after that is float64. The CTF is not here yet.
"""
import numpy as np


def read_record(path):
    """A record file as {name: array or float}: items of 'name dtype ydim xdim\\n' followed by raw values."""
    out = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return out
            name, dtype, y, x = line.decode().split()
            a = np.frombuffer(f.read(np.dtype(dtype).itemsize * int(y) * int(x)), dtype=dtype).reshape(int(y), int(x))
            out[name] = float(a[0, 0]) if a.shape == (1, 1) and dtype == "f8" and name != "power" else a


def shifted(img, norm, dx, dy):
    """img * norm moved by whole pixels; what leaves the box is dropped, what is uncovered is zero."""
    n = img.shape[0]
    out = np.zeros((n, n))
    y0, y1, x0, x1 = max(0, -dy), min(n, n - dy), max(0, -dx), min(n, n - dx)
    if y0 < y1 and x0 < x1:
        out[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = img[y0:y1, x0:x1].astype(np.float32).astype(np.float64) * norm
    return out


def mask_weights(n, radius, edge):
    """0 inside the mask, raised cosine across its edge, 1 outside."""
    if radius < 0:
        radius = n / 2
    y, x = np.mgrid[:n, :n] - n // 2
    r = np.sqrt(x * x + y * y)
    w = 0.5 + 0.5 * np.cos((radius + edge - r) / edge * np.pi)
    return np.where(r < radius, 0.0, np.where(r > radius + edge, 1.0, w))


def masked(a, w):
    """Outside the mask the image becomes its own weighted mean there."""
    bg = (w * a).sum() / w.sum()
    return a * (1 - w) + bg * w


def transform(a):
    """Centred FFT, normalised by the pixel count: n x (n/2+1)."""
    return np.fft.rfft2(np.fft.fftshift(a)) / a.size


def window(F, cur):
    """The cur x (cur/2+1) low-frequency window."""
    n, ox = F.shape[0], cur // 2 + 1
    rows = [i if i < ox else i - cur + n for i in range(cur)]
    return F[rows][:, :ox]


def power(F, cur):
    """Power per shell of the full transform and its sum over shells past the current size."""
    n, ix = F.shape
    y = np.arange(n)[:, None]
    y = np.where(y < ix, y, y - n)
    x = np.arange(ix)[None, :]
    shell = (np.sqrt(x * x + y * y) + 0.5).astype(int)
    ok = (shell < ix) & ~((x == 0) & (y < 0))
    spectrum = np.bincount(shell[ok], weights=(np.abs(F) ** 2)[ok], minlength=ix)
    return spectrum, spectrum[cur // 2 + 1:].sum()


def reference(rec):
    """{Fimg_nomask, Fimg, power, highres_Xi2} for one record."""
    img = rec["img"]
    n, cur = img.shape[0], int(rec["current_size"])
    a = shifted(img, rec["norm"], int(rec["dx"]), int(rec["dy"]))
    m = transform(masked(a, mask_weights(n, float(np.float32(rec["radius"])), float(np.float32(rec["edge"])))))
    out = {"Fimg_nomask": window(transform(a), cur), "Fimg": window(m, cur), "power": np.zeros((1, 0)), "highres_Xi2": 0.0}
    if cur < n:
        spectrum, out["highres_Xi2"] = power(m, cur)
        out["power"] = spectrum[None, :]
    return out
