"""The float64 reference of particle prep (tt_relion/prep.py) against RELION's definitions written out as loops.

Everything TTPrep and RELION's own path are graded against goes through this reference, so each piece is checked
here on a small box against a direct sum or a hand value. Host only:
    pytest tests/test_prep.py
"""
import numpy as np

from tt_relion import prep

N, CUR = 12, 8


def image(seed=0):
    return np.random.default_rng(seed).standard_normal((N, N)).astype(np.float32)


def dft(a, ky, kx):
    """One Fourier pixel of RELION's centred transform: origin at pixel n/2, divided by the pixel count."""
    n = a.shape[0]
    y, x = np.mgrid[:n, :n] - n // 2
    return (a * np.exp(-2j * np.pi * (ky * y + kx * x) / n)).sum() / a.size


def test_transform_and_window_are_the_centred_dft():
    a = image().astype(np.float64)
    F = prep.transform(a)
    assert F.shape == (N, N // 2 + 1)
    W = prep.window(F, CUR)
    assert W.shape == (CUR, CUR // 2 + 1)
    for i in range(CUR):
        ky = i if i < CUR // 2 + 1 else i - CUR
        for kx in range(CUR // 2 + 1):
            assert abs(W[i, kx] - dft(a, ky, kx)) < 1e-14


def test_shift_fills_with_zero_and_does_not_wrap():
    a = image()
    s = prep.shifted(a, 2.0, 3, -2)
    for y in range(N):
        for x in range(N):
            inside = 0 <= y + 2 < N and 0 <= x - 3 < N
            assert s[y, x] == (2.0 * float(a[y + 2, x - 3]) if inside else 0.0)
    assert not prep.shifted(a, 1.0, N, 0).any()


def test_mask_keeps_the_inside_and_levels_the_outside():
    w = prep.mask_weights(N, 3.0, 2.0)
    y, x = np.mgrid[:N, :N] - N // 2
    r = np.sqrt(x * x + y * y)
    edge = w[(r > 3) & (r < 5)]
    assert (w[r < 3] == 0).all() and (w[r > 5] == 1).all() and ((edge > 0) & (edge < 1)).all()
    assert abs(w[N // 2, N // 2 + 4] - 0.5) < 1e-15                     # half way across the edge
    a = image().astype(np.float64)
    m = prep.masked(a, w)
    assert np.array_equal(m[r < 3], a[r < 3])
    assert np.allclose(m[r > 5], (w * a).sum() / w.sum(), rtol=0, atol=1e-15)
    assert np.allclose(prep.masked(np.full((N, N), 1.5), w), 1.5, rtol=0, atol=1e-15)


def test_power_is_relions_shell_loop():
    F = prep.transform(image(1).astype(np.float64))
    spectrum, xi2 = np.zeros(N // 2 + 1), 0.0
    for i in range(N):                                                  # FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM
        ip = i if i < N // 2 + 1 else i - N
        for jp in range(N // 2 + 1):
            ires = int(np.sqrt(ip * ip + jp * jp) + 0.5)
            if ires < N // 2 + 1 and not (jp == 0 and ip < 0):
                spectrum[ires] += abs(F[i, jp]) ** 2
                if ires >= CUR // 2 + 1:
                    xi2 += abs(F[i, jp]) ** 2
    got, got_xi2 = prep.power(F, CUR)
    assert np.allclose(got, spectrum, rtol=1e-13, atol=0) and abs(got_xi2 - xi2) < 1e-13 * xi2


def test_ctf_hand_values():
    rec = dict(img=np.zeros((N, N), np.float32), current_size=CUR, ctf_padding=0, ctf_correction=1, ctf_kV=300.0,
               ctf_Cs=0.0, ctf_Q0=0.0, ctf_phase_shift=0.0, ctf_angle=30.0, ctf_defU=15000.0, ctf_defV=15000.0,
               angpix=1.5, ctf_intact_first_peak=0, ctf_Bfac=0.0, ctf_phase_flipped=0, ctf_only_flip_phases=0,
               ctf_scale=1.0, ctf_premultiplied=0)
    h, m, e, c0, v = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19, 299792458.0, 300e3
    lam = 1e10 * h / np.sqrt(2 * m * e * v * (1 + e * v / (2 * m * c0 * c0)))   # relativistic wavelength, Angstrom
    assert abs(lam - 0.019687) < 1e-6
    tol = 5e-4      # RELION's wavelength constants sit 6e-6 from today's: 2e-4 at these phases; a wrong sign or axis is O(1)
    c = prep.ctf(rec)
    assert c.shape == (CUR, CUR // 2 + 1) and c[0, 0] == 1e-8          # zero at the origin, clamped as RELION does
    for i, kx in ((0, 3), (2, 1), (6, 4)):                             # row 6 is ky = -2
        ky = i if i < CUR // 2 + 1 else i - CUR
        u2 = (kx * kx + ky * ky) / (N * 1.5) ** 2
        assert abs(c[i, kx] - np.sin(np.pi * lam * 15000.0 * u2)) < tol
    rec.update(ctf_defU=20000.0, ctf_defV=10000.0, ctf_angle=0.0, ctf_Cs=2.7, ctf_Q0=0.1)
    c, u = prep.ctf(rec), 3 / (N * 1.5)                                 # along x the defocus is defU
    gamma = -np.pi * lam * 20000.0 * u * u + np.pi / 2 * 2.7e7 * lam ** 3 * u ** 4 - np.arcsin(0.1)
    assert abs(c[0, 3] + np.sin(gamma)) < tol
    assert abs(prep.ctf(dict(rec, ctf_premultiplied=1))[0, 3] - c[0, 3] ** 2) < 1e-15
    c, u2 = prep.ctf(dict(rec, ctf_angle=45.0)), 2 * (2 / (N * 1.5)) ** 2   # defU along the direction of the angle
    for i, d in ((2, 20000.0), (6, 10000.0)):                          # (ky, kx) = (2, 2) and (-2, 2)
        gamma = -np.pi * lam * d * u2 + np.pi / 2 * 2.7e7 * lam ** 3 * u2 * u2 - np.arcsin(0.1)
        assert abs(c[i, 2] + np.sin(gamma)) < tol
