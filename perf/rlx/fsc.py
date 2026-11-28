"""FSC between two MRC maps of the same box, and the gate of one map against a seed pair.

  python fsc.py A.mrc B.mrc                 resolution at FSC 0.5 and 0.143, correlation, lowest shell
  python fsc.py A.mrc B.mrc --curve         the same, then one line per shell: index, resolution in A, FSC
  python fsc.py REF.mrc X.mrc --floor S.mrc X against REF, shell by shell beside the seed pair REF against S:
                                            passes when FSC(REF, X) >= FSC(REF, S) - 0.02 on every shell up to the
                                            shell where FSC(REF, S) drops under 0.143
"""
import sys

import numpy as np


def read(p):
    h = np.fromfile(p, np.int32, 256)
    n, mode, nsym = h[:3], h[3], h[23]
    assert mode == 2, mode
    px = np.fromfile(p, np.float32, 256)[10] / n[0]
    return np.fromfile(p, np.float32, offset=1024 + nsym).reshape(n[2], n[1], n[0]).astype(np.float64), px


def shells(n):
    f = [np.fft.fftfreq(n)] * 2 + [np.fft.rfftfreq(n)]
    s = np.rint(np.sqrt(sum(g ** 2 for g in np.meshgrid(*f, indexing="ij"))) * n).astype(int).ravel()
    return s, s < n // 2


def fsc(a, b):
    A, B = np.fft.rfftn(a), np.fft.rfftn(b)
    s, m = shells(a.shape[0])
    num = np.bincount(s[m], (A * B.conj()).real.ravel()[m])
    return num / np.sqrt(np.bincount(s[m], (abs(A) ** 2).ravel()[m]) * np.bincount(s[m], (abs(B) ** 2).ravel()[m]))


def cut(c, t):
    """First shell under t (len(c) when none is)."""
    i = int(np.argmax(c[1:] < t)) + 1
    return i if c[i] < t else len(c)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    a, px = read(args[0])
    b, _ = read(args[1])
    n, c = a.shape[0], fsc(a, b)
    res = lambda i: n * px / i if i < len(c) else 2 * px  # noqa: E731
    print("px %.4f A  corr %.6f  FSC=0.5 at %.2f A  FSC=0.143 at %.2f A  min FSC to Nyquist %.4f" % (
        px, np.corrcoef(a.ravel(), b.ravel())[0, 1], res(cut(c, 0.5)), res(cut(c, 0.143)), c[1:].min()))
    if "--curve" in sys.argv:
        for i in range(1, len(c)):
            print("%4d %8.2f %.4f" % (i, n * px / i, c[i]))
    if "--floor" in sys.argv:
        f = fsc(a, read(args[2])[0])
        top = cut(f, 0.143)
        d = c[1:top] - f[1:top]
        w = int(np.argmin(d)) + 1
        print("seed pair: FSC=0.143 at %.2f A (shell %d). Worst shell %d (%.2f A): %.4f against the pair's %.4f. %s" % (
            res(top), top, w, n * px / w, c[w], f[w], "PASS" if d.min() >= -0.02 else "FAIL"))


if __name__ == "__main__":
    main()
