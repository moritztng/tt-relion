"""Exact operand splitting for the accumulation probes."""
import numpy as np

def bits_split(a, nbits, npieces):
    """a (float64) -> pieces each with <= nbits significant bits (incl. the leading one), exact in bf16."""
    parts, r = [], a.copy()
    for _ in range(npieces):
        e = np.floor(np.log2(np.abs(r) + 1e-300))
        q = np.ldexp(np.trunc(np.ldexp(r, -(e - nbits + 1).astype(int))), (e - nbits + 1).astype(int))
        q[np.abs(r) < 1e-300] = 0
        parts.append(q)
        r = r - q
    return parts
