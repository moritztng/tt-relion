"""device_server.py with no device, so the bridge can be timed and checked on any host.

Runs the same tt_relion/cross.py server loop (bf16 split, operand build, float64 chunk sum, kept
operands). RLX_STUB_MATMUL=1 does the matmul in fp32 numpy, so bridge-side changes can be checked for
identical output; without it mm returns zeros of the right shape and the timing is host work only.
argv[1] is the fidelity (ignored).
"""
import os
import sys

import numpy as np


class _Numpy:
    def __init__(self):
        self.real = os.environ.get("RLX_STUB_MATMUL") == "1"
        self.models = {}

    def upload(self, lhs):
        return lhs

    def shape(self, a):
        return a.shape[0], a.shape[2]

    def mm(self, a, rhs):
        if self.real:
            return np.matmul(a, rhs)
        return np.zeros((a.shape[0], a.shape[1], rhs.shape[2]), np.float32)

    def set_model(self, mkey, mdl, geom):
        import torch
        self.models[mkey] = torch.from_numpy(mdl.copy())
        while len(self.models) > 2:
            self.models.pop(next(iter(self.models)))

    def project(self, mkey, eul, g, w):
        """The bridge's own torch fp32 _project, then the same lhs layout an uploaded reference gets."""
        import torch
        from tt_relion.cross import KCHUNK, pixels, proj_lhs
        from tt_relion.relion import _project
        x, y = (torch.from_numpy(v) for v in pixels(g))
        r, i = _project(torch, self.models[mkey], x, y, torch.from_numpy(eul.copy()), g["mdlX"], g["mdlY"],
                        g["mdlZ"], g["mdlInitY"], g["mdlInitZ"], g["maxR2"], g["pf"])
        ref = torch.cat([r, i], 1).numpy()
        ref2 = ((ref.astype(np.float64) ** 2) * np.concatenate([w, w])).sum(1)
        return proj_lhs(ref, g["P"], KCHUNK), ref2

    def free(self, a):
        pass

    def close(self):
        pass


def main():
    from tt_relion.cross import serve
    serve(_Numpy(), sys.stdout)


if __name__ == "__main__":
    main()
