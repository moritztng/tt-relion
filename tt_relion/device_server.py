"""The chip, in a process of its own.

RELION's ranks run on the system Open MPI, and ttnn ships a second Open MPI that tt-metal always
initialises. Both in one address space abort the rank ("MPI_Init invoked more than once"). So each rank's
bridge starts one of these servers with mpirun's variables removed; it owns the rank's chip (the inherited
TT_VISIBLE_DEVICES) and computes the cross term. Protocol and numerics: tt_relion/cross.py. Kept left
operands stay on the chip as device tensors, so a coarse call ships only the shifted image up.
"""
import os
import sys


class _Ttnn:
    def __init__(self, fidelity):
        import torch
        import ttnn
        self.torch, self.ttnn = torch, ttnn
        self.dev = ttnn.open_device(device_id=0)
        self.cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                                                    math_approx_mode=False, fp32_dest_acc_en=True,
                                                    packer_l1_acc=True)

    def _put(self, a):
        return self.ttnn.from_torch(self.torch.from_numpy(a), dtype=self.ttnn.float32,
                                    layout=self.ttnn.TILE_LAYOUT, device=self.dev)

    def upload(self, lhs):
        return (self._put(lhs), lhs.shape[0], lhs.shape[2])

    def shape(self, a):
        return a[1], a[2]

    def mm(self, a, rhs):
        ttnn = self.ttnn
        b = self._put(rhs)
        c = ttnn.matmul(a[0], b, compute_kernel_config=self.cfg)
        out = ttnn.to_torch(c).numpy()
        ttnn.deallocate(b)
        ttnn.deallocate(c)
        return out

    def free(self, a):
        self.ttnn.deallocate(a[0])

    def close(self):
        self.ttnn.close_device(self.dev)


def main():
    reply = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)                       # library chatter goes to stderr, never into the protocol
    from tt_relion.cross import serve
    serve(_Ttnn(sys.argv[1]), reply)


if __name__ == "__main__":
    main()
