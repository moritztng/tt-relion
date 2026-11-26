"""The chip, in a process of its own.

RELION's ranks run on the system Open MPI, and ttnn ships a second Open MPI that tt-metal always
initialises. Both in one address space abort the rank ("MPI_Init invoked more than once"). So each rank's
bridge starts one of these servers with mpirun's variables removed; it owns the rank's chip (the inherited
TT_VISIBLE_DEVICES) and does the matmuls.

Protocol, one JSON line per request on stdin, one line back:
    {"shm": <name>, "shape": [nc, m, kc, n]}  ->  "ok"
The shared-memory segment holds lhs [nc, m, kc], rhs [nc, kc, n] and out [nc, m, n], float32, back to back.
EOF on stdin closes the device and exits.
"""
import json
import os
import sys
import time


def main():
    reply = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)                       # library chatter goes to stderr, never into the protocol
    from multiprocessing import resource_tracker, shared_memory

    import numpy as np
    import torch
    import ttnn

    dev = ttnn.open_device(device_id=0)
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, sys.argv[1]),
                                          math_approx_mode=False, fp32_dest_acc_en=True,
                                          packer_l1_acc=True)
    shm = None
    t_open, busy, calls = time.time(), 0.0, 0
    for line in sys.stdin:
        req = json.loads(line)
        t0 = time.time()
        if shm is None or shm.name != req["shm"].lstrip("/"):
            if shm is not None:
                shm.close()
            shm = shared_memory.SharedMemory(name=req["shm"])
            resource_tracker.unregister(shm._name, "shared_memory")   # the bridge owns it
        nc, m, kc, n = req["shape"]
        buf = np.ndarray((nc * m * kc + nc * kc * n + nc * m * n,), np.float32, shm.buf)
        lhs = buf[:nc * m * kc].reshape(nc, m, kc)
        rhs = buf[nc * m * kc:nc * m * kc + nc * kc * n].reshape(nc, kc, n)
        out = buf[nc * m * kc + nc * kc * n:].reshape(nc, m, n)
        a = ttnn.from_torch(torch.from_numpy(lhs), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        b = ttnn.from_torch(torch.from_numpy(rhs), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        c = ttnn.matmul(a, b, compute_kernel_config=cfg)
        out[...] = ttnn.to_torch(c).numpy()
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.deallocate(c)
        del lhs, rhs, out, buf
        busy += time.time() - t0
        calls += 1
        reply.write("ok\n")
    if shm is not None:
        shm.close()
    ttnn.close_device(dev)
    # The WALLMAP reads this: how much of its life the rank's chip spent on requests (upload, matmul,
    # readback), against the life of the server.
    print("TTDEV chip=%s calls=%d busy_s=%.3f life_s=%.3f" % (os.environ.get("TT_VISIBLE_DEVICES", "?"),
          calls, busy, time.time() - t_open), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
