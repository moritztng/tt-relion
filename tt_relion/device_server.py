"""The chip, in a process of its own.

RELION's ranks run on the system Open MPI, and ttnn ships a second Open MPI that tt-metal always
initialises. Both in one address space abort the rank ("MPI_Init invoked more than once"). So each rank's
bridge starts one of these servers with mpirun's variables removed; it owns the rank's chip (the inherited
TT_VISIBLE_DEVICES) and does the matmuls.

Protocol, one JSON line per request on stdin, one line back:
    {"shm": <name>, "shape": [nc, m, kc, n], "client": {...}, "lhs_key": <str, optional>}  ->  "ok"
"lhs_key" keeps lhs on the chip: a request with the key of the lhs already held skips its upload (the
bridge does not even write it into the segment), a new key replaces it. The coarse pass keys its
reference projections by iteration, so a batch ships only its particles.
"sum0": true adds the nc partial products on the chip (fp32) and returns only out[0] = their sum.
"client" carries the bridge's cumulative counters (round trips, copies, bytes); the server only echoes the
last one in its exit line, so the WALLMAP gets the client side without RELION having to flush anything.
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
    held_key, held = None, None
    t_open, calls, client = time.time(), 0, {}
    up = mm = down = 0.0
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
        key = req.get("lhs_key")
        if key is not None and key == held_key:
            a = held
        else:
            a = ttnn.from_torch(torch.from_numpy(lhs), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            if key is not None:
                if held is not None:
                    ttnn.deallocate(held)
                held_key, held = key, a
        b = ttnn.from_torch(torch.from_numpy(rhs), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn.synchronize_device(dev)
        t1 = time.time()
        c = ttnn.matmul(a, b, compute_kernel_config=cfg)
        if req.get("sum0") and nc > 1:
            parts = [ttnn.slice(c, [i, 0, 0], [i + 1, m, n]) for i in range(nc)]
            acc = parts[0]
            for p_ in parts[1:]:
                nxt = ttnn.add(acc, p_)
                if acc is not parts[0]:
                    ttnn.deallocate(acc)
                acc = nxt
            for p_ in parts:
                ttnn.deallocate(p_)
            ttnn.deallocate(c)
            c = acc
        ttnn.synchronize_device(dev)
        t2 = time.time()
        r = ttnn.to_torch(c).numpy()
        out.reshape(-1)[:r.size] = r.reshape(-1)
        if a is not held:
            ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.deallocate(c)
        del lhs, rhs, out, buf
        t3 = time.time()
        up, mm, down = up + t1 - t0, mm + t2 - t1, down + t3 - t2
        client = req.get("client", client)
        calls += 1
        reply.write("ok\n")
    if shm is not None:
        shm.close()
    ttnn.close_device(dev)
    # The WALLMAP reads this (perf/rlx/wallmap.py). busy_s = up + mm + down; mm_s is the matmul alone,
    # synchronised; the client fields are the bridge's (round trip incl. the server, shm copies, bytes).
    print("TTDEV chip=%s calls=%d busy_s=%.3f up_s=%.3f mm_s=%.3f down_s=%.3f life_s=%.3f client=%s"
          % (os.environ.get("TT_VISIBLE_DEVICES", "?"), calls, up + mm + down, up, mm, down,
             time.time() - t_open, json.dumps(client, separators=(",", ":"))), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
