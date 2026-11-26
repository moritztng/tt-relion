"""device_server.py's protocol with no device, so perf/rlx/ipc_bench.py can price the transport alone.

argv[1] is the fidelity (ignored). RLX_STUB_COPY=1 makes the server read every input float and write
every output float once, the host-memory traffic the real server's from_torch / to_torch must do anyway;
without it the server only maps the segment and replies.
"""
import json
import os
import sys

import numpy as np
from multiprocessing import resource_tracker, shared_memory


def main():
    copy = os.environ.get("RLX_STUB_COPY") == "1"
    shm = None
    for line in sys.stdin:
        req = json.loads(line)
        if shm is None or shm.name != req["shm"].lstrip("/"):
            if shm is not None:
                shm.close()
            shm = shared_memory.SharedMemory(name=req["shm"])
            resource_tracker.unregister(shm._name, "shared_memory")
        if copy:
            nc, m, kc, n = req["shape"]
            buf = np.ndarray((nc * m * kc + nc * kc * n + nc * m * n,), np.float32, shm.buf)
            a = buf[:nc * m * kc].copy()
            b = buf[nc * m * kc:nc * m * kc + nc * kc * n].copy()
            buf[nc * m * kc + nc * kc * n:] = 0.0
            del a, b, buf
        sys.stdout.write("ok\n")
        sys.stdout.flush()
    if shm is not None:
        shm.close()


if __name__ == "__main__":
    main()
