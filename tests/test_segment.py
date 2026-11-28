"""The bridge's shared segment loses its name when the server attaches, and both sides keep the memory."""
import os
import subprocess
import sys

import numpy as np

SERVER = """
import sys
from multiprocessing import shared_memory
import numpy as np
shm = shared_memory.SharedMemory(name=sys.argv[1])
shm.unlink()                                    # the device server's two lines
buf = np.ndarray((int(sys.argv[2]),), np.float32, shm.buf)
print(float(buf.sum()), flush=True)
sys.stdin.readline()
print(float(buf.sum()), flush=True)
"""


def test_segment_name_goes_when_the_server_attaches():
    from tt_relion.relion import _Server
    s = object.__new__(_Server)
    s.shm = None
    buf = s._segment(4 * 1024)
    name = s.shm.name
    buf[:] = 1
    p = subprocess.Popen([sys.executable, "-c", SERVER, name, "1024"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    assert float(p.stdout.readline()) == 1024
    assert not os.path.exists("/dev/shm/" + name)
    buf[:] = 2                                  # a later request on the same segment still reaches the server
    p.stdin.write("\n")
    p.stdin.flush()
    assert float(p.stdout.readline()) == 2048
    assert p.communicate()[1] == "" and p.returncode == 0
    assert s._segment(4 * 512).size == 512 and s.shm.name == name       # a smaller request reuses it
    del buf
    big = s._segment(4 * 4096)                  # a larger one replaces it; the old one was already unlinked
    assert s.shm.name != name and big.size == 4096
    del big
    os.unlink("/dev/shm/" + s.shm.name)         # this one never met a server
    s.shm.close()
