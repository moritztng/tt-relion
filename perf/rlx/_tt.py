"""Device open + AICLK sampling shared by the perf/rlx prototypes.

The chip is opened through tt-bio's own path (tt_bio.tenstorrent.get_device), which sets whatever the box needs
(the 1x1 mesh descriptor on a p300 board, nothing extra on a Galaxy chip). A bare ttnn.open_device either spins or
TT_FATALs on a p300 and TT_THROWs in UMD on some boxes. Pin the chip with TT_VISIBLE_DEVICES.
"""
import os
import threading
import time

import numpy as np


def open_dev():
    """-> (device, sysfs node ids whose tt_aiclk to sample)."""
    from tt_bio import tenstorrent as T

    dev = T.get_device()
    # The nodes this process holds open: TT_VISIBLE_DEVICES ids are UMD (PCI-order) ids, not node numbers.
    nodes = _fd_nodes()
    print("device nodes:", nodes, flush=True)
    return dev, nodes


def _fd_nodes():
    n = set()
    for f in os.listdir("/proc/self/fd"):
        try:
            t = os.readlink("/proc/self/fd/" + f)
        except OSError:
            continue
        if t.startswith("/dev/tenstorrent/"):
            n.add(int(t.rsplit("/", 1)[1]))
    return sorted(n)


class Clock:
    """Samples tt_aiclk of the given nodes every 0.25 s until stop() -> (min, median) MHz."""

    def __init__(self, nodes):
        self.paths = ["/sys/class/tenstorrent/tenstorrent!%d/tt_aiclk" % n for n in nodes]
        self.v, self.run = [], True
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.th.start()

    def loop(self):
        while self.run:
            for p in self.paths:
                try:
                    self.v.append(int(open(p).read().split()[0]))
                except (OSError, ValueError):
                    pass
            time.sleep(0.25)

    def stop(self):
        self.run = False
        self.th.join()
        return (min(self.v), int(np.median(self.v))) if self.v else (None, None)
