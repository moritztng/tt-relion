#!/usr/bin/env python3
"""Throughput and accuracy of the fused FFT at box 256 bf16, data resident on device.

    python perf/fft2d/bench.py [--device 0] [--images-per-core 16]

Record the AICLK alongside any number you quote: throughput scales with the clock.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ttnn  # noqa: E402

from tt_relion.fft2d import FFT2D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--images-per-core", type=int, default=16)
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
dev = ttnn.open_device(device_id=a.device)
try:
    print(json.dumps(FFT2D(dev, box=256, images_per_core=a.images_per_core).benchmark(reps=a.reps), indent=1))
finally:
    ttnn.close_device(dev)
