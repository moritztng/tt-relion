#!/bin/bash
# KERNELS-PREP chain on one chip: box-360 diff2 parity sweep, then the wavg and FFT prototypes.
# Run via when_free.sh (it pins the card and sets the lease identity). R = run dir.
R=$1; PY=${PY:-$HOME/tt-bio-dev/env/bin/python}
cd "$(dirname "$0")/../.." && export PYTHONPATH=$PWD
timeout 7200 $PY perf/rlx/diff2_parity360.py --out $R/res.jsonl; echo "parity rc=$?"
timeout 3600 $PY perf/rlx/proto_wavg_fft.py --out $R/proto.jsonl; echo "proto rc=$?"
