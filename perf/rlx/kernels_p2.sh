#!/bin/bash
# Pass-2 device chain: BH matmul accumulation probe, then the on-device projection prototype.
set -u
cd "$(dirname "$0")/../.."
R=${1:?run dir}
PY=~/tt-bio-dev/env/bin/python
timeout 1200 $PY perf/rlx/probe_acc.py "$R/acc.jsonl"; echo "probe rc=$?"
timeout 2400 $PY perf/rlx/proj_proto.py --sizes 50,100,146 --out "$R/proj.jsonl"; echo "proj rc=$?"
timeout 2400 $PY perf/rlx/proj_proto.py --sizes 360 --oc 32 --out "$R/proj.jsonl"; echo "proj360 rc=$?"
echo P2-DONE
