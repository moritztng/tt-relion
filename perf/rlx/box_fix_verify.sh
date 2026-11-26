#!/bin/bash
# Verify the staging fix on a box that already holds the dev cell: restage (check_stacks.py re-fetches any stack
# carrying EBI's error body), then run the dev cell to convergence. PASS = run_class001.mrc written.
W=/work; cd $W; rm -f DONE FAILED
st(){ echo "$(date -u +%FT%TZ) $*" >> $W/box.log; }
bash stage.sh data dev.txt >> box.log 2>&1 || { st FAIL stage; touch FAILED; exit 1; }; cp dev.star data/
python3 check_stacks.py data/Particles | tail -3 >> box.log; st "stack check rc=${PIPESTATUS[0]}"
st "run V-n3-j4-auto"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 5 > smi-V.csv 2>&1 & SMI=$!
timeout 3600 bash refine_cmd.sh /opt/relion-fair data dev.star V-n3-j4-auto 3 4 30 0 > V-n3-j4-auto.out 2>&1; rc=$?; kill $SMI
if [ -s V-n3-j4-auto/run_class001.mrc ]; then st "V PASS rc=$rc $(grep -h Elapsed V-n3-j4-auto/time-v.txt)"
else st "V FAIL rc=$rc $(grep -m1 -h 'min_diff2\|ERROR' V-n3-j4-auto.out)"; fi
touch $W/DONE
