#!/bin/bash
# after.sh: once box_timed.sh is DONE, test the odd rank counts the 1-GPU auto sweep skipped (n=5,7).
# If one beats dev-n3-j6-p100 (1451.9 s) by >3 %, rerun full seed 0 with it as full-best-s0.
cd /work; W=/work
while [ ! -f $W/DONE ]; do sleep 60; done
st(){ echo "$(date -u +%FT%TZ) $*" >> box.log; }
wall(){ python3 -c "import sys;h,m,s=(['0']+sys.argv[1].split(':'))[-3:];print(int(h)*3600+int(m)*60+float(s))" "$(awk -F': ' '/Elapsed/{print $2}' $1/time-v.txt)"; }
best=3 bt=1451.9
for n in 5 7; do name=dev-n$n-j6-p100; rm -rf $name
  nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 1 > smi-$name.csv 2>&1 & sp=$!
  st "run $name start ($n ranks --j 6 --pool 100 seed 0)"
  bash refine_cmd.sh /opt/relion-fair data dev.star $name $n 6 100 0 "" "" > $name.out 2>&1; rc=$?; kill $sp
  [ $rc = 0 ] || { st "run $name rc=$rc"; continue; }
  t=$(wall $name); st "run $name rc=0 dev wall ${t}s $(grep -h 'Final resolution' $name.out | tail -1)"
  python3 -c "import sys;sys.exit(0 if $t < $bt*0.97 else 1)" && { best=$n bt=$t; }
done
st "AFTER BEST n=$best (${bt}s)"
if [ $best != 3 ]; then name=full-best-s0; rm -rf $name
  nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 1 > smi-$name.csv 2>&1 & sp=$!
  st "run $name start ($best ranks --j 6 --pool 100 seed 0)"
  bash refine_cmd.sh /opt/relion-fair data full.star $name $best 6 100 0 "" "" > $name.out 2>&1; rc=$?; kill $sp
  st "run $name rc=$rc wall=$(awk -F': ' '/Elapsed/{print $2}' $name/time-v.txt) $(grep -h 'Final resolution' $name.out | tail -1)"
fi
touch $W/AFTER_DONE; st after-done
