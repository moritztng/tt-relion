#!/bin/bash
# e2e_chain.sh: the prep A/B inside rlx-e2e's one-iteration arm T (iter_ab.sh: resident E-step on $CHIPS,
# one rank per chip plus the leader, --j 40, TT_RELION_BATCH 32, 13 digits, hugetlb, profiler on). One binary, three prep modes:
#   n  TT_RELION_PREP=0 (RELION's own code)   e  TTPrep, estimated plans (default)   w  TTPrep, plans from $R/wisdom.360
# and a control s: the binary without the prep patch ($SBIN), which says whether n is that binary's result.
# Per arm: prep thread CPU ms per particle (split.py), the E-step bar, AICLK of its chips sampled at 1 Hz while
# it ran, pose rows and half-map correlation against rlx-e2e's CPU arm and against this chain's own n arm.
# Arms never run into a booked whole-box window (~/fleet/HOLDER.windows), stop at ~/fleet/HANDBACK and start only
# with $MINMEM GB of MemAvailable, so a neighbour's memory-bound run is not pushed over.
# An arm whose line is already in the status file is skipped, so a restarted chain goes on where the last one stopped.
# Env: PLAN (arms as tag:mode:iteration), SBIN, CHIPS (16,...,23; chip C in 16-23 is node C+8), MINMEM (150), R (~/rlx-prep), SRC (tree under $R, src-f; exported, iter_ab.sh takes its Python tree and run.sh from it), RELION_BIN, P (tag prefix, pe), GIVEUP (epoch), E2E (~/rlx-e2e/runs).
R=${R:-$HOME/rlx-prep}; SRC=${SRC:-src-f}; P=${P:-pe}; S=$R/runs/$P.status; PY=${PY:-$HOME/fleet/env/bin/python}
E2E=${E2E:-$HOME/rlx-e2e/runs}; CHIPS=${CHIPS:-16,17,18,19,20,21,22,23}; MINMEM=${MINMEM:-150}; NC=$(echo $CHIPS | tr , '\n' | wc -l); NODES=$(for c in ${CHIPS//,/ }; do echo -n "$((c + 8)) "; done)
GIVEUP=${GIVEUP:-$(( $(date +%s) + 4 * 3600 ))}; W=$R/wisdom.360
export R SRC; BIN=${RELION_BIN:-$R/relion-f/build/bin/relion_refine_mpi}; SBIN=${SBIN:-$HOME/rlx-e2e/relion-s/build/bin/relion_refine_mpi}
PLAN=${PLAN:-n16a:n:16 e16a:e:16 w16a:w:16 e16b:e:16 n16b:n:16 s16a:s:16 n24a:n:24 e24a:e:24 w24a:w:24 e24b:e:24 n24b:n:24 s24a:s:24}
say() { echo "$(date -u +%FT%TZ) $* load1 $(cut -d' ' -f1 /proc/loadavg)" >> $S; }
stop() { [ -e $HOME/fleet/HANDBACK ] && { say HANDBACK; exit 0; }; [ $(date +%s) -gt $GIVEUP ] && { say "past give-up time"; exit 0; }; }
busy() { [ -n "$(fuser /dev/tenstorrent/$1 2>/dev/null)" ]; }
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
room() { now=$(date +%s); r=86400       # seconds until 2 min before the next window of another row; 0 inside one
  while read -r slug a b _; do case "$slug" in ""|\#*|rlx*) continue;; esac
    s=$(date -d "$a" +%s 2>/dev/null) || continue; e=$(date -d "$b" +%s 2>/dev/null) || continue
    [ $now -ge $e ] && continue
    d=$(( s - 120 - now )); [ $d -lt 0 ] && d=0; [ $d -lt $r ] && r=$d
  done < $HOME/fleet/HOLDER.windows; echo $r; }
slot() {    # wait for $1 s of room, the memory floor and every chip of $CHIPS free; sets END
  while :; do stop
    if [ $(room) -ge $1 ] && [ $(avail) -ge $MINMEM ]; then
      n=0; for d in $NODES; do busy $d || n=$((n + 1)); done
      [ $n = $NC ] && break
    fi; sleep 20
  done
  r=$(room); [ $r -gt 2700 ] && r=2700; END=$(( $(date +%s) + r )); }
estep() { tr '\r' '\n' < $1 | awk '/Maximization/{exit} match($0,/^ *[0-9.]+\/[0-9.]+ min/){x=substr($0,RSTART,RLENGTH); sub(/^ */,"",x); split(x,a,"[/ ]"); if(a[1]==a[2]) e=a[1]} END{print e}'; }
clk() { awk -F'\t' -v nodes="$NODES" 'BEGIN{split(nodes,q," "); for(k in q) want[q[k]]=1} NR==1{for(i=2;i<=NF;i++) if($i in want) c[i]=1; next} NF<33{next}
    {a=1; for(i in c) if($i+0<1200) a=0; if(a) up=1; if(!up) next
     for(i in c){n++; if($i+0<1200) lo++; if(m==""||$i+0<m) m=$i+0}} END{print "aiclk nodes " nodes "samples " n+0 " under1200 " lo+0 " min " m}' $1 2>/dev/null; }
arm() {     # arm <tag> <mode n|e|w|s> <it> <rlx-e2e CPU arm dir> [<own n arm tag>]
  grep -q " $1 mode=$2 it0$3 .* T rc=" $S && return
  tag=${P}_$1; it=$3; o=$R/runs/${tag}_T; slot 480; export RELION_BIN=$BIN
  case $2 in n) export TT_RELION_PREP=0; unset TT_RELION_PREP_WISDOM;; e) export TT_RELION_PREP=1; unset TT_RELION_PREP_WISDOM;;
             w) export TT_RELION_PREP=1 TT_RELION_PREP_WISDOM=$W;; s) export RELION_BIN=$SBIN; unset TT_RELION_PREP TT_RELION_PREP_WISDOM;; esac
  [ $it = 24 ] && export FROM=$HOME/rlx/runs/d32_l_n14/run_it023_optimiser.star || unset FROM
  IT=$it TAG=$tag ARMS=T TT_RELION_RESIDENT=1 TT_RELION_BATCHLOG=1 RANKS=$((NC + 1)) J=40 BATCH=32 RLX_CHIP_LIST=$CHIPS END_BY=$END \
    TT_RELION_PROFILE_FILE=$o/prof $R/$SRC/perf/rlx/iter_ab.sh
  $PY $R/$SRC/perf/rlx/wsum/split.py $o > $o/split.txt 2>&1
  say "$1 mode=$2 it0$it chips $CHIPS MemAvailable $(avail) GB: $(grep "^T rc" $R/runs/$tag.status | tail -1) E-step $(estep $o/refine.log) min; $(clk $o/aiclk.tsv) | $(grep -m1 fft_ctf $o/split.txt | xargs) | $(grep -m1 ttPrep $o/split.txt | xargs) | $(grep -c 'TTBridge: .* raised\|declined: ' $o/refine.log) fallback lines"
  for ref in $4 ${5:+$R/runs/${P}_$5_T}; do
    [ -e $ref/run_it0${it}_data.star ] || continue
    echo "-- $1 vs $ref" >> $S
    $PY $R/$SRC/perf/rlx/star_cmp.py $ref/run_it0${it}_data.star $o/run_it0${it}_data.star 2>&1 | grep "^rows\|Angle\|Origin" >> $S
    $PY $R/$SRC/perf/rlx/d32_cmp.py $it $ref $o 2>&1 | tail -1 >> $S
  done; }

export GLIBC_TUNABLES=glibc.malloc.hugetlb=1 TT_RELION_PROFILE=1 TT_RELION_DIGITS=13
unset TT_RELION_BATCH_ROWS TT_RELION_MIN_ORI
say "start bin $BIN tree $SRC chips $CHIPS plan $PLAN"
for a in $PLAN; do IFS=: read -r tag m it <<< "$a"
  [ $it = 24 ] && ref=$E2E/dc8_N || ref=$E2E/dc2_N
  arm $tag $m $it $ref $([ $tag != n${it}a ] && echo n${it}a)
done
say done
