#!/usr/bin/env bash
# 30 Sep sitting poll (read-only): queue, sacct since 29 Sep noon, maintenance evidence, TRAJ lines per cell.
B=/home/paretsky/scratch_audit/tree_v9b
C=/home/paretsky/scratch_audit/tree_v9c
echo "=== $(date '+%d %b %H:%M')"
squeue -u paretsky -h -o "%.10i %.24j %.2t %.10M %.10l %.5y %N %R" 2>/dev/null
echo "=== sacct since 29 Sep 12:00"
sacct -u paretsky -S 2026-09-29T12:00 -X -n \
  -o JobID%10,JobName%24,State%16,ExitCode%6,Elapsed%11,Start%19,End%19,NodeList%14 2>/dev/null
echo "=== reservations"
scontrol show reservation 2>/dev/null | grep -E "ReservationName|StartTime|Flags" | head -12
echo "=== restarts / reasons"
for j in $(squeue -u paretsky -h -o %i); do
  echo "$j $(scontrol show job $j 2>/dev/null | grep -oE 'Restarts=[0-9]+|Reason=[^ ]+|Dependency=[^ ]+' | tr '\n' ' ')"
done
for j in ${JOBS:-21729551 21729552 21729553 21729554 21729555 21729556 21729557 21729558 21730498 21730499 21730500 21730501 21730506 21730507 21730509 21730514 21730516}; do
  f=$B/runs/job$j/logs/rank0.log; [[ -f $f ]] || f=$C/runs/job$j/logs/rank0.log
  [[ -f $f ]] || { echo "--- $j: no log"; continue; }
  echo "--- $j $(sacct -j $j -X -n -o JobName%24,State%12,Elapsed,NodeList%14 2>/dev/null | head -1)"
  grep -E "\[eval\] (policy=|TRAJ (val_best|size_|final_ft))|Traceback|Error|eval_network_failed|TRAJ save failed|final_ft_failed" "$f" \
    | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar(10|100)_[a-z_-]+_[0-9._]+\.pth?//; s/ \| FLOPs x[0-9.]+//' | cut -c1-230 | tail -${TAIL:-40}
  n=$(grep -c "Step [0-9]* done" "$f")
  echo "    steps: $n; last: $(grep "Step [0-9]* done" "$f" | tail -1 | sed -E 's/^.*\| (Step)/\1/' | cut -c1-100)"
done
