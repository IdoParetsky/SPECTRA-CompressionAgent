#!/usr/bin/env bash
B=/home/paretsky/scratch_audit/tree_v9b/runs
sacct -j 21729555,21729556 -X -n -o JobID%10,Elapsed%10,NodeList%14 2>/dev/null
for J in 21729555 21729556; do
  echo "--- $J r56-w4 epoch timing (first step with 40 epochs)"
  grep -E "resnet56-width4.*Epoch [0-9]+/12" $B/job$J/logs/rank0.log | sed -n '20,24p' | sed -E 's/^([0-9-]+ [0-9:.]+) .*step=([0-9]+).*(Epoch [0-9]+\/12).*/\1 step \2 \3/'
done
echo "--- control batch evidence"
grep -m3 -iE "adaptive batch|batch_size=|batch size" /home/paretsky/scratch_audit/tree_v7/runs/job21536396/logs/rank0.log | sed -E 's/^.*\| //' | cut -c1-180
echo "--- train batch evidence"
grep -m3 -iE "adaptive batch|batch_size=|batch size" /home/paretsky/scratch_audit/tree_v9c/runs/job21737123/logs/rank0.log | sed -E 's/^.*\| //' | cut -c1-180
grep -nE "num_workers" /home/paretsky/scratch_audit/tree_v9c/src/utils.py | head -6