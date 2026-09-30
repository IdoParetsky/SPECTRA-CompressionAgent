#!/usr/bin/env bash
# Lean progress read for the thin-aug decider (556), the aug twins (553) and the queue.
B=/home/paretsky/scratch_audit/tree_v9b/runs
PY=/home/paretsky/.conda/envs/spectra/bin/python
date "+=== %d %b %H:%M"
squeue -u paretsky -h -o "%.10i %.24j %.2t %.10M %R" 2>/dev/null | grep -v JobHeldUser
for j in 21729556 21729553; do
  L=$B/job$j/logs/rank0.log
  echo "--- $j"; grep -E '\[eval\] TRAJ (val_best|size_)' "$L" | sed -E 's/^.*\[eval\]/[eval]/' | cut -c1-150 | tail -4
  grep -E 'Step [0-9]+ done' "$L" | tail -1 | sed -E 's/^.*(Step)/\1/' | cut -c1-110
done
cd /home/paretsky/scratch_audit/tree_v9c
$PY scripts/paired_steps.py $B/job21729556 $B/job21729555 2>/dev/null | grep resnet56
$PY scripts/paired_steps.py $B/job21729553 $B/job21726337 2>/dev/null
