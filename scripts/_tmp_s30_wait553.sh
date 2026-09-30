#!/usr/bin/env bash
B=/home/paretsky/scratch_audit/tree_v9b/runs; R=/home/paretsky/scratch_audit/readers_s30/scripts
PY=/home/paretsky/.conda/envs/spectra/bin/python
A=$B/job21729553/logs/rank0.log
for i in $(seq 1 45); do
  if grep -qE "\[eval\] TRAJ size_param0.70 resnet56" $A 2>/dev/null; then break; fi
  sleep 60
done
date '+%H:%M:%S'
echo "=== ARM 21729553 (aug) R56"
grep -E "\[eval\] TRAJ (val_best|size_|terminal|floor_hold) resnet56" $A | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar10_chenyaofo_[0-9._]+\.pt//' | cut -c1-160
echo "=== CTRL 21726337 (P) R56"
grep -E "\[eval\] TRAJ (val_best|size_|terminal|floor_hold) resnet56" $B/job21726337/logs/rank0.log | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar10_chenyaofo_[0-9._]+\.pt//' | cut -c1-160
echo "=== paired (arm vs ctrl), by step"
$PY $R/paired_steps.py $B/job21729553 $B/job21726337 2>&1 | grep -i resnet56 | cut -c1-200
echo "=== crossfit arm / ctrl"
$PY $R/crossfit_readout.py $B/job21729553 --taus 10,5 --sizes param:0.8,0.7 2>&1 | grep -A12 -i "resnet56" | head -14 | cut -c1-170
$PY $R/crossfit_readout.py $B/job21726337 --taus 10,5 --sizes param:0.8,0.7 2>&1 | grep -A12 -i "resnet56" | head -14 | cut -c1-170
squeue -j 21729553 -h -o "%i %T %M" 2>/dev/null