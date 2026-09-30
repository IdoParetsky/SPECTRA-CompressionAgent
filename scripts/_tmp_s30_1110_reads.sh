#!/usr/bin/env bash
R=/home/paretsky/scratch_audit/readers_s30/scripts; PY=/home/paretsky/.conda/envs/spectra/bin/python
C=/home/paretsky/scratch_audit/tree_v9c/runs; B=/home/paretsky/scratch_audit/tree_v9b/runs
echo "=== 21730500 final FT (fixed reader)"
$PY $R/final_ft_readout.py $C/job21730500 | cut -c1-200
echo "=== 21730500 crossfit"
$PY $R/crossfit_readout.py $C/job21730500 --taus 10,5 --sizes flop:0.6,0.47,0.39 2>&1 | cut -c1-170
echo "=== 21730501 final FT so far"
$PY $R/final_ft_readout.py $C/job21730501 | cut -c1-200
echo "=== P gate 552 last net state"
grep -E "\[eval\] TRAJ (val_best|size_)" $B/job21729552/logs/rank0.log | sed -E 's/^.*\| \[eval\]/[eval]/' | awk '{print $4}' | uniq -c
tail -c 300 $B/job21729552/logs/rank0.log | tr '\r' '\n' | tail -1 | sed -E 's/^.*net=([^ ]+) .*step=([0-9]+).*/\1 step \2/'
echo "=== aug gate 554 all TRAJ val_best"
grep -E "\[eval\] TRAJ val_best" $B/job21729554/logs/rank0.log | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar100_[a-z_-]+_[0-9._]+\.(pt|pth)//' | cut -c1-150
echo "=== P gate 552 all TRAJ val_best"
grep -E "\[eval\] TRAJ val_best" $B/job21729552/logs/rank0.log | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar100_[a-z_-]+_[0-9._]+\.(pt|pth)//' | cut -c1-150
echo "=== crossfit aug gate (census) nets 7-8"
$PY $R/crossfit_readout.py $B/job21729554 --taus 10 --sizes param:0.9,0.8 2>&1 | grep -A5 -E "mobilenet-v2x1_|densenet40" | cut -c1-170