#!/usr/bin/env bash
# 30 Sep: exact readouts for 21730500 (DepGraph R56, P walk + 100-ep final FT) with the fixed readers.
PY=/home/paretsky/.conda/envs/spectra/bin/python
RD=/home/paretsky/scratch_audit/readers_s30/scripts
R=/home/paretsky/scratch_audit/tree_v9c/runs/job21730500
echo "=== final_ft_readout"
$PY $RD/final_ft_readout.py "$R" 2>&1 | cut -c1-230
echo "=== crossfit_readout (10k, both halves)"
$PY $RD/crossfit_readout.py "$R" --taus 10,5 --sizes flop:0.6,0.47,0.39 2>&1 | cut -c1-230
echo "=== rows with FLOPs"
grep -E "\[eval\] TRAJ (val_best|size_|final_ft)" $R/logs/rank0.log | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar10_[a-z_-]+_[0-9._]+\.pth?//' | cut -c1-175
