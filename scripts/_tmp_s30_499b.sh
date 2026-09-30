#!/usr/bin/env bash
C=/home/paretsky/scratch_audit/tree_v9c
grep -E "\[eval\] TRAJ final_ft" $C/runs/job21730499/logs/rank0.log | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar10_thin-res-net_[0-9._]+\.pt//' | cut -c1-260
grep -cE "Traceback|final_ft_failed|TRAJ save failed" $C/runs/job21730499/logs/rank0.log
cd $C && /home/paretsky/.conda/envs/spectra/bin/python scripts/final_ft_readout.py $C/runs/job21730499 2>&1 | head -12