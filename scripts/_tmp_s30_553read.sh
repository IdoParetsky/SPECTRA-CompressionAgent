#!/usr/bin/env bash
B=/home/paretsky/scratch_audit/tree_v9b/runs; R=/home/paretsky/scratch_audit/readers_s30/scripts
PY=/home/paretsky/.conda/envs/spectra/bin/python
for J in 21729553 21726337; do
  echo "=== $J size rows R56"
  grep -E "\[eval\] TRAJ size_[a-z]+[0-9.]+ resnet56" $B/job$J/logs/rank0.log | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar10_chenyaofo_[0-9._]+\.pt//' | cut -c1-165
  echo "=== $J crossfit R56"
  $PY $R/crossfit_readout.py $B/job$J --taus 10,5 --sizes param:0.8,0.7 2>&1 | awk '/resnet56/{f=1} f&&/^[a-z]/&&!/resnet56/{f=0} f' | cut -c1-170
done
echo "=== 553 current net"
tail -c 400 $B/job21729553/logs/rank0.log | tr '\r' '\n' | tail -1 | sed -E 's/^.*net=([^ ]+) .*step=([0-9]+).*/\1 step \2/'