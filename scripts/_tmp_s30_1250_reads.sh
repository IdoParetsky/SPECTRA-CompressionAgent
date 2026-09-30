#!/usr/bin/env bash
# 30 Sep 12:50 IDT: rows for the ledger (21737104 VGG-19 C100 twin vs P twins 21726337; 21730501 thin final FT)
# and the ep0011 snapshot facts. Read-only.
B=/home/paretsky/scratch_audit/tree_v9b/runs
C=/home/paretsky/scratch_audit/tree_v9c/runs
RD=/home/paretsky/scratch_audit/readers_s30/scripts
PY=/home/paretsky/.conda/envs/spectra/bin/python
run() { for d in "$C/job$1" "$B/job$1" /home/paretsky/scratch_audit/tree_v9/runs/job$1; do [[ -f $d/logs/rank0.log ]] && { echo "$d"; return; }; done; }
cut_rows() { sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar(10|100)_[a-z_-]+_[0-9._]+\.pth?//' | cut -c1-230; }
echo "=== P twins 21726337 VGG-19 C100 rows (control)"
d=$(run 21726337); echo "$d"
grep -E "\[eval\] TRAJ (val_best|size_|floor_hold|terminal).*vgg19" $d/logs/rank0.log | cut_rows
echo "=== aug twins 21737104 VGG-19 C100 rows (arm), incl. floor_hold/terminal"
d=$(run 21737104)
grep -E "\[eval\] TRAJ (val_best|size_|floor_hold|terminal).*vgg19" $d/logs/rank0.log | cut_rows
grep -E "unpruned|origin acc|test acc" $d/logs/rank0.log | grep -i vgg19 | head -3 | cut -c1-200
echo "=== 10k / cross-fit readout on 21737104 vs 21726337 (vgg19)"
ls $RD
$PY $RD/crossfit_readout.py "$(run 21737104)" 2>&1 | grep -iE "vgg19|usage|error" | head -12
$PY $RD/crossfit_readout.py "$(run 21726337)" 2>&1 | grep -iE "vgg19|usage|error" | head -12
echo "=== census on arm 21737104 (vgg19): val>0 / test>0"
$PY $RD/paired_steps.py "$(run 21737104)" "$(run 21726337)" 2>&1 | grep -i vgg19
echo "=== 21730501 all TRAJ rows"
d=$(run 21730501)
grep -E "\[eval\] TRAJ (val_best|size_|final_ft|floor_hold)|final_ft_failed|ORIGIN" $d/logs/rank0.log | cut_rows
$PY $RD/final_ft_readout.py "$d" 2>&1 | head -30
echo "=== ep0011 snapshot"
ls -la $C/job21737123/snapshots/ep0011 2>/dev/null
grep -E "Snapshot frozen|PROBE ep" $C/job21737123/logs/rank0.log | sed -E 's/^(.{19}).*\| /\1 /' | cut -c1-200
exit 0
