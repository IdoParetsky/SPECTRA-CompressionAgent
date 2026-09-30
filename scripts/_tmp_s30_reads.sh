#!/usr/bin/env bash
# 30 Sep sitting, read-only: paired reads, honest gain, cross-fit, smoke-save checklist, priorities, gate nets.
B=/home/paretsky/scratch_audit/tree_v9b
C=/home/paretsky/scratch_audit/tree_v9c
PY=/home/paretsky/.conda/envs/spectra/bin/python
cd "$C" || exit 1
echo "=== paired: aug gate 21729554 vs P gate 21729552"
$PY scripts/paired_steps.py $B/runs/job21729554 $B/runs/job21729552 2>&1
echo "=== paired: aug twins 21729553 vs P twins 21726337"
$PY scripts/paired_steps.py $B/runs/job21729553 $B/runs/job21726337 2>&1
echo "=== paired: aug thin 12/4 21729556 vs P thin 12/4 21729555"
$PY scripts/paired_steps.py $B/runs/job21729556 $B/runs/job21729555 2>&1
echo "=== paired: P thin 12/4 21729555 vs P thin 40/10 21726335 (recipe length, info only)"
$PY scripts/paired_steps.py $B/runs/job21729555 $B/runs/job21726335 2>&1
echo "=== final_ft_readout 21729551"
$PY scripts/final_ft_readout.py $B/runs/job21729551 2>&1
echo "=== crossfit 21729551 (DG VGG-19 walk, 3-pass)"
$PY scripts/crossfit_readout.py $B/runs/job21729551 --taus 10,5 --sizes param:0.7,0.6 2>&1
echo "=== crossfit 21729555 (P thin 12/4)"
$PY scripts/crossfit_readout.py $B/runs/job21729555 --taus 10,5 --sizes param:0.8,0.6 2>&1
echo "=== smoke-save files"
ls -la $C/runs/job21730498/traj_models/ 2>&1 | head -30
grep -E "\[eval\] policy=" $C/runs/job21730498/logs/rank0.log | head -1 | grep -oE "final_ft=[^ ]+|save_traj[^ ]*|scratch[^ ]*" | tr '\n' ' '; echo
grep -cE "PicklingError|TRAJ save failed|Traceback" $C/runs/job21730498/logs/rank0.log
echo "=== sprio"
sprio -u paretsky -h -o "%.10i %.10Y %.8A %.8F %.8J %.8P %.8Q %.6N" 2>/dev/null | sort -k2 -nr | head -14
echo "=== gate nets"
$PY - <<'EOF'
import json
d = json.load(open("/home/paretsky/scratch_audit/tree_v9b/configs/v7_c100_candidates_input.json"))
rows = d if isinstance(d, list) else list(d.values()) if isinstance(d, dict) else d
print(type(d).__name__, len(d))
for k in (d if isinstance(d, dict) else range(len(d))):
    v = d[k]
    print(" ", k if isinstance(d, dict) else "", str(v)[:160])
EOF
