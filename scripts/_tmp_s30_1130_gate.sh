#!/usr/bin/env bash
# 30 Sep ~11:30 read-only: both C100 gates (every net), admit test, 21716380 backup, Stage-4 train progress.
B=/home/paretsky/scratch_audit/tree_v9b/runs
C=/home/paretsky/scratch_audit/tree_v9c/runs
for j in 21729554 21729552; do
  f=$B/job$j/logs/rank0.log
  echo "=== $j $(sacct -j $j -X -n -o JobName%20,State%11,Elapsed,NodeList%12 2>/dev/null | head -1)"
  grep -m2 -E "Val from test on cifar-100|FT aug on cifar-100" "$f" | sed -E 's/^.*\| //'
  grep -m1 -oE "'SPECTRA_FT_AUG': '[^']*'|'SPECTRA_EVAL_FINAL_FT_EPOCHS': '[^']*'" "$f" | tr '\n' ' '; echo
  grep -m1 -oE "'SPECTRA_NUM_EPOCHS': '[^']*'|'SPECTRA_FINETUNE_PATIENCE': '[^']*'|'SPECTRA_FT_LR': '[^']*'" "$f" | tr '\n' ' '; echo
  grep -E "\[eval\] TRAJ (val_best|floor_hold|terminal|size_)" "$f" | sed -E 's/^.*\| \[eval\]/[eval]/' \
    | sed -E 's/_cifar100_[a-z_-]+_[0-9._]+\.pth?//' | cut -c1-190
  echo "--- admit (val_best kept <= 0.98 and val dacc >= -10)"
  grep -E "\[eval\] TRAJ val_best" "$f" | sed -E 's/^.*TRAJ val_best ([^ ]+) step=([0-9]+) \| acc ([0-9.]+) -> ([0-9.]+) \(([-+0-9.]+)\) \| params x([0-9.]+).*val .acc ([-+0-9.]+) pp.*$/\1 \2 \3 \4 \5 \6 \7/' \
    | awk '{ad = ($6 <= 0.98 && $7 >= -10) ? "ADMIT" : "REJECT"; printf "%-58s step=%s test %s->%s (%+.1f pp) kept %s val %s %s\n", $1, $2, $3, $4, 100*$5, $6, $7, ad}'
  echo "nets with a val_best row: $(grep -c '\[eval\] TRAJ val_best' "$f"); nets started: $(grep -cE 'TRAJ start|=== Network|Evaluating network' "$f")"
  grep -E "TRAJ start|Evaluating network|=== Network" "$f" | sed -E 's/^.*\| //' | cut -c1-120 | tail -3
done
echo "=== 21716380 backup + state"
ls -la /home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/ 2>&1 | head -12
squeue -h -j 21716380 -o '%i %j %T %r' 2>&1
echo "=== train 21737123"
L=$C/job21737123/logs/rank0.log
grep -E "PPO update|PROBE ep|Snapshot frozen|REWIND" "$L" | sed -E 's/^.*\| (PPO|PROBE|Snapshot|REWIND)/\1/' | cut -c1-175 | tail -5
echo "episodes=$(grep -c 'DONE Episode' "$L") TB=$(grep -c Traceback "$L") snaps=$(ls -d $C/job21737123/snapshots/ep* 2>/dev/null | wc -l)"
ls -la $C/job21737123/agent_checkpoints/ 2>&1 | head -8
