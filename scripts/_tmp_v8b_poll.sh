#!/usr/bin/env bash
# 28 Sep (Fable): lean poll of the second V8 night.
squeue -u paretsky -h -S -p -o "%.9i %.26j %.2t %.4y %.10M %R" | grep -v JobHeldUser
for j in 21715233 21715234 21715235 21715236; do d=/home/paretsky/scratch_audit/tree_v8/runs/job$j; O=$d/logs/rank0.log
  echo "== $j $(squeue -h -j $j -o '%j %T %M' 2>/dev/null)"; [[ -s $O ]] || continue
  grep -m1 -oE 'num_epochs=[0-9]+' $O | tr '\n' ' '; grep -m1 -oE 'learning_rate=[0-9.e-]+' $O | tr '\n' ' '; grep -m1 -oE 'SPECTRA_FINETUNE_PATIENCE.: .[0-9]+' $O; 
  grep -m1 -oE 'Fine-tune recipe: optim=[a-z]+ lr=[0-9.e-]+[^|]{0,60}' $O
  grep -E '\[eval\] TRAJ val_best' $O | sed -E 's/^.*\[eval\]/[eval]/' | cut -c1-170
  echo "TB=$(grep -c Traceback $O)"; done
for j in 21715228 21716380; do O=$(ls /home/paretsky/scratch_audit/tree_v8*/runs/job$j/logs/rank0.log 2>/dev/null | head -1); [[ -s "$O" ]] || { echo "== $j not started"; continue; }
  echo "== $j"; grep -E 'PPO update|PROBE ep|Snapshot frozen|REWIND|Traceback' "$O" | tail -2
  echo "episodes=$(grep -c 'DONE Episode' "$O") stops=$(grep -c 'STOP' "$O") budget_steps=$(grep -c 'budget action:' "$O")"; done
