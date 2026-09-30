#!/usr/bin/env bash
# 30 Sep: a Slurm requeue of the Stage-4 train reruns the sbatch "always cold" block, which deletes the run's own
# train_resume.pt (spectra.sbatch ~1044). Turn requeue off on the train and its chained resume; back up the bundle.
scontrol show config 2>/dev/null | grep -iE "^JobRequeue|^PreemptMode|^PreemptType"
for j in 21737123 21767188; do
  echo "--- $j before: $(scontrol show job $j 2>/dev/null | grep -oE 'Requeue=[0-9]+|Restarts=[0-9]+' | tr '\n' ' ')"
  scontrol update JobId=$j Requeue=0 2>&1 | grep -v '^$'
  echo "    after:  $(scontrol show job $j 2>/dev/null | grep -oE 'Requeue=[0-9]+|Restarts=[0-9]+' | tr '\n' ' ')"
done
C=/home/paretsky/scratch_audit/tree_v9c/runs/job21737123
BK=/home/paretsky/spectra_backups/job21737123_$(date +%Y%m%d)
mkdir -p "$BK" && cp -a "$C/agent_checkpoints/." "$BK/" && ls -la "$BK"
df -h /home/paretsky 2>/dev/null | tail -1
