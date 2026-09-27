#!/usr/bin/env bash
# 28 Sep 00:55 (Fable): 21703443 ran the generic default branch (tree_v8 sbatch lacked the v7 gate line).
# Cancel it (Fable's own job, not a protected train), fix both trees, resubmit the real Budget+STOP cell.
set -u
T=/home/paretsky/scratch_audit/tree_v8
DEV=/home/paretsky/scratch_audit/tree_v6_dev
EXCL=ee-l40s-01,ee-l40s-02,cs-4090-09,ise-6000p-01,ise-6000p-02,ise-6000p-03,ise-6000p-04,ise-6000p-05,ise-6000p-06,ise-6000p-07
scancel 21703443 && echo "cancelled 21703443 (mis-profiled)"
sleep 3
# fixed scripts were scp'd to $DEV just before this script ran; copy into tree_v8 (no job from tree_v8 is PD/R now)
for f in scripts/spectra.sbatch scripts/submit.sh; do cp "$DEV/$f" "$T/$f"; sed -i 's/\r$//' "$T/$f" "$DEV/$f"; done
grep -c 'offline_train_v7_\*' $T/scripts/spectra.sbatch | sed 's/^/v7 gate lines in tree_v8: /'
bash -n $T/scripts/spectra.sbatch && bash -n $T/scripts/submit.sh && echo "bash -n ok"
cd "$T" || exit 1
export SPECTRA_REPO_DIR="$T" SPECTRA_EXCLUDE_NODES="$EXCL"
env SPECTRA_JOB_NAME=v7-budget-stop SPECTRA_NICE=20 SPECTRA_PROBE_SCORE=area bash scripts/submit.sh offline_train_v7_budget 2>&1 | grep -E 'submitted job|run dir|ERROR|error|Unknown'
squeue -u paretsky -h -S -p -o "%.9i %.26j %.2t %.4y %.10M %R" | grep -v JobHeldUser
