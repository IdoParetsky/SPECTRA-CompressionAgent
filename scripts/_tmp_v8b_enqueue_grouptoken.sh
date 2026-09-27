#!/usr/bin/env bash
# 28 Sep 01:15 (Fable): V8 representation cell. tree_v8b = tree_v6_dev (group tokens, 304/304 tests) frozen for this job.
set -u
SRC=/home/paretsky/scratch_audit/tree_v6_dev
T=/home/paretsky/scratch_audit/tree_v8b
EXCL=ee-l40s-01,ee-l40s-02,cs-4090-09,ise-6000p-01,ise-6000p-02,ise-6000p-03,ise-6000p-04,ise-6000p-05,ise-6000p-06,ise-6000p-07
rsync -a --delete --exclude runs --exclude '*.out' --exclude .git --exclude '__pycache__' "$SRC/" "$T/" && echo "tree_v8b synced"
grep -c offline_train_v8_grouptoken $T/scripts/spectra.sbatch $T/scripts/submit.sh
cd "$T" || exit 1
export SPECTRA_REPO_DIR="$T" SPECTRA_EXCLUDE_NODES="$EXCL"
if squeue -u paretsky -h -o '%j' | grep -qx v8-grouptoken; then echo "already queued"; else
  env SPECTRA_JOB_NAME=v8-grouptoken SPECTRA_NICE=30 bash scripts/submit.sh offline_train_v8_grouptoken 2>&1 | grep -E 'submitted job|run dir|ERROR|error|Unknown|missing'
fi
squeue -u paretsky -h -S -p -o "%.9i %.26j %.2t %.4y %.10M %R" | grep -v JobHeldUser
