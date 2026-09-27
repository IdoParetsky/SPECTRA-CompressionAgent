#!/usr/bin/env bash
# 28 Sep 01:00 (Fable): the constant-LR and schedule arms all failed the two-dataset gate inside 12 epochs.
# Next one-recipe candidate = budget, not LR: Adam, patience 4, cap 40 (cap rarely binds on CIFAR-10, binds on CIFAR-100).
# Two LR arms x (thin control + 8 C100 candidates), no agent, 2-pass mild, recipe A.
set -u
T=/home/paretsky/scratch_audit/tree_v8
EXCL=ee-l40s-01,ee-l40s-02,cs-4090-09,ise-6000p-01,ise-6000p-02,ise-6000p-03,ise-6000p-04,ise-6000p-05,ise-6000p-06,ise-6000p-07
cd "$T" || exit 1
export SPECTRA_REPO_DIR="$T" SPECTRA_EXCLUDE_NODES="$EXCL"
C=$T/configs
submit() { local name=$1 nice=$2 profile=$3; shift 3
  if squeue -u paretsky -h -o '%j' | grep -qx "$name"; then echo "== $name already queued"; return; fi
  echo "== $name (nice $nice) :: $*"
  env SPECTRA_JOB_NAME="$name" SPECTRA_NICE="$nice" "$@" bash scripts/submit.sh "$profile" 2>&1 | grep -E 'submitted job|run dir|ERROR|error|Unknown|missing' | head -2; }
GATE="SPECTRA_EVAL_PASSES=2 SPECTRA_NUM_EPOCHS=40 SPECTRA_FINETUNE_PATIENCE=4 SPECTRA_FT_RECIPE=a"
C100="SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$C/v7_c100_candidates_input.json SPECTRA_DATABASE=$C/v7_c100_candidates_input.json"
submit cap40-adam1e3-thin-ctl  5 baseline_c10_mild_traj_gonce $GATE
submit cap40-adam1e3-c100-gate 6 baseline_c10_mild_traj_gonce $GATE $C100
submit cap40-adam1e4-thin-ctl  7 baseline_c10_mild_traj_gonce $GATE SPECTRA_FT_LR=1e-4
submit cap40-adam1e4-c100-gate 8 baseline_c10_mild_traj_gonce $GATE SPECTRA_FT_LR=1e-4 $C100
squeue -u paretsky -h -S -p -o "%.9i %.26j %.2t %.4y %.10M %R" | grep -v JobHeldUser
