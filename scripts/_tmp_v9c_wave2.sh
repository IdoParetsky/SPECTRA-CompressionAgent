#!/usr/bin/env bash
# 29 Sep ~17:30 IDT, Opus 5.5 sitting. Wave 2 = tree_v9c (tree_v9b + state_dict saves, scratch control,
# final FT from a saved walk; CPU pytest green). Every job afterok on the v9c smoke except wave-1 re-ranks.
set -u
T=/home/paretsky/scratch_audit/tree_v9c
cd "$T" || exit 1
mkdir -p runs/slurm_logs
export SPECTRA_REPO_DIR=$PWD SPECTRA_EVAL_DETERMINISTIC=1
subid() {  # prints the job id; the full submit line goes to stderr for the log
  local name=$1; shift
  if squeue -u paretsky -h -o '%j' | grep -qx "$name"; then echo "== $name already queued" >&2; return; fi
  local out; out=$(env SPECTRA_JOB_NAME="$name" "$@" 2>&1 | grep -v "GPU Parameter")
  echo "$out" | grep -E "submitted job|WARNING|ERROR|error" >&2
  echo "$out" | grep -oE "submitted job [0-9]+" | grep -oE "[0-9]+$" | tail -1
}
P0="SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256"
FT="SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1"
CG="SPECTRA_FT_REINIT_EDITED=1 SPECTRA_FT_REINIT_SELECT=train SPECTRA_FT_REINIT_PATIENCE=10 SPECTRA_FT_REINIT_EPOCHS=100"
TW=$PWD/configs/input_catalog_l_twins.json
D=$PWD/configs/input_catalog_l_depgraph_r56.json
S="bash scripts/submit.sh baseline_c10_mild_traj_gonce"

echo "=== W2-0 smoke: state_dict saves + scratch on GPU (never ledger)"
id0=$(subid v9c-smoke-save $P0 SPECTRA_EVAL_FINAL_FT_EPOCHS=1 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1 \
  SPECTRA_EVAL_FINAL_FT_SCRATCH=both SPECTRA_EVAL_FINAL_FT_SCRATCH_EPOCHS=1 SPECTRA_EVAL_SIZE_POINTS=param:0.9 \
  SPECTRA_EVAL_PASSES=1 SPECTRA_NUM_EPOCHS=1 SPECTRA_FINETUNE_PATIENCE=1 \
  SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-01:00:00 SPECTRA_NICE=0 $S)
echo "W2-0 v9c-smoke-save=$id0"
[[ -n "$id0" ]] || { echo "smoke did not submit; stopping"; exit 1; }
DEP="SPECTRA_DEPENDENCY=afterok:$id0"
echo "=== W2-1 smoke: final FT from the saved walk (never ledger)"
id1=$(subid v9c-smoke-from $P0 SPECTRA_EVAL_FINAL_FT_EPOCHS=1 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 \
  SPECTRA_EVAL_FINAL_FT_FROM=$T/runs/job$id0/traj_models SPECTRA_EVAL_PASSES=1 SPECTRA_NUM_EPOCHS=1 \
  SPECTRA_FINETUNE_PATIENCE=1 SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-01:00:00 SPECTRA_NICE=0 $DEP $S)
echo "W2-1 v9c-smoke-from=$id1"
echo "=== W2-2 final FT 100 + origin, DepGraph R56 C10 at DepGraph's 2.11x / 2.57x FLOPs (bar 3)"
id2=$(subid v9c-ft100-dg-r56 $P0 $FT SPECTRA_EVAL_PASSES=5 SPECTRA_EVAL_SIZE_POINTS=flop:0.6,0.47,0.39 \
  SPECTRA_INPUT=$D SPECTRA_GPU_GRES=1 SPECTRA_WALL=1-06:00:00 SPECTRA_NICE=5 $DEP $S)
echo "W2-2 v9c-ft100-dg-r56=$id2"
echo "=== W2-3 final FT 100 + origin, P thin s42 (the fast C10 honest-gain cell; saves for scratch)"
id3=$(subid v9c-ft100-thin $P0 $FT SPECTRA_SEED=42 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6 \
  SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-18:00:00 SPECTRA_NICE=20 $DEP $S)
echo "W2-3 v9c-ft100-thin=$id3"
echo "=== W2-4 final FT 100 + origin, P twins C10 (R56 twin + VGG-16 = OCS/HRank cell), same walk as 21726337"
id4=$(subid v9c-ft100-twins-c10 $P0 $FT SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.7 \
  SPECTRA_DATASET_NAMES=cifar-10 SPECTRA_INPUT=$TW SPECTRA_GPU_GRES=1 SPECTRA_WALL=1-06:00:00 SPECTRA_NICE=40 $DEP $S)
echo "W2-4 v9c-ft100-twins-c10=$id4"
if [[ -n "$id3" ]]; then
  echo "=== W2-5 scratch-B (Liu et al. 2019) at the thin walk's saved architectures, 200 ep SGD 0.1, + origin scratch"
  id5=$(subid v9c-scratch-thin $P0 SPECTRA_SEED=42 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 \
    SPECTRA_EVAL_FINAL_FT_SCRATCH=only SPECTRA_EVAL_FINAL_FT_FROM=$T/runs/job$id3/traj_models \
    SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-16:00:00 SPECTRA_NICE=45 SPECTRA_DEPENDENCY=afterok:$id3 $S)
  echo "W2-5 v9c-scratch-thin=$id5"
fi
echo "=== W2-6 C-G (NEON-literal redraw) under P with NEON's own stop: train loss, patience 10, cap 100; R56 twin + VGG-16"
id6=$(subid v9c-cg-neon-twins $P0 $CG SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.7 \
  SPECTRA_DATASET_NAMES=cifar-10 SPECTRA_INPUT=$TW SPECTRA_GPU_GRES=1 SPECTRA_WALL=1-06:00:00 SPECTRA_NICE=50 $DEP $S)
echo "W2-6 v9c-cg-neon-twins=$id6"
echo "=== W2-7 same C-G cell on the thin pair (pairs with P thin 21726335)"
id7=$(subid v9c-cg-neon-thin $P0 $CG SPECTRA_SEED=42 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6 \
  SPECTRA_GPU_GRES=1 SPECTRA_WALL=1-00:00:00 SPECTRA_NICE=52 $DEP $S)
echo "W2-7 v9c-cg-neon-thin=$id7"
if [[ -n "$id2" ]]; then
  echo "=== W2-8 scratch-B at the DepGraph R56 walk's saved size points + origin scratch"
  id8=$(subid v9c-scratch-dg-r56 $P0 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 \
    SPECTRA_EVAL_FINAL_FT_SCRATCH=only SPECTRA_EVAL_FINAL_FT_FROM=$T/runs/job$id2/traj_models SPECTRA_INPUT=$D \
    SPECTRA_GPU_GRES=1 SPECTRA_WALL=1-06:00:00 SPECTRA_NICE=55 SPECTRA_DEPENDENCY=afterok:$id2 $S)
  echo "W2-8 v9c-scratch-dg-r56=$id8"
fi

echo "=== re-rank wave-1 PD (science order: 12/4 pair rule before the 40/10 thin aug; N2 last)"
for jn in "21729555 10" "21729556 11" "21729557 30" "21729558 80"; do
  set -- $jn
  scontrol update JobId=$1 Nice=$2 2>&1 | grep -v "^$" || true
done
sleep 8
echo "=== queue"
squeue -u paretsky -h -o "%.10i %.22j %.2t %.10l %.6Q %.4y %b %N %R" 2>/dev/null | grep -v JobHeldUser
