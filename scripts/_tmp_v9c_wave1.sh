#!/usr/bin/env bash
# 29 Sep ~16:30 IDT, Opus 5.5 sitting (Ido 15:49: fill QOS 4 with independent no-agent cells).
# Wave 1 = frozen tree_v9b, no code change. SPECTRA_EVAL_SAVE_TRAJ_MODELS is deliberately UNSET:
# the only torch.save of a live module sits behind it (runner line 170), so final_ft can run.
set -u
T=/home/paretsky/scratch_audit/tree_v9b
cd "$T" || exit 1
mkdir -p runs/slurm_logs
export SPECTRA_REPO_DIR=$PWD SPECTRA_EVAL_DETERMINISTIC=1
unset SPECTRA_EVAL_SAVE_TRAJ_MODELS
subid() {  # prints the job id; the full submit line goes to stderr for the log
  local name=$1; shift
  if squeue -u paretsky -h -o '%j' | grep -qx "$name"; then echo "== $name already queued" >&2; return; fi
  local out; out=$(env SPECTRA_JOB_NAME="$name" "$@" 2>&1 | grep -v "GPU Parameter")
  echo "$out" | grep -E "submitted job|WARNING|ERROR|error" >&2
  echo "$out" | grep -oE "submitted job [0-9]+" | grep -oE "[0-9]+$" | tail -1
}
P0="SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256"
FT="SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1"
AUG="SPECTRA_FT_AUG=1"
F12="SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4"
G=$PWD/configs/v7_c100_candidates_input.json
TW=$PWD/configs/input_catalog_l_twins.json
V=$PWD/configs/input_catalog_l_depgraph_vgg19_c100.json
S="bash scripts/submit.sh baseline_c10_mild_traj_gonce"

echo "=== W1-0 smoke: final_ft path on GPU (never ledger)"
id0=$(subid v9b-smoke-ft $P0 SPECTRA_EVAL_FINAL_FT_EPOCHS=1 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 \
  SPECTRA_EVAL_SIZE_POINTS=param:0.9 SPECTRA_EVAL_PASSES=1 SPECTRA_NUM_EPOCHS=1 SPECTRA_FINETUNE_PATIENCE=1 \
  SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-01:00:00 SPECTRA_NICE=0 $S)
echo "W1-0 v9b-smoke-ft=$id0"
echo "=== W1-1 final FT 100 + origin, DepGraph VGG-19 C100 (first honest-gain read)"
id1=$(subid v9b-ft100-dg-vgg19 $P0 $FT SPECTRA_EVAL_PASSES=3 SPECTRA_EVAL_SIZE_POINTS=param:0.7,0.6 \
  SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$V SPECTRA_DATABASE=$V SPECTRA_GPU_GRES=1 \
  SPECTRA_WALL=0-20:00:00 SPECTRA_NICE=10 $S)
echo "W1-1 v9b-ft100-dg-vgg19=$id1"
echo "=== W1-2 C100 8-net admission gate under P, train FT 12/4 (Q4 evidence; does NOT emit)"
id2=$(subid v9b-p-gate-c100 $P0 $F12 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.9,0.8 \
  SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$G SPECTRA_DATABASE=$G SPECTRA_GPU_GRES=1 \
  SPECTRA_WALL=0-16:00:00 SPECTRA_NICE=20 $S)
echo "W1-2 v9b-p-gate-c100=$id2"
echo "=== W1-3 crop+flip in the walk FT, twins, TEST 40/10 (pairs step-by-step with P twins 21726337)"
id3=$(subid v9b-aug-twins $P0 $AUG SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.7 \
  "SPECTRA_DATASET_NAMES=cifar-10 cifar-100" SPECTRA_INPUT=$TW SPECTRA_GPU_GRES=1 \
  SPECTRA_WALL=0-16:00:00 SPECTRA_NICE=30 $S)
echo "W1-3 v9b-aug-twins=$id3"
echo "=== W1-4 crop+flip, C100 8-net gate 12/4 (pairs with W1-2)"
id4=$(subid v9b-aug-gate-c100 $P0 $AUG $F12 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.9,0.8 \
  SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$G SPECTRA_DATABASE=$G SPECTRA_GPU_GRES=1 \
  SPECTRA_WALL=0-16:00:00 SPECTRA_NICE=40 $S)
echo "W1-4 v9b-aug-gate-c100=$id4"
echo "=== W1-5 P thin reference at the train FT 12/4 (control for W1-6)"
id5=$(subid v9b-p-thin-12x4 $P0 $F12 SPECTRA_SEED=42 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6 \
  SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-05:00:00 SPECTRA_NICE=50 $S)
echo "W1-5 v9b-p-thin-12x4=$id5"
echo "=== W1-6 crop+flip thin at 12/4 (the train-recipe pair rule, with W1-4)"
id6=$(subid v9b-aug-thin-12x4 $P0 $AUG $F12 SPECTRA_SEED=42 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6 \
  SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-05:00:00 SPECTRA_NICE=60 $S)
echo "W1-6 v9b-aug-thin-12x4=$id6"
echo "=== W1-7 crop+flip thin at TEST 40/10 (skinny guard; pairs with P thin 21726335)"
id7=$(subid v9b-aug-thin $P0 $AUG SPECTRA_SEED=42 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6 \
  SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-14:00:00 SPECTRA_NICE=70 $S)
echo "W1-7 v9b-aug-thin=$id7"
echo "=== W1-8 N2 stream protection under P, 3 passes (equal-keep vs P thin / P N4)"
id8=$(subid v9b-p-n2-streams $P0 SPECTRA_PROTECT_STREAMS=1 SPECTRA_SEED=42 SPECTRA_EVAL_PASSES=3 \
  SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6 SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-16:00:00 SPECTRA_NICE=80 $S)
echo "W1-8 v9b-p-n2-streams=$id8"

sleep 8
echo "=== queue"
squeue -u paretsky -h -o "%.10i %.22j %.2t %.10l %.6Q %.4y %b %N %R" 2>/dev/null | grep -v JobHeldUser
