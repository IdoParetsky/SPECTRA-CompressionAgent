#!/usr/bin/env bash
# 30 Sep ~02:45 IDT, Opus 5.5 sitting. Wave 3 = frozen tree_v9c, no code change, candidates saved.
# W3-1 finishes Pri 6 (aug as the TEST walk recipe): 21729553 walls out at 08:21 inside VGG-16, so the
#      ">= 2/3 twins kinder on TEST" rule can only resolve on a VGG-only re-walk. Pairs with 21726337.
# W3-2 is NEXT N4 (condition met: §148 gate rule): crop+flip walk + 100-ep final FT on DepGraph VGG-19
#      C100. Pairs with 21729551 by step (walk) and by final_ft row at equal keep.
set -u
T=/home/paretsky/scratch_audit/tree_v9c
cd "$T" || exit 1
mkdir -p runs/slurm_logs /home/paretsky/scratch_audit/configs_s30
export SPECTRA_REPO_DIR=$PWD SPECTRA_EVAL_DETERMINISTIC=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1
TWV=/home/paretsky/scratch_audit/configs_s30/input_catalog_l_twins_vgg.json
python3 - "$PWD/configs/input_catalog_l_twins.json" "$TWV" <<'PY'
import json, sys
src = json.load(open(sys.argv[1]))
out = {k: v for k, v in src.items() if "/vgg" in k}
assert len(out) == 2, out
json.dump(out, open(sys.argv[2], "w"), indent=2)
print("wrote", sys.argv[2], [k.rsplit("/", 1)[1] for k in out])
PY
subid() {
  local name=$1; shift
  if squeue -u paretsky -h -o '%j' | grep -qx "$name"; then echo "== $name already queued" >&2; return; fi
  local out; out=$(env SPECTRA_JOB_NAME="$name" "$@" 2>&1 | grep -v "GPU Parameter")
  echo "$out" | grep -E "submitted job|WARNING|ERROR|error" >&2
  echo "$out" | grep -oE "submitted job [0-9]+" | grep -oE "[0-9]+$" | tail -1
}
P0="SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256"
FT="SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1"
AUG="SPECTRA_FT_AUG=1"
V=$PWD/configs/input_catalog_l_depgraph_vgg19_c100.json
S="bash scripts/submit.sh baseline_c10_mild_traj_gonce"

echo "=== W3-1 crop+flip twins, VGG-16 C10 + VGG-19 C100 only, TEST 40/10 (pairs with P twins 21726337)"
w1=$(subid v9c-aug-twins-vgg $P0 $AUG SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.7 \
  "SPECTRA_DATASET_NAMES=cifar-10 cifar-100" SPECTRA_INPUT=$TWV SPECTRA_GPU_GRES=1 \
  SPECTRA_WALL=0-16:00:00 SPECTRA_NICE=25 $S)
echo "W3-1 v9c-aug-twins-vgg=$w1"
echo "=== W3-2 (N4) crop+flip walk + final FT 100 + origin, DepGraph VGG-19 C100 (pairs with 21729551)"
w2=$(subid v9c-aug-ft100-dg-vgg19 $P0 $AUG $FT SPECTRA_EVAL_PASSES=3 SPECTRA_EVAL_SIZE_POINTS=param:0.7,0.6 \
  SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$V SPECTRA_DATABASE=$V SPECTRA_GPU_GRES=1 \
  SPECTRA_WALL=0-20:00:00 SPECTRA_NICE=45 $S)
echo "W3-2 v9c-aug-ft100-dg-vgg19=$w2"
for j in $w1 $w2; do [[ -n "$j" ]] && scontrol update JobId=$j Features="rtx_6000|rtx_4090"; done
sleep 4
squeue -u paretsky -h -o "%.10i %.24j %.2t %.10l %.5y %f %R" 2>/dev/null | grep -v JobHeldUser
