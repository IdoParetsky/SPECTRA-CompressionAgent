#!/usr/bin/env bash
# 30 Sep ~13:05 IDT, Opus 5.5 sitting: the size-matched Catalog L VGG-16 C10 cell (bar 3, L2) under the adopted
# TEST walk. No earlier cell reaches a published VGG-16 size: 21809595 stops at 2 passes (0.66 kept); HRank is
# 46.5 % FLOPs / 17.1 % params kept, OCSPruner (pretrained start) 21.2 % FLOPs / 13.7 % params. Mild keeps
# params ~ FLOPs, so the size match is on FLOPs. Mild loses ~18 % FLOPs per pass on VGG-16 -> 10 passes.
# No-agent, frozen tree_v9c, no code change.
set -u
T=/home/paretsky/scratch_audit/tree_v9c
cd "$T" || exit 1
export SPECTRA_REPO_DIR=$PWD SPECTRA_EVAL_DETERMINISTIC=1
subid() {
  local name=$1; shift
  if squeue -u paretsky -h -o '%j' | grep -qx "$name"; then echo "== $name already queued" >&2; return; fi
  local out; out=$(env SPECTRA_JOB_NAME="$name" "$@" 2>&1 | grep -v "GPU Parameter")
  echo "$out" | grep -E "submitted job|WARNING|ERROR|error" >&2
  echo "$out" | grep -oE "submitted job [0-9]+" | grep -oE "[0-9]+$" | tail -1
}
feat() { [[ -n "$1" ]] && scontrol update JobId="$1" Features="rtx_6000|rtx_4090" 2>&1 | grep -v '^$'; }

V16=/home/paretsky/scratch_audit/configs_s30/input_catalog_l_vgg16_c10.json
python3 - "$PWD/configs/input_catalog_l_twins.json" "$V16" <<'PY'
import json, sys
src = json.load(open(sys.argv[1]))
out = {k: v for k, v in src.items() if "/vgg16_bn_cifar10" in k}
assert len(out) == 1, out
json.dump(out, open(sys.argv[2], "w"), indent=2)
print("wrote", sys.argv[2], [k.rsplit("/", 1)[1] for k in out])
PY
P0="SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256"
FT="SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1"
S="bash scripts/submit.sh baseline_c10_mild_traj_gonce"
id=$(subid v9c-aug-ft100-l2-vgg16 $P0 $FT SPECTRA_FT_AUG=1 SPECTRA_EVAL_PASSES=10 SPECTRA_EVAL_SIZE_POINTS=flop:0.465,0.212 \
  SPECTRA_DATASET_NAMES=cifar-10 SPECTRA_INPUT=$V16 SPECTRA_GPU_GRES=1 SPECTRA_WALL=1-06:00:00 SPECTRA_NICE=42 $S)
echo "L2=$id"; feat "$id"
sleep 5
scontrol show job "$id" 2>/dev/null | grep -oE "JobName=[^ ]+|Nice=[^ ]+|Features=[^ ]+|TimeLimit=[^ ]+|Dependency=[^ ]+|WorkDir=[^ ]+" | tr '\n' ' '; echo
squeue -u paretsky -h -o "%.10i %.26j %.2t %.10M %.5y %R" 2>/dev/null | grep -vE "c10-(similar|unlike)"
exit 0
