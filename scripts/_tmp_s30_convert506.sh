#!/usr/bin/env bash
# 30 Sep ~12:45 IDT, Opus 5.5 sitting (Ido GO 12:34): convert 21730506 (no-aug walk + 100-ep final FT,
# chenyaofo R56 + VGG-16 C10 twins) to the crop+flip walk adopted by decision (d) (ledger §152).
# Same line as runbook §10.5 (c): 21730506's own flags + SPECTRA_FT_AUG=1, frozen tree_v9c, no code change.
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

echo "=== 1. 21730506 must still be PENDING (never cancel a running walk)"
st=$(squeue -h -j 21730506 -o '%T' 2>/dev/null)
squeue -h -j 21730506 -o '%.10i %.24j %.2t %.5y %R' 2>/dev/null
if [[ "$st" != "PENDING" ]]; then echo "21730506 state='$st' (not PENDING); stop, report to Ido"; exit 1; fi
scancel 21730506; sleep 3
sacct -j 21730506 -X -n -o JobID,JobName%26,State%12 2>&1 | head -2

echo "=== 2. submit v9c-aug-ft100-twins-c10 (crop+flip walk + final FT 100 + origin, pairs with 21737104 / 21729553)"
P0="SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256"
FT="SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1"
TW=$PWD/configs/input_catalog_l_twins.json
S="bash scripts/submit.sh baseline_c10_mild_traj_gonce"
id=$(subid v9c-aug-ft100-twins-c10 $P0 $FT SPECTRA_FT_AUG=1 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.7 \
  SPECTRA_DATASET_NAMES=cifar-10 SPECTRA_INPUT=$TW SPECTRA_GPU_GRES=1 SPECTRA_WALL=1-06:00:00 SPECTRA_NICE=40 $S)
echo "NEW=$id"; feat "$id"
sleep 5
scontrol show job "$id" 2>/dev/null | grep -oE "JobName=[^ ]+|Nice=[^ ]+|Features=[^ ]+|TimeLimit=[^ ]+|Dependency=[^ ]+|Requeue=[^ ]+|WorkDir=[^ ]+" | tr '\n' ' '; echo

echo "=== 3. queue"
squeue -u paretsky -h -o "%.10i %.26j %.2t %.10M %.10l %.5y %f %R" 2>/dev/null
exit 0
