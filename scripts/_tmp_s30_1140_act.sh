#!/usr/bin/env bash
# 30 Sep ~11:40 IDT, Opus 5.5 sitting (Ido GO 11:08): scancel 21716380; chain the Stage-4 resume;
# submit N3 (crop+flip walk + 100-ep final FT, DepGraph R56) and N1/N2 (pre-registered, from 21730500's saves);
# park 21730506 behind them pending decision (d). Then read pace + the (d) control rows.
set -u
T=/home/paretsky/scratch_audit/tree_v9c
cd "$T" || exit 1
export SPECTRA_REPO_DIR=$PWD SPECTRA_EVAL_DETERMINISTIC=1
subid() {
  local name=$1; shift
  if squeue -u paretsky -h -o '%j' | grep -qx "$name"; then echo "== $name already queued" >&2; return; fi
  local out; out=$(env SPECTRA_JOB_NAME="$name" "$@" 2>&1 | grep -v "GPU Parameter")
  echo "$out" | grep -E "submitted job|WARNING|ERROR|error|resume" >&2
  echo "$out" | grep -oE "submitted job [0-9]+" | grep -oE "[0-9]+$" | tail -1
}
feat() { [[ -n "$1" ]] && scontrol update JobId="$1" Features="rtx_6000|rtx_4090" 2>&1 | grep -v '^$'; }

echo "=== 1. scancel 21716380 (backup verified 11:22)"
ls /home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/train_resume.pt || { echo "no backup; stop"; exit 1; }
scancel 21716380; sleep 3; squeue -h -j 21716380 -o '%i %T' 2>&1; sacct -j 21716380 -X -n -o JobID,State%12 2>&1 | head -2

echo "=== 2. Stage-4 resume, afterok:21737123"
R=$T/runs/job21737123
ls -la $R/agent_checkpoints/train_resume.pt
idR=$(subid v9c-paug-area-train-r1 SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_FT_AUG=1 SPECTRA_PROBE_SCORE=area \
  SPECTRA_RESUME_TRAIN=1 SPECTRA_RESUME_PATH=$R/agent_checkpoints/train_resume.pt SPECTRA_PARENT_RUN=$R \
  SPECTRA_DEPENDENCY=afterok:21737123 SPECTRA_GPU_GRES=1 SPECTRA_NICE=0 bash scripts/submit.sh offline_train_v6_inband_p5b2)
echo "resume=$idR"; feat "$idR"

P0="SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256"
FT="SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1"
D=$PWD/configs/input_catalog_l_depgraph_r56.json
S="bash scripts/submit.sh baseline_c10_mild_traj_gonce"
echo "=== 3. N3: 21730500's line + SPECTRA_FT_AUG=1 (crop+flip in the walk FT)"
id3=$(subid v9c-aug-ft100-dg-r56 $P0 $FT SPECTRA_FT_AUG=1 SPECTRA_EVAL_PASSES=5 SPECTRA_EVAL_SIZE_POINTS=flop:0.6,0.47,0.39 \
  SPECTRA_INPUT=$D SPECTRA_GPU_GRES=1 SPECTRA_WALL=1-06:00:00 SPECTRA_NICE=3 $S)
echo "N3=$id3"; feat "$id3"
echo "=== 4. N1 KD / N2 AutoAugment final FT from 21730500's saved walk (ops 8 lines, condition met 09:57)"
idK=$(subid v9c-kd-from-dg-r56 $P0 SPECTRA_FT_KD=1 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 \
  SPECTRA_EVAL_FINAL_FT_KD=1 SPECTRA_EVAL_FINAL_FT_FROM=$PWD/runs/job21730500/traj_models SPECTRA_INPUT=$D \
  SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-16:00:00 SPECTRA_NICE=60 $S)
echo "N1=$idK"; feat "$idK"
idA=$(subid v9c-autoaug-from-dg-r56 $P0 SPECTRA_FT_AUG=1 SPECTRA_FT_AUTOAUG=1 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 \
  SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_FINAL_FT_FROM=$PWD/runs/job21730500/traj_models SPECTRA_INPUT=$D \
  SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-16:00:00 SPECTRA_NICE=61 $S)
echo "N2=$idA"; feat "$idA"

echo "=== 5. park 21730506 (no-aug twins + FT) behind N1/N2 until decision (d)"
scontrol update JobId=21730506 Nice=70 2>&1 | grep -v '^$'
sleep 8
echo "=== queue"
squeue -u paretsky -h -o "%.10i %.24j %.2t %.10M %.10l %.5y %f %R" 2>/dev/null
scontrol show job "$idR" 2>/dev/null | grep -oE "Dependency=[^ ]+|Features=[^ ]+|TRES=[^ ]+|TimeLimit=[^ ]+" | tr '\n' ' '; echo

echo "=== 6. Stage-4 pace (timestamps)"
L=$R/logs/rank0.log
grep -E "PPO update [0-9]+ \|" "$L" | cut -c1-24 | tr '\n' ' '; echo
grep -E "best_score=.*since_improvement" "$L" | sed -E 's/^(.{19}).*(best_score=[^,]+, since_improvement=[^ ]+).*/\1 \2/' | tail -3
grep -E "PROBE|probe" "$L" | grep -vE "SPECTRA_\*|train_config" | sed -E 's/^(.{19}).*\| /\1 /' | cut -c1-170 | tail -6
grep -oE 'DONE Episode [0-9]+ in [0-9.]+s \| steps=[0-9]+' "$L" | tail -12 | tr '\n' ';'; echo
echo "episodes=$(grep -c 'DONE Episode' "$L") first=$(grep -m1 -E 'Episode 0/' "$L" | cut -c1-19) now=$(date '+%F %T')"

echo "=== 7. decision (d) control rows: P twins 21726337 (VGG-16) and P thin 21726335 (both nets)"
for j in 21726337 21726335; do
  for d in $T/runs/job$j /home/paretsky/scratch_audit/tree_v9b/runs/job$j /home/paretsky/scratch_audit/tree_v9/runs/job$j; do
    [[ -f $d/logs/rank0.log ]] || continue
    echo "--- $j ($d)"
    grep -E "\[eval\] TRAJ (val_best|size_)" $d/logs/rank0.log | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar10_[a-z_-]+_[0-9._]+\.pth?//; s/ \| FLOPs x[0-9.]+//' | cut -c1-150
    break
  done
done
echo "--- 21729557 (aug thin 40/10) so far"
grep -E "\[eval\] TRAJ (val_best|size_)" $(ls -d /home/paretsky/scratch_audit/tree_v9b/runs/job21729557)/logs/rank0.log \
  | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar10_[a-z_-]+_[0-9._]+\.pth?//; s/ \| FLOPs x[0-9.]+//' | cut -c1-150
echo "--- 21737104 (aug VGG twins) so far"
grep -E "\[eval\] TRAJ (val_best|size_)" $T/runs/job21737104/logs/rank0.log \
  | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar(10|100)_[a-z_-]+_[0-9._]+\.pth?//; s/ \| FLOPs x[0-9.]+//' | cut -c1-150
exit 0
