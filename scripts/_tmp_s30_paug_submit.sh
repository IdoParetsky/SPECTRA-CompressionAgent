#!/usr/bin/env bash
# 30 Sep ~02:35 IDT, Opus 5.5 sitting. Stage 4 = ONE train: the area train under P, plus crop+flip
# if O1 passes the training rule. The rule's decider (21729556 r56-w4 at 12/4) lands ~03:10. Both
# candidate trains are held so neither starts before the verdict; exactly one is released after it:
#   PASS (r56-w4 not > 0.5 pp worse at equal keep, r20 inside its 2 pp guard) -> release P+aug, scancel P-only
#   FAIL -> release P-only 21737095, scancel P+aug
set -u
C=/home/paretsky/scratch_audit/tree_v9c
cd "$C" || exit 1
export SPECTRA_REPO_DIR=$PWD
scontrol hold 21737095 && echo "held 21737095 (P-only)"
if squeue -u paretsky -h -o '%j' | grep -qx v9c-paug-area-train; then echo "== P+aug already queued"; else
  out=$(env SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_FT_AUG=1 SPECTRA_PROBE_SCORE=area \
    SPECTRA_JOB_NAME=v9c-paug-area-train SPECTRA_NICE=0 \
    bash scripts/submit.sh offline_train_v6_inband_p5b2 2>&1 | grep -vE "GPU Parameter")
  id=$(echo "$out" | grep -oE "submitted job [0-9]+" | grep -oE "[0-9]+$" | tail -1)
  [[ -n "$id" ]] && scontrol hold "$id" && echo "held $id (P+aug)"
  echo "$out" | tail -3
fi
sleep 3
squeue -u paretsky -h -o "%.10i %.22j %.2t %.5y %R" 2>/dev/null | grep -E "area-train|smoke-from|aug-thin-12"
scontrol show job $(squeue -u paretsky -h -n v9c-paug-area-train -o %i) 2>/dev/null | grep -oE 'TresPerJob=[^ ]*|Priority=[^ ]*' | tr '\n' ' '; echo
