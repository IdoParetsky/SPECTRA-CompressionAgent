#!/usr/bin/env bash
# 30 Sep ~02:30 IDT, Opus 5.5 sitting, Ido GO (Stage 4). One-change train: the area train 21536396
# (offline_train_v6_inband_p5b2 + SPECTRA_PROBE_SCORE=area, tree_v7) under protocol P = clean val
# from the held-out test split + batch 256 pinned. Nothing else changes (catalog, reward, menu,
# probes, 12/4 train FT, governor). Runs from the frozen, tested tree_v9c; no crop+flip.
set -u
C=/home/paretsky/scratch_audit/tree_v9c
cd "$C" || exit 1
export SPECTRA_REPO_DIR=$PWD
if squeue -u paretsky -h -o '%j' | grep -qx v9c-p-area-train; then echo "== already queued"; else
  env SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_PROBE_SCORE=area \
    SPECTRA_JOB_NAME=v9c-p-area-train SPECTRA_NICE=0 \
    bash scripts/submit.sh offline_train_v6_inband_p5b2 2>&1 | grep -vE "GPU Parameter" | tail -6
fi
echo "=== fast-card constraint on pending no-agent cells (the 5-min smoke-from stays untyped)"
for j in 21730500 21730501 21730506 21730507 21730509 21730514 21730516 21729557 21729558; do
  scontrol update JobId=$j Features="rtx_6000|rtx_4090" 2>&1 | head -1
  echo "$j $(scontrol show job $j 2>/dev/null | grep -oE 'Features=[^ ]*' | head -1)"
done
sleep 5
squeue -u paretsky -h -o "%.10i %.24j %.2t %.10l %.5y %b %f %R" 2>/dev/null | grep -v JobHeldUser
sprio -u paretsky -h -o "%.10i %.8Y %.6N" 2>/dev/null | sort -k2 -nr | head -12
