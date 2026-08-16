#!/usr/bin/env bash
# CIFAR-100 recoverability with a published-style FT recipe (no RL).
#
# Previous probes (Adam, 40/80 ep, crop+flip or none) recovered 0/all unaugmented
# and 1/26 with aug — that one cell kept 0.995 of params. This suite asks whether
# SGD+cosine+mixup+label-smoothing+AutoAugment+160 epochs recovers a real mid-layer
# 1–5% structured cut on competent C100 nets.
#
# Exit 0 if any cell is inside -tau pp (so afterok can start C100 DRL).
# Exit 2 if the matrix completed with zero recoverable cells.
# Exit 1 on a crash.
set -uo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
CKPT_ROOT="${SPECTRA_CKPT_ROOT:-/home/paretsky/spectra_pretrained_networks}"
INST="${SPECTRA_MODELS_INST:-/home/paretsky/spectra_models_instantiation}"

cd "$REPO_DIR"

RATES=(${SPECTRA_PROBE_RATES:-0.99 0.98 0.95})
EPOCHS=(${SPECTRA_PROBE_EPOCHS:-160})
ROWS="${SPECTRA_PROBE_ROWS:-mid}"
TAU="${SPECTRA_PROBE_TAU:-10}"
OK_CELLS=0
TOTAL_CELLS=0
CRASH=0

run_one() {
  local ckpt="$1" arch="$2" script="$3"
  if [[ -n "${SPECTRA_PROBE_ONLY:-}" && "$arch" != "${SPECTRA_PROBE_ONLY}" ]]; then
    echo "skip arch=${arch} (SPECTRA_PROBE_ONLY=${SPECTRA_PROBE_ONLY})"
    return
  fi
  echo "======== recipe_c100 arch=${arch} ckpt=$(basename "$ckpt") script=$(basename "$script") ========"
  if ! "$PYTHON" scripts/recovery_probe.py \
    --checkpoint "$ckpt" \
    --script "$script" \
    --arch "$arch" \
    --dataset cifar-100 \
    --rates "${RATES[@]}" \
    --epochs "${EPOCHS[@]}" \
    --scope single \
    --rows "$ROWS" \
    --train_compressed_layer_only False \
    --allowed_acc_reduction "$TAU"; then
    echo "NOTE: probe crashed for ${arch}; continuing."
    CRASH=1
    return
  fi
}

# Count OK cells from this job's jsonl after each net (append-only).
count_ok() {
  local jsonl="${SPECTRA_RUN_DIR:-}/results/recovery_probe.jsonl"
  if [[ -z "${SPECTRA_RUN_DIR:-}" || ! -f "$jsonl" ]]; then
    jsonl="$(ls -1t "$REPO_DIR"/runs/job"${SLURM_JOB_ID:-}"/results/recovery_probe.jsonl 2>/dev/null | head -1 || true)"
  fi
  if [[ -n "$jsonl" && -f "$jsonl" ]]; then
    OK_CELLS=$("$PYTHON" -c "
import json,sys
ok=total=0
for line in open(sys.argv[1], encoding='utf-8'):
    if not line.strip():
        continue
    row=json.loads(line)
    total+=1
    if row.get('within_budget'):
        ok+=1
print(ok, total)
" "$jsonl")
  fi
}

run_one \
  "$CKPT_ROOT/resnet20-width13_cifar100_thin-res-net_69.95_0.185_27.67.pt" \
  resnet20 "$INST/thin_res_net.py"
run_one \
  "$CKPT_ROOT/resnet56-width9_cifar100_thin-res-net_73.05_0.275_41.13.pt" \
  resnet56 "$INST/thin_res_net.py"
run_one \
  "$CKPT_ROOT/resnet56-width15_cifar100_thin-res-net_78.46_0.758_112.59.pt" \
  resnet56 "$INST/thin_res_net.py"
run_one \
  "$CKPT_ROOT/vgg11_bn_cifar100_chenyaofo_70.78_9.8_306.68.pt" \
  vgg11_bn "$INST/vgg_chenyaofo.py"
run_one \
  "$CKPT_ROOT/densenet40_cifar100_densenet-cifar_70.25_0.188_74.44.pt" \
  densenet40 "$INST/densenet_cifar.py"
run_one \
  "$CKPT_ROOT/mobilenet-v2x1_cifar100_chenyaofo_74.2_2.35_176.18.pt" \
  mobilenet_v2x1 "$INST/mobilenetv2_chenyaofo.py"

count_ok
echo "CIFAR-100 recipe probe finished crash=${CRASH} ok_total='${OK_CELLS}'"
if [[ "$CRASH" -ne 0 ]]; then
  exit 1
fi
OK_N=${OK_CELLS%% *}
if [[ -z "$OK_N" || "$OK_N" == "0" ]]; then
  echo "No recoverable cells inside -${TAU} pp; not starting afterok DRL."
  exit 2
fi
echo "Recoverable cells found (${OK_N}); afterok DRL may start."
exit 0
