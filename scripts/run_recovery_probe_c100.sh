#!/usr/bin/env bash
# CIFAR-100 recovery matrix (no RL): is a 5–20% structured prune recoverable
# with full-net fine-tune at 40/80 epochs? Mixed 6-net RL runs showed 0% of
# CIFAR-100 non-id steps inside −10 pp — this probe asks whether that is an
# environment/FT limit before another agent job.
set -uo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
SCRIPT="${SPECTRA_MODELS_SCRIPT:-/home/paretsky/spectra_models_instantiation/thin_res_net.py}"
CKPT_ROOT="${SPECTRA_CKPT_ROOT:-/home/paretsky/spectra_pretrained_networks}"

cd "$REPO_DIR"

RATES=(${SPECTRA_PROBE_RATES:-0.95 0.9 0.8})
EPOCHS=(${SPECTRA_PROBE_EPOCHS:-40 80})
ROWS="${SPECTRA_PROBE_ROWS:-0,mid}"
TAU="${SPECTRA_PROBE_TAU:-10}"

STATUS=0

run_one() {
  local ckpt="$1" arch="$2"
  echo "======== probe_c100 arch=${arch} ckpt=$(basename "$ckpt") ========"
  if ! "$PYTHON" scripts/recovery_probe.py \
    --checkpoint "$ckpt" \
    --script "$SCRIPT" \
    --arch "$arch" \
    --dataset cifar-100 \
    --rates "${RATES[@]}" \
    --epochs "${EPOCHS[@]}" \
    --scope single \
    --rows "$ROWS" \
    --train_compressed_layer_only False \
    --allowed_acc_reduction "$TAU"; then
    echo "NOTE: probe returned non-zero (0 cells within budget is a result); continuing."
    STATUS=1
  fi
}

run_one \
  "$CKPT_ROOT/resnet20-width13_cifar100_thin-res-net_69.95_0.185_27.67.pt" \
  resnet20
run_one \
  "$CKPT_ROOT/resnet56-width9_cifar100_thin-res-net_73.05_0.275_41.13.pt" \
  resnet56

echo "======== full-pass C100 (mild 0.95/0.9, 80 FT) ========"
if ! "$PYTHON" scripts/recovery_probe.py \
  --checkpoint "$CKPT_ROOT/resnet20-width13_cifar100_thin-res-net_69.95_0.185_27.67.pt" \
  --script "$SCRIPT" \
  --arch resnet20 \
  --dataset cifar-100 \
  --rates 0.95 0.9 \
  --epochs 80 \
  --scope pass \
  --train_compressed_layer_only False \
  --allowed_acc_reduction "$TAU"; then
  echo "NOTE: full-pass probe returned non-zero; continuing to summary."
  STATUS=1
fi

echo "CIFAR-100 recovery probe finished (aggregate_status=${STATUS})."
exit 0
