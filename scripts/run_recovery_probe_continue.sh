#!/usr/bin/env bash
# Finish the parts of the recovery suite that job 20018419 skipped after set -e
# aborted on a scientific "0 OK cells" exit from the layer-only resnet20 matrix.

set -uo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
SCRIPT="${SPECTRA_MODELS_SCRIPT:-/home/paretsky/spectra_models_instantiation/thin_res_net.py}"
CKPT_ROOT="${SPECTRA_CKPT_ROOT:-/home/paretsky/spectra_pretrained_networks}"

cd "$REPO_DIR"

RATES=(0.9 0.8 0.7 0.6)
EPOCHS=(0 10 20 40)

run_one() {
  local ckpt="$1" arch="$2" dataset="$3" ft_flag="$4" label="$5"
  echo "======== continue ${label} ========"
  "$PYTHON" scripts/recovery_probe.py \
    --checkpoint "$ckpt" \
    --script "$SCRIPT" \
    --arch "$arch" \
    --dataset "$dataset" \
    --rates "${RATES[@]}" \
    --epochs "${EPOCHS[@]}" \
    --scope single \
    --rows 0,mid \
    --train_compressed_layer_only "$ft_flag" \
    --allowed_acc_reduction 5 || echo "NOTE: non-zero exit for ${label} (0 OK is fine)"
}

run_one \
  "$CKPT_ROOT/resnet56-width6_cifar10_thin-res-net_92.88_0.122_18.60.pt" \
  resnet56 cifar-10 False "resnet56 full FT"
run_one \
  "$CKPT_ROOT/resnet56-width6_cifar10_thin-res-net_92.88_0.122_18.60.pt" \
  resnet56 cifar-10 True "resnet56 layer FT"

echo "======== full-pass recovery (mild rates, full FT) ========"
"$PYTHON" scripts/recovery_probe.py \
  --checkpoint "$CKPT_ROOT/resnet20-width10_cifar10_thin-res-net_91.90_0.107_16.55.pt" \
  --script "$SCRIPT" \
  --arch resnet20 \
  --dataset cifar-10 \
  --rates 0.9 0.8 \
  --epochs 20 40 \
  --scope pass \
  --train_compressed_layer_only False \
  --allowed_acc_reduction 5 || echo "NOTE: full-pass non-zero exit"

echo "Probe continuation finished."
exit 0
