#!/usr/bin/env bash
# Validate the group-aware freeze fix: layer-only FT but every rewritten module trains.
# Focused on resnet20-w10 so it finishes in ~1h and can be compared to job20018419's
# old layer-only matrix (0/32 OK).

set -uo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
SCRIPT="${SPECTRA_MODELS_SCRIPT:-/home/paretsky/spectra_models_instantiation/thin_res_net.py}"
CKPT_ROOT="${SPECTRA_CKPT_ROOT:-/home/paretsky/spectra_pretrained_networks}"

cd "$REPO_DIR"

echo "======== probe_groupft: resnet20 layer-only with group-edited params trainable ========"
"$PYTHON" scripts/recovery_probe.py \
  --checkpoint "$CKPT_ROOT/resnet20-width10_cifar10_thin-res-net_91.90_0.107_16.55.pt" \
  --script "$SCRIPT" \
  --arch resnet20 \
  --dataset cifar-10 \
  --rates 0.9 0.8 0.7 0.6 \
  --epochs 0 10 20 40 \
  --scope single \
  --rows 0,mid \
  --train_compressed_layer_only True \
  --allowed_acc_reduction 5 || echo "NOTE: non-zero exit (0 OK cells is fine)"

echo "Probe groupft finished."
exit 0
