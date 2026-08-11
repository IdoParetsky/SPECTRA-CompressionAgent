#!/usr/bin/env bash
# Recovery matrix on 1–2 thin ResNets: rate × epochs × fine-tune mode.
# Invoked by the `probe` SLURM profile (see spectra.sbatch).
#
# Each matrix may exit 1 when every cell is outside the accuracy budget; that is a
# scientific outcome, not a job failure. Keep going so later nets / scopes still run.

set -uo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
SCRIPT="${SPECTRA_MODELS_SCRIPT:-/home/paretsky/spectra_models_instantiation/thin_res_net.py}"
CKPT_ROOT="${SPECTRA_CKPT_ROOT:-/home/paretsky/spectra_pretrained_networks}"

cd "$REPO_DIR"

RATES=(${SPECTRA_PROBE_RATES:-0.9 0.8 0.7 0.6})
EPOCHS=(${SPECTRA_PROBE_EPOCHS:-0 10 20 40})
ROWS="${SPECTRA_PROBE_ROWS:-0,mid}"

STATUS=0

run_one() {
  local ckpt="$1" arch="$2" dataset="$3" ft_mode="$4"
  local ft_flag
  if [[ "$ft_mode" == "layer" ]]; then
    ft_flag=True
  else
    ft_flag=False
  fi
  echo "======== probe arch=${arch} dataset=${dataset} ft=${ft_mode} ========"
  if ! "$PYTHON" scripts/recovery_probe.py \
    --checkpoint "$ckpt" \
    --script "$SCRIPT" \
    --arch "$arch" \
    --dataset "$dataset" \
    --rates "${RATES[@]}" \
    --epochs "${EPOCHS[@]}" \
    --scope single \
    --rows "$ROWS" \
    --train_compressed_layer_only "$ft_flag" \
    --allowed_acc_reduction 5; then
    echo "NOTE: probe returned non-zero (often means 0 cells within budget); continuing."
    STATUS=1
  fi
}

# Two representative train-DB nets (CIFAR-10), both fine-tune policies.
run_one \
  "$CKPT_ROOT/resnet20-width10_cifar10_thin-res-net_91.90_0.107_16.55.pt" \
  resnet20 cifar-10 full
run_one \
  "$CKPT_ROOT/resnet20-width10_cifar10_thin-res-net_91.90_0.107_16.55.pt" \
  resnet20 cifar-10 layer
run_one \
  "$CKPT_ROOT/resnet56-width6_cifar10_thin-res-net_92.88_0.122_18.60.pt" \
  resnet56 cifar-10 full
run_one \
  "$CKPT_ROOT/resnet56-width6_cifar10_thin-res-net_92.88_0.122_18.60.pt" \
  resnet56 cifar-10 layer

# One full-pass check at mild rates (recovery after compressing every row once).
echo "======== full-pass recovery (mild rates, full FT) ========"
if ! "$PYTHON" scripts/recovery_probe.py \
  --checkpoint "$CKPT_ROOT/resnet20-width10_cifar10_thin-res-net_91.90_0.107_16.55.pt" \
  --script "$SCRIPT" \
  --arch resnet20 \
  --dataset cifar-10 \
  --rates 0.9 0.8 \
  --epochs 20 40 \
  --scope pass \
  --train_compressed_layer_only False \
  --allowed_acc_reduction 5; then
  echo "NOTE: full-pass probe returned non-zero; continuing to summary."
  STATUS=1
fi

echo "Recovery probe suite finished (aggregate_status=${STATUS})."
exit 0
