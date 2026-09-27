#!/bin/bash
# V5 diversity pretrain — from-scratch CIFAR nets SPECTRA does not yet have.
# Run on a GPU node AFTER a QOS hole. Do not steal v3/V4 GPUs.
# Instantiation-only; no leap src overlay.
set -euo pipefail
R=/home/paretsky/SPECTRA-CompressionAgent
PY=/home/paretsky/.conda/envs/spectra/bin/python
LEAP_INST=/home/paretsky/spectra_models_instantiation
cd "$R"

# Additive factories only — do not overwrite live chenyaofo / thin-res-net files
# while v3/V4 trains are reading them.
mkdir -p "$LEAP_INST"
for f in wide_resnet.py preact_resnet.py; do
  if [[ -f "$R/spectra_models_instantiation/$f" && ! -e "$LEAP_INST/$f" ]]; then
    cp -n "$R/spectra_models_instantiation/$f" "$LEAP_INST/$f"
    echo "copied $f -> leap instantiation"
  fi
done

# Wave A (native CIFAR stems; grouping-friendly residuals / DenseNet):
# WRN-16-4 / PreAct-20 / DenseNet-100 × C10 and C100.
# Wave B (chenyaofo hub leftovers) is scripts/fetch_chenyaofo_hub.py — do that first.

run_one() {
  local arch=$1 script=$2 dataset=$3 source=$4
  echo "=== pretrain $arch $dataset ==="
  "$PY" scripts/train_pretrained_checkpoint.py \
    --arch "$arch" \
    --script "$R/spectra_models_instantiation/$script" \
    --dataset "$dataset" \
    --source "$source" \
    --epochs 200
}

# Safer / cheaper first: C10 WRN-16-4 and PreAct-20, then C100 copies (harder origin).
run_one wrn_16_4 wide_resnet.py cifar-10 wide-resnet
run_one wrn_16_4 wide_resnet.py cifar-100 wide-resnet
run_one preact_resnet20 preact_resnet.py cifar-10 preact-resnet
run_one preact_resnet20 preact_resnet.py cifar-100 preact-resnet
run_one densenet100 densenet_cifar.py cifar-100 densenet-cifar
run_one wrn_28_2 wide_resnet.py cifar-10 wide-resnet
run_one preact_resnet56 preact_resnet.py cifar-10 preact-resnet
echo DONE
