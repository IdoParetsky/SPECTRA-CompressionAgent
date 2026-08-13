#!/bin/bash
# Train the missing thesis-pool cells (DenseNet-CIFAR, SVHN, Fashion-MNIST).
#
# Does not train on the login node. Submit from the cluster:
#   sbatch --gpus=rtx_6000:1 --mem=80G --cpus-per-task=6 --time=1-00:00:00 \
#     --exclude=ee-l40s-01,ee-l40s-02 scripts/train_pool.sbatch
#
#SBATCH --job-name=spectra-pretrain
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/paretsky/SPECTRA-CompressionAgent/runs/slurm_logs/pretrain_%j.out
#SBATCH --error=/home/paretsky/SPECTRA-CompressionAgent/runs/slurm_logs/pretrain_%j.out

set -uo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
INST="${SPECTRA_INSTANTIATION:-/home/paretsky/spectra_models_instantiation}"
cd "$REPO_DIR" || exit 1
mkdir -p runs/slurm_logs

echo "=============================================================="
echo " SPECTRA pretrained-pool job ${SLURM_JOB_ID:-local}"
echo " node    : $(hostname)"
echo " started : $(date -Is)"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 || echo "nvidia-smi unavailable"
if ! "$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: torch sees no CUDA device." >&2
  exit 2
fi

cp -f "$REPO_DIR/spectra_models_instantiation/densenet_cifar.py" "$INST/densenet_cifar.py"

train() {
  echo "----- $* -----"
  "$PYTHON" scripts/train_pretrained_checkpoint.py "$@"
  local rc=$?
  echo "exit $rc for $*"
  return $rc
}

FAIL=0

# DenseNet enters the *training* distribution (new connectivity pattern: concat).
train --arch densenet40 --script "$INST/densenet_cifar.py" --dataset cifar-10 \
  --source densenet-cifar --epochs 200 --milestones 100 150 \
  --append-catalogs /home/paretsky/database_generic_cifar.json \
                    /home/paretsky/database_cifar10_generic.json \
                    "$REPO_DIR/configs/database_generic_cifar.json" \
                    "$REPO_DIR/configs/database_cifar10_generic.json" || FAIL=1

train --arch densenet40 --script "$INST/densenet_cifar.py" --dataset cifar-100 \
  --source densenet-cifar --epochs 200 --milestones 100 150 \
  --append-catalogs /home/paretsky/database_generic_cifar.json \
                    /home/paretsky/database_cifar100_generic.json \
                    "$REPO_DIR/configs/database_generic_cifar.json" \
                    "$REPO_DIR/configs/database_cifar100_generic.json" || FAIL=1

# Unseen DenseNet depth for held-out eval.
train --arch densenet100 --script "$INST/densenet_cifar.py" --dataset cifar-10 \
  --source densenet-cifar --epochs 200 --milestones 100 150 \
  --append-catalogs /home/paretsky/input_generic_heldout.json \
                    /home/paretsky/input_cifar10_generic.json \
                    "$REPO_DIR/configs/input_generic_heldout.json" \
                    "$REPO_DIR/configs/input_cifar10_generic.json" || FAIL=1

# Unseen datasets (thesis: Fashion-MNIST, SVHN). Held out of the training JSON.
train --arch resnet20 --script "$INST/thin_res_net.py" --dataset svhn \
  --source thin-res-net --width 16 --epochs 160 --milestones 80 120 \
  --append-catalogs /home/paretsky/input_generic_heldout.json \
                    "$REPO_DIR/configs/input_generic_heldout.json" || FAIL=1

train --arch vgg11_bn --script "$INST/vgg_chenyaofo.py" --dataset svhn \
  --source vgg-chenyaofo --epochs 160 --milestones 80 120 \
  --append-catalogs /home/paretsky/input_generic_heldout.json \
                    "$REPO_DIR/configs/input_generic_heldout.json" || FAIL=1

train --arch resnet20 --script "$INST/thin_res_net.py" --dataset fashion-mnist \
  --source thin-res-net --width 16 --epochs 120 --milestones 60 90 \
  --append-catalogs /home/paretsky/input_generic_heldout.json \
                    "$REPO_DIR/configs/input_generic_heldout.json" || FAIL=1

train --arch vgg11_bn --script "$INST/vgg_chenyaofo.py" --dataset fashion-mnist \
  --source vgg-chenyaofo --epochs 120 --milestones 60 90 \
  --append-catalogs /home/paretsky/input_generic_heldout.json \
                    "$REPO_DIR/configs/input_generic_heldout.json" || FAIL=1

echo "=============================================================="
echo " finished : $(date -Is)  fail_flag=$FAIL"
echo "=============================================================="
exit $FAIL
