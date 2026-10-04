#!/usr/bin/env bash
# Allocation headroom probe (A0, docs/SITTING_GPU_QUEUE.md "A0") on one network: every coupled group cut once
# with L1 survivors, each allocation matched to the same kept params, then the walk's recipe A. Protocol P
# (val from the test split, batch 256) with the walk's crop+flip fine-tune, on the GPU (D5, §196).
#   AL_NET=thin-r56w4|dg-r56|cy-vgg16 [AL_ARGS="..."] sbatch --gpus=1 scripts/allocation_probe.sbatch
#SBATCH --job-name=alloc-probe
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=32G
#SBATCH --time=0-14:00:00

set -uo pipefail
REPO_DIR="${SPECTRA_REPO_DIR:-$PWD}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
NETS="${SPECTRA_NETS_DIR:-$HOME/spectra_pretrained_networks}"
INST="${SPECTRA_INST_DIR:-$HOME/spectra_models_instantiation}"
case "${AL_NET:?set AL_NET=thin-r56w4|dg-r56|cy-vgg16}" in
  thin-r56w4) NET=(--checkpoint "$NETS/resnet56-width4_cifar10_thin-res-net_88.80_0.054_8.49.pt"
                   --script "$INST/thin_res_net.py" --arch resnet56 --width 4 --dataset cifar-10 --tag thin-r56w4-c10) ;;
  dg-r56)     NET=(--checkpoint "$NETS/resnet56_cifar10_dep_graph_93.53.pth" --script "$INST/resnet_chenyaofo.py"
                   --arch resnet56 --dataset cifar-10 --tag dg-r56-c10) ;;
  cy-vgg16)   NET=(--checkpoint "$NETS/vgg16_bn_cifar10_chenyaofo_94.16_15.25_627.46.pt" --script "$INST/vgg_chenyaofo.py"
                   --arch vgg16_bn --dataset cifar-10 --tag cy-vgg16-c10) ;;
  *) echo "unknown AL_NET=$AL_NET" >&2; exit 2 ;;
esac
export SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_FT_AUG="${SPECTRA_FT_AUG:-1}" \
       SPECTRA_FT_AUG_GPU="${SPECTRA_FT_AUG_GPU:-1}"
export SPECTRA_RUN_ID="alloc_${AL_NET}_${SLURM_JOB_ID}"
export SPECTRA_RUN_DIR="$REPO_DIR/runs/allocation_probe/$SPECTRA_RUN_ID"
cd "$REPO_DIR" || exit 1
mkdir -p "$SPECTRA_RUN_DIR"
echo "alloc-probe ${AL_NET} ${SLURM_JOB_ID} on $(hostname) at $(date -Is); args: ${AL_ARGS:-defaults} aug=${SPECTRA_FT_AUG} aug_gpu=${SPECTRA_FT_AUG_GPU}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv
nvidia-smi --query-gpu=timestamp,index,name,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu \
  --format=csv,nounits -l 1 >> "$SPECTRA_RUN_DIR/gpu_samples.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

# shellcheck disable=SC2086
"$PYTHON" scripts/allocation_probe.py "${NET[@]}" ${AL_ARGS:-}
STATUS=$?
echo "alloc-probe finished with status ${STATUS} at $(date -Is); outputs in ${SPECTRA_RUN_DIR}"
exit $STATUS
