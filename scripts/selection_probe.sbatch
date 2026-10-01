#!/usr/bin/env bash
# Selection headroom probe (S0, docs/paper/FILTER_SELECTION_NAP_DESIGN.md) on one literature cell: every
# coupled group cut once to the same keep-rate, survivors chosen by each criterion, then the walk's recipe A
# at several budgets. Protocol P (val from the test split, batch 256) with the walk's crop+flip fine-tune;
# SPECTRA_FT_AUG=0 at submit gives the pre-30-Sep unaugmented recipe.
#   SEL_NET=r56|vgg16|vgg19 [SEL_NAP=1] [SEL_ARGS="..."] sbatch --gpus=rtx_4090:1 scripts/selection_probe.sbatch
#SBATCH --job-name=sel-probe
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
NAP="${NAPV2_DIR:-$HOME/scratch_audit/third_party/NAPv2}"
case "${SEL_NET:?set SEL_NET=r56|vgg16|vgg19}" in
  r56)   NET=(--checkpoint "$NETS/resnet56_cifar10_dep_graph_93.53.pth" --script "$INST/resnet_chenyaofo.py"
              --arch resnet56 --dataset cifar-10 --tag dg-r56-c10) ;;
  vgg16) NET=(--checkpoint "$NETS/vgg16_bn_cifar10_chenyaofo_94.16_15.25_627.46.pt" --script "$INST/vgg_chenyaofo.py"
              --arch vgg16_bn --dataset cifar-10 --tag cy-vgg16-c10) ;;
  vgg19) NET=(--checkpoint "$NETS/vgg19_cifar100_dep_graph_73.5.pth" --script "$INST/vgg_depgraph.py"
              --arch vgg19_bn --dataset cifar-100 --tag dg-vgg19-c100) ;;
  *) echo "unknown SEL_NET=$SEL_NET" >&2; exit 2 ;;
esac
export SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_FT_AUG="${SPECTRA_FT_AUG:-1}"
export SPECTRA_RUN_ID="sel_${SEL_NET}_${SLURM_JOB_ID}"
export SPECTRA_RUN_DIR="$REPO_DIR/runs/selection_probe/$SPECTRA_RUN_ID"
NAP_ARGS=()
[[ "${SEL_NAP:-0}" == 1 && -d "$NAP/nap2" ]] && NAP_ARGS=(--nap_dir "$NAP" --nap_interval 196)
cd "$REPO_DIR" || exit 1
mkdir -p "$SPECTRA_RUN_DIR"
echo "sel-probe ${SEL_NET} ${SLURM_JOB_ID} on $(hostname) at $(date -Is); args: ${SEL_ARGS:-defaults} nap=${#NAP_ARGS[@]} aug=${SPECTRA_FT_AUG}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv
nvidia-smi --query-gpu=timestamp,index,name,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu \
  --format=csv,nounits -l 5 >> "$SPECTRA_RUN_DIR/gpu_samples.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

# shellcheck disable=SC2086
"$PYTHON" scripts/selection_probe.py "${NET[@]}" ${SEL_ARGS:-} ${NAP_ARGS[@]+"${NAP_ARGS[@]}"}
STATUS=$?
echo "sel-probe finished with status ${STATUS} at $(date -Is); outputs in ${SPECTRA_RUN_DIR}"
exit $STATUS
