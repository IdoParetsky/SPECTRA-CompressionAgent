#!/usr/bin/env bash
# S2 (docs/paper/FILTER_SELECTION_NAP_DESIGN.md §8, gate G2): S0's probe plus S1's learned scorer (nap_f) on a
# cell none of S1's training nets came from. Keep 0.6 only; L1 and nap_f under the same 5 fine-tune seeds, the
# oracle under 3, 3 random masks, anti-L1 once; budgets 0 / BN / 1 / 3 / 10 / 40. Protocol P with the walk's
# crop+flip fine-tune, as S0.
#   S2_NET=mbv2|r56c100 S2_MODEL=<S1 export .pkl> sbatch --gpus=1 scripts/selection_s2.sbatch
#SBATCH --job-name=sel-s2
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=32G
#SBATCH --time=0-10:00:00

set -uo pipefail
REPO_DIR="${SPECTRA_REPO_DIR:-$PWD}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
NETS="${SPECTRA_NETS_DIR:-$HOME/spectra_pretrained_networks}"
INST="${SPECTRA_INST_DIR:-$HOME/spectra_models_instantiation}"
case "${S2_NET:?set S2_NET=mbv2|r56c100}" in
  mbv2)    NET=(--checkpoint "$NETS/mobilenet-v2x0.5_cifar10_chenyaofo_92.99_0.7_55.94.pt"
                --script "$INST/mobilenetv2_chenyaofo.py" --arch mobilenet_v2x05 --dataset cifar-10 --tag cy-mbv2x05-c10) ;;
  r56c100) NET=(--checkpoint "$NETS/resnet56_cifar100_chenyaofo_72.63_0.86_251.5.pt"
                --script "$INST/resnet_chenyaofo.py" --arch resnet56 --dataset cifar-100 --tag cy-r56-c100) ;;
  *) echo "unknown S2_NET=$S2_NET" >&2; exit 2 ;;
esac
export SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_FT_AUG="${SPECTRA_FT_AUG:-1}"
export SPECTRA_RUN_ID="s2_${S2_NET}_${SLURM_JOB_ID}"
export SPECTRA_RUN_DIR="$REPO_DIR/runs/selection_probe/$SPECTRA_RUN_ID"
cd "$REPO_DIR" || exit 1
mkdir -p "$SPECTRA_RUN_DIR"
echo "sel-s2 ${S2_NET} ${SLURM_JOB_ID} on $(hostname) at $(date -Is); model ${S2_MODEL:?set S2_MODEL} aug=${SPECTRA_FT_AUG}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv
nvidia-smi --query-gpu=timestamp,index,name,power.draw,utilization.gpu,memory.used,clocks.sm,temperature.gpu \
  --format=csv,nounits -l 5 >> "$SPECTRA_RUN_DIR/gpu_samples.csv" 2>/dev/null &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

# shellcheck disable=SC2086
"$PYTHON" scripts/selection_probe_s2.py --nap_f_model "$S2_MODEL" --oracle_seeds 3 "${NET[@]}" \
  --keep 0.6 --budgets 0 bn 1 3 10 40 --criteria l1 nap_f ablation random anti_l1 \
  --l1_seeds 5 --random_masks 3 ${S2_ARGS:-}
STATUS=$?
echo "sel-s2 finished with status ${STATUS} at $(date -Is); outputs in ${SPECTRA_RUN_DIR}"
exit $STATUS
