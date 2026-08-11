#!/usr/bin/env bash
# Launch the SPECTRA agent, adapting to however many GPUs the allocation actually has.
#
#   1 GPU  -> a single process, no NCCL, no debug servers
#   2 GPUs -> torchrun with one process per GPU; each rank explores a different network and
#             DDP averages the per-episode gradients (parallel A2C)
#
# Override the detected count with SPECTRA_GPUS, e.g. SPECTRA_GPUS=1 to force single-GPU.
#
#   bash scripts/run_agent.sh [extra runner arguments...]

set -euo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"

INPUT="${SPECTRA_INPUT:-/home/paretsky/input_no_imagenet.json}"
DATABASE="${SPECTRA_DATABASE:-/home/paretsky/database_no_imagenet.json}"
DATASETS="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100 svhn mnist}"

PASSES="${SPECTRA_PASSES:-2}"
ROLLOUT_LIMIT="${SPECTRA_ROLLOUT_LIMIT:-10}"
SEED="${SPECTRA_SEED:-42}"
N_SPLITS="${SPECTRA_N_SPLITS:-3}"
NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-2}"
SAVE_PRUNED="${SPECTRA_SAVE_PRUNED:-True}"

cd "$REPO_DIR"

# SPECTRA scales to at most two processes; a third would not add trajectories fast enough to
# offset the synchronisation cost of the per-episode update.
if [[ -n "${SPECTRA_GPUS:-}" ]]; then
  GPUS="$SPECTRA_GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPUS="$(nvidia-smi --list-gpus | wc -l)"
else
  GPUS=0
fi
(( GPUS > 2 )) && GPUS=2

if (( GPUS < 1 )); then
  echo "No GPU detected. SPECTRA experiments run on SLURM GPUs only - allocate one with:" >&2
  echo "  sinteractive --gpu rtx_6000:1 --time 0-15:00:00 --mem 120" >&2
  exit 1
fi

RUNNER_ARGS=(
  --input "$INPUT"
  --database "$DATABASE"
  --datasets $DATASETS
  --passes "$PASSES"
  --rollout_limit "$ROLLOUT_LIMIT"
  --seed "$SEED"
  --n_splits "$N_SPLITS"
  --num_epochs "$NUM_EPOCHS"
  --save_pruned_checkpoints "$SAVE_PRUNED"
  "$@"
)

if (( GPUS == 1 )); then
  echo "Detected 1 GPU -> single-process run"
  exec "$PYTHON" a2c_agent_reinforce_runner.py "${RUNNER_ARGS[@]}"
else
  echo "Detected $GPUS GPUs -> torchrun with $GPUS processes"
  exec "$PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$GPUS" \
    a2c_agent_reinforce_runner.py "${RUNNER_ARGS[@]}"
fi
