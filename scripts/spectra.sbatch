#!/bin/bash
# SPECTRA experiment as a batch job.
#
# A batch job (rather than sinteractive) is what makes an unattended, closed-loop experiment
# workable: it survives SSH disconnects, its stdout lands in a known file, and `squeue` /
# `sacct` report its state without a live terminal attached.
#
# Submit with the wrapper, which sets the profile and resources:
#     bash scripts/submit.sh smoke      # minutes: one network, few steps, correctness only
#     bash scripts/submit.sh full       # the real experiment
#
# Or directly:
#     sbatch --gpus=rtx_6000:1 --mem=120G --time=0-04:00:00 scripts/spectra.sbatch
#
#SBATCH --job-name=spectra
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home/paretsky/SPECTRA-CompressionAgent/runs/slurm_logs/spectra_%j.out
#SBATCH --error=/home/paretsky/SPECTRA-CompressionAgent/runs/slurm_logs/spectra_%j.out

set -uo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
PROFILE="${SPECTRA_PROFILE:-smoke}"

cd "$REPO_DIR" || exit 1
mkdir -p runs/slurm_logs

# Every artefact of this job lands in one directory, named after the job id, so the log,
# the structured events, the TensorBoard scalars and the result CSVs stay together.
export SPECTRA_RUN_ID="job${SLURM_JOB_ID}"
export SPECTRA_RUN_DIR="${REPO_DIR}/runs/${SPECTRA_RUN_ID}"
mkdir -p "$SPECTRA_RUN_DIR"

echo "=============================================================="
echo " SPECTRA job ${SLURM_JOB_ID} | profile=${PROFILE}"
echo " node       : $(hostname)"
echo " started    : $(date -Is)"
echo " run dir    : ${SPECTRA_RUN_DIR}"
echo " commit     : $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null)"
echo " gpus       : ${SLURM_JOB_GPUS:-?} / CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 || echo "nvidia-smi unavailable"

# A GPU-less allocation is a known SLURM failure mode on this cluster; fail fast and loudly
# rather than silently running a CPU job for hours.
if ! "$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: torch sees no CUDA device in this allocation. Resubmit (scancel + sbatch)." >&2
  exit 2
fi

# ---------------------------------------------------------------- profiles
# smoke: prove the pipeline end to end in minutes. Deliberately tiny so that a syntax or
#        logic bug surfaces immediately instead of after an hour of dataset preloading.
# full:  the actual experiment; every knob widened.
case "$PROFILE" in
  smoke)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/single_input.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/single_database.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=3
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=1
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-1}"
    EXTRA_ARGS=(--runtime_limit 900 --allowed_acc_reduction 5)
    ;;
  medium)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=1
    export SPECTRA_SAVE_PRUNED=False
    EXTRA_ARGS=(--runtime_limit 5400)
    ;;
  full)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES="${SPECTRA_PASSES:-2}"
    export SPECTRA_ROLLOUT_LIMIT="${SPECTRA_ROLLOUT_LIMIT:-10}"
    export SPECTRA_N_SPLITS="${SPECTRA_N_SPLITS:-3}"
    export SPECTRA_NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-2}"
    export SPECTRA_SAVE_PRUNED="${SPECTRA_SAVE_PRUNED:-True}"
    # Cache the database-wide feature statistics so later runs skip the fitting pass
    export SPECTRA_STANDARDIZER_PATH="${SPECTRA_STANDARDIZER_PATH:-${REPO_DIR}/runs/standardizer.pt}"
    EXTRA_ARGS=(--runtime_limit "${SPECTRA_RUNTIME_LIMIT:-36000}")
    ;;
  *)
    echo "Unknown SPECTRA_PROFILE='${PROFILE}' (expected smoke|medium|full)" >&2
    exit 1
    ;;
esac

echo "profile ${PROFILE}: input=${SPECTRA_INPUT} database=${SPECTRA_DATABASE}"
echo "datasets=${SPECTRA_DATASET_NAMES} passes=${SPECTRA_PASSES} rollout=${SPECTRA_ROLLOUT_LIMIT}"
env | grep '^SPECTRA_' | sort

bash scripts/run_agent.sh "${EXTRA_ARGS[@]}" ${SPECTRA_EXTRA_ARGS:-}
STATUS=$?

echo "=============================================================="
echo " SPECTRA job ${SLURM_JOB_ID} finished with status ${STATUS} at $(date -Is)"
echo "=============================================================="

# The digest is produced inside the job so the summary is available the moment the job ends,
# even if nobody is connected. Standard library only, so no environment activation needed.
python3 scripts/summarize_run.py "$SPECTRA_RUN_DIR" 2>&1 | tail -n 120 \
  || "$PYTHON" scripts/summarize_run.py "$SPECTRA_RUN_DIR" 2>&1 | tail -n 120

exit $STATUS
