#!/bin/bash
# SPECTRA experiment as a batch job.
#
# A batch job (rather than sinteractive) is what makes an unattended, closed-loop experiment
# workable: it survives SSH disconnects, its stdout lands in a known file, and `squeue` /
# `sacct` report its state without a live terminal attached.
#
# Submit with the wrapper, which sets the profile and resources:
#     bash scripts/submit.sh smoke      # minutes: one network, few steps, correctness only
#     bash scripts/submit.sh medium     # hours: realistic fine-tune + exploration, small DB
#     bash scripts/submit.sh full       # the real experiment
#     bash scripts/submit.sh probe      # ~2h: rate×epochs recovery matrix (no RL)
#     bash scripts/submit.sh diag       # ~2.5h: short agent run on 2 nets
#
# Or directly:
#     sbatch --gpus=rtx_6000:1 --mem=80G --time=0-08:00:00 scripts/spectra.sbatch
#
#SBATCH --job-name=spectra
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem=80G
#SBATCH --time=0-08:00:00
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
# smoke:  prove the pipeline end to end in minutes. Tiny fine-tune / short warmup on purpose.
# medium: realistic fine-tune + exploration on the small initial DB (progress, not correctness).
# full:   the actual experiment; every knob at thesis-scale defaults.
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
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-2}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-4}"
    EXTRA_ARGS=(--runtime_limit 900 --allowed_acc_reduction 5)
    ;;
  medium)
    # 20 fine-tune epochs (early-stop patience 10) + 20x network-count warmup. Enough for the
    # agent to see recoverable compressions, without the full 40-epoch / 2-pass budget.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=10
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=20
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    EXTRA_ARGS=(--runtime_limit 28800)
    ;;
  full)
    # NEON used 100 fine-tune epochs; SPECTRA's documented default is 40 (see argparse /
    # ClassificationHandler). Warmup and entropy match the medium profile's exploration fix.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES="${SPECTRA_PASSES:-2}"
    export SPECTRA_ROLLOUT_LIMIT="${SPECTRA_ROLLOUT_LIMIT:-10}"
    export SPECTRA_N_SPLITS="${SPECTRA_N_SPLITS:-3}"
    export SPECTRA_NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-40}"
    export SPECTRA_SAVE_PRUNED="${SPECTRA_SAVE_PRUNED:-True}"
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    # Cache the database-wide feature statistics so later runs skip the fitting pass
    export SPECTRA_STANDARDIZER_PATH="${SPECTRA_STANDARDIZER_PATH:-${REPO_DIR}/runs/standardizer.pt}"
    EXTRA_ARGS=(--runtime_limit "${SPECTRA_RUNTIME_LIMIT:-54000}")
    ;;
  probe)
    # No RL: rate × epochs recovery matrix on 1–2 nets, both freeze policies.
    # Answers whether fine-tune budgets can recover before another long agent run.
    RUN_KIND=probe
    ;;
  probe_continue)
    # Finish nets/scopes skipped when an earlier probe job aborted on set -e.
    RUN_KIND=probe_continue
    ;;
  diag)
    # Short agent probe on 2 CIFAR-10 nets: enough episodes to see whether exploration
    # ever produces recoverable compressions, without an 8h wall.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/diag_input.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/diag_database.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=20
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-5}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-10}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-1}"
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction 5)
    ;;
  recover)
    # Probe lead #1: full-network fine-tune + 40 epochs + milder rates (1.0/0.9/0.8).
    # Today's matrices showed only these settings ever stay inside the -5 pp cliff.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/diag_input.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/diag_database.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-5}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-10}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-1}"
    EXTRA_ARGS=(--runtime_limit 14400 --allowed_acc_reduction 5
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  recover_groupft)
    # Probe lead #2: keep layer-only FT but train every module rewritten by the group prune
    # (producers/consumers/norms), still with 40 epochs and mild rates.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/diag_input.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/diag_database.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-5}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-10}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-1}"
    EXTRA_ARGS=(--runtime_limit 14400 --allowed_acc_reduction 5
                --train_compressed_layer_only True
                --compression_rates 1.0 0.9 0.8)
    ;;
  *)
    echo "Unknown SPECTRA_PROFILE='${PROFILE}' (expected smoke|medium|full|probe|probe_continue|diag|recover|recover_groupft)" >&2
    exit 1
    ;;
esac

if [[ "${RUN_KIND:-agent}" == "probe" ]]; then
  echo "profile probe: recovery matrix (no RL)"
  env | grep '^SPECTRA_' | sort
  bash scripts/run_recovery_probe.sh
  STATUS=$?
elif [[ "${RUN_KIND:-agent}" == "probe_continue" ]]; then
  echo "profile probe_continue: remaining recovery matrices"
  env | grep '^SPECTRA_' | sort
  bash scripts/run_recovery_probe_continue.sh
  STATUS=$?
else
  echo "profile ${PROFILE}: input=${SPECTRA_INPUT} database=${SPECTRA_DATABASE}"
  echo "datasets=${SPECTRA_DATASET_NAMES} passes=${SPECTRA_PASSES} rollout=${SPECTRA_ROLLOUT_LIMIT}"
  echo "num_epochs=${SPECTRA_NUM_EPOCHS} entropy_coef=${SPECTRA_ENTROPY_COEF:-default} warmup_mult=${SPECTRA_WARMUP_MULTIPLIER:-default}"
  env | grep '^SPECTRA_' | sort

  bash scripts/run_agent.sh "${EXTRA_ARGS[@]}" ${SPECTRA_EXTRA_ARGS:-}
  STATUS=$?
fi

echo "=============================================================="
echo " SPECTRA job ${SLURM_JOB_ID} finished with status ${STATUS} at $(date -Is)"
echo "=============================================================="

# The digest is produced inside the job so the summary is available the moment the job ends,
# even if nobody is connected. Standard library only, so no environment activation needed.
python3 scripts/summarize_run.py "$SPECTRA_RUN_DIR" 2>&1 | tail -n 120 \
  || "$PYTHON" scripts/summarize_run.py "$SPECTRA_RUN_DIR" 2>&1 | tail -n 120

exit $STATUS
