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
# 15 minutes before the wall, ask the agent to soft-stop so eval can still run.
#SBATCH --signal=B:USR1@900
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
export SPECTRA_STOP_FILE="${SPECTRA_RUN_DIR}/SLURM_STOP"

# Soft-stop before the hard TIME LIMIT: USR1 writes a flag the agent polls each episode.
_spectra_on_usr1() {
  echo "$(date -Is) SLURM USR1: writing stop flag ${SPECTRA_STOP_FILE}" >&2
  : > "${SPECTRA_STOP_FILE}"
}
trap '_spectra_on_usr1' USR1

echo "=============================================================="
echo " SPECTRA job ${SLURM_JOB_ID} | profile=${PROFILE}"
echo " node       : $(hostname)"
echo " started    : $(date -Is)"
echo " run dir    : ${SPECTRA_RUN_DIR}"
echo " commit     : $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null)"
echo " gpus       : ${SLURM_JOB_GPUS:-?} / CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo " Timelimit  : ${SLURM_JOB_END_TIME:-unknown} (USR1 soft-stop -> ${SPECTRA_STOP_FILE})"
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
    EXTRA_ARGS=(--runtime_limit 25200)
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
    # Soft-stop 2h before the 15h SLURM wall so eval/summarize can finish.
    EXTRA_ARGS=(--runtime_limit "${SPECTRA_RUNTIME_LIMIT:-46800}")
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
    # Train soft-stops at 4h; submit.sh wall is 6h so eval is not SLURM-killed.
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
  recover_wide)
    # Include 0.7 so we can see whether the agent learns to avoid the unrecoverable rate
    # once 0.9/0.8 are actually recoverable under full FT.
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
                --compression_rates 1.0 0.9 0.8 0.7)
    ;;
  recover_pref10)
    # Same as recover but NEON-style preference knob at -10 pp (reward *formula* unchanged).
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
    EXTRA_ARGS=(--runtime_limit 14400 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  recover_king)
    # King's-road compose: mild rates + -10 pp preference + full-net FT + 40 epochs.
    # Non-id analysis: pref10 kept ~96% of real compressions inside budget; -5 pp only ~38%.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/diag_input.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/diag_database.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-8}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-20}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-1}"
    EXTRA_ARGS=(--runtime_limit 18000 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  recover_careful)
    # Literature-careful follow-up (AMC-style explore length + NEON multi-net):
    # king recipe knobs, but AMC-scale warmup floor, feature standardizer ON, and the
    # 6-net initial DB (CIFAR-10/100 × ResNet20/56 widths) instead of the 2-net diag slice.
    # Held-out eval nets stay on initial_input (3 nets). Not ImageNet-diverse yet — that
    # is a later scaling step once this recipe shows post-warmup learning.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    # AMC explores ~100 random episodes before trusting the policy; NEON used 2×#nets.
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-100}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  recover_careful_fortify)
    # A/B twin of recover_careful with SPECTRA_FORTIFY=1 (stem/width mask, entropy
    # anneal, depth/stem/coupling/width token channels). Same wall/DB/preference.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-100}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  recover_king_fortify)
    # King recipe + fortify A/B on the same 2-net diag slice as recover_king.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/diag_input.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/diag_database.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-8}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-20}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-1}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-80}"
    EXTRA_ARGS=(--runtime_limit 18000 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  recover_warm_king_fortify)
    # Continue training on the 6-net careful DB from king_fortify best weights.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-5}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-20}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_CONTINUE_TRAIN=1
    export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20063793/agent_checkpoints/latest_best_actor.pt}"
    export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20063793/agent_checkpoints/latest_best_critic.pt}"
    EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  recover_careful_fortify_ft80)
    # Careful+fortify with longer post-prune FT (closer to NEON ~100).
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=80
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-80}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-20}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    EXTRA_ARGS=(--runtime_limit 43200 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  reward_neon_ab|reward_structural_ab|reward_shaped_ab)
    # Matched A/B on king_fortify recipe; only SPECTRA_REWARD_MODE differs.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/diag_input.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/diag_database.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-8}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-20}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-1}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-80}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    case "$PROFILE" in
      reward_neon_ab) export SPECTRA_REWARD_MODE=neon ;;
      reward_structural_ab) export SPECTRA_REWARD_MODE=structural ;;
      reward_shaped_ab) export SPECTRA_REWARD_MODE=structural_shaped ;;
    esac
    EXTRA_ARGS=(--runtime_limit 18000 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  careful_fortify_structural|careful_fortify_shaped)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-100}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    if [[ "$PROFILE" == "careful_fortify_structural" ]]; then
      export SPECTRA_REWARD_MODE=structural
    else
      export SPECTRA_REWARD_MODE=structural_shaped
    fi
    EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  reward_structural_seed43)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/diag_input.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/diag_database.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-8}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-20}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-1}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_SEED=43
    export SPECTRA_REWARD_MODE=structural
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    EXTRA_ARGS=(--runtime_limit 18000 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  careful_fortify_tau15|careful_fortify_mildrates|careful_fortify_structural_guard|careful_fortify_structural_tau15)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-100}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    if [[ "$PROFILE" == "careful_fortify_structural_guard" ]]; then
      export SPECTRA_REWARD_MODE=structural_guard
      EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 10
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.9 0.8)
    elif [[ "$PROFILE" == "careful_fortify_structural_tau15" ]]; then
      export SPECTRA_REWARD_MODE=structural
      EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 15
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.9 0.8)
    elif [[ "$PROFILE" == "careful_fortify_tau15" ]]; then
      export SPECTRA_REWARD_MODE=neon
      EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 15
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.9 0.8)
    else
      # mildrates
      export SPECTRA_REWARD_MODE=neon
      EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 10
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.95 0.9)
    fi
    ;;
  probe_groupft)
    RUN_KIND=probe_groupft
    ;;
  probe_c100)
    RUN_KIND=probe_c100
    ;;
  probe_c100_extra)
    export SPECTRA_PROBE_C100_EXTRA=1
    RUN_KIND=probe_c100
    ;;
  probe_c100_aug)
    export SPECTRA_FT_AUG=1
    export SPECTRA_PROBE_RATES="${SPECTRA_PROBE_RATES:-0.99 0.98 0.95}"
    RUN_KIND=probe_c100
    ;;
  probe_c100_recipe)
    export SPECTRA_FT_AUG=1
    export SPECTRA_FT_AUTOAUG=1
    export SPECTRA_FT_OPTIM=sgd
    export SPECTRA_FT_COSINE=1
    export SPECTRA_FT_MIXUP="${SPECTRA_FT_MIXUP:-0.2}"
    export SPECTRA_FT_LABEL_SMOOTH="${SPECTRA_FT_LABEL_SMOOTH:-0.1}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-80}"
    export SPECTRA_PROBE_RATES="${SPECTRA_PROBE_RATES:-0.99 0.98 0.95}"
    export SPECTRA_PROBE_EPOCHS="${SPECTRA_PROBE_EPOCHS:-160}"
    export SPECTRA_PROBE_ROWS="${SPECTRA_PROBE_ROWS:-mid}"
    export SPECTRA_PROBE_TAU="${SPECTRA_PROBE_TAU:-10}"
    RUN_KIND=probe_c100_recipe
    ;;
  probe_c100_kd)
    export SPECTRA_FT_AUG=1
    export SPECTRA_FT_AUTOAUG=1
    export SPECTRA_FT_OPTIM=sgd
    export SPECTRA_FT_COSINE=1
    export SPECTRA_FT_MIXUP=0
    export SPECTRA_FT_LABEL_SMOOTH="${SPECTRA_FT_LABEL_SMOOTH:-0.1}"
    export SPECTRA_FT_KD=1
    export SPECTRA_FT_KD_T="${SPECTRA_FT_KD_T:-4}"
    export SPECTRA_FT_KD_ALPHA="${SPECTRA_FT_KD_ALPHA:-0.7}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-80}"
    export SPECTRA_PROBE_RATES="${SPECTRA_PROBE_RATES:-0.99 0.98 0.95}"
    export SPECTRA_PROBE_EPOCHS="${SPECTRA_PROBE_EPOCHS:-160}"
    export SPECTRA_PROBE_ROWS="${SPECTRA_PROBE_ROWS:-mid}"
    export SPECTRA_PROBE_TAU="${SPECTRA_PROBE_TAU:-10}"
    RUN_KIND=probe_c100_recipe
    ;;
  careful_fortify_cifar10|careful_fortify_cifar10_structural|careful_fortify_cifar10_mildrates|c10_curriculum_king|careful_fortify_cifar10_neon_rates|careful_fortify_cifar10_fine|encoder_c10_set|encoder_c10_wide|encoder_c10_bert|generic_c10_fortify|careful_fortify_cifar10_fast|c10_width_skinny_train|c10_budget_state)
    # Dedicated C10 JSON (not mixed initial_*). --datasets used to lazy-load C100.
    if [[ "$PROFILE" == "generic_c10_fortify" ]]; then
      export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/input_cifar10_generic.json}"
      export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/database_cifar10_generic.json}"
    elif [[ "$PROFILE" == "c10_width_skinny_train" ]]; then
      export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_c10_width_heldout.json}"
      export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_c10_thin_plus_skinny.json}"
    else
      export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/input_c10_thin.json}"
      export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/database_c10_thin.json}"
    fi
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-100}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    if [[ "$PROFILE" == "encoder_c10_set" ]]; then
      export SPECTRA_STATE_ENCODER=set
    elif [[ "$PROFILE" == "encoder_c10_wide" ]]; then
      export SPECTRA_STATE_ENCODER=transformer_wide
    elif [[ "$PROFILE" == "encoder_c10_bert" ]]; then
      export SPECTRA_STATE_ENCODER=bert
      export SPECTRA_BERT_INPUT_MODE="${SPECTRA_BERT_INPUT_MODE:-embeds}"
    else
      export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    fi
    if [[ "$PROFILE" == "careful_fortify_cifar10_fast" ]]; then
      # Experimental FT speed arm. Defaults stay off for every other profile.
      export SPECTRA_AMP=1
      export SPECTRA_CHANNELS_LAST=1
      export SPECTRA_SKIP_FT_GC=1
    fi
    if [[ "$PROFILE" == "c10_budget_state" ]]; then
      export SPECTRA_BUDGET_IN_STATE=1
    fi
    if [[ "$PROFILE" == "careful_fortify_cifar10_structural" ]]; then
      export SPECTRA_REWARD_MODE=structural
    else
      export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    fi
    if [[ "$PROFILE" == "c10_curriculum_king" ]]; then
      export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20063793/agent_checkpoints/latest_best_actor.pt}"
      export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20063793/agent_checkpoints/latest_best_critic.pt}"
      export SPECTRA_CONTINUE_TRAIN=1
    fi
    if [[ "$PROFILE" == "careful_fortify_cifar10_mildrates" ]]; then
      EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 10
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.95 0.9)
    elif [[ "$PROFILE" == "careful_fortify_cifar10_neon_rates" ]]; then
      EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 10
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.9 0.8 0.7 0.6)
    elif [[ "$PROFILE" == "careful_fortify_cifar10_fine" ]]; then
      EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 10
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.95 0.9 0.85 0.8)
    else
      EXTRA_ARGS=(--runtime_limit 36000 --allowed_acc_reduction 10
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.9 0.8)
    fi
    ;;
  offline_train)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_similar.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_offline_train.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 svhn fashion-mnist}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-100}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    EXTRA_ARGS=(--runtime_limit 43200 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  offline_wide)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_similar.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_offline_wide.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 svhn fashion-mnist}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-100}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-10}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    EXTRA_ARGS=(--runtime_limit 129600 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  eval_offline_similar|eval_offline_novel)
    if [[ "$PROFILE" == "eval_offline_novel" ]]; then
      export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_novel.json}"
    else
      export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_similar.json}"
    fi
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_offline_train.json}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_FORTIFY="${SPECTRA_FORTIFY:-1}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_CONTINUE_TRAIN=0
    if [[ -z "${SPECTRA_ACTOR_CHECKPOINT_PATH:-}" || -z "${SPECTRA_CRITIC_CHECKPOINT_PATH:-}" ]]; then
      echo "${PROFILE} requires SPECTRA_ACTOR_CHECKPOINT_PATH and SPECTRA_CRITIC_CHECKPOINT_PATH" >&2
      exit 1
    fi
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  c100_mild_structural|c100_mild_neon|c100_curriculum_king|c100_ultra_mild|c100_mild_structural_seed43)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/input_c100_thin.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/database_c100_thin.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=80
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-100}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-15}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    if [[ "$PROFILE" == "c100_mild_neon" ]]; then
      export SPECTRA_REWARD_MODE=neon
    else
      export SPECTRA_REWARD_MODE=structural
    fi
    if [[ "$PROFILE" == "c100_curriculum_king" ]]; then
      export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20063793/agent_checkpoints/latest_best_actor.pt}"
      export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20063793/agent_checkpoints/latest_best_critic.pt}"
      export SPECTRA_CONTINUE_TRAIN=1
    fi
    if [[ "$PROFILE" == "c100_mild_structural_seed43" ]]; then
      export SPECTRA_SEED=43
    fi
    if [[ "$PROFILE" == "c100_ultra_mild" ]]; then
      EXTRA_ARGS=(--runtime_limit 43200 --allowed_acc_reduction 15
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.98 0.95)
    else
      EXTRA_ARGS=(--runtime_limit 43200 --allowed_acc_reduction 15
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.95 0.9)
    fi
    ;;
  c100_wide_drl)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_c100.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_c100_wide.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-80}"
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-100}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-20}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_FT_AUG=1
    export SPECTRA_FT_AUTOAUG=1
    export SPECTRA_FT_OPTIM=sgd
    export SPECTRA_FT_COSINE=1
    export SPECTRA_FT_MIXUP="${SPECTRA_FT_MIXUP:-0.2}"
    export SPECTRA_FT_LABEL_SMOOTH="${SPECTRA_FT_LABEL_SMOOTH:-0.1}"
    EXTRA_ARGS=(--runtime_limit 129600 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.98 0.95)
    ;;
  c100_recoverable_drl)
    # Train a CIFAR-100 agent only on families that already recover / transfer:
    # VGG-11 BN, VGG-16 BN, ShuffleNet-v2×1. Held-out eval is thin residuals.
    # SGD+cosine+MixUp+AutoAugment. Do not mix unrecovered residuals into train.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_c100_residuals.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_c100_recoverable.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-80}"
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_ENTROPY_COEF="${SPECTRA_ENTROPY_COEF:-0.05}"
    export SPECTRA_WARMUP_MULTIPLIER="${SPECTRA_WARMUP_MULTIPLIER:-20}"
    export SPECTRA_WARMUP_FLOOR="${SPECTRA_WARMUP_FLOOR:-100}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-20}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_FORTIFY=1
    export SPECTRA_STEM_ROWS="${SPECTRA_STEM_ROWS:-1}"
    export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_ENTROPY_ANNEAL_HORIZON:-100}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_FT_AUG=1
    export SPECTRA_FT_AUTOAUG=1
    export SPECTRA_FT_OPTIM=sgd
    export SPECTRA_FT_COSINE=1
    export SPECTRA_FT_MIXUP="${SPECTRA_FT_MIXUP:-0.2}"
    export SPECTRA_FT_LABEL_SMOOTH="${SPECTRA_FT_LABEL_SMOOTH:-0.1}"
    EXTRA_ARGS=(--runtime_limit 129600 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.98 0.95)
    ;;
  eval_diag_structural_c100|eval_king_fortify_c100|eval_neon_c100)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/input_c100_thin.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/database_c100_thin.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_FORTIFY=1
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_CONTINUE_TRAIN=0
    if [[ "$PROFILE" == "eval_king_fortify_c100" ]]; then
      export SPECTRA_REWARD_MODE=neon
      export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20063793/agent_checkpoints/latest_best_actor.pt}"
      export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20063793/agent_checkpoints/latest_best_critic.pt}"
    elif [[ "$PROFILE" == "eval_neon_c100" ]]; then
      export SPECTRA_REWARD_MODE=neon
      export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20066578/agent_checkpoints/latest_best_actor.pt}"
      export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20066578/agent_checkpoints/latest_best_critic.pt}"
    else
      export SPECTRA_REWARD_MODE=structural
      export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20066579/agent_checkpoints/latest_best_actor.pt}"
      export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20066579/agent_checkpoints/latest_best_critic.pt}"
    fi
    EXTRA_ARGS=(--runtime_limit 3600 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  eval_offline_c100)
    # Frozen-agent held-out dataset eval (claim C9). Database must be C100 too:
    # instantiate_networks filters --database by --datasets and crashes if the
    # 10-net C10/SVHN/Fashion catalog is passed with datasets=cifar-100.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_c100.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/input_offline_c100.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-40}"
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_FORTIFY="${SPECTRA_FORTIFY:-1}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_SKIP_TRAIN=1
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_CONTINUE_TRAIN=0
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    if [[ -z "${SPECTRA_ACTOR_CHECKPOINT_PATH:-}" || -z "${SPECTRA_CRITIC_CHECKPOINT_PATH:-}" ]]; then
      echo "eval_offline_c100 requires SPECTRA_ACTOR_CHECKPOINT_PATH and SPECTRA_CRITIC_CHECKPOINT_PATH" >&2
      exit 1
    fi
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction "${SPECTRA_ALLOWED_ACC_REDUCTION:-10}"
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  eval_only)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/initial_input_no_imagenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/initial_database_no_imagenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-40}"
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_FORTIFY="${SPECTRA_FORTIFY:-1}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_SKIP_TRAIN="${SPECTRA_SKIP_TRAIN:-1}"
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_CONTINUE_TRAIN=0
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    if [[ -z "${SPECTRA_ACTOR_CHECKPOINT_PATH:-}" || -z "${SPECTRA_CRITIC_CHECKPOINT_PATH:-}" ]]; then
      echo "eval_only requires SPECTRA_ACTOR_CHECKPOINT_PATH and SPECTRA_CRITIC_CHECKPOINT_PATH" >&2
      exit 1
    fi
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction "${SPECTRA_ALLOWED_ACC_REDUCTION:-10}"
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  eval_imagenet_short)
    # Frozen 10-net actor, MobileNet-v2 ImageNet-1k, 3-epoch FT, param floor 0.70.
    # Transfer probe, not ImageNet DRL train. Prefer a 4090/6000; 2080 will OOM.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_imagenet_mobilenet.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/input_offline_imagenet_mobilenet.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-imagenet1k}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-3}"
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_FORTIFY="${SPECTRA_FORTIFY:-1}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    unset SPECTRA_EVAL_MIN_FLOP_RATIO
    export SPECTRA_SKIP_TRAIN=1
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_CONTINUE_TRAIN=0
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    export SPECTRA_BATCH_SIZE="${SPECTRA_BATCH_SIZE:-128}"
    if [[ -z "${SPECTRA_ACTOR_CHECKPOINT_PATH:-}" || -z "${SPECTRA_CRITIC_CHECKPOINT_PATH:-}" ]]; then
      echo "eval_imagenet_short requires SPECTRA_ACTOR_CHECKPOINT_PATH and SPECTRA_CRITIC_CHECKPOINT_PATH" >&2
      exit 1
    fi
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction "${SPECTRA_ALLOWED_ACC_REDUCTION:-10}"
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  eval_c10_thin_flop_floor)
    # Eval-only FLOP floor + look-ahead on C10-thin (r20-w2, r56-w4).
    # Same frozen 10-net s42 actor as 20189046 (control: r56-w4 TEST −15.9 @ 0.704/0.550).
    # Default-off everywhere else: this profile is the only one that sets the FLOP floor.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_c10_thin.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_c10_thin.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-40}"
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_FORTIFY="${SPECTRA_FORTIFY:-1}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_EVAL_MIN_FLOP_RATIO="${SPECTRA_EVAL_MIN_FLOP_RATIO:-0.70}"
    export SPECTRA_SKIP_TRAIN=1
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_CONTINUE_TRAIN=0
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-/home/paretsky/SPECTRA-CompressionAgent/runs/job20158274/agent_checkpoints/latest_best_actor.pt}"
    export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-/home/paretsky/SPECTRA-CompressionAgent/runs/job20158274/agent_checkpoints/latest_best_critic.pt}"
    if [[ -z "${SPECTRA_ACTOR_CHECKPOINT_PATH:-}" || -z "${SPECTRA_CRITIC_CHECKPOINT_PATH:-}" ]]; then
      echo "eval_c10_thin_flop_floor requires SPECTRA_ACTOR_CHECKPOINT_PATH and SPECTRA_CRITIC_CHECKPOINT_PATH" >&2
      exit 1
    fi
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction "${SPECTRA_ALLOWED_ACC_REDUCTION:-10}"
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  baseline_c10_l1|baseline_c10_mild|baseline_c10_random)
    # Same-loop structured baselines vs DRL: same rates, FT, fortify, floor 0.70.
    # Env still L1-ranks filters; these policies only pick the compression rate.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_c10_thin.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_c10_thin.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=40
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_FORTIFY="${SPECTRA_FORTIFY:-1}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_SKIP_TRAIN=1
    export SPECTRA_SKIP_STANDARDIZER=1
    export SPECTRA_CONTINUE_TRAIN=0
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    if [[ "$PROFILE" == "baseline_c10_mild" ]]; then
      export SPECTRA_EVAL_POLICY=mild
    elif [[ "$PROFILE" == "baseline_c10_random" ]]; then
      export SPECTRA_EVAL_POLICY=random
    else
      export SPECTRA_EVAL_POLICY=l1
    fi
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  *)
    echo "Unknown SPECTRA_PROFILE='${PROFILE}'" >&2
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
elif [[ "${RUN_KIND:-agent}" == "probe_groupft" ]]; then
  echo "profile probe_groupft: layer-only recovery under group-aware freeze"
  env | grep '^SPECTRA_' | sort
  bash scripts/run_recovery_probe_groupft.sh
  STATUS=$?
elif [[ "${RUN_KIND:-agent}" == "probe_c100" ]]; then
  echo "profile probe_c100: CIFAR-100 rate×epochs recovery (no RL)"
  env | grep '^SPECTRA_' | sort
  bash scripts/run_recovery_probe_c100.sh
  STATUS=$?
elif [[ "${RUN_KIND:-agent}" == "probe_c100_recipe" ]]; then
  echo "profile ${PROFILE}: CIFAR-100 published-style FT recovery (no RL)"
  env | grep '^SPECTRA_' | sort
  bash scripts/run_recovery_probe_c100_recipe.sh
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
