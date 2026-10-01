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
#     sbatch --gpus=rtx_6000:1 --mem-per-gpu=24G --time=0-08:00:00 scripts/spectra.sbatch
#
#SBATCH --job-name=spectra
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_6000:1
#SBATCH --mem-per-gpu=24G
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
# Continue-train: copy resume + latest_best into THIS job, then drop SPECTRA_RESUME_PATH
# so later save_train_resume writes here instead of overwriting the parent snapshot.
# A requeue re-enters with the same run dir: its own (newer) bundle and bests are kept.
if [[ -n "${SPECTRA_PARENT_RUN:-}" || -n "${SPECTRA_RESUME_PATH:-}" ]]; then
  mkdir -p "${SPECTRA_RUN_DIR}/agent_checkpoints"
  if [[ -n "${SPECTRA_RESUME_PATH:-}" && -f "${SPECTRA_RESUME_PATH}" ]]; then
    if [[ -s "${SPECTRA_RUN_DIR}/agent_checkpoints/train_resume.pt" ]]; then
      echo "requeue: keeping this run's train_resume.pt, not re-copying ${SPECTRA_RESUME_PATH}"
    else
      cp -f "${SPECTRA_RESUME_PATH}" "${SPECTRA_RUN_DIR}/agent_checkpoints/train_resume.pt"
      echo "copied resume bundle from ${SPECTRA_RESUME_PATH}"
    fi
    unset SPECTRA_RESUME_PATH
  fi
  if [[ -n "${SPECTRA_PARENT_RUN:-}" ]]; then
    for _f in latest_best_actor.pt latest_best_critic.pt; do
      if [[ -s "${SPECTRA_RUN_DIR}/agent_checkpoints/${_f}" ]]; then
        echo "requeue: keeping this run's ${_f}"
      elif [[ -f "${SPECTRA_PARENT_RUN}/agent_checkpoints/${_f}" ]]; then
        cp -f "${SPECTRA_PARENT_RUN}/agent_checkpoints/${_f}" "${SPECTRA_RUN_DIR}/agent_checkpoints/${_f}"
        echo "seeded ${_f} from ${SPECTRA_PARENT_RUN}"
      fi
    done
  fi
fi

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
  smoke_v2)
    # Minutes-long integration gate for the 13 Sep v2 recipe (PPO + slack state + group-once
    # + (rate, ranking) actions + train-FT cap). Chain the long v2 arms afterok this job.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-/home/paretsky/single_input.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-/home/paretsky/single_database.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=6
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS=1
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_SKIP_STANDARDIZER=0
    export SPECTRA_FORTIFY=1
    export SPECTRA_ALGO=ppo
    export SPECTRA_PPO_EPISODES=2
    export SPECTRA_PPO_EPOCHS=2
    export SPECTRA_PPO_WARMUP_EPISODES=0
    export SPECTRA_AGENT_LR=3e-4
    export SPECTRA_ENTROPY_COEF=0.01
    export SPECTRA_ENCODER_DROPOUT=0
    export SPECTRA_POLICY_HEAD_ZERO_INIT=1
    export SPECTRA_STATE_ALIGN=next
    export SPECTRA_STATE_SLACK=1
    export SPECTRA_BUDGET_IN_STATE=1
    export SPECTRA_GROUP_ONCE_PER_PASS=1
    export SPECTRA_TRAIN_FT_EPOCHS=1
    export SPECTRA_CHECKPOINT=val_best
    export SPECTRA_SNAPSHOT_BASELINE=0
    export SPECTRA_REWARD_MODE=structural
    export SPECTRA_REWARD_SCALE=cbrt
    export SPECTRA_REWARD_TRACE=1
    export SPECTRA_EVAL_DETERMINISTIC=1
    export SPECTRA_EVAL_TRAJECTORY=1
    export SPECTRA_EVAL_LOOKAHEAD=0
    export SPECTRA_SKIP_EVAL_TRAIN=1
    export SPECTRA_EVAL_MIN_PARAM_RATIO=0.70
    unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH SPECTRA_RESUME_PATH SPECTRA_PARENT_RUN
    export SPECTRA_CONTINUE_TRAIN=0
    export SPECTRA_SKIP_TRAIN=0
    EXTRA_ARGS=(--runtime_limit 600 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8 0.9 0.8
                --action_rankings l1 l1 l1 fpgm fpgm)
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
  reward_neon_ab|reward_structural_ab|reward_shaped_ab|reward_band_ab)
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
      reward_band_ab) export SPECTRA_REWARD_MODE=structural_band ;;
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
  offline_train|offline_train_cbrt|offline_train_band_cbrt|offline_train_unified|offline_train_prefer|offline_train_unified_full|offline_train_prefer_floor|offline_train_neon_full|offline_train_gonce_cold|offline_train_v2a|offline_train_v2b|offline_train_v2c|offline_train_v3_fpgm|offline_train_v3_svd|offline_train_v3_bnscale|offline_train_v3_fpgm_neonraw|offline_train_v3_fpgm_structraw|offline_train_v4_factored|offline_train_v4_factored_tau6|offline_train_v5_p5b3|offline_train_v5_p5b3_cgp|offline_train_v5_ft40|offline_train_v6_inband|offline_train_v6_inband_p5b2|offline_train_v6_inband_p5b2_factored|offline_train_v7_budget|offline_train_v8_grouptoken|offline_train_v9_diverse)
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_similar.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_offline_train.json}"
    if [[ "$PROFILE" == offline_train_v5_p5b3* ]]; then
      # P5-B3 trains on CIFAR-10 + recoverable CIFAR-100 only (SVHN / Fashion-MNIST are TEST hold-outs).
      export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
    fi
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
    # Named A/B profiles pin the switch in the sbatch file so --export=ALL
    # dropping extras cannot void the arm. eval of these actors uses argmax.
    if [[ "$PROFILE" == "offline_train_cbrt" ]]; then
      export SPECTRA_REWARD_MODE=neon
      export SPECTRA_REWARD_SCALE=cbrt
      export SPECTRA_REWARD_TRACE=1
      export SPECTRA_EVAL_DETERMINISTIC=1
    elif [[ "$PROFILE" == "offline_train_band_cbrt" ]]; then
      # Full-net 128-step twin of neon_full. Do NOT skip over-budget actor
      # updates: overshoot grading is the empty-band layer-selection signal.
      # 20945744 was this profile with leftover rollout_limit=5 — invalid.
      export SPECTRA_REWARD_MODE=structural_band
      export SPECTRA_REWARD_SCALE=cbrt
      export SPECTRA_REWARD_TRACE=1
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_CHECKPOINT=inbudget_compression
      export SPECTRA_STATE_ALIGN=next
      export SPECTRA_ROLLOUT_LIMIT=128
      export SPECTRA_WARMUP_MULTIPLIER=3
      export SPECTRA_WARMUP_FLOOR=30
      export SPECTRA_SKIP_EVAL=1
      unset SPECTRA_ACTOR_SKIP_OVERBUDGET
      unset SPECTRA_SKIP_EVAL_TRAIN
      export SPECTRA_SKIP_TRAIN=0
      if [[ -z "${SPECTRA_RESUME_PATH:-}" && "${SPECTRA_CONTINUE_TRAIN:-}" != "1" ]]; then
        unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH
        export SPECTRA_CONTINUE_TRAIN=0
      else
        export SPECTRA_CONTINUE_TRAIN=1
      fi
    elif [[ "$PROFILE" == "offline_train_unified" ]]; then
      # F1: hard unified, no cubes, no cbrt. Fresh actor (not a frozen warm-start).
      export SPECTRA_REWARD_MODE=structural_unified
      export SPECTRA_REWARD_SCALE=raw
      export SPECTRA_REWARD_TRACE=1
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_CHECKPOINT=inbudget_compression
      export SPECTRA_UNIFIED_EPS="${SPECTRA_UNIFIED_EPS:-1}"
      unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH
      unset SPECTRA_SKIP_EVAL_TRAIN
      export SPECTRA_SKIP_TRAIN=0
      export SPECTRA_CONTINUE_TRAIN=0
    elif [[ "$PROFILE" == "offline_train_prefer" ]]; then
      # Full-net prefer: F1 over-budget + in-budget ρ (no slack taper).
      export SPECTRA_REWARD_MODE=structural_prefer
      export SPECTRA_REWARD_SCALE=raw
      export SPECTRA_REWARD_TRACE=1
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_CHECKPOINT=inbudget_compression
      export SPECTRA_UNIFIED_EPS="${SPECTRA_UNIFIED_EPS:-1}"
      export SPECTRA_STATE_ALIGN=next
      export SPECTRA_ROLLOUT_LIMIT=64
      export SPECTRA_WARMUP_MULTIPLIER=3
      export SPECTRA_WARMUP_FLOOR=30
      export SPECTRA_SKIP_EVAL=1
      unset SPECTRA_SKIP_EVAL_TRAIN
      export SPECTRA_SKIP_TRAIN=0
      if [[ -z "${SPECTRA_RESUME_PATH:-}" && "${SPECTRA_CONTINUE_TRAIN:-}" != "1" ]]; then
        unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH
        export SPECTRA_CONTINUE_TRAIN=0
      else
        export SPECTRA_CONTINUE_TRAIN=1
      fi
    elif [[ "$PROFILE" == "offline_train_unified_full" ]]; then
      # F1 slack-taper, same full-net loop as prefer (fair reward A/B vs 21168773).
      export SPECTRA_REWARD_MODE=structural_unified
      export SPECTRA_REWARD_SCALE=raw
      export SPECTRA_REWARD_TRACE=1
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_CHECKPOINT=inbudget_compression
      export SPECTRA_UNIFIED_EPS="${SPECTRA_UNIFIED_EPS:-1}"
      export SPECTRA_STATE_ALIGN=next
      export SPECTRA_ROLLOUT_LIMIT=128
      export SPECTRA_WARMUP_MULTIPLIER=3
      export SPECTRA_WARMUP_FLOOR=30
      export SPECTRA_SKIP_EVAL=1
      unset SPECTRA_SKIP_EVAL_TRAIN
      export SPECTRA_SKIP_TRAIN=0
      if [[ -z "${SPECTRA_RESUME_PATH:-}" && "${SPECTRA_CONTINUE_TRAIN:-}" != "1" ]]; then
        unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH
        export SPECTRA_CONTINUE_TRAIN=0
      else
        export SPECTRA_CONTINUE_TRAIN=1
      fi
    elif [[ "$PROFILE" == "offline_train_prefer_floor" ]]; then
      # Prefer + train-time 0.70 floor/look-ahead + budget channel (paper-protocol MDP).
      export SPECTRA_REWARD_MODE=structural_prefer
      export SPECTRA_REWARD_SCALE=raw
      export SPECTRA_REWARD_TRACE=1
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_CHECKPOINT=inbudget_compression
      export SPECTRA_UNIFIED_EPS="${SPECTRA_UNIFIED_EPS:-1}"
      export SPECTRA_STATE_ALIGN=next
      export SPECTRA_ROLLOUT_LIMIT=128
      export SPECTRA_WARMUP_MULTIPLIER=3
      export SPECTRA_WARMUP_FLOOR=30
      export SPECTRA_SKIP_EVAL=1
      export SPECTRA_TRAIN_RESPECT_FLOOR=1
      export SPECTRA_EVAL_LOOKAHEAD=1
      export SPECTRA_BUDGET_IN_STATE=1
      unset SPECTRA_SKIP_EVAL_TRAIN
      export SPECTRA_SKIP_TRAIN=0
      if [[ -z "${SPECTRA_RESUME_PATH:-}" && "${SPECTRA_CONTINUE_TRAIN:-}" != "1" ]]; then
        unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH
        export SPECTRA_CONTINUE_TRAIN=0
      else
        export SPECTRA_CONTINUE_TRAIN=1
      fi
    elif [[ "$PROFILE" == "offline_train_neon_full" || "$PROFILE" == "offline_train_gonce_cold" ]]; then
      # Original NEON trichotomy on *realized* CNN cuts, cube-root critic scale.
      # In-budget checkpoint; empty-band walks skip the actor; mixed walks mask
      # only the violating steps so cubes cannot fold latest_best to identity.
      export SPECTRA_REWARD_MODE=structural
      export SPECTRA_REWARD_SCALE=cbrt
      export SPECTRA_REWARD_TRACE=1
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_CHECKPOINT=inbudget_compression
      export SPECTRA_ACTOR_SKIP_OVERBUDGET=1
      export SPECTRA_STATE_ALIGN=next
      export SPECTRA_ROLLOUT_LIMIT=128
      export SPECTRA_WARMUP_MULTIPLIER=3
      export SPECTRA_WARMUP_FLOOR=30
      export SPECTRA_SKIP_EVAL=1
      unset SPECTRA_SKIP_EVAL_TRAIN
      export SPECTRA_SKIP_TRAIN=0
      if [[ "$PROFILE" == "offline_train_gonce_cold" ]]; then
        # ONE lever on the frozen neon_full recipe (control: cold 21168838 lineage):
        # each coupled group is cut at most once per pass. Always a cold actor — a
        # warm-start from a near-uniform policy (job20158274 family) is not a new MDP.
        # SPECTRA_SNAPSHOT_BASELINE freezes a TESTable copy whenever latest_best
        # clears it (default: any completed in-budget episode).
        export SPECTRA_GROUP_ONCE_PER_PASS=1
        export SPECTRA_SNAPSHOT_BASELINE="${SPECTRA_SNAPSHOT_BASELINE:-0}"
        unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH SPECTRA_RESUME_PATH SPECTRA_PARENT_RUN
        # The continue-train prologue above may already have seeded this run dir.
        rm -f "${SPECTRA_RUN_DIR}/agent_checkpoints/train_resume.pt" \
              "${SPECTRA_RUN_DIR}/agent_checkpoints/latest_best_actor.pt" \
              "${SPECTRA_RUN_DIR}/agent_checkpoints/latest_best_critic.pt"
        export SPECTRA_CONTINUE_TRAIN=0
      elif [[ -z "${SPECTRA_RESUME_PATH:-}" && "${SPECTRA_CONTINUE_TRAIN:-}" != "1" ]]; then
        unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH
        export SPECTRA_CONTINUE_TRAIN=0
      else
        export SPECTRA_CONTINUE_TRAIN=1
      fi
    fi
    if [[ "$PROFILE" == offline_train_v2* ]]; then
      # ---- v2 (13 Sep audit): the recipe that can actually move a policy off uniform.
      # One optimiser (PPO over 4-episode batches, GAE, return scaling, clipped surrogate),
      # a state that can see the τ band (accuracy slack + progress + kept ratio), a walk
      # that cuts each stream once per pass (group-once), a cold zero-init policy head,
      # no encoder dropout, next-row marker, and a shorter recovery only during policy
      # training (TEST keeps --num_epochs 40 for every method). Arms:
      #   v2a: 3 rates, structural + cbrt (NEON trichotomy on realised cut)
      #   v2b: v2a + (rate, ranking) actions: 1.0 | 0.9/0.8 x {l1, fpgm}  (5 actions)
      #   v2c: v2a with the original NEON reward on the nominal rate, raw scale
      export SPECTRA_ALGO=ppo
      export SPECTRA_PPO_EPISODES="${SPECTRA_PPO_EPISODES:-4}"
      export SPECTRA_PPO_EPOCHS="${SPECTRA_PPO_EPOCHS:-4}"
      export SPECTRA_PPO_CLIP="${SPECTRA_PPO_CLIP:-0.2}"
      export SPECTRA_PPO_GAE_LAMBDA="${SPECTRA_PPO_GAE_LAMBDA:-0.95}"
      export SPECTRA_PPO_TARGET_KL="${SPECTRA_PPO_TARGET_KL:-0.03}"
      export SPECTRA_PPO_WARMUP_EPISODES="${SPECTRA_PPO_WARMUP_EPISODES:-0}"
      export SPECTRA_AGENT_LR="${SPECTRA_AGENT_LR:-3e-4}"
      # The shared arm header above already exported SPECTRA_ENTROPY_COEF=0.05 (A2C value),
      # so a plain ":-" default would keep it. PPO uses 0.01 unless SPECTRA_V2_ENTROPY_COEF.
      export SPECTRA_ENTROPY_COEF="${SPECTRA_V2_ENTROPY_COEF:-0.01}"
      export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_V2_ENTROPY_ANNEAL_HORIZON:-200}"
      export SPECTRA_ENTROPY_MIN="${SPECTRA_V2_ENTROPY_MIN:-0.002}"
      export SPECTRA_ENCODER_DROPOUT=0
      export SPECTRA_POLICY_HEAD_ZERO_INIT=1
      export SPECTRA_STATE_ALIGN=next
      export SPECTRA_STATE_SLACK=1
      export SPECTRA_BUDGET_IN_STATE=1
      export SPECTRA_GROUP_ONCE_PER_PASS=1
      export SPECTRA_TRAIN_FT_EPOCHS="${SPECTRA_TRAIN_FT_EPOCHS:-12}"
      export SPECTRA_TRAIN_FT_PATIENCE="${SPECTRA_TRAIN_FT_PATIENCE:-4}"
      export SPECTRA_ROLLOUT_LIMIT=128
      export SPECTRA_CHECKPOINT=val_best
      export SPECTRA_SNAPSHOT_BASELINE="${SPECTRA_SNAPSHOT_BASELINE:-0.05}"
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_REWARD_TRACE=1
      export SPECTRA_SKIP_EVAL=1
      unset SPECTRA_ACTOR_SKIP_OVERBUDGET SPECTRA_TRAIN_RESPECT_FLOOR SPECTRA_SKIP_EVAL_TRAIN
      unset SPECTRA_FILTER_IMPORTANCE
      if [[ "$PROFILE" == "offline_train_v2c" ]]; then
        export SPECTRA_REWARD_MODE=neon
        export SPECTRA_REWARD_SCALE=raw
      else
        export SPECTRA_REWARD_MODE=structural
        export SPECTRA_REWARD_SCALE=cbrt
      fi
      # Always cold: a warm-start from a near-uniform policy is not a new MDP.
      unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH SPECTRA_RESUME_PATH SPECTRA_PARENT_RUN
      rm -f "${SPECTRA_RUN_DIR}/agent_checkpoints/train_resume.pt" \
            "${SPECTRA_RUN_DIR}/agent_checkpoints/latest_best_actor.pt" \
            "${SPECTRA_RUN_DIR}/agent_checkpoints/latest_best_critic.pt"
      export SPECTRA_CONTINUE_TRAIN=0
      export SPECTRA_SKIP_TRAIN=0
    fi
    if [[ "$PROFILE" == offline_train_v3_* || "$PROFILE" == offline_train_v4_* || "$PROFILE" == offline_train_v5_* || "$PROFILE" == offline_train_v6_* || "$PROFILE" == offline_train_v7_* || "$PROFILE" == offline_train_v8_* || "$PROFILE" == offline_train_v9_* ]]; then
      # ---- v3 (16 Sep last train). Everything v2a/v2b had, plus what the v2 traces showed
      # was missing: (1) band-edge exposure — B saw the τ edge on 2.5 % of steps, so "stop
      # from slack" had no signal → --passes 2 in training (TEST replays 2 passes via
      # policy_config); (2) per-layer group cost in the state (SPECTRA_STATE_GROUPCOST);
      # (3) a learning governor: deterministic fixed-probe selection score (two thin train
      # ResNets, argmax walks), min lifetime 250, patience 150 on the probe, and Ido's
      # probe-gated rewind to the elite with an entropy bump (SPECTRA_REWIND_BEST);
      # (4) entropy floor 0.005 so a late peaked policy can still explore.
      # Arms differ only in RANK (5-action menu 1.0 | 0.9/0.8 x {l1, RANK}) and, for the
      # 4th, the reward scale. v2 profiles above are untouched (byte-identical replays).
      # Catalog: the 24-net wide train (superset of the 10-net set; adds thin ResNets at
      # widths 7-14 that sit between the train widths and the held-out r20-w2 / r56-w4,
      # zero overlap with similar / unlike / thin / C100 / ImageNet hold-outs — checked
      # 16 Sep). Override with SPECTRA_V3_DATABASE=.../database_offline_train.json for 10.
      export SPECTRA_DATABASE="${SPECTRA_V3_DATABASE:-${REPO_DIR}/configs/database_offline_wide.json}"
      export SPECTRA_ALGO=ppo
      export SPECTRA_PPO_EPISODES="${SPECTRA_PPO_EPISODES:-4}"
      export SPECTRA_PPO_EPOCHS="${SPECTRA_PPO_EPOCHS:-4}"
      export SPECTRA_PPO_CLIP="${SPECTRA_PPO_CLIP:-0.2}"
      export SPECTRA_PPO_GAE_LAMBDA="${SPECTRA_PPO_GAE_LAMBDA:-0.95}"
      export SPECTRA_PPO_TARGET_KL="${SPECTRA_PPO_TARGET_KL:-0.03}"
      export SPECTRA_PPO_WARMUP_EPISODES="${SPECTRA_PPO_WARMUP_EPISODES:-0}"
      export SPECTRA_AGENT_LR="${SPECTRA_AGENT_LR:-3e-4}"
      export SPECTRA_ENTROPY_COEF="${SPECTRA_V3_ENTROPY_COEF:-0.01}"
      export SPECTRA_ENTROPY_ANNEAL_HORIZON="${SPECTRA_V3_ENTROPY_ANNEAL_HORIZON:-300}"
      export SPECTRA_ENTROPY_MIN="${SPECTRA_V3_ENTROPY_MIN:-0.005}"
      export SPECTRA_ENCODER_DROPOUT=0
      export SPECTRA_POLICY_HEAD_ZERO_INIT=1
      export SPECTRA_STATE_ALIGN=next
      export SPECTRA_STATE_SLACK=1
      export SPECTRA_BUDGET_IN_STATE=1
      export SPECTRA_STATE_GROUPCOST=1
      export SPECTRA_GROUP_ONCE_PER_PASS=1
      export SPECTRA_TRAIN_FT_EPOCHS="${SPECTRA_TRAIN_FT_EPOCHS:-12}"
      export SPECTRA_TRAIN_FT_PATIENCE="${SPECTRA_TRAIN_FT_PATIENCE:-4}"
      export SPECTRA_ROLLOUT_LIMIT=128
      export SPECTRA_PASSES=2
      export SPECTRA_CHECKPOINT=val_best
      export SPECTRA_SNAPSHOT_BASELINE="${SPECTRA_SNAPSHOT_BASELINE:-0.05}"
      export SPECTRA_PROBE_EVERY="${SPECTRA_PROBE_EVERY:-12}"
      # This default fires before the v5/v6 blocks below, so every v3–V8 actor was selected on
      # the two thin ResNets (audit 28 Sep), never on the VGG-13 + r56-w6 set V7 §1.1 asked for.
      # SPECTRA_PROBE_SET=v7 switches to that set — the one change of a new train, never mid-run.
      case "${SPECTRA_PROBE_SET:-}" in
        v7) export SPECTRA_PROBE_NETS="vgg13_bn_cifar10_,resnet56-width6" ;;
        ""|thin) ;;
        *) echo "unknown SPECTRA_PROBE_SET=${SPECTRA_PROBE_SET} (v7|thin)" >&2; exit 1 ;;
      esac
      export SPECTRA_PROBE_NETS="${SPECTRA_PROBE_NETS:-resnet56-width6,resnet20-width10}"
      export SPECTRA_MIN_EPISODES="${SPECTRA_MIN_EPISODES:-250}"
      export SPECTRA_PATIENCE_EPISODES="${SPECTRA_PATIENCE_EPISODES:-150}"
      export SPECTRA_REWIND_BEST="${SPECTRA_REWIND_BEST:-1}"
      export SPECTRA_REWIND_PATIENCE="${SPECTRA_REWIND_PATIENCE:-50}"
      export SPECTRA_REWIND_MAX="${SPECTRA_REWIND_MAX:-3}"
      export SPECTRA_REWIND_ENTROPY="${SPECTRA_REWIND_ENTROPY:-0.02}"
      export SPECTRA_REWIND_ENTROPY_EPISODES="${SPECTRA_REWIND_ENTROPY_EPISODES:-30}"
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_REWARD_TRACE=1
      export SPECTRA_SKIP_EVAL=1
      unset SPECTRA_ACTOR_SKIP_OVERBUDGET SPECTRA_TRAIN_RESPECT_FLOOR SPECTRA_SKIP_EVAL_TRAIN
      unset SPECTRA_FILTER_IMPORTANCE SPECTRA_TRAIN_TAU
      case "$PROFILE" in
        offline_train_v3_fpgm_neonraw)
          # The B-menu x original-NEON cell v2 never ran: nominal-rate trichotomy, no cube-root.
          export SPECTRA_REWARD_MODE=neon; export SPECTRA_REWARD_SCALE=raw ;;
        offline_train_v3_fpgm_structraw)
          # Fallback if neonraw collapses like v2c: NEON trichotomy on the realised cut, no cube-root.
          export SPECTRA_REWARD_MODE=structural; export SPECTRA_REWARD_SCALE=raw ;;
        offline_train_v6_inband|offline_train_v6_inband_p5b2|offline_train_v6_inband_p5b2_factored|offline_train_v7_budget|offline_train_v8_grouptoken|offline_train_v9_diverse)
          # Ido 19 Sep GO: cbrt on cubed arms only; in-band stays linear +ρ.
          # Isolated vs live v3-fpgm 21385158 (same 24-net, same PPO, same fpgm menu).
          export SPECTRA_REWARD_MODE=structural; export SPECTRA_REWARD_SCALE=cbrt_cubes
          # G2 one-change reward trains (Ido 1 Oct GO): the arm swaps the scale and nothing else.
          # cbrt_miss = raw +ρ³ on gains; raw = NEON's trichotomy on the realised cut.
          if [[ -n "${SPECTRA_REWARD_SCALE_ARM:-}" ]]; then
            case "$SPECTRA_REWARD_SCALE_ARM" in
              raw|cbrt|cbrt_cubes|cbrt_miss) export SPECTRA_REWARD_SCALE="$SPECTRA_REWARD_SCALE_ARM" ;;
              *) echo "unknown SPECTRA_REWARD_SCALE_ARM=${SPECTRA_REWARD_SCALE_ARM} (raw|cbrt|cbrt_cubes|cbrt_miss)" >&2; exit 1 ;;
            esac
          fi ;;
        *)
          export SPECTRA_REWARD_MODE=structural; export SPECTRA_REWARD_SCALE=cbrt ;;
      esac
      case "$PROFILE" in
        offline_train_v3_svd) V3_RANK=svd ;;
        offline_train_v3_bnscale) V3_RANK=bn_scale ;;
        *) V3_RANK=fpgm ;;
      esac
      if [[ "$PROFILE" == "offline_train_v6_inband_p5b2" || "$PROFILE" == "offline_train_v6_inband_p5b2_factored" || "$PROFILE" == "offline_train_v7_budget" || "$PROFILE" == "offline_train_v8_grouptoken" ]]; then
        # ---- V6 next-cycle thesis actor (Fable 21 Sep, Catalog L lock). In-band linear reward
        # (the only reward whose frozen actor left the 90% keep on r56-w4, ledger §111) on the
        # Catalog-L-clean catalog: 9-net C10 core (VGG-13, not VGG-16) + one SVHN net (P5-B2;
        # CIFAR-100 admitted nothing, §109). Train ∩ {L1, L2, L3, thin, similar, unlike, FMNIST,
        # ImageNet} = ∅ (tests/test_v5_catalog.py). Probes: the thin pair from the v3 default
        # above (VGG-13 + r56-w6 only with SPECTRA_PROBE_SET=v7). Train FT stays
        # 12/4 unless the ft40 TRAJ (21535193) says 40/10 walks differently — then submit with
        # SPECTRA_TRAIN_FT_EPOCHS=40 SPECTRA_TRAIN_FT_PATIENCE=10 and caption it.
        export SPECTRA_DATABASE="${SPECTRA_V6_DATABASE:-${REPO_DIR}/configs/database_offline_v6_p5b2.json}"
        export SPECTRA_DATASET_NAMES="cifar-10 svhn"
        if [[ ! -s "$SPECTRA_DATABASE" ]]; then
          echo "offline_train_v6_inband_p5b2: ${SPECTRA_DATABASE} missing" >&2; exit 1
        fi
        export SPECTRA_FT_REINIT_EDITED=0 SPECTRA_FT_REINIT_THEN_POLISH=0
      fi
      if [[ "$PROFILE" == "offline_train_v9_diverse" ]]; then
        # ---- N8 diverse train (roadmap §5). The Stage-4 recipe of 21737123 pinned here, not at
        # submit: P (val = half of the CIFAR test split, batch 256), crop+flip train FT, area probe
        # score; the in-band linear reward, keep-rate menu and 12/4 train FT come from the blocks
        # above. Three changes against Stage 4: catalog A (8 C10 + the 8 admitted C100 nets, §148),
        # datasets cifar-10 + cifar-100, and r20-w13 C100 as a third probe net so the selection
        # score sees C100. SVHN, Fashion-MNIST and ImageNet stay held out (tests/test_v5_catalog.py).
        # If M1 changes the recipe, change the pin here.
        export SPECTRA_DATABASE="${SPECTRA_V9_DATABASE:-${REPO_DIR}/configs/database_offline_v7_diverse_admitted.json}"
        export SPECTRA_DATASET_NAMES="cifar-10 cifar-100"
        if [[ ! -s "$SPECTRA_DATABASE" ]]; then
          echo "offline_train_v9_diverse: ${SPECTRA_DATABASE} missing" >&2; exit 1
        fi
        # --check-admitted passes a C10-only file (it only rejects unadmitted C100 rows).
        if ! grep -q '"cifar-100"' "$SPECTRA_DATABASE"; then
          echo "offline_train_v9_diverse: no CIFAR-100 rows in ${SPECTRA_DATABASE}; catalog A not deployed" >&2; exit 1
        fi
        "$PYTHON" "${REPO_DIR}/scripts/build_v5_catalog.py" --check-admitted "$SPECTRA_DATABASE" \
          --gate "${REPO_DIR}/configs/v7_c100_gate.json" || exit 1
        export SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_FT_AUG=1 SPECTRA_PROBE_SCORE=area
        export SPECTRA_PROBE_NETS="resnet56-width6,resnet20-width10,resnet20-width13_cifar100"
        export SPECTRA_FT_REINIT_EDITED=0 SPECTRA_FT_REINIT_THEN_POLISH=0
      fi
      if [[ "$PROFILE" == offline_train_v5_* ]]; then
        # ---- V5 (18 Sep, Fable sitting). One cell per profile on the v3-fpgm recipe
        # (5-action fpgm menu, structural+cbrt, 2 passes, group-cost, rewind governor).
        #   offline_train_v5_p5b3      P5-B3 catalog only: rebalanced C10 core U *admitted*
        #                              recoverable C100 (configs/database_offline_v5_p5b3_admitted.json,
        #                              emitted by scripts/build_v5_catalog.py --emit-admitted from the
        #                              recovery-probe gate table). Probes: a C10 non-ResNet and a C100 net.
        #                              FT recipe A. Control = v3-fpgm 21385158 on the 24-net.
        #   offline_train_v5_p5b3_cgp  P5-B3 catalog + P8 C-G+ recipe (only after the no-agent
        #                              recovery probe lands and the group budget is set from it).
        #   offline_train_v5_ft40      Isolated FT A/B: the live v3-fpgm 24-net recipe with the TEST
        #                              recovery budget (40/10) instead of 12/4. Nothing else changes.
        case "$PROFILE" in
          offline_train_v5_ft40)
            export SPECTRA_TRAIN_FT_EPOCHS=40
            export SPECTRA_TRAIN_FT_PATIENCE=10
            ;;
          offline_train_v5_p5b3|offline_train_v5_p5b3_cgp)
            export SPECTRA_DATABASE="${SPECTRA_V5_DATABASE:-${REPO_DIR}/configs/database_offline_v5_p5b3_admitted.json}"
            if [[ ! -s "$SPECTRA_DATABASE" ]]; then
              echo "offline_train_v5_p5b3: ${SPECTRA_DATABASE} is missing or empty." >&2
              echo "Run scripts/build_v5_catalog.py --emit-admitted after the C100 recovery probe (gate table configs/v5_p5b3_c100_gate.json)." >&2
              exit 1
            fi
            # P5-B3 gate: every C100 net in the database must be 'admitted' in the gate table.
            "$PYTHON" "${REPO_DIR}/scripts/build_v5_catalog.py" --check-admitted "$SPECTRA_DATABASE" || exit 1
            # Catalog L lock (21 Sep): VGG-16 C10 is L2 and left the core; the non-ResNet probe
            # VGG-13 needs SPECTRA_PROBE_SET=v7 (the v3 default above already set the thin pair).
            ;;
        esac
        if [[ "$PROFILE" == "offline_train_v5_p5b3_cgp" ]]; then
          export SPECTRA_FT_REINIT_EDITED=1
          export SPECTRA_FT_REINIT_THEN_POLISH=1
          export SPECTRA_REFRESH_ALL_FEATURES=1
          # Group-training budget for policy training: set from the recovery probe's
          # finetune epochs_ran (docs/V5_P8_RECOVERY_PROBE.md). Defaults are placeholders.
          export SPECTRA_FT_REINIT_EPOCHS="${SPECTRA_FT_REINIT_EPOCHS:-24}"
          export SPECTRA_FT_REINIT_PATIENCE="${SPECTRA_FT_REINIT_PATIENCE:-4}"
          export SPECTRA_FT_POLISH_EPOCHS="${SPECTRA_FT_POLISH_EPOCHS:-4}"
          export SPECTRA_FT_POLISH_PATIENCE="${SPECTRA_FT_POLISH_PATIENCE:-2}"
        else
          export SPECTRA_FT_REINIT_EDITED=0
          export SPECTRA_FT_REINIT_THEN_POLISH=0
          export SPECTRA_REFRESH_ALL_FEATURES="${SPECTRA_REFRESH_ALL_FEATURES:-0}"
        fi
      fi
      # Always cold (group-cost changes the token width; no v2 checkpoint can be loaded) —
      # unless SPECTRA_RESUME_TRAIN=1 continues a run of this same profile and tree: then the
      # bundle the prologue copied from SPECTRA_RESUME_PATH stays, and the parent's feature
      # standardizer is reused. The bundle restores weights, optimizers, the episode index and
      # the governor's best probe score and patience counter (PPO saves gov.best_score), so a
      # probe after the resume freezes only if it beats the pre-resume best. Only the rewind
      # count starts again at 0. Trains submit with --no-requeue; without the switch this
      # branch deletes the bundle.
      if [[ "${SPECTRA_RESUME_TRAIN:-0}" == "1" && -s "${SPECTRA_RUN_DIR}/agent_checkpoints/train_resume.pt" ]]; then
        if [[ -n "${SPECTRA_PARENT_RUN:-}" ]]; then
          for _std in "${SPECTRA_PARENT_RUN}/standardizer.pt" "${SPECTRA_PARENT_RUN}/agent_checkpoints/standardizer.pt"; do
            if [[ -s "$_std" ]]; then
              cp -f "$_std" "${SPECTRA_RUN_DIR}/standardizer.pt"
              cp -f "$_std" "${SPECTRA_RUN_DIR}/agent_checkpoints/standardizer.pt"
              echo "resume: standardizer from ${_std}"
              break
            fi
          done
        fi
        echo "resume: keeping ${SPECTRA_RUN_DIR}/agent_checkpoints/train_resume.pt"
        unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH SPECTRA_RESUME_PATH SPECTRA_PARENT_RUN
      else
        unset SPECTRA_ACTOR_CHECKPOINT_PATH SPECTRA_CRITIC_CHECKPOINT_PATH SPECTRA_RESUME_PATH SPECTRA_PARENT_RUN
        rm -f "${SPECTRA_RUN_DIR}/agent_checkpoints/train_resume.pt" \
              "${SPECTRA_RUN_DIR}/agent_checkpoints/latest_best_actor.pt" \
              "${SPECTRA_RUN_DIR}/agent_checkpoints/latest_best_critic.pt"
      fi
      export SPECTRA_CONTINUE_TRAIN=0
      export SPECTRA_SKIP_TRAIN=0
      if [[ "$PROFILE" == "offline_train_v7_budget" ]]; then
        # ---- V7 Budget + STOP (Fable 27 Sep): the action is "remove this fraction of the
        # network's parameters through this group" (0 / 1 / 2 / 4 %) or STOP (-1). Same in-band
        # linear reward, same 10-net Catalog-L-clean catalog, area probe score, recipe A, L1
        # ranking on every cut. New actor (the action list changed). Cost-shaped action after
        # AMC (He et al., ECCV 2018) plus an explicit stop AMC does not have. Control = the
        # area train 21536396 (same catalog/reward/score, rates menu).
        export SPECTRA_ACTION_MENU=budget
        export SPECTRA_PROBE_SCORE="${SPECTRA_PROBE_SCORE:-area}"
        export SPECTRA_STOP_REWARD_SCALE="${SPECTRA_STOP_REWARD_SCALE:-100}"
        EXTRA_ARGS=(--runtime_limit "${SPECTRA_RUNTIME_LIMIT:-518400}" --allowed_acc_reduction 10
                    --train_compressed_layer_only False
                    --compression_rates 1.0 0.01 0.02 0.04 -1
                    --action_rankings l1 l1 l1 l1 l1)
      elif [[ "$PROFILE" == "offline_train_v8_grouptoken" ]]; then
        # ---- V8 representation cell (Fable 28 Sep): the prune unit is the token. One token per
        # coupling group (mean of member layer tokens + 4 structure columns) with a learned
        # feeds / fed-by relation attention bias (src/group_tokens.py). Everything else is the
        # area train 21536396: in-band linear reward, P5-B2 catalog, area probe, fpgm 5-action
        # menu, recipe A, 12/4 train FT. New actor (token width changed; pinned in policy_config).
        export SPECTRA_STATE_TOKENS=groups
        export SPECTRA_PROBE_SCORE="${SPECTRA_PROBE_SCORE:-area}"
        EXTRA_ARGS=(--runtime_limit "${SPECTRA_RUNTIME_LIMIT:-518400}" --allowed_acc_reduction 10
                    --train_compressed_layer_only False
                    --compression_rates 1.0 0.9 0.8 0.9 0.8
                    --action_rankings l1 l1 l1 "$V3_RANK" "$V3_RANK")
      elif [[ "$PROFILE" == offline_train_v4_* || "$PROFILE" == "offline_train_v6_inband_p5b2_factored" ]]; then
        # ---- V4-1: factored (rate x ranking) policy on the v3 recipe. Two heads: rate over
        # {1.0, 0.9, 0.8}, ranking over {l1, fpgm, bn_scale, svd, taylor}; the ranking head is
        # inactive on identity. Grows the criterion menu without a 13-way softmax. Taylor is
        # first-order |w*grad| on one train batch, bound right before each cut (data-dependent).
        # v4_factored_tau6 adds the train-only tau curriculum (SPECTRA_TRAIN_TAU=6).
        export SPECTRA_FACTORED_HEAD=1
        if [[ "$PROFILE" == "offline_train_v4_factored_tau6" ]]; then export SPECTRA_TRAIN_TAU=6; fi
        # Colon-separated so the value survives sbatch --export (no spaces / commas).
        IFS=':' read -r -a V4_MENU <<< "${SPECTRA_V4_RANKING_MENU:-l1:fpgm:bn_scale:svd:taylor}"
        EXTRA_ARGS=(--runtime_limit "${SPECTRA_RUNTIME_LIMIT:-518400}" --allowed_acc_reduction 10
                    --train_compressed_layer_only False
                    --compression_rates 1.0 0.9 0.8
                    --ranking_menu "${V4_MENU[@]}")
      else
        EXTRA_ARGS=(--runtime_limit "${SPECTRA_RUNTIME_LIMIT:-518400}" --allowed_acc_reduction 10
                    --train_compressed_layer_only False
                    --compression_rates 1.0 0.9 0.8 0.9 0.8
                    --action_rankings l1 l1 l1 "$V3_RANK" "$V3_RANK")
      fi
    elif [[ "$PROFILE" == "offline_train_v2b" ]]; then
      EXTRA_ARGS=(--runtime_limit "${SPECTRA_RUNTIME_LIMIT:-518400}" --allowed_acc_reduction 10
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.9 0.8 0.9 0.8
                  --action_rankings l1 l1 l1 fpgm fpgm)
    elif [[ "$PROFILE" == "offline_train_prefer" || "$PROFILE" == "offline_train_unified_full" || "$PROFILE" == "offline_train_prefer_floor" || "$PROFILE" == "offline_train_neon_full" || "$PROFILE" == "offline_train_band_cbrt" || "$PROFILE" == "offline_train_gonce_cold" || "$PROFILE" == offline_train_v2* ]]; then
      # Wall is a fuse. Stop is min_episodes (130) + 100-ep in-budget patience.
      # 36h/48h runtime_limit truncated actor learning. 6d fits a 4090 160-ep run.
      EXTRA_ARGS=(--runtime_limit "${SPECTRA_RUNTIME_LIMIT:-518400}" --allowed_acc_reduction 10
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.9 0.8)
    else
      EXTRA_ARGS=(--runtime_limit 43200 --allowed_acc_reduction 10
                  --train_compressed_layer_only False
                  --compression_rates 1.0 0.9 0.8)
    fi
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
  eval_offline_similar|eval_offline_similar_det|eval_offline_novel)
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
    export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20158274/agent_checkpoints/latest_best_actor.pt}"
    export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20158274/agent_checkpoints/latest_best_critic.pt}"
    if [[ "$PROFILE" == "eval_offline_similar_det" ]]; then
      export SPECTRA_EVAL_DETERMINISTIC=1
    fi
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
  c10_c100_matched_vgg_drl)
    # Same VGG-16 BN, C10 (10-way) + C100 (100-way). Isolates class count vs task
    # heaviness on a family that already recovers. Held-out eval: C100 residuals.
    # Do not mix unrecovered residuals into this train catalog.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_c100_residuals.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_c10_c100_matched_vgg.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
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
                --compression_rates 1.0 0.9 0.8)
    ;;
  eval_c100_spoof_classes)
    # Frozen 10-net C10 actor, C9 catalog, Adam-40. Tokens only: last Linear 100→10.
    # Compare TEST to ledger §21. Does not change the CNN head.
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
    export SPECTRA_SPOOF_NUM_CLASSES="${SPECTRA_SPOOF_NUM_CLASSES:-10}"
    export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20158274/agent_checkpoints/latest_best_actor.pt}"
    export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20158274/agent_checkpoints/latest_best_critic.pt}"
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction "${SPECTRA_ALLOWED_ACC_REDUCTION:-10}"
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  eval_c100_residuals_sgd)
    # Skip-train held-out C100 residuals, SGD-80 MixUp (ledger §31 recipe). Needs actor.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_offline_c100_residuals.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/input_offline_c100_residuals.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-100}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    export SPECTRA_NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-80}"
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_FORTIFY="${SPECTRA_FORTIFY:-1}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_SKIP_TRAIN=1
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_CONTINUE_TRAIN=0
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    export SPECTRA_FT_AUG=1
    export SPECTRA_FT_AUTOAUG=1
    export SPECTRA_FT_OPTIM=sgd
    export SPECTRA_FT_COSINE=1
    export SPECTRA_FT_MIXUP="${SPECTRA_FT_MIXUP:-0.2}"
    export SPECTRA_FT_LABEL_SMOOTH="${SPECTRA_FT_LABEL_SMOOTH:-0.1}"
    export SPECTRA_FINETUNE_PATIENCE="${SPECTRA_FINETUNE_PATIENCE:-20}"
    if [[ -z "${SPECTRA_ACTOR_CHECKPOINT_PATH:-}" || -z "${SPECTRA_CRITIC_CHECKPOINT_PATH:-}" ]]; then
      echo "eval_c100_residuals_sgd requires SPECTRA_ACTOR_CHECKPOINT_PATH and SPECTRA_CRITIC_CHECKPOINT_PATH" >&2
      exit 1
    fi
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction "${SPECTRA_ALLOWED_ACC_REDUCTION:-10}"
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  diag_reward_band)
    # Why C10 trains and C100 does not. compute_reward only credits a cut when
    # -tau <= delta_acc <= 0; outside that band a cut scores -reduction**3, so the
    # argmax policy is "never prune". Catalog crosses dataset x known outcome:
    #   VGG-16 BN C10 (inside tau)   vs  VGG-16 BN C100 (inside tau, ledger §21)
    #   r56-w4 C10    (misses tau)   vs  r20-w16 C100   (misses tau, ledger §21)
    # If empty bands track the OUTCOME rather than the DATASET, the root cause is
    # recoverability under the FT recipe, not the 100-way classifier token.
    # Frozen 10-net actor (job20158274) has 3 actions and encoder dim 48.
    # Extra rates rebuild a 5-way head (dim 52) and fail to load — 20900185.
    # Probe the frozen menu 1.0/0.9/0.8; the fine ladder belongs on TRAIN jobs.
    # Read with: python scripts/reward_band_report.py runs/job<id>
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_reward_band_diag.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/input_reward_band_diag.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10 cifar-100}"
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
    export SPECTRA_REWARD_TRACE=1
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20158274/agent_checkpoints/latest_best_actor.pt}"
    export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20158274/agent_checkpoints/latest_best_critic.pt}"
    EXTRA_ARGS=(--runtime_limit "${SPECTRA_RUNTIME_LIMIT:-72000}"
                --allowed_acc_reduction "${SPECTRA_ALLOWED_ACC_REDUCTION:-10}"
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  c100_recoverable_drl_fine|c100_recoverable_drl_fine_shaped)
    # C100 DRL, recoverable families, with the two levers that can re-open the reward
    # band: a fine rate ladder (so some action lands inside tau) and SGD-80 + aug in
    # the loop (so the cut is recoverable at all). Same held-out residual eval as
    # c100_recoverable_drl, so the two are directly comparable.
    # Named _shaped profile pins structural_shaped in sbatch so --export cannot drop it.
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
    if [[ "$PROFILE" == "c100_recoverable_drl_fine_shaped" ]]; then
      export SPECTRA_REWARD_MODE=structural_shaped
    fi
    export SPECTRA_REWARD_TRACE=1
    export SPECTRA_FT_AUG=1
    export SPECTRA_FT_AUTOAUG=1
    export SPECTRA_FT_OPTIM=sgd
    export SPECTRA_FT_COSINE=1
    export SPECTRA_FT_MIXUP="${SPECTRA_FT_MIXUP:-0.2}"
    export SPECTRA_FT_LABEL_SMOOTH="${SPECTRA_FT_LABEL_SMOOTH:-0.1}"
    EXTRA_ARGS=(--runtime_limit 129600 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates 1.0 0.99 0.98 0.96 0.94 0.90)
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
  eval_offline_c100|eval_offline_c100_det)
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
    export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20158274/agent_checkpoints/latest_best_actor.pt}"
    export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20158274/agent_checkpoints/latest_best_critic.pt}"
    if [[ "$PROFILE" == "eval_offline_c100_det" ]]; then
      export SPECTRA_EVAL_DETERMINISTIC=1
    fi
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
    # Transfer probe, not ImageNet DRL train. submit.sh takes the strongest
    # free SKU at or above rtx_4090 (2080/1080 OOM).
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
  eval_c10_thin|eval_c10_thin_det|eval_c10_thin_fpgm|eval_c10_thin_bnscale|eval_c10_thin_traj|eval_c10_thin_traj_gonce)
    # Plain C10-thin held-out eval (easy r20-w2, hard r56-w4) for one frozen actor:
    # param floor 0.70, no FLOP floor. Look-ahead is on so a 0.8 cut cannot jump
    # past 0.70 to ~0.60 (leaky Path 3 TESTs: set SPECTRA_EVAL_LOOKAHEAD=0).
    # eval_c10_thin_traj: no identity-pad; Phase-A look-ahead labels ~0.70 then
    # continues; quote floor_hold / val_best / terminal. Determinism A/B: submit
    # twice, once with SPECTRA_EVAL_DETERMINISTIC=1.
    # eval_c10_thin_traj_gonce: same frozen actor and TRAJ protocol, plus
    # SPECTRA_GROUP_ONCE_PER_PASS=1 — each residual stream is cut at most once per
    # pass (13 Sep audit: Path 3 walked the r56-w4 stage-2 stream 8->2 on six owning
    # rows). Compare its [eval] TRAJ lines to 21233223 (same actor, flag off).
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
    unset SPECTRA_EVAL_MIN_FLOP_RATIO
    export SPECTRA_SKIP_TRAIN=1
    export SPECTRA_SKIP_STANDARDIZER="${SPECTRA_SKIP_STANDARDIZER:-0}"
    export SPECTRA_CONTINUE_TRAIN=0
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_REWARD_TRACE="${SPECTRA_REWARD_TRACE:-1}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    export SPECTRA_ACTOR_CHECKPOINT_PATH="${SPECTRA_ACTOR_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20158274/agent_checkpoints/latest_best_actor.pt}"
    export SPECTRA_CRITIC_CHECKPOINT_PATH="${SPECTRA_CRITIC_CHECKPOINT_PATH:-${REPO_DIR}/runs/job20158274/agent_checkpoints/latest_best_critic.pt}"
    if [[ "$PROFILE" == "eval_c10_thin_det" ]]; then
      export SPECTRA_EVAL_DETERMINISTIC=1
    fi
    if [[ "$PROFILE" == "eval_c10_thin_fpgm" ]]; then
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_FILTER_IMPORTANCE=fpgm
    fi
    if [[ "$PROFILE" == "eval_c10_thin_bnscale" ]]; then
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_FILTER_IMPORTANCE=bn_scale
    fi
    if [[ "$PROFILE" == "eval_c10_thin_traj" || "$PROFILE" == "eval_c10_thin_traj_gonce" ]]; then
      export SPECTRA_EVAL_DETERMINISTIC=1
      export SPECTRA_EVAL_TRAJECTORY=1
      export SPECTRA_EVAL_LOOKAHEAD=0
      export SPECTRA_SKIP_EVAL_TRAIN=1
    fi
    if [[ "$PROFILE" == "eval_c10_thin_traj_gonce" ]]; then
      export SPECTRA_GROUP_ONCE_PER_PASS=1
    fi
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction "${SPECTRA_ALLOWED_ACC_REDUCTION:-10}"
                --train_compressed_layer_only False
                --compression_rates 1.0 0.9 0.8)
    ;;
  baseline_c10_l1|baseline_c10_mild|baseline_c10_random|baseline_c10_mild_traj|baseline_c10_mild_traj_gonce|baseline_c10_l1_traj|baseline_c10_l1_traj_gonce|baseline_c10_mildest95_traj_gonce)
    # Same-loop structured baselines vs DRL: same rates, FT, fortify, floor 0.70.
    # Env still L1-ranks filters; these policies only pick the compression rate.
    # *_traj: unconstrained TRAJ protocol (quote [eval] TRAJ). *_gonce: plus
    # SPECTRA_GROUP_ONCE_PER_PASS=1. Path 3 argmax on the thin nets is "0.9 on every
    # legal row" (20945568 step records), i.e. the mild picker: baseline_c10_mild_traj
    # is the same-loop control for eval_c10_thin_traj, and the _gonce pair isolates
    # the group-once lever from the actor.
    export SPECTRA_INPUT="${SPECTRA_INPUT:-${REPO_DIR}/configs/input_c10_thin.json}"
    export SPECTRA_DATABASE="${SPECTRA_DATABASE:-${REPO_DIR}/configs/database_c10_thin.json}"
    export SPECTRA_DATASET_NAMES="${SPECTRA_DATASET_NAMES:-cifar-10}"
    export SPECTRA_PASSES=1
    export SPECTRA_ROLLOUT_LIMIT=5
    export SPECTRA_N_SPLITS=0
    # 40 is the TEST recovery budget. The P5-B3 C100 gate probe replays the same walk under the
    # *train* budget (SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4); everything else keeps 40.
    export SPECTRA_NUM_EPOCHS="${SPECTRA_NUM_EPOCHS:-40}"
    export SPECTRA_SAVE_PRUNED=False
    export SPECTRA_FORTIFY="${SPECTRA_FORTIFY:-1}"
    export SPECTRA_EVAL_MIN_PARAM_RATIO="${SPECTRA_EVAL_MIN_PARAM_RATIO:-0.70}"
    export SPECTRA_SKIP_TRAIN=1
    export SPECTRA_SKIP_STANDARDIZER=1
    export SPECTRA_CONTINUE_TRAIN=0
    export SPECTRA_REWARD_MODE="${SPECTRA_REWARD_MODE:-neon}"
    export SPECTRA_STATE_ENCODER="${SPECTRA_STATE_ENCODER:-transformer}"
    BASELINE_RATES=(1.0 0.9 0.8)
    case "$PROFILE" in
      baseline_c10_mild|baseline_c10_mild_traj|baseline_c10_mild_traj_gonce) export SPECTRA_EVAL_POLICY=mild ;;
      baseline_c10_mildest95_traj_gonce)
        # V9 fine menu: weakest legal cut. 0.95 is finer than 0.9 only on groups >= 16 wide;
        # dedupe masks it below that, where the walk is the mild walk.
        export SPECTRA_EVAL_POLICY=mildest
        export SPECTRA_ACTION_DEDUPE=1
        BASELINE_RATES=(1.0 0.95 0.9 0.8) ;;
      baseline_c10_random) export SPECTRA_EVAL_POLICY=random ;;
      *) export SPECTRA_EVAL_POLICY=l1 ;;
    esac
    if [[ "$PROFILE" == *_traj || "$PROFILE" == *_traj_gonce ]]; then
      export SPECTRA_EVAL_TRAJECTORY=1
      export SPECTRA_EVAL_LOOKAHEAD=0
      export SPECTRA_SKIP_EVAL_TRAIN=1
    fi
    if [[ "$PROFILE" == *_gonce ]]; then
      export SPECTRA_GROUP_ONCE_PER_PASS=1
    fi
    EXTRA_ARGS=(--runtime_limit 7200 --allowed_acc_reduction 10
                --train_compressed_layer_only False
                --compression_rates "${BASELINE_RATES[@]}")
    ;;
  *)
    echo "Unknown SPECTRA_PROFILE='${PROFILE}'" >&2
    exit 1
    ;;
esac

# P8 recovery recipe for eval / heuristic-control profiles (SPECTRA_FT_RECIPE=a|cg|cgp).
# The no-agent recovery probe (docs/V5_P8_RECOVERY_PROBE.md) submits the *same* baseline
# TRAJ profile three times with a different recipe; a frozen NEON-C actor pins its recipe
# through policy_config.json instead. Train profiles set the flags themselves above.
case "${SPECTRA_FT_RECIPE:-}" in
  cg)
    export SPECTRA_FT_REINIT_EDITED=1 SPECTRA_FT_REINIT_THEN_POLISH=0 SPECTRA_REFRESH_ALL_FEATURES=1 ;;
  cgp)
    export SPECTRA_FT_REINIT_EDITED=1 SPECTRA_FT_REINIT_THEN_POLISH=1 SPECTRA_REFRESH_ALL_FEATURES=1 ;;
  alsq)
    # V7 A-LSQ: keep-leftover cut, least-squares refit of every consumer kernel to its pre-cut
    # output (He, Zhang, Sun, ICCV 2017), BatchNorm running stats re-estimated, then recipe A's
    # full-net fine-tune. The comparison with recipe A is the initialisation only.
    export SPECTRA_FT_REINIT_EDITED=0 SPECTRA_FT_REINIT_THEN_POLISH=0
    export SPECTRA_FT_LSQ_CONSUMERS=1 SPECTRA_FT_BN_RECAL=1 ;;
  pca)
    # V7 C-PCA: new producer filters = top principal directions of the pre-cut output, consumers
    # premultiplied by that basis, group norms reset, BN recalibrated, recipe A fine-tune.
    export SPECTRA_FT_REINIT_EDITED=pca SPECTRA_FT_REINIT_THEN_POLISH=0
    export SPECTRA_FT_LSQ_CONSUMERS=0 SPECTRA_FT_BN_RECAL=1 ;;
  a)
    export SPECTRA_FT_REINIT_EDITED=0 SPECTRA_FT_REINIT_THEN_POLISH=0 ;;
  "") : ;;
  *)
    echo "Unknown SPECTRA_FT_RECIPE='${SPECTRA_FT_RECIPE}' (use a | cg | cgp | alsq | pca)" >&2
    exit 1 ;;
esac
if [[ "${SPECTRA_FT_REINIT_EDITED:-0}" == "1" ]]; then
  # TEST-scale group budget: NEON trained the new layer until convergence (num_epoch 100,
  # patience 10 on the train loss). 60 / 6 on val is the eval default; polish 8 / 3 at 0.1x lr.
  export SPECTRA_FT_REINIT_EPOCHS="${SPECTRA_FT_REINIT_EPOCHS:-60}"
  export SPECTRA_FT_REINIT_PATIENCE="${SPECTRA_FT_REINIT_PATIENCE:-6}"
  export SPECTRA_FT_REINIT_SELECT="${SPECTRA_FT_REINIT_SELECT:-val}"
  # group = NEON-source-literal (producers + consumer slices + norms); producers = Gilad's oral wording.
  export SPECTRA_FT_REINIT_SCOPE="${SPECTRA_FT_REINIT_SCOPE:-group}"
  export SPECTRA_FT_POLISH_EPOCHS="${SPECTRA_FT_POLISH_EPOCHS:-8}"
  export SPECTRA_FT_POLISH_PATIENCE="${SPECTRA_FT_POLISH_PATIENCE:-3}"
  export SPECTRA_FT_POLISH_LR_MULT="${SPECTRA_FT_POLISH_LR_MULT:-0.1}"
fi

# Eval-only jobs: skip the duplicated train-loader prune+FT walk. Paper quotes
# eval_test only. Train profiles leave this unset (default off in code). Override
# with SPECTRA_SKIP_EVAL_TRAIN=0 on an eval profile if a walk must keep both.
if [[ "${SPECTRA_SKIP_TRAIN:-0}" == "1" ]]; then
  export SPECTRA_SKIP_EVAL_TRAIN="${SPECTRA_SKIP_EVAL_TRAIN:-1}"
fi
# Eval-only jobs: SPECTRA_EVAL_PASSES overrides the profile's --passes (v3 actors train with
# 2 passes and their policy_config replays 2; heuristic controls need the same walk length).
if [[ "${SPECTRA_SKIP_TRAIN:-0}" == "1" && -n "${SPECTRA_EVAL_PASSES:-}" ]]; then
  export SPECTRA_PASSES="${SPECTRA_EVAL_PASSES}"
fi

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
