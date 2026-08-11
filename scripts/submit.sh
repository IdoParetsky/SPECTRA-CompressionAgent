#!/usr/bin/env bash
# Submit a SPECTRA batch job and print everything needed to follow it.
#
#   bash scripts/submit.sh smoke            # 1 GPU, 1h  - pipeline correctness
#   bash scripts/submit.sh medium           # 1 GPU, 10h - small DB (7h train + eval buffer)
#   bash scripts/submit.sh full             # 2 GPUs, 15h - the real experiment
#   bash scripts/submit.sh probe            # 1 GPU, 3h  - recovery matrix (no RL)
#   bash scripts/submit.sh recover          # 1 GPU, 6h  - full FT + mild rates (probe lead)
#   bash scripts/submit.sh recover_groupft  # 1 GPU, 6h  - group-aware freeze + mild rates
#   bash scripts/submit.sh full 1           # ... forced onto 1 GPU
#
# Wall clocks are intentionally longer than --runtime_limit inside spectra.sbatch: the agent
# soft-stops training at runtime_limit, then still needs wall time for eval + summarize_run.
# Matching the two (as we did for medium=8h/8h and diag=2.5h with 2h train) causes SLURM
# TIMEOUT mid-eval — not a library crash.
#
# Resource choice per profile is deliberate: debugging on two GPUs adds an NCCL failure mode
# on top of whatever bug is being chased, so only `full` requests two.

set -euo pipefail

PROFILE="${1:-smoke}"
GPU_COUNT="${2:-}"
REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"

case "$PROFILE" in
  # HPC: do not request >80G with an rtx_6000 (IT policy). Our CIFAR thin-ResNet
  # jobs peak around ~3G host RSS / ~2G GPU; 64–80G is already a large cushion.
  smoke)  GPUS="${GPU_COUNT:-1}"; TIME="0-01:00:00"; MEM="64G"  ;;
  medium) GPUS="${GPU_COUNT:-1}"; TIME="0-10:00:00"; MEM="80G"  ;;
  full)   GPUS="${GPU_COUNT:-2}"; TIME="0-15:00:00"; MEM="80G"  ;;
  probe)  GPUS="${GPU_COUNT:-1}"; TIME="0-03:00:00"; MEM="64G"  ;;
  probe_continue) GPUS="${GPU_COUNT:-1}"; TIME="0-02:00:00"; MEM="64G"  ;;
  diag)   GPUS="${GPU_COUNT:-1}"; TIME="0-04:00:00"; MEM="64G"  ;;
  recover) GPUS="${GPU_COUNT:-1}"; TIME="0-06:00:00"; MEM="64G"  ;;
  recover_groupft) GPUS="${GPU_COUNT:-1}"; TIME="0-06:00:00"; MEM="64G"  ;;
  *) echo "usage: $0 {smoke|medium|full|probe|probe_continue|diag|recover|recover_groupft} [gpu_count]" >&2; exit 1 ;;
esac

cd "$REPO_DIR"
mkdir -p runs/slurm_logs

JOB_ID=$(sbatch --parsable \
  --gpus="rtx_6000:${GPUS}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --job-name="spectra-${PROFILE}" \
  --export=ALL,SPECTRA_PROFILE="${PROFILE}" \
  scripts/spectra.sbatch)

LOG="${REPO_DIR}/runs/slurm_logs/spectra_${JOB_ID}.out"
echo "submitted job ${JOB_ID} (profile=${PROFILE}, gpus=${GPUS}, mem=${MEM}, time=${TIME})"
echo "log     : ${LOG}"
echo "run dir : ${REPO_DIR}/runs/job${JOB_ID}"
echo "follow  : tail -f ${LOG}"
echo "status  : squeue -j ${JOB_ID}"
echo "cancel  : scancel ${JOB_ID}"
echo "${JOB_ID}"
