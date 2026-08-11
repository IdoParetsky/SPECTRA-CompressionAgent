#!/usr/bin/env bash
# Allocate an interactive GPU session on BGU SLURM (run on login node: bgu-slurm).
# Edit GPUs / time / mem below as needed.
set -euo pipefail

# Defaults match a typical Cursor experiment session; override e.g. GPUS=rtx_6000:2
GPUS="${GPUS:-1}"
TIME="${TIME:-0-3:00:00}"
MEM="${MEM:-120}"

echo "Requesting: sinteractive --gpu ${GPUS} --time ${TIME} --mem ${MEM}"
echo "After allocation: update local ~/.ssh/config Host bgu-slurm-gpu HostName to this node."
echo "From Windows Cursor:  .\\scripts\\slurm_sync.ps1  then  .\\scripts\\slurm_run.ps1"
echo "Or on the node:       bash scripts/check_gpu.sh && bash scripts/run_agent.sh"
echo "If torch sees no CUDA devices: scancel \$SLURM_JOB_ID  and re-run this script."

exec sinteractive --gpu "${GPUS}" --time "${TIME}" --mem "${MEM}"
