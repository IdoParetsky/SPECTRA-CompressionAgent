#!/usr/bin/env bash
# Deployment benchmark of saved TRAJ candidates (scripts/bench_deploy.py), BENCH_REPEATS fresh processes.
#   sbatch --gpus=rtx_4090:1 scripts/bench_deploy.sbatch <run dir> ...
#SBATCH --job-name=bench-deploy
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=0-03:00:00

set -uo pipefail
REPO_DIR="${SPECTRA_REPO_DIR:-$PWD}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
cd "$REPO_DIR" || exit 1
OUT="${BENCH_OUT:-runs/bench_deploy/bench_${SLURM_JOB_ID}}"
mkdir -p "$(dirname "$OUT")"
echo "bench-deploy ${SLURM_JOB_ID} on $(hostname) at $(date -Is): $*"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv
STATUS=0
for r in $(seq 1 "${BENCH_REPEATS:-3}"); do
  "$PYTHON" scripts/bench_deploy.py "$@" ${BENCH_ARGS:-} --out "${OUT}_r${r}.jsonl" || STATUS=$?
done
echo "bench-deploy finished with status ${STATUS} at $(date -Is); rows in ${OUT}_r*.jsonl"
exit $STATUS
