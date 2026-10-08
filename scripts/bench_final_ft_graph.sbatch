#!/usr/bin/env bash
# Final fine-tune CUDA-graph speed bench (scripts/bench_final_ft_graph.py; queue "FT cost", wave 28 G0). Speed only.
#   sbatch --gpus=rtx_4090:1 scripts/bench_final_ft_graph.sbatch <stem> ...
#SBATCH --job-name=bench-ft-graph
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=0-01:30:00

set -uo pipefail
REPO_DIR="${SPECTRA_REPO_DIR:-$PWD}"
PYTHON="${SPECTRA_PYTHON:-$HOME/.conda/envs/spectra/bin/python}"
cd "$REPO_DIR" || exit 1
OUT="${BENCH_OUT:-runs/bench_ft_graph/bench_${SLURM_JOB_ID}.jsonl}"
mkdir -p "$(dirname "$OUT")"
echo "bench-ft-graph ${SLURM_JOB_ID} on $(hostname) at $(date -Is): $*"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv
"$PYTHON" scripts/bench_final_ft_graph.py "$@" ${BENCH_ARGS:-} --out "$OUT"
STATUS=$?
echo "bench-ft-graph finished with status ${STATUS} at $(date -Is); rows in ${OUT}"
exit $STATUS
