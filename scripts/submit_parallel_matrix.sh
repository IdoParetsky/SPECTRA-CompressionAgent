#!/usr/bin/env bash
# Submit today's parallel SPECTRA matrix entirely on the cluster (no laptop watcher).
#
# Parallel now:
#   - recover_warm_king_fortify   (warm-start from job20063793 best on 6-net DB)
#   - recover_careful_fortify_ft80 (longer FT + fortify + eval param budget)
#
# Cluster-chained (after currently running careful A/B if present):
#   - after careful_fortify (20066522 by default, override CAREFUL_FORTIFY_JOB):
#       another warm-start from THAT job's latest_best_* once it finishes
#
# Usage (on bgu-slurm login or any host with sbatch to the cluster):
#   bash scripts/submit_parallel_matrix.sh
#   CAREFUL_FORTIFY_JOB=20066522 bash scripts/submit_parallel_matrix.sh
set -euo pipefail

REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
cd "$REPO_DIR"
mkdir -p runs/slurm_logs
STAMP="$(date -Is)"
LOG="runs/slurm_logs/parallel_matrix_${STAMP//[:]//_}.txt"

{
  echo "=== SPECTRA parallel matrix @ ${STAMP} ==="
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

  echo "--- warm-start from king_fortify ---"
  WARM=$(bash scripts/submit.sh recover_warm_king_fortify | tee /dev/stderr | tail -1)

  echo "--- careful_fortify with 80 FT epochs ---"
  FT80=$(bash scripts/submit.sh recover_careful_fortify_ft80 | tee /dev/stderr | tail -1)

  CAREFUL_FORTIFY_JOB="${CAREFUL_FORTIFY_JOB:-20066522}"
  if sacct -j "${CAREFUL_FORTIFY_JOB}" -n -X -o State --parsable2 2>/dev/null | head -1 | grep -Eq 'RUNNING|PENDING|COMPLETING|COMPLETED'; then
    echo "--- chain: afterok:${CAREFUL_FORTIFY_JOB} -> warm-start from that run's best ---"
    # Child overrides checkpoint paths to the parent run dir (known once job id is known).
    export SPECTRA_ACTOR_CHECKPOINT_PATH="${REPO_DIR}/runs/job${CAREFUL_FORTIFY_JOB}/agent_checkpoints/latest_best_actor.pt"
    export SPECTRA_CRITIC_CHECKPOINT_PATH="${REPO_DIR}/runs/job${CAREFUL_FORTIFY_JOB}/agent_checkpoints/latest_best_critic.pt"
    export SPECTRA_CONTINUE_TRAIN=1
    CHAIN=$(SPECTRA_DEPENDENCY="afterok:${CAREFUL_FORTIFY_JOB}" \
      bash scripts/submit.sh recover_warm_king_fortify | tee /dev/stderr | tail -1)
  else
    echo "NOTE: CAREFUL_FORTIFY_JOB=${CAREFUL_FORTIFY_JOB} not found; skipping chain"
    CHAIN="none"
  fi

  echo "=== submitted ==="
  echo "warm_king_fortify=$WARM"
  echo "careful_fortify_ft80=$FT80"
  echo "chained_after_${CAREFUL_FORTIFY_JOB}=$CHAIN"
  squeue -u "$USER"
} | tee "$LOG"

echo "matrix log: $REPO_DIR/$LOG"
