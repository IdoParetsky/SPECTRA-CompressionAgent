#!/usr/bin/env bash
# One-shot status of a SPECTRA job: queue state, log tail, recorded errors, event counts.
# Designed to be the single command an unattended monitoring loop runs.
#
#   bash scripts/job_status.sh <job_id> [log_tail_lines]

set -uo pipefail

JOB_ID="${1:?usage: job_status.sh <job_id> [tail_lines]}"
TAIL_LINES="${2:-40}"
REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"
LOG="${REPO_DIR}/runs/slurm_logs/spectra_${JOB_ID}.out"
RUN_DIR="${REPO_DIR}/runs/job${JOB_ID}"

echo "=== queue ==="
squeue -j "$JOB_ID" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null | tail -n +1
if ! squeue -j "$JOB_ID" >/dev/null 2>&1 || [ -z "$(squeue -h -j "$JOB_ID" 2>/dev/null)" ]; then
  echo "(not in queue - finished or never started)"
  sacct -j "$JOB_ID" --format=JobID,JobName%20,State,ExitCode,Elapsed,MaxRSS -n 2>/dev/null | head -5
fi

echo
echo "=== slurm log tail (${TAIL_LINES}) ==="
[ -f "$LOG" ] && tail -n "$TAIL_LINES" "$LOG" || echo "(no log at $LOG)"

echo
echo "=== recorded errors ==="
if [ -f "${RUN_DIR}/logs/errors.log" ]; then
  wc -l < "${RUN_DIR}/logs/errors.log" | xargs echo "lines:"
  tail -n 30 "${RUN_DIR}/logs/errors.log"
else
  echo "(none)"
fi

echo
echo "=== event counts ==="
if [ -d "${RUN_DIR}/events" ]; then
  cat "${RUN_DIR}"/events/rank*.jsonl 2>/dev/null \
    | sed -n 's/.*"event": *"\([a-z_]*\)".*/\1/p' | sort | uniq -c | sort -rn
else
  echo "(no events yet at ${RUN_DIR}/events)"
fi
