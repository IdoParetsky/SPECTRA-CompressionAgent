#!/usr/bin/env bash
# Submit a SPECTRA batch job and print everything needed to follow it.
#
#   bash scripts/submit.sh smoke            # pipeline correctness
#   bash scripts/submit.sh medium           # small DB agent
#   bash scripts/submit.sh full             # thesis-scale
#   bash scripts/submit.sh probe            # recovery matrix (no RL)
#   bash scripts/submit.sh probe_groupft    # recovery matrix under group-aware freeze
#   bash scripts/submit.sh recover          # full FT + mild rates
#   bash scripts/submit.sh recover_groupft  # group-aware freeze + mild rates
#   bash scripts/submit.sh recover_wide     # full FT + rates incl. 0.7
#   bash scripts/submit.sh recover_pref10   # full FT + mild rates, -10 pp preference
#
# Wall clocks MUST exceed the profile's --runtime_limit (agent soft-stop) so eval and
# summarize_run can finish. Matching the two caused SLURM TIMEOUT mid-eval (not a
# library crash). This script refuses to submit if wall_sec <= train_sec.
#
# HPC notes that have bitten us:
#   - rtx_6000: request <=80G RAM (IT policy)
#   - gpu partition MaxMemPerCPU=16G -> for 80G use >=5 CPUs
#   - scontrol update TimeLimit is denied for users; cancel+resubmit instead
#   - #SBATCH --signal=USR1@900 soft-stops the agent before the hard kill
#   - MaxTime on gpu is 7 days; keep walls well under that

set -euo pipefail

PROFILE="${1:-smoke}"
GPU_COUNT="${2:-}"
REPO_DIR="${SPECTRA_REPO_DIR:-/home/paretsky/SPECTRA-CompressionAgent}"

# train_sec = --runtime_limit inside spectra.sbatch for agent profiles (0 for probe suites).
# wall must be strictly greater; keep >=90 min buffer for eval when possible.
case "$PROFILE" in
  smoke)
    GPUS="${GPU_COUNT:-1}"; TIME="0-01:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=900 ;;
  medium)
    GPUS="${GPU_COUNT:-1}"; TIME="0-10:00:00"; MEM="80G"; CPUS=6; TRAIN_SEC=25200 ;;
  full)
    GPUS="${GPU_COUNT:-2}"; TIME="0-15:00:00"; MEM="80G"; CPUS=6; TRAIN_SEC=46800 ;;
  probe)
    GPUS="${GPU_COUNT:-1}"; TIME="0-03:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=0 ;;
  probe_continue)
    GPUS="${GPU_COUNT:-1}"; TIME="0-02:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=0 ;;
  probe_groupft)
    GPUS="${GPU_COUNT:-1}"; TIME="0-02:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=0 ;;
  diag)
    GPUS="${GPU_COUNT:-1}"; TIME="0-04:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=7200 ;;
  recover)
    GPUS="${GPU_COUNT:-1}"; TIME="0-06:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=14400 ;;
  recover_groupft)
    GPUS="${GPU_COUNT:-1}"; TIME="0-06:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=14400 ;;
  recover_wide)
    GPUS="${GPU_COUNT:-1}"; TIME="0-06:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=14400 ;;
  recover_pref10)
    GPUS="${GPU_COUNT:-1}"; TIME="0-06:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=14400 ;;
  recover_king)
    # Compose today's winners: mild rates + -10 pp preference + full FT + 40 epochs.
    GPUS="${GPU_COUNT:-1}"; TIME="0-08:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=18000 ;;
  recover_careful)
    # AMC-scale warmup + 6-net DB + standardizer; wall >> train for eval buffer.
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; MEM="80G"; CPUS=6; TRAIN_SEC=36000 ;;
  recover_careful_fortify)
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; MEM="80G"; CPUS=6; TRAIN_SEC=36000 ;;
  recover_king_fortify)
    GPUS="${GPU_COUNT:-1}"; TIME="0-08:00:00"; MEM="64G"; CPUS=4; TRAIN_SEC=18000 ;;
  recover_warm_king_fortify)
    # Warm-start 6-net careful-fortify from king_fortify best actor/critic.
    GPUS="${GPU_COUNT:-1}"; TIME="0-12:00:00"; MEM="80G"; CPUS=6; TRAIN_SEC=36000 ;;
  recover_careful_fortify_ft80)
    # Same as careful_fortify but 80 FT epochs (NEON-closer recovery budget).
    GPUS="${GPU_COUNT:-1}"; TIME="0-14:00:00"; MEM="80G"; CPUS=6; TRAIN_SEC=43200 ;;
  *)
    echo "usage: $0 {smoke|medium|full|probe|probe_continue|probe_groupft|diag|recover|recover_groupft|recover_wide|recover_pref10|recover_king|recover_careful|recover_careful_fortify|recover_king_fortify|recover_warm_king_fortify|recover_careful_fortify_ft80} [gpu_count]" >&2
    exit 1
    ;;
esac

# Parse TIME (D-HH:MM:SS or HH:MM:SS) into seconds for the guardrail.
_wall_to_sec() {
  local t="$1" days=0 rest h m s
  if [[ "$t" == *-* ]]; then
    days="${t%%-*}"
    rest="${t#*-}"
  else
    rest="$t"
  fi
  IFS=: read -r h m s <<<"$rest"
  echo $(( days*86400 + 10#$h*3600 + 10#$m*60 + 10#$s ))
}

WALL_SEC="$(_wall_to_sec "$TIME")"
if (( TRAIN_SEC > 0 && WALL_SEC <= TRAIN_SEC )); then
  echo "REFUSING submit: wall ${TIME} (${WALL_SEC}s) <= train runtime_limit ${TRAIN_SEC}s" >&2
  echo "Eval would be SLURM-killed. Raise TIME or lower runtime_limit." >&2
  exit 2
fi
if (( TRAIN_SEC > 0 && WALL_SEC - TRAIN_SEC < 1800 )); then
  echo "WARNING: only $(( WALL_SEC - TRAIN_SEC ))s wall buffer after train stop; prefer >=90min." >&2
fi

cd "$REPO_DIR"
mkdir -p runs/slurm_logs

# Prefer non-preemptible rtx_6000-class nodes. ee-l40s-* preempted recover_pref10/wide
# mid-run and wiped progress (no mid-train resume yet).
EXCLUDE_NODES="${SPECTRA_EXCLUDE_NODES:-ee-l40s-01,ee-l40s-02}"

# Cluster-side chaining: SPECTRA_DEPENDENCY=afterok:JOBID (or afterany:JOBID).
# Survives laptop/VPN disconnects — do not rely on a local PowerShell watcher.
SBATCH_EXTRA=()
if [[ -n "${SPECTRA_DEPENDENCY:-}" ]]; then
  SBATCH_EXTRA+=(--dependency="${SPECTRA_DEPENDENCY}")
fi

JOB_ID=$(sbatch --parsable \
  --gpus="rtx_6000:${GPUS}" \
  --mem="${MEM}" \
  --cpus-per-task="${CPUS}" \
  --time="${TIME}" \
  --exclude="${EXCLUDE_NODES}" \
  --job-name="spectra-${PROFILE}" \
  --export=ALL,SPECTRA_PROFILE="${PROFILE}" \
  "${SBATCH_EXTRA[@]}" \
  scripts/spectra.sbatch)

LOG="${REPO_DIR}/runs/slurm_logs/spectra_${JOB_ID}.out"
echo "submitted job ${JOB_ID} (profile=${PROFILE}, gpus=${GPUS}, cpus=${CPUS}, mem=${MEM}, time=${TIME}, train_limit=${TRAIN_SEC}s, exclude=${EXCLUDE_NODES}${SPECTRA_DEPENDENCY:+, dependency=${SPECTRA_DEPENDENCY}})"
echo "log     : ${LOG}"
echo "run dir : ${REPO_DIR}/runs/job${JOB_ID}"
# Confirm the scheduler accepted the Timelimit we asked for (qos/partition can silently clamp).
sleep 1
TL=$(squeue -j "${JOB_ID}" -h -o "%l" 2>/dev/null || sacct -j "${JOB_ID}" -n -X -o Timelimit --parsable2 2>/dev/null | head -1 || true)
echo "Timelimit (scheduler): ${TL:-unknown}"
echo "follow  : tail -f ${LOG}"
echo "status  : squeue -j ${JOB_ID}"
echo "cancel  : scancel ${JOB_ID}"
echo "${JOB_ID}"
