#!/usr/bin/env bash
# Ops heartbeat for the V9b/V9c cells + the Stage-4 train and its resume chain. Opus 5.5 sitting, 30 Sep.
# Run: powershell -NoProfile -File scripts/rexec.ps1 -Quiet -File scripts/_tmp_s30_ops_hb.sh
# TRAIN = the running v9c-paug-area-train* job, else the newest one with a log; override with TRAIN=<id>.
# Only side effect: one copy per day of the train's agent_checkpoints/ under ~/spectra_backups (last 3 kept),
# because a requeue of the same job id deletes its own train_resume.pt (spectra.sbatch "always cold").
B=/home/paretsky/scratch_audit/tree_v9b/runs
C=/home/paretsky/scratch_audit/tree_v9c/runs
PY=/home/paretsky/.conda/envs/spectra/bin/python
RD=/home/paretsky/scratch_audit/readers_s30/scripts
run() { for d in "$C/job$1" "$B/job$1"; do [[ -f $d/logs/rank0.log ]] && { echo "$d"; return; }; done; }
echo "=== $(date '+%d %b %H:%M') queue"
squeue -u paretsky -h -o "%.10i %.24j %.2t %.10M %.10l %.5y %N %R" 2>/dev/null | grep -vE "c10-(similar|unlike)"
sacct -u paretsky -S "$(date -d '-1 day' +%Y-%m-%dT%H:%M)" -X -n -s CD,F,CA,TO,NF,OOM,PR \
  -o JobID%10,JobName%24,State%12,ExitCode%6,Elapsed%11,End%19,NodeList%14 2>/dev/null

echo "=== Stage-4 train chain"
chain=$(sacct -u paretsky -S 2026-09-30T00:00 -X -n -o JobID%10,JobName%30,State%12 2>/dev/null \
  | awk '$2 ~ /^v9c-paug-area-train/' | sort -k1,1nr)
echo "$chain"
if [[ -z "${TRAIN:-}" ]]; then
  TRAIN=$(echo "$chain" | awk '$3 == "RUNNING" {print $1; exit}')
  [[ -z "$TRAIN" ]] && for t in $(echo "$chain" | awk '{print $1}'); do
    [[ -s $C/job$t/logs/rank0.log ]] && { TRAIN=$t; break; }; done
fi
echo "TRAIN=${TRAIN:-none} $(squeue -h -j ${TRAIN:-0} -o '%j %T %M %N' 2>/dev/null) $(scontrol show job ${TRAIN:-0} 2>/dev/null | grep -oE 'Requeue=[0-9]' )"
L=$C/job${TRAIN:-0}/logs/rank0.log
if [[ -s $L ]]; then
  grep -m1 -oE "'SPECTRA_VAL_FROM_TEST': '[^']*'|'SPECTRA_FT_AUG': '[^']*'" "$L" | tr '\n' ' '
  grep -m1 -oE "'SPECTRA_BATCH_SIZE': '[^']*'" "$L"; grep -m1 -oE "'SPECTRA_PROBE_SCORE': '[^']*'" "$L"
  grep -E "Val from test|FT aug on" "$L" | sed -E 's/^.*\| (Val|FT)/\1/' | sort -u | head -4
  grep -m1 -E "GPUs:" "$L" | sed -E 's/^.*\| //'
  grep -m3 -E "resume:|copied resume|seeded latest|Resumed|resumed_episode" "$L" | sed -E 's/^.*\| //' | cut -c1-170
  grep -E "PPO update|PROBE ep|Snapshot frozen|REWIND|Stopping PPO" "$L" \
    | sed -E 's/^.*\| (PPO|PROBE|Snapshot|REWIND|Stopping)/\1/' | cut -c1-170 | tail -6
  grep -E "best_score=.*since_improvement" "$L" | tail -1 | sed -E 's/^(.{19}).*(best_score=[^,]+, since_improvement=[^ ]+).*/\1 \2/'
  echo "new episodes in this job=$(grep -c 'DONE Episode' "$L") TB=$(grep -c Traceback "$L")"
  grep -E "DONE Episode" "$L" | tail -1 | sed -E 's/^.*(DONE Episode)/\1/' | cut -c1-170
  grep -oE 'DONE Episode [0-9]+ in [0-9.]+s' "$L" | awk '{print $5}' | tr -d s | sort -n \
    | awk '{a[NR]=$1; s+=$1} END {if (NR) printf "s/episode median=%.0f mean=%.0f (30 Sep first 12: 1296 / 2315)\n", a[int((NR+1)/2)], s/NR}'
  BK=/home/paretsky/spectra_backups/job${TRAIN}_$(date +%Y%m%d)
  if [[ ! -d $BK && -s $C/job$TRAIN/agent_checkpoints/train_resume.pt ]]; then
    mkdir -p "$BK" && cp -a "$C/job$TRAIN/agent_checkpoints/." "$BK/" && echo "backup -> $BK"
  fi
  ls -d /home/paretsky/spectra_backups/job${TRAIN}_20[0-9][0-9][0-1][0-9][0-3][0-9] 2>/dev/null | sort | head -n -3 | xargs -r rm -rf
else
  echo "no train log yet"
fi
echo "snapshots: $(ls -d $C/job21737123/snapshots/ep* $C/job21767188/snapshots/ep* 2>/dev/null | sed -E 's#.*/job([0-9]+)/snapshots/#\1/#' | tr '\n' ' ')"

echo "=== TRAJ rows (last 6 per cell)"
for j in 21729557 21730501 21809595 21730507 21730509 21730514 21730516 21737104 21737105 \
         21767189 21767190 21767192 21729558 21814029; do
  d=$(run $j); [[ -n "$d" ]] || continue
  f=$d/logs/rank0.log
  echo "--- $j $(sacct -j $j -X -n -o JobName%24,State%11,Elapsed,NodeList%14 2>/dev/null | head -1) TB=$(grep -c Traceback "$f")"
  grep -E "\[eval\] TRAJ (val_best|size_|final_ft)|TRAJ save failed|final_ft_failed|final_ft from" "$f" \
    | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar(10|100)_[a-z_-]+_[0-9._]+\.pth?//; s/ \| FLOPs x[0-9.]+//' | cut -c1-200 | tail -6
  echo "    steps=$(grep -c 'Step [0-9]* done' "$f")"
done

echo "=== paired reads (arm vs control; KILL on an arm -> scancel that arm, runbook 10.3)"
pair() { local a=$(run $1) c=$(run $2); [[ -n "$a" && -n "$c" ]] || return; shift 2
  $PY $RD/paired_steps.py "$a" "$c" "$@" 2>/dev/null | sed "s/^/  /"; }
echo "- aug thin 40/10 557 vs P thin 335";      pair 21729557 21726335
echo "- aug twins VGG 104 vs P twins 337";      pair 21737104 21726337
echo "- N3 aug DG R56 189 vs 500";              pair 21767189 21730500
echo "- N4 aug DG VGG-19 105 vs 551";           pair 21737105 21729551
echo "- N2 streams 558 vs P thin 335 (params)"; pair 21729558 21726335 --by params
echo "- C-G twins 509 vs 337 (big-effect)";     pair 21730509 21726337 --min-steps 5 --kill 3 --frac 0.8
echo "- C-G thin 514 vs 335 (big-effect)";      pair 21730514 21726335 --min-steps 5 --kill 3 --frac 0.8
echo "- determinism (must be ~0): 501 vs 335; crop+flip twins 21809595 vs 21737104 (VGG-16) and vs 21729553 (R56)"
pair 21730501 21726335; pair 21809595 21737104; pair 21809595 21729553
echo "- L2 VGG-16 10-pass 21814029 vs 21737104 (its first 2 passes, must be ~0)"; pair 21814029 21737104

echo "=== decision (d) thin guard: r56-w4 rows, aug 557 vs P 335 (TEST, equal keep)"
for j in 21729557 21726335; do
  d=$(run $j); [[ -n "$d" ]] || continue
  echo "--- $j"; grep -E "\[eval\] TRAJ (val_best|size_).*resnet56-width4" $d/logs/rank0.log \
    | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar10_[a-z_-]+_[0-9._]+\.pth?//; s/ \| FLOPs x[0-9.]+//' | cut -c1-150
done
exit 0
