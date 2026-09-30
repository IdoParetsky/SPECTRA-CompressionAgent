#!/usr/bin/env bash
# Ops heartbeat for the V9b/V9c wave + Stage-4 train (read-only). Opus 5.5 sitting, 30 Sep.
# Run: powershell -NoProfile -File scripts/rexec.ps1 -Quiet -File scripts/_tmp_s30_ops_hb.sh
# TRAIN defaults to whichever Stage-4 train is not held/cancelled; override with TRAIN=<id>.
B=/home/paretsky/scratch_audit/tree_v9b/runs
C=/home/paretsky/scratch_audit/tree_v9c/runs
PY=/home/paretsky/.conda/envs/spectra/bin/python
run() { for d in "$C/job$1" "$B/job$1"; do [[ -f $d/logs/rank0.log ]] && { echo "$d"; return; }; done; }
echo "=== $(date '+%d %b %H:%M') queue"
squeue -u paretsky -h -o "%.10i %.24j %.2t %.10M %.10l %.5y %N %R" 2>/dev/null | grep -vE "c10-(similar|unlike)"
sacct -u paretsky -S 2026-09-30T00:00 -X -n -s CD,F,CA,TO,NF,OOM,PR \
  -o JobID%10,JobName%24,State%12,ExitCode%6,Elapsed%11,End%19,NodeList%14 2>/dev/null

echo "=== Stage-4 train"
if [[ -z "${TRAIN:-}" ]]; then
  for t in 21737123 21737095; do
    st=$(squeue -h -j $t -o '%T %r' 2>/dev/null)
    [[ -n "$st" && "$st" != *JobHeldUser* ]] && { TRAIN=$t; break; }
    [[ -z "$st" ]] && sacct -j $t -X -n -o State 2>/dev/null | grep -qE "RUNNING|COMPLETED|FAILED|TIMEOUT|NODE_FAIL" && { TRAIN=$t; break; }
  done
fi
echo "TRAIN=${TRAIN:-none} $(squeue -h -j ${TRAIN:-0} -o '%j %T %M %N' 2>/dev/null)"
L=$C/job${TRAIN:-0}/logs/rank0.log
if [[ -s $L ]]; then
  grep -m1 -oE "'SPECTRA_VAL_FROM_TEST': '[^']*'|'SPECTRA_FT_AUG': '[^']*'" "$L" | tr '\n' ' '
  grep -m1 -oE "'SPECTRA_BATCH_SIZE': '[^']*'" "$L"; grep -m1 -oE "'SPECTRA_PROBE_SCORE': '[^']*'" "$L"
  grep -E "Val from test|FT aug on" "$L" | sed -E 's/^.*\| (Val|FT)/\1/' | sort -u | head -4
  grep -m1 -E "GPUs:" "$L" | sed -E 's/^.*\| //'
  grep -E "PPO update|PROBE ep|Snapshot frozen|REWIND|rewind" "$L" | sed -E 's/^.*\| (PPO|PROBE|Snapshot|REWIND|rewind)/\1/' | cut -c1-170 | tail -6
  echo "episodes=$(grep -c 'DONE Episode' "$L") TB=$(grep -c Traceback "$L") snaps=$(ls -d $C/job$TRAIN/snapshots/ep* 2>/dev/null | wc -l)"
  grep -E "DONE Episode" "$L" | tail -1 | sed -E 's/^.*(DONE Episode)/\1/' | cut -c1-170
  grep -oE 'DONE Episode [0-9]+ in [0-9.]+s' "$L" | awk '{print $5}' | tr -d s | sort -n \
    | awk '{a[NR]=$1} END {if (NR) printf "median s/episode=%.0f (control 21536396: 580 first 40, 629 all)\n", a[int((NR+1)/2)]}'
else
  echo "no train log yet"
fi

echo "=== TRAJ rows (last 6 per cell)"
for j in 21729552 21729553 21729554 21729556 21729557 21729558 21730499 21730500 21730501 21730506 \
         21730507 21730509 21730514 21730516 21737104 21737105; do
  d=$(run $j); [[ -n "$d" ]] || continue
  f=$d/logs/rank0.log
  echo "--- $j $(sacct -j $j -X -n -o JobName%24,State%11,Elapsed,NodeList%14 2>/dev/null | head -1) TB=$(grep -c Traceback "$f")"
  grep -E "\[eval\] TRAJ (val_best|size_|final_ft)|TRAJ save failed|final_ft_failed|final_ft from" "$f" \
    | sed -E 's/^.*\| \[eval\]/[eval]/; s/_cifar(10|100)_[a-z_-]+_[0-9._]+\.pth?//; s/ \| FLOPs x[0-9.]+//' | cut -c1-200 | tail -6
  echo "    steps=$(grep -c 'Step [0-9]* done' "$f")"
done

echo "=== paired reads (arm vs control; KILL on an arm -> scancel that arm, ops 8)"
pair() { local a=$(run $1) c=$(run $2); [[ -n "$a" && -n "$c" ]] || return; shift 2
  $PY /home/paretsky/scratch_audit/tree_v9c/scripts/paired_steps.py "$a" "$c" "$@" 2>/dev/null | sed "s/^/  /"; }
echo "- aug gate 554 vs P gate 552";            pair 21729554 21729552
echo "- aug twins 553 vs P twins 337";          pair 21729553 21726337
echo "- aug thin 12/4 556 vs 555";              pair 21729556 21729555
echo "- aug thin 40/10 557 vs P thin 335";      pair 21729557 21726335
echo "- N2 streams 558 vs P thin 335 (params)"; pair 21729558 21726335 --by params
echo "- aug twins VGG 104 vs P twins 337";      pair 21737104 21726337
echo "- N4 aug DG VGG-19 105 vs 551";           pair 21737105 21729551
echo "- C-G twins 509 vs 337 (big-effect)";     pair 21730509 21726337 --min-steps 5 --kill 3 --frac 0.8
echo "- C-G thin 514 vs 335 (big-effect)";      pair 21730514 21726335 --min-steps 5 --kill 3 --frac 0.8
echo "- determinism (must be ~0): 500 vs 340, 501 vs 335, 506 vs 337"
pair 21730500 21726340; pair 21730501 21726335; pair 21730506 21726337
exit 0
