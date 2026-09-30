#!/usr/bin/env bash
# Early telemetry of the control area train 21536396 (read-only), for the P-train go/no-go.
L=/home/paretsky/scratch_audit/tree_v7/runs/job21536396/logs/rank0.log
[[ -s $L ]] || L=/home/paretsky/scratch_audit/tree_v7/runs/slurm_logs/spectra_21536396.out
echo "log: $L"
echo "=== PPO updates 1-3, 5, 10, 15, 20, 30"
grep -E '^.*PPO update (1|2|3|5|10|15|20|30) \|' "$L" | sed -E 's/^.*(PPO update)/\1/' | cut -c1-190
echo "=== gap_to_uniform / pmax, episodes 0-3, 39-40, 79-80"
grep -E 'DONE Episode (0|1|2|3|39|40|79|80) in' "$L" | sed -E 's/^.*(DONE Episode)/\1/' | cut -c1-200
echo "=== first 8 probe lines + snapshots"
grep -E 'PROBE|Probe score|probe score' "$L" | head -8 | cut -c1-200
grep -E 'Snapshot frozen|snapshot' "$L" | head -6 | cut -c1-200
echo "=== seconds per episode (median of first 40 and of all)"
grep -oE 'DONE Episode [0-9]+ in [0-9.]+s' "$L" | awk '{print $5}' | tr -d s > /tmp/_s30_eps.txt
python3 - <<'PY'
v = [float(x) for x in open('/tmp/_s30_eps.txt') if x.strip()]
import statistics as s
if v:
    print(f"episodes={len(v)} median_first40={s.median(v[:40]):.0f}s median_all={s.median(v):.0f}s mean_all={s.mean(v):.0f}s")
PY
grep -m3 -E 'batch size|Batch size|BATCH_SIZE|GPU:|CUDA device|NVIDIA' "$L" | cut -c1-160
echo "=== 21729556 progress + P-train state"
grep -cE '\[eval\] TRAJ' /home/paretsky/scratch_audit/tree_v9b/runs/job21729556/logs/rank0.log
grep -E 'Step [0-9]+ done' /home/paretsky/scratch_audit/tree_v9b/runs/job21729556/logs/rank0.log | tail -1 | sed -E 's/^.*(Step)/\1/' | cut -c1-120
squeue -h -j 21737095,21730499,21729556 -o "%.10i %.20j %.2t %.10M %R" 2>/dev/null
scontrol show job 21737095 2>/dev/null | grep -oE 'TresPerNode=[^ ]*|TresPerJob=[^ ]*|Features=[^ ]*|StartTime=[^ ]*' | tr '\n' ' '; echo
