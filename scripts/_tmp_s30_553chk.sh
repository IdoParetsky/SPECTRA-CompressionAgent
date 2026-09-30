#!/usr/bin/env bash
L=/home/paretsky/scratch_audit/tree_v9b/runs/job21729553/logs/rank0.log
grep -E "\[eval\] TRAJ (val_best|size_|floor_hold|terminal)" $L 2>/dev/null | sed -E 's/^.*\| \[eval\]/[eval]/' | cut -c1-170
grep -E "phase=eval_test" /home/paretsky/scratch_audit/tree_v9b/runs/job21729553/events/rank0.jsonl 2>/dev/null | tail -1 | cut -c1-200
tail -c 600 $L | tr '\r' '\n' | tail -3 | cut -c1-200