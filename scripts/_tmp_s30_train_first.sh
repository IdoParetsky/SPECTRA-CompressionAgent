#!/usr/bin/env bash
L=/home/paretsky/scratch_audit/tree_v9c/runs/job21737123/logs/rank0.log
date '+%H:%M:%S'
grep -E "DONE Episode|PPO update|PROBE|Snapshot frozen|REWIND|Traceback" $L | sed -E 's/^([0-9-]+ [0-9:]+)\.[0-9]+ .*\| /\1 | /' | cut -c1-230 | tail -8
grep -cE "Traceback" $L