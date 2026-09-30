#!/usr/bin/env bash
L=/home/paretsky/scratch_audit/tree_v9c/runs/job21737123/logs/rank0.log
C=/home/paretsky/scratch_audit/tree_v7/runs/job21536396/logs/rank0.log
date '+%H:%M:%S'
echo "--- train DONE lines"
grep -E "DONE Episode" $L | sed -E 's/^([0-9-]+ [0-9:]+)\.[0-9]+ .*(DONE Episode)/\1 \2/' | cut -c1-140
echo "--- control first 4 DONE lines"
grep -E "DONE Episode" $C | head -4 | sed -E 's/^([0-9-]+ [0-9:]+)\.[0-9]+ .*(DONE Episode)/\1 \2/' | cut -c1-140
echo "--- train now"
tail -c 3000 $L | tr '\r' '\n' | grep -E "net=|Epoch" | tail -2 | sed -E 's/^([0-9-]+ [0-9:]+)\.[0-9]+ .*\| (phase=[^|]*)\|(.*)$/\1 \2|\3/' | cut -c1-200
echo "--- epoch timing (train, last 6 Epoch lines)"
grep -E "Epoch [0-9]+/12" $L | tail -6 | sed -E 's/^([0-9-]+ [0-9:.]+) .*net=([^ ]+) .*step=([0-9]+).*(Epoch [0-9]+\/12).*/\1 \2 step \3 \4/' | cut -c1-150
echo "--- epoch timing (control, 6 Epoch lines in episode 2)"
grep -E "Epoch [0-9]+/12" $C | sed -n '400,405p' | sed -E 's/^([0-9-]+ [0-9:.]+) .*net=([^ ]+) .*step=([0-9]+).*(Epoch [0-9]+\/12).*/\1 \2 step \3 \4/' | cut -c1-150
nvidia-smi >/dev/null 2>&1; sstat -j 21737123.batch -n -o AveCPU,MaxRSS 2>/dev/null | head -2