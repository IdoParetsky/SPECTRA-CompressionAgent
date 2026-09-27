#!/usr/bin/env bash
# 28 Sep (Fable): exact [eval] TRAJ val_best rows + wall time for the 11 completed V8 walks (for ledger 126+ and the Gilad note).
cd /home/paretsky/scratch_audit/tree_v8/runs || exit 1
for j in 21703433 21703434 21703435 21703436 21703437 21703438 21703439 21703440 21703441 21703466 21703467; do
  d=job$j; O=$(ls $d/logs/rank0.log 2>/dev/null | head -1)
  name=$(sacct -j $j -n -X -o JobName%30 | head -1 | tr -d ' ')
  el=$(sacct -j $j -n -X -o Elapsed,State,NodeList%14 | head -1 | tr -s ' ')
  echo "== $j $name |$el"
  [[ -n "$O" ]] || { echo "   no rank0.log"; continue; }
  grep -oE 'FLAGS.{0,400}' "$O" | head -1 | tr ' ' '\n' | grep -E 'ft_recipe|FT_OPTIM|FT_SCHEDULE|FT_LR=|FT_BN_RECAL|FT_LSQ|REINIT_EDITED' | sort -u | tr '\n' ' '; echo
  grep -E '\[eval\] TRAJ val_best' "$O" | sed -E 's/^.*\[eval\]/[eval]/' | cut -c1-230
  grep -cE 'Traceback' "$O" | sed 's/^/   tracebacks: /'
done
