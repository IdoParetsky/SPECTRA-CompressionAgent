#!/usr/bin/env bash
T=/home/paretsky/scratch_audit/tree_v9c
sed -n '620,640p' $T/src/A2C_Agent_Reinforce.py
echo "--- prologue copy of the resume bundle"
grep -nE "SPECTRA_RESUME_PATH|train_resume" $T/scripts/spectra.sbatch | head -12
echo "--- fortify governor since/best restore"
grep -nE "def __init__\(self, \*, patience|since_improvement: int|self.since_improvement = |self.rewinds = " $T/src/fortify.py | head -6
ls -la $T/runs/job21737123/agent_checkpoints/ | cut -c1-140