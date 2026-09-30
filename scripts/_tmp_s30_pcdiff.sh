#!/usr/bin/env bash
A=/home/paretsky/scratch_audit/tree_v7/runs/job21536396/agent_checkpoints/policy_config.json
B=/home/paretsky/scratch_audit/tree_v9c/runs/job21737123/agent_checkpoints/policy_config.json
ls -la $B 2>&1 | cut -c1-150
/home/paretsky/.conda/envs/spectra/bin/python - "$A" "$B" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        key = f"{p}.{k}" if p else k
        if isinstance(v, dict):
            out.update(flat(v, key))
        else:
            out[key] = v
    return out
fa, fb = flat(a), flat(b)
for k in sorted(set(fa) | set(fb)):
    if fa.get(k) != fb.get(k):
        print(f"{k}: {str(fa.get(k))[:70]!s} -> {str(fb.get(k))[:70]!s}")
PY
L=/home/paretsky/scratch_audit/tree_v9c/runs/job21737123/logs/rank0.log
grep -E "DONE Episode|PPO update|PROBE|Snapshot frozen|Traceback" $L 2>/dev/null | tail -3 | sed -E 's/^.*\| //' | cut -c1-200
grep -cE "Traceback" /home/paretsky/scratch_audit/tree_v9c/runs/slurm_logs/spectra_21737123.out 2>/dev/null