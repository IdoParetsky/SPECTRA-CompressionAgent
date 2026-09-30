#!/usr/bin/env bash
# Wait (<= 90 min) for 21729556's last r56-w4 TRAJ row, then print the thin-rule read at equal keep.
B=/home/paretsky/scratch_audit/tree_v9b/runs
L=$B/job21729556/logs/rank0.log
for i in $(seq 1 180); do
  grep -q 'TRAJ size_param0.60 resnet56-width4' "$L" 2>/dev/null && break
  squeue -h -j 21729556 2>/dev/null | grep -q . || break
  sleep 30
done
date "+=== %d %b %H:%M:%S"
squeue -h -j 21729556,21730499 -o "%.10i %.20j %.2t %.8M %R" 2>/dev/null
python3 - "$B/job21729555/logs/rank0.log" "$L" <<'PY'
import re, sys
pat = re.compile(r"TRAJ (val_best|size_param0\.\d+) (\S+?)(?:_cifar10\S*)? (?:step=(\d+) \| acc ([\d.]+) -> ([\d.]+) \(([-+][\d.]+)\) \| params x([\d.]+)|NONE)")
def rows(path):
    out = {}
    for line in open(path, errors="replace"):
        if "[eval] TRAJ" not in line or "final_ft" in line:
            continue
        m = pat.search(line)
        if m:
            lab, net = m.group(1), m.group(2).split("_")[0]
            out[(net, lab)] = None if m.group(3) is None else (int(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(7)))
    return out
p, a = rows(sys.argv[1]), rows(sys.argv[2])
for net in ("resnet20-width2", "resnet56-width4"):
    for lab in ("val_best", "size_param0.80", "size_param0.60"):
        rp, ra = p.get((net, lab)), a.get((net, lab))
        if rp is None or ra is None:
            print(f"{net:16s} {lab:15s} P={rp} aug={ra}")
            continue
        d = (ra[2] - ra[1]) - (rp[2] - rp[1])
        same = "same step" if rp[0] == ra[0] else f"steps {rp[0]}/{ra[0]}"
        print(f"{net:16s} {lab:15s} P {100*(rp[2]-rp[1]):+6.2f} @ {rp[3]:.3f} | aug {100*(ra[2]-ra[1]):+6.2f} @ {ra[3]:.3f} | aug-P {100*d:+5.2f} pp ({same})")
PY
cd /home/paretsky/scratch_audit/tree_v9c
/home/paretsky/.conda/envs/spectra/bin/python scripts/paired_steps.py $B/job21729556 $B/job21729555 2>/dev/null
/home/paretsky/.conda/envs/spectra/bin/python scripts/crossfit_readout.py $B/job21729556 --taus 10 --sizes param:0.8,0.6 2>/dev/null | grep -E "census|size_param|crossfit@10"
