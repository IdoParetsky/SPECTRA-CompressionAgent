#!/usr/bin/env python
"""
Paired early read of a no-agent arm against a finished control, from ``step`` events (any tree).

    python scripts/paired_steps.py <arm run_dir | events.jsonl> <control run_dir | events.jsonl>
        [--by step|params] [--min-steps 15] [--kill 1.0] [--adopt 1.0] [--frac 0.75]

A mild / L1 walk cuts the same groups in the same order whatever the recovery recipe, so at step k
the arm and the control hold the same architecture and only the recovery differs. The paired
statistic is val Δacc(arm) − val Δacc(control) at each cut both runs have made. It can be read while
the arm is still running, and needs no re-run of the control. Arms that change the geometry (stream
protection, rollback, a new menu) must use ``--by params``: each arm cut is paired with the control
cut nearest in parameters kept (within 1 % relative).

Verdict per network, pre-registered:

* ``KILL``      — at least ``--min-steps`` pairs, mean ≤ −kill pp and ≥ frac of pairs worse;
* ``ADOPT?``    — at least ``--min-steps`` pairs, mean ≥ +adopt pp and ≥ frac of pairs better
                  (a candidate: the arm still has to finish and win on TEST at the selected point);
* ``CONTINUE``  — otherwise.

Val only: this never looks at TEST, so it cannot select on TEST.
"""
import argparse
import glob
import json
import os
import statistics
import sys


def load_cuts(path):
    """``{network: {step: (val_dacc_pp, params_m)}}`` from structural ``step`` events; later rows win."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "events", "rank*.jsonl")))
    else:
        files = [path]
    out = {}
    for name in files:
        with open(name, encoding="utf-8") as fh:
            for line in fh:
                if '"event": "step"' not in line and '"event":"step"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                phase = ev.get("phase") or ev.get("mode")
                if ev.get("event") != "step" or (phase and phase != "eval_test"):
                    continue
                if ev.get("prune_mode") != "structural" or ev.get("new_acc") is None:
                    continue
                net = os.path.basename(str(ev.get("network") or ev.get("net") or "?"))
                val_pp = (float(ev["new_acc"]) - float(ev["baseline_acc"])) * 100.0
                out.setdefault(net, {})[int(ev["step_index"])] = (val_pp, float(ev["params_after_m"]))
    return out


def pair(arm, ctrl, by="step", tol=0.01, resolution_m=1e-4):
    """``[(step, arm_val, ctrl_val)]`` in arm step order. ``params_after_m`` is logged to 1e-4 M, so a
    params pair is accepted within ``max(tol × params, resolution_m)``."""
    pairs = []
    if by == "step":
        for step in sorted(set(arm) & set(ctrl)):
            pairs.append((step, arm[step][0], ctrl[step][0]))
        return pairs
    ctrl_rows = list(ctrl.values())
    for step in sorted(arm):
        val, params = arm[step]
        best = min(ctrl_rows, key=lambda r: abs(r[1] - params), default=None)
        if best is not None and abs(best[1] - params) <= max(tol * params, resolution_m) + 1e-12:
            pairs.append((step, val, best[0]))
    return pairs


def verdict(pairs, min_steps=15, kill=1.0, adopt=1.0, frac=0.75):
    if not pairs:
        return "CONTINUE", None, None
    diffs = [a - c for _, a, c in pairs]
    mean = statistics.fmean(diffs)
    better = sum(d > 0 for d in diffs) / len(diffs)
    if len(diffs) >= min_steps and mean <= -kill and (1.0 - better) >= frac:
        return "KILL", mean, better
    if len(diffs) >= min_steps and mean >= adopt and better >= frac:
        return "ADOPT?", mean, better
    return "CONTINUE", mean, better


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("arm")
    ap.add_argument("control")
    ap.add_argument("--by", choices=("step", "params"), default="step")
    ap.add_argument("--min-steps", type=int, default=15)
    ap.add_argument("--kill", type=float, default=1.0)
    ap.add_argument("--adopt", type=float, default=1.0)
    ap.add_argument("--frac", type=float, default=0.75)
    args = ap.parse_args(argv)
    arm, ctrl = load_cuts(args.arm), load_cuts(args.control)
    if not arm:
        print(f"{args.arm}: no structural step events yet")
        return 0
    for net in sorted(arm):
        if net not in ctrl:
            print(f"{net}: not in the control")
            continue
        pairs = pair(arm[net], ctrl[net], args.by)
        label, mean, better = verdict(pairs, args.min_steps, args.kill, args.adopt, args.frac)
        if mean is None:
            print(f"{net}: no paired cuts ({len(arm[net])} arm cuts)")
            continue
        last = pairs[-1]
        print(f"{net}: {label} | {len(pairs)} paired cuts by {args.by} | mean arm−control val "
              f"{mean:+.2f} pp | arm better on {better:.0%} | last step {last[0]}: "
              f"arm {last[1]:+.2f} vs control {last[2]:+.2f} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
