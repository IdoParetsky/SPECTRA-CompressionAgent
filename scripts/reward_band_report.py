#!/usr/bin/env python3
"""
Reward-band report: where does each dataset land in the NEON trichotomy?

``compute_reward`` only credits a cut when ``-tau <= delta_acc <= 0`` (reward ``+reduction``).
Above the band the step scores ``+reduction ** 3``; below it, ``-reduction ** 3``. If a dataset
never lands inside the band, every real cut is punished cubically and the argmax policy is
"never prune" -- no encoder or agent change can fix that.

Usage:
    python scripts/reward_band_report.py runs/job20884670 [runs/job... ...]
    python scripts/reward_band_report.py --glob 'runs/job*/reward_trace.jsonl'
"""
import argparse
import glob as globlib
import json
import os
import sys
from collections import defaultdict

BRANCHES = ("gain", "in_budget", "over_budget")


def load(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def tabulate(rows, key):
    counts = defaultdict(lambda: defaultdict(int))
    cuts = defaultdict(list)
    for row in rows:
        # Rate 1.0 is the identity action; it carries no pruning signal.
        if float(row.get("rate", 1.0)) >= 1.0:
            continue
        bucket = row.get(key, "unknown")
        counts[bucket][row.get("branch", "unknown")] += 1
        cuts[bucket].append(float(row.get("delta_acc_pp", 0.0)))
    return counts, cuts


def report(rows, key, title):
    counts, cuts = tabulate(rows, key)
    if not counts:
        return
    print(f"\n=== {title} (identity rate 1.0 excluded) ===")
    width = max(len(str(k)) for k in counts)
    header = f"{'':<{width}}  {'n':>6}  " + "  ".join(f"{b:>11}" for b in BRANCHES) + "   median Δacc"
    print(header)
    print("-" * len(header))
    for bucket in sorted(counts, key=lambda b: -sum(counts[b].values())):
        row = counts[bucket]
        total = sum(row.values())
        cells = []
        for branch in BRANCHES:
            n = row.get(branch, 0)
            cells.append(f"{n:>4} {100.0 * n / total:>5.1f}%")
        deltas = sorted(cuts[bucket])
        median = deltas[len(deltas) // 2] if deltas else float("nan")
        print(f"{bucket:<{width}}  {total:>6}  " + "  ".join(cells) + f"   {median:>8.2f} pp")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", help="run directories or reward_trace.jsonl paths")
    parser.add_argument("--glob", dest="pattern", help="glob for reward_trace.jsonl files")
    args = parser.parse_args()

    paths = []
    for item in args.run_dirs:
        paths.append(item if item.endswith(".jsonl") else os.path.join(item, "reward_trace.jsonl"))
    if args.pattern:
        paths.extend(globlib.glob(args.pattern))
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("no reward_trace.jsonl found (set SPECTRA_REWARD_TRACE=1 on the job)", file=sys.stderr)
        return 1

    rows = load(paths)
    print(f"traces: {len(paths)} file(s), {len(rows)} step(s)")
    report(rows, "dataset", "Reward band by dataset")
    report(rows, "net", "Reward band by network")

    empty = [b for b, c in tabulate(rows, "dataset")[0].items()
             if c.get("in_budget", 0) + c.get("gain", 0) == 0]
    if empty:
        print("\nEMPTY BAND (every non-identity cut scores -reduction**3): " + ", ".join(sorted(empty)))
        print("The agent cannot learn to prune these; the environment offers no rewarded cut.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
