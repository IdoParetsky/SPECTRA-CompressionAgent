#!/usr/bin/env python
"""
O26 memorization census: per catalog net, the unpruned val accuracy a run recorded at episode
reset against the TEST accuracy in the checkpoint name (CPU, reads only the events file).

    python scripts/memorization_census.py <run_dir | events.jsonl> [...] [--gap-pp 3] [--val-bar 0.995]

A net whose val sits far above its TEST accuracy has memorized its val split (ledger §141): every
reward and gate on that net scored cuts on memorization loss, not on generalization. Under P
(``SPECTRA_VAL_FROM_TEST=1``) val is half of the test split, so the gap is split noise (~±1 pp).
Checkpoint names end ``_<acc %>_<params M>_<MFLOPs>.pt``; a name without that tail reports no TEST.
"""
import argparse
import glob
import json
import os
import re
import statistics

_NAME_TAIL = re.compile(r"_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.pt$")


def name_test_acc(path):
    """TEST accuracy (fraction) from a ``..._<acc %>_<params M>_<MFLOPs>.pt`` name, or None."""
    match = _NAME_TAIL.search(os.path.basename(path))
    return float(match.group(1)) / 100.0 if match else None


def event_files(target):
    if os.path.isfile(target):
        return [target]
    return sorted(glob.glob(os.path.join(target, "events", "rank*.jsonl")))


def baseline_vals(paths):
    """``{network: [baseline val acc at each episode reset]}`` from ``episode_reset`` events."""
    vals = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if '"episode_reset"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "episode_reset" and event.get("baseline_acc") is not None:
                    vals.setdefault(event.get("network", "?"), []).append(float(event["baseline_acc"]))
    return vals


def census(vals, gap_pp=3.0, val_bar=0.995):
    rows = []
    for network, accs in sorted(vals.items()):
        test = name_test_acc(network)
        val = statistics.median(accs)
        gap = None if test is None else (val - test) * 100.0
        if test is None:
            verdict = "no TEST in name"
        elif val >= val_bar or gap >= gap_pp:
            verdict = "MEMORIZED"
        else:
            verdict = "ok"
        rows.append({"net": os.path.basename(network), "resets": len(accs), "val": val,
                     "val_min": min(accs), "val_max": max(accs), "test": test, "gap_pp": gap,
                     "verdict": verdict})
    return rows


def render(target, rows):
    out = [f"### {target}", "",
           "| net | resets | val (median) | val range | TEST (name) | val − TEST | verdict |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        test = "—" if r["test"] is None else f"{r['test']:.4f}"
        gap = "—" if r["gap_pp"] is None else f"{r['gap_pp']:+.2f} pp"
        out.append(f"| {r['net']} | {r['resets']} | {r['val']:.4f} | {r['val_min']:.4f}–{r['val_max']:.4f} "
                   f"| {test} | {gap} | {r['verdict']} |")
    flagged = sum(r["verdict"] == "MEMORIZED" for r in rows)
    out += ["", f"{flagged}/{len(rows)} nets MEMORIZED", ""]
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("targets", nargs="+", help="run directory (reads events/rank*.jsonl) or one events file")
    parser.add_argument("--gap-pp", type=float, default=3.0, help="flag val − TEST at or above this (pp)")
    parser.add_argument("--val-bar", type=float, default=0.995, help="flag an unpruned val at or above this")
    args = parser.parse_args(argv)
    for target in args.targets:
        paths = event_files(target)
        if not paths:
            print(f"### {target}\n\nno events/rank*.jsonl\n")
            continue
        vals = baseline_vals(paths)
        if not vals:
            print(f"### {target}\n\nno episode_reset events\n")
            continue
        print(render(target, census(vals, args.gap_pp, args.val_bar)))


if __name__ == "__main__":
    main()
