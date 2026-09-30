#!/usr/bin/env python
"""
Honest gain of the TRAJ final fine-tune, from ``eval_traj_final_ft`` events (trees v9b and later).

    python scripts/final_ft_readout.py <run_dir | events.jsonl> [...] [--adopt 2.0] [--kill 0.5]

Per network and label (TEST = the 5k TEST half under protocol P):

* ``walk``   — TEST Δacc at the point before the final fine-tune (the walk's own recovery);
* ``final``  — TEST Δacc after it;
* ``honest`` — (final − walk) − (origin's final − origin's start): the gain the long recipe gives the
  pruned net beyond what it gives any net. The rule (sitting 29 Sep): ≥ adopt pp → paper tables use
  ``final_ft`` (captioned); < kill pp → the long recipe is not the lever; otherwise HOLD.
  ``ORIGIN-HURT`` instead when the recipe costs the unpruned origin more than 0.5 pp: subtracting a
  negative origin change inflates ``honest`` by that much (the 1-epoch smoke 21730499 printed
  "honest +5.78 ADOPT" on a raw gain of −0.06). Read the raw gain there.
* ``10k``    — for size points and origin only: Δacc on the whole CIFAR test set (val half + TEST
  half, weighted by their sizes). Those points are chosen by size and fine-tuned without val, so
  neither half selected them. ``val_best`` was selected on the val half: 5k only.

``<label>+scratch`` rows (``SPECTRA_EVAL_FINAL_FT_SCRATCH``) are read against ``origin+scratch`` and
against the inherited ``<label>`` row at the same architecture.
"""
import argparse
import glob
import json
import os
import sys


def load_rows(path):
    """``{network: {label: event}}``; later rows win."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "events", "rank*.jsonl")))
    else:
        files = [path]
    out = {}
    for name in files:
        with open(name, encoding="utf-8") as fh:
            for line in fh:
                if "eval_traj_final_ft" not in line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("event") == "eval_traj_final_ft":
                    out.setdefault(os.path.basename(str(ev.get("network", "?"))), {})[ev["label"]] = ev
    return out


def _pp(a, b):
    return (float(a) - float(b)) * 100.0


def full_test_dacc(row, n_val=5000, n_test=5000):
    """Δacc (pp) on val half + TEST half together."""
    final = (float(row["val_final"]) * n_val + float(row["test_final"]) * n_test) / (n_val + n_test)
    start = (float(row["val_origin"]) * n_val + float(row["test_origin"]) * n_test) / (n_val + n_test)
    return (final - start) * 100.0


def honest_gain(row, origin):
    gain = _pp(row["test_final"], row["test_walk"])
    if origin is None:
        return gain, None
    return gain, gain - _pp(origin["test_final"], origin["test_walk"])


def verdict(honest, adopt=2.0, kill=0.5, origin_change=None, origin_floor=-0.5):
    if honest is None:
        return "NO-ORIGIN"
    if origin_change is not None and origin_change < origin_floor:
        return "ORIGIN-HURT"
    if honest >= adopt:
        return "ADOPT"
    if honest < kill:
        return "CROSS-OFF"
    return "HOLD"


def readout(rows, adopt=2.0, kill=0.5, n_val=5000, n_test=5000):
    lines = []
    for label in sorted(rows, key=lambda k: (k.startswith("origin"), k.endswith("+scratch"), k)):
        row = rows[label]
        scratch = label.endswith("+scratch")
        origin = rows.get("origin+scratch" if scratch else "origin")
        walk = _pp(row["test_walk"], row["test_origin"])
        final = _pp(row["test_final"], row["test_origin"])
        text = (f"  {label:<24} step={int(row['step']):>4} | params x{float(row['param']):.3f} | "
                f"FLOPs x{float(row['flop']):.3f} | walk {walk:+6.2f} | final {final:+6.2f} pp")
        if label.startswith("origin"):
            text += f" | origin change {final - walk:+.2f} pp"
        else:
            gain, honest = honest_gain(row, origin)
            origin_change = _pp(origin["test_final"], origin["test_walk"]) if origin is not None else None
            text += f" | gain {gain:+.2f}"
            text += (f" | honest {honest:+.2f} pp {verdict(honest, adopt, kill, origin_change)}"
                     if honest is not None else " | honest n/a (no origin row)")
            if scratch and label[:-len("+scratch")] in rows:
                text += (f" | scratch−inherit "
                         f"{_pp(row['test_final'], rows[label[:-len('+scratch')]]['test_final']):+.2f} pp")
        if not label.startswith("val_best"):
            text += f" | 10k {full_test_dacc(row, n_val, n_test):+.2f} pp"
        else:
            text += " | 10k n/a (val-selected)"
        lines.append(text)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--adopt", type=float, default=2.0)
    ap.add_argument("--kill", type=float, default=0.5)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=5000)
    args = ap.parse_args(argv)
    for path in args.paths:
        rows = load_rows(path)
        if not rows:
            print(f"{path}: no eval_traj_final_ft rows yet")
            continue
        for network in sorted(rows):
            print(f"{network}  [{path}]")
            print("\n".join(readout(rows[network], args.adopt, args.kill, args.n_val, args.n_test)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
