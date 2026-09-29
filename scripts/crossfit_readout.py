#!/usr/bin/env python
"""
Full-test-set readouts of a protocol-P walk, from recorded TRAJ points (``eval_traj_summary``).

    python scripts/crossfit_readout.py <run_dir | events.jsonl> [...] [--taus 10,5] [--sizes param:0.8,0.6]

Under ``SPECTRA_VAL_FROM_TEST=1`` the CIFAR test set is split into a val half (A) and a TEST half
(B). A heuristic walk that reads neither half — mild / L1 geometry, walk fine-tune early-stopped
on the **train** loss, no rollback — produces the same points whichever half is called val, and
every point records both halves. So, with no new GPU run:

* ``crossfit@τ`` — two folds of the τ rule: select on A and report B (the live rule), select on B
  and report A. Their mean estimates the rule on the whole 10k test set without selecting on the
  half it is reported on. The two folds may pick different points; both are printed.
* ``size_* 10k`` — size points are chosen by size, not by either half: their Δacc on the whole test
  set is the size-weighted mean of the two halves.
* ``census`` — cut points whose Δacc is above zero on val, on TEST, and on both. The legacy
  memorised val sat at 0.999–1.000 on full-width zoo nets, where no cut could raise it.

Not valid for walks that read val (``SPECTRA_EVAL_ROLLBACK=1``, or a policy whose state carries
accuracy): there the walk itself depends on which half is val.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traj_readout  # noqa: E402


def _swap(point):
    out = dict(point)
    out["val_dacc_pp"], out["test_dacc_pp"] = point["test_dacc_pp"], point["val_dacc_pp"]
    return out


def crossfit(points, tau):
    """``(fold_a, fold_b, mean_pp)``; each fold is ``(point, reported Δacc pp)`` or ``None``."""
    a = traj_readout.val_best(points, tau)
    b = traj_readout.val_best([_swap(p) for p in points], tau)
    fold_a = (a, float(a["test_dacc_pp"])) if a else None
    fold_b = (b, float(b["test_dacc_pp"])) if b else None           # b is swapped: this is half A
    if fold_a is None or fold_b is None:
        return fold_a, fold_b, None
    return fold_a, fold_b, (fold_a[1] + fold_b[1]) / 2.0


def full_test_dacc(point, n_val=5000, n_test=5000):
    return (float(point["val_dacc_pp"]) * n_val + float(point["test_dacc_pp"]) * n_test) / (n_val + n_test)


def census(points):
    cuts = [p for p in points if int(p["step"]) >= 0 and float(p["param"]) < 1.0 - 1e-9]
    val_up = [p for p in cuts if float(p["val_dacc_pp"]) > 0]
    test_up = [p for p in cuts if float(p["test_dacc_pp"]) > 0]
    both = [p for p in val_up if float(p["test_dacc_pp"]) > 0]
    deepest = min(both, key=lambda p: float(p["param"])) if both else None
    return {"cuts": len(cuts), "val_up": len(val_up), "test_up": len(test_up), "both_up": len(both),
            "max_val_pp": max((float(p["val_dacc_pp"]) for p in cuts), default=None),
            "deepest_both_up": deepest}


def _fmt(fold):
    if fold is None:
        return "NONE"
    p, d = fold
    return f"{d:+.2f} @ {float(p['param']):.3f}/{float(p['flop']):.3f} step {int(p['step'])}"


def readout(points, taus=(10.0,), sizes=(), n_val=5000, n_test=5000):
    lines = []
    for tau in taus:
        a, b, mean = crossfit(points, tau)
        text = f"  crossfit@{tau:g}: A→B {_fmt(a)} | B→A {_fmt(b)}"
        text += f" | mean {mean:+.2f} pp (10k)" if mean is not None else " | mean n/a"
        lines.append(text)
    for kind, target in sizes:
        hit = next((p for p in points if int(p["step"]) >= 0 and float(p[kind]) <= target + 1e-9), None)
        label = f"size_{kind}{target:.2f}"
        if hit is None:
            lines.append(f"  {label}: NONE")
            continue
        lines.append(f"  {label}: TEST half {float(hit['test_dacc_pp']):+.2f} | val half "
                     f"{float(hit['val_dacc_pp']):+.2f} | 10k {full_test_dacc(hit, n_val, n_test):+.2f} pp "
                     f"@ {float(hit['param']):.3f}/{float(hit['flop']):.3f} step {int(hit['step'])}")
    c = census(points)
    deep = c["deepest_both_up"]
    lines.append(f"  census: {c['cuts']} cut points | val Δ>0 {c['val_up']} | TEST Δ>0 {c['test_up']} | "
                 f"both {c['both_up']} | max val Δ "
                 f"{'n/a' if c['max_val_pp'] is None else format(c['max_val_pp'], '+.2f')}"
                 + (f" | deepest both>0 @ {float(deep['param']):.3f} step {int(deep['step'])}" if deep else ""))
    return lines


def _sizes(raw):
    if not raw or ":" not in raw:
        return ()
    kind, _, values = raw.partition(":")
    return tuple((kind.strip(), float(v)) for v in values.split(",") if v.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--taus", default="10")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=5000)
    args = ap.parse_args(argv)
    taus = tuple(float(t) for t in args.taus.split(",") if t.strip())
    for path in args.paths:
        runs = traj_readout.load_points(path)
        if not runs:
            print(f"{path}: no eval_traj_summary events")
            continue
        for network, points, _ in runs:
            print(f"{network}  [{path}]")
            print("\n".join(readout(points, taus, _sizes(args.sizes), args.n_val, args.n_test)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
