#!/usr/bin/env python
"""
Selection readouts from recorded TRAJ points (``eval_traj_summary`` events, trees v9 and later).

    python scripts/traj_readout.py <run_dir | events.jsonl> [...] [--taus 10,5,2]
        [--sizes param:0.8,0.6] [--k 3] [--edge 1.5]

Per network and run:

* ``val_best@τ`` — the live rule: the most compressed point with val Δacc ≥ −τ anywhere on the walk.
* ``first_exit@τ`` — the deepest point before val first leaves −τ.
* ``smooth@τ`` — the deepest point whose centred ``k``-point median of val stays ≥ −τ.
* ``size_<kind><t>`` — the first point at or below each pre-registered fraction kept.
* ``gap`` — mean and max of (test Δacc − val Δacc) over cut points. With the legacy train-split
  val, a zoo checkpoint's gap is its memorisation offset; with ``SPECTRA_VAL_FROM_TEST=1`` it
  should be split noise only (about ±1 pp on 5k images).
* ``edge`` — how many cut points sit within ``±edge`` pp of −τ. A long edge makes ``val_best`` a
  maximum over noisy draws: the more points hover at the edge, the deeper the lottery can reach.

Only test Δacc at a val-selected (or pre-registered) point may be quoted; this script never picks
on test.
"""
import argparse
import glob
import json
import os
import statistics
import sys


def load_points(path):
    """``[(network, points, source)]`` from a run dir (``events/rank*.jsonl``) or one jsonl file."""
    if os.path.isdir(path):
        files = (sorted(glob.glob(os.path.join(path, "events", "rank*.jsonl")))
                 or sorted(glob.glob(os.path.join(path, "run_records.jsonl"))))
    else:
        files = [path]
    out = []
    for name in files:
        with open(name, encoding="utf-8") as fh:
            for line in fh:
                if '"eval_traj_summary"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "eval_traj_summary" and event.get("points"):
                    out.append((os.path.basename(str(event.get("network", "?"))), event["points"], path))
    return out


def _in_band(point, tau):
    return float(point["val_dacc_pp"]) + 1e-12 >= -float(tau)


def _depth_key(point):
    return float(point["param"]), float(point["flop"]), -float(point["val_dacc_pp"])


def val_best(points, tau):
    band = [p for p in points if _in_band(p, tau)]
    return min(band, key=_depth_key) if band else None


def first_exit(points, tau):
    best = None
    for p in points:
        if not _in_band(p, tau):
            break
        if best is None or _depth_key(p) < _depth_key(best):
            best = p
    return best


def smooth(points, tau, k=3):
    half = max(0, int(k) // 2)
    vals = [float(p["val_dacc_pp"]) for p in points]
    best = None
    for i, p in enumerate(points):
        window = vals[max(0, i - half): i + half + 1]
        if statistics.median(window) + 1e-12 >= -float(tau):
            if best is None or _depth_key(p) < _depth_key(best):
                best = p
    return best


def size_points(points, targets):
    return {f"size_{kind}{t:.2f}": next((p for p in points if float(p[kind]) <= t + 1e-12), None)
            for kind, t in targets}


def gap(points):
    diffs = [float(p["test_dacc_pp"]) - float(p["val_dacc_pp"]) for p in points if int(p["step"]) >= 0]
    if not diffs:
        return None, None
    return statistics.fmean(diffs), max(diffs, key=abs)


def edge_count(points, tau, width=1.5):
    return sum(1 for p in points if int(p["step"]) >= 0
               and abs(float(p["val_dacc_pp"]) + float(tau)) <= float(width))


def parse_sizes(raw):
    if not raw or ":" not in raw:
        return ()
    kind, _, values = raw.partition(":")
    kind = {"flops": "flop", "params": "param"}.get(kind.strip(), kind.strip())
    return tuple((kind, float(v)) for v in values.split(",") if v.strip())


def _row(label, p):
    if p is None:
        return f"  {label:<18} NONE"
    return (f"  {label:<18} step={int(p['step']):>4} | params x{float(p['param']):.3f} | "
            f"FLOPs x{float(p['flop']):.3f} | test {float(p['test_dacc_pp']):+6.2f} pp | "
            f"val {float(p['val_dacc_pp']):+6.2f} pp")


def readout(points, taus, sizes, k, width):
    lines = []
    for tau in taus:
        lines.append(_row(f"val_best@{tau:g}", val_best(points, tau)))
        lines.append(_row(f"first_exit@{tau:g}", first_exit(points, tau)))
        lines.append(_row(f"smooth{k}@{tau:g}", smooth(points, tau, k)))
        lines.append(f"  edge@{tau:g}             {edge_count(points, tau, width)} cut points within "
                     f"±{width:g} pp of -{tau:g}")
    for label, p in size_points(points, sizes).items():
        lines.append(_row(label, p))
    mean_gap, worst = gap(points)
    if mean_gap is not None:
        lines.append(f"  gap (test-val)     mean {mean_gap:+.2f} pp, largest {worst:+.2f} pp")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--taus", default="10")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--edge", type=float, default=1.5)
    args = ap.parse_args(argv)
    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    sizes = parse_sizes(args.sizes)
    for path in args.paths:
        rows = load_points(path)
        if not rows:
            print(f"{path}: no eval_traj_summary points (tree older than v9?)")
            continue
        for network, points, source in rows:
            print(f"{network}  [{source}]  {len(points)} points")
            print("\n".join(readout(points, taus, sizes, args.k, args.edge)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
