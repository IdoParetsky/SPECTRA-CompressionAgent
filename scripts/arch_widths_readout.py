#!/usr/bin/env python
"""
Per-stage residual and inner widths of a run's saved candidates (``traj_models/*.json``, key ``arch``).

    python scripts/arch_widths_readout.py <run_dir> [...] [--labels val_best,size_param0.60]

ResNets: a stage's *residual* width is the stream its blocks add into (the downsample conv and every
block's last conv form one coupled group); *inner* are the other convs of each block (``conv1`` in a
basic block, ``conv1`` / ``conv2`` in a bottleneck). Other families print their conv widths in order.
On the thin ResNet-56 the sens allocation walk keeps every residual stream full and the walk
heuristics cut it (ledger §231); this prints the same read for any run, e.g. a freeze TEST.
"""
import argparse
import glob
import json
import os
import re
import statistics

STAGE = re.compile(r"^layer(\d+)\.(\d+)\.(.+)$")


def candidates(run_dir, labels=None):
    """Saved walk candidates of ``run_dir``, without their fine-tuned copies (``__ft<epochs>``)."""
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "traj_models", "*.json"))):
        if "__ft" in os.path.basename(path):
            continue
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if labels and meta.get("label") not in labels:
            continue
        out.append(meta)
    return out


def stage_widths(arch):
    """``{stage: (residual widths, inner widths)}``, or ``None`` when no ``layer<k>.<block>.`` convs exist."""
    convs = [(name, spec["out_channels"]) for name, spec in arch.items() if spec.get("type") == "Conv2d"]
    stages = {}
    for name, width in convs:
        m = STAGE.match(name)
        if m:
            stages.setdefault(int(m.group(1)), []).append((m.group(3), width))
    if not stages:
        return None
    out = {}
    for s, members in sorted(stages.items()):
        last = "conv3" if any(tail == "conv3" for tail, _ in members) else "conv2"
        res = sorted({w for tail, w in members if tail == last or tail.startswith("downsample")})
        inner = [w for tail, w in members if tail.startswith("conv") and tail != last]
        out[s] = (res, inner)
    return out


def describe(meta):
    arch = meta.get("arch") or {}
    point = meta.get("point") or {}
    head = (f"{os.path.basename(str(meta.get('network', '?')))[:28]:28s} {str(meta.get('label', '?')):16s} "
            f"step={point.get('step', '?'):>4} params x{float(point.get('param', float('nan'))):.3f} "
            f"FLOPs x{float(point.get('flop', float('nan'))):.3f}")
    stages = stage_widths(arch)
    if stages is None:
        widths = [spec["out_channels"] for spec in arch.values() if spec.get("type") == "Conv2d"]
        return f"{head} | convs {' '.join(map(str, widths))}"
    parts = []
    for s, (res, inner) in stages.items():
        tail = f"in {min(inner)}/{statistics.median(inner):g}/{max(inner)}" if inner else "in -"
        parts.append(f"s{s} res {'/'.join(map(str, res))} | {tail}")
    return f"{head} | " + "  ||  ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--labels", default="", help="comma-separated labels (default: every saved candidate)")
    args = ap.parse_args()
    labels = {x.strip() for x in args.labels.split(",") if x.strip()} or None
    print("inner = min/median/max of the block-inner conv widths; res = the residual stream width(s)")
    for run_dir in args.run_dirs:
        rows = candidates(run_dir, labels)
        print(f"######## {run_dir}: {len(rows)} candidate(s)")
        for meta in rows:
            print("  " + describe(meta))


if __name__ == "__main__":
    main()
