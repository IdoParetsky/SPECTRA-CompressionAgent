#!/usr/bin/env python
"""
Readout of Torch-Pruning ``reproduce/main.py --mode prune`` runs: the output dir of
``scripts/h2h_depgraph.sbatch``, or a log file (the repo ships its own under ``reproduce/run/``).

Per run: wall-clock of each stage (sparse learning, pruning, fine-tune), accuracy right before / after the
cut, the best epoch (their selection, made on the test set) and the last epoch (no selection), params and
FLOPs kept, and board energy per stage when ``gpu_samples.csv`` sits in the dir. Deployment rows of
``bench_r*.jsonl`` are reduced to the median over repeats, with speedups against the ``origin`` row.

    python scripts/h2h_readout.py <out dir | log> ... [--jsonl out.jsonl]
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_readout  # noqa: E402

LINE = re.compile(r"^\[(\d\d)/(\d\d) (\d\d):(\d\d):(\d\d)\] (\S+) INFO: (.*)$")
EPOCH = re.compile(r"Epoch (\d+)/(\d+), Acc=([\d.]+)")
SIZE = re.compile(r"^(Params|FLOPs): ([\d.]+) M => ([\d.]+) M")
CUT_ACC = re.compile(r"^Acc: ([\d.]+) => ([\d.]+)")
BEST = re.compile(r"^Best Acc=([\d.]+)")
STAGES = {"Regularizing...": "sl", "Pruning...": "prune", "Finetuning...": "ft"}


def parse_log(path, year):
    """Stage windows, epoch accuracies and the cut's size / accuracy lines of one log."""
    out = {"log": path, "name": None, "stages": {}, "params_m": None, "flops_m": None, "cut_acc": None}
    stage = previous = None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            match = LINE.match(raw.strip())
            if not match:
                continue
            month, day, hh, mm, ss, name, message = match.groups()
            stamp = datetime(year, int(month), int(day), int(hh), int(mm), int(ss))
            if previous and stamp < previous:
                stamp = stamp.replace(year=stamp.year + 1)
            previous, out["name"], message = stamp, name, message.strip()
            if message in STAGES:
                if stage:
                    out["stages"][stage]["end"] = stamp
                stage = STAGES[message]
                out["stages"][stage] = {"start": stamp, "end": stamp, "epochs": [], "best": None}
                continue
            if stage is None:
                continue
            row = out["stages"][stage]
            row["end"] = stamp
            if EPOCH.search(message):
                epoch = EPOCH.search(message)
                row["epochs"].append((int(epoch.group(1)), float(epoch.group(3))))
            elif BEST.match(message):
                row["best"] = float(BEST.match(message).group(1))
            elif SIZE.match(message):
                size = SIZE.match(message)
                out[f"{size.group(1).lower()}_m"] = (float(size.group(2)), float(size.group(3)))
            elif CUT_ACC.match(message) and stage == "prune":
                out["cut_acc"] = tuple(float(v) for v in CUT_ACC.match(message).groups())
    return out


def summarize_log(parsed, samples=()):
    stages = {}
    for stage, row in parsed["stages"].items():
        stages[stage] = {"minutes": round((row["end"] - row["start"]).total_seconds() / 60, 2),
                         "epochs": len(row["epochs"]), "best": row["best"],
                         "last": row["epochs"][-1][1] if row["epochs"] else None,
                         "energy_wh": (round(cost_readout.energy_wh(samples, row["start"], row["end"]), 2)
                                       if samples else None)}
    params, flops = parsed["params_m"], parsed["flops_m"]
    return {"name": parsed["name"], "log": parsed["log"], "stages": stages,
            "total_min": round(sum(s["minutes"] for s in stages.values()), 2),
            "params_m": params, "flops_m": flops,
            "param_kept": round(params[1] / params[0], 4) if params else None,
            "flop_kept": round(flops[1] / flops[0], 4) if flops else None,
            "speed_up": round(flops[0] / flops[1], 3) if flops else None, "cut_acc": parsed["cut_acc"]}


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def read_bench(folder):
    """Median over ``bench_r*.jsonl`` repeats per (network group, label), with speedups against ``origin``."""
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "bench_r*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    groups = {}
    for row in rows:
        groups.setdefault((row["network"], row["label"]), []).append(row)
    cells = []
    for (network, label), items in sorted(groups.items()):
        cell = {"network": network, "label": label, "repeats": len(items), "params": items[0]["params"],
                "macs": items[0]["macs"], "gpu": items[0].get("gpu"), "host": items[0].get("host"), "by_batch": {}}
        for bs in items[0]["by_batch"]:
            keys = items[0]["by_batch"][bs].keys()
            cell["by_batch"][bs] = {k: _median(i["by_batch"][bs].get(k) for i in items) for k in keys}
        cells.append(cell)
    for cell in cells:
        origin = next((c for c in cells if c["network"] == cell["network"] and c["label"] == "origin"), None)
        for bs, b in cell["by_batch"].items():
            o = origin["by_batch"].get(bs) if origin else None
            b["speedup"] = (round(o["latency_ms_median"] / b["latency_ms_median"], 3)
                            if o and b.get("latency_ms_median") else None)
    return cells


def _logs(arg):
    if os.path.isfile(arg):
        return [arg]
    top = sorted(glob.glob(os.path.join(arg, "*.log")))
    return top or sorted(glob.glob(os.path.join(arg, "**", "*.txt"), recursive=True))


def readout(arg):
    folder = arg if os.path.isdir(arg) else os.path.dirname(arg)
    samples = cost_readout.read_power(folder)
    year = samples[0][0].year if samples else datetime.now().year
    runs = [summarize_log(parse_log(path, year), samples) for path in _logs(arg)]
    return {"path": arg, "runs": [r for r in runs if r["stages"]], "bench": read_bench(folder)}


def _print(result):
    print(f"=== {result['path']}")
    for run in result["runs"]:
        parts = []
        for stage in ("sl", "prune", "ft"):
            s = run["stages"].get(stage)
            if s:
                acc = f" best {s['best']} last {s['last']}" if s["epochs"] else ""
                wh = f" {s['energy_wh']} Wh" if s["energy_wh"] is not None else ""
                parts.append(f"{stage} {s['minutes']} min{acc}{wh}")
        print(f"  {run['name']}: {run['total_min']} min | " + " | ".join(parts)
              + f" | params kept {run['param_kept']} FLOPs kept {run['flop_kept']} ({run['speed_up']}x)"
              + f" | cut acc {run['cut_acc']}")
    for cell in result["bench"]:
        batches = " | ".join(f"bs{bs} {b['latency_ms_median']:.3f} ms {b['throughput_img_s']:.0f} img/s"
                             + (f" x{b['speedup']:.2f}" if b.get("speedup") else "")
                             for bs, b in cell["by_batch"].items())
        print(f"  [bench {cell['gpu']} @ {cell['host']}, {cell['repeats']} repeats] {cell['network']} {cell['label']}: "
              f"{cell['params'] / 1e6:.3f} M params {cell['macs'] / 1e6:.2f} M MACs | {batches}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--jsonl")
    args = parser.parse_args(argv)
    for path in args.paths:
        result = readout(path)
        _print(result)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(result, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
