#!/usr/bin/env python
"""
What a SPECTRA run cost, from the files it already writes.

Per network: walk minutes split into fine-tune / prune / evaluate / state features, the time between
steps (agent decision plus bookkeeping), the mean decision itself when the run set
``SPECTRA_TIME_DECIDE=1``, fine-tune epochs actually run per budget, final fine-tune minutes and
GPU-hours. Per run: the GPU named in the manifest (the Slurm label is not the card:
``rtx_6000`` nodes carry an RTX 6000 Ada), peak GPU memory from the heartbeat, and board energy when
the run has ``gpu_samples.csv`` (spectra.sbatch's 1 s nvidia-smi sampler).

    python scripts/cost_readout.py <run dir | job id> ... [--jsonl out.jsonl]

Standard library only, so it runs on the login node.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta

STAGES = {"step.finetune": "ft_s", "step.prune": "prune_s", "step.evaluate": "eval_s",
          "step.feature_extraction": "features_s", "step.compute_results": "results_s",
          "step.decide": "decide_s"}
EPOCH_LINE = re.compile(r"\bnet=(\S+).*?\bstep=(-?\d+)\b.*?\| Epoch (\d+)/(\d+):")
PEAK_ALLOC = re.compile(r"gpu_max_alloc_gb=([0-9.]+)")
MAX_SAMPLE_GAP_S = 5.0


def run_roots():
    roots = [r for r in os.environ.get("SPECTRA_RUN_ROOTS", "").split(":") if r]
    if roots:
        return roots
    return sorted(glob.glob("/home/paretsky/scratch_audit/tree_v*/runs")) + [
        os.path.expanduser("~/SPECTRA-CompressionAgent/runs")]


def resolve(arg):
    if os.path.isdir(arg):
        return arg
    job = arg[3:] if arg.startswith("job") else arg
    for root in run_roots():
        path = os.path.join(root, f"job{job}")
        if os.path.isdir(path):
            return path
    return None


def _deep(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _deep(value, key)
            if found is not None:
                return found
    return None


def read_manifest(run):
    try:
        with open(os.path.join(run, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        manifest = {}
    started = _deep(manifest, "started_at")
    try:
        started = datetime.fromisoformat(started) if started else None
    except ValueError:
        started = None
    env = _deep(manifest, "spectra_env") or {}
    return {"gpus": _deep(manifest, "gpus") or [], "host": _deep(manifest, "hostname") or "?",
            "started": started, "profile": env.get("SPECTRA_PROFILE", "?")}


def _blank():
    row = defaultdict(float)
    row["gaps"] = []
    row["finals"] = []
    row["first_t"] = None
    row["last_t"] = None
    return row


def read_events(run):
    """``(per-network rows, run-level fields)`` from ``events/rank0.jsonl``."""
    nets = defaultdict(_blank)
    run_info = {"run_s": None, "episodes": 0, "episode_s": []}
    pending = defaultdict(float)
    last_end = {}
    path = os.path.join(run, "events", "rank0.jsonl")
    if not os.path.exists(path):
        return nets, run_info
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("event")
            t = float(event.get("t", 0.0) or 0.0)
            if kind == "stage" and event.get("stage") in STAGES:
                pending[STAGES[event["stage"]]] += float(event.get("seconds", 0.0) or 0.0)
                if event["stage"] == "step.decide":
                    pending["decide_n"] += 1
            elif kind == "step":
                name = os.path.basename(str(event.get("network", "?")))
                row = nets[name]
                seconds = float(event.get("seconds", 0.0) or 0.0)
                row["steps"] += 1
                row["cuts"] += int(event.get("compression_rate", 1) != 1)
                row["step_s"] += seconds
                for key, value in pending.items():
                    row[key] += value
                pending.clear()
                if name in last_end:
                    row["gaps"].append(max(0.0, t - seconds - last_end[name]))
                last_end[name] = t
                row["first_t"] = t - seconds if row["first_t"] is None else row["first_t"]
                row["last_t"] = t
            elif kind == "eval_traj_final_ft":
                name = os.path.basename(str(event.get("network", "?")))
                row = nets[name]
                row["finals"].append((float(event.get("minutes") or 0.0), int(event.get("epochs") or 0)))
                row["last_t"] = t if row["last_t"] is None else max(row["last_t"], t)
            elif kind == "episode":
                run_info["episodes"] += 1
                run_info["episode_s"].append(float(event.get("seconds", 0.0) or 0.0))
            elif kind == "run_end":
                run_info["run_s"] = event.get("seconds")
    return nets, run_info


def read_log(run):
    """``({network: {epoch budget: epochs run}}, peak allocated GB)`` from ``logs/rank0.log``."""
    best = {}
    peak = 0.0
    path = os.path.join(run, "logs", "rank0.log")
    if not os.path.exists(path):
        return {}, None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "| Epoch " in line:
                match = EPOCH_LINE.search(line)
                if match:
                    key = (match.group(1), int(match.group(2)), int(match.group(4)))
                    best[key] = max(best.get(key, 0), int(match.group(3)))
            elif "[heartbeat]" in line:
                match = PEAK_ALLOC.search(line)
                if match:
                    peak = max(peak, float(match.group(1)))
    epochs = defaultdict(lambda: defaultdict(int))
    for (name, _step, budget), ran in best.items():
        epochs[name][budget] += ran
    return epochs, (peak or None)


def read_power(run):
    """``[(datetime, watts)]`` from ``gpu_samples.csv``; repeated headers (requeues) are skipped."""
    path = os.path.join(run, "gpu_samples.csv")
    if not os.path.exists(path):
        return []
    samples = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for row in csv.reader(fh):
            if len(row) < 4 or row[0].strip().startswith("timestamp"):
                continue
            try:
                stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                samples.append((stamp, float(row[3])))
            except ValueError:
                continue
    samples.sort()
    return samples


def energy_wh(samples, start=None, end=None):
    """Trapezoid integral of board power over ``[start, end]``; gaps over 5 s are not bridged."""
    total = 0.0
    previous = None
    for stamp, watts in samples:
        if (start and stamp < start) or (end and stamp > end):
            previous = None
            continue
        if previous is not None:
            dt = (stamp - previous[0]).total_seconds()
            if 0 < dt <= MAX_SAMPLE_GAP_S:
                total += 0.5 * (watts + previous[1]) * dt
        previous = (stamp, watts)
    return total / 3600.0


def summarize(arg):
    run = resolve(arg)
    if run is None:
        return None
    manifest = read_manifest(run)
    nets, run_info = read_events(run)
    epochs, peak = read_log(run)
    samples = read_power(run)
    job = os.path.basename(run.rstrip("/"))
    out = {"run": job, "path": run, "profile": manifest["profile"], "gpus": manifest["gpus"],
           "host": manifest["host"], "run_h": round(run_info["run_s"] / 3600.0, 3) if run_info["run_s"] else None,
           "peak_alloc_gb": peak, "episodes": run_info["episodes"],
           "episode_median_min": (round(statistics.median(run_info["episode_s"]) / 60.0, 1)
                                  if run_info["episode_s"] else None),
           "energy_wh": round(energy_wh(samples), 1) if samples else None,
           "mean_w": (round(statistics.mean(w for _, w in samples), 1) if samples else None),
           "networks": []}
    for name, row in sorted(nets.items()):
        steps = max(1.0, row["steps"])
        finals = row["finals"]
        final_min = sum(m for m, _ in finals)
        net = {"network": name, "steps": int(row["steps"]), "cuts": int(row["cuts"]),
               "walk_min": round(row["step_s"] / 60.0, 1),
               "ft_share": round(row["ft_s"] / row["step_s"], 4) if row["step_s"] else None,
               "per_step_s": {key: round(row[key] / steps, 3) for key in
                              ("ft_s", "prune_s", "eval_s", "features_s", "results_s")},
               "between_steps_s": round(statistics.median(row["gaps"]), 3) if row["gaps"] else None,
               # SPECTRA_TIME_DECIDE runs only: mean policy decision (actor or heuristic) per call.
               "decide_ms": (round(1000.0 * row["decide_s"] / row["decide_n"], 3)
                             if row["decide_n"] else None),
               "decisions": int(row["decide_n"]),
               "epochs_by_budget": dict(sorted(epochs.get(name, {}).items())),
               "finals": len(finals), "final_min": round(final_min, 1),
               "gpu_h": round((row["step_s"] + 60.0 * final_min) / 3600.0, 3)}
        if samples and manifest["started"] and row["first_t"] is not None:
            start = manifest["started"] + timedelta(seconds=row["first_t"])
            end = manifest["started"] + timedelta(seconds=row["last_t"])
            net["energy_wh"] = round(energy_wh(samples, start, end), 1)
        out["networks"].append(net)
    return out


def _print(summary):
    gpus = ",".join(summary["gpus"]) or "?"
    print(f"=== {summary['run']} {summary['profile']} | {gpus} @ {summary['host'].split('.')[0]} | "
          f"run {summary['run_h']} h | peak alloc {summary['peak_alloc_gb']} GB"
          + (f" | {summary['episodes']} episodes, median {summary['episode_median_min']} min"
             if summary["episodes"] else "")
          + (f" | {summary['energy_wh']} Wh at mean {summary['mean_w']} W" if summary["energy_wh"] else ""))
    for net in summary["networks"]:
        per = net["per_step_s"]
        epochs = " ".join(f"{ran}@{budget}" for budget, ran in net["epochs_by_budget"].items()) or "-"
        print(f"  {net['network'][:40]:40s} cuts {net['cuts']:4d}/{net['steps']:<4d} walk {net['walk_min']:7.1f} min "
              f"(FT {100 * (net['ft_share'] or 0):5.1f}%) | per step FT {per['ft_s']:6.1f}s prune {per['prune_s']:.2f}s "
              f"eval {per['eval_s']:.2f}s features {per['features_s']:.2f}s between {net['between_steps_s']}s "
              f"decide {net['decide_ms'] if net['decide_ms'] is not None else '-'} ms | "
              f"epochs run {epochs} | finals {net['finals']} x {net['final_min'] / max(1, net['finals']):.1f} min "
              f"| {net['gpu_h']} GPU-h" + (f" | {net['energy_wh']} Wh" if "energy_wh" in net else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="run directories or job ids")
    parser.add_argument("--jsonl", help="append one JSON object per run here")
    args = parser.parse_args(argv)
    for arg in args.runs:
        summary = summarize(arg)
        if summary is None:
            print(f"=== {arg}: no run directory under {run_roots()}")
            continue
        _print(summary)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
