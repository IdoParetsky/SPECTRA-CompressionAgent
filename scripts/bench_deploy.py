#!/usr/bin/env python
"""
Deployment cost of saved TRAJ candidates (``<run>/traj_models``) on the device this runs on.

Per architecture (a fine-tuned copy has its walk twin's shapes, so it is timed once): params and MACs
from the env's counters, state-dict size, and per batch size the latency (median / p90 / mean / std of
per-batch times), throughput, peak allocated memory and board energy per image.

Protocol (docs/paper/EFFICIENCY_AND_TRANSFER.md, measurement protocol): eval mode, synthetic input on
the device at the dataset's resolution, FP32 with TF32 off so Turing and Ampere-or-newer cards run the
same arithmetic, cudnn.benchmark on, ``--warmup`` iterations, then at least ``--min-iters`` timed
iterations and ``--min-seconds``, each timed with CUDA events. Energy is nvidia-smi ``power.draw``
sampled every 100 ms over the timed loop.

    python scripts/bench_deploy.py <run dir> ... [--batches 1,64,256] [--min-seconds 10] [--out f.jsonl]

``--model FILE:DATASET[:LABEL[:GROUP]]`` also times a whole pickled ``nn.Module`` (another method's pruned
network, e.g. Torch-Pruning's output); rows sharing GROUP print speedups against the one labelled ``origin``.
The module's own package must be importable (``PYTHONPATH``).
"""
import argparse
import glob
import hashlib
import io
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

DATASETS = {"cifar-10": (10, (3, 32, 32)), "cifar-100": (100, (3, 32, 32)), "svhn": (10, (3, 32, 32)),
            "imagenet": (1000, (3, 224, 224))}


def _stem_parts(stem):
    """``(network file, label, step, suffix)`` from ``<net>__<label>__step<N>[__ft<E>]``."""
    base = os.path.basename(stem)
    head, _, tail = base.rpartition("__step")
    net, _, label = head.partition("__")
    step, _, suffix = tail.partition("__")
    return net, label, step, suffix


def candidates(run):
    """Saved candidates of one run, grouped ``{network path: [(stem, doc)]}``."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run, "traj_models", "*.json"))):
        stem = path[:-len(".json")]
        if not os.path.exists(stem + ".pt"):
            continue
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        out.setdefault(doc.get("network") or _stem_parts(stem)[0], []).append((stem, doc))
    return out


def load_template(net_path, doc):
    """``(original network, per-sample input shape)`` through the env's own loader."""
    from src import utils
    input_json = (doc.get("recipe_env") or {}).get("SPECTRA_INPUT")
    with open(input_json, encoding="utf-8") as fh:
        entry = json.load(fh)[net_path]
    arch, script, dataset = entry[:3]
    kwargs = entry[3] if len(entry) > 3 else {}
    if dataset not in DATASETS:
        raise KeyError(f"no input shape known for dataset {dataset!r}")
    num_classes, shape = DATASETS[dataset]
    model = utils.load_model_from_script(arch, dataset, script, net_path, kwargs, num_classes, shape)
    return model, shape


class PowerSampler:
    """nvidia-smi ``power.draw`` every ``interval_ms`` while the ``with`` block runs; no-op off CUDA."""

    def __init__(self, device, interval_ms=100):
        self.device, self.interval_ms, self.watts = device, interval_ms, []
        self.proc = self.thread = None

    def _query(self):
        cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
               "-lms", str(self.interval_ms)]
        if torch.cuda.device_count() > 1:
            uuid = str(getattr(torch.cuda.get_device_properties(self.device), "uuid", ""))
            if uuid:
                cmd += ["-i", uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"]
        return cmd

    def __enter__(self):
        if self.device.type != "cuda":
            return self
        try:
            self.proc = subprocess.Popen(self._query(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True)
        except OSError:
            return self
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.watts.append(float(line.strip().split(",")[0]))
            except ValueError:
                continue

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait(timeout=5)
            self.thread.join(timeout=5)
        return False

    def mean(self):
        return statistics.mean(self.watts) if self.watts else None


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_loop(model, x, device, warmup, min_iters, min_seconds, max_iters):
    """Per-batch times in ms and the wall seconds of the timed loop."""
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        _sync(device)
        times = []
        started = time.perf_counter()
        while len(times) < max_iters and (len(times) < min_iters or time.perf_counter() - started < min_seconds):
            if device.type == "cuda":
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                model(x)
                end.record()
                end.synchronize()
                times.append(begin.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - t0) * 1e3)
        return times, time.perf_counter() - started


def _state_bytes(model, half=False):
    buffer = io.BytesIO()
    state = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save(state, buffer)
    return buffer.tell()


def _arch_key(doc):
    return hashlib.sha1(json.dumps(doc.get("arch"), sort_keys=True).encode()).hexdigest()[:12]


def environment(device):
    info = {"device": str(device), "host": socket.gethostname().split(".")[0], "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "tf32": False, "cudnn_benchmark": True,
            "cpu_threads": torch.get_num_threads()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        try:
            info["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, text=True).strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            info["driver"] = None
    return info


def bench_model(model, shape, device, batches, args, env):
    from src import utils
    model = model.to(device).eval()
    row = {"params": int(utils.calc_num_parameters(model)),
           "macs": float(utils.calc_flops(model, shape, device)),
           "state_mb_fp32": round(_state_bytes(model) / 2 ** 20, 3),
           "state_mb_fp16": round(_state_bytes(model, half=True) / 2 ** 20, 3), "by_batch": {}}
    with PowerSampler(device) as idle:
        time.sleep(args.idle_seconds if device.type == "cuda" else 0)
    row["idle_w"] = idle.mean()
    for bs in batches:
        x = torch.randn(bs, *shape, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        with PowerSampler(device) as power:
            times, wall = time_loop(model, x, device, args.warmup, args.min_iters, args.min_seconds,
                                    args.max_iters)
        ordered = sorted(times)
        median = statistics.median(ordered)
        watts = power.mean()
        row["by_batch"][str(bs)] = {
            "iters": len(times), "seconds": round(wall, 2),
            "latency_ms_median": round(median, 4),
            "latency_ms_p90": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 4),
            "latency_ms_mean": round(statistics.mean(times), 4),
            "latency_ms_std": round(statistics.pstdev(times), 4),
            "throughput_img_s": round(bs * 1e3 / median, 1),
            "peak_alloc_mb": (round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1)
                              if device.type == "cuda" else None),
            "power_w_mean": round(watts, 1) if watts else None,
            "energy_mj_per_img": (round(1e3 * watts * wall / (len(times) * bs), 4) if watts else None)}
    return row


def run(args):
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.threads:
        torch.set_num_threads(args.threads)
    env = environment(device)
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    rows = []
    for run_dir in args.runs:
        for net_path, items in candidates(run_dir).items():
            try:
                template, shape = load_template(net_path, items[0][1])
            except Exception as error:  # noqa: BLE001 - one unloadable network must not cost the rest
                print(f"[bench] {os.path.basename(net_path)}: template failed: {type(error).__name__}: {error}")
                continue
            from src import traj_models
            timed = {}
            for stem, doc in items:
                net, label, step, suffix = _stem_parts(stem)
                key = _arch_key(doc)
                if key not in timed:
                    model, _ = traj_models.load_candidate(template, stem)
                    timed[key] = bench_model(model, shape, device, batches, args, env)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                point = doc.get("point") or {}
                record = {"run": os.path.basename(run_dir.rstrip("/")), "network": os.path.basename(net_path),
                          "label": label, "step": step, "suffix": suffix, "arch_key": key,
                          "param_ratio": point.get("param"), "flop_ratio": point.get("flop"),
                          "test_origin": point.get("test_origin"), "test_walk": point.get("test_acc"),
                          "test_final": (doc.get("final_ft") or {}).get("test_final"), **timed[key], **env}
                rows.append(_emit(record, args.out))
    for spec in args.model or []:
        path, dataset, label, group = (spec.split(":") + ["", ""])[:4]
        model = torch.load(path, map_location="cpu", weights_only=False)
        timed = bench_model(model, DATASETS[dataset][1], device, batches, args, env)
        del model
        record = {"run": os.path.basename(os.path.dirname(path)), "network": group or os.path.basename(path),
                  "label": label or os.path.basename(path), "step": None, "suffix": "", "arch_key": None,
                  "file": path, **timed, **env}
        rows.append(_emit(record, args.out))
    _print(rows, batches)
    return rows


def _emit(record, out):
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    return record


def _print(rows, batches):
    by_net = {}
    for row in rows:
        by_net.setdefault((row["run"], row["network"]), []).append(row)
    for (run_name, net), items in by_net.items():
        origin = next((r for r in items if r["label"] == "origin"), None)
        print(f"=== {run_name} {net} on {items[0].get('gpu', items[0]['device'])}")
        seen = set()
        for r in sorted(items, key=lambda r: -r["params"]):
            key = r["arch_key"] or r.get("file")
            if key in seen:
                continue
            seen.add(key)
            cells = []
            for bs in batches:
                b = r["by_batch"][str(bs)]
                speed = (origin["by_batch"][str(bs)]["latency_ms_median"] / b["latency_ms_median"]
                         if origin else None)
                cells.append(f"bs{bs} {b['latency_ms_median']:.3f} ms {b['throughput_img_s']:.0f} img/s"
                             + (f" x{speed:.2f}" if speed else "")
                             + (f" {b['energy_mj_per_img']:.3f} mJ/img" if b["energy_mj_per_img"] else ""))
            macs_x = origin["macs"] / r["macs"] if origin and r["macs"] else None
            print(f"  {r['label'][:22]:22s} {r['params'] / 1e6:7.3f} M params {r['macs'] / 1e6:8.2f} M MACs"
                  + (f" (MACs x{macs_x:.2f})" if macs_x else "") + " | " + " | ".join(cells))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="*", help="run directories with traj_models/")
    parser.add_argument("--model", action="append", help="FILE:DATASET[:LABEL[:GROUP]] of a pickled nn.Module")
    parser.add_argument("--batches", default="1,64,256")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--min-iters", type=int, default=300)
    parser.add_argument("--min-seconds", type=float, default=10.0)
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--idle-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    parser.add_argument("--out", help="append one JSON row per candidate here")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
