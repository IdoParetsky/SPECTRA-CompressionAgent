#!/usr/bin/env python
"""
Speed bench for the final fine-tune's CUDA-graph path (``SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH``; queue
"FT cost", wave 28 G0). Speed and parity only, never a TEST row.

Each saved pre-fine-tune candidate (``<run>/traj_models/<stem>``) gets the final recipe (SGD lr 0.1,
momentum 0.9, wd 5e-4, cosine per epoch, clip 1, last epoch kept) through
``ClassificationHandler.train_model`` on synthetic CIFAR-shaped uint8 data fed by the GPU crop+flip
loader (50 000 images, so an epoch has the real step count). Eager and graphed at each batch size get a
1-epoch and a 3-epoch run on fresh copies: steady epoch = (t3 - t1) / 2, fixed cost = t1 - steady.
``--compile`` adds ``torch.compile`` (default and ``reduce-overhead``) on the first candidate at the first
batch size, reported only.

Parity: the gradients and BatchNorm buffers after one step from the same weights and batch, eager
against :class:`TrainGraph`, in FP32 with TF32 off and cuDNN deterministic; and the epoch loss and weights
after one epoch from the same start and data order, with the timed runs' settings.

    python scripts/bench_final_ft_graph.py <stem> [<stem> ...] [--batches 128,256] [--compile] [--out f.jsonl]
"""
import argparse
import copy
import json
import os
import socket
import sys
import time
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO, os.path.join(REPO, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

RECIPE = {"SPECTRA_FT_OPTIM": "sgd", "SPECTRA_FT_SGD_LR": "0.1", "SPECTRA_FT_MOMENTUM": "0.9",
          "SPECTRA_FT_WD": "5e-4", "SPECTRA_FT_COSINE": "1", "SPECTRA_FT_SCHEDULE": "", "SPECTRA_FT_MIXUP": "0",
          "SPECTRA_FT_LABEL_SMOOTH": "0", "SPECTRA_FT_KD": "0"}
OFF = ("SPECTRA_AMP", "SPECTRA_CHANNELS_LAST", "SPECTRA_SKIP_FT_GC", "SPECTRA_FT_AUTOAUG")
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
N_TRAIN = 50000
PARITY_BAR = 1e-3
RECORDS = []


def _init(device):
    from src.Configuration.ConfigurationValues import ConfigurationValues
    from src.Configuration.StaticConf import StaticConf
    import src.run_recorder as run_recorder
    if StaticConf.get_instance() is None:
        StaticConf(ConfigurationValues(
            device=device, test_name="bench-final-ft-graph", input_dict={}, compression_rates_dict={0: 1.0},
            runtime_limit=60, num_epochs=1, train_compressed_layer_only=False, allowed_acc_reduction=5,
            discount_factor=0.99, learning_rate=1e-3, rollout_limit=10, passes=1, prune=True, seed=42,
            n_splits=0, train_split=0.7, val_split=0.2, database_dict={}, actor_checkpoint_path=None,
            critic_checkpoint_path=None, save_pruned_checkpoints=False, test_ts="ts"))
    run_recorder.record = lambda kind, **fields: RECORDS.append((kind, fields))
    os.environ.update(RECIPE)
    for key in OFF:
        os.environ.pop(key, None)


def load(stem):
    """``(pre-fine-tune candidate on CPU, number of classes, label)``."""
    from bench_deploy import load_template
    from src import traj_models
    with open(stem + ".json", encoding="utf-8") as fh:
        doc = json.load(fh)
    template, _shape = load_template(doc["network"], doc)
    model, doc = traj_models.load_candidate(template, stem)
    classes = [m for m in model.modules() if isinstance(m, nn.Linear)][-1].out_features
    return model.cpu(), classes, os.path.basename(stem)


def synthetic_loader(batch, classes, device, seed=0):
    from src.utils import GpuCropFlipLoader
    rng = np.random.default_rng(seed)
    base = SimpleNamespace(data=rng.integers(0, 256, (N_TRAIN, 32, 32, 3), dtype=np.uint8),
                           targets=rng.integers(0, classes, N_TRAIN).tolist())
    return GpuCropFlipLoader(base, batch, CIFAR_MEAN, CIFAR_STD, device)


def fine_tune(model, loader, epochs, graph, compile_mode=""):
    """Wall seconds of one ``train_model`` call on a fresh copy, and the trained copy."""
    from src.ModelHandlers.ClassificationHandler import ClassificationHandler
    net = copy.deepcopy(model)
    if compile_mode:
        net = torch.compile(net, mode=None if compile_mode == "default" else compile_mode)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ClassificationHandler(net, nn.CrossEntropyLoss()).train_model(
        loader, allow_reinit_retry=False, max_epochs=epochs, patience=epochs + 1, keep_last=True,
        **({"cuda_graph": True} if graph else {}))
    torch.cuda.synchronize()
    return time.perf_counter() - t0, net


def last_loss():
    return next((f.get("best_loss") for kind, f in reversed(RECORDS) if kind == "finetune"), None)


def time_config(model, loader, graph, compile_mode=""):
    t1, _ = fine_tune(model, loader, 1, graph, compile_mode)
    t3, _ = fine_tune(model, loader, 3, graph, compile_mode)
    steady = (t3 - t1) / 2.0
    return {"t1_s": round(t1, 3), "t3_s": round(t3, 3), "epoch_s": round(steady, 3),
            "fixed_s": round(t1 - steady, 3), "ms_step": round(1000.0 * steady / len(loader), 3),
            "ft100_min": round((t1 - steady + 100.0 * steady) / 60.0, 2)}


def grad_parity(model, loader, device):
    """Max relative gradient and buffer difference after one step, eager against TrainGraph."""
    from src.ModelHandlers.ClassificationHandler import TrainGraph
    saved = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(0)
        x, y = next(iter(loader))
        results = []
        for graphed in (False, True):
            net = copy.deepcopy(model).float().to(device).train()
            for p in net.parameters():
                p.requires_grad_(True)
            forward = TrainGraph(net) if graphed else net
            nn.CrossEntropyLoss()(forward(x), y.long()).backward()
            results.append(([p.grad.detach().clone() for p in net.parameters()],
                            [b.detach().clone().float() for b in net.buffers()]))
            if graphed:
                forward.release()
        (g0, b0), (g1, b1) = results

        def worst(a, b):
            return max((float((u - v).abs().max() / (u.abs().max() + 1e-12)) for u, v in zip(a, b)), default=0.0)

        return worst(g0, g1), worst(b0, b1)
    finally:
        (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
         torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = saved


def epoch_parity(model, loader):
    """Epoch loss eager / graphed and the max relative weight difference after one epoch, same data order."""
    out = []
    for graphed in (False, True):
        torch.manual_seed(1)
        _, net = fine_tune(model, loader, 1, graphed)
        out.append((last_loss(), {k: v.detach().float().cpu() for k, v in net.state_dict().items()}))
    (l0, s0), (l1, s1) = out
    wdiff = max(float((s0[k] - s1[k]).abs().max() / (s0[k].abs().max() + 1e-12))
                for k in s0 if s0[k].is_floating_point())
    return l0, l1, wdiff


def environment(device):
    props = torch.cuda.get_device_properties(device)
    return {"host": socket.gethostname(), "gpu": props.name, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "slurm_job": os.environ.get("SLURM_JOB_ID")}


def emit(row, out):
    print("ROW " + json.dumps(row, sort_keys=True), flush=True)
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def run(args):
    device = torch.device("cuda")
    _init(device)
    env = environment(device)
    print("ENV " + json.dumps(env, sort_keys=True), flush=True)
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    summary = {}
    for i, stem in enumerate(args.stems):
        model, classes, label = load(stem)
        params = sum(p.numel() for p in model.parameters())
        base = synthetic_loader(batches[0], classes, device)
        fine_tune(model, base, 1, False)  # cuDNN autotune, allocator and loader materialisation
        for batch in batches:
            loader = base if batch == batches[0] else base.with_batch_size(batch)
            for graphed in (False, True):
                row = dict(env, stem=label, params=params, batch=batch, mode="graph" if graphed else "eager",
                           steps=len(loader), **time_config(model, loader, graphed))
                summary[(label, batch, row["mode"])] = row
                emit(row, args.out)
        g_rel, b_rel = grad_parity(model, base, device)
        l0, l1, w_rel = epoch_parity(model, base)
        parity = dict(env, stem=label, mode="parity", batch=batches[0], grad_rel=g_rel, buffer_rel=b_rel,
                      epoch_loss_eager=l0, epoch_loss_graph=l1, weight_rel_1ep=w_rel)
        summary[(label, "parity")] = parity
        emit(parity, args.out)
        if args.compile and i == 0:
            for mode in ("default", "reduce-overhead"):
                try:
                    row = dict(env, stem=label, params=params, batch=batches[0], mode=f"compile-{mode}",
                               steps=len(base), **time_config(model, base, False, mode))
                except Exception as error:  # noqa: BLE001 - reported, the bench goes on
                    row = dict(env, stem=label, batch=batches[0], mode=f"compile-{mode}",
                               error=f"{type(error).__name__}: {str(error)[:300]}")
                emit(row, args.out)
        del model, base
        torch.cuda.empty_cache()
    report(summary, args.stems, batches)


def report(summary, stems, batches):
    print("=== steady epoch (s) / ms per step / fixed (s) / 100-epoch projection (min)")
    for stem in stems:
        label = os.path.basename(stem)
        for batch in batches:
            e, g = summary[(label, batch, "eager")], summary[(label, batch, "graph")]
            print(f"{label[:60]:60s} b{batch:<4d} eager {e['epoch_s']:6.2f} s {e['ms_step']:6.2f} ms "
                  f"| graph {g['epoch_s']:6.2f} s {g['ms_step']:6.2f} ms fixed {g['fixed_s']:5.2f} s "
                  f"| x{e['epoch_s'] / max(g['epoch_s'], 1e-9):.2f} | 100 ep {e['ft100_min']:.1f} -> {g['ft100_min']:.1f} min")
        p = summary[(label, "parity")]
        print(f"{'':60s} parity grad {p['grad_rel']:.2e} buffers {p['buffer_rel']:.2e} | 1-epoch loss "
              f"{p['epoch_loss_eager']} vs {p['epoch_loss_graph']} | weights {p['weight_rel_1ep']:.2e}")
    first = os.path.basename(stems[0])
    e, g = summary[(first, batches[0], "eager")], summary[(first, batches[0], "graph")]
    ratio = e["epoch_s"] / max(g["epoch_s"], 1e-9)
    call = "GRAPH-WIN" if ratio >= 1.5 else ("GRAPH-PARTIAL" if ratio >= 1.2 else "GRAPH-NO-GAIN")
    parity = summary[(first, "parity")]["grad_rel"]
    print(f"=== CALL ({first}, batch {batches[0]}): {call} at x{ratio:.2f}; parity "
          f"{'PARITY-FAIL' if parity > PARITY_BAR else 'ok'} (grad {parity:.2e}, bar {PARITY_BAR:g})")
    if len(batches) > 1:
        g2 = summary[(first, batches[1], "graph")]
        gain = g["epoch_s"] / max(g2["epoch_s"], 1e-9)
        print(f"=== G2 condition (graphed b{batches[1]} >= 1.3x graphed b{batches[0]}): "
              f"{'met' if gain >= 1.3 else 'not met'} at x{gain:.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stems", nargs="+", help="<run>/traj_models/<net>__<label>__step<N> (no extension)")
    parser.add_argument("--batches", default="128,256")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("bench_final_ft_graph needs a CUDA device")
    torch.backends.cudnn.benchmark = True
    run(args)


if __name__ == "__main__":
    main()
