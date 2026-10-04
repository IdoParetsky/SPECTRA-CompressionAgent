#!/usr/bin/env python
"""
Allocation headroom probe (A0, docs/SITTING_GPU_QUEUE.md "A0"): at a fixed overall size, does *how many*
channels each coupled group keeps change recovered accuracy, once *which* channels is fixed to SPECTRA's
L1 group vote?

S0 (scripts/selection_probe.py) fixed the allocation and varied the selection: after 40 epochs nothing beat
L1. A0 fixes the selection and varies the allocation. Every allocation cuts each coupled group once with the
walk's own structural edit, scaled to the same kept-parameter fraction, then recovers with the walk's recipe A.

  uniform   every group keeps the same fraction: what mild and the constant 0.8 agent do. Its
            --uniform_seeds fine-tune seeds give the noise every other allocation has to clear
  sens      keep ∝ s^α (α = --alpha): groups whose cut hurts more keep more. Li et al. (ICLR 2017)'s
            per-layer sensitivity analysis, made continuous. sens2 uses 2α
  anti      keep ∝ s^-α: the reverse, the lever's other side
  random    keep ∝ exp(σ z), z ~ N(0, 1), σ = --sigma: --random_draws allocations

s = a group's calibration-loss increase when it alone is cut to --sens_keep (train batches, no fine-tune).
Keeps are clip(c · weight, --min_keep, 1), with c found by bisection. Uniform is cut first, as close to --keep
as integer widths allow; every other allocation is bisected to uniform's realised params, so they compare
at equal size even on nets whose groups are a few channels wide. ``--match flops`` bisects on kept FLOPs
instead: on VGG-16, allocations matched on params kept up to twice uniform's FLOPs (ledger §205).

    python scripts/allocation_probe.py --checkpoint NET.pth --script thin_res_net.py --arch resnet56 \
        --width 4 --dataset cifar-10 --keep 0.6 0.35 --budgets 0 bn 40

Writes to $SPECTRA_RUN_DIR/results/: allocation_probe.jsonl (one row per allocation, fine-tune seed and
budget, then one summary row per keep and budget) and allocation_keeps.jsonl (each allocation's per-group
keep and realised width). Accuracies are on protocol P's val/test halves. They measure a lever; they are
never a method's TEST row.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import selection_probe as sp  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.logging_utils as logging_utils  # noqa: E402
import src.utils as utils  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src.Configuration.StaticConf import StaticConf  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
from src.NetworkEnv import prune_current_model  # noqa: E402

POWERS = {"sens": 1.0, "sens2": 2.0, "anti": -1.0}


def cut_alloc(model, plan, keeps, input_shape):
    """One structural L1 cut per group at its own keep; a keep of 1 leaves the group alone."""
    mwr = ModelWithRows(copy.deepcopy(model))
    modes = []
    for key, _, row in plan:
        rate = float(keeps.get(key, 1.0))
        if rate >= 1.0 - 1e-9:
            modes.append("identity")
            continue
        mwr = prune_current_model(mwr, rate, row, quiet=True, record=False, input_shape=input_shape,
                                  importance="l1")
        modes.append((getattr(mwr, "last_prune_outcome", None) or {}).get("mode"))
        mwr = ModelWithRows(mwr.model)
    return mwr.model, modes


def group_widths(model):
    """Width of every coupled group, keyed like the plan (module names survive structural edits)."""
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    names = sp._module_names(mwr.model)
    return {sp.group_key(mwr.model, group, names): int(group.width) for group in groups}


@torch.no_grad()
def calib_loss(model, batches):
    model.eval()
    loss = nn.CrossEntropyLoss(reduction="sum")
    total = sum(float(loss(model(x), y)) for x, y in batches)
    return total / max(1, sum(int(y.numel()) for _, y in batches))


def group_sensitivity(model, plan, batches, keep, input_shape):
    """Calibration-loss increase when each group alone is cut to ``keep`` (no fine-tune)."""
    base = calib_loss(model, batches)
    out = {}
    for entry in plan:
        cut_model, _ = cut_alloc(model, [entry], {entry[0]: keep}, input_shape)
        out[entry[0]] = calib_loss(cut_model, batches) - base
        del cut_model
    return out, base


def allocation_weights(kind, plan, sens=None, alpha=0.5, sigma=0.35, rng=None):
    """Relative keep per group before scaling: 1 (uniform), s^±α (sens / sens2 / anti) or exp(σ z) (random)."""
    keys = [key for key, _, _ in plan]
    if kind == "uniform":
        return {key: 1.0 for key in keys}
    if kind == "random":
        rng = rng if rng is not None else np.random.default_rng(0)
        return {key: float(math.exp(sigma * rng.standard_normal())) for key in keys}
    if kind not in POWERS:
        raise ValueError(f"unknown allocation {kind!r}")
    positive = [max(0.0, float(sens[key])) for key in keys]
    floor = max(1e-6, 0.05 * statistics.median(positive))
    s = {key: max(float(sens[key]), floor) for key in keys}
    mid = statistics.median(s.values())
    return {key: (s[key] / mid) ** (POWERS[kind] * alpha) for key in keys}


def match_params(model, plan, weights, target, input_shape, params0, min_keep=0.1, iters=16, tol=0.003,
                 measure=None):
    """Keeps clip(c · w, min_keep, 1), with c bisected so the cut keeps ``target`` of the params.

    ``measure(cut_model)`` replaces the kept-params fraction when given (``--match flops``). Returns
    ``(keeps, kept, cut_model, modes)`` of the closest cut found. Widths are integers, so the kept fraction
    is a step function of c and may stop short of ``tol``.
    """
    def keeps_at(c):
        return {key: min(1.0, max(min_keep, c * w)) for key, w in weights.items()}

    lo, hi = 0.0, 1.0 / min(weights.values())
    best = None
    for _ in range(iters):
        c = 0.5 * (lo + hi)
        keeps = keeps_at(c)
        cut_model, modes = cut_alloc(model, plan, keeps, input_shape)
        frac = measure(cut_model) if measure is not None else utils.calc_num_parameters(cut_model) / params0
        if best is None or abs(frac - target) < abs(best[1] - target):
            best = (keeps, frac, cut_model, modes)
        else:
            del cut_model
        if abs(frac - target) <= tol:
            break
        if frac > target:
            hi = c
        else:
            lo = c
    return best


def summarize(rows, keep, budget, matched_tol=0.02, match_key="params_kept"):
    """The registered A0 call for one keep and budget (queue file "A0").

    bar = max(0.5, 2 × the uniform allocation's fine-tune-seed SD of val Δ). HEADROOM: sens, sens2 or the
    val-best random draw is ≥ bar above uniform on val **and** above it on test. HARM: every matched
    allocation is ≥ bar below uniform on val. FLAT otherwise. Allocations whose ``match_key`` fraction
    (params, or FLOPs under ``--match flops``) differs from uniform's by more than ``matched_tol`` are
    reported but never counted.
    """
    cell = [r for r in rows if r.get("keep") == keep and r.get("budget") == str(budget)]
    uni = [r for r in cell if r["alloc"] == "uniform"]
    if len(uni) < 2:
        return None
    mean_val = statistics.mean(r["d_val_pp"] for r in uni)
    mean_test = statistics.mean(r["d_test_pp"] for r in uni)
    sd = statistics.stdev(r["d_val_pp"] for r in uni)
    bar = max(0.5, 2.0 * sd)
    params_uni = statistics.mean(r["params_kept"] for r in uni)
    size_uni = statistics.mean(r[match_key] for r in uni)
    allocs = {}
    for r in cell:
        if r["alloc"] == "uniform":
            continue
        name = f"random{r['draw']}" if r["alloc"] == "random" else r["alloc"]
        allocs[name] = {"dval": round(r["d_val_pp"] - mean_val, 3), "dtest": round(r["d_test_pp"] - mean_test, 3),
                        "params": r["params_kept"], "flops": r["flops_kept"],
                        "matched": abs(r[match_key] - size_uni) <= matched_tol}
    candidates = [(name, v) for name, v in allocs.items() if name in ("sens", "sens2") and v["matched"]]
    randoms = [(name, v) for name, v in allocs.items() if name.startswith("random") and v["matched"]]
    out = {"uniform_dval": round(mean_val, 3), "uniform_dtest": round(mean_test, 3), "uniform_val_sd": round(sd, 3),
           "bar": round(bar, 3), "uniform_params": round(params_uni, 4), "match": match_key.split("_")[0],
           "allocs": allocs, "random_valbest": None}
    if randoms:
        best = max(randoms, key=lambda kv: kv[1]["dval"])
        out["random_valbest"] = best[0]
        candidates.append(best)
    matched = [v for v in allocs.values() if v["matched"]]
    if any(v["dval"] >= bar and v["dtest"] > 0 for _, v in candidates):
        out["call"] = "HEADROOM"
    elif matched and all(v["dval"] <= -bar for v in matched):
        out["call"] = "HARM"
    else:
        out["call"] = "FLAT"
    return out


def _budget(token: str):
    return "bn" if token == "bn" else int(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--script", required=True, help="Instantiation script path")
    parser.add_argument("--arch", required=True, help="Constructor name inside the script")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", default=None, help="Network label in the results (default: checkpoint stem)")
    parser.add_argument("--width", type=int, default=None, help="Width kwarg for thin-ResNet constructors")
    parser.add_argument("--keep", type=float, nargs="+", default=[0.6, 0.35],
                        help="Target fraction of the network's parameters (or FLOPs, --match flops) kept")
    parser.add_argument("--match", choices=("params", "flops"), default="params",
                        help="Size every allocation is bisected to and the matched check reads")
    parser.add_argument("--budgets", type=_budget, nargs="+", default=[0, "bn", 40],
                        help="Fine-tune epochs per allocation; 'bn' = BatchNorm re-estimation only")
    parser.add_argument("--uniform_seeds", type=int, default=3)
    parser.add_argument("--random_draws", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.35, help="Spread of the random allocations (log keep)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Sensitivity power of sens / anti")
    parser.add_argument("--sens_keep", type=float, default=0.5, help="Keep of the one-group sensitivity cut")
    parser.add_argument("--min_keep", type=float, default=0.1)
    parser.add_argument("--matched_tol", type=float, default=0.02,
                        help="Params-kept gap to uniform beyond which an allocation is not counted")
    parser.add_argument("--calib_batches", type=int, default=4, help="Train batches for the sensitivity")
    parser.add_argument("--bn_batches", type=int, default=50)
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stop patience (default SPECTRA_FINETUNE_PATIENCE, as the walk)")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random allocations")
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    logging_utils.setup()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    numeric = [b for b in args.budgets if b != "bn"]
    sp._ensure_conf(device, max(numeric) if numeric else 0)
    StaticConf.get_instance().conf_values.device = device
    tag = args.tag or Path(args.checkpoint).stem

    registry = utils.preload_datasets([args.dataset], args.train_split, args.val_split)
    optional = {"width": args.width} if args.width is not None else {}
    model = utils.load_model_from_script(
        args.arch, args.dataset, args.script, args.checkpoint, optional,
        num_classes=registry.num_classes(args.dataset),
        input_shape=registry.input_shape(args.dataset),
    ).to(device).eval()
    loaders = registry.loaders(args.dataset)
    input_shape = tuple(registry.input_shape(args.dataset))

    base = ClassificationHandler(copy.deepcopy(model), nn.CrossEntropyLoss())
    base_val, base_test = float(base.evaluate_model(loaders[1])), float(base.evaluate_model(loaders[2]))
    params0 = utils.calc_num_parameters(model)
    flops0 = utils.calc_flops(model, input_shape, device)
    del base

    plan = sp.cut_plan(model)
    utils.print_flush(
        f"Allocation probe {tag} ({args.dataset}) on {device}: val {base_val:.4f} test {base_test:.4f}, "
        f"{params0 / 1e6:.3f} M params, {len(plan)} groups / {sum(g.width for _, g, _ in plan)} channels; "
        f"keep {args.keep} (match {args.match}) budgets {args.budgets}; uniform x{args.uniform_seeds}, "
        f"random x{args.random_draws} "
        f"(sigma {args.sigma}), alpha {args.alpha}, sens_keep {args.sens_keep}, min_keep {args.min_keep}; "
        f"FT_AUG={os.environ.get('SPECTRA_FT_AUG', '0')} FT_AUG_GPU={os.environ.get('SPECTRA_FT_AUG_GPU', '0')} "
        f"VAL_FROM_TEST={os.environ.get('SPECTRA_VAL_FROM_TEST', '0')}")

    started = time.perf_counter()
    batches = sp.calibration_batches(loaders[0], args.calib_batches, device)
    sens, base_loss = group_sensitivity(model, plan, batches, args.sens_keep, input_shape)
    values = list(sens.values())
    utils.print_flush(
        f"Sensitivity at keep {args.sens_keep}: {len(values)} groups, median {statistics.median(values):.4f}, "
        f"min {min(values):.4f}, max {max(values):.4f} (calibration loss {base_loss:.4f}) "
        f"in {time.perf_counter() - started:.0f}s")

    out_dir = os.path.join(logging_utils.run_dir(), "results")
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "allocation_probe.jsonl")
    keeps_path = os.path.join(out_dir, "allocation_keeps.jsonl")
    with open(keeps_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"net": tag, "kind": "sensitivity", "sens_keep": args.sens_keep,
                                 "base_loss": base_loss, "sens": {"|".join(k): v for k, v in sens.items()}}) + "\n")

    specs = [("uniform", 0, list(range(args.uniform_seeds)))]
    specs += [(kind, 0, [0]) for kind in ("sens", "sens2", "anti")]
    specs += [("random", draw, [0]) for draw in range(args.random_draws)]
    measure = ((lambda cut: utils.calc_flops(cut, input_shape, device) / flops0)
               if args.match == "flops" else None)
    match_key = f"{args.match}_kept"
    rows = []
    for keep in args.keep:
        reference = None
        for kind, draw, seeds in specs:
            t0 = time.perf_counter()
            weights = allocation_weights(kind, plan, sens, args.alpha, args.sigma,
                                         np.random.default_rng([args.seed, draw]))
            target = keep if reference is None else reference
            keeps, kept, cut_model, modes = match_params(model, plan, weights, target, input_shape, params0,
                                                         args.min_keep, measure=measure)
            if kind == "uniform":
                reference = kept
            params_kept = utils.calc_num_parameters(cut_model) / params0
            flops_kept = utils.calc_flops(cut_model, input_shape, device) / flops0
            widths = group_widths(cut_model)
            with open(keeps_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"net": tag, "keep": keep, "alloc": kind, "draw": draw, "target": target,
                                         "match": args.match, "params_kept": params_kept, "flops_kept": flops_kept,
                                         "keeps": {"|".join(k): round(v, 4) for k, v in keeps.items()},
                                         "widths": {"|".join(k): widths.get(k) for k, _, _ in plan},
                                         "origin_widths": {"|".join(k): g.width for k, g, _ in plan},
                                         "modes": modes}) + "\n")
            for seed in seeds:
                for budget in args.budgets:
                    t1 = time.perf_counter()
                    val, test, _ = sp.recover(cut_model, budget, loaders, device, seed,
                                              patience=args.patience, bn_batches=args.bn_batches)
                    row = {"net": tag, "dataset": args.dataset, "keep": keep, "match": args.match, "alloc": kind,
                           "draw": draw, "ft_seed": seed, "budget": str(budget), "val": round(val, 5),
                           "test": round(test, 5),
                           "base_val": round(base_val, 5), "base_test": round(base_test, 5),
                           "d_val_pp": round((val - base_val) * 100, 3),
                           "d_test_pp": round((test - base_test) * 100, 3),
                           "params_kept": round(params_kept, 5), "flops_kept": round(flops_kept, 5),
                           "seconds": round(time.perf_counter() - t1, 1)}
                    rows.append(row)
                    with open(results_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    utils.print_flush(
                        f"[alloc] {tag} keep={keep} {kind}{draw}/s{seed} budget={budget}: "
                        f"val {row['d_val_pp']:+.2f} pp test {row['d_test_pp']:+.2f} pp | params x{params_kept:.3f} "
                        f"FLOPs x{flops_kept:.3f} | {row['seconds']:.0f}s")
            utils.print_flush(f"[alloc] {kind}{draw} at keep {keep} done in {time.perf_counter() - t0:.0f}s")
            del cut_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for budget in args.budgets:
            cell = summarize(rows, keep, budget, args.matched_tol, match_key)
            if cell is None:
                continue
            parts = " ".join(f"{name} {v['dval']:+.2f}/{v['dtest']:+.2f}{'' if v['matched'] else '(unmatched)'}"
                             for name, v in cell["allocs"].items())
            utils.print_flush(
                f"[alloc-call] {tag} keep={keep} match={args.match} budget={budget}: "
                f"uniform val {cell['uniform_dval']:+.2f} "
                f"(sd {cell['uniform_val_sd']:.2f}) test {cell['uniform_dtest']:+.2f} | bar {cell['bar']:.2f} | "
                f"vs uniform, val/test: {parts} | random val-best {cell['random_valbest']} | {cell['call']}")
            with open(results_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "summary", "net": tag, "keep": keep, "budget": str(budget),
                                         **cell}) + "\n")
    utils.print_flush(f"Wrote {results_path} and {keeps_path} in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
