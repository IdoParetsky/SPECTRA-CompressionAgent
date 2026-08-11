"""
Short diagnostic: how much post-prune fine-tuning recovers accuracy at each rate.

Answers what the 8h RL run learns only slowly: for a given network and compression
rate, does ``num_epochs`` of fine-tuning bring accuracy back inside the allowed drop?
If not, longer agent runs cannot learn useful compression either.

Default mode matches one agent *step* (single row prune + fine-tune). Optional
``--scope pass`` compresses every prunable row once before one fine-tune.

    python scripts/recovery_probe.py \
        --checkpoint /path/to/net.pt \
        --script /path/to/thin_res_net.py --arch resnet20 \
        --dataset cifar-10 \
        --rates 0.9 0.8 0.7 0.6 \
        --epochs 0 10 20 40

Results: ``$SPECTRA_RUN_DIR/results/recovery_probe.jsonl`` (+ stdout matrix).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.logging_utils as logging_utils
import src.run_recorder as recorder
import src.utils as utils
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.ClassificationHandler import ClassificationHandler
from src.NetworkEnv import build_param_names_to_keep_trainable, prune_current_model


def _ensure_conf(device, num_epochs: int, train_compressed_layer_only: bool):
    if StaticConf.get_instance() is not None:
        return
    StaticConf(ConfigurationValues(
        device=device, test_name="recovery-probe", input_dict={},
        compression_rates_dict={0: 1.0}, runtime_limit=1, num_epochs=num_epochs,
        train_compressed_layer_only=train_compressed_layer_only,
        allowed_acc_reduction=5, discount_factor=0.99, learning_rate=1e-3,
        rollout_limit=None, passes=1, prune=True, seed=0, n_splits=0,
        train_split=0.7, val_split=0.2, database_dict={},
        actor_checkpoint_path=None, critic_checkpoint_path=None,
        save_pruned_checkpoints=False, test_ts="probe",
    ))


def _row_indices(model_with_rows):
    return sorted(model_with_rows.row_to_main_layer.keys())


def _resolve_rows(model_with_rows, row_spec: str):
    """``all`` | ``mid`` | comma-separated ints | single int."""
    rows = _row_indices(model_with_rows)
    if not rows:
        raise RuntimeError("Model has no prunable rows")
    if row_spec == "all":
        return rows
    if row_spec == "mid":
        return [rows[len(rows) // 2]]
    out = []
    for part in row_spec.split(","):
        part = part.strip()
        if part == "mid":
            out.append(rows[len(rows) // 2])
        else:
            idx = int(part)
            if idx not in model_with_rows.row_to_main_layer:
                raise ValueError(f"row {idx} not in {rows}")
            out.append(idx)
    return out


def evaluate(handler, loader) -> float:
    return float(handler.evaluate_model(loader))


def _apply_finetune_mode(handler, mwr, row_idx, train_compressed_layer_only: bool):
    if train_compressed_layer_only:
        keep = build_param_names_to_keep_trainable(mwr, row_idx)
        handler.freeze_all_layers_but_pruned(keep)
    else:
        handler.unfreeze_all_layers()


def probe_one(model, train_loader, val_loader, test_loader, rates, epoch_budgets,
              train_compressed_layer_only: bool, device, allowed_drop: float,
              scope: str, row_spec: str):
    """
    For each (rate, epochs): clone the pristine model, prune (one row or full pass),
    fine-tune under the chosen freeze policy, measure val (+ test) accuracy.
    """
    baseline_handler = ClassificationHandler(copy.deepcopy(model).to(device),
                                             torch.nn.CrossEntropyLoss())
    baseline_acc = evaluate(baseline_handler, val_loader)
    baseline_test = evaluate(baseline_handler, test_loader) if test_loader is not None else None
    utils.print_flush(
        f"Baseline val={baseline_acc:.4f}"
        + (f" test={baseline_test:.4f}" if baseline_test is not None else ""))

    out = []
    for rate in rates:
        for epochs in epoch_budgets:
            StaticConf.get_instance().conf_values.num_epochs = epochs
            StaticConf.get_instance().conf_values.train_compressed_layer_only = (
                train_compressed_layer_only)

            working = copy.deepcopy(model).to(device)
            mwr = ModelWithRows(working)
            rows_to_prune = _resolve_rows(mwr, "all" if scope == "pass" else row_spec)
            prune_modes = []
            started = time.perf_counter()
            last_row = rows_to_prune[-1]

            for row_idx in rows_to_prune:
                mwr = prune_current_model(mwr, rate, row_idx)
                prune_modes.append(getattr(mwr, "last_prune_outcome", {}).get("mode"))
                mwr = ModelWithRows(mwr.model)

            handler = ClassificationHandler(mwr.model, torch.nn.CrossEntropyLoss())
            # Agent freezes relative to the row just pruned; for a full pass that is the
            # last row. Full-network FT always unfreezes everything.
            _apply_finetune_mode(handler, mwr, last_row, train_compressed_layer_only)
            if epochs > 0:
                handler.train_model(train_loader)

            new_acc = evaluate(handler, val_loader)
            new_test = evaluate(handler, test_loader) if test_loader is not None else None
            elapsed = time.perf_counter() - started
            delta_pp = (new_acc - baseline_acc) * 100
            params_before = utils.calc_num_parameters(model)
            params_after = utils.calc_num_parameters(mwr.model)
            record = {
                "scope": scope,
                "rows": rows_to_prune,
                "rate": rate,
                "epochs": epochs,
                "baseline_acc": round(baseline_acc, 5),
                "new_acc": round(new_acc, 5),
                "delta_pp": round(delta_pp, 3),
                "baseline_test": None if baseline_test is None else round(baseline_test, 5),
                "new_test": None if new_test is None else round(new_test, 5),
                "within_budget": delta_pp >= -allowed_drop,
                "param_ratio": round(params_after / max(params_before, 1), 5),
                "structural_share": (round(
                    sum(1 for m in prune_modes if m == "structural") / max(len(prune_modes), 1), 3)
                    if prune_modes else None),
                "seconds": round(elapsed, 1),
                "train_compressed_layer_only": train_compressed_layer_only,
            }
            out.append(record)
            recorder.record("recovery", **record)
            flag = "OK" if record["within_budget"] else "DROP"
            test_bit = (f" test {baseline_test:.3f}->{new_test:.3f}"
                        if baseline_test is not None and new_test is not None else "")
            utils.print_flush(
                f"[{flag}] scope={scope} rows={rows_to_prune} rate={rate} epochs={epochs} "
                f"ft={'layer' if train_compressed_layer_only else 'full'} | "
                f"val {baseline_acc:.3f}->{new_acc:.3f} ({delta_pp:+.2f} pp)"
                f"{test_bit} | params x{record['param_ratio']:.3f} | {elapsed:.0f}s | "
                f"structural={record['structural_share']}")
            del working, mwr, handler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--script", required=True, help="Instantiation script path")
    parser.add_argument("--arch", required=True, help="Constructor name inside the script")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.9, 0.8, 0.7, 0.6])
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 10, 20, 40])
    parser.add_argument("--scope", choices=("single", "pass"), default="single",
                        help="single = one row (agent step); pass = every prunable row once")
    parser.add_argument("--rows", default="0,mid",
                        help="For --scope single: 'mid', '0,mid', or comma-separated row ids")
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--allowed_acc_reduction", type=float, default=5.0,
                        help="Percentage-point budget used to flag OK vs DROP")
    parser.add_argument("--train_compressed_layer_only", type=utils.str2bool, default=False,
                        help="True = agent freeze policy; False = full-network fine-tune")
    parser.add_argument("--width", type=int, default=None,
                        help="Optional width kwarg for thin-ResNet constructors")
    args = parser.parse_args()

    logging_utils.setup()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _ensure_conf(device, max(args.epochs) if args.epochs else 0, args.train_compressed_layer_only)
    StaticConf.get_instance().conf_values.device = device

    utils.print_flush(f"Recovery probe on {args.checkpoint}")
    utils.print_flush(
        f"scope={args.scope} rows={args.rows} rates={args.rates} epochs={args.epochs} "
        f"train_compressed_layer_only={args.train_compressed_layer_only} device={device}")

    registry = utils.preload_datasets([args.dataset], args.train_split, args.val_split)
    optional = {"width": args.width} if args.width is not None else {}
    model = utils.load_model_from_script(
        args.arch, args.dataset, args.script, args.checkpoint, optional,
        num_classes=registry.num_classes(args.dataset),
        input_shape=registry.input_shape(args.dataset),
    ).to(device).eval()
    train_loader, val_loader, test_loader = registry.loaders(args.dataset)

    # When --rows lists several targets under --scope single, run each target as its own
    # matrix (not a multi-row prune), matching one agent step at a time.
    all_results = []
    if args.scope == "single" and "," in args.rows:
        # Expand each token against a throwaway ModelWithRows so 'mid' resolves once.
        probe_mwr = ModelWithRows(copy.deepcopy(model).to(device))
        row_targets = []
        for token in args.rows.split(","):
            row_targets.extend(_resolve_rows(probe_mwr, token.strip()))
        del probe_mwr
        # Deduplicate while preserving order
        seen = set()
        row_targets = [r for r in row_targets if not (r in seen or seen.add(r))]
        for row_idx in row_targets:
            utils.print_flush(f"--- single-row probe row={row_idx} ---")
            all_results.extend(probe_one(
                model, train_loader, val_loader, test_loader, args.rates, args.epochs,
                args.train_compressed_layer_only, device, args.allowed_acc_reduction,
                "single", str(row_idx),
            ))
    else:
        all_results = probe_one(
            model, train_loader, val_loader, test_loader, args.rates, args.epochs,
            args.train_compressed_layer_only, device, args.allowed_acc_reduction,
            args.scope, args.rows,
        )

    out_dir = os.path.join(logging_utils.run_dir(), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "recovery_probe.jsonl")
    with open(out_path, "a", encoding="utf-8") as handle:
        for row in all_results:
            handle.write(json.dumps(row) + "\n")

    utils.print_flush("=== recovery matrix (val delta pp; * = outside budget) ===")
    # Group by (scope, rows, ft) so multi-row runs stay readable
    groups = {}
    for row in all_results:
        key = (row["scope"], tuple(row["rows"]), row["train_compressed_layer_only"])
        groups.setdefault(key, []).append(row)
    for (scope, rows, layer_only), rows_data in groups.items():
        ft = "layer" if layer_only else "full"
        utils.print_flush(f"-- scope={scope} rows={list(rows)} ft={ft} --")
        epoch_list = sorted({r["epochs"] for r in rows_data})
        rate_list = sorted({r["rate"] for r in rows_data}, reverse=True)
        header = f"{'rate':>6} " + " ".join(f"{e:>8}" for e in epoch_list)
        utils.print_flush(header)
        for rate in rate_list:
            cells = []
            for epochs in epoch_list:
                match = next(r for r in rows_data
                             if r["rate"] == rate and r["epochs"] == epochs)
                mark = " " if match["within_budget"] else "*"
                cells.append(f"{match['delta_pp']:>+7.2f}{mark}")
            utils.print_flush(f"{rate:>6.2f} " + " ".join(cells))

    ok = sum(1 for r in all_results if r["within_budget"])
    utils.print_flush(
        f"* = outside -{args.allowed_acc_reduction:g} pp budget; "
        f"{ok}/{len(all_results)} OK; wrote {out_path}")
    recorder.record("recovery_summary",
                    checkpoint=os.path.basename(args.checkpoint),
                    ok=ok, total=len(all_results),
                    allowed_drop=args.allowed_acc_reduction)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
