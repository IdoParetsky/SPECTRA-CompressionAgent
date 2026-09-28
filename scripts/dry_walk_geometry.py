"""
Dry prune walk (CPU, no fine-tune): the parameter / FLOP fraction a rate policy reaches, row by
row, under the live legal mask, group-once lock and structural pruner.

Widths do not depend on the weights, so the kept fraction at every step is exactly what a TRAJ
walk of the same heuristic reaches; accuracy is not simulated. It predicts where a fine menu
(0.95), the width ladder, stream protection or a min-width floor can change a walk before a GPU
is spent, and it must reproduce the kept fraction of quoted TRAJ rows at their step (§93 mild:
r56-w4 x0.923 at step 38, r20-w2 x0.536 at step 40).

    python scripts/dry_walk_geometry.py --nets r20w2 r56w4 r56 --rules mild mildest95 mild_streams
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Configuration.ConfigurationValues import ConfigurationValues  # noqa: E402
from src.Configuration.StaticConf import StaticConf  # noqa: E402

if StaticConf.get_instance() is None:
    StaticConf(ConfigurationValues(
        device=torch.device("cpu"), test_name="dry-walk", input_dict={},
        compression_rates_dict={0: 1.0, 1: 0.9, 2: 0.8},
        runtime_limit=60, num_epochs=0, train_compressed_layer_only=False,
        allowed_acc_reduction=10, discount_factor=0.99, learning_rate=1e-3,
        rollout_limit=None, passes=2, prune=True, seed=0, n_splits=0,
        train_split=0.7, val_split=0.2, database_dict={},
        actor_checkpoint_path=None, critic_checkpoint_path=None,
        save_pruned_checkpoints=False, test_ts="dry",
    ))

import src.channel_groups as channel_groups  # noqa: E402
import src.fortify as fortify  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.utils as utils  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src.NetworkEnv import NetworkEnv, prune_current_model  # noqa: E402

RULES = {
    # name: (rate menu, heuristic, env overrides)
    "mild": ((1.0, 0.9, 0.8), "mild", {}),
    "l1": ((1.0, 0.9, 0.8), "l1", {}),
    "mildest95": ((1.0, 0.95, 0.9, 0.8), "mildest", {"SPECTRA_ACTION_DEDUPE": "1"}),
    "mild_ladder8": ((1.0, 0.9, 0.8), "mild", {"SPECTRA_WIDTH_LADDER": "8"}),
    "l1_ladder8": ((1.0, 0.9, 0.8), "l1", {"SPECTRA_WIDTH_LADDER": "8"}),
    "mild_streams": ((1.0, 0.9, 0.8), "mild", {"SPECTRA_PROTECT_STREAMS": "1"}),
    "l1_streams": ((1.0, 0.9, 0.8), "l1", {"SPECTRA_PROTECT_STREAMS": "1"}),
    "mild_minw4": ((1.0, 0.9, 0.8), "mild", {"SPECTRA_MIN_WIDTH_FOR_PRUNE": "4"}),
}
RULE_KEYS = ("SPECTRA_ACTION_DEDUPE", "SPECTRA_WIDTH_LADDER", "SPECTRA_PROTECT_STREAMS",
             "SPECTRA_MIN_WIDTH_FOR_PRUNE")


def build_net(name: str, num_classes: int = 10):
    from spectra_models_instantiation import thin_res_net
    if name == "r20w2":
        return thin_res_net.resnet20(num_classes=num_classes, large_input=False, width=2)
    if name == "r56w4":
        return thin_res_net.resnet56(num_classes=num_classes, large_input=False, width=4)
    if name == "r56":
        from spectra_models_instantiation import resnet_chenyaofo
        return resnet_chenyaofo.resnet56(num_classes=num_classes, large_input=False)
    if name == "vgg19dg":
        from spectra_models_instantiation import vgg_depgraph
        return vgg_depgraph.vgg19_bn(num_classes=100, large_input=False)
    raise ValueError(name)


def _layer_names(model):
    return {id(m): n for n, m in model.named_modules()}


def dry_walk(model, rule: str, passes: int, group_once: bool = True, input_shape=(3, 32, 32),
             flops: bool = True):
    menu, policy, overrides = RULES[rule]
    saved = {k: os.environ.get(k) for k in RULE_KEYS}
    for key in RULE_KEYS:
        os.environ.pop(key, None)
    os.environ.update(overrides)
    try:
        rates = {i: float(r) for i, r in enumerate(menu)}
        origin_p = float(utils.calc_num_parameters(model))
        origin_f = float(utils.calc_flops(model, input_shape)) if flops else None
        mwr = ModelWithRows(model)
        n_rows = len(mwr.all_rows) - 1  # the env never walks the classifier row (NetworkEnv.step)
        names = _layer_names(model)
        steps = []
        for p in range(passes):
            locked = set()
            for row in range(n_rows):
                mwr = ModelWithRows(mwr.model)
                layer_idx = mwr.row_to_main_layer[row]
                layer = mwr.all_layers[layer_idx]
                alive = int(pruning.alive_filters(layer).numel()) if hasattr(layer, "weight") else 1
                force = layer_idx in locked
                stream = NetworkEnv._is_stream_row(mwr, layer)
                if not force and fortify.protect_streams():
                    force = stream
                legal = fortify.legal_action_mask(rates, row_index=row, alive_count=alive,
                                                  device="cpu", force_identity=force)
                action = int(fortify.heuristic_eval_action(legal, rates, policy=policy, device="cpu").item())
                rate = rates[action]
                keep = NetworkEnv._ladder_keep_rate(mwr, row, rate)
                width_after = alive
                if abs(keep - 1.0) >= 1e-9:
                    mwr = prune_current_model(mwr, keep, row, quiet=True, record=False,
                                              input_shape=input_shape)
                    outcome = dict(getattr(mwr, "last_prune_outcome", {}) or {})
                    if group_once and outcome.get("mode") == "structural":
                        locked.update(int(i) for i in outcome.get("group_layer_indices") or [])
                    new_layer = ModelWithRows(mwr.model).all_layers[layer_idx]
                    width_after = int(pruning.layer_width(new_layer))
                steps.append({
                    "step": p * n_rows + row, "pass": p, "row": row,
                    "layer": names.get(id(layer), type(layer).__name__),
                    "stream": bool(stream), "rate": rate, "width": f"{alive}->{width_after}",
                    "param": utils.calc_num_parameters(mwr.model) / origin_p,
                    "flop": (utils.calc_flops(mwr.model, input_shape) / origin_f) if flops else None,
                })
        return steps
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def summarize(net: str, rule: str, steps, marks=(38, 40)):
    n = len(steps)
    per_pass = {}
    for s in steps:
        per_pass[s["pass"]] = s
    cuts = sum(1 for s in steps if abs(s["rate"] - 1.0) >= 1e-9)
    line = {"net": net, "rule": rule, "steps": n, "cuts": cuts}
    for p, s in sorted(per_pass.items()):
        line[f"pass{p + 1}_param"] = round(s["param"], 4)
        if s["flop"] is not None:
            line[f"pass{p + 1}_flop"] = round(s["flop"], 4)
    for m in marks:
        if m < n:
            line[f"step{m}_param"] = round(steps[m]["param"], 4)
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nets", nargs="+", default=["r20w2", "r56w4", "r56"])
    parser.add_argument("--rules", nargs="+", default=list(RULES))
    parser.add_argument("--passes", type=int, nargs="+", default=[2])
    parser.add_argument("--steps", action="store_true", help="print every step")
    args = parser.parse_args()
    torch.manual_seed(0)
    for net in args.nets:
        for rule in args.rules:
            for passes in args.passes:
                steps = dry_walk(build_net(net), rule, passes)
                if args.steps:
                    for s in steps:
                        print(json.dumps({"net": net, "rule": rule, **s}), flush=True)
                summary = summarize(net, rule, steps)
                summary["passes"] = passes
                print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
