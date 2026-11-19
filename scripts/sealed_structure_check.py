#!/usr/bin/env python3
"""
Sealed-set structure check: are SPECTRA's structured cuts REAL on a network, or masked?

CPU only. No data, no training, no GPU; nothing is written outside ``--out``. Run it on the BGU login node in a
staged copy of the tree (never on the Windows laptop: it imports torch):

    PY="/home/paretsky/.conda/envs/spectra/bin/python -I"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    $PY scripts/sealed_structure_check.py --out out/sealed --only mobilenetv1_x025    # one network, one JSON
    for a in $($PY scripts/sealed_structure_check.py --names); do                       # every candidate:
        setsid -w $PY scripts/sealed_structure_check.py --out out/sealed --only $a      # one process per network
    done
    $PY scripts/sealed_structure_check.py --out out/sealed --report    # the markdown table from the JSON files

One network per process is not a nicety there: on 18 Nov 2026 the login node killed, silently (no OOM event, no
exit code), every python process of this check that lived longer than five to ten minutes, a waiting parent
python with no CPU use included. One network takes 10 s to 5 min on one thread (all 49: about 45 CPU-minutes).
Run the loop detached (``setsid nohup bash -c '...' > log 2>&1 &``), at most four at a time (the user slice has
four CPUs); a JSON that exists is kept, so a stopped run resumes where it stopped.

Why. SPECTRA's cut of a channel group is real only when ``src/channel_groups.py`` understands every layer that
reads the group; otherwise ``NetworkEnv.prune_current_model`` zeroes filters and keeps every shape ("masked":
the ``prune_fallback_masked`` issue, the "shape preserved" log line), and the size the plan promised is not reached
(ShuffleNetV2, ledger 439). This script asks that question of each candidate network of the sealed test set
before anything is trained.

What it does per network (random well-conditioned weights; the answer is a property of the graph):

1. builds the network through the pipeline's own loader contract (the script file by path,
   ``utils.resolve_instantiation_func``), runs it, counts parameters and MACs (``utils.calc_flops``);
2. traces the channel groups (``channel_groups.build_channel_groups``), in eval and in train mode, and lists the
   groups the tracer refuses, with the share of parameters and MACs held by the layers that produce them;
3. builds the plan the walk would cut (``group_sensitivity.group_plan``, the default: every prunable group) and
   asks the pipeline's own per-group test of each one (``group_sensitivity.realizes_structurally``: the cut is a
   structural edit, lands on its width and computes what zeroing the same channels computes);
4. cuts every planned group once to a uniform keep of 0.75 / 0.5 / 0.25 through the path evaluation uses to cut a
   plan (``alloc_walk.cut_to``, i.e. ``NetworkEnv.prune_current_model`` group by group, L1 ranking), recording
   every prune's outcome, and checks: the forward still runs, parameters and MACs fall, no edit is masked, every
   group lands on its target width, and the cut net computes what a copy with the same channels zeroed computes
   (float64; the comparison must be informative: the output is not all zeros and the cut moved it). It also says
   whether ``plan_agent.ParamModel`` / ``FlopModel`` predict the size the cut reached;
5. cuts the groups that passed to the floor (keep 0.1, at least 2 channels) to show how far the network can
   really be shrunk.

Verdict per network:

* ``REAL``            every group but the classifier's output is cut for real and every structure check passes;
* ``PARTLY-MASKED``   some groups are cut for real, others are refused by the tracer or masked (they are named);
* ``MASKED``          no group is cut for real;
* ``FAILS-TO-TRACE``  ``torch.fx`` cannot trace the network (in eval or in train mode): every cut would be masked;
* ``ERROR``           building, running or cutting raised (the message is kept).

A cost model that mispredicts a real cut does not change the verdict; it is reported beside it (``cost models``).

``mode == "masked"`` in ``last_prune_outcome`` is the same event as the ``prune_fallback_masked`` issue and the
"shape preserved" log line (``NetworkEnv.prune_current_model`` sets all three in one branch); ``cut_to`` calls it
with ``quiet=True, record=False``, so the outcome is what this script reads.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "spectra_models_instantiation"

SHAPE = (3, 32, 32)
KEEPS = (0.75, 0.5, 0.25)
FLOOR_KEEP = 0.1     # the plans' keep floor (plan_agent.decode's k_min)
MIN_WIDTH = 2        # the eval walk's width floor on the menus in use (SPECTRA_PLAN_MIN_WIDTH=walk)
EQ_TOL = 1e-9        # float64 output gap, relative to the output's scale (group_sensitivity.REALIZE_TOL)
COST_TOL = 1e-6      # |predicted - real| kept MACs fraction under which FlopModel is called exact
VERDICTS = ("REAL", "PARTLY-MASKED", "MASKED", "FAILS-TO-TRACE", "ERROR")

# (family, instantiation script, arch = factory name, role)
#   sealed   one of the family's four size variants proposed for the sealed set
#   extra    a further factory of a sealed family (the reference size, above the 10 M aim)
#   control  a grouped-convolution family, expected unprunable in part (decision D6)
#   alias    the same network as another row (skipped by the check, covered by the tests)
CANDIDATES = (
    ("GoogLeNet", "googlenet_width_cifar.py", "googlenet_x025", "sealed"),
    ("GoogLeNet", "googlenet_width_cifar.py", "googlenet_x050", "sealed"),
    ("GoogLeNet", "googlenet_width_cifar.py", "googlenet_x075", "sealed"),
    ("GoogLeNet", "googlenet_cifar.py", "googlenet", "sealed"),
    ("GoogLeNet", "googlenet_width_cifar.py", "googlenet_x100", "alias"),
    ("MobileNet v1", "mobilenetv1_cifar.py", "mobilenetv1_x025", "sealed"),
    ("MobileNet v1", "mobilenetv1_cifar.py", "mobilenetv1_x050", "sealed"),
    ("MobileNet v1", "mobilenetv1_cifar.py", "mobilenetv1_x075", "sealed"),
    ("MobileNet v1", "mobilenetv1_cifar.py", "mobilenetv1_x100", "sealed"),
    ("SqueezeNet", "squeezenet_cifar.py", "squeezenet1_1_x050", "sealed"),
    ("SqueezeNet", "squeezenet_cifar.py", "squeezenet1_1", "sealed"),
    ("SqueezeNet", "squeezenet_cifar.py", "squeezenet1_0", "sealed"),
    ("SqueezeNet", "squeezenet_cifar.py", "squeezenet1_0_x200", "sealed"),
    ("DLA", "dla_cifar.py", "simpledla_x025", "sealed"),
    ("DLA", "dla_cifar.py", "simpledla_x050", "sealed"),
    ("DLA", "dla_cifar.py", "dla_x025", "sealed"),
    ("DLA", "dla_cifar.py", "dla_x050", "sealed"),
    ("DLA", "dla_cifar.py", "simpledla_x100", "extra"),
    ("DLA", "dla_cifar.py", "dla_x100", "extra"),
    ("EfficientNet", "efficientnet_cifar.py", "efficientnet_b0_x050", "sealed"),
    ("EfficientNet", "efficientnet_cifar.py", "efficientnet_b0_x075", "sealed"),
    ("EfficientNet", "efficientnet_cifar.py", "efficientnet_b0", "sealed"),
    ("EfficientNet", "efficientnet_cifar.py", "efficientnet_b1", "sealed"),
    ("MobileNetV3", "mobilenetv3_cifar.py", "mobilenetv3_small_x075", "sealed"),
    ("MobileNetV3", "mobilenetv3_cifar.py", "mobilenetv3_small_x100", "sealed"),
    ("MobileNetV3", "mobilenetv3_cifar.py", "mobilenetv3_large_x075", "sealed"),
    ("MobileNetV3", "mobilenetv3_cifar.py", "mobilenetv3_large_x100", "sealed"),
    ("ConvNeXt", "convnext_cifar.py", "convnext_cifar_w24", "sealed"),
    ("ConvNeXt", "convnext_cifar.py", "convnext_cifar_atto", "sealed"),
    ("ConvNeXt", "convnext_cifar.py", "convnext_cifar_femto", "sealed"),
    ("ConvNeXt", "convnext_cifar.py", "convnext_cifar_pico", "sealed"),
    ("HarDNet", "hardnet_cifar.py", "hardnet39ds", "sealed"),
    ("HarDNet", "hardnet_cifar.py", "hardnet68ds", "sealed"),
    ("HarDNet", "hardnet_cifar.py", "hardnet68_x050", "sealed"),
    ("HarDNet", "hardnet_cifar.py", "hardnet68_x075", "sealed"),
    ("GhostNet", "ghostnet_cifar.py", "ghostnet_x050", "sealed"),
    ("GhostNet", "ghostnet_cifar.py", "ghostnet_x075", "sealed"),
    ("GhostNet", "ghostnet_cifar.py", "ghostnet_x100", "sealed"),
    ("GhostNet", "ghostnet_cifar.py", "ghostnet_x130", "sealed"),
    ("SE-ResNet", "seresnet_cifar.py", "se_resnet20", "sealed"),
    ("SE-ResNet", "seresnet_cifar.py", "se_resnet32", "sealed"),
    ("SE-ResNet", "seresnet_cifar.py", "se_resnet56", "sealed"),
    ("SE-ResNet", "seresnet_cifar.py", "se_resnet110", "sealed"),
    ("ResNeXt-29", "resnext_cifar.py", "resnext29_8x4d", "control"),
    ("ResNeXt-29", "resnext_cifar.py", "resnext29_16x4d", "control"),
    ("ResNeXt-29", "resnext_cifar.py", "resnext29_32x4d", "control"),
    ("ResNeXt-29", "resnext_cifar.py", "resnext29_2x64d", "control"),
    ("RegNet", "regnet_cifar.py", "regnetx_200mf", "control"),
    ("RegNet", "regnet_cifar.py", "regnetx_400mf", "control"),
    ("RegNet", "regnet_cifar.py", "regnety_400mf", "control"),
)


# ----------------------------------------------------------------------------------------------- setup and loading


def setup():
    """Put the repo on ``sys.path`` and give ``ModelWithRows`` the minimal ``StaticConf`` it reads a device from."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import torch
    from src.Configuration.ConfigurationValues import ConfigurationValues
    from src.Configuration.StaticConf import StaticConf
    if StaticConf.get_instance() is None:
        StaticConf(ConfigurationValues(
            device=torch.device("cpu"), test_name="sealed-structure-check", input_dict={},
            compression_rates_dict={0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6},
            runtime_limit=60, num_epochs=0, train_compressed_layer_only=True,
            allowed_acc_reduction=5, discount_factor=0.99, learning_rate=1e-3,
            rollout_limit=10, passes=1, prune=True, seed=42, n_splits=0,
            train_split=0.7, val_split=0.2, database_dict={},
            actor_checkpoint_path=None, critic_checkpoint_path=None,
            save_pruned_checkpoints=False, test_ts="ts",
        ))


def load_factory(script, arch):
    """The factory ``arch`` of the instantiation script, found as ``utils.load_model_from_script`` finds it: the file
    is executed by path under its stem (not as a package member) and the name goes through
    ``utils.resolve_instantiation_func``. Returns ``(factory, resolved name, module)``."""
    setup()
    import src.utils as utils
    path = Path(script)
    if not path.is_absolute():
        path = MODELS_DIR / path
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory, resolved = utils.resolve_instantiation_func(module, arch)
    return factory, resolved, module


def build(script, arch, num_classes=10):
    """The network as the loader builds it for a 32-px dataset: ``num_classes`` and ``large_input=False`` by name."""
    factory, _resolved, _module = load_factory(script, arch)
    params = list(inspect.signature(factory).parameters)
    kwargs = {}
    if "num_classes" in params:
        kwargs["num_classes"] = num_classes
    if "large_input" in params:
        kwargs["large_input"] = False
    return factory(**kwargs)


def condition(model, seed=0):
    """Well-conditioned random weights in place (``group_sensitivity._condition``'s recipe for Conv / Linear /
    BatchNorm; every other parameter, e.g. a LayerNorm's or a layer scale, is drawn around 1): no filter is dead
    and every channel reaches the output, so a misrouted or forgotten channel shows in the output."""
    import math
    import torch
    from torch import nn
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                fan_in = max(1, module.weight[0].numel())
                module.weight.copy_(torch.randn(module.weight.shape, generator=gen) * math.sqrt(2.0 / fan_in))
                if module.bias is not None:
                    module.bias.copy_(0.1 * torch.randn(module.bias.shape, generator=gen))
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                n = module.num_features
                if module.affine:
                    module.weight.copy_(torch.rand(n, generator=gen) + 0.5)
                    module.bias.copy_(0.1 * torch.randn(n, generator=gen))
                if module.track_running_stats:
                    module.running_mean.copy_(0.1 * torch.randn(n, generator=gen))
                    module.running_var.copy_(torch.rand(n, generator=gen) + 0.5)
            else:
                for name, param in module.named_parameters(recurse=False):
                    if name == "bias":
                        param.copy_(0.1 * torch.randn(param.shape, generator=gen))
                    else:
                        param.copy_(torch.rand(param.shape, generator=gen) + 0.5)
    return model


# ----------------------------------------------------------------------------------------------- the real cut, recorded


@contextlib.contextmanager
def recording_cuts(log):
    """While active, every ``NetworkEnv.prune_current_model`` call (the one ``alloc_walk.cut_to`` makes per group)
    appends its outcome to ``log``; the cut itself is the pipeline's, unchanged."""
    import src.NetworkEnv as network_env
    real = network_env.prune_current_model

    def recording(model_with_rows, compression_rate, row_to_prune_idx, *args, **kwargs):
        out = real(model_with_rows, compression_rate, row_to_prune_idx, *args, **kwargs)
        outcome = dict(getattr(out, "last_prune_outcome", None) or {})
        edit = getattr(out, "last_group_edit", None) or {}
        structural = outcome.get("mode") == "structural"
        log.append({"row": int(row_to_prune_idx), "rate": float(compression_rate), "mode": outcome.get("mode"),
                    "reason": outcome.get("reason"), "old_width": outcome.get("old_width"),
                    "new_width": outcome.get("new_width"),
                    "keep_idx": [int(i) for i in edit.get("keep_idx", [])] if structural else None})
        return out

    network_env.prune_current_model = recording
    try:
        yield log
    finally:
        network_env.prune_current_model = real


def group_label(group, names):
    """``first producer's module name (+n more) w=<width>``: how a group is named in the report."""
    owners = list(group.producers) + list(group.depthwise)
    if not owners:
        return f"(no producer) w={int(group.width)}"
    first = names.get(id(owners[0]), "?")
    more = f" (+{len(owners) - 1})" if len(owners) > 1 else ""
    return f"{first}{more} w={int(group.width)}"


def _apply(model, plan, rates, labels, shape, x, origin64):
    """Cut ``plan`` at ``rates`` (``{row: keep}``) through ``alloc_walk.cut_to`` and measure the result.
    ``origin64`` is the uncut network's float64 output on ``x``."""
    import torch
    from src import alloc_walk, plan_agent
    import src.pruning as pruning
    import src.utils as utils
    from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows

    want_shape = origin64.shape

    log = []
    with recording_cuts(log):
        cut = alloc_walk.cut_to(model, plan, rates, shape)
    cut.eval()
    todo = [(group, row) for group, row in plan if float(rates.get(row, 1.0)) < 1.0 - 1e-9]
    if [item["row"] for item in log] != [int(row) for _group, row in todo]:
        raise RuntimeError("the recorded prunes are not the plan's rows in plan order")

    out = {"params": int(utils.calc_num_parameters(cut)), "macs": float(utils.calc_flops(cut, shape))}
    try:
        with torch.no_grad():
            got32 = cut(x)
        out["forward_ok"] = tuple(got32.shape) == tuple(want_shape)
        out["forward_error"] = None if out["forward_ok"] else f"output shape {tuple(got32.shape)}"
    except Exception as error:  # noqa: BLE001 - a cut net that cannot run is the finding
        out["forward_ok"], out["forward_error"] = False, f"{type(error).__name__}: {error}"

    cut_mwr = ModelWithRows(cut)
    widths, targets, masked, off_target = {}, {}, [], []
    for (group, row), item in zip(todo, log):
        row = int(row)
        widths[row] = int(pruning.layer_width(cut_mwr.all_layers[cut_mwr.row_to_main_layer[row]]))
        targets[row] = int(pruning.target_width(int(group.width), float(rates[row])))
        if item["mode"] != "structural":
            masked.append({"row": row, "group": labels[row], "mode": item["mode"], "reason": item["reason"]})
        elif widths[row] != targets[row]:
            off_target.append({"row": row, "group": labels[row], "target": targets[row], "width": widths[row]})
    out.update(cuts=len(log), structural=sum(item["mode"] == "structural" for item in log), masked=masked,
               off_target=off_target, widths={str(r): w for r, w in widths.items()},
               targets={str(r): w for r, w in targets.items()})

    # The same decisions written as zeros on a copy that keeps its shapes: a structural edit zeroes what
    # prune_group_structurally removed (plan_agent._zero_group, the reward's MaskedCut), a masked one zeroes the
    # filters mask_layer_filters zeroed. The real cut must compute the same function.
    twin, twin_groups = copy.deepcopy((model, [group for group, _row in todo]))
    twin_mwr = ModelWithRows(twin)
    with torch.no_grad():
        try:
            for item, twin_group in zip(log, twin_groups):
                if item["mode"] == "structural":
                    plan_agent._zero_group(twin_group, torch.tensor(item["keep_idx"], dtype=torch.long))
                else:
                    layer = twin_mwr.all_layers[twin_mwr.row_to_main_layer[item["row"]]]
                    pruning.mask_layer_filters(layer, pruning.select_surviving_filters(layer, item["rate"], mode="l1"))
            want = twin.double().eval()(x.double())
            got = cut.double().eval()(x.double())
            finite = bool(torch.isfinite(want).all() and torch.isfinite(got).all())
            if want.shape != got.shape:
                out.update(eq_gap=None, eq_ok=False, finite=finite, eq_informative=False)
            else:
                gap = float((got - want).abs().max()) / (1.0 + float(want.abs().max()))
                # the comparison says something only if the output is not all zeros and the cut moved it
                scale = float(want.abs().max())
                moved = float((want - origin64).abs().max()) / (1.0 + float(origin64.abs().max()))
                out.update(eq_gap=gap, eq_ok=bool(finite and gap <= EQ_TOL), finite=finite, out_scale=scale,
                           moved=moved,
                           eq_informative=bool(finite and scale > 0.0 and (moved > 1e3 * EQ_TOL or not log)))
        except Exception as error:  # noqa: BLE001
            out.update(eq_gap=None, eq_ok=False, finite=False, eq_informative=False,
                       eq_error=f"{type(error).__name__}: {error}")
    return out


# ----------------------------------------------------------------------------------------------- one network


def check_model(model, shape=SHAPE, keeps=KEEPS, floor_keep=FLOOR_KEEP, min_width=MIN_WIDTH, seed=0, cost_models=True):
    """
    The structure check of one network (module docstring, steps 1-5). ``model`` is re-initialised in place with
    ``condition``; every cut is made on copies. Returns a JSON-able dict with ``verdict`` and ``masked_groups``.
    """
    setup()
    import torch
    from torch import nn
    from src import plan_agent
    import src.channel_groups as channel_groups
    import src.group_sensitivity as group_sensitivity
    import src.utils as utils
    from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows

    started = time.perf_counter()
    res = {"shape": list(shape), "keeps": [float(k) for k in keeps], "floor_keep": float(floor_keep),
           "min_width": int(min_width), "torch": torch.__version__, "error": None, "masked_groups": [],
           "spectra_env": sorted(k for k in os.environ if k.startswith("SPECTRA_"))}
    try:
        model.eval()
        condition(model, seed)
        x = torch.randn(2, *shape, generator=torch.Generator().manual_seed(seed + 1))
        with torch.no_grad():
            y0 = model(x)
            origin64 = copy.deepcopy(model).double().eval()(x.double())
        res["out_shape"] = list(y0.shape)
        params0 = int(utils.calc_num_parameters(model))
        macs0 = float(utils.calc_flops(model, shape))
        res.update(params=params0, macs=macs0)

        # --- 2. channel groups, eval and train mode
        names = {id(module): name for name, module in model.named_modules()}
        groups = channel_groups.build_channel_groups(model)
        model.train()
        groups_train = channel_groups.build_channel_groups(model)
        model.eval()
        res["trace_eval"] = groups is not None

        def signature(found):
            return sorted((group_label(g, names), bool(g.prunable), g.reason) for g in found
                          if g.producers or g.depthwise)

        if groups is None:
            res["trace_train"] = "fails" if groups_train is None else "works"
            res.update(verdict="FAILS-TO-TRACE", masked_groups=["every group: torch.fx cannot trace the network"],
                       seconds=time.perf_counter() - started)
            return res
        res["trace_train"] = ("fails" if groups_train is None
                              else "same" if signature(groups_train) == signature(groups) else "differs")

        mwr = ModelWithRows(model)
        res.update(layers=len(mwr.all_layers), rows=len(mwr.row_to_main_layer),
                   first_layer=type(mwr.all_layers[0]).__name__)
        plan = group_sensitivity.group_plan(mwr, groups, structural_only=False)
        rows = [int(row) for _group, row in plan]
        labels = {int(row): group_label(group, names) for group, row in plan}
        planned_ids = {id(group) for group, _row in plan}

        # --- ownership: a Conv / Linear belongs to the group whose channels it produces, a BatchNorm to the groups
        # it normalises (by share of its features). "blocked" is what sits in layers whose output width SPECTRA
        # cannot cut: the groups the tracer refuses, and Conv / Linear / BatchNorm layers that are in no group at
        # all. "other" is the rest (activations' MACs, LayerNorm, bare parameters).
        per_macs = utils.per_module_macs(model, shape)
        own_params = {id(m): int(sum(p.numel() for p in m.parameters(recurse=False))) for m in model.modules()}
        buckets = {key: [0.0, 0.0] for key in ("planned", "blocked", "output", "narrow", "unplanned", "other")}
        blocked, narrow, unplanned, seen = [], [], [], set()
        real_groups = [g for g in groups if g.producers or g.depthwise]
        for group in groups:
            owned = bool(group.producers or group.depthwise)
            if not owned:
                # no layer produces these channels: the network input (its norms are "other"), or a tensor the
                # tracer lost track of (its norms cannot be resized: "blocked")
                kind = "other" if group.reason == "model input" else "blocked"
            elif id(group) in planned_ids:
                kind = "planned"
            elif not group.prunable:
                kind = "output" if group.reason == "model output" else "blocked"
            elif int(group.width) < 2:
                kind = "narrow"
            else:
                kind = "unplanned"
            mine = [0.0, 0.0]
            for module in list(group.producers) + list(group.depthwise):
                if id(module) in seen:
                    continue
                seen.add(id(module))
                mine[0] += own_params.get(id(module), 0)
                mine[1] += per_macs.get(id(module), 0)
            for ref in group.norms:
                seen.add(id(ref.module))
                share = (len(ref.positions) / float(ref.total)) if ref.total else 1.0
                mine[0] += share * own_params.get(id(ref.module), 0)
                mine[1] += share * per_macs.get(id(ref.module), 0)
            if kind != "other":  # "other" is the remainder, taken below
                buckets[kind][0] += mine[0]
                buckets[kind][1] += mine[1]
            if not owned:
                continue
            entry = {"group": group_label(group, names), "width": int(group.width), "reason": group.reason,
                     "producers": [names.get(id(m), "?") for m in list(group.producers) + list(group.depthwise)],
                     "params_share": mine[0] / params0, "macs_share": mine[1] / macs0}
            if kind == "blocked":
                blocked.append(entry)
            elif kind == "narrow":
                narrow.append(entry)
            elif kind == "unplanned":
                unplanned.append(entry)
        untracked = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)) and id(module) not in seen:
                untracked.append(name)
                buckets["blocked"][0] += own_params.get(id(module), 0)
                buckets["blocked"][1] += per_macs.get(id(module), 0)
        buckets["other"] = [params0 - sum(v[0] for k, v in buckets.items() if k != "other"),
                            macs0 - sum(v[1] for k, v in buckets.items() if k != "other")]
        res.update(groups=len(real_groups), planned=len(plan), blocked=blocked, narrow=narrow, unplanned=unplanned,
                   untracked_layers=untracked,
                   shares={key: {"params": v[0] / params0, "macs": v[1] / macs0} for key, v in buckets.items()})

        # --- 3. the pipeline's own per-group test (fresh per network: its cache is keyed by module names only)
        group_sensitivity._REALIZED.clear()
        realized = {int(row): bool(group_sensitivity.realizes_structurally(model, group, row)) for group, row in plan}
        res["realized"] = sum(realized.values())
        res["not_realized"] = [{"row": row, "group": labels[row]} for row in rows if not realized[row]]

        # --- the cost models a plan is decoded on
        pm = fm = None
        if cost_models and plan:
            pm = plan_agent.ParamModel(model, plan)
            try:
                fm = plan_agent.FlopModel(model, plan, shape)
            except Exception as error:  # noqa: BLE001
                res["flop_model_error"] = f"{type(error).__name__}: {error}"

        # --- 4. uniform cuts of every planned group
        res["cuts"] = {}
        reasons = {}  # row -> why its group is not cut for real (the cut's own reason first)
        for keep in keeps:
            if not plan:
                break
            cut = _apply(model, plan, {row: float(keep) for row in rows}, labels, shape, x, origin64)
            cut.update(params_kept=cut["params"] / params0, macs_kept=cut["macs"] / macs0,
                       shrinks=bool(cut["params"] < params0 and cut["macs"] < macs0))
            targets = {int(r): w for r, w in cut["targets"].items()}
            if pm is not None:
                cut["pm_params"] = int(pm.params(targets))
                cut["pm_exact"] = cut["pm_params"] == cut["params"]
            if fm is not None:
                cut["fm_kept"] = float(fm.kept(targets))
                cut["fm_err"] = abs(cut["fm_kept"] - cut["macs_kept"])
                cut["fm_exact"] = cut["fm_err"] <= COST_TOL
            for item in cut["masked"]:
                reasons.setdefault(item["row"], f"{item['mode']} at keep {float(keep):g}: {item['reason']}")
            for item in cut["off_target"]:
                reasons.setdefault(item["row"], f"landed on width {item['width']} at keep {float(keep):g}, "
                                                f"target {item['target']}")
            res["cuts"][f"{float(keep):g}"] = cut
        for row in rows:
            if not realized[row]:
                reasons.setdefault(row, "realizes_structurally is False (the cut is not a structural edit that "
                                        "computes the zeroed copy's function, or a layer reads only part of the group)")

        # --- 5. the floor cut of the groups that passed
        good = [(group, row) for group, row in plan if int(row) not in reasons]
        if good:
            widths0 = {int(row): int(group.width) for group, row in good}
            floor = plan_agent.widths_of({row: float(floor_keep) for row in widths0}, widths0, min_width)
            cut = _apply(model, good, plan_agent.rates_of(floor, widths0), labels, shape, x, origin64)
            cut.update(params_kept=cut["params"] / params0, macs_kept=cut["macs"] / macs0)
            for item in cut["masked"]:
                reasons.setdefault(item["row"], f"{item['mode']} at the floor cut: {item['reason']}")
            for item in cut["off_target"]:
                reasons.setdefault(item["row"], f"landed on width {item['width']} at the floor cut, "
                                                f"target {item['target']}")
            res["floor_cut"] = cut
            good = [(group, row) for group, row in good if int(row) not in reasons]
        res["real_groups"] = len(good)

        # --- verdict
        res["masked_planned"] = [{"row": row, "group": labels[row], "reason": reasons[row]} for row in sorted(reasons)]
        cuts = list(res["cuts"].values()) + ([res["floor_cut"]] if "floor_cut" in res else [])
        res["structure_ok"] = bool(all(c["forward_ok"] and c["eq_ok"] and c["eq_informative"] for c in cuts)
                                   and all(c["shrinks"] for c in res["cuts"].values()))
        res["cost_models_ok"] = bool(all(c.get("pm_exact", True) and c.get("fm_exact", True)
                                         for c in res["cuts"].values()) and "flop_model_error" not in res)
        res["masked_groups"] = ([f"{b['group']} [{b['reason']}]" for b in blocked]
                                + [f"{b['group']} [{b['reason']}]" for b in res["masked_planned"]]
                                + [f"{b['group']} [prunable but in no plan row]" for b in unplanned]
                                + [f"{name} [layer in no channel group]" for name in untracked])
        if res["trace_train"] != "same":
            res["verdict"] = "FAILS-TO-TRACE"
            res["masked_groups"] = [f"train-mode trace {res['trace_train']}"] + res["masked_groups"]
        elif not good:
            res["verdict"] = "MASKED"
        elif blocked or reasons or unplanned or untracked or not res["structure_ok"]:
            res["verdict"] = "PARTLY-MASKED"
            if not res["masked_groups"]:
                res["masked_groups"] = ["no single group: the cut net runs, shrinks or computes differently from the "
                                        "zeroed copy only when the groups are cut together"]
        else:
            res["verdict"] = "REAL"
    except Exception as error:  # noqa: BLE001 - the report keeps what raised
        res.update(verdict="ERROR", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    res["seconds"] = time.perf_counter() - started
    return res


def check_candidate(family, script, arch, role, num_classes=10, **kwargs):
    """``check_model`` of one registry row, with its identity; a factory that cannot build is an ``ERROR`` row."""
    head = {"family": family, "script": script, "arch": arch, "role": role, "num_classes": int(num_classes)}
    try:
        model = build(script, arch, num_classes)
    except Exception as error:  # noqa: BLE001
        head.update(verdict="ERROR", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                    masked_groups=[], seconds=0.0)
        return head
    head.update(check_model(model, **kwargs))
    return head


# ----------------------------------------------------------------------------------------------- report


def _pct(value):
    return "-" if value is None else f"{100.0 * value:.1f}%"


def _per_keep(result, key, fmt):
    cuts = result.get("cuts") or {}
    cells = []
    for keep in result.get("keeps", KEEPS):
        cut = cuts.get(f"{float(keep):g}")
        cells.append(fmt(cut[key]) if cut is not None and cut.get(key) is not None else "-")
    return " / ".join(cells)


def _check_notes(result):
    """``(structure, cost models)``: what failed among the network-level checks, or ``pass`` / ``exact``."""
    structure, cost = [], []
    for key, cut in (result.get("cuts") or {}).items():
        if not cut["forward_ok"]:
            structure.append(f"forward breaks at {key} ({cut['forward_error']})")
        if not cut["shrinks"]:
            structure.append(f"no shrink at {key}")
        if not cut["eq_ok"]:
            structure.append(f"not the zeroed copy's function at {key} (gap {cut['eq_gap']})")
        elif not cut.get("eq_informative", True):
            structure.append(f"the comparison with the zeroed copy says nothing at {key} (the output is all zeros "
                             f"or the cut did not move it)")
        if cut.get("pm_exact") is False:
            cost.append(f"ParamModel at {key}: {cut['pm_params']} predicted, {cut['params']} real")
        if cut.get("fm_exact") is False:
            cost.append(f"FlopModel at {key}: {cut['fm_kept']:.4f} predicted, {cut['macs_kept']:.4f} real")
    floor = result.get("floor_cut")
    if floor and not (floor["forward_ok"] and floor["eq_ok"] and floor.get("eq_informative", True)):
        structure.append("floor cut does not run or is not the zeroed copy's function")
    if result.get("flop_model_error"):
        cost.append(f"FlopModel raised: {result['flop_model_error']}")
    if result.get("spectra_env"):
        structure.append(f"SPECTRA_* set: {','.join(result['spectra_env'])}")
    return ("; ".join(structure) if structure else "pass"), ("; ".join(cost) if cost else "exact")


def _verdict_cell(result):
    """The verdict in bold; a PARTLY-MASKED / MASKED one also says what kind: groups the tracer refuses are never in
    a plan (they stay whole and plans land on their size), groups masked in the plan are cuts the walk would play
    and not get."""
    cell = f"**{result.get('verdict')}**"
    parts = []
    if result.get("blocked"):
        parts.append(f"{len(result['blocked'])} unprunable")
    if result.get("untracked_layers"):
        parts.append(f"{len(result['untracked_layers'])} layers in no group")
    if result.get("masked_planned"):
        parts.append(f"{len(result['masked_planned'])} masked in plan")
    if result.get("unplanned"):
        parts.append(f"{len(result['unplanned'])} in no plan row")
    return f"{cell} ({', '.join(parts)})" if parts and result.get("verdict") != "REAL" else cell


def render_report(results):
    """The markdown report: the table (one row per network), the verdict per family, then the named groups of every
    network that is not REAL."""
    order = {(c[1], c[2]): i for i, c in enumerate(CANDIDATES)}
    results = sorted(results, key=lambda r: order.get((r.get("script"), r.get("arch")), len(order)))
    header = ["family", "network", "role", "params M", "MACs M", "groups", "planned", "real", "blocked",
              "blocked share (params / MACs)", "params kept @0.75/0.5/0.25", "MACs kept @0.75/0.5/0.25",
              "masked edits @0.75/0.5/0.25", "floor (params / MACs)", "structure checks", "cost models", "verdict"]
    versions = ", ".join(sorted({str(r["torch"]) for r in results if r.get("torch")})) or "?"
    lines = ["# Sealed-set structure check", "",
             f"{len(results)} networks, input {tuple(SHAPE)}, random well-conditioned weights, torch {versions}; "
             "MACs are `utils.calc_flops`; cuts go through `alloc_walk.cut_to` (L1).", "",
             "- `groups`: channel groups that have a producer; `planned`: the groups the walk's plan cuts "
             "(`group_sensitivity.group_plan`, default flags; a group one channel wide cannot be cut and is counted "
             "beside it); `real`: planned groups whose cut is structural at every keep, lands on its width and "
             "passes `realizes_structurally`.",
             "- `blocked`: groups the tracer refuses (the classifier's output aside); the share is the parameters "
             "and MACs of the layers that produce those groups, plus Conv / Linear / BatchNorm layers in no group.",
             "- `kept`: parameters and MACs kept after a uniform keep of 0.75 / 0.5 / 0.25 of every planned group; "
             "`masked edits`: prunes of that cut that fell back to masking.",
             "- `floor`: parameters / MACs kept after cutting every real group to keep 0.1 (at least 2 channels).",
             "- `structure checks`: the cut net runs, shrinks and computes what a copy with the same channels zeroed "
             "computes (float64); `cost models`: `ParamModel` / `FlopModel` predict the size each uniform cut "
             "reached.",
             "- `verdict`: REAL = every group but the classifier's output is cut for real; PARTLY-MASKED = some are, "
             "and the rest are either `unprunable` (refused by the tracer: never in a plan, so they stay whole and "
             "plans still land on their size) or `masked in plan` (planned, and the walk's cut falls back to masking: "
             "the plan's size is not reached); MASKED = no group is cut for real; FAILS-TO-TRACE = `torch.fx` cannot "
             "trace the network.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        if "groups" not in r or r.get("verdict") == "ERROR":
            # the network did not trace, or something raised: what is known, and the reason in the checks column
            size = f"{r['params'] / 1e6:.3f}" if "params" in r else "-"
            macs = f"{r['macs'] / 1e6:.1f}" if "macs" in r else "-"
            note = r.get("error") or "; ".join(r.get("masked_groups", [])) or "-"
            cells = [r.get("family"), r.get("arch"), r.get("role"), size, macs] + ["-"] * 9 + [note, "-",
                                                                                               _verdict_cell(r)]
        else:
            shares, floor = r["shares"], r.get("floor_cut")
            structure, cost = _check_notes(r)
            planned = f"{r['planned']} (+{len(r['narrow'])} one-wide)" if r.get("narrow") else str(r["planned"])
            cells = [r["family"], r["arch"], r["role"], f"{r['params'] / 1e6:.3f}", f"{r['macs'] / 1e6:.1f}",
                     r["groups"], planned, r.get("real_groups", 0), len(r["blocked"]),
                     f"{_pct(shares['blocked']['params'])} / {_pct(shares['blocked']['macs'])}",
                     _per_keep(r, "params_kept", lambda v: f"{v:.3f}"),
                     _per_keep(r, "macs_kept", lambda v: f"{v:.3f}"),
                     _per_keep(r, "masked", lambda v: str(len(v))),
                     (f"{floor['params_kept']:.3f} / {floor['macs_kept']:.3f}" if floor else "-"),
                     structure, cost, _verdict_cell(r)]
        # a table cell cannot hold a pipe
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")

    lines += ["", "## Verdict per family", ""]
    by_family = {}
    for r in results:
        by_family.setdefault(r.get("family"), []).append(r)
    for family, rows in by_family.items():
        verdicts = ", ".join(sorted({str(r.get("verdict")) for r in rows}))
        sizes = [r["params"] / 1e6 for r in rows if "params" in r]
        span = f"{min(sizes):.2f}-{max(sizes):.2f} M" if sizes else "-"
        why = {}
        for r in rows:
            for b in r.get("blocked", []):
                why[f"tracer: {b['reason']}"] = why.get(f"tracer: {b['reason']}", 0) + 1
            for b in r.get("masked_planned", []):
                # "masked at keep 0.75: <the walk's reason>" -> the walk's reason; other reasons up to their bracket
                short = b["reason"].split(": ", 1)[1] if ": " in b["reason"] else b["reason"].split(" (")[0]
                why[f"cut: {short}"] = why.get(f"cut: {short}", 0) + 1
        cause = ("; ".join(f"{reason} ({count})" for reason, count in sorted(why.items(), key=lambda kv: -kv[1]))
                 if why else "no group refused or masked")
        lines.append(f"- **{family}** ({len(rows)} networks, {span}): {verdicts}. {cause}.")

    lines += ["", "## Groups that are not cut for real", ""]
    for r in results:
        if r.get("verdict") == "REAL":
            continue
        lines.append(f"### {r.get('arch')} ({r.get('family')}): {r.get('verdict')}")
        lines.append("")
        if r.get("error"):
            lines += ["```", (r.get("traceback") or r["error"]).rstrip(), "```", ""]
            continue
        by_reason = {}
        for b in r.get("blocked", []):
            by_reason.setdefault(f"tracer: {b['reason']}", []).append(b["group"])
        for b in r.get("masked_planned", []):
            by_reason.setdefault(f"cut: {b['reason']}", []).append(b["group"])
        for b in r.get("unplanned", []):
            by_reason.setdefault("prunable but in no plan row", []).append(b["group"])
        if r.get("untracked_layers"):
            by_reason["Conv / Linear / BatchNorm layer in no channel group"] = list(r["untracked_layers"])
        if r.get("trace_train") not in (None, "same"):
            lines.append(f"- train-mode trace: {r['trace_train']}")
        if "groups" not in r:
            lines.append("- " + "; ".join(r.get("masked_groups", [])))
        for reason, found in by_reason.items():
            lines.append(f"- {reason} ({len(found)}): " + "; ".join(found))
        if r.get("shares"):
            lines.append("- shares (params / MACs): " + ", ".join(
                f"{key} {_pct(v['params'])} / {_pct(v['macs'])}" for key, v in r["shares"].items()))
        if r.get("narrow"):
            lines.append(f"- too narrow to cut (width 1): {len(r['narrow'])} group(s)")
        lines.append("")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------- command line


def _write_atomic(path, text):
    """Write ``text`` in full to a temp file of this process, then replace ``path`` with it."""
    tmp = Path(f"{path}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def _selected(args):
    wanted = [c for c in CANDIDATES if c[3] != "alias" or args.aliases]
    if args.role:
        roles = {r.strip() for r in args.role.split(",")}
        wanted = [c for c in wanted if c[3] in roles]
    if args.family:
        fams = [f.strip().lower() for f in args.family.split(",")]
        wanted = [c for c in wanted if any(f in c[0].lower() for f in fams)]
    if args.only:
        names = {n.strip() for n in args.only.split(",")}
        missing = names - {c[2] for c in CANDIDATES}
        if missing:
            raise SystemExit(f"unknown network(s): {sorted(missing)}")
        wanted = [c for c in CANDIDATES if c[2] in names]
    return wanted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="directory for one JSON per network and the report")
    parser.add_argument("--only", default="", help="comma-separated factory names")
    parser.add_argument("--family", default="", help="comma-separated family names (substring, any case)")
    parser.add_argument("--role", default="", help="comma-separated roles: sealed, extra, control, alias")
    parser.add_argument("--aliases", action="store_true", help="also take alias rows (the same network twice)")
    parser.add_argument("--keeps", default=",".join(f"{k:g}" for k in KEEPS))
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="recompute networks whose JSON exists")
    parser.add_argument("--report", action="store_true", help="only write the report from the JSON files in --out")
    parser.add_argument("--list", action="store_true", help="print every candidate (family, arch|script, role)")
    parser.add_argument("--names", action="store_true",
                        help="print the selected networks' factory names, one per line (for a shell loop)")
    args = parser.parse_args(argv)

    if args.list:
        for family, script, arch, role in CANDIDATES:
            print(f"{family}\t{arch}|{script}\t{role}")
        return 0
    if args.names:
        for _family, _script, arch, _role in _selected(args):
            print(arch)
        return 0
    if args.out is None:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.report:
        keeps = tuple(float(k) for k in args.keeps.split(","))
        if any(not 0.0 < k < 1.0 for k in keeps):
            parser.error("--keeps: every keep must be inside (0, 1)")
        stray = sorted(k for k in os.environ if k.startswith("SPECTRA_"))
        if stray:
            print(f"WARNING: SPECTRA_* variables are set and may change the cut: {stray}", flush=True)
        for family, script, arch, role in _selected(args):
            path = args.out / f"{Path(script).stem}__{arch}.json"
            if path.exists() and not args.force:
                print(f"skip {arch} (exists)", flush=True)
                continue
            print(f"check {arch} ...", flush=True)
            result = check_candidate(family, script, arch, role, num_classes=args.num_classes, keeps=keeps)
            _write_atomic(path, json.dumps(result, indent=1) + "\n")
            floor = result.get("floor_cut") or {}
            tail = f" | {result['error']}" if result.get("error") else ""
            print(f"  {arch}: {result['verdict']} | params {result.get('params', 0) / 1e6:.3f} M | groups "
                  f"{result.get('groups', '-')} planned {result.get('planned', '-')} real "
                  f"{result.get('real_groups', '-')} blocked {len(result.get('blocked', []))} | floor "
                  f"{floor.get('params_kept', float('nan')):.3f} / {floor.get('macs_kept', float('nan')):.3f} | "
                  f"{result.get('seconds', 0.0):.0f}s{tail}", flush=True)

    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.out.glob("*__*.json"))]
    if results:
        _write_atomic(args.out / "sealed_structure_report.md", render_report(results))
        counts = {}
        for r in results:
            counts[r.get("verdict")] = counts.get(r.get("verdict"), 0) + 1
        print(f"report: {args.out / 'sealed_structure_report.md'} | {len(results)} networks | "
              + ", ".join(f"{v} {counts[v]}" for v in VERDICTS if v in counts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
