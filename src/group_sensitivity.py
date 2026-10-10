"""
Per-group sensitivity channels for the agent's state (``SPECTRA_STATE_SENS``, v10).

A0 (ledger §201 / §204 / §205): at equal kept parameters, keeping more of the groups whose cut
hurts most beat a uniform allocation after a 40-epoch recovery on both ResNet-56 cells. The rule
needs one cheap number per coupled group: the calibration-loss rise when that group alone is cut
to half its width, without fine-tuning (Li et al., ICLR 2017, per-layer sensitivity). This is
``scripts/allocation_probe.py``'s measurement, taken once per episode on the origin network and
given to every layer that produces the group's channels.

``SPECTRA_PLAN_STRUCTURAL_ONLY`` (v18, default off) makes ``group_plan`` keep only the groups whose cut the walk
realizes as a structural edit (``realizes_structurally``), so every plan built on it predicts the size its cut reaches.
"""

from __future__ import annotations

import copy
import math
import statistics
import time

import torch
from torch import nn

import src.channel_groups as channel_groups
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows

SENS_KEEP = 0.5
CALIB_BATCHES = 4
LOG_SPAN = 3.0  # a group 20x (e^3) more or less sensitive than the median saturates the channel
PROBE_SPATIAL = (32, 224, 28)  # the input sizes NetworkEnv.dummy_forward_ok tries
REALIZE_TOL = 1e-9  # float64 output gap (relative to the output's scale) under which a cut computes the masked function

_REALIZED = {}  # (architecture, the group's producer names) -> realizes_structurally's answer
_ANNOUNCED = set()  # architectures whose structural-only plan has been logged


def calibration_batches(loader, n_batches, device):
    """The loader's next ``n_batches`` train batches on ``device`` (labels as class indices)."""
    batches = []
    for x, y in loader:
        if y.dim() > 1 and y.shape[1] > 1:
            y = y.argmax(dim=1)
        batches.append((x.to(device), y.long().to(device)))
        if len(batches) >= n_batches:
            break
    return batches


@torch.no_grad()
def calib_loss(model, batches):
    """Mean cross-entropy over ``batches`` in eval mode (the model's mode is restored)."""
    was_training = model.training
    model.eval()
    loss = nn.CrossEntropyLoss(reduction="sum")
    total = sum(float(loss(model(x), y)) for x, y in batches)
    model.train(was_training)
    return total / max(1, sum(int(y.numel()) for _, y in batches))


def group_plan(model_with_rows, groups=None, structural_only=None):
    """``[(group, row)]``: each coupled group the walk can cut, at its first row, in walk order.

    ``structural_only`` (``None``: ``SPECTRA_PLAN_STRUCTURAL_ONLY``, ``fortify.plan_structural_only``) also leaves out
    every group ``realizes_structurally`` rejects, so a plan only cuts what the walk shrinks.
    """
    if groups is None:
        groups = channel_groups.build_channel_groups(model_with_rows.model) or []
    plan, seen = [], set()
    for row in sorted(model_with_rows.row_to_main_layer)[:-1]:
        layer = model_with_rows.all_layers[model_with_rows.row_to_main_layer[row]]
        group = channel_groups.group_of(groups, layer)
        if group is None or not group.prunable or group.width < 2 or id(group) in seen:
            continue
        seen.add(id(group))
        plan.append((group, row))
    if structural_only is None:
        import src.fortify as fortify
        structural_only = fortify.plan_structural_only()
    if structural_only:
        plan = structural_plan(model_with_rows.model, plan)
    return plan


def architecture(model):
    """The module tree's names and classes. A group's realizability is a property of this graph, not of the weights or
    the widths, so one check serves every checkpoint of an architecture and every step of a walk on it."""
    return tuple((name, type(module).__module__, type(module).__qualname__) for name, module in model.named_modules())


def structural_plan(model, plan):
    """The ``(group, row)`` of ``plan`` that ``realizes_structurally`` passes; the first plan of an architecture is
    logged with the rows it leaves whole."""
    arch = architecture(model)
    names = {id(module): name for name, module in model.named_modules()}
    kept = [(group, row) for group, row in plan if realizes_structurally(model, group, row, arch, names)]
    if arch not in _ANNOUNCED:
        _ANNOUNCED.add(arch)
        import src.utils as utils
        rows = {row for _group, row in kept}
        utils.print_flush(f"[plan] structural only ({type(model).__name__}): {len(kept)} of {len(plan)} groups "
                          f"planned; left whole (cut masked or not the masked function): rows "
                          f"{[row for _group, row in plan if row not in rows]}")
    return kept


def reads_whole(group) -> bool:
    """Every consumer and normalisation of ``group`` reads all of its channels. Under a chunk / split view the kept size
    depends on which channels survive (``ParamModel`` counts a share), and ``pruning.prune_group_structurally`` masks
    such a cut or replays it to another width."""
    width = int(group.width)
    return all(not ref.positions or len(ref.positions) == width for ref in list(group.consumers) + list(group.norms))


def realizes_structurally(model, group, row, arch=None, names=None) -> bool:
    """
    True when the walk's cut of ``group`` (first walk row ``row``) is a structural edit that computes what zeroing the
    same channels computes, so the cut shrinks the net by what ``ParamModel`` / ``FlopModel`` predict.

    The group must be read whole (``reads_whole``). It is then cut for real (``NetworkEnv.prune_current_model``, L1)
    on a copy with well-conditioned random weights, to two adjacent widths: each cut must be structural, at that
    width, and agree in float64 with a copy whose removed channels are zeroed (``plan_agent._zero_group``). Two adjacent
    widths catch a width the graph constrains (ShuffleNetV2's channel shuffle needs an even concatenation); the
    agreement catches an edit that runs but misroutes channels (a cut whose survivors a later channel shuffle
    re-interleaves, while the consumer is sliced on the old order). Cached per ``architecture`` and producer names.
    """
    arch = architecture(model) if arch is None else arch
    if names is None:
        names = {id(module): name for name, module in model.named_modules()}
    key = (arch, tuple(sorted(names.get(id(module), "?") for module in list(group.producers) + list(group.depthwise))))
    if key not in _REALIZED:
        _REALIZED[key] = _realizes(model, group, row)
    return _REALIZED[key]


def _realizes(model, group, row) -> bool:
    """``realizes_structurally`` without the cache. The copies' cuts leave the global RNG (module construction draws
    from it) and the Network Slimming scores ``prune_current_model`` rebinds as they were."""
    from src import pruning
    width = int(group.width)
    if width < 2 or not reads_whole(group):
        return False
    saved = dict(pruning._BN_ABS_GAMMA)
    try:
        with torch.random.fork_rng(devices=[]):
            probe, twin = copy.deepcopy((model, group))
            _condition(probe)
            x = _probe_input(probe)
            if x is None:
                return False
            half = max(1, width // 2)
            for k in (half, half + 1 if half + 1 < width else half - 1):
                if k >= 1 and not _cut_matches_mask(probe, twin, row, k, x):
                    return False
            return True
    except Exception as error:  # noqa: BLE001 - a cut that cannot be checked is not planned
        import src.utils as utils
        utils.print_flush(f"[plan] structural check of row {row} raised {type(error).__name__}: {error}; "
                          f"the group stays whole")
        return False
    finally:
        pruning._BN_ABS_GAMMA.clear()
        pruning._BN_ABS_GAMMA.update(saved)


def _cut_matches_mask(probe, group, row, width, x) -> bool:
    """The walk's cut of ``group`` (on ``probe``) to ``width``, made on a copy of ``probe``, is structural, lands on
    ``width`` and computes on ``x``, in float64, what another copy computes with the same channels zeroed."""
    from src.NetworkEnv import prune_current_model
    from src.plan_agent import _zero_group
    cut = prune_current_model(ModelWithRows(copy.deepcopy(probe)), width / float(group.width), row, quiet=True,
                              record=False, input_shape=tuple(x.shape[1:]), importance="l1")
    outcome = getattr(cut, "last_prune_outcome", None) or {}
    edit = getattr(cut, "last_group_edit", None) or {}
    if outcome.get("mode") != "structural" or int(edit.get("new_width", -1)) != int(width):
        return False
    masked, twin = copy.deepcopy((probe, group))
    _zero_group(twin, torch.tensor(edit["keep_idx"], dtype=torch.long))
    with torch.no_grad():
        want = masked.double().eval()(x.double())
        got = cut.model.to(x.device).double().eval()(x.double())
    if want.shape != got.shape:
        return False
    return float((got - want).abs().max()) <= REALIZE_TOL * (1.0 + float(want.abs().max()))


@torch.no_grad()
def _condition(model, seed=0):
    """Well-conditioned random weights: no filter is dead (``pruning.target_width`` would keep fewer of a checkpoint's)
    and every channel reaches the output, so a misrouted one shows."""
    gen = torch.Generator().manual_seed(seed)
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


def _probe_input(model, batch=2):
    """A fixed random batch at the first of ``PROBE_SPATIAL`` the model runs on, or None."""
    device = next(model.parameters()).device
    channels = next((int(m.in_channels) for m in model.modules() if isinstance(m, nn.Conv2d)), 3)
    gen = torch.Generator().manual_seed(1)
    model.eval()
    with torch.no_grad():
        for spatial in PROBE_SPATIAL:
            x = torch.randn(batch, channels, spatial, spatial, generator=gen).to(device)
            try:
                model(x)
            except Exception:  # noqa: BLE001 - the next size, as dummy_forward_ok does
                continue
            return x
    return None


def group_sensitivity(model, plan, batches, input_shape, keep=SENS_KEEP, costs=None):
    """``({row: loss rise}, base loss)``: each planned group cut alone to ``keep`` (L1 survivors).

    With a dict ``costs`` it is also filled with ``{row: (params saved, MACs saved)}`` measured on that same cut
    (``sens_cost``, v16); with ``None`` no cost is measured.
    """
    from src.NetworkEnv import prune_current_model
    base = calib_loss(model, batches)
    if costs is not None:
        import src.utils as utils
        p0 = utils.calc_num_parameters(model)
        f0 = utils.calc_flops(model, input_shape)
    out = {}
    for _group, row in plan:
        cut = prune_current_model(ModelWithRows(copy.deepcopy(model)), keep, row, quiet=True, record=False,
                                  input_shape=input_shape, importance="l1")
        out[row] = calib_loss(cut.model, batches) - base
        if costs is not None:
            costs[row] = (float(p0 - utils.calc_num_parameters(cut.model)),
                          float(f0 - utils.calc_flops(cut.model, input_shape)))
        del cut
    return out, base


def normalise(sens):
    """``{row: (log-ratio channel, percentile channel)}`` from raw loss rises.

    Log-ratio: ``log(s / median) / LOG_SPAN`` clipped to [−1, 1], with s floored at 5 % of the
    median so a cut that lowers the loss reads as very cheap rather than undefined. Percentile:
    the group's rank among the net's groups, 0 (cheapest) to 1 (most sensitive).
    """
    rows = list(sens)
    if not rows:
        return {}
    values = [max(0.0, float(sens[row])) for row in rows]
    floor = max(1e-6, 0.05 * statistics.median(values))
    floored = [max(v, floor) for v in values]
    mid = statistics.median(floored)
    order = sorted(range(len(rows)), key=lambda k: floored[k])
    pct = [0.0] * len(rows)
    for rank, k in enumerate(order):
        pct[k] = rank / (len(rows) - 1) if len(rows) > 1 else 0.5
    return {row: (max(-1.0, min(1.0, math.log(floored[k] / mid) / LOG_SPAN)), pct[k])
            for k, row in enumerate(rows)}


def layer_features(model, batches, input_shape, keep=SENS_KEEP):
    """
    ``(features, summary)``: features is a ``(len(all_layers), 2)`` CPU tensor holding, for each
    layer that produces a cuttable group's channels, that group's two channels (``normalise``);
    zeros elsewhere. Indices follow ``ModelWithRows(model).all_layers``, which structural cuts
    keep, so one measurement on the origin serves the whole episode. The state describes every
    cuttable group, whatever ``SPECTRA_PLAN_STRUCTURAL_ONLY`` leaves out of the plans.
    """
    started = time.perf_counter()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    plan = group_plan(mwr, groups, structural_only=False)
    sens, base = group_sensitivity(model, plan, batches, input_shape, keep)
    channels = normalise(sens)
    by_group = {id(group): channels[row] for group, row in plan if row in channels}
    features = torch.zeros(len(mwr.all_layers), 2)
    for index, layer in enumerate(mwr.all_layers):
        group = channel_groups.group_of(groups, layer)
        if group is not None and id(group) in by_group:
            features[index] = torch.tensor(by_group[id(group)])
    values = list(sens.values())
    summary = {"groups": len(values), "base_loss": base,
               "median": statistics.median(values) if values else 0.0,
               "min": min(values) if values else 0.0, "max": max(values) if values else 0.0,
               "layers": int((features.abs().sum(dim=1) > 0).sum().item()),
               "seconds": time.perf_counter() - started}
    return features, summary
