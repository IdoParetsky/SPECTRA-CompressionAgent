"""
Per-group sensitivity channels for the agent's state (``SPECTRA_STATE_SENS``, v10).

A0 (ledger §201 / §204 / §205): at equal kept parameters, keeping more of the groups whose cut
hurts most beat a uniform allocation after a 40-epoch recovery on both ResNet-56 cells. The rule
needs one cheap number per coupled group: the calibration-loss rise when that group alone is cut
to half its width, without fine-tuning (Li et al., ICLR 2017, per-layer sensitivity). This is
``scripts/allocation_probe.py``'s measurement, taken once per episode on the origin network and
given to every layer that produces the group's channels.
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


def group_plan(model_with_rows, groups=None):
    """``[(group, row)]``: each coupled group the walk can cut, at its first row, in walk order."""
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
    return plan


def group_sensitivity(model, plan, batches, input_shape, keep=SENS_KEEP):
    """``({row: loss rise}, base loss)``: each planned group cut alone to ``keep`` (L1 survivors)."""
    from src.NetworkEnv import prune_current_model
    base = calib_loss(model, batches)
    out = {}
    for _group, row in plan:
        cut = prune_current_model(ModelWithRows(copy.deepcopy(model)), keep, row, quiet=True, record=False,
                                  input_shape=input_shape, importance="l1")
        out[row] = calib_loss(cut.model, batches) - base
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
    keep, so one measurement on the origin serves the whole episode.
    """
    started = time.perf_counter()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    plan = group_plan(mwr, groups)
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
