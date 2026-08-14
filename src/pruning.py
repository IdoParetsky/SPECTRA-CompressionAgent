"""
Structured pruning primitives for SPECTRA.

Two compression strategies live here:

``prune_group_structurally``
    Physically removes channels from every layer of a coupled dependency group (see
    src/channel_groups.py) and resizes the normalisations and consumers that read them.
    Only this path actually reduces parameters and FLOPs.

``mask_layer_filters``
    Zeroes the least important filters in place, leaving tensor shapes untouched. Used as a
    fallback wherever the dependency group cannot be resized -- a concatenation, an
    unrecognised module, or a model that cannot be symbolically traced. Masking keeps the
    network runnable but does not shrink it, so the reporting helpers count masked filters
    as removed.

Note on compounding: importance is always computed over the *currently alive* filters,
so applying rate ``r`` twice leaves ``r^2`` of the original width. ``torch.nn.utils.prune``
ranks all filters including previously zeroed ones, whose L1 norm is 0, so it re-selects
them first and the second pass removes almost nothing new.
"""

from typing import Optional

import numpy as np
import torch
from torch import nn

import src.channel_groups as channel_groups
import src.distributed as ddp
import src.utils as utils

PRUNABLE_TYPES = (nn.Conv2d, nn.Linear)
NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)

# Re-exported so callers can unwrap replicas without importing the distributed helper
ddp_unwrap = ddp.unwrap


def layer_width(layer: nn.Module) -> int:
    """Number of output filters (Conv2d) or neurons (Linear)."""
    return layer.out_channels if isinstance(layer, nn.Conv2d) else layer.out_features


def filter_importance(layer: nn.Module) -> torch.Tensor:
    """L1 norm of each output filter, flattened over its input/spatial dimensions."""
    weight = layer.weight.detach()
    return weight.reshape(weight.size(0), -1).abs().sum(dim=1)


def alive_filters(layer: nn.Module) -> torch.Tensor:
    """Indices of filters that are not entirely zero (i.e. not previously masked out)."""
    return torch.nonzero(filter_importance(layer) > 0, as_tuple=False).flatten()


def target_width(alive_count: int, compression_rate: float) -> int:
    """
    How many channels a compression rate should leave alive.

    This used to be ``ceil(rate * alive)``, which silently turned small rates on narrow
    layers into no-ops: ``ceil(0.9 * 6) == 6`` keeps every filter, so the environment applied
    nothing while the agent was still rewarded or punished for the action. An action that
    cannot change the environment is pure noise in the RL signal, and on the thin ResNets in
    the database (widths of 6-16 channels) it fired constantly.

    The target is now the nearest width, with the guarantee that asking for *any* compression
    removes at least one channel.
    """
    if alive_count <= 1 or compression_rate >= 1.0:
        return max(alive_count, 1)
    target = int(round(compression_rate * alive_count))
    return max(1, min(target, alive_count - 1))


def select_surviving_filters(layer: nn.Module, compression_rate: float) -> torch.Tensor:
    """
    Choose which output filters to keep.

    The target width is a fraction of the filters that are still alive, so repeated
    compression of the same layer compounds as the caller expects.
    """
    importance = filter_importance(layer)
    alive = torch.nonzero(importance > 0, as_tuple=False).flatten()
    if alive.numel() == 0:  # fully masked already; keep one filter to stay runnable
        return torch.zeros(1, dtype=torch.long, device=importance.device)

    target = target_width(alive.numel(), compression_rate)

    # Highest-importance survivors, restored to ascending order so channel order is stable
    best = torch.topk(importance[alive], k=target).indices
    return torch.sort(alive[best]).values


def mask_layer_filters(layer: nn.Module, keep_idx: torch.Tensor) -> None:
    """Zero every filter outside `keep_idx`, preserving the layer's shape."""
    with torch.no_grad():
        mask = torch.zeros(layer_width(layer), dtype=torch.bool, device=layer.weight.device)
        mask[keep_idx] = True
        layer.weight[~mask] = 0
        if layer.bias is not None:
            layer.bias[~mask] = 0


def count_effective_parameters(model: nn.Module) -> int:
    """
    Parameter count that treats structurally-zero filters as removed.

    Needed so that *masked* fallback can be compared against structural ``numel`` counts.
    Eval ``param_ratio`` uses ``calc_num_parameters`` (shapes) and does **not** treat
    masked zeros as removed; quoting this helper as the thesis compression number
    overstates real size cut.
    """
    total = 0
    for module in model.modules():
        if isinstance(module, PRUNABLE_TYPES):
            weight = module.weight.detach()
            per_filter = weight.reshape(weight.size(0), -1)
            alive = (per_filter.abs().sum(dim=1) > 0)
            total += int(alive.sum().item()) * per_filter.size(1)
            if module.bias is not None:
                total += int(alive.sum().item())
        else:
            total += sum(p.numel() for p in module.parameters(recurse=False))
    return total


def _clone_conv(layer: nn.Conv2d, out_idx: torch.Tensor, in_idx: Optional[torch.Tensor],
                depthwise: bool = False) -> nn.Conv2d:
    weight = layer.weight.detach()
    if in_idx is not None and not depthwise:
        weight = weight[:, in_idx]
    weight = weight[out_idx]

    # A depthwise convolution holds one filter per input channel, so slicing the output
    # rows already slices the input channels; `groups` must shrink with them
    groups = out_idx.numel() if depthwise else layer.groups

    new_layer = nn.Conv2d(
        in_channels=weight.size(1) * groups,
        out_channels=weight.size(0),
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=groups,
        bias=layer.bias is not None,
        padding_mode=layer.padding_mode,
    ).to(layer.weight.device)

    with torch.no_grad():
        new_layer.weight.copy_(weight)
        if layer.bias is not None:
            new_layer.bias.copy_(layer.bias.detach()[out_idx])
    return new_layer


def _clone_linear(layer: nn.Linear, out_idx: torch.Tensor, in_idx: Optional[torch.Tensor]) -> nn.Linear:
    weight = layer.weight.detach()
    if in_idx is not None:
        weight = weight[:, in_idx]
    weight = weight[out_idx]

    new_layer = nn.Linear(
        in_features=weight.size(1),
        out_features=weight.size(0),
        bias=layer.bias is not None,
    ).to(layer.weight.device)

    with torch.no_grad():
        new_layer.weight.copy_(weight)
        if layer.bias is not None:
            new_layer.bias.copy_(layer.bias.detach()[out_idx])
    return new_layer


def _clone_norm(layer: nn.Module, keep_idx: torch.Tensor) -> nn.Module:
    new_layer = type(layer)(
        num_features=keep_idx.numel(),
        eps=layer.eps,
        momentum=layer.momentum,
        affine=layer.affine,
        track_running_stats=layer.track_running_stats,
    ).to(next(layer.parameters(), torch.zeros(1)).device if layer.affine else keep_idx.device)

    with torch.no_grad():
        if layer.affine:
            new_layer.weight.copy_(layer.weight.detach()[keep_idx])
            new_layer.bias.copy_(layer.bias.detach()[keep_idx])
        if layer.track_running_stats:
            new_layer.running_mean.copy_(layer.running_mean.detach()[keep_idx])
            new_layer.running_var.copy_(layer.running_var.detach()[keep_idx])
            new_layer.num_batches_tracked.copy_(layer.num_batches_tracked.detach())
    return new_layer


def _expand_indices_for_flatten(keep_idx: torch.Tensor, old_width: int, in_features: int) -> Optional[torch.Tensor]:
    """
    Map surviving channels onto the input features of a Linear placed after a flatten.

    A flatten turns (C, H, W) into C*H*W, so channel ``c`` owns the contiguous block
    ``[c * H*W, (c + 1) * H*W)``.
    """
    if in_features % old_width != 0:
        return None
    spatial = in_features // old_width
    offsets = torch.arange(spatial, device=keep_idx.device)
    return (keep_idx.unsqueeze(1) * spatial + offsets.unsqueeze(0)).reshape(-1)


def surviving_input_channels(ref, group_width: int, keep_idx: torch.Tensor) -> torch.Tensor:
    """
    Input channels a consumer retains once one segment of its input has been pruned.

    A layer reading a concatenated tensor (a DenseNet block, an Inception branch merge) sees
    the pruned group as one slice of a wider input. ShuffleNet channel-shuffle scatters that
    slice across even/odd indices; a later ``chunk`` may expose only a subset of the group.
    ``ref.positions[j]`` is the consumer index for producer channel ``ref.producer_idx[j]``.
    """
    positions = list(ref.positions)
    producer_idx = list(getattr(ref, "producer_idx", None) or ())
    if not producer_idx or len(producer_idx) != len(positions):
        if len(positions) == group_width:
            producer_idx = list(range(group_width))
        else:
            # Legacy contiguous slice, clamped so arange cannot invert.
            device = keep_idx.device
            start = max(0, min(int(ref.offset), int(ref.total)))
            end = max(start, min(int(ref.offset) + int(group_width), int(ref.total)))
            before = torch.arange(0, start, device=device)
            span = end - start
            if span <= 0:
                kept = torch.arange(ref.total, device=device)
                return kept if kept.numel() else torch.zeros(1, dtype=torch.long, device=device)
            local = keep_idx[(keep_idx >= 0) & (keep_idx < span)]
            inside = start + local
            after = torch.arange(end, ref.total, device=device)
            parts = [p for p in (before, inside, after) if p.numel()]
            if not parts:
                return torch.zeros(1, dtype=torch.long, device=device)
            return torch.cat(parts)

    keep_local = {int(i) for i in keep_idx.detach().cpu().tolist()}
    drop = {int(pos) for pos, pidx in zip(positions, producer_idx) if int(pidx) not in keep_local}
    kept = [i for i in range(ref.total) if i not in drop]
    if not kept:
        kept = [0]
    return torch.tensor(kept, dtype=torch.long, device=keep_idx.device)


def group_importance(group) -> Optional[torch.Tensor]:
    """
    Importance of each channel position of a coupled group.

    Every producer votes on the same channel index, so their per-filter norms are summed.
    Each producer is normalised by its own maximum first, otherwise a layer with larger
    weights (a 3x3 conv against a 1x1 shortcut) would decide the ranking on its own.
    """
    votes = []
    for producer in list(group.producers) + list(group.depthwise):
        importance = filter_importance(producer)
        if importance.numel() != group.width:
            return None
        votes.append(importance / importance.max().clamp(min=1e-12))
    if not votes:
        return None
    return torch.stack(votes).sum(dim=0)


def select_group_survivors(group, compression_rate: float) -> Optional[torch.Tensor]:
    """Channel indices to retain for a whole coupled group."""
    importance = group_importance(group)
    if importance is None:
        return None

    alive = torch.nonzero(importance > 0, as_tuple=False).flatten()
    if alive.numel() == 0:
        return torch.zeros(1, dtype=torch.long, device=importance.device)

    target = target_width(alive.numel(), compression_rate)
    best = torch.topk(importance[alive], k=target).indices
    return torch.sort(alive[best]).values


def _replay_consumer_indices(old_units, new_units, device):
    """Map post-prune (token, origin) channel order onto pre-prune input indices."""
    if not old_units or new_units is None:
        return None
    index = {unit: i for i, unit in enumerate(old_units)}
    kept = [index.get(unit) for unit in new_units]
    if any(i is None for i in kept):
        return None
    if not kept:
        kept = [0]
    return torch.tensor(kept, dtype=torch.long, device=device)


def _survivors_of_width(group, k: int, device) -> Optional[torch.Tensor]:
    importance = group_importance(group)
    if importance is None:
        return None
    alive = torch.nonzero(importance > 0, as_tuple=False).flatten()
    if alive.numel() == 0:
        return torch.zeros(1, dtype=torch.long, device=importance.device)
    k = max(1, min(int(k), int(alive.numel()) - (1 if alive.numel() > 1 else 0)))
    if alive.numel() == 1:
        return alive
    best = torch.topk(importance[alive], k=k).indices
    return torch.sort(alive[best]).values


def _producer_keep_dict(group, keep_idx):
    kept = [int(i) for i in keep_idx.detach().cpu().tolist()]
    return {id(module): kept for module in list(group.producers) + list(group.depthwise)}


def prune_group_structurally(model_with_rows, group, keep_idx: torch.Tensor) -> bool:
    """
    Shrink every layer tied to a coupled channel group in one consistent edit.

    All producers of the group (a residual block's ``conv2`` together with whatever feeds
    the shortcut) drop the same channel indices, the normalisations over that dimension are
    resized, and every consumer's input dimension is sliced to match. This is what allows a
    residual-coupled convolution to actually shrink instead of merely being masked.

    ShuffleNet channel-shuffle is a permutation of the *current* width, so consumer slices
    are taken from a replay of the FX layout with the kept producer origins, not by deleting
    indices from the pre-prune permutation. Nearby keep-counts are tried when a candidate
    would leave a shuffle/chunk with an odd channel count.

    Returns:
        bool: False when the edit cannot be expressed (unexpected widths); the model is left
              untouched because all replacements are prepared before any is applied.
    """
    if not group.prunable or keep_idx.numel() == group.width:
        return False

    # A later ShuffleNet chunk sees only a subset of this group. One such consumer
    # (MiniShuffle's head) can be resized by replaying the shuffle; a chain of
    # them desynchronizes keep-counts from the real cat/shuffle widths, so mask.
    partial_views = sum(
        1 for ref in group.consumers
        if ref.positions and len(ref.positions) != group.width)
    if partial_views > 1:
        return False

    model = model_with_rows.model
    new_units = {}
    if partial_views == 1:
        target_k = int(keep_idx.numel())
        chosen = None
        for k in [target_k] + [k for delta in range(1, 9) for k in (target_k + delta, target_k - delta)]:
            if k < 1 or k >= group.width:
                continue
            trial = keep_idx if k == target_k else _survivors_of_width(group, k, keep_idx.device)
            if trial is None:
                continue
            flag, units = [], {}
            replayed = channel_groups.build_channel_groups(
                model, producer_keep=_producer_keep_dict(group, trial),
                input_units_out=units, shuffle_ok_out=flag)
            if replayed is not None and flag and flag[0]:
                chosen = trial
                new_units = units
                break
        if chosen is None:
            return False
        keep_idx = chosen

    index_of = {id(layer): idx for idx, layer in enumerate(model_with_rows.all_layers)}
    edits: dict = {}

    def stage(module, out_idx=None, in_idx=None) -> bool:
        if id(module) not in index_of:
            return False
        current_out, current_in = edits.get(id(module), (None, None))
        edits[id(module)] = (out_idx if out_idx is not None else current_out,
                             in_idx if in_idx is not None else current_in)
        return True

    for producer in group.producers:
        if layer_width(producer) != group.width or not stage(producer, out_idx=keep_idx):
            return False

    for module in group.depthwise:
        if layer_width(module) != group.width or not stage(module, out_idx=keep_idx):
            return False

    for ref in group.consumers:
        consumer = ref.module
        kept = _replay_consumer_indices(ref.units, new_units.get(id(consumer)), keep_idx.device)
        if kept is None:
            kept = surviving_input_channels(ref, group.width, keep_idx)

        if isinstance(consumer, nn.Conv2d):
            if consumer.in_channels != ref.total or not stage(consumer, in_idx=kept):
                return False
        else:
            in_idx = (kept if consumer.in_features == ref.total
                      else _expand_indices_for_flatten(kept, ref.total, consumer.in_features))
            if in_idx is None or not stage(consumer, in_idx=in_idx):
                return False

    norm_edits = []
    for ref in group.norms:
        norm = ref.module
        if norm.num_features != ref.total or id(norm) not in index_of:
            return False
        kept = _replay_consumer_indices(ref.units, new_units.get(id(norm)), keep_idx.device)
        if kept is None:
            kept = surviving_input_channels(ref, group.width, keep_idx)
        norm_edits.append((index_of[id(norm)], _clone_norm(norm, kept)))

    depthwise_ids = {id(m) for m in group.depthwise}
    replacements = []
    for module_id, (out_idx, in_idx) in edits.items():
        module = model_with_rows.all_layers[index_of[module_id]]
        if isinstance(module, nn.Conv2d):
            new_module = _clone_conv(module,
                                     out_idx if out_idx is not None else torch.arange(module.out_channels),
                                     in_idx,
                                     depthwise=module_id in depthwise_ids)
        else:
            new_module = _clone_linear(module,
                                       out_idx if out_idx is not None else torch.arange(module.out_features),
                                       in_idx)
        replacements.append((index_of[module_id], new_module))

    edited_param_ids = []
    for idx, new_layer in replacements + norm_edits:
        model_with_rows.replace_layer(idx, new_layer)
        edited_param_ids.extend(id(param) for param in new_layer.parameters())
    model_with_rows.last_edited_param_ids = edited_param_ids
    return True


