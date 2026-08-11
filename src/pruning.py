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

    target = int(np.ceil(compression_rate * alive.numel()))
    target = max(1, min(target, alive.numel()))

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

    Needed so that masked (fallback) pruning is not reported as zero compression.
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
    the pruned group as one slice of a wider input. Everything before and after that slice is
    untouched; only the slice itself is filtered, and it is filtered at its true offset.
    """
    device = keep_idx.device
    before = torch.arange(ref.offset, device=device)
    inside = ref.offset + keep_idx
    after = torch.arange(ref.offset + group_width, ref.total, device=device)
    return torch.cat([before, inside, after])


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

    target = max(1, min(int(np.ceil(compression_rate * alive.numel())), alive.numel()))
    best = torch.topk(importance[alive], k=target).indices
    return torch.sort(alive[best]).values


def prune_group_structurally(model_with_rows, group, keep_idx: torch.Tensor) -> bool:
    """
    Shrink every layer tied to a coupled channel group in one consistent edit.

    All producers of the group (a residual block's ``conv2`` together with whatever feeds
    the shortcut) drop the same channel indices, the normalisations over that dimension are
    resized, and every consumer's input dimension is sliced to match. This is what allows a
    residual-coupled convolution to actually shrink instead of merely being masked.

    Returns:
        bool: False when the edit cannot be expressed (unexpected widths); the model is left
              untouched because all replacements are prepared before any is applied.
    """
    if not group.prunable or keep_idx.numel() == group.width:
        return False

    index_of = {id(layer): idx for idx, layer in enumerate(model_with_rows.all_layers)}
    # module -> (out_idx, in_idx); a layer can be both a producer and a consumer of the
    # same group (e.g. a same-width residual convolution), so edits are merged per module
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
        kept = surviving_input_channels(ref, group.width, keep_idx)

        if isinstance(consumer, nn.Conv2d):
            if consumer.in_channels != ref.total or not stage(consumer, in_idx=kept):
                return False
        else:
            # A Linear behind a flatten sees each channel as a block of spatial positions
            in_idx = (kept if consumer.in_features == ref.total
                      else _expand_indices_for_flatten(kept, ref.total, consumer.in_features))
            if in_idx is None or not stage(consumer, in_idx=in_idx):
                return False

    norm_edits = []
    for ref in group.norms:
        norm = ref.module
        if norm.num_features != ref.total or id(norm) not in index_of:
            return False
        norm_edits.append((index_of[id(norm)], _clone_norm(norm, surviving_input_channels(ref, group.width, keep_idx))))

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

    for idx, new_layer in replacements + norm_edits:
        model_with_rows.replace_layer(idx, new_layer)
    return True


