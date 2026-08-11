"""
What each candidate compression rate would actually cost.

The state described in "Extending BERT Input Mechanisms" characterises what the network *is*
-- topology, activation statistics, weight statistics -- but never what the available actions
would *do*. Choosing a compression rate is a cost/benefit decision, and the cost side is
exactly computable in advance.

It is also not a property of the target layer alone. Pruning a residual-coupled convolution
also shrinks every other producer of its dependency group and narrows every layer that reads
that group, so two layers with identical statistics can differ by an order of magnitude in
what pruning them removes. Making that observable turns a latent inference problem into an
input feature.

For each rate the agent can pick, this module reports the fraction of the whole network's
parameters and MACs that selecting it would eliminate.
"""

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

import src.channel_groups as channel_groups
import src.pruning as pruning
import src.utils as utils

# Each action token carries (compression_rate, parameter_fraction, mac_fraction)
ACTION_FEATURE_DIM = 3


def _parameters_per_output(layer: nn.Module) -> int:
    """Parameter count owned by a single output filter/neuron of `layer`."""
    if isinstance(layer, nn.Conv2d):
        per_filter = (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
    else:
        per_filter = layer.in_features
    return per_filter + (1 if layer.bias is not None else 0)


def _parameters_per_input(layer: nn.Module, removed_inputs: int) -> int:
    """Parameters a layer sheds when `removed_inputs` of its input channels disappear."""
    if isinstance(layer, nn.Conv2d):
        return removed_inputs * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] // layer.groups
    return removed_inputs * layer.out_features


def group_removal_cost(group, removed: int, macs: Dict[int, float]) -> (int, float):
    """
    Parameters and MACs eliminated by dropping `removed` channels from a coupled group.

    MACs are linear in a layer's output width and, separately, in its input width, so each
    module's measured cost can simply be scaled by the fraction of that dimension removed.
    """
    if group.width <= 0 or removed <= 0:
        return 0, 0.0

    fraction = removed / group.width
    params = 0
    flops = 0.0

    for producer in list(group.producers) + list(group.depthwise):
        params += removed * _parameters_per_output(producer)
        flops += macs.get(id(producer), 0.0) * fraction

    for ref in group.norms:
        norm = ref.module
        params += removed * (2 if getattr(norm, "affine", False) else 0)
        flops += macs.get(id(norm), 0.0) * (removed / max(ref.total, 1))

    for ref in group.consumers:
        consumer = ref.module
        if isinstance(consumer, nn.Linear) and consumer.in_features != ref.total:
            # A Linear behind a flatten sees each channel as a block of spatial positions
            spatial = consumer.in_features // ref.total if ref.total else 1
            removed_inputs = removed * spatial
            total_inputs = consumer.in_features
        else:
            removed_inputs = removed
            total_inputs = ref.total

        params += _parameters_per_input(consumer, removed_inputs)
        flops += macs.get(id(consumer), 0.0) * (removed_inputs / max(total_inputs, 1))

    return params, flops


def estimate_action_costs(model: nn.Module, target_layer: nn.Module,
                          compression_rates: Sequence[float], input_shape,
                          groups: Optional[List] = None,
                          device=None) -> torch.Tensor:
    """
    Cost of every action available at the current state.

    Args:
        model (nn.Module):           Network being compressed.
        target_layer (nn.Module):    Layer the agent is about to act on.
        compression_rates (Sequence[float]): The action space, in action-index order.
        input_shape (tuple):         Per-sample input shape, needed for MAC accounting.
        groups (list, optional):     Precomputed dependency groups, to avoid re-tracing.
        device (torch.device):       Device for the returned tensor.

    Returns:
        torch.Tensor: (num_actions, 3) of (rate, parameter fraction removed, MAC fraction
        removed). Fractions are of the *whole network*, so they are comparable across layers
        and across architectures.
    """
    model = pruning.ddp_unwrap(model)
    device = device or next(model.parameters()).device
    rates = torch.tensor(list(compression_rates), dtype=torch.float32, device=device)
    costs = torch.zeros(len(compression_rates), ACTION_FEATURE_DIM, device=device)
    costs[:, 0] = rates

    total_params = sum(p.numel() for p in model.parameters())
    if total_params == 0:
        return costs

    try:
        macs = utils.per_module_macs(model, input_shape, device)
    except Exception:  # a probe pass can fail on exotic architectures; features degrade to 0
        return costs
    total_macs = sum(macs.values()) or 1.0

    if groups is None:
        groups = channel_groups.build_channel_groups(model) or []
    group = channel_groups.group_of(groups, target_layer)

    for index, rate in enumerate(compression_rates):
        if group is not None and group.prunable:
            keep = pruning.select_group_survivors(group, rate)
            if keep is None:
                continue
            removed = group.width - keep.numel()
            params, flops = group_removal_cost(group, removed, macs)
        else:
            # Masking removes weights but no shapes, so the MAC count is unchanged
            keep = pruning.select_surviving_filters(target_layer, rate)
            removed = pruning.layer_width(target_layer) - keep.numel()
            params, flops = removed * _parameters_per_output(target_layer), 0.0

        costs[index, 1] = params / total_params
        costs[index, 2] = flops / total_macs

    return costs
