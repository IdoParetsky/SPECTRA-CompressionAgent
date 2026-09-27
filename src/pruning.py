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

Filter ranking (``SPECTRA_FILTER_IMPORTANCE``, default ``l1``)
    The DRL *agent* only chooses a compression *rate* for the current layer. Which channels
    die is an environment decision, shared by the learned policy and by the greedy/random
    baselines. That split follows SPA's group-level wrapping of any criterion
    (Wang, Rachwan, Günnemann, Charpentier, "Structurally Prune Anything", arXiv:2403.18955,
    2024) — not a per-model search, and not SPA's OBSPA Hessian (too slow for an RL step).

    ``l1``  Li, Kadav, Durdanovic, Samet, Graf, "Pruning Filters for Efficient ConvNets"
            (ICLR 2017 workshop; arXiv:1608.08710). Default; matches the frozen 10-net actors.
    ``l2``  Per-filter Frobenius / Euclidean norm (He, Kang, Dong, Fu, Yang,
            "Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks",
            IJCAI 2018). Same interface, no extra data.
    ``svd`` Nuclear norm (sum of singular values) of each filter unfolded to (Cin × spatial).
            Weight-only score in the family of Pham, Zniyed, Nguyen, "Singular values-driven
            automated filter pruning", Neural Networks 192:107857, 2025
            (doi:10.1016/j.neunet.2025.107857). We take *only* the per-filter nuclear score,
            not SLIMING's combinatorial search over layer-wise rates — that search would
            replace SPECTRA's offline agent.
    ``fpgm`` He, Kang, Dong, Fu, Yang, "Filter Pruning via Geometric Median for Deep
            Convolutional Neural Networks Acceleration" (CVPR 2019). Survival score is the
            sum of L2 distances to other filters in the layer (far from the geometric median
            → keep). Weight-only; no extra data. Default stays ``l1`` so the frozen agents
            keep the ranking they were trained with.
    ``bn_scale`` Liu, Li, Shen, Huang, "Learning Efficient Convolutional Networks through
            Network Slimming" (ICCV 2017). Survival score is ``|γ|`` of the BatchNorm that
            immediately follows the conv. Falls back to L1 when there is no matching BN.
            Call ``bind_bn_scales(model)`` before ranking (NetworkEnv does this).

    Poplar / metaheuristic channel search, MLPruner, DAGP, flow-guided pruning, and
    activation spectral-entropy scores are *per-model* solvers (they need a new search or
    extra forwards on each CNN). They would replace SPECTRA rather than sit under it.
"""

from typing import Dict, Optional
import os

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


def normalize_importance_mode(raw) -> str:
    """Canonical ranking name; unknown values fall back to L1 so a typo cannot silent-skip prune."""
    raw = (raw or "l1").strip().lower()
    if raw in ("l2", "frobenius", "euclidean"):
        return "l2"
    if raw in ("svd", "nuclear", "spectral"):
        return "svd"
    if raw in ("fpgm", "geometric_median", "geometric-median"):
        return "fpgm"
    if raw in ("bn_scale", "bn-scale", "bn", "slimming", "network_slimming"):
        return "bn_scale"
    if raw in ("taylor", "taylor_fo", "first_order", "molchanov"):
        return "taylor"
    return "l1"


_TAYLOR_SCORES: Dict[int, torch.Tensor] = {}


def bind_taylor_scores(model: nn.Module, loader, device=None, n_batches: int = 1,
                       loss_fn: Optional[nn.Module] = None) -> int:
    """
    First-order Taylor filter importance (Molchanov et al., ICLR 2017 / CVPR 2019 variant on
    weights): per output filter, ``|Σ_w w · ∂L/∂w|`` accumulated over ``n_batches`` mini-batches
    of ``loader`` — the first-order estimate of the loss change from removing that filter.

    Data-dependent, unlike L1/L2/SVD/FPGM/BN-scale; costs one forward+backward per bound batch.
    Call right before ranking with that criterion (``NetworkEnv.step`` does so when the chosen
    ranking is ``taylor``); scores are keyed by ``id(layer)`` and cleared on every call, so a
    resized model never reads stale scores. Returns the number of layers bound.
    """
    _TAYLOR_SCORES.clear()
    if model is None or loader is None:
        return 0
    model = ddp.unwrap(model)
    device = device or next(model.parameters()).device
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()  # BN uses running stats; gradients still flow to weights
    grads: Dict[int, torch.Tensor] = {}
    params = [(m, m.weight) for m in model.modules()
              if isinstance(m, PRUNABLE_TYPES) and getattr(m, "weight", None) is not None]
    for _, w in params:
        w.requires_grad_(True)
    seen = 0
    try:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if y.dim() > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)
            model.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y.long())
            loss.backward()
            for m, w in params:
                if w.grad is None:
                    continue
                contrib = (w.detach() * w.grad.detach()).reshape(w.size(0), -1).sum(dim=1)
                grads[id(m)] = grads.get(id(m), 0) + contrib
            seen += 1
            if seen >= max(1, int(n_batches)):
                break
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    for m, _ in params:
        if id(m) in grads:
            _TAYLOR_SCORES[id(m)] = grads[id(m)].abs()
    return len(_TAYLOR_SCORES)


def _taylor_per_filter(layer: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """Bound Taylor score; L1 fallback when no score is bound for this layer. Dead filters 0."""
    score = _TAYLOR_SCORES.get(id(layer))
    cout = int(weight.size(0))
    dead = weight.reshape(cout, -1).abs().sum(dim=1) <= 0
    if score is None or int(score.numel()) != cout:
        scores = weight.reshape(cout, -1).abs().sum(dim=1)
    else:
        scores = score.to(device=weight.device, dtype=weight.dtype)
    scores = scores.clone()
    scores[dead] = 0
    return scores


def filter_importance_mode(mode: Optional[str] = None) -> str:
    """
    Ranking in force: an explicit ``mode`` (the agent's *action* may carry one when the
    action menu is ``(rate, ranking)`` pairs — ``--action_rankings``), else the environment
    default ``SPECTRA_FILTER_IMPORTANCE``.
    """
    if mode:
        return normalize_importance_mode(mode)
    return normalize_importance_mode(os.environ.get("SPECTRA_FILTER_IMPORTANCE", "l1"))


_BN_ABS_GAMMA: Dict[int, torch.Tensor] = {}


def bind_bn_scales(model: Optional[nn.Module]) -> None:
    """
    Pair each Conv/Linear with the next BatchNorm in module order (Network Slimming).

    Required before ``bn_scale`` ranking. Safe to call on every prune: the map is rebuilt
    from current weights, including after a structural resize.
    """
    _BN_ABS_GAMMA.clear()
    if model is None:
        return
    mods = list(model.modules())
    for i, module in enumerate(mods):
        if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue
        out_n = int(module.out_channels) if hasattr(module, "out_channels") else int(module.out_features)
        for nxt in mods[i + 1 : i + 8]:
            if isinstance(nxt, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                gamma = nxt.weight.detach().abs()
                if int(gamma.numel()) == out_n:
                    _BN_ABS_GAMMA[id(module)] = gamma
                break
            if isinstance(nxt, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                break


def filter_importance(layer: nn.Module, mode: Optional[str] = None) -> torch.Tensor:
    """
    Per-output-filter score; higher = more likely to survive.

    Default is L1 magnitude (Li et al. 2017). ``l2`` / ``svd`` / ``fpgm`` are drop-in
    weight-only criteria at group level (SPA 2024). ``bn_scale`` uses the following BN's
    ``|γ|`` when ``bind_bn_scales`` has paired it. All of these are zero iff the filter is
    all zeros, so ``alive_filters`` stays well-defined. ``mode`` overrides the environment
    default for one call (ranking chosen by the action).
    """
    weight = layer.weight.detach()
    mode = filter_importance_mode(mode)
    if mode == "l2":
        return weight.reshape(weight.size(0), -1).pow(2).sum(dim=1).sqrt()
    if mode == "svd":
        return _nuclear_per_filter(weight)
    if mode == "fpgm":
        return _fpgm_per_filter(weight)
    if mode == "bn_scale":
        return _bn_scale_per_filter(layer, weight)
    if mode == "taylor":
        return _taylor_per_filter(layer, weight)
    return weight.reshape(weight.size(0), -1).abs().sum(dim=1)


def _fpgm_per_filter(weight: torch.Tensor) -> torch.Tensor:
    """Sum of L2 distances to other filters (He et al. CVPR 2019 FPGM). Dead filters score 0."""
    cout = int(weight.size(0))
    flat = weight.reshape(cout, -1)
    alive = flat.abs().sum(dim=1) > 0
    dist = torch.cdist(flat.to(dtype=torch.float32), flat.to(dtype=torch.float32), p=2)
    scores = dist.sum(dim=1).to(dtype=weight.dtype)
    return scores * alive.to(dtype=weight.dtype)


def _bn_scale_per_filter(layer: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """|γ| of the paired BN; L1 fallback when no BN is bound or widths differ."""
    gamma = _BN_ABS_GAMMA.get(id(layer))
    cout = int(weight.size(0))
    dead = weight.reshape(cout, -1).abs().sum(dim=1) <= 0
    if gamma is None or int(gamma.numel()) != cout:
        scores = weight.reshape(cout, -1).abs().sum(dim=1)
    else:
        scores = gamma.to(device=weight.device, dtype=weight.dtype)
    scores = scores.clone()
    scores[dead] = 0
    return scores


def _nuclear_per_filter(weight: torch.Tensor) -> torch.Tensor:
    """Sum of singular values of each output filter as a matrix (Cin × spatial)."""
    cout = int(weight.size(0))
    if weight.dim() <= 2:
        # Linear / 1-D: a row's only singular value is its Euclidean norm.
        return weight.reshape(cout, -1).norm(dim=1)
    cin = int(weight.size(1))
    unfolded = weight.reshape(cout, cin, -1)
    # svdvals is defined on the last two dims; fully zero filters yield 0.
    return torch.linalg.svdvals(unfolded).sum(dim=1)


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


def select_surviving_filters(layer: nn.Module, compression_rate: float,
                             mode: Optional[str] = None) -> torch.Tensor:
    """
    Choose which output filters to keep.

    The target width is a fraction of the filters that are still alive, so repeated
    compression of the same layer compounds as the caller expects.
    """
    importance = filter_importance(layer, mode)
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


def group_importance(group, mode: Optional[str] = None) -> Optional[torch.Tensor]:
    """
    Importance of each channel position of a coupled group.

    DepGraph (Fang, Ma, Song, Mi, Wang, "DepGraph: Towards Any Structural Pruning",
    CVPR 2023, arXiv:2301.12900) requires coupled parameters to be *consistently*
    unimportant, not merely unimportant in one layer of the residual/concat group. SPA
    (Wang et al., arXiv:2403.18955) then wraps any per-filter criterion into that group.
    SPECTRA does both: every producer votes on the same channel index via
    ``filter_importance``, each vote is normalised by its own maximum (so a 3x3 conv cannot
    outvote a 1x1 shortcut just by having larger weights), and the votes are summed.
    """
    votes = []
    for producer in list(group.producers) + list(group.depthwise):
        importance = filter_importance(producer, mode)
        if importance.numel() != group.width:
            return None
        votes.append(importance / importance.max().clamp(min=1e-12))
    if not votes:
        return None
    return torch.stack(votes).sum(dim=0)


def select_group_survivors(group, compression_rate: float,
                           mode: Optional[str] = None) -> Optional[torch.Tensor]:
    """Channel indices to retain for a whole coupled group."""
    importance = group_importance(group, mode)
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


def _survivors_of_width(group, k: int, device, mode: Optional[str] = None) -> Optional[torch.Tensor]:
    importance = group_importance(group, mode)
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


def prune_group_structurally(model_with_rows, group, keep_idx: torch.Tensor,
                             mode: Optional[str] = None) -> bool:
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
            trial = keep_idx if k == target_k else _survivors_of_width(group, k, keep_idx.device, mode)
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
    norm_slices = {}
    for ref in group.norms:
        norm = ref.module
        if norm.num_features != ref.total or id(norm) not in index_of:
            return False
        kept = _replay_consumer_indices(ref.units, new_units.get(id(norm)), keep_idx.device)
        if kept is None:
            kept = surviving_input_channels(ref, group.width, keep_idx)
        norm_edits.append((index_of[id(norm)], _clone_norm(norm, kept)))
        group_channels = set(int(p) for p in ref.positions)
        norm_slices[index_of[id(norm)]] = [
            j for j, old in enumerate(kept.detach().cpu().tolist()) if int(old) in group_channels]

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

    # P8 bookkeeping: which all_layers indices were rewritten and, for every consumer, which
    # positions of its *new* input read the group's surviving channels. reinit_group_edit
    # uses this to install NEON's "new layer" (fresh producers, reset norms, fresh consumer
    # input slices) instead of the surviving pretrained filters.
    consumer_slices = {}
    for ref in group.consumers:
        consumer = ref.module
        if id(consumer) not in edits:
            continue
        in_idx = edits[id(consumer)][1]
        if in_idx is None:
            continue
        group_inputs = set(int(p) for p in ref.positions)
        if not isinstance(consumer, nn.Conv2d) and consumer.in_features != ref.total:
            expanded = _expand_indices_for_flatten(
                torch.tensor(sorted(group_inputs), dtype=torch.long), ref.total, consumer.in_features)
            group_inputs = set(int(p) for p in expanded.tolist()) if expanded is not None else set()
        positions = [j for j, old in enumerate(in_idx.detach().cpu().tolist()) if int(old) in group_inputs]
        consumer_slices[index_of[id(consumer)]] = positions
    consumer_in_idx = {}
    consumer_group_positions = {}
    for ref in group.consumers:
        consumer = ref.module
        if id(consumer) not in edits or id(consumer) not in index_of:
            continue
        in_idx = edits[id(consumer)][1]
        if in_idx is None:
            continue
        consumer_in_idx[index_of[id(consumer)]] = [int(v) for v in in_idx.detach().cpu().tolist()]
        # Old input positions of *every* channel of this group inside the consumer's input
        # (C-PCA projects that whole slice; a concat consumer keeps its other slices).
        consumer_group_positions[index_of[id(consumer)]] = sorted(int(p) for p in ref.positions)
    group_edit = {
        "producers": sorted(index_of[id(m)] for m in group.producers if id(m) in edits),
        "depthwise": sorted(index_of[id(m)] for m in group.depthwise if id(m) in edits),
        "norms": norm_slices,
        "consumers": consumer_slices,
        "consumer_in_idx": consumer_in_idx,
        "consumer_group_positions": consumer_group_positions,
        "keep_idx": [int(v) for v in keep_idx.detach().cpu().tolist()],
        "old_width": int(group.width),
        "new_width": int(keep_idx.numel()),
    }

    edited_param_ids = []
    for idx, new_layer in replacements + norm_edits:
        model_with_rows.replace_layer(idx, new_layer)
        edited_param_ids.extend(id(param) for param in new_layer.parameters())
    model_with_rows.last_edited_param_ids = edited_param_ids
    model_with_rows.last_group_edit = group_edit
    return True


# --------------------------------------------------------------------------- P8: NEON layer replacement

def _kaiming_like(weight: torch.Tensor) -> torch.Tensor:
    """A fresh tensor shaped like ``weight`` drawn the way PyTorch would for a new layer."""
    fresh = torch.empty_like(weight)
    if weight.dim() >= 2:
        nn.init.kaiming_normal_(fresh, mode="fan_out" if weight.dim() == 4 else "fan_in",
                                nonlinearity="relu")
    else:
        fresh.zero_()
    return fresh


def reinit_group_edit(model_with_rows, edit: Optional[dict], scope: str = "group") -> dict:
    """
    NEON "layer replacement" on the group that ``prune_group_structurally`` just resized.

    ``scope="group"`` (NEON-source-literal) re-draws producers, norms and consumer input
    slices; ``scope="producers"`` (Gilad's oral wording) re-draws producers and norms only and
    leaves the consumers' surviving input slices to adapt by training.

    The structural prune installed modules that *keep* the surviving pretrained filters —
    the neuron-removal NEON explicitly rejected (Hirsch & Katz 2022, Sec. 3). This throws
    those weights away and leaves a randomly initialised layer of the new width in place:

    * producers / depthwise owners: full re-initialisation (kaiming, zero bias) — this is
      NEON's ``nn.Linear(in, new_size)``;
    * group norms: affine reset to (1, 0), running stats reset — NEON's fresh ``BatchNorm1d``;
    * consumers: the input slice that reads the group's channels is re-drawn (kaiming scale
      of the whole weight). When the consumer reads *only* this group (plain chains, residual
      streams) that is the whole weight — NEON's ``nn.Linear(new_size, out)``. A concat
      consumer (DenseNet) keeps the slices that read other groups.

    Returns a summary dict (counts) for the step record. No-op on ``None`` (masked / identity).
    """
    summary = {"reinit": False, "producers": 0, "norms": 0, "consumers_full": 0,
               "consumers_slice": 0, "params_reinit": 0, "scope": scope}
    if not edit:
        return summary
    consumer_items = (edit.get("consumers") or {}).items() if scope != "producers" else ()
    layers = model_with_rows.all_layers
    with torch.no_grad():
        for idx in list(edit.get("producers", [])) + list(edit.get("depthwise", [])):
            module = layers[idx]
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            module.weight.copy_(_kaiming_like(module.weight))
            summary["params_reinit"] += module.weight.numel()
            if module.bias is not None:
                module.bias.zero_()
                summary["params_reinit"] += module.bias.numel()
            summary["producers"] += 1
        norms = edit.get("norms") or {}
        norm_items = norms.items() if isinstance(norms, dict) else ((i, None) for i in norms)
        for idx, positions in norm_items:
            norm = layers[int(idx)]
            n_feat = int(getattr(norm, "num_features", 0) or 0)
            full = positions is None or len(positions) >= n_feat
            if full:
                if hasattr(norm, "reset_running_stats"):
                    norm.reset_running_stats()
                if getattr(norm, "affine", False):
                    norm.weight.fill_(1.0)
                    norm.bias.zero_()
                    summary["params_reinit"] += norm.weight.numel() + norm.bias.numel()
            else:
                # A norm over a concat sees this group as one slice; the other branches keep
                # their statistics (only the replaced channels are fresh).
                if not positions:
                    continue
                pos = torch.tensor(sorted(int(p) for p in positions), dtype=torch.long,
                                   device=norm.weight.device if getattr(norm, "affine", False)
                                   else norm.running_mean.device)
                if getattr(norm, "track_running_stats", False) and norm.running_mean is not None:
                    norm.running_mean[pos] = 0.0
                    norm.running_var[pos] = 1.0
                if getattr(norm, "affine", False):
                    norm.weight[pos] = 1.0
                    norm.bias[pos] = 0.0
                    summary["params_reinit"] += 2 * int(pos.numel())
            summary["norms"] += 1
        for idx, positions in consumer_items:
            module = layers[int(idx)]
            if not isinstance(module, (nn.Conv2d, nn.Linear)) or not positions:
                continue
            in_dim = module.weight.size(1)
            grouped = isinstance(module, nn.Conv2d) and module.groups > 1
            if grouped and len(positions) < module.in_channels:
                # weight dim 1 of a grouped conv is per-group; a partial input slice has no
                # column-wise image there. Leave the surviving weights (recipe-A behaviour
                # for this consumer) and say so.
                summary["consumers_skipped"] = summary.get("consumers_skipped", 0) + 1
                continue
            fresh = _kaiming_like(module.weight)
            if len(positions) >= in_dim or grouped:
                module.weight.copy_(fresh)
                if module.bias is not None:
                    module.bias.zero_()
                    summary["params_reinit"] += module.bias.numel()
                summary["consumers_full"] += 1
                summary["params_reinit"] += module.weight.numel()
            else:
                pos = torch.tensor(sorted(int(p) for p in positions), dtype=torch.long,
                                   device=module.weight.device)
                module.weight[:, pos] = fresh[:, pos]
                summary["consumers_slice"] += 1
                summary["params_reinit"] += module.weight[:, pos].numel()
    summary["reinit"] = summary["producers"] + summary["norms"] + summary["consumers_full"] + summary["consumers_slice"] > 0
    return summary


