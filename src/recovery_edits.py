"""
Closed-form weight edits and the budget-action map. Every caller is default-off.

A-LSQ (``SPECTRA_FT_LSQ_CONSUMERS=1``)
    After a keep-leftover cut, replace each consumer's surviving kernel with the
    least-squares kernel that reproduces that consumer's pre-cut output from the
    kept input channels. This is the reconstruction step of He, Zhang and Sun,
    "Channel Pruning for Accelerating Very Deep Neural Networks", ICCV 2017.
    ThiNet (Luo et al., ICCV 2017) solves the smaller problem of a per-channel
    scale; the solver here fits the remaining kernel, which is He et al.

C-PCA (``SPECTRA_FT_REINIT_EDITED=pca``)
    Build the new producer filters as the top-k principal directions of the
    pre-cut output, and premultiply each consumer by that basis so the linear
    map is the same map restricted to the kept subspace. This is not PCA-Pruner
    (Zhang et al. use PCA only to choose how many filters to drop, then rank the
    old filters by L1). Identity skip-adds are not rotated by a weight edit, so
    the equality is exact on a plain chain and approximate on a residual add.
    The fine-tune that follows is recipe A's full-net fine-tune, so a comparison
    with recipe A is a comparison of the initialisation.

BN recalibration (``SPECTRA_FT_BN_RECAL=1``)
    After the edit, reset every BatchNorm's running statistics and re-estimate
    them with a cumulative average over a few training batches, then restore the
    original momentum. No gradient.

Budget actions (``SPECTRA_ACTION_MENU=budget``)
    A rate in (0, 1) means "remove this fraction of the whole network through
    the current group". A negative rate is STOP: the episode ends and the reward
    is the slack-weighted area accumulated so far. AMC (He et al., ECCV 2018)
    also makes each action a sparsity that is clipped so the network lands on a
    resource budget; it does not add an explicit stop, because it walks every
    layer exactly once. STOP is the part that lets the agent choose the depth.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
from torch import nn

import src.action_costs as action_costs
import src.channel_groups as channel_groups


def calib_images_per_batch(default: int = 32) -> int:
    """Images kept per calibration batch (``SPECTRA_FT_CALIB_IMAGES``, default 32).

    Two images (the first draft) leave a 256-channel 3×3 consumer at 8×8 with 256 positions
    against 2 300 unknowns — an underdetermined fit that memorises noise. 32 images × 2
    batches × 64 positions = 4 096 rows is the floor; the solver below adds a ridge term.
    """
    raw = os.environ.get("SPECTRA_FT_CALIB_IMAGES", "").strip()
    try:
        return max(1, int(raw)) if raw else int(default)
    except ValueError:
        return int(default)


def _subsample_batch(tensor: torch.Tensor, max_images: Optional[int] = None) -> Optional[torch.Tensor]:
    """Keep up to ``max_images`` images. Spatial structure is preserved so a later unfold is valid."""
    if not torch.is_tensor(tensor) or tensor.dim() not in (2, 4):
        return None
    limit = calib_images_per_batch() if max_images is None else int(max_images)
    take = min(int(tensor.shape[0]), limit)
    if take < 1:
        return None
    return tensor[:take].detach().cpu().contiguous()


def _ridge_solve(design: torch.Tensor, target: torch.Tensor, ridge: float) -> Optional[torch.Tensor]:
    """min ‖X w − y‖² + λ‖w‖² in float64 via the normal equations; λ scales with the data."""
    x = design.double()
    y = target.double()
    gram = x.T @ x
    if ridge > 0.0:
        scale = float(gram.diagonal().mean().item()) if gram.numel() else 1.0
        gram = gram + torch.eye(gram.shape[0], dtype=gram.dtype) * (ridge * max(scale, 1e-12))
    try:
        return torch.linalg.solve(gram, x.T @ y).float()
    except RuntimeError:
        try:
            return torch.linalg.lstsq(x, y).solution.float()
        except RuntimeError:
            return None


def _subsample_rows(design: torch.Tensor, target: torch.Tensor, max_rows: int, seed: int = 0):
    if design.shape[0] <= max_rows:
        return design, target
    gen = torch.Generator().manual_seed(seed)
    sel = torch.randperm(design.shape[0], generator=gen)[:max_rows]
    return design[sel], target[sel]


def lsq_conv2d_weight(x: torch.Tensor, y: torch.Tensor, in_keep, conv: nn.Conv2d,
                      max_rows: int = 16384, ridge: float = 1e-5):
    """Least-squares kernel (out, kept_in, kH, kW) reconstructing ``y`` from ``x[:, in_keep]``.

    ``x`` is the consumer's pre-cut input, ``y`` its pre-cut output (the conv's own
    output, before BatchNorm and ReLU). He, Zhang, Sun (ICCV 2017) eq. (5): after channel
    selection, refit the remaining kernel to the original response. Ridge λ = 1e-5 of the
    mean Gram diagonal keeps a wide consumer with few positions from memorising the
    calibration images while leaving a well-determined fit exact to ~1e-4.
    Returns None when the shapes cannot meet.
    """
    if not isinstance(conv, nn.Conv2d) or conv.groups != 1:
        return None
    if x.dim() != 4 or y.dim() != 4:
        return None
    keep = [int(i) for i in in_keep]
    if not keep or max(keep) >= x.shape[1]:
        return None
    x_kept = x[:, keep].float()
    y = y.float()
    unfold = nn.Unfold(kernel_size=conv.kernel_size, dilation=conv.dilation,
                       padding=conv.padding, stride=conv.stride)
    patches = unfold(x_kept)  # (N, kept*kH*kW, L)
    if patches.shape[-1] != y.shape[-2] * y.shape[-1]:
        return None
    design = patches.permute(0, 2, 1).reshape(-1, patches.shape[1])
    target = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1])
    if conv.bias is not None:
        design = torch.cat([design, torch.ones(design.shape[0], 1)], dim=1)
    design, target = _subsample_rows(design, target, max_rows)
    solution = _ridge_solve(design, target, ridge)
    if solution is None:
        return None
    kernel_w = solution[:-1] if conv.bias is not None else solution
    weight = kernel_w.T.reshape(y.shape[1], len(keep), conv.kernel_size[0], conv.kernel_size[1])
    bias = solution[-1] if conv.bias is not None else None
    return weight.contiguous(), bias


def lsq_linear_weight(x: torch.Tensor, y: torch.Tensor, in_keep, linear: nn.Linear,
                      max_rows: int = 16384, ridge: float = 1e-5):
    """Least-squares weight (out, kept_in) for a Linear consumer (e.g. the VGG classifier after flatten).

    ``x`` is the consumer's pre-cut input (N, in_features) and ``y`` its pre-cut output.
    ``in_keep`` are the kept input features (already expanded across the flatten).
    """
    if not isinstance(linear, nn.Linear) or x.dim() != 2 or y.dim() != 2:
        return None
    keep = [int(i) for i in in_keep]
    if not keep or max(keep) >= x.shape[1]:
        return None
    design = x[:, keep].float()
    target = y.float()
    if linear.bias is not None:
        design = torch.cat([design, torch.ones(design.shape[0], 1)], dim=1)
    design, target = _subsample_rows(design, target, max_rows)
    solution = _ridge_solve(design, target, ridge)
    if solution is None:
        return None
    weight = (solution[:-1] if linear.bias is not None else solution).T
    bias = solution[-1] if linear.bias is not None else None
    return weight.contiguous(), bias


def channel_basis(activations: torch.Tensor, k: int, max_positions: int = 4096) -> Optional[torch.Tensor]:
    """Top-k eigenvectors of the channel covariance, shape (channels, k), largest first."""
    if activations.dim() != 4 or k < 1:
        return None
    channels = int(activations.shape[1])
    k = min(int(k), channels)
    flat = activations.float().permute(1, 0, 2, 3).reshape(channels, -1)
    if flat.shape[1] < 2:
        return None
    if flat.shape[1] > max_positions:
        sel = torch.linspace(0, flat.shape[1] - 1, max_positions).long()
        flat = flat[:, sel]
    flat = flat - flat.mean(dim=1, keepdim=True)
    cov = flat @ flat.T / float(max(flat.shape[1] - 1, 1))
    try:
        _evals, evecs = torch.linalg.eigh(cov)
    except RuntimeError:
        return None
    basis = evecs[:, -k:].contiguous()
    # Pin the sign so the same activations always yield the same basis.
    for col in range(basis.shape[1]):
        nz = (basis[:, col].abs() > 1e-8).nonzero(as_tuple=False)
        pivot = int(nz[0]) if nz.numel() else 0
        if float(basis[pivot, col]) < 0:
            basis[:, col] = -basis[:, col]
    return basis


def mix_out_channels(weight: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """New producer filters: each is a combination of the old output channels. ``basis`` is (C_out, k)."""
    return torch.einsum("oihw,ok->kihw", weight.float(), basis.float()).contiguous()


def mix_in_channels(weight: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """New consumer kernel so ``W' (U^T y)`` is ``W`` restricted to the column space of ``U``.

    ``basis`` is (C_in, k). The product ``mix @ U^T`` equals ``W @ U @ U^T``, which is
    ``W`` itself only when every channel is kept.
    """
    return torch.einsum("oihw,ik->okhw", weight.float(), basis.float()).contiguous()


def group_param_fraction(model_with_rows, row_index: int) -> float:
    """Fraction of the network's parameters that a full cut of this row's group removes."""
    model = model_with_rows.model
    total = sum(int(p.numel()) for p in model.parameters()) or 1
    try:
        groups = channel_groups.build_channel_groups(model)
        layer = model_with_rows.all_layers[model_with_rows.row_to_main_layer[row_index]]
        group = channel_groups.group_of(groups, layer) if groups else None
        if group is not None and int(group.width) > 0:
            removed, _flops = action_costs.group_removal_cost(group, int(group.width), {})
            if removed > 0:
                return min(1.0, float(removed) / float(total))
    except Exception:
        pass
    layer = model_with_rows.all_layers[model_with_rows.row_to_main_layer[row_index]]
    owned = sum(int(p.numel()) for p in layer.parameters())
    return min(1.0, float(owned) / float(total))


def capture_pre_prune(model_with_rows, row_index: int, loader, device, n_batches: int = 2) -> Optional[dict]:
    """One or two training batches of the group's producer/consumer inputs and outputs.

    Keys are ``all_layers`` indices, which structural replacement keeps stable.
    Returns None when the group cannot be traced; the caller then skips the edit.
    """
    try:
        groups = channel_groups.build_channel_groups(model_with_rows.model)
    except Exception:
        return None
    if not groups:
        return None
    main_idx = model_with_rows.row_to_main_layer[row_index]
    main = model_with_rows.all_layers[main_idx]
    group = channel_groups.group_of(groups, main)
    if group is None:
        return None
    index_of = {id(layer): i for i, layer in enumerate(model_with_rows.all_layers)}
    wanted: Dict[int, nn.Module] = {}
    producer_ids = []
    for module in list(group.producers) + list(group.depthwise):
        if id(module) in index_of and isinstance(module, (nn.Conv2d, nn.Linear)):
            idx = index_of[id(module)]
            wanted[idx] = module
            producer_ids.append(idx)
    for ref in group.consumers:
        module = ref.module
        if id(module) in index_of and isinstance(module, (nn.Conv2d, nn.Linear)):
            wanted[index_of[id(module)]] = module
    if not wanted:
        return None

    captured_io: Dict[int, dict] = {idx: {"x": [], "y": []} for idx in wanted}
    hooks = []

    def _hook(idx):
        def fn(_module, inputs, output):
            # One record per calibration batch (the loop below stops after n_batches).
            x = inputs[0] if inputs else None
            y = output if torch.is_tensor(output) else None
            xs = _subsample_batch(x)
            ys = _subsample_batch(y)
            if xs is not None and ys is not None:
                captured_io[idx]["x"].append(xs)
                captured_io[idx]["y"].append(ys)
        return fn

    for idx, module in wanted.items():
        hooks.append(module.register_forward_hook(_hook(idx)))
    model = model_with_rows.model
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            seen = 0
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(x.to(device))
                seen += 1
                if seen >= int(n_batches):
                    break
    except Exception:
        return None
    finally:
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()

    io = {}
    for idx, parts in captured_io.items():
        if not parts["x"] or not parts["y"]:
            continue
        io[idx] = {"x": torch.cat(parts["x"], dim=0), "y": torch.cat(parts["y"], dim=0)}
    if not io:
        return None
    return {
        "io": io,
        "weights": {idx: module.weight.detach().cpu().clone() for idx, module in wanted.items()},
        "bias": {idx: (None if module.bias is None else module.bias.detach().cpu().clone())
                 for idx, module in wanted.items()},
        "producer_indices": producer_ids,
    }


def apply_lsq(model_with_rows, captured) -> dict:
    """Write least-squares consumer kernels into the already-pruned modules (A-LSQ).

    Every consumer whose input the cut resized is refit: plain and concat convolutions
    (the kept input indices span every surviving channel, so a concat consumer's other
    slices are refit together with the group's slice) and Linear consumers behind a
    flatten. Grouped convolutions (depthwise readers) are the one skipped kind — their
    kernel has no column per input channel to fit.
    """
    summary = {"lsq": 0, "lsq_linear": 0, "skipped": 0}
    edit = getattr(model_with_rows, "last_group_edit", None) or {}
    if not captured or not edit:
        return summary
    layers = model_with_rows.all_layers
    for idx, old_in in (edit.get("consumer_in_idx") or {}).items():
        module = layers[int(idx)]
        sample = (captured.get("io") or {}).get(int(idx))
        if sample is None:
            summary["skipped"] += 1
            continue
        if isinstance(module, nn.Conv2d):
            fitted = lsq_conv2d_weight(sample["x"], sample["y"], old_in, module)
            kind = "lsq"
        elif isinstance(module, nn.Linear):
            fitted = lsq_linear_weight(sample["x"], sample["y"], old_in, module)
            kind = "lsq_linear"
        else:
            fitted = None
            kind = "lsq"
        if fitted is None:
            summary["skipped"] += 1
            continue
        weight, bias = fitted
        if tuple(weight.shape) != tuple(module.weight.shape):
            summary["skipped"] += 1
            continue
        with torch.no_grad():
            module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
            if bias is not None and module.bias is not None and bias.numel() == module.bias.numel():
                module.bias.copy_(bias.to(device=module.bias.device, dtype=module.bias.dtype))
        summary[kind] += 1
    return summary


def apply_pca(model_with_rows, captured) -> dict:
    """Replace producer filters by the activation basis and premultiply consumers (C-PCA).

    One basis ``U`` (old_width × k) for the whole group, from the first producer's pre-cut
    output, so every producer of a residual stream is rotated the same way and the identity
    adds stay consistent (``x' + F'(x') = Uᵀ(x + F(x))`` up to BN/ReLU, which are not
    rotation-invariant — that is the approximation the fine-tune has to close). Consumers get
    ``W U`` on the group's slice of their input; a concat consumer keeps its other slices.
    The group's BatchNorms are reset to identity affine (their per-channel γ/β described the
    old channels) and their running stats cleared — pair this recipe with BN recalibration.
    Groups with a depthwise owner are skipped whole: a depthwise kernel is one filter per
    channel and cannot be mixed across channels.
    """
    summary = {"pca_producers": 0, "pca_consumers": 0, "pca_norms": 0, "skipped": 0, "width": 0}
    edit = getattr(model_with_rows, "last_group_edit", None) or {}
    if not captured or not edit:
        return summary
    if edit.get("depthwise"):
        summary["skipped"] += 1
        summary["reason"] = "depthwise owner"
        return summary
    k = int(edit.get("new_width") or 0)
    layers = model_with_rows.all_layers
    basis = None
    for idx in captured.get("producer_indices") or []:
        sample = (captured.get("io") or {}).get(int(idx))
        old_w = (captured.get("weights") or {}).get(int(idx))
        if sample is None or old_w is None or old_w.dim() != 4:
            continue
        basis = channel_basis(sample["y"], k)
        if basis is not None and basis.shape[0] == old_w.shape[0]:
            break
        basis = None
    if basis is None:
        summary["skipped"] += 1
        return summary
    summary["width"] = int(basis.shape[1])
    producer_set = set(int(i) for i in (captured.get("producer_indices") or []))
    with torch.no_grad():
        for idx in captured.get("producer_indices") or []:
            module = layers[int(idx)]
            old_w = (captured.get("weights") or {}).get(int(idx))
            if not isinstance(module, nn.Conv2d) or old_w is None:
                summary["skipped"] += 1
                continue
            if old_w.shape[0] != basis.shape[0] or module.out_channels != basis.shape[1]:
                summary["skipped"] += 1
                continue
            new_w = mix_out_channels(old_w, basis)
            if tuple(new_w.shape) != tuple(module.weight.shape):
                summary["skipped"] += 1
                continue
            module.weight.copy_(new_w.to(device=module.weight.device, dtype=module.weight.dtype))
            old_b = (captured.get("bias") or {}).get(int(idx))
            if module.bias is not None and old_b is not None and old_b.numel() == basis.shape[0]:
                new_b = basis.T.float() @ old_b.float()
                module.bias.copy_(new_b.to(device=module.bias.device, dtype=module.bias.dtype))
            summary["pca_producers"] += 1

        group_positions = edit.get("consumer_group_positions") or {}
        new_positions = edit.get("consumers") or {}
        for idx, old_w in (captured.get("weights") or {}).items():
            if int(idx) in producer_set:
                continue
            module = layers[int(idx)]
            if not isinstance(module, nn.Conv2d) or old_w is None or old_w.dim() != 4 or module.groups != 1:
                summary["skipped"] += 1
                continue
            old_pos = [int(p) for p in group_positions.get(int(idx), group_positions.get(str(idx), []))]
            new_pos = [int(p) for p in new_positions.get(int(idx), new_positions.get(str(idx), []))]
            if len(old_pos) != basis.shape[0] or len(new_pos) != basis.shape[1]:
                # Consumer whose input is exactly the group (legacy path) or one we cannot place.
                if old_w.shape[1] == basis.shape[0] and module.in_channels == basis.shape[1]:
                    new_w = mix_in_channels(old_w, basis)
                    module.weight.copy_(new_w.to(device=module.weight.device, dtype=module.weight.dtype))
                    summary["pca_consumers"] += 1
                else:
                    summary["skipped"] += 1
                continue
            slice_w = old_w[:, old_pos]                          # (out, old_width, kH, kW)
            mixed = mix_in_channels(slice_w, basis)               # (out, k, kH, kW)
            module.weight[:, new_pos] = mixed.to(device=module.weight.device, dtype=module.weight.dtype)
            summary["pca_consumers"] += 1

        # Group norms: the sliced γ/β/stats belong to the old channels; the rotated channels
        # start from identity affine and empty stats (BN recalibration fills them).
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
            elif positions:
                pos = torch.tensor(sorted(int(p) for p in positions), dtype=torch.long,
                                   device=norm.weight.device if getattr(norm, "affine", False)
                                   else norm.running_mean.device)
                if getattr(norm, "track_running_stats", False) and norm.running_mean is not None:
                    norm.running_mean[pos] = 0.0
                    norm.running_var[pos] = 1.0
                if getattr(norm, "affine", False):
                    norm.weight[pos] = 1.0
                    norm.bias[pos] = 0.0
            summary["pca_norms"] += 1
    return summary


def recalibrate_batchnorm(model: nn.Module, loader, device, n_batches: int = 8) -> int:
    """Cumulative re-estimate of every BatchNorm's running stats. Returns how many modules moved."""
    norms = [m for m in model.modules()
             if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not norms or loader is None:
        return 0
    saved = [(m, m.momentum, m.training) for m in norms]
    was_training = model.training
    model.train()
    for module in norms:
        module.reset_running_stats()
        module.momentum = None
    seen = 0
    try:
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(x.to(device))
                seen += 1
                if seen >= int(n_batches):
                    break
    finally:
        for module, momentum, training in saved:
            module.momentum = momentum
            module.train(training)
        if not was_training:
            model.eval()
    return len(norms) if seen else 0
