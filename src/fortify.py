"""
SPECTRA fortifications. On by default (SPECTRA_FORTIFY=1); set SPECTRA_FORTIFY=0 to disable.

When enabled:
  * Stem / narrow-width action masking (identity-only on fragile layers)
  * Post-warmup entropy coefficient anneal (AMC-style exploration decay)
  * Extra per-layer representation channels (depth, stem, coupling, width)

Always available (not gated): mid-run ``train_resume.pt`` save/load helpers used by the
agent so USR1 / preempt can warm-continue without a full cold start.
"""

from __future__ import annotations

import os
from typing import Dict, Sequence

import torch
from torch.distributions import Categorical

import src.pruning as pruning

FORTIFY_TOKEN_DIM = 4  # relative_depth, is_stem, is_coupled, width_norm


def fortify_enabled() -> bool:
    raw = os.environ.get("SPECTRA_FORTIFY", "1").strip().lower()
    return raw in ("1", "true", "yes")


def budget_in_state() -> bool:
    """Broadcast remaining-param ratio as an extra token channel (default off)."""
    raw = os.environ.get("SPECTRA_BUDGET_IN_STATE", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def stem_rows() -> int:
    """How many leading prunable rows are treated as stem (identity-only under fortify)."""
    return max(0, int(os.environ.get("SPECTRA_STEM_ROWS", "1")))


def min_width_for_prune() -> int:
    """Layers at or below this alive width may only take the identity action under fortify."""
    return max(1, int(os.environ.get("SPECTRA_MIN_WIDTH_FOR_PRUNE", "2")))


def entropy_anneal_horizon() -> int:
    return max(1, int(os.environ.get("SPECTRA_ENTROPY_ANNEAL_HORIZON", "100")))


def entropy_min_coef(base: float) -> float:
    raw = os.environ.get("SPECTRA_ENTROPY_MIN", "").strip()
    if raw:
        return float(raw)
    return 0.2 * base


def entropy_coef(episode_idx: int, warmup_len: int, base: float) -> float:
    """Constant base during warmup; linear decay toward entropy_min after warmup when fortify on."""
    if not fortify_enabled() or episode_idx < warmup_len:
        return base
    t = min((episode_idx - warmup_len) / float(entropy_anneal_horizon()), 1.0)
    lo = entropy_min_coef(base)
    return base + (lo - base) * t


def fortify_token_dim() -> int:
    n = FORTIFY_TOKEN_DIM if fortify_enabled() else 0
    if budget_in_state():
        n += 1
    return n


def build_fortify_features(
    num_layers: int,
    coupling_ids: torch.Tensor,
    topology: Sequence,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Representation fortification channels (already ~[0,1], not z-scored with raw moments):

      0 relative_depth in [0,1]
      1 is_stem (first SPECTRA_STEM_ROWS layers)
      2 is_coupled (shares coupling id with another layer — skip / group signal)
      3 width_norm = out_channels_or_features / max_width (from topology cols)
    """
    if num_layers == 0:
        return torch.zeros(0, FORTIFY_TOKEN_DIM, device=device, dtype=dtype)

    depths = torch.arange(num_layers, device=device, dtype=dtype) / max(num_layers - 1, 1)
    stem = torch.zeros(num_layers, device=device, dtype=dtype)
    stem[: min(stem_rows(), num_layers)] = 1.0

    coupled = torch.zeros(num_layers, device=device, dtype=dtype)
    if coupling_ids is not None and coupling_ids.numel() == num_layers:
        for cid in torch.unique(coupling_ids):
            idx = (coupling_ids == cid).nonzero(as_tuple=False).flatten()
            if idx.numel() > 1:
                coupled[idx] = 1.0

    widths = []
    for i in range(num_layers):
        row = topology[i] if i < len(topology) and topology[i] else [0.0] * 7
        kind = int(row[0]) if row else 0
        if kind == 2:  # Conv
            w = float(row[2]) if len(row) > 2 else 0.0
        elif kind == 1:  # Linear
            w = float(row[6]) if len(row) > 6 else 0.0
        else:
            w = float(row[2]) if len(row) > 2 else 0.0
        widths.append(max(w, 0.0))
    width_t = torch.tensor(widths, device=device, dtype=dtype)
    width_norm = width_t / width_t.max().clamp_min(1.0)

    return torch.stack([depths, stem, coupled, width_norm], dim=1)


def legal_action_mask(
    compression_rates: Dict[int, float],
    *,
    row_index: int,
    alive_count: int,
    device,
) -> torch.Tensor:
    """
    Bool mask over discrete actions.

    Always on (library hygiene):
      * rates that cannot change width (target_width == alive) → illegal if rate < 1
      * layers already at 1 alive channel → identity only

    When fortify is enabled (default) additionally:
      * stem rows → only rate == 1.0
      * narrow layers (alive <= min_width) → only rate == 1.0
    """
    n = len(compression_rates)
    mask = torch.ones(n, dtype=torch.bool, device=device)

    force_identity = alive_count <= 1
    if fortify_enabled():
        force_identity = force_identity or (row_index < stem_rows()) or (
            alive_count <= min_width_for_prune())

    for idx, rate in compression_rates.items():
        if force_identity:
            mask[idx] = abs(float(rate) - 1.0) < 1e-9
            continue
        if float(rate) >= 1.0:
            continue
        # No-op compressions are pure credit-assignment noise (always illegal).
        if pruning.target_width(alive_count, float(rate)) >= alive_count:
            mask[idx] = False

    if not mask.any():
        for idx, rate in compression_rates.items():
            if abs(float(rate) - 1.0) < 1e-9:
                mask[idx] = True
                break
        if not mask.any():
            mask[0] = True
    return mask


def eval_min_param_ratio() -> float:
    """Stop applying non-identity prune actions in eval once params fall below this fraction."""
    return float(os.environ.get("SPECTRA_EVAL_MIN_PARAM_RATIO", "0.70"))


def critic_huber_delta() -> float:
    """Smooth-L1 beta for critic; override with SPECTRA_CRITIC_HUBER_DELTA (0 disables)."""
    return float(os.environ.get("SPECTRA_CRITIC_HUBER_DELTA", "100.0"))


def apply_action_mask(dist: Categorical, legal: torch.Tensor) -> Categorical:
    """Zero illegal probs and renorm; used for both sample and log_prob."""
    if legal is None or bool(legal.all()):
        return dist
    probs = dist.probs.clone()
    legal_b = legal
    while legal_b.dim() < probs.dim():
        legal_b = legal_b.unsqueeze(0)
    probs = probs * legal_b.to(dtype=probs.dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return Categorical(probs=probs)


def sample_masked_action(
    dist: Categorical,
    legal: torch.Tensor,
    *,
    uniform: bool,
    device,
) -> torch.Tensor:
    """Sample from masked policy, or uniform over legal actions during warmup."""
    masked = apply_action_mask(dist, legal)
    if not uniform:
        return masked.sample()
    legal_idx = legal.nonzero(as_tuple=False).flatten()
    choice = legal_idx[torch.randint(0, legal_idx.numel(), (1,), device=device)]
    return choice
