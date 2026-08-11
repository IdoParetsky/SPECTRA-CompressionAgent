"""
Per-feature standardisation of layer tokens across the training database.

Motivation (docs/BERT_INPUT_CRITIQUE.md §6): topology integers, channel counts, L1 norms and
kurtosis live on incompatible scales. A signed log1p squash is a local fix; what the
"generic agent" premise actually needs is features that are comparable *across
architectures*, which means fitting (mean, std) once over every layer token seen in the
database and applying the same transform at train and eval time.

Cost note: fitting walks every network once and runs the activation probe for each. That is
a one-time cost proportional to (#networks × probe batches × forward). Prefer caching the
fitted stats to ``SPECTRA_STANDARDIZER_PATH`` so subsequent runs skip the pass. Skip entirely
with ``SPECTRA_SKIP_STANDARDIZER=1`` for short correctness smoke tests.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch

import src.utils as utils

_EPS = 1e-6


class FeatureStandardizer:
    """Welford running mean/variance over layer-token rows, then z-score transform."""

    _instance: Optional["FeatureStandardizer"] = None

    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.mean = torch.zeros(dim)
        self.m2 = torch.zeros(dim)  # sum of squared deviations (Welford)
        self._frozen = False

    @classmethod
    def instance(cls, dim: int) -> "FeatureStandardizer":
        if cls._instance is None or cls._instance.dim != dim:
            cls._instance = cls(dim)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    @property
    def is_fitted(self) -> bool:
        return self._frozen and self.count > 1

    def update(self, tokens: torch.Tensor):
        """
        Accumulate rows from a (num_layers, dim) token matrix.

        Only the first ``self.dim`` columns are standardised; callers that append action-cost
        slots should pass the *base* features here (fractions in [0, 1] need no z-score).
        """
        if self._frozen:
            return
        rows = tokens.detach().float().cpu()
        if rows.dim() != 2 or rows.size(1) < self.dim:
            raise ValueError(f"expected (L, >={self.dim}) tokens, got {tuple(rows.shape)}")
        for row in rows[:, : self.dim]:
            self.count += 1
            delta = row - self.mean
            self.mean = self.mean + delta / self.count
            self.m2 = self.m2 + delta * (row - self.mean)

    def finalize(self):
        self._frozen = True
        utils.print_flush(
            f"FeatureStandardizer fitted on {self.count} layer tokens "
            f"(dim={self.dim})")

    @property
    def std(self) -> torch.Tensor:
        if self.count < 2:
            return torch.ones(self.dim)
        return torch.sqrt(self.m2 / (self.count - 1)).clamp(min=_EPS)

    def transform(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Z-score the leading ``dim`` columns in-place-safe fashion; trailing columns (e.g.
        action-cost slots) are left unchanged.
        """
        if not self.is_fitted:
            return tokens
        out = tokens.clone()
        mean = self.mean.to(device=tokens.device, dtype=tokens.dtype)
        std = self.std.to(device=tokens.device, dtype=tokens.dtype)
        out[:, : self.dim] = (out[:, : self.dim] - mean) / std
        return out

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "dim": torch.tensor(self.dim),
            "count": torch.tensor(self.count),
            "mean": self.mean.clone(),
            "m2": self.m2.clone(),
            "frozen": torch.tensor(int(self._frozen)),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        self.dim = int(state["dim"].item())
        self.count = int(state["count"].item())
        self.mean = state["mean"].float().cpu()
        self.m2 = state["m2"].float().cpu()
        self._frozen = bool(state["frozen"].item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        utils.print_flush(f"FeatureStandardizer saved to {path}")

    def load(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state)
        utils.print_flush(f"FeatureStandardizer loaded from {path} (n={self.count})")


def ensure_fitted(database_dict, device, token_base_dim: int) -> FeatureStandardizer:
    """
    Fit (or load) the database-wide standardiser before RL training begins.

    Args:
        database_dict: ``{path: (model, (train, val, test))}`` as produced by preload.
        device:        Torch device for the activation probes.
        token_base_dim: Width of the base layer token (excluding action-cost slots).
    """
    std = FeatureStandardizer.instance(token_base_dim)

    if os.environ.get("SPECTRA_SKIP_STANDARDIZER", "").strip() in {"1", "true", "True"}:
        utils.print_flush("FeatureStandardizer skipped (SPECTRA_SKIP_STANDARDIZER=1)")
        # Identity transform: freeze with count=0 so transform() is a no-op
        std.count = 0
        std._frozen = True
        return std

    cache_path = os.environ.get("SPECTRA_STANDARDIZER_PATH", "").strip()
    if cache_path and os.path.isfile(cache_path):
        std.load(cache_path)
        return std

    if std.is_fitted:
        return std

    if not database_dict:
        utils.print_flush("FeatureStandardizer: empty database; leaving unfitted (log1p fallback)")
        return std

    # Local import avoids a circular dependency at module load time
    from NetworkFeatureExtraction.src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
    from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows

    utils.print_flush(
        f"Fitting FeatureStandardizer over {len(database_dict)} database networks "
        f"(one activation probe each; cache with SPECTRA_STANDARDIZER_PATH to skip next time)")

    for net_path, (model, loaders) in database_dict.items():
        train_loader = loaders[0]
        try:
            extractor = FeatureExtractor(train_loader, device)
            model_with_rows = ModelWithRows(model)
            feature_maps = extractor.extract_features(model_with_rows)
            # Raw tokens — must not apply the (still-fitting) transform while accumulating
            tokens = extractor.state_builder.build_base_tokens(feature_maps)
            std.update(tokens)
        except Exception as error:
            utils.print_flush(f"FeatureStandardizer: skipped {net_path} ({error})")

    std.finalize()
    if cache_path:
        std.save(cache_path)
    return std
