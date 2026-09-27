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


def cache_path_from_actor(actor_path: str) -> str:
    """
    Standardizer cache that belongs to an actor checkpoint.

    Preference: ``<ckpt dir>/standardizer.pt`` (written next to every ``latest_best_*``
    since 13 Sep, and carried by ``freeze_snapshot`` copies and manual snapshot dirs), then
    the historical ``<run>/standardizer.pt`` one level up. A snapshot copied without the
    cache used to resolve to a *wrong* directory and silently fall back to log1p (ledger §71).
    """
    if not actor_path:
        return ""
    ckpt_dir = os.path.dirname(os.path.abspath(actor_path))
    beside = os.path.join(ckpt_dir, "standardizer.pt")
    if os.path.isfile(beside):
        return beside
    run = os.path.dirname(ckpt_dir)
    if not run:
        return ""
    # Historical location (also the write target when nothing exists yet).
    return os.path.join(run, "standardizer.pt")


def resolve_standardizer_path(*, for_write: bool = False) -> str:
    """
    Env path, then the training run next to the loaded actor, then this job's run dir.

    Eval-only jobs used to skip ``ensure_fitted``, so TEST tokens fell back to log1p
    while training used z-scores. Prefer the actor's run cache so chained thin evals
    see the same features the policy was trained on.
    """
    env_path = os.environ.get("SPECTRA_STANDARDIZER_PATH", "").strip()
    if env_path and (for_write or os.path.isfile(env_path)):
        return env_path
    actor = os.environ.get("SPECTRA_ACTOR_CHECKPOINT_PATH", "").strip()
    inferred = cache_path_from_actor(actor)
    if inferred and (for_write or os.path.isfile(inferred)):
        return inferred
    if env_path:
        return env_path
    try:
        import src.logging_utils as logging_utils
        rd = logging_utils.run_dir()
        if rd:
            candidate = os.path.join(rd, "standardizer.pt")
            if for_write or os.path.isfile(candidate):
                return candidate
    except Exception:
        pass
    return inferred


def ensure_fitted(database_dict, device, token_base_dim: int, *,
                  load_only: bool = False) -> FeatureStandardizer:
    """
    Fit (or load) the database-wide standardiser before RL training begins.

    Args:
        database_dict: ``{path: (model, (train, val, test))}`` as produced by preload.
        device:        Torch device for the activation probes.
        token_base_dim: Width of the base layer token (excluding action-cost slots).
        load_only:     Eval-only: load a cache, never fit the eval catalog (thin
                       held-out nets are the wrong population).
    """
    std = FeatureStandardizer.instance(token_base_dim)

    if os.environ.get("SPECTRA_SKIP_STANDARDIZER", "").strip() in {"1", "true", "True"}:
        utils.print_flush("FeatureStandardizer skipped (SPECTRA_SKIP_STANDARDIZER=1)")
        # Identity transform: freeze with count=0 so transform() is a no-op
        std.count = 0
        std._frozen = True
        return std

    cache_path = resolve_standardizer_path(for_write=False)
    if cache_path and os.path.isfile(cache_path):
        std.load(cache_path)
        return std

    if std.is_fitted:
        return std

    if load_only:
        utils.print_flush(
            "FeatureStandardizer: eval-only with no cache at "
            f"{cache_path or '(unset)'}; log1p fallback "
            "(train z-score / eval log1p mismatch)")
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
    save_path = resolve_standardizer_path(for_write=True)
    if save_path:
        std.save(save_path)
        # Second copy next to the checkpoints, so any copy of ``agent_checkpoints/`` (snapshots,
        # continue-train seeds) carries the feature scale the policy was trained with.
        try:
            import src.logging_utils as logging_utils
            ckpt_dir = os.path.join(logging_utils.run_dir(), "agent_checkpoints")
            beside = os.path.join(ckpt_dir, "standardizer.pt")
            if os.path.abspath(beside) != os.path.abspath(save_path):
                std.save(beside)
        except Exception:
            pass
    return std
