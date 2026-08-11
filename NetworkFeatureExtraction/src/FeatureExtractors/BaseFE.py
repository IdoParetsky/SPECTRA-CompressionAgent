import torch
from typing import List, Dict


class BaseFE:
    """
    Shared helpers for topology / activation / weight feature extractors.

    Moment set (see docs/BERT_INPUT_CRITIQUE.md §8 and the thesis briefing):
      * classical moments + L1/L2 norms;
      * order statistics min / p25 / median / p75 / max (shape of the distribution);
      * ``abs_p10`` — 10th percentile of absolute magnitudes, replacing the old
        ``scale_exponent`` which was dominated by a single near-zero outlier.
    """

    def extract_feature_map(self, model_with_rows) -> List[List[float]]:
        """
        Placeholder method to extract features for a specific layer.
        Should be overridden by child classes.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
        """
        raise NotImplementedError("Child class must implement extract_feature_map.")

    # Canonical statistic order, shared by every helper below.
    # Length must stay in sync with TOKEN_FEATURE_DIM in src/BERTInputModeler.py.
    MOMENT_NAMES = (
        "mean", "std", "skewness", "kurtosis",
        "min", "p25", "median", "p75", "max",
        "l1_norm", "l2_norm", "abs_p10",
    )

    # Shape of a 1-D importance / magnitude distribution (e.g. per-filter L1 norms).
    # Used by WeightStatisticsFE; deliberately separate from MOMENT_NAMES so the layer-token
    # layout stays readable.
    SHAPE_NAMES = ("mean", "std", "min", "p25", "median", "p75", "max")

    @staticmethod
    def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
        """
        Scalar quantile of a 1-D tensor.

        ``torch.quantile`` on multi-million-element tensors is one of the more expensive
        feature ops; callers should reduce (e.g. spatial mean of activations) before
        invoking this. For SPECTRA's typical layer sizes it is cheap enough to run every step.
        """
        if values.numel() == 0:
            return torch.tensor(0.0, device=values.device)
        # Flatten and use a float32 view; quantile does not support every dtype on every device
        flat = values.detach().float().reshape(-1)
        return torch.quantile(flat, q)

    @classmethod
    def distribution_shape(cls, values: torch.Tensor) -> torch.Tensor:
        """
        (mean, std, min, p25, median, p75, max) of a 1-D sample.

        This is the recommended summary of *per-filter importance*: the agent's action is a
        compression *rate*, so what matters is the shape of the importance distribution, not
        a token per filter (see critique §4).
        """
        flat = values.detach().float().reshape(-1)
        if flat.numel() == 0:
            return torch.zeros(len(cls.SHAPE_NAMES), device=values.device)

        mean = flat.mean()
        std = flat.std(unbiased=False) if flat.numel() > 1 else torch.zeros((), device=flat.device)
        # One quantile call with several q's is cheaper than five separate ones
        qs = torch.quantile(flat, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=flat.device))
        return torch.stack([mean, std, qs[0], qs[1], qs[2], qs[3], qs[4]])

    def compute_moments_batch(self, rows: torch.Tensor) -> torch.Tensor:
        """
        Statistics of every row of a 2-D tensor, computed in one shot.

        Returns a (num_rows, len(MOMENT_NAMES)) device tensor. Summarising filters one at a
        time through host-side loops would cost one synchronisation per statistic per filter.

        Note: the default WeightStatisticsFE no longer emits per-filter moment vectors into
        the agent state (critique §4); this helper remains for callers that still need a
        per-row view (e.g. diagnostics).
        """
        flat = rows.float()
        mean = flat.mean(dim=1)
        std = flat.std(dim=1, unbiased=False) + 1e-8

        centralized = flat - mean.unsqueeze(1)
        skewness = (centralized ** 3).mean(dim=1) / (std ** 3)
        kurtosis = (centralized ** 4).mean(dim=1) / (std ** 4)

        abs_flat = flat.abs()
        l1 = abs_flat.sum(dim=1)
        l2 = flat.pow(2).sum(dim=1).sqrt()

        # Order statistics along each row. torch.quantile over dim=1 is supported for float tensors.
        qs = torch.quantile(
            flat, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=flat.device), dim=1)
        abs_p10 = torch.quantile(abs_flat, 0.10, dim=1)

        return torch.stack([
            mean, std, skewness, kurtosis,
            qs[0], qs[1], qs[2], qs[3], qs[4],
            l1, l2, abs_p10,
        ], dim=1)

    def compute_moments_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Layer-level statistics as a (len(MOMENT_NAMES),) device tensor so callers that only
        need to accumulate them avoid one host synchronisation per statistic.
        """
        if tensor.dim() == 4:
            # Activations (B, C, H, W): average over spatial dims before the global summary.
            # Quantiling the full HxW map every step would be the expensive path to avoid.
            tensor = tensor.mean(dim=(2, 3))  # → (B, C)
        elif tensor.dim() > 2:
            tensor = tensor.view(tensor.size(0), -1)

        flat = tensor.float().reshape(-1)
        if flat.numel() == 0:
            return torch.zeros(len(self.MOMENT_NAMES), device=tensor.device)

        mean = flat.mean()
        std = flat.std(unbiased=False) + 1e-8
        centralized = flat - mean
        skewness = (centralized ** 3).mean() / (std ** 3)
        kurtosis = (centralized ** 4).mean() / (std ** 4)

        abs_flat = flat.abs()
        l1 = abs_flat.sum()
        l2 = flat.pow(2).sum().sqrt()
        qs = torch.quantile(flat, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=flat.device))
        abs_p10 = torch.quantile(abs_flat, 0.10)

        return torch.stack([
            mean, std, skewness, kurtosis,
            qs[0], qs[1], qs[2], qs[3], qs[4],
            l1, l2, abs_p10,
        ])

    def compute_moments(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Efficient, GPU-optimized computation of statistical moments and norms.
        Applies adaptive reshaping and averaging for activation maps.

        Args:
            tensor (torch.Tensor): Input tensor, either weights or activations.

        Returns:
            Dict[str, float]: Dictionary of computed statistics.
        """
        values = self.compute_moments_tensor(tensor).tolist()  # one synchronisation
        return dict(zip(self.MOMENT_NAMES, values))

    def empty_vector(self) -> Dict[str, float]:
        return {name: 0.0 for name in self.MOMENT_NAMES}
