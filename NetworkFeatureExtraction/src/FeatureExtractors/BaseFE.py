from ..ModelWithRows import ModelWithRows
import torch
from typing import List, Dict


class BaseFE:
    model_with_rows: ModelWithRows

    def __init__(self, model_with_rows: ModelWithRows):
        self.model_with_rows: ModelWithRows = model_with_rows

    def extract_feature_map(self) -> List[List[float]]:
        """
        Placeholder method to extract features for a specific layer.
        Should be overridden by child classes.
        """
        raise NotImplementedError("Child class must implement extract_feature_map.")

    def compute_moments(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Efficient, GPU-optimized computation of statistical moments and norms.
        Applies adaptive reshaping and averaging for activation maps.

        Args:
            tensor (torch.Tensor): Input tensor, either weights or activations.

        Returns:
            Dict[str, float]: Dictionary of computed statistics.
        """
        if tensor.dim() == 4:
            # For activations (B, C, H, W), average over spatial dims
            tensor = tensor.mean(dim=(2, 3))  # → (B, C)
        elif tensor.dim() > 2:
            tensor = tensor.view(tensor.size(0), -1)  # Flatten for general case

        flat = tensor.float()
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False) + 1e-8

        centralized = flat - mean
        skewness = (centralized ** 3).mean(dim=0) / (std ** 3)
        kurtosis = (centralized ** 4).mean(dim=0) / (std ** 4)

        abs_flat = flat.abs()
        min_val = flat.min(dim=0).values
        max_val = flat.max(dim=0).values
        l1 = abs_flat.sum(dim=0)
        l2 = flat.pow(2).sum(dim=0).sqrt()

        nonzero = abs_flat[abs_flat > 0]
        if nonzero.numel() > 0:
            scale_exponent = torch.floor(torch.log10(nonzero.min()))
        else:
            scale_exponent = torch.tensor(0.0, device=tensor.device)

        return {
            "mean": mean.mean().item(),
            "std": std.mean().item(),
            "skewness": skewness.mean().item(),
            "kurtosis": kurtosis.mean().item(),
            "min": min_val.mean().item(),
            "max": max_val.mean().item(),
            "l1_norm": l1.mean().item(),
            "l2_norm": l2.mean().item(),
            "scale_exponent": scale_exponent.item()
        }

    def empty_vector(self) -> Dict[str, float]:
        return {
            "mean": 0.0,
            "std": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "min": 0.0,
            "max": 0.0,
            "l1_norm": 0.0,
            "l2_norm": 0.0,
            "scale_exponent": 0.0
        }
