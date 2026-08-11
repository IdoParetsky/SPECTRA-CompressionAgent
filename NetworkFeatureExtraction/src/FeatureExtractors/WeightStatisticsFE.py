import torch
from torch.nn import Linear, Conv2d
from .BaseFE import BaseFE
from typing import List


class WeightStatisticsFE(BaseFE):
    """
    Fixed-width weight features per layer.

    Earlier drafts flattened nine moments *per filter* into a variable-length vector, then
    summarised with mean/std. Per-filter tokens do not scale (critique §4): a ResNet-50 layer
    alone can exceed any reasonable sequence budget, and the agent's action is a compression
    *rate*, so the informative signal is the *shape* of the filter-importance distribution.

    Each layer now contributes:
      1. layer-level moments of the weight tensor (MOMENT_NAMES);
      2. shape statistics of the per-filter L1 norms (SHAPE_NAMES).
    """

    def __init__(self, device):
        self.device = device
        self.vector_width = len(self.MOMENT_NAMES) + len(self.SHAPE_NAMES)

    def extract_feature_map(self, model_with_rows) -> List[List[float]]:
        """
        Extracts a fixed-width statistical vector from each layer's weights.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.

        Returns:
            List[List[float]]: Per-layer vectors of length ``vector_width``.
        """
        features = []

        for layer in model_with_rows.all_layers:
            if hasattr(layer, "weight") and layer.weight is not None:
                weights = layer.weight.data.to(self.device)

                # Per-filter (or per-neuron) rows for the L1-importance distribution
                if isinstance(layer, (Conv2d, Linear)) or weights.dim() > 1:
                    per_filter = weights.reshape(weights.size(0), -1)
                else:
                    per_filter = weights.reshape(1, -1)

                # L1 importance of each filter — the quantity structured pruning ranks by
                l1_per_filter = per_filter.abs().sum(dim=1)
                shape = self.distribution_shape(l1_per_filter)

                # Layer-level moments of the full weight tensor (not one token per filter)
                moments = self.compute_moments_tensor(weights)

                features.append(torch.cat([moments, shape]).tolist())
            else:
                features.append([0.0] * self.vector_width)

        return features
