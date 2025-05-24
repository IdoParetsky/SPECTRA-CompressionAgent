import torch
from torch.nn import Linear, Conv2d
from .BaseFE import BaseFE
from typing import List
import src.utils as utils


class WeightStatisticsFE(BaseFE):
    def __init__(self, device):
        """
        Extracts statistical features from model weights.
        """
        self.device = device

    def extract_feature_map(self, model_with_rows) -> List[List[float]]:
        """
        Extracts statistical features from weights of Linear and Conv2d layers.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
        
        Returns:
            List[List[float]]: Per-layer statistical vectors.
        """
        # utils.print_flush("Starting Weights FE")
        features = []

        for layer in model_with_rows.all_layers:
            if hasattr(layer, 'weight') and layer.weight is not None:
                weights = layer.weight.data.to(self.device)

                # Flatten each neuron/filter into a 1D row for per-neuron stats
                if isinstance(layer, Conv2d):
                    weights = weights.view(weights.size(0), -1)
                elif isinstance(layer, Linear):
                    weights = weights  # Already shaped as (out, in)

                # Compute stats per neuron/filter
                layer_stats = []
                for w in weights:
                    stats = self.compute_moments(w)
                    layer_stats.append(list(stats.values()))

                # Flatten into single vector per layer
                features.append([val for stat in layer_stats for val in stat])
            else:
                features.append([0.0] * len(self.compute_moments(torch.zeros(1))))
        # utils.print_flush("Finished Weights FE")
        return features
