import numpy as np
import torch
from scipy.stats import skew, kurtosis
from torch.nn import Linear, Conv2d
from .BaseFE import BaseFE
from ..utils import get_scaler_exponent


class WeightStatisticsFE(BaseFE):
    def __init__(self, model_with_rows):
        """
        Extracts statistical features from model weights.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
        """
        super(WeightStatisticsFE, self).__init__(model_with_rows)

    def extract_feature_map(self):
        """
        Extracts weight statistics (moments, min/max, norms) for all layers.

        Returns:
            weight_sequence (List[List[float]]): A sequential representation of weight statistics.
        """
        weight_sequence = []

        # Process weights for each layer
        for row in self.model_with_rows.all_rows:
            main_layer = row[0]  # The main computational layer (Conv2D/Linear)

            if isinstance(main_layer, Linear):
                weight_sequence.append(self.process_linear_layer(main_layer))
            elif isinstance(main_layer, Conv2d):
                weight_sequence.append(self.process_conv2d_layer(main_layer))

        return weight_sequence

    def process_linear_layer(self, layer):
        """
        Extracts statistical features from Linear layers' weights.

        Args:
            layer (torch.nn.Linear): Linear layer.

        Returns:
            List[float]: Statistical features.
        """
        weights = layer.weight.detach().cpu().numpy()
        return self.compute_weight_statistics(weights)

    def process_conv2d_layer(self, layer):
        """
        Extracts statistical features from Conv2D layers' weights.

        Args:
            layer (torch.nn.Conv2d): Conv2D layer.

        Returns:
            List[float]: Statistical features.
        """
        weights = layer.weight.detach().cpu().numpy()
        flattened_weights = weights.reshape(weights.shape[0], -1)  # Flatten per filter
        return self.compute_weight_statistics(flattened_weights)

    def compute_weight_statistics(self, weights):
        """
        Computes weight distribution statistics.

        Args:
            weights (np.array): Weights of a layer.

        Returns:
            List[float]: Statistical features for BERT tokenization.
        """
        # Scale weights for numerical stability
        scaler_exp = get_scaler_exponent(weights)
        weights_scaled = weights * (10 ** -scaler_exp)

        # Compute statistical moments
        mean = np.mean(weights, axis=1)
        std = np.std(weights, axis=1)
        skewness = skew(weights_scaled, axis=1)
        kurt = kurtosis(weights_scaled, axis=1)

        # Compute Min/Max
        min_val = np.min(weights, axis=1)
        max_val = np.max(weights, axis=1)

        # Compute L1/L2 norms
        l1_norm = np.linalg.norm(weights, axis=1, ord=1)
        l2_norm = np.linalg.norm(weights, axis=1, ord=2)

        return [*mean, *std, *skewness, *kurt, *min_val, *max_val, *l1_norm, *l2_norm]
