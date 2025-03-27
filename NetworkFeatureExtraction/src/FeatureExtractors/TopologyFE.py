from ..FeatureExtractors.BaseFE import BaseFE
from torch import nn
from typing import List


class TopologyFE(BaseFE):
    def __init__(self, model_with_rows):
        """
        Extracts the architecture of a CNN as a sequence for BERT tokenization.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
        """
        super(TopologyFE, self).__init__(model_with_rows)

        # Dictionary to map layer types to feature extractors
        self.layer_type_to_function = {
            nn.Linear: self.handle_linear,
            nn.Conv2d: self.handle_conv2d,
            nn.BatchNorm1d: self.handle_batchnorm,
            nn.BatchNorm2d: self.handle_batchnorm,
            nn.ReLU: self.handle_activation,
            nn.ELU: self.handle_activation,
            nn.SiLU: self.handle_activation,
            nn.Softmax: self.handle_activation,
            nn.Tanh: self.handle_activation,
            nn.Sigmoid: self.handle_activation,
            nn.Dropout: self.handle_dropout
        }

    def extract_feature_map(self) -> List[List[float]]:
        """
        Extracts a sequence representation of the CNN topology for BERT tokenization.

        Returns:
            topology_sequence (List[List[float]]): A tokenized representation of the CNN topology.
        """
        topology_sequence = []

        for row in self.model_with_rows.all_rows:
            row_features = []
            for layer in row:
                handler = self.layer_type_to_function.get(type(layer), None)
                if handler:
                    row_features.extend(handler(layer))  # Append extracted features
            if row_features:  # Avoid empty rows
                topology_sequence.append(row_features)

        return topology_sequence

    @staticmethod
    def handle_linear(layer) -> List[float]:
        """Handles Linear (Fully Connected) layers."""
        return [1, 0, 0, 0, 0, layer.in_features, layer.out_features]  # 1 indicates Linear Layer

    @staticmethod
    def handle_conv2d(layer) -> List[float]:
        """Handles Conv2D layers and extracts key attributes."""
        return [2, layer.in_channels, layer.out_channels, layer.kernel_size[0],
                layer.stride[0], layer.padding[0], 0]  # 2 indicates Conv Layer

    @staticmethod
    def handle_batchnorm(layer) -> List[float]:
        """Handles BatchNorm layers."""
        return [3, 0, 0, 0, 0, 0, 0]  # 3 indicates BatchNorm

    @staticmethod
    def handle_activation(layer) -> List[float]:
        """Handles Activation layers."""
        return [4, 0, 0, 0, 0, 0, 0]  # 4 indicates Activation

    @staticmethod
    def handle_dropout(layer) -> List[float]:
        """Handles Dropout layers."""
        return [5, 0, 0, 0, 0, 0, 0]  # 5 indicates Dropout
