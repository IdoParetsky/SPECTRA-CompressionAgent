import torch.nn as nn
from typing import List
from .BaseFE import BaseFE
import src.utils as utils


class TopologyFE(BaseFE):
    def __init__(self, model_with_rows):
        """
        Extracts the architecture of a CNN as a sequence for BERT tokenization.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
        """
        super().__init__(model_with_rows)

        # Map layer types to handler functions
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
            nn.Dropout: self.handle_dropout,
            nn.Flatten: self.handle_flatten,
            nn.MaxPool2d: self.handle_pooling,
            nn.AvgPool2d: self.handle_pooling
        }

    def extract_feature_map(self) -> List[List[float]]:
        """
        Extracts a per-layer representation of the CNN topology for BERT tokenization.

        Returns:
            List[List[float]]: A sequence of feature vectors, one per layer.
        """
        utils.print_flush("Starting Topology FE")
        topology_sequence = []

        for layer in self.model_with_rows.all_layers:
            handler = self.layer_type_to_function.get(type(layer), None)
            if handler:
                topology_sequence.append(handler(layer))
            else:
                topology_sequence.append([0.0] * 7)  # Default for unrecognized layers
        utils.print_flush("Finished Topology FE")
        return topology_sequence

    @staticmethod
    def handle_linear(layer) -> List[float]:
        return [1, 0, 0, 0, 0, layer.in_features, layer.out_features]

    @staticmethod
    def handle_conv2d(layer) -> List[float]:
        return [2, layer.in_channels, layer.out_channels, layer.kernel_size[0],
                layer.stride[0], layer.padding[0], 0]

    @staticmethod
    def handle_batchnorm(layer) -> List[float]:
        return [3, layer.num_features, 0, 0, 0, 0, 0]

    @staticmethod
    def handle_activation(layer) -> List[float]:
        return [4, 0, 0, 0, 0, 0, 0]

    @staticmethod
    def handle_dropout(layer) -> List[float]:
        return [5, layer.p if hasattr(layer, 'p') else 0, 0, 0, 0, 0, 0]

    @staticmethod
    def handle_flatten(layer) -> List[float]:
        return [6, 0, 0, 0, 0, 0, 0]

    @staticmethod
    def handle_pooling(layer) -> List[float]:
        return [7, layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0],
                layer.stride if isinstance(layer.stride, int) else layer.stride[0],
                layer.padding if isinstance(layer.padding, int) else layer.padding[0], 0, 0, 0]
