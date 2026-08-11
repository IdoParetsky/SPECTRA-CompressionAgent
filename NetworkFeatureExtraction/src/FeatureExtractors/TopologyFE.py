import torch.nn as nn
from typing import List
from .BaseFE import BaseFE
import src.utils as utils


class TopologyFE(BaseFE):
    def __init__(self):
        """
        Extracts the architecture of a CNN as a sequence for BERT tokenization.
        """

        # Map layer types to handler functions. Coverage matters for a generic agent: an
        # unrecognised module contributes an all-zero token, so a network built from
        # activations or pooling variants outside this map would be partly invisible.
        self.layer_type_to_function = {
            nn.Linear: self.handle_linear,
            nn.Conv1d: self.handle_conv,
            nn.Conv2d: self.handle_conv,
            nn.Conv3d: self.handle_conv,
            nn.ConvTranspose2d: self.handle_conv,
            nn.BatchNorm1d: self.handle_batchnorm,
            nn.BatchNorm2d: self.handle_batchnorm,
            nn.BatchNorm3d: self.handle_batchnorm,
            nn.GroupNorm: self.handle_groupnorm,
            nn.LayerNorm: self.handle_layernorm,
            nn.InstanceNorm2d: self.handle_batchnorm,
            nn.Dropout: self.handle_dropout,
            nn.Dropout2d: self.handle_dropout,
            nn.Flatten: self.handle_flatten,
            nn.Identity: self.handle_activation,
        }

        for activation in (nn.ReLU, nn.ReLU6, nn.ELU, nn.SiLU, nn.Softmax, nn.Tanh, nn.Sigmoid,
                           nn.LeakyReLU, nn.GELU, nn.Hardtanh, nn.Hardswish, nn.Hardsigmoid,
                           nn.Softplus, nn.Softsign, nn.PReLU, nn.LogSigmoid, nn.SELU, nn.CELU,
                           nn.Mish, nn.GLU, nn.LogSoftmax):
            self.layer_type_to_function[activation] = self.handle_activation

        for pooling in (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d,
                        nn.AvgPool3d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
                        nn.AdaptiveMaxPool2d):
            self.layer_type_to_function[pooling] = self.handle_pooling

    def extract_feature_map(self, model_with_rows) -> List[List[float]]:
        """
        Extracts a per-layer representation of the CNN topology for BERT tokenization.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.

        Returns:
            List[List[float]]: A sequence of feature vectors, one per layer.
        """
        # utils.print_flush("Starting Topology FE")
        topology_sequence = []

        for layer in model_with_rows.all_layers:
            handler = self.layer_type_to_function.get(type(layer), None)
            if handler:
                topology_sequence.append(handler(layer))
            else:
                topology_sequence.append([0.0] * 7)  # Default for unrecognized layers
        # utils.print_flush("Finished Topology FE")
        return topology_sequence

    @staticmethod
    def handle_linear(layer) -> List[float]:
        return [1, 0, 0, 0, 0, layer.in_features, layer.out_features]

    @staticmethod
    def handle_conv(layer) -> List[float]:
        as_scalar = lambda value: value[0] if isinstance(value, (tuple, list)) else value
        return [2, layer.in_channels, layer.out_channels, as_scalar(layer.kernel_size),
                as_scalar(layer.stride), as_scalar(layer.padding), layer.groups]

    @staticmethod
    def handle_batchnorm(layer) -> List[float]:
        return [3, layer.num_features, 0, 0, 0, 0, 0]

    @staticmethod
    def handle_groupnorm(layer) -> List[float]:
        return [3, layer.num_channels, layer.num_groups, 0, 0, 0, 0]

    @staticmethod
    def handle_layernorm(layer) -> List[float]:
        size = layer.normalized_shape[-1] if layer.normalized_shape else 0
        return [3, size, 0, 0, 0, 0, 0]

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
        # Adaptive pooling has an output_size instead of a kernel/stride/padding triple
        def scalar(attribute):
            value = getattr(layer, attribute, 0)
            if isinstance(value, (tuple, list)):
                return value[0] if value and value[0] is not None else 0
            return value if value is not None else 0

        return [7, scalar('kernel_size'), scalar('stride'), scalar('padding'),
                scalar('output_size'), 0, 0]
