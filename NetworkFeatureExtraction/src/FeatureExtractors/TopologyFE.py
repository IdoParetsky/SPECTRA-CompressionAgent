from enum import Enum

import numpy as np
import torch
import pandas as pd
from ..FeatureExtractors.BaseFE import BaseFE


class Activations(Enum):
    ReLU = 1  # Results in NaN values in some datasets
    ELU = 1
    SiLU = 1
    Softmax = 2
    Tanh = 3
    Sigmoid = 4


class BatchNorm(Enum):
    BatchNorm1d = 1
    BatchNorm2d = 2


class TopologyFE(BaseFE):
    BatchNorm = 0
    Activation = 1
    Dropout = 2
    InFeatures = 3
    OutFeatures = 4
    KernelSize = 5
    Stride = 6
    Padding = 7

    num_features = 8

    def __init__(self, model_with_rows):
        super(TopologyFE, self).__init__(model_with_rows)
        self.all_layers = []
        self.MAX_LAYERS = 10  # TODO: To be overridden via BERT mechanism

        self.layer_type_to_function = {
            torch.nn.modules.linear.Linear: self.handle_linear,

            torch.nn.modules.conv.Conv2d: self.handle_conv2d,

            torch.nn.modules.batchnorm.BatchNorm1d: self.handle_batchnorm(BatchNorm.BatchNorm1d),
            torch.nn.modules.batchnorm.BatchNorm2d: self.handle_batchnorm(BatchNorm.BatchNorm2d),

            torch.nn.modules.activation.ReLU: self.handle_activation(Activations.ReLU),  # NEON - Results in NaN values in some datasets
            torch.nn.modules.activation.ELU: self.handle_activation(Activations.ELU),  # NEON - Works fine
            torch.nn.modules.activation.SiLU: self.handle_activation(Activations.SiLU),  # NEON - Yet to be verified
            torch.nn.modules.activation.Softmax: self.handle_activation(Activations.Softmax),
            torch.nn.modules.activation.Tanh: self.handle_activation(Activations.Tanh),
            torch.nn.modules.activation.Sigmoid: self.handle_activation(Activations.Sigmoid),

            torch.nn.modules.dropout.Dropout: self.handle_dropout,
        }

    def extract_feature_map(self, layer_index):
        topology_map = np.zeros((self.MAX_LAYERS, self.num_features))  # TODO: To be overridden via BERT mechanism

        all_category_columns = ['activation_0.0', 'activation_1.0', 'activation_2.0', 'activation_3.0',
                                'activation_4.0']

        for i, curr_row in enumerate(self.model_with_rows.all_rows):
            for curr_layer in curr_row:
                # Use the layer type to determine which handler to call
                handler = self.layer_type_to_function.get(type(curr_layer), None)
                if handler is not None:
                    handler(curr_layer, topology_map[i])

        df = pd.DataFrame(topology_map, columns=['batchnorm', 'activation', 'dropout', 'in_features', 'out_features',
                                                 'kernel_size', 'stride', 'padding'])
        df_activations = pd.DataFrame({'activation': df['activation'].astype('category')})
        df_activations = pd.get_dummies(df_activations)
        df_activations['activation_0.0'] = 0

        df_activations = df_activations.T.reindex(all_category_columns).T.fillna(0)
        df = pd.concat([df, df_activations], axis=1)
        df = df.drop(columns=['activation', all_category_columns[0]])

        topology_map = df.to_numpy()

        return topology_map, topology_map[layer_index]

    @staticmethod
    def handle_linear(curr_layer, row_to_fill):
        """Handles Linear (Fully-Connected) layers."""
        row_to_fill[TopologyFE.InFeatures] = curr_layer.in_features
        row_to_fill[TopologyFE.OutFeatures] = curr_layer.out_features

    @staticmethod
    def handle_conv2d(curr_layer, row_to_fill):
        """Handles Conv2D layers and extracts kernel size, stride, and padding."""
        row_to_fill[TopologyFE.InFeatures] = curr_layer.in_channels
        row_to_fill[TopologyFE.OutFeatures] = curr_layer.out_channels
        row_to_fill[TopologyFE.KernelSize] = curr_layer.kernel_size[0] if isinstance(curr_layer.kernel_size, tuple) else curr_layer.kernel_size
        row_to_fill[TopologyFE.Stride] = curr_layer.stride[0] if isinstance(curr_layer.stride, tuple) else curr_layer.stride
        row_to_fill[TopologyFE.Padding] = curr_layer.padding[0] if isinstance(curr_layer.padding, tuple) else curr_layer.padding

    @staticmethod
    def handle_dropout(curr_layer, row_to_fill):
        """Handles Dropout layers."""
        row_to_fill[TopologyFE.Dropout] = 1

    @staticmethod
    def handle_batchnorm(batchnorm_type):
        """Handles BatchNorm layers (both 1D and 2D)."""
        def handler(curr_layer, row_to_fill):
            row_to_fill[TopologyFE.BatchNorm] = batchnorm_type.value
        return handler

    @staticmethod
    def handle_activation(activation_type):
        """Handles Activation layers."""
        def handler(curr_layer, row_to_fill):
            row_to_fill[TopologyFE.Activation] = activation_type.value
        return handler
