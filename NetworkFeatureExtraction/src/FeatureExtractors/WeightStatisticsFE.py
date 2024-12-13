from typing import List

import numpy as np
from scipy.stats import skew, kurtosis
from torch.nn import Linear, Conv2d

from .BaseFE import BaseFE
from ..utils import pad_with_columns, pad_with_rows


class WeightStatisticsFE(BaseFE):

    def __init__(self, model_with_rows):
        super(WeightStatisticsFE, self).__init__(model_with_rows)
        self.MAX_LAYER_SIZE = 1000  # TODO: To be overridden via BERT mechanism
        self.MAX_LAYERS = 10  # TODO: To be overridden via BERT mechanism

    def extract_feature_map(self, layer_index):
        """
        Extracts weight statistics (moments, min/max, norms) for each layer row.
        """
        layer_weights_for_each_row: List[Linear] = list(map(lambda row: row[0], self.model_with_rows.all_rows))

        moment_map = [[], [], [], []]  # Mean, Std, Skew, Kurtosis
        min_max_map = [[], []]         # Min, Max
        norm_map = [[], []]            # L1 Norm, L2 Norm

        for curr_layer in layer_weights_for_each_row:
            if isinstance(curr_layer, Linear):
                self.handle_linear_layer(curr_layer, moment_map, min_max_map, norm_map)
            elif isinstance(curr_layer, Conv2d):
                self.handle_conv2d_layer(curr_layer, moment_map, min_max_map, norm_map)

        weights_map = np.array([*moment_map, *min_max_map, *norm_map])

        weights_map = np.array(list(map(lambda f_map: pad_with_rows(f_map, self.MAX_LAYERS),weights_map)))  # TODO: To be overridden via BERT mechanism

        return weights_map, weights_map[:, layer_index, :]

    def handle_linear_layer(self, curr_layer, moment_map, min_max_map, norm_map):
        """
        Extracts statistics for Linear layers' weights.
        """
        curr_layer_weights = curr_layer.weight.detach().cpu().numpy()

        self.compute_weight_statistics(curr_layer_weights, moment_map, min_max_map, norm_map)

    def handle_conv2d_layer(self, curr_layer, moment_map, min_max_map, norm_map):
        """
        Extracts statistics for Conv2D layers' weights.
        """
        # Conv2D weights have shape: (out_channels, in_channels, kernel_height, kernel_width)
        curr_layer_weights = curr_layer.weight.detach().cpu().numpy()
        flattened_weights = curr_layer_weights.reshape(curr_layer_weights.shape[0], -1)  # Flatten each filter

        self.compute_weight_statistics(flattened_weights, moment_map, min_max_map, norm_map)

    def compute_weight_statistics(self, weights, moment_map, min_max_map, norm_map):
        """
        Computes and pads statistical features: mean, std, skew, kurtosis, min/max, L1/L2 norms.
        """
        mean = np.mean(weights, axis=1)
        std_val = np.std(weights, axis=1)
        skew_val = skew(weights, axis=1)
        kurtosis_val = kurtosis(weights, axis=1)
        all_moments = [mean, std_val, skew_val, kurtosis_val]

        all_moments_padded = list(map(lambda moment: pad_with_columns(moment, self.MAX_LAYER_SIZE), all_moments))  # TODO: To be overridden via BERT mechanism
        min_per_neuron = pad_with_columns(np.min(weights, axis=1), self.MAX_LAYER_SIZE)  # TODO: To be overridden via BERT mechanism
        max_per_neuron = pad_with_columns(np.max(weights, axis=1), self.MAX_LAYER_SIZE)  # TODO: To be overridden via BERT mechanism

        min_max_map[0].append(min_per_neuron)
        min_max_map[1].append(max_per_neuron)

        l1_per_neuron = np.linalg.norm(weights, axis=1, ord=1)
        l2_per_neuron = np.linalg.norm(weights, axis=1, ord=2)

        norm_map[0].append(pad_with_columns(l1_per_neuron, self.MAX_LAYER_SIZE))  # TODO: To be overridden via BERT mechanism
        norm_map[1].append(pad_with_columns(l2_per_neuron, self.MAX_LAYER_SIZE))  # TODO: To be overridden via BERT mechanism

        for i, curr_moment in enumerate(all_moments_padded):
            moment_map[i].append(curr_moment)