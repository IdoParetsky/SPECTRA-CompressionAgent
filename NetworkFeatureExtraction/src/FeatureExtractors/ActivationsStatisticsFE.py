import numpy as np
import torch
from scipy.stats import skew, kurtosis
from .BaseFE import BaseFE
from ..utils import pad_with_columns, pad_with_rows, get_scaler_exponent


class ActivationsStatisticsFE(BaseFE):
    def __init__(self, model_with_rows, dataset_x, device):
        super(ActivationsStatisticsFE, self).__init__(model_with_rows)
        self.MAX_LAYER_SIZE = 1000  # TODO: To be overridden via BERT mechanism
        self.MAX_LAYERS = 10  # TODO: To be overridden via BERT mechanism
        self.dataset_x = dataset_x
        self.device = device

    def extract_feature_map(self, layer_index):
        """
        Extracts activation statistics (moments, min/max, norms) for each layer row.
        """
        moment_map = [[], [], [], []]  # Mean, Std, Skew, Kurtosis
        min_max_map = [[], []]         # Min, Max
        norm_map = [[], []]            # L1 Norm, L2 Norm

        # Register hooks to collect activations
        unregister_hook_functions = self.build_register_forward_hooks_to_important_layers()
        all_activations_in_important_layers.clear()

        # Forward pass to collect activations
        self.model_with_rows.model(torch.Tensor(self.dataset_x).to(self.device))
        [x.remove() for x in unregister_hook_functions]

        # Calculate statistics
        self.calculate_stats_for_each_layer(moment_map, min_max_map, norm_map)
        activations_map = np.array([*moment_map, *min_max_map, *norm_map])

        activations_map = np.array(list(map(lambda f_map: pad_with_rows(f_map, self.MAX_LAYERS), activations_map)))  # TODO: To be overridden via BERT mechanism

        return activations_map, activations_map[:, layer_index, :]

    def calculate_stats_for_each_layer(self, moment_map, min_max_map, norm_map):
        """
        Calculates statistics for activations collected in all_activations_in_important_layers.
        """
        for layer_activations in all_activations_in_important_layers:
            # Handle 4D activations (CNN) and flatten them per channel
            if len(layer_activations.shape) == 4:  # Shape: (batch_size, channels, height, width)
                flattened_activations = layer_activations.reshape(
                    layer_activations.shape[0], layer_activations.shape[1], -1
                )  # Shape: (batch_size, channels, num_elements_per_channel)
                layer_activations_transposed = np.mean(flattened_activations, axis=0)  # Average over batch
            else:
                # For dense layers or activations with no spatial dimensions
                layer_activations_transposed = np.array(layer_activations).T

            # Scale activations
            scaler_exponent = get_scaler_exponent(layer_activations_transposed)
            layer_activations_transposed_scaled = layer_activations_transposed * (10 ** -scaler_exponent)
            layer_activations_transposed_scaled = np.array(layer_activations_transposed_scaled, dtype=float)

            # Compute moments
            mean = np.mean(layer_activations_transposed, axis=1)
            std_val = np.std(layer_activations_transposed, axis=1)
            skew_val = skew(layer_activations_transposed_scaled, axis=1)
            kurtosis_val = kurtosis(layer_activations_transposed_scaled, axis=1)

            all_moments = [mean, std_val, skew_val, kurtosis_val]

            # Min/Max
            min_per_neuron = np.min(layer_activations_transposed, axis=1)
            max_per_neuron = np.max(layer_activations_transposed, axis=1)

            min_max_map[0].append(pad_with_columns(min_per_neuron, self.MAX_LAYER_SIZE))  # TODO: To be overridden via BERT mechanism
            min_max_map[1].append(pad_with_columns(max_per_neuron, self.MAX_LAYER_SIZE))  # TODO: To be overridden via BERT mechanism

            # L1/L2 Norms
            l1_per_neuron = np.linalg.norm(layer_activations_transposed, axis=1, ord=1)
            l2_per_neuron = np.linalg.norm(layer_activations_transposed, axis=1, ord=2)

            norm_map[0].append(pad_with_columns(l1_per_neuron, self.MAX_LAYER_SIZE))  # TODO: To be overridden via BERT mechanism
            norm_map[1].append(pad_with_columns(l2_per_neuron, self.MAX_LAYER_SIZE))  # TODO: To be overridden via BERT mechanism

            for m in range(4):
                moment_map[m].append(pad_with_columns(all_moments[m], self.MAX_LAYER_SIZE))  # TODO: To be overridden via BERT mechanism

    def build_register_forward_hooks_to_important_layers(self):
        """
        Registers forward hooks to collect activations from important layers.
        """
        unregister_hook_function = []
        for curr_row in self.model_with_rows.all_rows[:-1]:
            most_important_layer = curr_row[0]

            for curr_layer in curr_row[1:]:
                if 'activation' in str(type(curr_layer)):
                    most_important_layer = curr_layer

            unregister_hook_function.append(most_important_layer.register_forward_hook(save_activations))

        return unregister_hook_function


# Global variable to store activations from hooks
all_activations_in_important_layers = []


def save_activations(self, input, output):
    """
    Hook function to save activations during forward pass.
    """
    global all_activations_in_important_layers
    if len(output.shape) == 4:  # Handle Conv2D activations
        activations = output.detach().cpu().numpy()
    else:
        activations = output.detach().cpu().numpy()
    all_activations_in_important_layers.append(activations)
    return
