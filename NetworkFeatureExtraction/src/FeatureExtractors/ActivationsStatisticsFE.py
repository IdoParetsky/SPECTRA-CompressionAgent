import numpy as np
import torch
from scipy.stats import skew, kurtosis
from .BaseFE import BaseFE
from ..utils import get_scaler_exponent


class ActivationsStatisticsFE(BaseFE):
    def __init__(self, model_with_rows, train_dataloader, device):
        """
        Extracts statistical features from activations across all layers of a CNN.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
            train_dataloader: DataLoader providing input samples for activation extraction.
            device: Device (CPU/GPU) for computation.
        """
        super(ActivationsStatisticsFE, self).__init__(model_with_rows)
        self.train_dataloader = train_dataloader
        self.device = device

    def extract_feature_map(self):
        """
        Extracts activation statistics (moments, min/max, norms) across all layers.

        Returns:
            activations_sequence (List[List[float]]): A sequential representation of activation statistics.
        """
        activations_sequence = []

        # Register hooks to collect activations
        unregister_hooks = self.register_hooks()
        all_activations.clear()

        # Forward pass to collect activations from the dataset
        with torch.no_grad():
            for batch_x, _ in self.train_dataloader:
                self.model_with_rows.model(batch_x.to(self.device))

        # Remove hooks after forward pass
        [hook.remove() for hook in unregister_hooks]

        # Compute activation statistics per layer
        for layer_activations in all_activations:
            activations_sequence.append(self.compute_statistics(layer_activations))

        return activations_sequence

    def register_hooks(self):
        """
        Registers forward hooks to capture activations from key layers.

        Returns:
            List: Unregistration handles for hooks.
        """
        unregister_hooks = []
        for row in self.model_with_rows.all_rows[:-1]:  # Exclude output layer
            important_layer = row[0]

            for layer in row[1:]:
                if 'activation' in str(type(layer)):
                    important_layer = layer

            unregister_hooks.append(important_layer.register_forward_hook(save_activations))

        return unregister_hooks

    @staticmethod
    def compute_statistics(layer_activations):
        if len(layer_activations.shape) == 4:  # (batch, channels, height, width)
            flattened = layer_activations.reshape(layer_activations.shape[0], layer_activations.shape[1], -1)
            layer_activations = np.mean(flattened, axis=0)  # Shape: (channels, features)

        scaler_exp = get_scaler_exponent(layer_activations)
        layer_activations_scaled = layer_activations * (10 ** -scaler_exp)

        # Compute raw statistics
        mean = np.mean(layer_activations, axis=1)
        std = np.std(layer_activations, axis=1)
        skewness = skew(layer_activations_scaled, axis=1)
        kurt = kurtosis(layer_activations_scaled, axis=1)
        min_val = np.min(layer_activations, axis=1)
        max_val = np.max(layer_activations, axis=1)
        l1_norm = np.linalg.norm(layer_activations, axis=1, ord=1)
        l2_norm = np.linalg.norm(layer_activations, axis=1, ord=2)

        # Combine all features
        all_features = np.concatenate([
            mean, std, skewness, kurt, min_val, max_val, l1_norm, l2_norm
        ])

        # TODO: Delve deeper if more issues / inaccuracies arise
        # Sanitize NaNs (noticed 1,216 NaNs out of a 2387668 feature vector
        all_features = np.nan_to_num(all_features, nan=0.0)

        return all_features.tolist()


# Global storage for activation outputs
all_activations = []


def save_activations(self, input, output):
    """
    Hook function to save activations during forward pass.

    Args:
        input (Tensor): Input tensor to the layer.
        output (Tensor): Output tensor from the layer.
    """
    global all_activations
    all_activations.append(output.detach().cpu().numpy())
