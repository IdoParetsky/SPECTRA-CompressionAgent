import os
import torch
from torch import nn
from typing import List
from .BaseFE import BaseFE


class ActivationsStatisticsFE(BaseFE):
    def __init__(self, train_dataloader, device):
        """
        Extracts statistical features from activations across all layers of a CNN.

        Args:
            train_dataloader: DataLoader providing input samples for activation extraction.
            device: Device GPU(s) for computation.
        """
        self.train_dataloader = train_dataloader
        self.device = device
        self.cached_activation_maps = None

        # Fixed probe batch(es), materialised once and reused for every state extraction.
        # Drawing from a shuffled loader each visit made the same network yield a different
        # state — noise that the RL signal then has to absorb (critique §8). Override the
        # count with SPECTRA_PROBE_BATCHES; larger values are for later experiment runs, not
        # for the short correctness smoke tests.
        self.num_probe_batches = max(1, int(os.environ.get("SPECTRA_PROBE_BATCHES", "2")))
        self._probe_batches = None  # list[Tensor] on CPU; moved to device only when used

    def _ensure_probe_batches(self):
        """Capture the first N batches once; subsequent calls reuse them."""
        if self._probe_batches is not None:
            return
        batches = []
        for i, (batch_x, _) in enumerate(self.train_dataloader):
            # Keep on CPU so a long-lived FeatureExtractor does not pin GPU memory for the
            # whole episode; the copy to device is cheap relative to the forward pass.
            batches.append(batch_x.detach().cpu().contiguous())
            if i + 1 >= self.num_probe_batches:
                break
        if not batches:
            raise RuntimeError("ActivationsStatisticsFE: training loader yielded no batches")
        self._probe_batches = batches

    def extract_feature_map(self, model_with_rows, update_indices=None) -> List[List[float]]:
        if update_indices is None:
            update_indices = list(range(len(model_with_rows.all_layers)))

        num_layers = len(model_with_rows.all_layers)
        if self.cached_activation_maps is None:
            self.cached_activation_maps = [None] * num_layers
        elif len(self.cached_activation_maps) != num_layers:
            # Structural changes can alter the layer count; keep the cache aligned with it
            self.cached_activation_maps = (self.cached_activation_maps + [None] * num_layers)[:num_layers]

        update_indices = [idx for idx in update_indices if idx < num_layers]

        activation_maps_collector = [[] for _ in model_with_rows.all_layers]
        hooks = []

        def is_observable_layer(layer: nn.Module) -> bool:
            # Convolutions, linear layers and normalisations are instrumented alongside the
            # activation modules. Many architectures call activations functionally
            # (F.relu inside forward), and hooking only activation *modules* left those
            # networks with an all-zero activation feature map -- precisely the networks a
            # generic agent is meant to handle.
            return isinstance(layer, (
                nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear,
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm,
                nn.ReLU, nn.ReLU6, nn.ELU, nn.SiLU, nn.Softmax, nn.Tanh, nn.Sigmoid,
                nn.LeakyReLU, nn.GELU, nn.Hardtanh, nn.Hardswish, nn.Softplus, nn.Softsign,
                nn.PReLU, nn.LogSigmoid, nn.SELU, nn.CELU, nn.GLU, nn.Mish,
            )) or "activation" in layer.__class__.__name__.lower()

        for idx, layer in enumerate(model_with_rows.all_layers):
            if idx in update_indices and is_observable_layer(layer):
                def get_activation_hook(index):
                    def hook(module, input, output):
                        activation_maps_collector[index].append(
                            self.compute_moments_tensor(output.detach()))

                    return hook

                hooks.append(layer.register_forward_hook(get_activation_hook(idx)))

        self._ensure_probe_batches()
        with torch.no_grad():
            for batch_x in self._probe_batches:
                model_with_rows.model(batch_x.to(self.device, non_blocking=True))

        for hook in hooks:
            hook.remove()

        # Average statistics across the fixed probe batches for each layer
        empty_stats = [0.0] * len(self.MOMENT_NAMES)
        for idx in update_indices:
            if activation_maps_collector[idx]:
                mean_stats = torch.stack(activation_maps_collector[idx]).mean(dim=0)
                self.cached_activation_maps[idx] = mean_stats.tolist()
            elif self.cached_activation_maps[idx] is None:
                self.cached_activation_maps[idx] = list(empty_stats)

        for idx in range(len(self.cached_activation_maps)):
            if self.cached_activation_maps[idx] is None:
                self.cached_activation_maps[idx] = list(empty_stats)

        return self.cached_activation_maps
