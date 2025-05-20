import torch
from torch import nn
from typing import List
from .BaseFE import BaseFE
import src.utils as utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class ActivationsStatisticsFE(BaseFE):
    def __init__(self, model_with_rows, train_dataloader, device):
        """
        Extracts statistical features from activations across all layers of a CNN.

        Args:
            model_with_rows: ModelWithRows instance containing structured layer representation.
            train_dataloader: DataLoader providing input samples for activation extraction.
            device: Device GPU(s) for computation.
        """
        super().__init__(model_with_rows)
        self.train_dataloader = train_dataloader
        self.device = device

        # DDP Wrapping
        self.model_with_rows.model = self.model_with_rows.model.to(self.device)
        if torch.cuda.device_count() > 1 and dist.is_initialized():
            self.model_with_rows.model = DDP(self.model_with_rows.model, device_ids=[self.device.index],
                                             output_device=self.device.index, find_unused_parameters=True)
        self.model_with_rows.model.eval()
        self.cached_activation_maps = [None] * len(self.model_with_rows.all_layers)


    def extract_feature_map(self, update_indices=None) -> List[List[float]]:
        if update_indices is None:
            update_indices = list(range(len(self.model_with_rows.all_layers)))

        utils.print_flush("Starting Activations FE")

        activation_maps_collector = [[] for _ in self.model_with_rows.all_layers]
        hooks = []

        def is_activation_layer(layer: nn.Module) -> bool:
            return "activation" in layer.__class__.__name__.lower() or isinstance(layer, (
                nn.ReLU, nn.ELU, nn.SiLU, nn.Softmax, nn.Tanh, nn.Sigmoid,
                nn.LeakyReLU, nn.GELU, nn.Hardtanh, nn.Softplus, nn.Softsign,
                nn.PReLU, nn.LogSigmoid, nn.SELU, nn.CELU, nn.GLU
            ))

        for idx, layer in enumerate(self.model_with_rows.all_layers):
            if idx in update_indices and is_activation_layer(layer):
                def get_activation_hook(index):
                    def hook(module, input, output):
                        stats = self.compute_moments(output.detach())
                        activation_maps_collector[index].append(torch.tensor(list(stats.values()), device=self.device))

                    return hook

                hooks.append(layer.register_forward_hook(get_activation_hook(idx)))

        # TODO: Experiment with 2 batches only and later compare against a full forward pass (2-3 sec per step)
        #  Add to args so this becomes configurable (maybe even dynamic according to architecture and dataset?)

        # with torch.no_grad():
        #     utils.print_flush("Running full forward pass...")
        #     for i, (batch_x, _) in enumerate(self.train_dataloader):
        #         self.model_with_rows.model(batch_x.to(self.device, non_blocking=True))
        #     utils.print_flush("Full forward pass completed")

        with torch.no_grad():
            utils.print_flush("Running a 2-batches forward pass...")
            for i, (batch_x, _) in enumerate(self.train_dataloader):
                # Limit to 2 batches due to runtime considerations,
                # assuming stable mean and std can be extracted after only 2 batches
                if i > 1:
                    break
                self.model_with_rows.model(batch_x.to(self.device, non_blocking=True))
            utils.print_flush("2-batches forward pass completed")

        for hook in hooks:
            hook.remove()

        # Average statistics across batches for each layer
        for idx in update_indices:
            if activation_maps_collector[idx]:
                mean_stats = torch.stack(activation_maps_collector[idx]).mean(dim=0)
                self.cached_activation_maps[idx] = mean_stats.tolist()
            elif self.cached_activation_maps[idx] is None:
                # Ensure it’s not left as None
                self.cached_activation_maps[idx] = [0.0] * len(self.compute_moments(torch.zeros(1, device=self.device)))

        # Ensure all entries are valid (not just updated ones)
        for idx in range(len(self.cached_activation_maps)):
            if self.cached_activation_maps[idx] is None:
                self.cached_activation_maps[idx] = [0.0] * len(self.compute_moments(torch.zeros(1, device=self.device)))

        utils.print_flush("Finished Activations FE")
        return self.cached_activation_maps
