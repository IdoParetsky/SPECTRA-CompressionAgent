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
                                             output_device=self.device.index)
        self.model_with_rows.model.eval()

    def extract_feature_map(self) -> List[List[float]]:
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
            if is_activation_layer(layer):
                def get_activation_hook(index):
                    def hook(module, input, output):
                        stats = self.compute_moments(output.detach())
                        activation_maps_collector[index].append(torch.tensor(list(stats.values()), device=self.device))

                    return hook

                hooks.append(layer.register_forward_hook(get_activation_hook(idx)))

        with torch.no_grad():
            utils.print_flush("Running full forward pass...")
            for i, (batch_x, _) in enumerate(self.train_dataloader):
                self.model_with_rows.model(batch_x.to(self.device, non_blocking=True))
            utils.print_flush("Full forward pass completed")

        for hook in hooks:
            hook.remove()

        # Average statistics across batches for each layer
        activation_maps = []
        for stats_list in activation_maps_collector:
            if stats_list:
                mean_stats = torch.stack(stats_list).mean(dim=0)
                activation_maps.append(mean_stats.tolist())
            else:
                activation_maps.append([0.0] * len(self.compute_moments(torch.zeros(1, device=self.device))))

        utils.print_flush("Finished Activations FE")
        return activation_maps
