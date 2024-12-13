from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class BasicHandler:
    def __init__(self, model: nn.Module, loss_function: _Loss, optimizer: Optimizer):
        self.model = model
        self.loss_func = loss_function
        self.optimizer = optimizer

    def evaluate_model(self, loader: DataLoader) -> float:
        pass

    def train_model(self, train_loader: DataLoader):
        pass

    def freeze_layers(self, layers_to_freeze):
        for curr_l in self.model.parameters():
            if id(curr_l) in layers_to_freeze:
                curr_l.requires_grad = True
            else:
                curr_l.requires_grad = False

    def unfreeze_all_layers(self):
        for curr_l in self.model.parameters():
            curr_l.requires_grad = True
