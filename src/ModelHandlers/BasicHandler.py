from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


class BasicHandler:
    def __init__(self, model: nn.Module, loss_function: _Loss):
        self.model = model
        self.loss_func = loss_function

    def evaluate_model(self, loader: DataLoader) -> float:
        pass

    def train_model(self, train_loader: DataLoader):
        pass

    def freeze_all_layers_but_pruned(self, params_to_keep_trainable):
        for curr_l in self.model.parameters():
            if id(curr_l) in params_to_keep_trainable:
                curr_l.requires_grad = True
            else:
                curr_l.requires_grad = False

    def unfreeze_all_layers(self):
        for curr_l in self.model.parameters():
            curr_l.requires_grad = True
