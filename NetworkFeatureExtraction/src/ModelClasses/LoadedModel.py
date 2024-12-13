from enum import Enum, auto

from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer


class MissionTypes(Enum):
    Regression = auto()
    Classification = auto()


class LoadedModel:
    model: nn.Module
    optimizer: Optimizer
    mission_type: MissionTypes
    loss: _Loss

    def __init__(self, model, optimizer, missionType, loss):
        self.model = model
        self.optimizer = optimizer
        self.mission_type = missionType
        self.loss = loss
