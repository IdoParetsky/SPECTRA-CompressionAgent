from torch import nn
from src.Model.Agent import Agent


class Actor(Agent):
    def __init__(self, device, num_outputs):
        super().__init__(device, num_outputs)
        self.is_actor = True

        self.actor = nn.Sequential(
            nn.Linear(170, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, num_outputs),
            nn.Softmax(dim=1),
        )
