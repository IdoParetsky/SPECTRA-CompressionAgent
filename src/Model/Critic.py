from torch import nn
from src.Model.Agent import Agent


class Critic(Agent):
    def __init__(self, device, num_outputs):
        super().__init__(device, num_outputs)
        self.is_critic = True

        self.critic = nn.Sequential(
            nn.Linear(170, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
