from torch import nn
from src.Model.Agent import Agent


class Critic(Agent):
    def __init__(self, device, num_outputs):
        super().__init__(device, num_outputs)
        self.is_critic = True
