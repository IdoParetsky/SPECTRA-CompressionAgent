from torch import nn


class NetX(nn.Module):
    def __init__(self):
        super(NetX, self).__init__()
    def forward(self, x):
        x = self.model(x)
        return x