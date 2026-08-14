import torch
import torch.nn as nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class LinearMaskedRelu(nn.Module):
    def __init__(self, size, value=1.0, bias=0.0):
        super().__init__()
        self.size = size
        self.weight = nn.Parameter(torch.full(size, value))
        self.bias = nn.Parameter(torch.full(size, bias))

    def forward(self, x):
        return F.relu(x * self.weight + self.bias)


class CifarCNNB(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(3, 32, (5, 5), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 128, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(8192, 250),
            nn.ReLU(),
            nn.Linear(250, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CifarCNNBGraft(nn.Module):
    def __init__(self, num_classes=10, v=1.0, b=0.0):
        super().__init__()
        self.layer0 = nn.ZeroPad2d((1, 2, 1, 2))
        self.layer1 = nn.Conv2d(3, 32, (5, 5), stride=2, padding=0)
        self.linear_masked_relu2 = LinearMaskedRelu(size=(32, 16, 16), value=v, bias=b)
        self.layer3 = nn.Conv2d(32, 128, (4, 4), stride=2, padding=1)
        self.linear_masked_relu4 = LinearMaskedRelu(size=(128, 8, 8), value=v, bias=b)
        self.layer5 = Flatten()
        self.layer6 = nn.Linear(8192, 250)
        self.linear_masked_relu7 = LinearMaskedRelu(size=(250,), value=v, bias=b)
        self.layer8 = nn.Linear(250, num_classes)
        self._initialize_weights()

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.linear_masked_relu2(out)
        out = self.layer3(out)
        out = self.linear_masked_relu4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.linear_masked_relu7(out)
        out = self.layer8(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def cnn_b_2_255_baseline(num_classes=10, large_input=False):
    return CifarCNNB(num_classes=num_classes)


def cnn_b_2_255_graft(num_classes=10, large_input=False):
    return CifarCNNBGraft(num_classes=num_classes, v=1.0, b=0.0)
