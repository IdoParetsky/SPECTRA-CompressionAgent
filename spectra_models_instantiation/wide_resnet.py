"""CIFAR Wide ResNet (Zagoruyko & Komodakis). Native 32×32 stem.

Factories used by V5 catalog prep. Signature matches SPECTRA catalogs:
``fn(num_classes, large_input)``. ``large_input`` is ignored (CIFAR stem only).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class WideBasic(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_in_out = in_planes == out_planes
        self.conv_shortcut = None
        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(in_planes, out_planes, 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        residual = x if self.equal_in_out else self.conv_shortcut(x)
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + residual


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, drop_rate=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, n_stages[0], 3, stride=1, padding=1, bias=False)
        self.layer1 = self._block(n_stages[0], n_stages[1], n, 1, drop_rate)
        self.layer2 = self._block(n_stages[1], n_stages[2], n, 2, drop_rate)
        self.layer3 = self._block(n_stages[2], n_stages[3], n, 2, drop_rate)
        self.bn = nn.BatchNorm2d(n_stages[3])
        self.fc = nn.Linear(n_stages[3], num_classes)

    def _block(self, in_planes, out_planes, num_blocks, stride, drop_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasic(in_planes, out_planes, s, drop_rate))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def wrn_16_4(num_classes, large_input):
    return WideResNet(16, 4, num_classes=num_classes, drop_rate=0.3)


def wrn_16_8(num_classes, large_input):
    return WideResNet(16, 8, num_classes=num_classes, drop_rate=0.3)


def wrn_28_2(num_classes, large_input):
    return WideResNet(28, 2, num_classes=num_classes, drop_rate=0.3)


def wrn_28_10(num_classes, large_input):
    return WideResNet(28, 10, num_classes=num_classes, drop_rate=0.3)
