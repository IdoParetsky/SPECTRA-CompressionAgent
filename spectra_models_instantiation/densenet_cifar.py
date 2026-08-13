"""
CIFAR DenseNet-BC (Huang et al., CVPR 2017).

Thesis pool names DenseNet-40 / DenseNet-100. Both are the bottleneck-compressed
(DenseNet-BC) variants used in the CIFAR literature:

* densenet40  — depth 40, growth 12, 3 blocks of 6 bottleneck layers
* densenet100 — depth 100, growth 12, 3 blocks of 16 bottleneck layers

``large_input`` selects an ImageNet-style 7x7 stem; CIFAR/SVHN keep the 3x3 stride-1 stem.
The block uses ``torch.cat`` on the channel axis so SPECTRA's channel-group tracer can
resize DenseNet concatenations the same way it does torchvision DenseNet-121.
"""
from collections import OrderedDict
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

    def forward(self, input_features: Tensor) -> Tensor:  # noqa: A003 - torchvision name
        if isinstance(input_features, Tensor):
            prev_features = [input_features]
        else:
            prev_features = input_features
        concated = torch.cat(prev_features, 1)
        bottleneck = self.conv1(self.relu1(self.norm1(concated)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))
        if self.drop_rate > 0.0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int,
                 growth_rate: int, drop_rate: float):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for layer in self.values():
            features.append(layer(features))
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    def __init__(self, growth_rate: int = 12, block_config: Tuple[int, ...] = (6, 6, 6),
                 num_init_features: int = 24, bn_size: int = 4, drop_rate: float = 0.0,
                 num_classes: int = 10, large_input: bool = False):
        super().__init__()
        if large_input:
            self.features = nn.Sequential(OrderedDict([
                ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ("norm0", nn.BatchNorm2d(num_init_features)),
                ("relu0", nn.ReLU(inplace=True)),
                ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ("conv0", nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
                ("norm0", nn.BatchNorm2d(num_init_features)),
                ("relu0", nn.ReLU(inplace=True)),
            ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


def _bc(depth: int, growth_rate: int, num_classes: int, large_input: bool) -> DenseNet:
    # DenseNet-BC: 3 blocks, bottleneck (1x1+3x3) counts as 2 layers in the depth formula.
    if (depth - 4) % 6 != 0:
        raise ValueError(f"DenseNet-BC depth must satisfy (depth-4) % 6 == 0, got {depth}")
    n = (depth - 4) // 6
    return DenseNet(
        growth_rate=growth_rate,
        block_config=(n, n, n),
        num_init_features=2 * growth_rate,
        num_classes=num_classes,
        large_input=large_input,
    )


def densenet40(num_classes: int, large_input: bool = False) -> DenseNet:
    return _bc(40, 12, num_classes, large_input)


def densenet100(num_classes: int, large_input: bool = False) -> DenseNet:
    return _bc(100, 12, num_classes, large_input)
