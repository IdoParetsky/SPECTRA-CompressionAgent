"""MobileNet v1 for 32x32 CIFAR inputs, with the paper's width multiplier.

A plain chain of depthwise-separable blocks (3x3 depthwise conv + BN + ReLU, 1x1 pointwise
conv + BN + ReLU), no residual connection and no expansion. 3,217,226 parameters at width
1.0 for 10 classes (the kuangliu network).

Source: kuangliu/pytorch-cifar, ``models/mobilenet.py``
(https://github.com/kuangliu/pytorch-cifar, MIT licence, Copyright (c) 2017 liukuang), after
Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications" (arXiv:1704.04861, 2017), whose width multiplier alpha (1, 0.75, 0.5, 0.25)
scales every channel count.

Changes from the kuangliu file, none of which changes the function at width 1.0 on 32-px
inputs: ReLU modules instead of ``F.relu`` (same state_dict keys), adaptive global average
pooling instead of ``F.avg_pool2d(out, 2)`` (the final map is 2x2), and the width multiplier.

SPECTRA notes: the stem conv is the first registered layer and the classifier the last, in
forward order. Each depthwise conv has ``groups == in == out``, which SPECTRA's channel
groups tie to the pointwise conv before it (the MobileNetV2 primitive). ``large_input`` is
accepted for the loader's contract and ignored: this is a 32-px definition.
"""
import torch
import torch.nn as nn

__all__ = ["MobileNetV1", "mobilenetv1_x025", "mobilenetv1_x050", "mobilenetv1_x075", "mobilenetv1_x100"]

# (128, 2) means a block with 128 output planes and stride 2; stride 1 by default
CFG = (64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024)
STEM = 32
MIN_WIDTH = 8


def scale_width(channels: int, mult: float) -> int:
    """``channels * mult`` rounded to an integer, at least ``MIN_WIDTH``."""
    return max(MIN_WIDTH, int(round(channels * mult)))


class Block(nn.Module):
    """Depthwise conv + pointwise conv."""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1,
                               groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


class MobileNetV1(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        self.width_mult = float(width_mult)
        in_planes = scale_width(STEM, width_mult)
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        layers = []
        for entry in CFG:
            planes, stride = (entry, 1) if isinstance(entry, int) else entry
            out_planes = scale_width(planes, width_mult)
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_planes, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


def mobilenetv1_x025(num_classes, large_input):
    return MobileNetV1(num_classes=num_classes, width_mult=0.25)


def mobilenetv1_x050(num_classes, large_input):
    return MobileNetV1(num_classes=num_classes, width_mult=0.5)


def mobilenetv1_x075(num_classes, large_input):
    return MobileNetV1(num_classes=num_classes, width_mult=0.75)


def mobilenetv1_x100(num_classes, large_input):
    return MobileNetV1(num_classes=num_classes, width_mult=1.0)
