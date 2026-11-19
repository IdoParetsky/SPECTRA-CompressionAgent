"""EfficientNet for 32x32 CIFAR inputs: B0, two narrower B0s and B1.

MBConv blocks: 1x1 expansion conv, depthwise 3x3 / 5x5 conv, squeeze-and-excitation (two 1x1
convs and a sigmoid gate that multiplies the feature map), 1x1 linear projection, residual
add when the shapes match; swish (SiLU) activations. B0 has 3,598,598 parameters for 10
classes (torchvision's EfficientNet-B0 trunk; the kuangliu network).

Source: kuangliu/pytorch-cifar, ``models/efficientnet.py``
(https://github.com/kuangliu/pytorch-cifar, MIT licence, Copyright (c) 2017 liukuang), after
Tan and Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
(ICML 2019, arXiv:1905.11946). As in the kuangliu file, the stem has stride 1, the stage
strides are (1, 2, 2, 2, 1, 2, 1), and the classifier reads the last stage directly (there is
no 1280-channel head conv), with dropout 0.2 before it.

Changes from the kuangliu file:
* the paper's compound scaling is added (``width_mult`` with the reference ``round_filters``,
  divisor 8, and ``depth_mult`` with ``ceil``): B1 is width 1.0, depth 1.1; the x0.5 and x0.75
  nets are narrower B0s;
* a block with expansion ratio 1 has no expansion conv (kuangliu builds one and never calls
  it; an uncalled layer would sit in SPECTRA's layer rows and parameter count);
* no drop-connect: the kuangliu file computes every block's rate as 0 (its block counter is
  never advanced), so its networks train without stochastic depth, and so do these;
* ``nn.SiLU`` / ``nn.Sigmoid`` / ``nn.Dropout`` modules instead of the functional forms (same
  function, same state_dict keys for the layers that have weights).

SPECTRA notes: the stem conv is the first registered layer and the classifier the last, in
forward order. The squeeze-and-excitation gate is ``x * gate``: SPECTRA's channel groups tie
the gate's second conv to the expanded feature map, so they are cut together.
``large_input`` is accepted for the loader's contract and ignored: this is a 32-px definition.
"""
import math

import torch
import torch.nn as nn

__all__ = ["EfficientNet", "efficientnet_b0", "efficientnet_b0_x050", "efficientnet_b0_x075", "efficientnet_b1"]

# EfficientNet-B0 (kuangliu's CIFAR strides)
B0 = {
    "num_blocks": (1, 2, 2, 3, 3, 4, 1),
    "expansion": (1, 6, 6, 6, 6, 6, 6),
    "out_channels": (16, 24, 40, 80, 112, 192, 320),
    "kernel_size": (3, 3, 5, 3, 5, 5, 3),
    "stride": (1, 2, 2, 2, 1, 2, 1),
}
STEM = 32
SE_RATIO = 0.25


def round_filters(filters: int, width_mult: float, divisor: int = 8) -> int:
    """The reference implementation's channel rounding: nearest multiple of ``divisor``, never under 90 %."""
    filters = filters * width_mult
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_mult: float) -> int:
    return int(math.ceil(depth_mult * repeats))


class SE(nn.Module):
    """Squeeze-and-excitation with swish."""

    def __init__(self, in_channels: int, se_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se1 = nn.Conv2d(in_channels, se_channels, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.se2 = nn.Conv2d(se_channels, in_channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.se1(self.pool(x)))
        out = self.gate(self.se2(out))
        return x * out


class Block(nn.Module):
    """Expansion + depthwise + squeeze-and-excitation + pointwise projection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 expand_ratio: int = 1, se_ratio: float = 0.0):
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        channels = expand_ratio * in_channels

        # Expansion (absent at ratio 1)
        if expand_ratio != 1:
            self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.act1 = nn.SiLU(inplace=True)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                               padding=(1 if kernel_size == 3 else 2), groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.SiLU(inplace=True)

        # Squeeze-and-excitation, sized on the block's input width
        self.se = SE(channels, max(1, int(in_channels * se_ratio)))

        # Linear projection
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MobileNetV2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x if self.expand_ratio == 1 else self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0, depth_mult: float = 1.0,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.width_mult, self.depth_mult = float(width_mult), float(depth_mult)
        in_channels = round_filters(STEM, width_mult)
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU(inplace=True)
        layers = []
        for expansion, base_out, base_blocks, kernel_size, stride in zip(
                B0["expansion"], B0["out_channels"], B0["num_blocks"], B0["kernel_size"], B0["stride"]):
            out_channels = round_filters(base_out, width_mult)
            for index in range(round_repeats(base_blocks, depth_mult)):
                layers.append(Block(in_channels, out_channels, kernel_size, stride if index == 0 else 1,
                                    expansion, se_ratio=SE_RATIO))
                in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        return self.linear(out)


def efficientnet_b0_x050(num_classes, large_input):
    return EfficientNet(num_classes=num_classes, width_mult=0.5, depth_mult=1.0)


def efficientnet_b0_x075(num_classes, large_input):
    return EfficientNet(num_classes=num_classes, width_mult=0.75, depth_mult=1.0)


def efficientnet_b0(num_classes, large_input):
    return EfficientNet(num_classes=num_classes, width_mult=1.0, depth_mult=1.0)


def efficientnet_b1(num_classes, large_input):
    return EfficientNet(num_classes=num_classes, width_mult=1.0, depth_mult=1.1)
