"""GoogLeNet width variants for 32x32 CIFAR inputs (Inception v1 with batch norm).

The full-width network is ``googlenet`` in ``googlenet_cifar.py`` (same folder, about 6.2 M
parameters). This file adds the narrower members of the family for the sealed test set:
every channel count of that network times a width multiplier (rounded, floor of 8, the rule
``vgg_chenyaofo.scale_cfg`` uses), nothing else changed. At multiplier 1.0 the module names
and every state_dict shape are those of ``googlenet_cifar.GoogLeNet``
(``tests/test_sealed_families.py`` checks it), so ``googlenet_x100`` here and ``googlenet``
there are the same network.

Source: kuangliu/pytorch-cifar, ``models/googlenet.py``
(https://github.com/kuangliu/pytorch-cifar, MIT licence, Copyright (c) 2017 liukuang), after
Szegedy et al., "Going Deeper with Convolutions" (CVPR 2015). The width multiplier is ours.

SPECTRA notes: the stem conv is the first registered layer and the classifier the last, in
forward order (``ModelWithRows`` reads rows in registration order); plain ``torch.cat`` in
``forward`` so ``torch.fx`` traces the four-branch merge. ``large_input`` is accepted for the
loader's contract and ignored: this is a 32-px definition (global pooling is adaptive).
"""
import torch
import torch.nn as nn

__all__ = ["GoogLeNet", "googlenet_x025", "googlenet_x050", "googlenet_x075", "googlenet_x100"]

STEM = 192
MIN_WIDTH = 8
# name -> (n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes), the kuangliu table
CFG = {
    "a3": (64, 96, 128, 16, 32, 32),
    "b3": (128, 128, 192, 32, 96, 64),
    "a4": (192, 96, 208, 16, 48, 64),
    "b4": (160, 112, 224, 24, 64, 64),
    "c4": (128, 128, 256, 24, 64, 64),
    "d4": (112, 144, 288, 32, 64, 64),
    "e4": (256, 160, 320, 32, 128, 128),
    "a5": (256, 160, 320, 32, 128, 128),
    "b5": (384, 192, 384, 48, 128, 128),
}


def scale_width(channels: int, mult: float) -> int:
    """``channels * mult`` rounded to an integer, at least ``MIN_WIDTH``."""
    return max(MIN_WIDTH, int(round(channels * mult)))


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int) -> list:
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True),
    ]


class Inception(nn.Module):
    def __init__(self, in_planes: int, n1x1: int, n3x3red: int, n3x3: int,
                 n5x5red: int, n5x5: int, pool_planes: int):
        super().__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(*_conv_bn_relu(in_planes, n1x1, 1))
        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            *_conv_bn_relu(in_planes, n3x3red, 1),
            *_conv_bn_relu(n3x3red, n3x3, 3),
        )
        # 1x1 conv -> 3x3 conv -> 3x3 conv branch (replaces the 5x5 conv)
        self.b3 = nn.Sequential(
            *_conv_bn_relu(in_planes, n5x5red, 1),
            *_conv_bn_relu(n5x5red, n5x5, 3),
            *_conv_bn_relu(n5x5, n5x5, 3),
        )
        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            *_conv_bn_relu(in_planes, pool_planes, 1),
        )
        self.out_planes = n1x1 + n3x3 + n5x5 + pool_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


class GoogLeNet(nn.Module):
    """``googlenet_cifar.GoogLeNet`` with every width times ``width_mult``."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        self.width_mult = float(width_mult)
        planes = scale_width(STEM, width_mult)
        self.pre_layers = nn.Sequential(*_conv_bn_relu(3, planes, 3))

        def block(name: str, in_planes: int) -> Inception:
            return Inception(in_planes, *[scale_width(c, width_mult) for c in CFG[name]])

        self.a3 = block("a3", planes)
        self.b3 = block("b3", self.a3.out_planes)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = block("a4", self.b3.out_planes)
        self.b4 = block("b4", self.a4.out_planes)
        self.c4 = block("c4", self.b4.out_planes)
        self.d4 = block("d4", self.c4.out_planes)
        self.e4 = block("e4", self.d4.out_planes)

        self.a5 = block("a5", self.e4.out_planes)
        self.b5 = block("b5", self.a5.out_planes)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.b5.out_planes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


def googlenet_x025(num_classes, large_input):
    return GoogLeNet(num_classes=num_classes, width_mult=0.25)


def googlenet_x050(num_classes, large_input):
    return GoogLeNet(num_classes=num_classes, width_mult=0.5)


def googlenet_x075(num_classes, large_input):
    return GoogLeNet(num_classes=num_classes, width_mult=0.75)


def googlenet_x100(num_classes, large_input):
    """The same network as ``googlenet_cifar.googlenet`` (kept there; listed here for one-file use)."""
    return GoogLeNet(num_classes=num_classes, width_mult=1.0)
