"""GoogLeNet for 32x32 CIFAR inputs (Inception v1 with batch norm, 5x5 branch as two 3x3).

Source: the widely used kuangliu/pytorch-cifar implementation
(https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py), which follows
Szegedy et al., "Going Deeper with Convolutions" (CVPR 2015). A 3x3 pre-layer replaces the
ImageNet stem; the 5x5 branch is a 1x1 -> 3x3 -> 3x3 stack; every conv is followed by
BatchNorm2d and ReLU. About 6.2 M parameters for 10 classes.

Plain nn.Sequential / nn.Module code with ``torch.cat`` in ``forward`` so that
``torch.fx.symbolic_trace`` (SPECTRA's channel-group discovery) traces it.
"""
import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.pre_layers = nn.Sequential(*_conv_bn_relu(3, 192, 3))

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(1024, num_classes)

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


def googlenet(num_classes, large_input):
    return GoogLeNet(num_classes=num_classes)
