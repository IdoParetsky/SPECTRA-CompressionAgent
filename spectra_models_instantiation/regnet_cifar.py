"""RegNetX-200MF / X-400MF / Y-400MF for 32x32 CIFAR inputs: a grouped-convolution control for the sealed set.

Bottleneck blocks (bottleneck ratio 1) whose 3x3 conv is a grouped convolution with a fixed
group width (8 for 200MF, 16 for 400MF), four stages; RegNetY adds a squeeze-and-excitation
gate after the grouped conv. For 10 classes: X-200MF about 2.3 M parameters, X-400MF about
4.8 M, Y-400MF about 5.7 M (the kuangliu networks).

Source: kuangliu/pytorch-cifar, ``models/regnet.py``
(https://github.com/kuangliu/pytorch-cifar, MIT licence, Copyright (c) 2017 liukuang), after
Radosavovic et al., "Designing Network Design Spaces" (CVPR 2020, arXiv:2003.13678). As in the
kuangliu file the stem is a 3x3 stride-1 conv with 64 channels and the stage strides are
(1, 1, 2, 2).

Changes from the kuangliu file, none of which changes the function: ReLU / Sigmoid modules
instead of the functional forms, an identity shortcut is ``None`` instead of an empty
``nn.Sequential``, and an ``nn.AdaptiveAvgPool2d`` module for the global pooling.

SPECTRA notes: this family is a CONTROL. ``src/channel_groups.py`` (as of tree_v36) marks the
input and the output channels of a grouped convolution that is not depthwise as unprunable
("grouped convolution consumer" / "grouped convolution producer"; open decision D6), so each
block's inner width stays whole and only the residual streams (and RegNetY's gate widths)
can be cut. The stem conv is the first registered layer and the classifier the last.
``large_input`` is accepted for the loader's contract and ignored: this is a 32-px definition.
"""
import torch
import torch.nn as nn

__all__ = ["RegNet", "regnetx_200mf", "regnetx_400mf", "regnety_400mf"]

CFGS = {
    "x_200mf": dict(depths=(1, 1, 4, 7), widths=(24, 56, 152, 368), strides=(1, 1, 2, 2),
                    group_width=8, bottleneck_ratio=1, se_ratio=0),
    "x_400mf": dict(depths=(1, 2, 7, 12), widths=(32, 64, 160, 384), strides=(1, 1, 2, 2),
                    group_width=16, bottleneck_ratio=1, se_ratio=0),
    "y_400mf": dict(depths=(1, 2, 7, 12), widths=(32, 64, 160, 384), strides=(1, 1, 2, 2),
                    group_width=16, bottleneck_ratio=1, se_ratio=0.25),
}


class SE(nn.Module):
    """Squeeze-and-excitation block."""

    def __init__(self, in_planes: int, se_planes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se1 = nn.Conv2d(in_planes, se_planes, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.se2 = nn.Conv2d(se_planes, in_planes, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.se1(self.pool(x)))
        out = self.gate(self.se2(out))
        return x * out


class Block(nn.Module):
    def __init__(self, w_in: int, w_out: int, stride: int, group_width: int, bottleneck_ratio: float,
                 se_ratio: float):
        super().__init__()
        # 1x1
        w_b = int(round(w_out * bottleneck_ratio))
        self.conv1 = nn.Conv2d(w_in, w_b, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w_b)
        self.relu1 = nn.ReLU(inplace=True)
        # 3x3
        num_groups = w_b // group_width
        self.conv2 = nn.Conv2d(w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_groups, bias=False)
        self.bn2 = nn.BatchNorm2d(w_b)
        self.relu2 = nn.ReLU(inplace=True)
        # se
        self.with_se = se_ratio > 0
        if self.with_se:
            w_se = int(round(w_in * se_ratio))
            self.se = SE(w_b, w_se)
        # 1x1
        self.conv3 = nn.Conv2d(w_b, w_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(w_out)

        self.shortcut = None
        if stride != 1 or w_in != w_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(w_out),
            )
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        if self.with_se:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out += x if self.shortcut is None else self.shortcut(x)
        return self.relu3(out)


class RegNet(nn.Module):
    def __init__(self, cfg: dict, num_classes: int = 10):
        super().__init__()
        self.cfg = cfg
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(0)
        self.layer2 = self._make_layer(1)
        self.layer3 = self._make_layer(2)
        self.layer4 = self._make_layer(3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.cfg["widths"][-1], num_classes)
        self._initialize_weights()

    def _make_layer(self, idx: int) -> nn.Sequential:
        depth = self.cfg["depths"][idx]
        width = self.cfg["widths"][idx]
        stride = self.cfg["strides"][idx]
        group_width = self.cfg["group_width"]
        bottleneck_ratio = self.cfg["bottleneck_ratio"]
        se_ratio = self.cfg["se_ratio"]

        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            layers.append(Block(self.in_planes, width, s, group_width, bottleneck_ratio, se_ratio))
            self.in_planes = width
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


def regnetx_200mf(num_classes, large_input):
    return RegNet(CFGS["x_200mf"], num_classes=num_classes)


def regnetx_400mf(num_classes, large_input):
    return RegNet(CFGS["x_400mf"], num_classes=num_classes)


def regnety_400mf(num_classes, large_input):
    return RegNet(CFGS["y_400mf"], num_classes=num_classes)
