"""GhostNet for 32x32 CIFAR inputs, with the width multiplier (x0.5, x0.75, x1.0, x1.3).

A Ghost module makes half of its output channels with an ordinary ("primary") conv and the
other half by a cheap 3x3 depthwise conv applied to the primary half, then concatenates the
two. Ghost bottlenecks stack two Ghost modules, with a stride-2 depthwise conv and an optional
squeeze-and-excitation gate between them and a depthwise + 1x1 shortcut when shapes change.
About 3.9 M parameters at width 1.0 for 10 classes.

Sources: Han et al., "GhostNet: More Features from Cheap Operations" (CVPR 2020,
arXiv:1911.11907), and the authors' implementation (huawei-noah/Efficient-AI-Backbones,
``ghostnet_pytorch/ghostnet.py``, https://github.com/huawei-noah/Efficient-AI-Backbones,
Apache License 2.0, Copyright (c) 2020 Huawei Technologies Co., Ltd.), whose modules, stage
table, channel rounding (``_make_divisible`` to 4) and head are copied here. With the
1000-class head the x1.0 network has the published 5.2 M parameters
(``tests/test_sealed_families.py`` checks it).

Changes for CIFAR (the adaptation ``mobilenetv2_chenyaofo.py`` makes for MobileNetV2): the stem
conv has stride 1 and stage 2 runs at stride 1, so the total stride is 8 and the last feature
map is 4x4.

Changes that keep the function: the gate is ``nn.Hardsigmoid`` (the original's
``relu6(x + 3) / 6``), an absent activation or an identity shortcut is left out instead of
being an empty ``nn.Sequential``, dropout is an ``nn.Dropout`` module, and a Ghost module
slices its output to ``oup`` channels only when the two halves exceed it (never with ratio 2
and the even widths ``_make_divisible`` gives; the original always slices).

SPECTRA notes: the stem conv is the first registered layer and the classifier the last, in
forward order. A Ghost module's two halves are one coupled set of channels (ghost channel i
is made from primary channel i), which every later layer therefore reads twice;
``src/channel_groups.py`` (as of tree_v36) refuses such a group ("overlapping channel reads by
the same layer"), and a stride-2 bottleneck's depthwise conv over the concatenation is not a
recorded reader at all. See ``scripts/sealed_structure_check.py`` for what can be cut.
``large_input`` is accepted for the loader's contract and ignored: this is a 32-px definition.
"""
import math

import torch
import torch.nn as nn

__all__ = ["GhostNet", "ghostnet_x050", "ghostnet_x075", "ghostnet_x100", "ghostnet_x130"]

# One list per stage of blocks: kernel, expansion size, output channels, SE ratio, stride (ImageNet strides)
CFGS = (
    # stage1
    ((3, 16, 16, 0, 1),),
    # stage2 (stride 1 at CIFAR size)
    ((3, 48, 24, 0, 2),),
    ((3, 72, 24, 0, 1),),
    # stage3
    ((5, 72, 40, 0.25, 2),),
    ((5, 120, 40, 0.25, 1),),
    # stage4
    ((3, 240, 80, 0, 2),),
    ((3, 200, 80, 0, 1),
     (3, 184, 80, 0, 1),
     (3, 184, 80, 0, 1),
     (3, 480, 112, 0.25, 1),
     (3, 672, 112, 0.25, 1)),
    # stage5
    ((5, 672, 160, 0.25, 2),),
    ((5, 960, 160, 0, 1),
     (5, 960, 160, 0.25, 1),
     (5, 960, 160, 0, 1),
     (5, 960, 160, 0.25, 1)),
)
CIFAR_STRIDE_ONE = (1,)  # indices into CFGS whose stride-2 block runs at stride 1 at CIFAR size (stage 2)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs: int, se_ratio: float = 0.25, divisor: int = 4):
        super().__init__()
        reduced_chs = _make_divisible(in_chs * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvBnAct(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act1(self.bn1(self.conv(x)))


class GhostModule(nn.Module):
    def __init__(self, inp: int, oup: int, kernel_size: int = 1, ratio: int = 2, dw_size: int = 3,
                 stride: int = 1, relu: bool = True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.trim = init_channels + new_channels != oup

        primary = [nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
                   nn.BatchNorm2d(init_channels)]
        cheap = [nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
                 nn.BatchNorm2d(new_channels)]
        if relu:
            primary.append(nn.ReLU(inplace=True))
            cheap.append(nn.ReLU(inplace=True))
        self.primary_conv = nn.Sequential(*primary)
        self.cheap_operation = nn.Sequential(*cheap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        if self.trim:
            out = out[:, :self.oup, :, :]
        return out


class GhostBottleneck(nn.Module):
    """Ghost bottleneck with optional SE."""

    def __init__(self, in_chs: int, mid_chs: int, out_chs: int, dw_kernel_size: int = 3,
                 stride: int = 1, se_ratio: float = 0.0):
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio) if has_se else None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = None
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += residual if self.shortcut is None else self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0, dropout: float = 0.2,
                 cifar_strides: bool = True):
        super().__init__()
        self.width_mult = float(width_mult)

        # building first layer (stride 1 at CIFAR size)
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 1 if cifar_strides else 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        exp_size = 16
        for index, cfg in enumerate(CFGS):
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                if cifar_strides and index in CIFAR_STRIDE_ONE:
                    s = 1
                output_channel = _make_divisible(c * width_mult, 4)
                hidden_channel = _make_divisible(exp_size * width_mult, 4)
                layers.append(GhostBottleneck(input_channel, hidden_channel, output_channel, k, s, se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width_mult, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_channel, num_classes)
        self._initialize_weights()

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
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def ghostnet_x050(num_classes, large_input):
    return GhostNet(num_classes=num_classes, width_mult=0.5)


def ghostnet_x075(num_classes, large_input):
    return GhostNet(num_classes=num_classes, width_mult=0.75)


def ghostnet_x100(num_classes, large_input):
    return GhostNet(num_classes=num_classes, width_mult=1.0)


def ghostnet_x130(num_classes, large_input):
    return GhostNet(num_classes=num_classes, width_mult=1.3)
