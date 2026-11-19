"""MobileNetV3 small and large for 32x32 CIFAR inputs, with the width multiplier.

Inverted residual blocks (1x1 expansion, 3x3 / 5x5 depthwise conv, optional
squeeze-and-excitation with a hard-sigmoid gate, 1x1 linear projection), ReLU or hard-swish
activations, a 1x1 "last conv" to six times the last block's width and a two-layer Linear
classifier. For 10 classes at width 1.0: small about 1.5 M parameters, large about 4.2 M.

Sources: Howard et al., "Searching for MobileNetV3" (ICCV 2019, arXiv:1905.02244), and its
torchvision implementation (``torchvision/models/mobilenetv3.py``,
https://github.com/pytorch/vision, BSD 3-Clause licence), whose two block tables, channel
rounding (``_make_divisible`` to 8), squeeze width (``_make_divisible(expanded // 4, 8)``),
BatchNorm settings (eps 0.001, momentum 0.01) and initialisation are copied here. With the
1000-class head the parameter counts are torchvision's (large 5,483,032, small 2,542,856;
``tests/test_sealed_families.py`` checks them).

Changes for CIFAR (the adaptation ``mobilenetv2_chenyaofo.py`` makes for MobileNetV2): the stem
conv has stride 1 and the first stride-2 block runs at stride 1 (small: block 1, large:
block 2), so the total stride is 8 and the last feature map is 4x4. Small's block 1 (16 to 16
channels) then has matching shapes and takes the residual connection the block rule gives; it
has none at stride 2.

SPECTRA notes: the stem conv is the first registered layer and the classifier's last Linear
the last, in forward order. The gate is ``nn.Hardsigmoid`` (a functional ``relu6(x + 3) / 6``
would hide the gate from SPECTRA's channel groups) and multiplies the feature map as
``scale * x``, which ties the gate's second conv to the expanded feature map.
``large_input`` is accepted for the loader's contract and ignored: this is a 32-px definition.
"""
from typing import List, Optional

import torch
import torch.nn as nn

__all__ = ["MobileNetV3", "mobilenetv3_small_x075", "mobilenetv3_small_x100",
           "mobilenetv3_large_x075", "mobilenetv3_large_x100"]

# (input channels, kernel, expanded channels, out channels, use SE, activation, stride) at width 1.0, ImageNet strides
LARGE = (
    (16, 3, 16, 16, False, "RE", 1),
    (16, 3, 64, 24, False, "RE", 2),   # C1; stride 1 at CIFAR size
    (24, 3, 72, 24, False, "RE", 1),
    (24, 5, 72, 40, True, "RE", 2),    # C2
    (40, 5, 120, 40, True, "RE", 1),
    (40, 5, 120, 40, True, "RE", 1),
    (40, 3, 240, 80, False, "HS", 2),  # C3
    (80, 3, 200, 80, False, "HS", 1),
    (80, 3, 184, 80, False, "HS", 1),
    (80, 3, 184, 80, False, "HS", 1),
    (80, 3, 480, 112, True, "HS", 1),
    (112, 3, 672, 112, True, "HS", 1),
    (112, 5, 672, 160, True, "HS", 2),  # C4
    (160, 5, 960, 160, True, "HS", 1),
    (160, 5, 960, 160, True, "HS", 1),
)
SMALL = (
    (16, 3, 16, 16, True, "RE", 2),    # C1; stride 1 at CIFAR size
    (16, 3, 72, 24, False, "RE", 2),   # C2
    (24, 3, 88, 24, False, "RE", 1),
    (24, 5, 96, 40, True, "HS", 2),    # C3
    (40, 5, 240, 40, True, "HS", 1),
    (40, 5, 240, 40, True, "HS", 1),
    (40, 5, 120, 48, True, "HS", 1),
    (48, 5, 144, 48, True, "HS", 1),
    (48, 5, 288, 96, True, "HS", 2),   # C4
    (96, 5, 576, 96, True, "HS", 1),
    (96, 5, 576, 96, True, "HS", 1),
)
ARCHS = {"large": (LARGE, 1280), "small": (SMALL, 1024)}
BN_EPS, BN_MOMENTUM = 0.001, 0.01


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv_bn_act(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1,
                 activation=None) -> nn.Sequential:
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, (kernel_size - 1) // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch, eps=BN_EPS, momentum=BN_MOMENTUM),
    ]
    if activation is not None:
        layers.append(activation(inplace=True))
    return nn.Sequential(*layers)


class SqueezeExcitation(nn.Module):
    """torchvision's ``SqueezeExcitation`` with a ReLU squeeze and a hard-sigmoid gate."""

    def __init__(self, input_channels: int, squeeze_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.activation = nn.ReLU()
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.scale_activation = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.activation(self.fc1(scale))
        scale = self.scale_activation(self.fc2(scale))
        return scale * x


class InvertedResidual(nn.Module):
    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int,
                 use_se: bool, use_hs: bool, stride: int):
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")
        self.use_res_connect = stride == 1 and input_channels == out_channels
        activation = nn.Hardswish if use_hs else nn.ReLU
        layers: List[nn.Module] = []
        # expand
        if expanded_channels != input_channels:
            layers.append(_conv_bn_act(input_channels, expanded_channels, kernel_size=1, activation=activation))
        # depthwise
        layers.append(_conv_bn_act(expanded_channels, expanded_channels, kernel_size=kernel, stride=stride,
                                   groups=expanded_channels, activation=activation))
        if use_se:
            layers.append(SqueezeExcitation(expanded_channels, _make_divisible(expanded_channels // 4, 8)))
        # project
        layers.append(_conv_bn_act(expanded_channels, out_channels, kernel_size=1, activation=None))
        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(self, arch: str = "small", num_classes: int = 10, width_mult: float = 1.0, dropout: float = 0.2,
                 cifar_strides: bool = True):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"Unsupported MobileNetV3 arch {arch}: small or large expected")
        self.arch, self.width_mult = arch, float(width_mult)
        table, last = ARCHS[arch]

        def adjust(channels: int) -> int:
            return _make_divisible(channels * width_mult, 8)

        layers: List[nn.Module] = []
        # building first layer (stride 1 at CIFAR size)
        first = adjust(table[0][0])
        layers.append(_conv_bn_act(3, first, kernel_size=3, stride=1 if cifar_strides else 2, activation=nn.Hardswish))
        # building inverted residual blocks; the first stride-2 block runs at stride 1 at CIFAR size
        relaxed = not cifar_strides
        for in_ch, kernel, exp_ch, out_ch, use_se, act, stride in table:
            if stride == 2 and not relaxed:
                stride, relaxed = 1, True
            layers.append(InvertedResidual(adjust(in_ch), kernel, adjust(exp_ch), adjust(out_ch), use_se,
                                           act == "HS", stride))
        # building last several layers
        lastconv_input = adjust(table[-1][3])
        lastconv_output = 6 * lastconv_input
        layers.append(_conv_bn_act(lastconv_input, lastconv_output, kernel_size=1, activation=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        last_channel = adjust(last)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenetv3_small_x075(num_classes, large_input):
    return MobileNetV3("small", num_classes=num_classes, width_mult=0.75)


def mobilenetv3_small_x100(num_classes, large_input):
    return MobileNetV3("small", num_classes=num_classes, width_mult=1.0)


def mobilenetv3_large_x075(num_classes, large_input):
    return MobileNetV3("large", num_classes=num_classes, width_mult=0.75)


def mobilenetv3_large_x100(num_classes, large_input):
    return MobileNetV3("large", num_classes=num_classes, width_mult=1.0)
