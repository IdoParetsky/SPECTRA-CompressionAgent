"""Wave-3 (Tiny-ImageNet, 64x64, 200 classes) factories: ResNet-18, VGG-16-BN and MobileNetV2 x1.0.

Signature as every SPECTRA catalog factory: ``fn(num_classes, large_input)``; ``large_input`` is accepted and
ignored, the three nets are 64-px by construction. The checkpoints are trained by
``scripts/train_pretrained_checkpoint.py --dataset tiny-imagenet --script <this file>`` and referenced from an
input JSON as ``[arch, <this file>, "tiny-imagenet"]``; ``utils.resolve_instantiation_func`` also accepts the
hyphenated spellings a checkpoint filename carries (``vgg16-bn``, ``mobilenet-v2``).

* ``resnet18``: torchvision's ResNet-18 body with the CIFAR stem (3x3 conv, stride 1, no max-pool), the usual
  choice at 64 px: stages at 64 / 32 / 16 / 8 px, a 512 x 8 x 8 map into the global average pool (about 2.2 GMACs).
* ``vgg16_bn``: the CIFAR VGG-16-BN features (13 convs, five 2x2 max-pools, so 64 px -> 2 x 2 x 512) with the
  classifier's first Linear adapted to that map (2048 -> 512 -> 512 -> classes), the rest of the head as the
  chenyaofo CIFAR VGGs in the pool (about 1.3 GMACs).
* ``mobilenet_v2``: the chenyaofo CIFAR MobileNetV2 (width 1.0) with the ImageNet stage strides but a stride-1
  stem (stage strides 1, 2, 2, 2, 1, 2, 1): 64 px -> 4 x 4 before the 1280-wide head, the same final geometry as
  the pool's CIFAR MobileNetV2 at 32 px (about 0.2 GMACs).

Everything here is self-contained (an instantiation script is loaded from its path, not as a package).
"""
from __future__ import annotations

from typing import List, Optional, Union, cast

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

INPUT_SIZE = 64


# ------------------------------------------------------------------ ResNet-18, CIFAR stem

def resnet18(num_classes=200, large_input=False):
    """ResNet-18 with a 3x3 stride-1 stem and no max-pool (64 px -> 8 x 8 x 512 before the pool)."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model


# ------------------------------------------------------------------ VGG-16-BN, CIFAR features, 2x2 head

class TinyVGG(nn.Module):
    """CIFAR VGG features over a 64-px input; the classifier reads the flattened 2 x 2 x 512 map."""

    def __init__(self, features: nn.Module, num_classes: int = 200, classifier_in: int = 512 * 2 * 2):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


VGG16_CFG: List[Union[str, int]] = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
                                    512, 512, 512, 'M']


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16_bn(num_classes=200, large_input=False):
    """VGG-16-BN (CIFAR features) at 64 px: five pools leave 2 x 2 x 512, read by Linear(2048, 512)."""
    pools = sum(1 for v in VGG16_CFG if v == 'M')
    side = INPUT_SIZE // (2 ** pools)
    return TinyVGG(make_layers(VGG16_CFG, batch_norm=True), num_classes=num_classes, classifier_in=512 * side * side)


# ------------------------------------------------------------------ MobileNetV2 x1.0, stride-1 stem

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None,
                 activation_layer=None, dilation=1):
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True),
        )
        self.out_channels = out_planes


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        norm_layer = norm_layer or nn.BatchNorm2d
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


# (expand ratio t, channels c, repeats n, stride s): the ImageNet schedule; the stem below is stride 1, so a
# 64-px input reaches the 1280-wide head at 4 x 4 (ImageNet's 224 reaches it at 7 x 7 with a stride-2 stem).
TINY_MBV2_SETTING = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class TinyMobileNetV2(nn.Module):
    def __init__(self, num_classes=200, width_mult=1.0, setting=None, stem_stride=1):
        super().__init__()
        block = InvertedResidual
        norm_layer = nn.BatchNorm2d
        input_channel = 32
        last_channel = 1280
        setting = TINY_MBV2_SETTING if setting is None else setting

        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        features = [ConvBNReLU(3, input_channel, stride=stem_stride, norm_layer=norm_layer)]

        for t, c, n, s in setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenet_v2(num_classes=200, large_input=False):
    """MobileNetV2 x1.0 for 64 px: stride-1 stem, ImageNet stage strides (64 px -> 4 x 4 before the head)."""
    return TinyMobileNetV2(num_classes=num_classes, width_mult=1.0)


def mobilenet_v2x1(num_classes=200, large_input=False):
    """Alias in the pool's spelling (``mobilenet_v2x1``)."""
    return mobilenet_v2(num_classes=num_classes, large_input=large_input)


FACTORIES = ("resnet18", "vgg16_bn", "mobilenet_v2")
