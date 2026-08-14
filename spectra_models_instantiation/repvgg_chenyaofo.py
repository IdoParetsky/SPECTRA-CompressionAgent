import torch
import torch.nn as nn
from typing import List


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                        padding, groups=groups, bias=False))
    result.add_module("bn", nn.BatchNorm2d(out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, groups=groups, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups)
            self.rbr_1x1 = conv_bn(in_channels, out_channels, kernel_size=1, stride=stride,
                                   padding=0, groups=groups)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(x))
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)
        return self.nonlinearity(out)


class RepVGG(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, deploy=False):
        super().__init__()
        assert len(width_multiplier) == 4

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(3, self.in_planes, stride=1, padding=1, deploy=deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=1, deploy=deploy)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2, deploy=deploy)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2, deploy=deploy)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2, deploy=deploy)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

        self._initialize_weights()

    def _make_stage(self, planes, num_blocks, stride, deploy):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(RepVGGBlock(self.in_planes, planes, stride=s, padding=1, deploy=deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.linear(x)


def repvgg_a0(num_classes, large_input, deploy=False):
    num_blocks, width_mult = ([2, 4, 14, 1], [0.75, 0.75, 0.75, 2.5])
    return RepVGG(num_blocks=num_blocks, width_multiplier=width_mult,
                  num_classes=num_classes, deploy=deploy)


def repvgg_a1(num_classes, large_input, deploy=False):
    num_blocks, width_mult = ([2, 4, 14, 1], [1.0, 1.0, 1.0, 2.5])
    return RepVGG(num_blocks=num_blocks, width_multiplier=width_mult,
                  num_classes=num_classes, deploy=deploy)


def repvgg_a2(num_classes, large_input, deploy=False):
    num_blocks, width_mult = ([2, 4, 14, 1], [1.5, 1.5, 1.5, 2.75])
    return RepVGG(num_blocks=num_blocks, width_multiplier=width_mult,
                  num_classes=num_classes, deploy=deploy)


# Filename stems in spectra_pretrained_networks are ``repvgga0`` (no underscore).
repvgga0 = repvgg_a0
repvgga1 = repvgg_a1
repvgga2 = repvgg_a2
