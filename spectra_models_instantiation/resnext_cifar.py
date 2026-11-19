"""ResNeXt-29 for 32x32 CIFAR inputs (small sizes): a grouped-convolution control for the sealed set.

Bottleneck blocks whose 3x3 conv is a grouped convolution (``cardinality`` groups of
``bottleneck_width`` channels), three stages of three blocks, the bottleneck width doubling
per stage. For 10 classes: 32x4d has 4.77 M parameters and 2x64d 9.13 M (the kuangliu
networks); 16x4d and 8x4d are narrower nets of the same shape (ours, not published sizes).

Source: kuangliu/pytorch-cifar, ``models/resnext.py``
(https://github.com/kuangliu/pytorch-cifar, MIT licence, Copyright (c) 2017 liukuang), after
Xie et al., "Aggregated Residual Transformations for Deep Neural Networks" (CVPR 2017,
arXiv:1611.05431). As in the kuangliu file the stem is a 1x1 conv.

Changes from the kuangliu file, none of which changes the function on 32-px inputs: ReLU
modules instead of ``F.relu``, an identity shortcut is ``None`` instead of an empty
``nn.Sequential``, and adaptive global average pooling instead of ``F.avg_pool2d(out, 8)``
(the final map is 8x8).

SPECTRA notes: this family is a CONTROL. ``src/channel_groups.py`` (as of tree_v36) marks the
input and the output channels of a grouped convolution that is not depthwise as unprunable
("grouped convolution consumer" / "grouped convolution producer"; open decision D6), so each
block's two inner widths stay whole and only the residual streams can be cut. The stem conv
is the first registered layer and the classifier the last. ``large_input`` is accepted for
the loader's contract and ignored: this is a 32-px definition.
"""
import torch
import torch.nn as nn

__all__ = ["ResNeXt", "resnext29_8x4d", "resnext29_16x4d", "resnext29_32x4d", "resnext29_2x64d"]


class Block(nn.Module):
    """Grouped convolution block."""
    expansion = 2

    def __init__(self, in_planes: int, cardinality: int = 32, bottleneck_width: int = 4, stride: int = 1):
        super().__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1,
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * group_width),
            )
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x if self.shortcut is None else self.shortcut(x)
        return self.relu3(out)


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality: int, bottleneck_width: int, num_classes: int = 10):
        super().__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # _make_layer has doubled bottleneck_width three times: the last stage is cardinality * width * 4 * 2 wide
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)
        self._initialize_weights()

    def _make_layer(self, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


def resnext29_8x4d(num_classes, large_input):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=8, bottleneck_width=4, num_classes=num_classes)


def resnext29_16x4d(num_classes, large_input):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=16, bottleneck_width=4, num_classes=num_classes)


def resnext29_32x4d(num_classes, large_input):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=32, bottleneck_width=4, num_classes=num_classes)


def resnext29_2x64d(num_classes, large_input):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=2, bottleneck_width=64, num_classes=num_classes)
