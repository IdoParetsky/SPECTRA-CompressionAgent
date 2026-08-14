"""CIFAR ResNets matching akamaster/pytorch_resnet_cifar10 checkpoints.

The published ``.th`` files were trained with paper option A: identity shortcuts
inside a stage, and a parameter-free channel-pad downsample between stages.
SPECTRA previously instantiated option B (1x1 conv + BN shortcuts). Those extra
keys are absent from the checkpoints, so all six akamaster files failed to load.

``large_input`` is accepted for catalog compatibility; the CIFAR stem is always
3x3 stride-1 (these weights were never trained with an ImageNet stem).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PadShortcut(nn.Module):
    """Option-A downsample, expressed with modules the structured pruner can resize.

    akamaster's paper shortcut is ``pad(x[:, :, ::2, ::2], planes//4)`` — no learned
    weights, so the ``.th`` files do not store shortcut keys. ``MaxPool2d(1, 2)`` is
    the same spatial subsample; a frozen 1x1 identity in the middle of the output
    channels is the same zero-pad. After load, missing ``shortcut.channel.weight``
    is expected and left at this identity.
    """

    def __init__(self, in_planes: int, planes: int):
        super().__init__()
        self.spatial = nn.MaxPool2d(kernel_size=1, stride=2)
        self.channel = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        with torch.no_grad():
            self.channel.weight.zero_()
            start = planes // 4
            for i in range(in_planes):
                self.channel.weight[start + i, i, 0, 0] = 1.0
        self.channel.weight.requires_grad_(False)

    def forward(self, x):
        return self.channel(self.spatial(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = PadShortcut(in_planes, planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


def resnet20(num_classes=10, large_input=False):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes=10, large_input=False):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes=10, large_input=False):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes=10, large_input=False):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes=10, large_input=False):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)


def resnet1202(num_classes=10, large_input=False):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)
