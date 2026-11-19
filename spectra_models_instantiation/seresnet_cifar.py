"""SE-ResNet-20 / 32 / 56 / 110 for 32x32 CIFAR inputs.

The CIFAR ResNet of He et al. (three stages of basic blocks at 16, 32 and 64 channels) with a
squeeze-and-excitation gate on every residual branch: global average pooling, a bias-free
Linear to ``channels // 16``, ReLU, a bias-free Linear back, a sigmoid, and the branch is
multiplied by the result before the shortcut is added. About 0.27 M (20) to 1.7 M (110)
parameters. The family is close to the plain CIFAR ResNets SPECTRA's agents have seen: the
gate is the only new element.

Sources: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018, arXiv:1709.01507), whose
CIFAR experiments include SE-ResNet-110, and moskomule/senet.pytorch (``senet/se_resnet.py``
and ``senet/se_module.py``, https://github.com/moskomule/senet.pytorch, MIT licence, Copyright
(c) 2018 Ryuichiro Hataya), whose ``CifarSEResNet`` / ``CifarSEBasicBlock`` / ``SELayer``
(reduction 16, 1x1 conv + BN shortcut where the width changes) are followed for depths 20, 32
and 56; depth 110 is the same network with 18 blocks per stage.

Changes that keep the function: the gate is applied by broadcasting (``x * y`` with ``y`` of
shape (N, C, 1, 1)) instead of ``x * y.expand_as(x)``, each ReLU call has its own module, and
an identity shortcut is ``None`` instead of a lambda.

SPECTRA notes: the stem conv is the first registered layer and the classifier the last. The
gate's Linear layers are rows of their own; ``x * y`` ties the second Linear to the residual
stream, so SPECTRA's channel groups cut them together (``expand_as`` is an operation
``src/channel_groups.py`` does not know: it would block the stream and the gate). The stage-1
gate is one unit wide (16 // 16) and cannot be cut. ``large_input`` is accepted for the
loader's contract and ignored: this is a 32-px definition.
"""
import torch
import torch.nn as nn

__all__ = ["CifarSEResNet", "se_resnet20", "se_resnet32", "se_resnet56", "se_resnet110"]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.flatten(self.avg_pool(x), 1)
        y = self.fc(y)
        return x * y.view(y.size(0), y.size(1), 1, 1)


class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
        else:
            self.downsample = None
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = out + residual
        return self.relu2(out)


class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size: int, num_classes: int = 10, reduction: int = 16):
        super().__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int, reduction: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def se_resnet20(num_classes, large_input):
    return CifarSEResNet(CifarSEBasicBlock, 3, num_classes=num_classes)


def se_resnet32(num_classes, large_input):
    return CifarSEResNet(CifarSEBasicBlock, 5, num_classes=num_classes)


def se_resnet56(num_classes, large_input):
    return CifarSEResNet(CifarSEBasicBlock, 9, num_classes=num_classes)


def se_resnet110(num_classes, large_input):
    return CifarSEResNet(CifarSEBasicBlock, 18, num_classes=num_classes)
