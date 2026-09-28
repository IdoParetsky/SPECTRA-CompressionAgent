"""
VGG-BN in the module layout of Torch-Pruning's CIFAR benchmark models
(``reproduce/engine/models/cifar/vgg.py``, from RepDistiller), so the checkpoints DepGraph
(Fang et al., CVPR 2023) released — ``vgg19_cifar100_dep_graph_73.5.pth`` — load with strict keys.

Layout: five ``blockN`` Sequentials of Conv(bias)-BN-ReLU with each block's last ReLU applied in
``forward``; max-pool after blocks 0-2; ``pool3`` only on 64-pixel inputs (the reference branches
on ``x.shape[2] == 64`` at run time — here ``large_input`` decides it at construction, so the
graph is static and fx-traceable); global average pool; one ``Linear(512, num_classes)``.
The ``vgg_chenyaofo`` twins use a different layout (``features`` + a three-layer classifier).
"""
import math

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VGGDepGraph", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]

CFG = {
    "A": [[64], [128], [256, 256], [512, 512], [512, 512]],
    "B": [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    "D": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    "E": [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
}


class VGGDepGraph(nn.Module):
    def __init__(self, cfg, num_classes=100, large_input=False):
        super().__init__()
        self.block0 = self._make_layers(cfg[0], 3)
        self.block1 = self._make_layers(cfg[1], cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], cfg[3][-1])
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_after_block3 = bool(large_input)
        self.classifier = nn.Linear(cfg[4][-1], num_classes)
        self._initialize_weights()

    @staticmethod
    def _make_layers(cfg, in_channels):
        layers = []
        for v in cfg:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                       nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers[:-1])

    def forward(self, x):
        x = self.pool0(F.relu(self.block0(x)))
        x = self.pool1(F.relu(self.block1(x)))
        x = self.pool2(F.relu(self.block2(x)))
        x = F.relu(self.block3(x))
        if self.pool_after_block3:
            x = self.pool3(x)
        x = F.relu(self.block4(x))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg11_bn(num_classes=100, large_input=False):
    return VGGDepGraph(CFG["A"], num_classes=num_classes, large_input=large_input)


def vgg13_bn(num_classes=100, large_input=False):
    return VGGDepGraph(CFG["B"], num_classes=num_classes, large_input=large_input)


def vgg16_bn(num_classes=100, large_input=False):
    return VGGDepGraph(CFG["D"], num_classes=num_classes, large_input=large_input)


def vgg19_bn(num_classes=100, large_input=False):
    return VGGDepGraph(CFG["E"], num_classes=num_classes, large_input=large_input)
