import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def make_layer(in_planes, out_planes, blocks, stride, bn):
    layers = []
    for i in range(blocks):
        s = stride if i == 0 else 1
        conv = conv3x3(in_planes, out_planes, stride=s)
        block = [conv]
        if bn:
            block.append(nn.BatchNorm2d(out_planes))
        block.append(nn.ReLU(inplace=True))
        layers.append(nn.Sequential(*block))
    return nn.Sequential(*layers)


class ResNet_LG(nn.Module):
    def __init__(self, in_planes, block_structure, bn, last_layer):
        super().__init__()
        self.in_planes = in_planes
        self.last_layer = last_layer

        self.conv1 = conv3x3(3, in_planes)

        self.layers = nn.Sequential(
            make_layer(in_planes, in_planes, block_structure[0], stride=1, bn=bn),
            make_layer(in_planes, in_planes, block_structure[1], stride=2, bn=bn),
            make_layer(in_planes, in_planes, block_structure[2], stride=2, bn=bn)
        )

        if last_layer == "dense":
            self.classifier = nn.Linear(in_planes, 10)
        elif last_layer == "avg":
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(in_planes, 10)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)

        if self.last_layer == "dense":
            x = torch.flatten(x, 1)
        elif self.last_layer == "avg":
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet2b(num_classes=10, large_input=False):
    return ResNet_LG(in_planes=8, block_structure=[2, 0, 0], bn=False, last_layer="dense")

def resnet4b(num_classes=10, large_input=False):
    return ResNet_LG(in_planes=16, block_structure=[2, 2, 0], bn=False, last_layer="dense")

def resnet8px(num_classes=10, large_input=False):
    return ResNet_LG(in_planes=8, block_structure=[2, 2, 0], bn=True, last_layer="avg")

def resnetbase(num_classes=10, large_input=False):
    return ResNet_LG(in_planes=16, block_structure=[2, 2, 0], bn=True, last_layer="avg")

def resnetdeep(num_classes=10, large_input=False):
    return ResNet_LG(in_planes=8, block_structure=[2, 2, 2], bn=True, last_layer="avg")
