"""CIFAR-stem ResNet-18 (3x3 conv1, no 7x7 / max-pool).

pruning-bench ``resnet18_cifar100`` checkpoints use a 64-channel 3x3 stem
``conv1.weight`` of shape ``(64, 3, 3, 3)``, which does not load into
torchvision's ImageNet ResNet-18.
"""
from torchvision.models.resnet import BasicBlock, ResNet
from torch import nn


def resnet18_cifar(num_classes=100, large_input=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if not large_input:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model
