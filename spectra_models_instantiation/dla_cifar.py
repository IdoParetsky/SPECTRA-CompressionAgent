"""Deep Layer Aggregation for 32x32 CIFAR inputs: SimpleDLA and DLA (DLA-34 style), with a width multiplier.

Trees of residual BasicBlocks whose outputs are concatenated and fused by a 1x1 ``Root``
conv. ``SimpleDLA`` nests two-leaf trees; ``DLA`` is the deeper aggregation of the paper's
DLA-34 (a level-2 tree's root also reads the tree's first block and its level-1 subtree).
At width 1.0 for 10 classes: SimpleDLA 15.1 M parameters, DLA 16.3 M (the kuangliu networks).

Source: kuangliu/pytorch-cifar, ``models/dla_simple.py`` and ``models/dla.py``
(https://github.com/kuangliu/pytorch-cifar, MIT licence, Copyright (c) 2017 liukuang), after
Yu et al., "Deep Layer Aggregation" (CVPR 2018, arXiv:1707.06484). The width multiplier is ours
(every channel count times it, rounded, floor of 4).

Changes from the kuangliu files, none of which changes the function at width 1.0 on 32-px
inputs: modules are registered in forward order (kuangliu registers a tree's ``root`` before
its children; only the order of the state_dict keys differs), ReLU modules instead of
``F.relu``, an identity shortcut is ``None`` instead of an empty ``nn.Sequential``, and
adaptive global average pooling instead of ``F.avg_pool2d(out, 4)`` (the final map is 4x4).

SPECTRA notes: the stem conv is the first registered layer and the classifier the last.
A tree's root reads the same coupled channels twice: its second block adds its input (the
first block's output) back, so both operands of the root's ``torch.cat`` are one residual
stream. ``src/channel_groups.py`` (as of tree_v36) refuses such a group ("overlapping channel
reads by the same layer"), so those streams stay whole; see ``scripts/sealed_structure_check.py``.
``large_input`` is accepted for the loader's contract and ignored: this is a 32-px definition.
"""
import torch
import torch.nn as nn

__all__ = ["SimpleDLA", "DLA", "simpledla_x025", "simpledla_x050", "simpledla_x100",
           "dla_x025", "dla_x050", "dla_x100"]

BASE = (16, 16, 32)            # base, layer1, layer2
TREES = (64, 128, 256, 512)    # layer3 .. layer6
MIN_WIDTH = 4


def scale_width(channels: int, mult: float) -> int:
    """``channels * mult`` rounded to an integer, at least ``MIN_WIDTH``."""
    return max(MIN_WIDTH, int(round(channels * mult)))


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x if self.shortcut is None else self.shortcut(x)
        return self.relu2(out)


class Root(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xs) -> torch.Tensor:
        x = torch.cat(xs, 1)
        return self.relu(self.bn(self.conv(x)))


class SimpleTree(nn.Module):
    """``dla_simple.Tree``: two children (blocks at level 1, trees above) and a root over both."""

    def __init__(self, block, in_channels: int, out_channels: int, level: int = 1, stride: int = 1):
        super().__init__()
        if level == 1:
            self.left_tree = block(in_channels, out_channels, stride=stride)
            self.right_tree = block(out_channels, out_channels, stride=1)
        else:
            self.left_tree = SimpleTree(block, in_channels, out_channels, level=level - 1, stride=stride)
            self.right_tree = SimpleTree(block, out_channels, out_channels, level=level - 1, stride=1)
        self.root = Root(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.left_tree(x)
        out2 = self.right_tree(out1)
        return self.root([out1, out2])


class Tree(nn.Module):
    """``dla.Tree``: above level 1 the root also reads ``prev_root`` and every lower-level subtree."""

    def __init__(self, block, in_channels: int, out_channels: int, level: int = 1, stride: int = 1):
        super().__init__()
        self.level = level
        if level > 1:
            self.prev_root = block(in_channels, out_channels, stride=stride)
            for i in reversed(range(1, level)):
                setattr(self, "level_%d" % i, Tree(block, in_channels, out_channels, level=i, stride=stride))
            self.left_node = block(out_channels, out_channels, stride=1)
        else:
            self.left_node = block(in_channels, out_channels, stride=stride)
        self.right_node = block(out_channels, out_channels, stride=1)
        self.root = Root((2 if level == 1 else level + 2) * out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            x = getattr(self, "level_%d" % i)(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        return self.root(xs)


class _DLABase(nn.Module):
    tree = None  # the Tree class of the subclass

    def __init__(self, block=BasicBlock, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        self.width_mult = float(width_mult)
        b0, b1, b2 = [scale_width(c, width_mult) for c in BASE]
        t3, t4, t5, t6 = [scale_width(c, width_mult) for c in TREES]
        tree = type(self).tree
        self.base = _conv_bn_relu(3, b0)
        self.layer1 = _conv_bn_relu(b0, b1)
        self.layer2 = _conv_bn_relu(b1, b2)
        self.layer3 = tree(block, b2, t3, level=1, stride=1)
        self.layer4 = tree(block, t3, t4, level=2, stride=2)
        self.layer5 = tree(block, t4, t5, level=2, stride=2)
        self.layer6 = tree(block, t5, t6, level=1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(t6, num_classes)
        self._initialize_weights()

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
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


class SimpleDLA(_DLABase):
    tree = SimpleTree


class DLA(_DLABase):
    tree = Tree


def simpledla_x025(num_classes, large_input):
    return SimpleDLA(num_classes=num_classes, width_mult=0.25)


def simpledla_x050(num_classes, large_input):
    return SimpleDLA(num_classes=num_classes, width_mult=0.5)


def simpledla_x100(num_classes, large_input):
    """kuangliu's SimpleDLA (15.1 M parameters)."""
    return SimpleDLA(num_classes=num_classes, width_mult=1.0)


def dla_x025(num_classes, large_input):
    return DLA(num_classes=num_classes, width_mult=0.25)


def dla_x050(num_classes, large_input):
    return DLA(num_classes=num_classes, width_mult=0.5)


def dla_x100(num_classes, large_input):
    """kuangliu's DLA (16.3 M parameters)."""
    return DLA(num_classes=num_classes, width_mult=1.0)
