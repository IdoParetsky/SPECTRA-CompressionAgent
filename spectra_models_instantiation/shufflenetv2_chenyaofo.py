import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Callable


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    b, c, h, w = x.size()
    cpg = c // groups
    x = x.view(b, groups, cpg, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, -1, h, w)


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError("Stride must be 1 or 2")
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, self.stride, 1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, repeats: List[int], out_channels: List[int],
                 num_classes: int = 10, inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_channels[0], 3, 1, 1, bias=False),  # stride=1 for CIFAR
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        )
        input_channels = out_channels[0]

        stages = []
        for idx, (repeat, out_channel) in enumerate(zip(repeats, out_channels[1:-1])):
            blocks = [inverted_residual(input_channels, out_channel, 2)]
            for _ in range(repeat - 1):
                blocks.append(inverted_residual(out_channel, out_channel, 1))
            stages.append(nn.Sequential(*blocks))
            input_channels = out_channel

        self.stage2 = stages[0]
        self.stage3 = stages[1]
        self.stage4 = stages[2]

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(out_channels[-1], num_classes)

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        return self.fc(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def shufflenetv2x05(num_classes, large_input):
    repeats, out_channels = ([4, 8, 4], [24, 48, 96, 192, 1024])
    return ShuffleNetV2(repeats, out_channels, num_classes=num_classes)


def shufflenetv2x1(num_classes, large_input):
    repeats, out_channels = ([4, 8, 4], [24, 116, 232, 464, 1024])
    return ShuffleNetV2(repeats, out_channels, num_classes=num_classes)


def shufflenetv2x15(num_classes, large_input):
    repeats, out_channels = ([4, 8, 4], [24, 176, 352, 704, 1024])
    return ShuffleNetV2(repeats, out_channels, num_classes=num_classes)


def shufflenetv2x2(num_classes, large_input):
    repeats, out_channels = ([4, 8, 4], [24, 244, 488, 976, 2048])
    return ShuffleNetV2(repeats, out_channels, num_classes=num_classes)
