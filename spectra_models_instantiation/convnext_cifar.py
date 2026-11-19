"""ConvNeXt for 32x32 CIFAR inputs: four small widths (stage dims from 24 to 64).

The ConvNeXt block: 7x7 depthwise conv, LayerNorm over channels, a pointwise MLP (Linear to
four times the width, GELU, Linear back) with layer scale, and a residual add. Four stages
with a LayerNorm + 2x2 stride-2 conv between them, global average pooling, LayerNorm, Linear.
There is no BatchNorm anywhere.

Sources: Liu et al., "A ConvNet for the 2020s" (CVPR 2022, arXiv:2201.03545), and the official
implementation (facebookresearch/ConvNeXt, ``models/convnext.py``,
https://github.com/facebookresearch/ConvNeXt, MIT licence, Copyright (c) Meta Platforms, Inc.
and affiliates), whose block (the "permute to channels-last, LayerNorm, Linear, GELU, Linear,
permute back" form), downsampling layers, head and initialisation (truncated normal, std 0.02)
are followed. The depths (2, 2, 6, 2) and the dims of the atto / femto / pico sizes
((40, 80, 160, 320), (48, 96, 192, 384), (64, 128, 256, 512)) are those of the small ConvNeXts
of Woo et al., "ConvNeXt V2" (CVPR 2023, arXiv:2301.00808) and timm; ``convnext_cifar_w24``
((24, 48, 96, 192)) is a narrower net of the same shape and is not a published size.

Changes for CIFAR: the 4x4 stride-4 patchify stem is a 2x2 stride-2 conv, so the four stages
run at 16, 8, 4 and 2 px. Stochastic depth is left out (the official default rate is 0).
The channels-first LayerNorm of the stem and of the downsampling layers is written as
``permute -> nn.LayerNorm -> permute`` (``LayerNorm2d``), the same function as the official
channels-first branch.

SPECTRA notes: the stem conv is the first registered layer and the head the last, in forward
order (the official file registers all downsampling layers before all stages).
``src/channel_groups.py`` (as of tree_v36) cannot resize ``nn.LayerNorm`` and blocks a group
whose tensor is permuted to channels-last, so every stage's residual stream stays whole and
only the MLP's hidden width (4 x dim) can be cut; see ``scripts/sealed_structure_check.py``.
The layer-scale ``gamma`` is a bare per-channel parameter that the channel groups do not track.
The official recipe trains with AdamW; whether SPECTRA's SGD origin recipe trains these nets
is not checked here. ``large_input`` is accepted for the loader's contract and ignored.
"""
import torch
import torch.nn as nn

__all__ = ["ConvNeXt", "convnext_cifar_w24", "convnext_cifar_atto", "convnext_cifar_femto", "convnext_cifar_pico"]

DEPTHS = (2, 2, 6, 2)


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel axis of an (N, C, H, W) tensor (the official ``channels_first`` LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class Block(nn.Module):
    """ConvNeXt block, the official form (2): depthwise conv, then LayerNorm and the MLP in channels-last."""

    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise / 1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
                      if layer_scale_init_value > 0 else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return shortcut + x


class ConvNeXt(nn.Module):
    def __init__(self, num_classes: int = 10, depths=DEPTHS, dims=(40, 80, 160, 320), patch: int = 2,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNeXt takes four stages")
        self.depths, self.dims = tuple(depths), tuple(dims)
        layers = [nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=patch, stride=patch), LayerNorm2d(dims[0]))]
        for i in range(4):
            if i > 0:
                layers.append(nn.Sequential(LayerNorm2d(dims[i - 1]),
                                            nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2)))
            layers.append(nn.Sequential(*[Block(dims[i], layer_scale_init_value) for _ in range(depths[i])]))
        # stem, stage 1, then (downsampling layer, stage) three times, in forward order
        self.features = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        return self.head(x)


def convnext_cifar_w24(num_classes, large_input):
    return ConvNeXt(num_classes=num_classes, dims=(24, 48, 96, 192))


def convnext_cifar_atto(num_classes, large_input):
    return ConvNeXt(num_classes=num_classes, dims=(40, 80, 160, 320))


def convnext_cifar_femto(num_classes, large_input):
    return ConvNeXt(num_classes=num_classes, dims=(48, 96, 192, 384))


def convnext_cifar_pico(num_classes, large_input):
    return ConvNeXt(num_classes=num_classes, dims=(64, 128, 256, 512))
