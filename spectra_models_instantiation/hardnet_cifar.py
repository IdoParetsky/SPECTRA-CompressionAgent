"""HarDNet (Harmonic DenseNet) for 32x32 CIFAR inputs: 39-DS, 68-DS and two widths of 68.

A HarDNet block is a chain of conv layers in which layer k reads the concatenation of the
outputs of layers k - 2^i (for every 2^i that divides k), so most links skip several layers;
the block's output concatenates its odd layers and its last one, and a 1x1 transition conv
follows. The DS networks replace each 3x3 conv with a 1x1 conv followed by a 3x3 depthwise
conv, and each max-pool with a stride-2 depthwise conv.

Sources: Chao et al., "HarDNet: A Low Memory Traffic Network" (ICCV 2019, arXiv:1909.00948),
and the authors' implementation (PingoLH/Pytorch-HarDNet, ``hardnet.py``,
https://github.com/PingoLH/Pytorch-HarDNet, MIT licence, Copyright (c) 2020 NEUCHIPS
CORPORATION), whose link rule (``get_link``), block, three architecture tables (39, 68, 85)
and layer names are copied here. With the 1000-class head the parameter counts are the
published ones (HarDNet-68 17.6 M, 68-DS 4.2 M, 39-DS 3.5 M; ``tests/test_sealed_families.py``
checks them).

Changes for CIFAR: the stem does not downsample (the first conv has stride 1 and the stem's
max-pool is dropped; in the DS nets the stem's depthwise conv has stride 1), so with the
three downsampling steps between the blocks the last feature map is 4x4. ``width_mult``
(ours) scales the stem widths, the transition widths and the growth rates; layer widths stay
even, as in the original. The authors' ``Flatten`` module is ``nn.Flatten``.

SPECTRA notes: the stem conv is the first registered layer and the classifier the last, in
forward order (the original's single ``base`` ModuleList). The links are plain ``torch.cat``
calls over a fixed Python list, which ``torch.fx`` traces; no layer output is read twice by
one layer. ``large_input`` is accepted for the loader's contract and ignored: this is a 32-px
definition.
"""
import torch
import torch.nn as nn

__all__ = ["HarDNet", "hardnet39ds", "hardnet68ds", "hardnet68_x050", "hardnet68_x075"]

# arch -> stem widths, transition widths, growth-rate multiplier, growth rates, layers per block, downsample flags,
#         dropout before the classifier
ARCHS = {
    68: dict(first_ch=(32, 64), ch_list=(128, 256, 320, 640, 1024), grmul=1.7, gr=(14, 16, 20, 40, 160),
             n_layers=(8, 16, 16, 16, 4), down_samp=(1, 0, 1, 1, 0), drop_rate=0.1),
    85: dict(first_ch=(48, 96), ch_list=(192, 256, 320, 480, 720, 1280), grmul=1.7, gr=(24, 24, 28, 36, 48, 256),
             n_layers=(8, 16, 16, 16, 16, 4), down_samp=(1, 0, 1, 0, 1, 0), drop_rate=0.2),
    39: dict(first_ch=(24, 48), ch_list=(96, 320, 640, 1024), grmul=1.6, gr=(16, 20, 64, 160),
             n_layers=(4, 16, 8, 4), down_samp=(1, 1, 1, 0), drop_rate=0.1),
}


def _even(channels: float) -> int:
    """The nearest even integer, at least 2."""
    return max(2, int(round(channels / 2.0)) * 2)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1, bias: bool = False):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride,
                                          padding=kernel // 2, groups=1, bias=bias))
        self.add_module("norm", nn.BatchNorm2d(out_channels))
        self.add_module("relu", nn.ReLU6(True))


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, bias: bool = False):
        super().__init__()
        groups = in_channels
        self.add_module("dwconv", nn.Conv2d(groups, groups, kernel_size=3, stride=stride, padding=1,
                                            groups=groups, bias=bias))
        self.add_module("norm", nn.BatchNorm2d(groups))


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 1, stride: int = 1):
        super().__init__()
        self.add_module("layer1", ConvLayer(in_channels, out_channels, kernel))
        self.add_module("layer2", DWConvLayer(out_channels, out_channels, stride=stride))


class HarDBlock(nn.Module):
    def get_link(self, layer: int, base_ch: int, growth_rate: float, grmul: float):
        """``(out channels, in channels, linked layers)`` of ``layer`` (0 is the block's input)."""
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self) -> int:
        return self.out_channels

    def __init__(self, in_channels: int, growth_rate: float, grmul: float, n_layers: int,
                 keep_base: bool = False, dwconv: bool = False):
        super().__init__()
        self.keep_base = keep_base
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keep_base) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        return torch.cat(out_, 1)


class HarDNet(nn.Module):
    def __init__(self, arch: int = 68, depth_wise: bool = False, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"Unsupported HarDNet arch {arch}: 39, 68 or 85 expected")
        self.arch, self.depth_wise, self.width_mult = arch, bool(depth_wise), float(width_mult)
        cfg = ARCHS[arch]
        first_ch = [_even(c * width_mult) for c in cfg["first_ch"]]
        ch_list = [_even(c * width_mult) for c in cfg["ch_list"]]
        gr = [g * width_mult for g in cfg["gr"]]
        grmul, n_layers, down_samp = cfg["grmul"], cfg["n_layers"], cfg["down_samp"]
        second_kernel, max_pool, drop_rate = 3, True, cfg["drop_rate"]
        if depth_wise:
            second_kernel, max_pool, drop_rate = 1, False, 0.05

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First layer: standard conv 3x3 (stride 2 on ImageNet, stride 1 at CIFAR size)
        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=1, bias=False))
        # Second layer
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))
        # ImageNet: max-pooling or a stride-2 depthwise conv here. CIFAR: no pool; the depthwise conv stays, stride 1
        if not max_pool:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=1))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i == blks - 1 and arch == 85:
                self.base.append(nn.Dropout(0.1))

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if down_samp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))

        ch = ch_list[blks - 1]
        self.base.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(ch, num_classes)))
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
        for layer in self.base:
            x = layer(x)
        return x


def hardnet39ds(num_classes, large_input):
    return HarDNet(arch=39, depth_wise=True, num_classes=num_classes)


def hardnet68ds(num_classes, large_input):
    return HarDNet(arch=68, depth_wise=True, num_classes=num_classes)


def hardnet68_x050(num_classes, large_input):
    return HarDNet(arch=68, depth_wise=False, num_classes=num_classes, width_mult=0.5)


def hardnet68_x075(num_classes, large_input):
    return HarDNet(arch=68, depth_wise=False, num_classes=num_classes, width_mult=0.75)
