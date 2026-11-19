"""SqueezeNet 1.0 and 1.1 for 32x32 CIFAR inputs, with batch norm and a width multiplier.

Fire modules: a 1x1 squeeze conv, then a 1x1 and a 3x3 expand conv whose outputs are
concatenated. The classifier is a 1x1 conv to ``num_classes`` followed by ReLU and global
average pooling (there is no Linear layer). About 0.73 M parameters at width 1.0 for 10
classes.

Sources: Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
<0.5MB model size" (arXiv:1602.07360, 2016), and its torchvision implementation
(``torchvision/models/squeezenet.py``, https://github.com/pytorch/vision, BSD 3-Clause
licence), whose fire tables for 1.0 and 1.1 and whose classifier are copied here.

Changes for CIFAR:
* the ImageNet stem (7x7 stride 2 for 1.0, 3x3 stride 2 for 1.1) is a 3x3 stride-1 conv with
  padding 1; the three 3x3 stride-2 max-pools stay, so the fire stages run at 16, 8 and 4 px;
* ``batch_norm=True`` (the factories' setting) puts BatchNorm2d after every conv but the
  classifier, as kuangliu's CIFAR GoogLeNet does for Inception v1. The 2016 network has none,
  and a network without normalisation is hard to train at the learning rate SPECTRA's origin
  recipe uses (0.1; not tried here). ``batch_norm=False`` is the paper's network (conv biases,
  no norm); with the 1000-class head it has torchvision's parameter count (1.1: 1,235,496;
  1.0: 1,248,424 less the 11,520 weights the 7x7 stem has over a 3x3 one), which
  ``tests/test_sealed_families.py`` checks;
* ``width_mult`` scales every channel count (rounded, floor of 8).

SPECTRA notes: the stem conv is the first registered layer and the classifier conv the last
Conv2d, in forward order. ``large_input`` is accepted for the loader's contract and ignored:
this is a 32-px definition.
"""
import torch
import torch.nn as nn

__all__ = ["SqueezeNet", "squeezenet1_0", "squeezenet1_0_x200", "squeezenet1_1", "squeezenet1_1_x050"]

MIN_WIDTH = 8
# version -> (stem width, [fire (squeeze, expand1x1, expand3x3) or "M" for a max-pool]); a pool follows the stem
CFG = {
    "1_0": (96, [(16, 64, 64), (16, 64, 64), (32, 128, 128), "M",
                 (32, 128, 128), (48, 192, 192), (48, 192, 192), (64, 256, 256), "M",
                 (64, 256, 256)]),
    "1_1": (64, [(16, 64, 64), (16, 64, 64), "M",
                 (32, 128, 128), (32, 128, 128), "M",
                 (48, 192, 192), (48, 192, 192), (64, 256, 256), (64, 256, 256)]),
}


def scale_width(channels: int, mult: float) -> int:
    """``channels * mult`` rounded to an integer, at least ``MIN_WIDTH``."""
    return max(MIN_WIDTH, int(round(channels * mult)))


def _conv_relu(in_ch: int, out_ch: int, kernel_size: int, batch_norm: bool) -> list:
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2, bias=not batch_norm)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return layers


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int,
                 batch_norm: bool = True):
        super().__init__()
        self.squeeze = nn.Sequential(*_conv_relu(inplanes, squeeze_planes, 1, batch_norm))
        self.expand1x1 = nn.Sequential(*_conv_relu(squeeze_planes, expand1x1_planes, 1, batch_norm))
        self.expand3x3 = nn.Sequential(*_conv_relu(squeeze_planes, expand3x3_planes, 3, batch_norm))
        self.out_planes = expand1x1_planes + expand3x3_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)


class SqueezeNet(nn.Module):
    def __init__(self, version: str = "1_1", num_classes: int = 10, width_mult: float = 1.0,
                 batch_norm: bool = True, dropout: float = 0.5):
        super().__init__()
        if version not in CFG:
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")
        self.version, self.width_mult, self.batch_norm = version, float(width_mult), bool(batch_norm)
        stem, fires = CFG[version]
        planes = scale_width(stem, width_mult)
        layers = _conv_relu(3, planes, 3, batch_norm)
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        for entry in fires:
            if entry == "M":
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
                continue
            fire = Fire(planes, *[scale_width(c, width_mult) for c in entry], batch_norm=batch_norm)
            layers.append(fire)
            planes = fire.out_planes
        self.features = nn.Sequential(*layers)

        # The final convolution is the classifier (initialised differently from the rest, as in torchvision)
        final_conv = nn.Conv2d(planes, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def squeezenet1_0(num_classes, large_input):
    return SqueezeNet("1_0", num_classes=num_classes, width_mult=1.0)


def squeezenet1_0_x200(num_classes, large_input):
    return SqueezeNet("1_0", num_classes=num_classes, width_mult=2.0)


def squeezenet1_1(num_classes, large_input):
    return SqueezeNet("1_1", num_classes=num_classes, width_mult=1.0)


def squeezenet1_1_x050(num_classes, large_input):
    return SqueezeNet("1_1", num_classes=num_classes, width_mult=0.5)
