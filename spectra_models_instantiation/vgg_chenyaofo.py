import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast


class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 10, classifier_width: int = 512):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(classifier_width, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[int, List[Union[str, int]]] = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_bn(num_classes, large_input):
    return VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']), num_classes=num_classes)


def vgg13_bn(num_classes, large_input):
    return VGG(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']), num_classes=num_classes)


def vgg16_bn(num_classes, large_input):
    return VGG(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']), num_classes=num_classes)


def vgg19_bn(num_classes, large_input):
    return VGG(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']), num_classes=num_classes)
    
def vgg19(num_classes, large_input):
    return VGG(make_layers(
        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
         512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        batch_norm=False), num_classes=num_classes)


def vgg19_bn_linear(num_classes, large_input=False):
    """VGG-19-BN with a single Linear head (DFPC CIFAR checkpoints)."""
    model = VGG(make_layers(cfgs[19]), num_classes=num_classes)
    model.classifier = nn.Linear(512, num_classes)
    return model


def scale_cfg(cfg: List[Union[str, int]], mult: float) -> List[Union[str, int]]:
    """Scale every conv width of a VGG cfg list by ``mult`` (floor of 8); 'M' entries stay."""
    return [v if v == 'M' else max(8, int(round(cast(int, v) * mult))) for v in cfg]


def _vgg_bn_scaled(depth: int, mult: float, num_classes: int) -> VGG:
    """Width-scaled VGG-BN: same cfg as ``cfgs[depth]``, classifier input = round(512 * mult)."""
    return VGG(make_layers(scale_cfg(cfgs[depth], mult), batch_norm=True), num_classes=num_classes,
               classifier_width=int(round(512 * mult)))


def vgg11_bn_x050(num_classes, large_input):
    return _vgg_bn_scaled(11, 0.5, num_classes)


def vgg11_bn_x075(num_classes, large_input):
    return _vgg_bn_scaled(11, 0.75, num_classes)


def vgg13_bn_x050(num_classes, large_input):
    return _vgg_bn_scaled(13, 0.5, num_classes)


def vgg13_bn_x075(num_classes, large_input):
    return _vgg_bn_scaled(13, 0.75, num_classes)


def vgg16_bn_x050(num_classes, large_input):
    return _vgg_bn_scaled(16, 0.5, num_classes)


def vgg16_bn_x075(num_classes, large_input):
    return _vgg_bn_scaled(16, 0.75, num_classes)
