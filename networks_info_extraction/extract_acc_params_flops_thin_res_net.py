import os
import re
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet
from thop import profile
from torch.utils.data import DataLoader

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 base_width: int = 64, dilation: int = 1):
        super().__init__()
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = torch.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride, downsample: Optional[nn.Module] = None, base_width=64, dilation=1):
        super().__init__()
        width = int(planes * (base_width / 64.0))
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = Conv3x3(width, width, stride, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, large_input, width, zero_init_residual=False):
        super().__init__()
        self.inplanes = width
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.base_width = width
        if large_input:
            self.embed = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.embed = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True)
            )

        self.layer1 = self._make_layer(block, width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, width * 2, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, width * 4, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        if len(layers) > 3:
            self.layer4 = self._make_layer(block, width * 8, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
            self.fc = nn.Linear(width * 8 * block.expansion, num_classes)
        else:
            self.layer4 = nn.Identity()
            self.fc = nn.Linear(width * 4 * block.expansion, num_classes)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.base_width, previous_dilation
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    base_width=self.base_width,
                    dilation=self.dilation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embed(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(-1).mean(-1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes, large_input, width):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, large_input, width)


def resnet34(num_classes, large_input, width):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, large_input, width)


def resnet50(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, large_input, width)


def resnet101(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, large_input, width)


def resnet152(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, large_input, width)


def resnet20(num_classes, large_input, width):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, large_input, width)


def resnet32(num_classes, large_input, width):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, large_input, width)


def resnet44(num_classes, large_input, width):
    return ResNet(BasicBlock, [7, 7, 7], num_classes, large_input, width)


def resnet56(num_classes, large_input, width):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, large_input, width)


def resnet110(num_classes, large_input, width):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, large_input, width)


def resnet1202(num_classes, large_input, width):
    return ResNet(BasicBlock, [200, 200, 200], num_classes, large_input, width)

def extract_width(filename):
    # Using regex to find width from filename pattern "..._width<X>..."
    match = re.search(r'_width(\d+)', filename)
    return int(match.group(1)) if match else None

# # Load ResNet-20/56
# def load_net(path):
#     filename = path.split('/')[-1]  # Get filename from path
#     global width
#     width = extract_width(filename)  # Extract width from filename
#     large_input = False  # CIFAR-10/100
#     num_classes = 10|100  # CIFAR-10 has 10 classes, CIFAR-100 has 100
#
#     model = resnet20|56(num_classes, large_input, width)
#     model.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
#     return model

# ImageNet
def load_net(path):
    filename = path.split('/')[-1]  # Get filename from path
    global width
    width = extract_width(filename)  # Extract width from filename
    large_input = True  # ImageNet
    num_classes = 1000  # ImageNet has 1000 classes

    model = resnet50(num_classes, large_input, width)

    # Load the state_dict
    state_dict = torch.load(path, map_location=torch.device('cuda'))

    # Remove 'module.' prefix from the keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # Strip "module." from the key
        new_state_dict[name] = v

    # Load the modified state_dict into the model
    model.load_state_dict(new_state_dict)

    return model


# Define CIFAR-10 /100 transformations
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
# ])


# Define ImageNet transformations
transform = transforms.Compose([
    transforms.Resize(256),                    # Resize shorter side to 256
    transforms.CenterCrop(224),                # Center crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Load CIFAR-10/100 / ImageNet test dataset
#test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

val_root = './data'  # Path to ImageNet validation images
test_set = torchvision.datasets.ImageNet(root=val_root, split='val', transform=transform)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

# Calculate top-1 accuracy
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Main function to process each model file
def process_models(directory):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for filename in os.listdir(directory):
        if filename.endswith(".pth") or filename.endswith(".pt"):
            file_path = os.path.join(directory, filename)
            model = load_net(file_path).to(device)

            # Calculate FLOPs and #Params using a dummy input
            dummy_input = torch.randn(1, 3, 224, 224).to(device)  # torch.randn(1, 3, 32, 32).to(device)
            flops, num_params = profile(model, inputs=(dummy_input,), verbose=False)

            # Evaluate accuracy
            top1_accuracy = evaluate_accuracy(model, test_loader, device)

            # Format new filename with extracted values
            acc_str = f"{top1_accuracy:.2f}"
            params_str = f"{(num_params / 1e6):.3f}"
            flops_str = f"{(flops / 1e6):.2f}"
            base_name, ext = os.path.splitext(filename)
            new_filename = f"resnet50_width{width}_imagenet1k_thin_res_net_{acc_str}_{params_str}_{flops_str}{ext}"
            new_file_path = os.path.join(directory, new_filename)

            # Rename file with new name format
            os.rename(file_path, new_file_path)
            print(f"Processed {filename} -> {new_filename}")

if __name__ == '__main__':
    directory_path = r"C:\Users\idopa\Documents\BGU\MSc\SPECTRA-CompressionAgent\pretrained networks\trained_models\imagenet"
    width = None
    process_models(directory_path)
