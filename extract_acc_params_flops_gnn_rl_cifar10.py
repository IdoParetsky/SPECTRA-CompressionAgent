import os
import re
import torch
import torchvision
import torchvision.transforms as transforms
from thop import profile
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import numpy as np


# Define ResNet Implementation for CIFAR-10
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


# Load ResNet model
def load_net(path, net_type='resnet20'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = globals()[net_type]().to(device)

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    return net


# ImageNet transformations for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 validation dataset
val_root = './data'  # Path to CIFAR-10 validation images
test_set = torchvision.datasets.CIFAR10(root=val_root, train=False, download=True, transform=transform)
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
def process_models(path, net_type='resnet20'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_net(path, net_type=net_type).to(device)

    # Calculate FLOPs and Params using a dummy input
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    flops, num_params = profile(model, inputs=(dummy_input,), verbose=False)

    # Evaluate accuracy
    top1_accuracy = evaluate_accuracy(model, test_loader, device)

    # Format new filename
    acc_str = f"{top1_accuracy:.2f}"
    params_str = f"{(num_params / 1e6):.2f}"
    flops_str = f"{(flops / 1e6):.2f}"
    new_filename = rf"C:\Users\idopa\Documents\BGU\MSc\SPECTRA-CompressionAgent\pretrained networks\{net_type}_cifar10_gnn_rl_{acc_str}_{params_str}_{flops_str}.th"

    print(f"Processed {path} -> {new_filename}")

    # Rename file with new name format
    os.rename(path, new_filename)


if __name__ == '__main__':
    path = r"C:\Users\idopa\Documents\BGU\MSc\SPECTRA-CompressionAgent\pretrained networks\resnet1202_cifar10_gnn_rl.th"
    process_models(path, net_type='resnet1202')
