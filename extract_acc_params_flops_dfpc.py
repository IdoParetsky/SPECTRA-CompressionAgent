import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from thop import profile
import re

# Note: Due to shape mismatches, only the Params (M) and FLOPs (M) are extracted, while the accuracy is as reported in
#       the paper (except for MobileNetV2, whose accuracy was recalculated successfully)

''''MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def load_vgg19(num_classes):
    """Load VGG19 model adapted for CIFAR-10 or CIFAR-100."""
    model = models.vgg19(pretrained=False)

    # Modify the `features` to handle 32x32 input
    model.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # Output will be 2x2 feature maps

    # Modify the classifier for the target dataset
    model.classifier[0] = nn.Linear(512 * 2 * 2, 4096)  # Adjust input size to match AdaptiveAvgPool2d output
    model.classifier[6] = nn.Linear(4096, num_classes)  # Output layer matches the number of classes
    return model


def load_weights(model, path):
    """Load weights from a .pth file."""
    state_dict = torch.load(path, map_location=torch.device('cuda'))

    # Check if the state_dict is nested
    if "net" in state_dict:
        state_dict = state_dict["net"]  # Extract the actual weights

    # Adjust for potential key mismatches
    model_state_dict = model.state_dict()
    matched_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            matched_state_dict[k] = v
        else:
            print(f"Skipping layer: {k} due to shape mismatch.")

    # Load the adjusted state_dict into the model
    model_state_dict.update(matched_state_dict)
    model.load_state_dict(model_state_dict)
    return model


def load_model(model_name, num_classes):
    """Load the specified model with the appropriate number of classes."""
    if model_name == "MobileNetV2":
        model = MobileNetV2(num_classes=num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet101":
        model = models.resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "VGG19":
        model = load_vgg19(num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def evaluate_accuracy(model, loader, device):
    """Evaluate model accuracy."""
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


def process_models(model_name, path, dataset="CIFAR-10"):
    """Process model files and rename with accuracy, params, and FLOPs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Load CIFAR-10 or CIFAR-100 dataset and adjust classes dynamically
    if dataset == "CIFAR-100":
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    elif dataset == "CIFAR-10":
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset: Choose 'CIFAR-10' or 'CIFAR-100'")

    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    # Load specified model and weights
    model = load_model(model_name, num_classes).to(device)
    model = load_weights(model, path)

    # Calculate FLOPs and #Params
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    flops, num_params = profile(model, inputs=(dummy_input,), verbose=False)

    # Evaluate accuracy
    accuracy = evaluate_accuracy(model, test_loader, device)

    # Generate new filename
    acc_str = f"{accuracy:.2f}"
    params_str = f"{num_params / 1e6:.2f}"
    flops_str = f"{flops / 1e6:.2f}"
    base_name, ext = os.path.splitext(path)
    new_filename = rf"{base_name}_{model_name}_{dataset}_{acc_str}_{params_str}_{flops_str}{ext}"

    # Rename file
    os.rename(path, new_filename)
    print(f"Processed {path} -> {new_filename}")


if __name__ == '__main__':
    # Define the directory path, model, and dataset
    path = r"C:\Users\idopa\Documents\BGU\MSc\SPECTRA-CompressionAgent\pretrained networks\resnet50_cifar100_dfpc_pre_78.85.pth"
    model_choice = "ResNet50"  # Options: "MobileNetV2", "ResNet50", "ResNet101", "VGG19"
    dataset_choice = "CIFAR-100"   # Change to "CIFAR-100" for CIFAR-100 processing

    process_models(model_choice, path, dataset=dataset_choice)
