import os
import re
import torch
import torchvision
import torchvision.transforms as transforms
from thop import profile
from torch.utils.data import DataLoader
import torch.nn as nn
import math


# Define MobileNet
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, n_class, profile='normal'):
        super(MobileNet, self).__init__()

        # Profile configurations
        if profile == 'normal':
            in_planes = 32
            cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
        elif profile == '0.5flops':
            in_planes = 24
            cfg = [48, (96, 2), 80, (192, 2), 200, (328, 2), 352, 368, 360, 328, 400, (736, 2), 752]
        else:
            raise NotImplementedError

        self.conv1 = conv_bn(3, in_planes, stride=2)
        self.features = self._make_layers(in_planes, cfg, conv_dw)
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], n_class),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.mean(3).mean(2)  # Global average pooling
        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, layer):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# Load MobileNet model
def load_net(path):
    filename = path.split('/')[-1]
    n_class = 1000  # ImageNet classes

    model = MobileNet(n_class=n_class, profile='normal')

    # Load state_dict
    checkpoint = torch.load(path, map_location=torch.device('cuda'))

    # If 'state_dict' is nested inside the checkpoint, extract it
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    # Load weights into the model
    model.load_state_dict(new_state_dict)

    return model



# ImageNet transformations
transform = transforms.Compose([
    transforms.Resize(256),                    # Resize shorter side to 256
    transforms.CenterCrop(224),                # Center crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet validation dataset
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
def process_models(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_net(path).to(device)

    # Calculate FLOPs and Params using a dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, num_params = profile(model, inputs=(dummy_input,), verbose=False)

    # Evaluate accuracy
    top1_accuracy = evaluate_accuracy(model, test_loader, device)

    # Format new filename
    acc_str = f"{top1_accuracy:.2f}"
    params_str = f"{(num_params / 1e6):.3f}"
    flops_str = f"{(flops / 1e6):.2f}"
    new_filename = rf"C:\Users\idopa\Documents\BGU\MSc\SPECTRA-CompressionAgent\pretrained networks\mobilenet_imagenet1k_gnn_rl_{acc_str}_{params_str}_{flops_str}.pth"

    print(f"Processed {path} -> {new_filename}")

    # Rename file with new name format
    os.rename(path, new_filename)


if __name__ == '__main__':
    path = r"C:\Users\idopa\Documents\BGU\MSc\SPECTRA-CompressionAgent\pretrained networks\mobilenet_imagenet_gnn_rl_70.9.pth"
    process_models(path)
