import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSmallCifar10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 6 * 6, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        self._init_weights()

    def forward(self, x):
        return self.model(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def conv_small_cifar10(num_classes=10, large_input=False):
    return ConvSmallCifar10(num_classes)


class ConvSmallMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 5 * 5, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        self._init_weights()
        
    def forward(self, x):
        return self.model(x)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def conv_small_mnist(num_classes=10, large_input=False):
    return ConvSmallMNIST(num_classes)


class ConvBigCifar10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        self._init_weights()

    def forward(self, x):
        return self.model(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def conv_big_cifar10(num_classes=10, large_input=False):
    return ConvBigCifar10(num_classes)


class ConvBigMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        self._init_weights()

    def forward(self, x):
        return self.model(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def conv_big_mnist(num_classes=10, large_input=False):
    return ConvBigMNIST(num_classes)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MNISTConvBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)

class MNISTConvBigFC(nn.Module):
    def __init__(self, depth=6, width=100, num_classes=10):
        super().__init__()
        self.features = MNISTConvBackbone()
        layers = [Flatten(), nn.Linear(32 * 5 * 5, width), nn.ReLU()]
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, num_classes))
        self.classifier = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def conv_big_6_100(num_classes=10, large_input=False):
    return MNISTConvBigFC(depth=6, width=100, num_classes=num_classes)

def conv_big_6_200(num_classes=10, large_input=False):
    return MNISTConvBigFC(depth=6, width=200, num_classes=num_classes)

def conv_big_9_100(num_classes=10, large_input=False):
    return MNISTConvBigFC(depth=9, width=100, num_classes=num_classes)

def conv_big_9_200(num_classes=10, large_input=False):
    return MNISTConvBigFC(depth=9, width=200, num_classes=num_classes)

