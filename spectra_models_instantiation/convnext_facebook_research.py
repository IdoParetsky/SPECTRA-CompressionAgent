import torch
import torch.nn as nn

# ---- Core Block ----
class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + self.drop_path(x)

# ---- ConvNeXt Standard ----
class ConvNeXt(nn.Module):
    def __init__(self, depths, dims, num_classes=1000):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6, elementwise_affine=True),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6, elementwise_affine=True),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])
        x = self.norm(x)
        return self.head(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ---- ConvNeXt Isotropic ----
class ConvNeXtISO(nn.Module):
    def __init__(self, depth, dim, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=4, stride=4),
            nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True),
        )
        self.blocks = nn.Sequential(*[Block(dim=dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean([-2, -1])
        x = self.norm(x)
        return self.head(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ---- ConvNeXt 1K Variants ----
def convnext_tiny_1k(num_classes=1000, large_input=True):
    return ConvNeXt([3, 3, 9, 3], [96, 192, 384, 768], num_classes)

def convnext_small_1k(num_classes=1000, large_input=True):
    return ConvNeXt([3, 3, 27, 3], [96, 192, 384, 768], num_classes)

def convnext_base_224_1k(num_classes=1000, large_input=True):
    return ConvNeXt([3, 3, 27, 3], [128, 256, 512, 1024], num_classes)

def convnext_large_224_1k(num_classes=1000, large_input=True):
    return ConvNeXt([3, 3, 27, 3], [192, 384, 768, 1536], num_classes)

def convnext_base_384_1k(num_classes=1000, large_input=True):
    return ConvNeXt([3, 3, 27, 3], [128, 256, 512, 1024], num_classes)

def convnext_large_384_1k(num_classes=1000, large_input=True):
    return ConvNeXt([3, 3, 27, 3], [192, 384, 768, 1536], num_classes)

# ---- ConvNeXt 22K Variants ----
def convnext_tiny_22k(num_classes=21841, large_input=True):
    return ConvNeXt([3, 3, 9, 3], [96, 192, 384, 768], num_classes)

def convnext_small_22k(num_classes=21841, large_input=True):
    return ConvNeXt([3, 3, 27, 3], [96, 192, 384, 768], num_classes)

def convnext_base_22k(num_classes=21841, large_input=True):
    return ConvNeXt([3, 3, 27, 3], [128, 256, 512, 1024], num_classes)

def convnext_large_22k(num_classes=21841, large_input=True):
    return ConvNeXt([3, 3, 27, 3], [192, 384, 768, 1536], num_classes)

def convnext_xlarge_22k(num_classes=21841, large_input=True):
    return ConvNeXt([3, 3, 27, 3], [256, 512, 1024, 2048], num_classes)

# ---- ConvNeXt Isotropic Variants ----
def convnext_isotropic_small_1k(num_classes=1000, large_input=True):
    return ConvNeXtISO(22, 320, num_classes)

def convnext_isotropic_base_1k(num_classes=1000, large_input=True):
    return ConvNeXtISO(87, 640, num_classes)

def convnext_isotropic_large_1k(num_classes=1000, large_input=True):
    return ConvNeXtISO(306, 1024, num_classes)
