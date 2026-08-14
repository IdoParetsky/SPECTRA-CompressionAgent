import timm
import torch
import torch.nn as nn
from timm.models.deit import VisionTransformerDistilled
from timm.layers.patch_embed import PatchEmbed


torch.serialization.add_safe_globals([VisionTransformerDistilled, PatchEmbed, set])

def deit_0_6g_imagenet1k(num_classes=1000, large_input=True):
    model = timm.create_model('deit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def deit_1_2g_imagenet1k(num_classes=1000, large_input=True):
    model = timm.create_model('deit_small_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def deit_2_6g_imagenet1k(num_classes=1000, large_input=True):
    model = timm.create_model('deit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def deit_4_2g_imagenet1k(num_classes=1000, large_input=True):
    model = timm.create_model('deit_base_patch16_384', pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
