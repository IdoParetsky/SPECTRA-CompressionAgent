import torch
import torch.nn as nn
import torchvision.models as models

def replace_final_linear(model, num_classes):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            parent = model
            submodules = name.split('.')
            for sub in submodules[:-1]:
                parent = getattr(parent, sub)
            setattr(parent, submodules[-1], nn.Linear(module.in_features, num_classes))
            return model
    raise ValueError("No Linear layer found to replace.")
    
def replace_final_squeezenet(model, num_classes):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            parent = model
            submodules = name.split('.')
            for sub in submodules[:-1]:
                parent = getattr(parent, sub)
            setattr(parent, submodules[-1], nn.Conv2d(module.in_channels, num_classes, kernel_size=1))
            return model

    raise ValueError("No suitable Conv2d layer found to replace.")

def generic_model(model_fn, num_classes=1000, large_input=True):
    model = model_fn(weights=None)
    if model_fn in [models.squeezenet1_0, models.squeezenet1_1]:
      model = replace_final_squeezenet(model, num_classes)
    else:
      model = replace_final_linear(model, num_classes)
    return model

def alexnet(num_classes=1000, large_input=True): return generic_model(models.alexnet, num_classes)

def convnext_base(num_classes=1000, large_input=True): return generic_model(models.convnext_base, num_classes)
def convnext_large(num_classes=1000, large_input=True): return generic_model(models.convnext_large, num_classes)
def convnext_small(num_classes=1000, large_input=True): return generic_model(models.convnext_small, num_classes)
def convnext_tiny(num_classes=1000, large_input=True): return generic_model(models.convnext_tiny, num_classes)

def densenet121(num_classes=1000, large_input=True): return generic_model(models.densenet121, num_classes)
def densenet161(num_classes=1000, large_input=True): return generic_model(models.densenet161, num_classes)
def densenet169(num_classes=1000, large_input=True): return generic_model(models.densenet169, num_classes)
def densenet201(num_classes=1000, large_input=True): return generic_model(models.densenet201, num_classes)

def efficientnet_b0(num_classes=1000, large_input=True): return generic_model(models.efficientnet_b0, num_classes)
def efficientnet_b1(num_classes=1000, large_input=True): return generic_model(models.efficientnet_b1, num_classes)
def efficientnet_b2(num_classes=1000, large_input=True): return generic_model(models.efficientnet_b2, num_classes)
def efficientnet_b3(num_classes=1000, large_input=True): return generic_model(models.efficientnet_b3, num_classes)
def efficientnet_b4(num_classes=1000, large_input=True): return generic_model(models.efficientnet_b4, num_classes)
def efficientnet_b5(num_classes=1000, large_input=True): return generic_model(models.efficientnet_b5, num_classes)
def efficientnet_b6(num_classes=1000, large_input=True): return generic_model(models.efficientnet_b6, num_classes)
def efficientnet_b7(num_classes=1000, large_input=True): return generic_model(models.efficientnet_b7, num_classes)

def efficientnet_v2_large(num_classes=1000, large_input=True): return generic_model(models.efficientnet_v2_l, num_classes)
def efficientnet_v2_medium(num_classes=1000, large_input=True): return generic_model(models.efficientnet_v2_m, num_classes)
def efficientnet_v2_small(num_classes=1000, large_input=True): return generic_model(models.efficientnet_v2_s, num_classes)

def googlenet(num_classes=1000, large_input=True):
    return generic_model(
        lambda weights=None: models.googlenet(weights=weights, aux_logits=False, init_weights=False),
        num_classes)


def inception_v3(num_classes=1000, large_input=True):
    return generic_model(lambda weights=None: models.inception_v3(weights=weights, aux_logits=False, init_weights=False), num_classes)

def maxvit_t(num_classes=1000, large_input=True): return generic_model(models.maxvit_t, num_classes)

def mnasnet_05(num_classes=1000, large_input=True): return generic_model(models.mnasnet0_5, num_classes)
def mnasnet_075(num_classes=1000, large_input=True): return generic_model(models.mnasnet0_75, num_classes)
def mnasnet_1(num_classes=1000, large_input=True): return generic_model(models.mnasnet1_0, num_classes)
def mnasnet_13(num_classes=1000, large_input=True): return generic_model(models.mnasnet1_3, num_classes)

def mobilenet_v2(num_classes=1000, large_input=True): return generic_model(models.mobilenet_v2, num_classes)
def mobilenet_v3_large(num_classes=1000, large_input=True): return generic_model(models.mobilenet_v3_large, num_classes)
def mobilenet_v3_small(num_classes=1000, large_input=True): return generic_model(models.mobilenet_v3_small, num_classes)

def regnet_x_16gf(num_classes=1000, large_input=True): return generic_model(models.regnet_x_16gf, num_classes)
def regnet_x_1_6gf(num_classes=1000, large_input=True): return generic_model(models.regnet_x_1_6gf, num_classes)
def regnet_x_32gf(num_classes=1000, large_input=True): return generic_model(models.regnet_x_32gf, num_classes)
def regnet_x_3_2gf(num_classes=1000, large_input=True): return generic_model(models.regnet_x_3_2gf, num_classes)
def regnet_x_400mf(num_classes=1000, large_input=True): return generic_model(models.regnet_x_400mf, num_classes)
def regnet_x_800mf(num_classes=1000, large_input=True): return generic_model(models.regnet_x_800mf, num_classes)
def regnet_x_8gf(num_classes=1000, large_input=True): return generic_model(models.regnet_x_8gf, num_classes)

def regnet_y_400mf(num_classes=1000, large_input=True): return generic_model(models.regnet_y_400mf, num_classes)
def regnet_y_800mf(num_classes=1000, large_input=True): return generic_model(models.regnet_y_800mf, num_classes)
def regnet_y_1_6gf(num_classes=1000, large_input=True): return generic_model(models.regnet_y_1_6gf, num_classes)
def regnet_y_3_2gf(num_classes=1000, large_input=True): return generic_model(models.regnet_y_3_2gf, num_classes)
def regnet_y_8gf(num_classes=1000, large_input=True): return generic_model(models.regnet_y_8gf, num_classes)
def regnet_y_16gf(num_classes=1000, large_input=True): return generic_model(models.regnet_y_16gf, num_classes)
def regnet_y_32gf(num_classes=1000, large_input=True): return generic_model(models.regnet_y_32gf, num_classes)

def resnet18(num_classes=1000, large_input=True): return generic_model(models.resnet18, num_classes)
def resnet34(num_classes=1000, large_input=True): return generic_model(models.resnet34, num_classes)
def resnet50(num_classes=1000, large_input=True): return generic_model(models.resnet50, num_classes)
def resnet101(num_classes=1000, large_input=True): return generic_model(models.resnet101, num_classes)
def resnet152(num_classes=1000, large_input=True): return generic_model(models.resnet152, num_classes)

def resnext101_32x8d(num_classes=1000, large_input=True):
    return generic_model(lambda weights=None: models.resnext101_32x8d(weights=weights), num_classes)
def resnext101_64x4d(num_classes=1000, large_input=True):
    return generic_model(lambda weights=None: models.resnext101_64x4d(weights=weights), num_classes)
def resnext50_32x4d(num_classes=1000, large_input=True):
    return generic_model(lambda weights=None: models.resnext50_32x4d(weights=weights), num_classes)


def shufflenet_v2_x0_5(num_classes=1000, large_input=True): return generic_model(models.shufflenet_v2_x0_5, num_classes)
def shufflenet_v2_x1_0(num_classes=1000, large_input=True): return generic_model(models.shufflenet_v2_x1_0, num_classes)
def shufflenet_v2_x1_5(num_classes=1000, large_input=True): return generic_model(models.shufflenet_v2_x1_5, num_classes)
def shufflenet_v2_x2_0(num_classes=1000, large_input=True): return generic_model(models.shufflenet_v2_x2_0, num_classes)

def squeezenet1_0(num_classes=1000, large_input=True): return generic_model(models.squeezenet1_0, num_classes)
def squeezenet1_1(num_classes=1000, large_input=True): return generic_model(models.squeezenet1_1, num_classes)

def swin_s(num_classes=1000, large_input=True): return generic_model(models.swin_s, num_classes)
def swin_t(num_classes=1000, large_input=True): return generic_model(models.swin_t, num_classes)
def swin_b(num_classes=1000, large_input=True): return generic_model(models.swin_b, num_classes)
def swin_v2_s(num_classes=1000, large_input=True): return generic_model(models.swin_v2_s, num_classes)
def swin_v2_t(num_classes=1000, large_input=True): return generic_model(models.swin_v2_t, num_classes)
def swin_v2_b(num_classes=1000, large_input=True): return generic_model(models.swin_v2_b, num_classes)

def vgg11(num_classes=1000, large_input=True): return generic_model(models.vgg11, num_classes)
def vgg11_bn(num_classes=1000, large_input=True): return generic_model(models.vgg11_bn, num_classes)
def vgg13(num_classes=1000, large_input=True): return generic_model(models.vgg13, num_classes)
def vgg13_bn(num_classes=1000, large_input=True): return generic_model(models.vgg13_bn, num_classes)
def vgg16(num_classes=1000, large_input=True): return generic_model(models.vgg16, num_classes)
def vgg16_bn(num_classes=1000, large_input=True): return generic_model(models.vgg16_bn, num_classes)
def vgg19(num_classes=1000, large_input=True): return generic_model(models.vgg19, num_classes)
def vgg19_bn(num_classes=1000, large_input=True): return generic_model(models.vgg19_bn, num_classes)

def vit_b_16(num_classes=1000, large_input=True): return generic_model(models.vit_b_16, num_classes)
def vit_b_32(num_classes=1000, large_input=True): return generic_model(models.vit_b_32, num_classes)
def vit_l_16(num_classes=1000, large_input=True): return generic_model(models.vit_l_16, num_classes)
def vit_l_32(num_classes=1000, large_input=True): return generic_model(models.vit_l_32, num_classes)

def wide_resnet50_2(num_classes=1000, large_input=True): return generic_model(models.wide_resnet50_2, num_classes)
def wide_resnet101_2(num_classes=1000, large_input=True): return generic_model(models.wide_resnet101_2, num_classes)
