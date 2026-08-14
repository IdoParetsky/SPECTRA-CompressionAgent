#!/usr/bin/env python3
"""Map every file in spectra_pretrained_networks to (arch, script, dataset) and optionally load.

This is the 287-file inventory, not an agent database. SPECTRA still trains only on
catalog JSONs. The mapper exists so held-out families can be added without guessing
factory names.

    python scripts/map_checkpoint_pool.py              # print mapping, no load
    python scripts/map_checkpoint_pool.py --load       # CPU instantiate + 1-sample forward
    python scripts/map_checkpoint_pool.py --load --skip-huge
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CKPT_ROOT = Path("/home/paretsky/spectra_pretrained_networks")
SCRIPT_ROOT = Path("/home/paretsky/spectra_models_instantiation")
REPO = Path(__file__).resolve().parents[1]

DATASET_JSON = {
    "cifar10": "cifar-10", "cifar100": "cifar-100", "svhn": "svhn",
    "fashionmnist": "fashion-mnist", "mnist": "mnist",
    "imagenet1k": "imagenet1k", "imagenet1kv1": "imagenet1k", "imagenet1kv2": "imagenet1kv2",
    "imagenet": "imagenet1k",
}

TV_ARCH = {
    "alexnet": "alexnet",
    "convnext-base": "convnext_base", "convnext-large": "convnext_large",
    "convnext-small": "convnext_small", "convnext-tiny": "convnext_tiny",
    "densenet121": "densenet121", "densenet161": "densenet161",
    "densenet169": "densenet169", "densenet201": "densenet201",
    "densenet": "densenet161",
    "efficientnet-b0": "efficientnet_b0", "efficientnet-b1": "efficientnet_b1",
    "efficientnet-b2": "efficientnet_b2", "efficientnet-b3": "efficientnet_b3",
    "efficientnet-b4": "efficientnet_b4", "efficientnet-b5": "efficientnet_b5",
    "efficientnet-b6": "efficientnet_b6", "efficientnet-b7": "efficientnet_b7",
    "efficientnet-v2-large": "efficientnet_v2_large",
    "efficientnet-v2-medium": "efficientnet_v2_medium",
    "efficientnet-v2-small": "efficientnet_v2_small",
    "googlenet": "googlenet", "inception-v3": "inception_v3", "inception": "inception_v3",
    "maxvit-t": "maxvit_t",
    "mnasnet-05": "mnasnet_05", "mnasnet-075": "mnasnet_075",
    "mnasnet-1": "mnasnet_1", "mnasnet-13": "mnasnet_13", "mnasnet": "mnasnet_1",
    "mobilenet-v2": "mobilenet_v2", "mobilenet-v3_large": "mobilenet_v3_large",
    "mobilenet-v3_small": "mobilenet_v3_small", "mobilenet_v3_large": "mobilenet_v3_large",
    "regnet-x-1-6gf": "regnet_x_1_6gf", "regnet-x-16gf": "regnet_x_16gf",
    "regnet-x-3-2gf": "regnet_x_3_2gf", "regnet-x-32gf": "regnet_x_32gf",
    "regnet-x-400mf": "regnet_x_400mf", "regnet-x-800mf": "regnet_x_800mf",
    "regnet-x-8gf": "regnet_x_8gf",
    "regnet-y-1-6gf": "regnet_y_1_6gf", "regnet-y-16gf": "regnet_y_16gf",
    "regnet-y-3-2gf": "regnet_y_3_2gf", "regnet-y-32gf": "regnet_y_32gf",
    "regnet-y-400mf": "regnet_y_400mf", "regnet-y-800mf": "regnet_y_800mf",
    "regnet-y-8gf": "regnet_y_8gf",
    "resnet18": "resnet18", "resnet34": "resnet34", "resnet50": "resnet50",
    "resnet101": "resnet101", "resnet152": "resnet152",
    "resnext50-32x4d": "resnext50_32x4d", "resnext101-32x8d": "resnext101_32x8d",
    "resnext101-64x4d": "resnext101_64x4d",
    "shufflenet-v2-x0-5": "shufflenet_v2_x0_5", "shufflenet-v2-x1-0": "shufflenet_v2_x1_0",
    "shufflenet-v2-x1-5": "shufflenet_v2_x1_5", "shufflenet-v2-x2-0": "shufflenet_v2_x2_0",
    "shufflenet": "shufflenet_v2_x1_0",
    "squeezenet-1-0": "squeezenet1_0", "squeezenet-1-1": "squeezenet1_1",
    "squeezenet1-0": "squeezenet1_0", "squeezenet1-1": "squeezenet1_1",
    "swin-s": "swin_s", "swin-t": "swin_t", "swin-b": "swin_b",
    "swin-small": "swin_s", "swin-tiny": "swin_t", "swin-big": "swin_b",
    "swin-v2-s": "swin_v2_s", "swin-v2-t": "swin_v2_t", "swin-v2-b": "swin_v2_b",
    "swin-v2-small": "swin_v2_s", "swin-v2-tiny": "swin_v2_t", "swin-v2-big": "swin_v2_b",
    "swin-v2-s": "swin_v2_s", "swin-v2-t": "swin_v2_t", "swin-v2-b": "swin_v2_b",
    "vgg11": "vgg11", "vgg11-bn": "vgg11_bn", "vgg13": "vgg13", "vgg13-bn": "vgg13_bn",
    "vgg16": "vgg16", "vgg16-bn": "vgg16_bn", "vgg19": "vgg19", "vgg19-bn": "vgg19_bn",
    "vit-b-16": "vit_b_16", "vit-b-32": "vit_b_32", "vit-l-16": "vit_l_16", "vit-l-32": "vit_l_32",
    "wide-resnet50-2": "wide_resnet50_2", "wide-resnet101-2": "wide_resnet101_2",
}

HUGE = re.compile(
    r"(vit-l|convnext.large|convnext.xlarge|efficientnet-b[67]|efficientnet-v2-large|"
    r"regnet-[xy]-32gf|resnet152|resnext101|wide-resnet101|densenet201|"
    r"resnet1202|deit_4.2G|convnext_iso_large)",
    re.I,
)


def _dataset_from_name(name: str) -> str | None:
    lower = name.lower()
    for key, canon in DATASET_JSON.items():
        if f"_{key}_" in f"_{lower}_" or f"_{key}." in f"_{lower}.":
            return canon
    return None


def _scale_token(raw: str) -> str:
    return raw.replace(".", "") if "." in raw else raw


def map_file(name: str):
    ds = _dataset_from_name(name)
    ncls = {"cifar-100": 100, "imagenet1k": 1000, "imagenet1kv2": 1000, "places365": 365}.get(ds, 10)
    shape = (3, 224, 224) if ds in ("imagenet1k", "imagenet1kv2") else (3, 32, 32)
    if ds == "mnist":
        shape = (1, 28, 28)

    # thin-res-net
    m = re.match(r"(resnet\d+)-width\d+_(cifar10|cifar100|svhn|fashionmnist|imagenet1k)_thin-res-net_", name)
    if m:
        ds = DATASET_JSON[m.group(2)]
        ncls = 100 if ds == "cifar-100" else (1000 if "imagenet" in ds else 10)
        shape = (3, 224, 224) if "imagenet" in ds else (3, 32, 32)
        spec = ds
        if ds == "fashion-mnist":
            spec = {"name": "fashion-mnist", "image_size": 32, "to_rgb": True}
        return m.group(1), "thin_res_net.py", spec, ncls, shape

    # chenyaofo families
    m = re.match(r"(resnet\d+)_(cifar10|cifar100)_chenyaofo_", name)
    if m:
        ds = DATASET_JSON[m.group(2)]
        return m.group(1), "resnet_chenyaofo.py", ds, 100 if ds == "cifar-100" else 10, (3, 32, 32)
    m = re.match(r"(vgg\d+_bn)_(cifar10|cifar100)_chenyaofo_", name)
    if m:
        ds = DATASET_JSON[m.group(2)]
        return m.group(1), "vgg_chenyaofo.py", ds, 100 if ds == "cifar-100" else 10, (3, 32, 32)
    m = re.match(r"mobilenet-v2x([0-9.]+)_(cifar10|cifar100)_chenyaofo_", name)
    if m:
        ds = DATASET_JSON[m.group(2)]
        arch = "mobilenet_v2x" + _scale_token(m.group(1))
        return arch, "mobilenetv2_chenyaofo.py", ds, 100 if ds == "cifar-100" else 10, (3, 32, 32)
    m = re.match(r"shufflenetv2x([0-9.]+)_(cifar10|cifar100)_chenyaofo_", name)
    if m:
        ds = DATASET_JSON[m.group(2)]
        arch = "shufflenetv2x" + _scale_token(m.group(1))
        return arch, "shufflenetv2_chenyaofo.py", ds, 100 if ds == "cifar-100" else 10, (3, 32, 32)
    m = re.match(r"repvgga(\d+)_(cifar10|cifar100)_chenyaofo_", name)
    if m:
        ds = DATASET_JSON[m.group(2)]
        return f"repvgg_a{m.group(1)}", "repvgg_chenyaofo.py", ds, 100 if ds == "cifar-100" else 10, (3, 32, 32)

    # akamaster
    m = re.match(r"(resnet\d+)_cifar10_akamaster_", name)
    if m:
        return m.group(1), "resnet_akamaster.py", "cifar-10", 10, (3, 32, 32)

    # densenet-cifar
    m = re.match(r"(densenet(?:40|100))_(cifar10|cifar100)_densenet-cifar_", name)
    if m:
        ds = DATASET_JSON[m.group(2)]
        return m.group(1), "densenet_cifar.py", ds, 100 if ds == "cifar-100" else 10, (3, 32, 32)

    # gnn-rl
    m = re.match(r"(resnet\d+)_cifar10_gnn_rl_", name)
    if m:
        return m.group(1), "resnet_gnn_rl.py", "cifar-10", 10, (3, 32, 32)
    if name.startswith("mobilenet_imagenet1k_gnn_rl"):
        return "mobilenet", "mobilenet_gnn_rl.py", "imagenet1k", 1000, (3, 224, 224)

    # grafting
    if name.startswith("cnn-b-2-255_cifar10_baseline"):
        return "cnn_b_2_255_baseline", "cnn_b_2_lin_grafting.py", "cifar-10", 10, (3, 32, 32)
    if name.startswith("cnn-b-2-255_cifar10_graft"):
        return "cnn_b_2_255_graft", "cnn_b_2_lin_grafting.py", "cifar-10", 10, (3, 32, 32)
    graft_conv = {
        "conv-small_cifar10": ("conv_small_cifar10", "cifar-10", (3, 32, 32), 10),
        "conv-small_mnist": ("conv_small_mnist", "mnist", (1, 28, 28), 10),
        "conv-big_cifar10": ("conv_big_cifar10", "cifar-10", (3, 32, 32), 10),
        "conv-big_mnist": ("conv_big_mnist", "mnist", (1, 28, 28), 10),
        "conv-big-6-100_mnist": ("conv_big_6_100", "mnist", (1, 28, 28), 10),
        "conv-big-6-200_mnist": ("conv_big_6_200", "mnist", (1, 28, 28), 10),
        "conv-big-9-100_mnist": ("conv_big_9_100", "mnist", (1, 28, 28), 10),
        "conv-big-9-200_mnist": ("conv_big_9_200", "mnist", (1, 28, 28), 10),
    }
    for prefix, (arch, ds, shape, ncls) in graft_conv.items():
        if name.startswith(prefix):
            return arch, "conv_lin_grafting.py", ds, ncls, shape
    m = re.match(r"(resnet(?:2b|4b|8px|base|deep))_cifar10_lin_grafting", name)
    if m:
        return m.group(1), "resnet_lin_grafting.py", "cifar-10", 10, (3, 32, 32)

    # sublinear
    if name.startswith("lenet_mnist_sublinear"):
        return "lenet_mnist_sublinear", "sublinear_instantiation.py", "mnist", 10, (1, 28, 28)
    if name.startswith("vgg11_bn_small_cifar10_sublinear"):
        return "vgg11_bn_small_cifar10_sublinear", "sublinear_instantiation.py", "cifar-10", 10, (3, 32, 32)
    if "resnet56" in name and "sublinear" in name:
        return "resnet56_cifar10_sublinear", "sublinear_instantiation.py", "cifar-10", 10, (3, 32, 32)

    # vgg chenyaofo on svhn / fashionmnist (hyphenated bn)
    m = re.match(r"vgg11-bn_(fashionmnist|svhn)_vgg-chenyaofo_", name)
    if m:
        ds = DATASET_JSON[m.group(1)]
        spec = ds if ds != "fashion-mnist" else {"name": "fashion-mnist", "image_size": 32, "to_rgb": True}
        return "vgg11_bn", "vgg_chenyaofo.py", spec, 10, (3, 32, 32)

    # facebook convnext
    fb = {
        "convnext_tiny_imagenet1k_facebook_research": "convnext_tiny_1k",
        "convnext_small_imagenet1k_facebook_research": "convnext_small_1k",
        "convnext_base_224_imagenet1k_facebook_research": "convnext_base_224_1k",
        "convnext_large_224_imagenet1k_facebook_research": "convnext_large_224_1k",
        "convnext_base_384_imagenet1k_facebook_research": "convnext_base_384_1k",
        "convnext_large_384_imagenet1k_facebook_research": "convnext_large_384_1k",
        "convnext_iso_small_imagenet1k_facebook_research": "convnext_isotropic_small_1k",
        "convnext_iso_base_imagenet1k_facebook_research": "convnext_isotropic_base_1k",
        "convnext_iso_large_imagenet1k_facebook_research": "convnext_isotropic_large_1k",
    }
    for prefix, arch in fb.items():
        if name.startswith(prefix):
            return arch, "convnext_facebook_research.py", "imagenet1k", 1000, (3, 224, 224)

    # deit isomorphic
    m = re.match(r"deit_([0-9.]+)G_imagenet1k_isomorphic", name)
    if m:
        token = m.group(1).replace(".", "_")
        return f"deit_{token}g_imagenet1k", "deit_instantiation.py", "imagenet1k", 1000, (3, 224, 224)

    # torchvision / pytorch ImageNet dumps (hyphenated stems)
    if any(tag in name for tag in ("_torchvision_", "_pytorch_")):
        stem = name.split("_imagenet")[0]
        arch = TV_ARCH.get(stem) or TV_ARCH.get(stem.replace("_", "-"))
        if arch:
            ds = "imagenet1kv2" if "imagenet1kv2" in name else "imagenet1k"
            return arch, "torchvision_instantiation.py", ds, 1000, (3, 224, 224)

    # Same ImageNet CNNs under third-party wrappers (CRAM / HALP / pruning-bench).
    if name.startswith("resnet50_imagenet"):
        return "resnet50", "torchvision_instantiation.py", "imagenet1k", 1000, (3, 224, 224)
    if name.startswith("resnet56_cifar10_dep_graph"):
        return "resnet56", "resnet_chenyaofo.py", "cifar-10", 10, (3, 32, 32)
    if name.startswith("resnet18_cifar100_pruning_bench"):
        return "resnet18_cifar", "resnet_cifar.py", "cifar-100", 100, (3, 32, 32)
    if name.startswith("vgg19_cifar100_dfpc_pre"):
        return "vgg19_bn_linear", "vgg_chenyaofo.py", "cifar-100", 100, (3, 32, 32)
    if name.startswith("vgg19_cifar100_dep_graph") or name.startswith("vgg19_cifar100_pruning_bench"):
        return "vgg19_bn", "vgg_repdistiller.py", "cifar-100", 100, (3, 32, 32)
    if name.startswith("mobilenet-v2_cifar10_dfpc"):
        return "mobilenet_v2_dfpc", "mobilenetv2_dfpc.py", "cifar-10", 10, (3, 32, 32)
    if name.startswith("mobilenet-v2_cifar100_dfpc"):
        return "mobilenet_v2_dfpc", "mobilenetv2_dfpc.py", "cifar-100", 100, (3, 32, 32)
    if name.startswith("convnext_tiny_") and "isomorphic" in name:
        return "convnext_tiny_1k", "convnext_facebook_research.py", "imagenet1k", 1000, (3, 224, 224)
    if name.startswith("convnext_small_") and "isomorphic" in name:
        return "convnext_small_1k", "convnext_facebook_research.py", "imagenet1k", 1000, (3, 224, 224)

    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--skip-huge", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    files = sorted(p.name for p in args.ckpt_root.iterdir() if p.is_file())
    mapped, unmapped = [], []
    for name in files:
        row = map_file(name)
        if row is None:
            unmapped.append(name)
        else:
            arch, script, ds, ncls, shape = row
            mapped.append({
                "file": name, "arch": arch, "script": script, "dataset": ds,
                "num_classes": ncls, "input_shape": list(shape),
            })

    print(f"files={len(files)} mapped={len(mapped)} unmapped={len(unmapped)}")
    for name in unmapped:
        print(f"  UNMAPPED {name}")

    results = {"mapped": mapped, "unmapped": unmapped, "load": []}
    if args.load:
        sys.path.insert(0, str(REPO))
        from tests.test_pruning import _init_static_conf  # noqa: WPS433
        _init_static_conf()
        import torch
        import src.utils as utils

        failures = []
        for item in mapped:
            name = item["file"]
            if args.skip_huge and HUGE.search(name):
                item["skipped"] = "huge"
                print(f"  SKIP huge {name}")
                continue
            ckpt = args.ckpt_root / name
            script = SCRIPT_ROOT / item["script"]
            ds = item["dataset"]
            ncls = item["num_classes"]
            shape = tuple(item["input_shape"])
            print(f"  LOAD {name} -> {item['arch']} {item['script']}", flush=True)
            try:
                model = utils.load_model_from_script(
                    item["arch"], ds, str(script), str(ckpt), {},
                    num_classes=ncls, input_shape=shape)
                model = model.cpu().eval()
                # MNIST grafting nets may be 1-channel; ImageNet 3x224
                c, h, w = shape
                y = model(torch.randn(1, c, h, w))
                item["ok"] = True
                item["out"] = list(y.shape)
                item["params"] = int(sum(p.numel() for p in model.parameters()))
                print(f"    ok out={tuple(y.shape)} params={item['params']}")
                del model
            except Exception as exc:
                item["ok"] = False
                item["error"] = f"{type(exc).__name__}: {exc}"
                failures.append((name, item["error"]))
                print(f"    FAIL {item['error']}")
        print(f"load failures={len(failures)}/{len(mapped)}")
        results["load_failures"] = [{"file": n, "error": e} for n, e in failures]

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
