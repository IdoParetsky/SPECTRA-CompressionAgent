#!/usr/bin/env python3
"""Fetch remaining chenyaofo CIFAR hub weights into SPECTRA's checkpoint pool.

Does **not** train. Downloads public CIFAR-10/100 zoos (ResNet / VGG / MobileNetV2 /
ShuffleNetV2 / RepVGG) that our instantiation files already know, evaluates TEST
acc, writes the SPECTRA filename convention.

Run on a BGU GPU node (eval is cheap; hub download is the wait):

    python scripts/fetch_chenyaofo_hub.py --dry-run
    python scripts/fetch_chenyaofo_hub.py --ckpt-root /home/paretsky/spectra_pretrained_networks

Do not overlay leap ``src/``. Instantiation lives in ``spectra_models_instantiation/``.
Skip ImageNet. Skip ViT. Skip any hub name we already have a SPECTRA ckpt for.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.utils as utils  # noqa: E402

LEAP_INST = Path("/home/paretsky/spectra_models_instantiation")
REPO_INST = REPO / "spectra_models_instantiation"
SCRIPT_ROOT = LEAP_INST if LEAP_INST.is_dir() else REPO_INST


def ensure_leap_new_factories() -> None:
    """Copy additive factories into the leap instantiation tree. Never overwrite."""
    if not LEAP_INST.is_dir():
        return
    for name in ("wide_resnet.py", "preact_resnet.py"):
        src, dst = REPO_INST / name, LEAP_INST / name
        if src.is_file() and not dst.exists():
            dst.write_bytes(src.read_bytes())
            print(f"copied {name} -> {dst}")


HUB_REPO = "chenyaofo/pytorch-cifar-models"

# hub_name -> (spectra_arch, instantiation file, dataset)
HUB = {
    "cifar10_resnet20": ("resnet20", "resnet_chenyaofo.py", "cifar-10"),
    "cifar10_resnet32": ("resnet32", "resnet_chenyaofo.py", "cifar-10"),
    "cifar10_resnet44": ("resnet44", "resnet_chenyaofo.py", "cifar-10"),
    "cifar10_resnet56": ("resnet56", "resnet_chenyaofo.py", "cifar-10"),
    "cifar10_vgg11_bn": ("vgg11_bn", "vgg_chenyaofo.py", "cifar-10"),
    "cifar10_vgg13_bn": ("vgg13_bn", "vgg_chenyaofo.py", "cifar-10"),
    "cifar10_vgg16_bn": ("vgg16_bn", "vgg_chenyaofo.py", "cifar-10"),
    "cifar10_vgg19_bn": ("vgg19_bn", "vgg_chenyaofo.py", "cifar-10"),
    "cifar10_mobilenetv2_x0_5": ("mobilenet_v2x05", "mobilenetv2_chenyaofo.py", "cifar-10"),
    "cifar10_mobilenetv2_x0_75": ("mobilenet_v2x075", "mobilenetv2_chenyaofo.py", "cifar-10"),
    "cifar10_mobilenetv2_x1_0": ("mobilenet_v2x1", "mobilenetv2_chenyaofo.py", "cifar-10"),
    "cifar10_mobilenetv2_x1_4": ("mobilenet_v2x14", "mobilenetv2_chenyaofo.py", "cifar-10"),
    "cifar10_shufflenetv2_x0_5": ("shufflenetv2x05", "shufflenetv2_chenyaofo.py", "cifar-10"),
    "cifar10_shufflenetv2_x1_0": ("shufflenetv2x1", "shufflenetv2_chenyaofo.py", "cifar-10"),
    "cifar10_shufflenetv2_x1_5": ("shufflenetv2x15", "shufflenetv2_chenyaofo.py", "cifar-10"),
    "cifar10_shufflenetv2_x2_0": ("shufflenetv2x2", "shufflenetv2_chenyaofo.py", "cifar-10"),
    "cifar10_repvgg_a0": ("repvgg_a0", "repvgg_chenyaofo.py", "cifar-10"),
    "cifar10_repvgg_a1": ("repvgg_a1", "repvgg_chenyaofo.py", "cifar-10"),
    "cifar10_repvgg_a2": ("repvgg_a2", "repvgg_chenyaofo.py", "cifar-10"),
    "cifar100_resnet20": ("resnet20", "resnet_chenyaofo.py", "cifar-100"),
    "cifar100_resnet32": ("resnet32", "resnet_chenyaofo.py", "cifar-100"),
    "cifar100_resnet44": ("resnet44", "resnet_chenyaofo.py", "cifar-100"),
    "cifar100_resnet56": ("resnet56", "resnet_chenyaofo.py", "cifar-100"),
    "cifar100_vgg11_bn": ("vgg11_bn", "vgg_chenyaofo.py", "cifar-100"),
    "cifar100_vgg13_bn": ("vgg13_bn", "vgg_chenyaofo.py", "cifar-100"),
    "cifar100_vgg16_bn": ("vgg16_bn", "vgg_chenyaofo.py", "cifar-100"),
    "cifar100_vgg19_bn": ("vgg19_bn", "vgg_chenyaofo.py", "cifar-100"),
    "cifar100_mobilenetv2_x0_5": ("mobilenet_v2x05", "mobilenetv2_chenyaofo.py", "cifar-100"),
    "cifar100_mobilenetv2_x0_75": ("mobilenet_v2x075", "mobilenetv2_chenyaofo.py", "cifar-100"),
    "cifar100_mobilenetv2_x1_0": ("mobilenet_v2x1", "mobilenetv2_chenyaofo.py", "cifar-100"),
    "cifar100_mobilenetv2_x1_4": ("mobilenet_v2x14", "mobilenetv2_chenyaofo.py", "cifar-100"),
    "cifar100_shufflenetv2_x0_5": ("shufflenetv2x05", "shufflenetv2_chenyaofo.py", "cifar-100"),
    "cifar100_shufflenetv2_x1_0": ("shufflenetv2x1", "shufflenetv2_chenyaofo.py", "cifar-100"),
    "cifar100_shufflenetv2_x1_5": ("shufflenetv2x15", "shufflenetv2_chenyaofo.py", "cifar-100"),
    "cifar100_shufflenetv2_x2_0": ("shufflenetv2x2", "shufflenetv2_chenyaofo.py", "cifar-100"),
    "cifar100_repvgg_a0": ("repvgg_a0", "repvgg_chenyaofo.py", "cifar-100"),
    "cifar100_repvgg_a1": ("repvgg_a1", "repvgg_chenyaofo.py", "cifar-100"),
    "cifar100_repvgg_a2": ("repvgg_a2", "repvgg_chenyaofo.py", "cifar-100"),
}


def already_have(ckpt_root: Path, arch: str, dataset: str) -> Path | None:
    token = {"cifar-10": "cifar10", "cifar-100": "cifar100"}[dataset]
    slugs = {arch, arch.replace("_", "-"), arch.replace("_", "")}
    for p in ckpt_root.glob("*.pt"):
        name = p.name.lower()
        if token not in name or "chenyaofo" not in name:
            continue
        for slug in slugs:
            s = slug.lower()
            if name.startswith(s + "_") or name.startswith(s + "-"):
                return p
    return None


def instantiate_ours(arch: str, script: str, num_classes: int):
    import importlib.util
    path = SCRIPT_ROOT / script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, arch)(num_classes=num_classes, large_input=False)


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            pred = model(images).argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / max(total, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-root", type=Path, default=Path("/home/paretsky/spectra_pretrained_networks"))
    p.add_argument("--manifest", type=Path, default=Path("/home/paretsky/pretrained_pool_manifest.json"))
    p.add_argument("--only", nargs="*", default=None, help="Subset of hub names")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wanted = args.only or sorted(HUB)
    args.ckpt_root.mkdir(parents=True, exist_ok=True)
    ensure_leap_new_factories()

    for hub_name in wanted:
        if hub_name not in HUB:
            print(f"skip unknown {hub_name}")
            continue
        arch, script, dataset = HUB[hub_name]
        existing = already_have(args.ckpt_root, arch, dataset)
        if existing is not None and not args.force:
            print(f"have {hub_name} -> {existing.name}")
            continue
        print(f"fetch {hub_name} -> {arch} {dataset}")
        if args.dry_run:
            continue
        num_classes = 100 if dataset == "cifar-100" else 10
        hub_model = torch.hub.load(HUB_REPO, hub_name, pretrained=True)
        ours = instantiate_ours(arch, script, num_classes)
        missing, unexpected = ours.load_state_dict(hub_model.state_dict(), strict=False)
        if missing:
            raise SystemExit(f"{hub_name}: missing keys {missing[:8]}")
        ours.to(device)
        name_or_path, options = utils.parse_dataset_spec(dataset)
        eval_t = utils.build_transform(name_or_path, options)
        _, test_set = utils.DATASET_BUILDERS[utils.canonical_dataset_name(dataset)](eval_t)
        loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
        acc = accuracy(ours, loader, device)
        params_m = sum(p.numel() for p in ours.parameters()) / 1e6
        flops_m = utils.calc_flops(ours, (3, 32, 32), device) / 1e6
        fname = (f"{arch.replace('_', '-')}_{dataset.replace('-', '')}_chenyaofo_"
                 f"{acc:.2f}_{params_m:.3f}_{flops_m:.2f}.pt")
        out = args.ckpt_root / fname
        torch.save({k: v.detach().cpu() for k, v in ours.state_dict().items()}, out)
        print(f"saved {out} acc={acc:.2f} unexpected={list(unexpected)[:4]}")
        manifest = {}
        if args.manifest.is_file():
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest[str(out)] = {
            "hub": hub_name, "row": [arch, str(SCRIPT_ROOT / script), dataset],
            "acc": round(acc, 2), "params_m": round(params_m, 3), "flops_m": round(flops_m, 2),
        }
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
