#!/usr/bin/env python3
"""
Train a CNN and save it with SPECTRA's checkpoint filename convention.

    {arch}[-widthW]_{dataset}_{source}_{acc}_{paramsM}_{flopsM}.pt

Intended for the missing thesis-pool cells (DenseNet-CIFAR, SVHN, Fashion-MNIST).
Runs on a SLURM GPU node; do not use the Windows laptop.

Example:

    python scripts/train_pretrained_checkpoint.py \\
        --arch densenet40 --script /home/paretsky/spectra_models_instantiation/densenet_cifar.py \\
        --dataset cifar-10 --source densenet-cifar --epochs 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.utils as utils  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True)
    p.add_argument("--script", required=True, type=Path)
    p.add_argument("--dataset", required=True,
                   help="cifar-10 / cifar-100 / svhn / fashion-mnist")
    p.add_argument("--source", required=True,
                   help="Filename source tag, e.g. densenet-cifar or thin-res-net")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--milestones", type=int, nargs="*", default=[100, 150])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-root", type=Path, default=Path("/home/paretsky/spectra_pretrained_networks"))
    p.add_argument("--manifest", type=Path,
                   default=Path("/home/paretsky/pretrained_pool_manifest.json"))
    p.add_argument("--append-catalogs", type=Path, nargs="*", default=None,
                   help="JSON catalogs to append this net to on success")
    return p.parse_args()


def dataset_token(name: str) -> str:
    return {
        "cifar-10": "cifar10", "cifar-100": "cifar100",
        "fashion-mnist": "fashionmnist", "svhn": "svhn", "mnist": "mnist",
    }.get(name, name.replace("-", ""))


def dataset_spec(name: str):
    """Match the preprocessing SPECTRA will use when it later prunes this net."""
    if name == "fashion-mnist":
        return {"name": "fashion-mnist", "image_size": 32, "to_rgb": True}
    return name


def train_transform(name: str):
    """CIFAR-style aug for training; SPECTRA's unaugmented pipeline for the reported test acc."""
    name_or_path, options = utils.parse_dataset_spec(dataset_spec(name))
    eval_t = utils.build_transform(name_or_path, options)
    steps = []
    if options.get("to_rgb"):
        steps.append(transforms.Grayscale(num_output_channels=3))
    image_size = options.get("image_size")
    if image_size:
        size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        steps.append(transforms.Resize(size))
    steps.append(transforms.RandomCrop(32, padding=4))
    if name != "svhn":
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    mean, std = utils.DATASET_STATS.get(utils.canonical_dataset_name(name_or_path), utils.DEFAULT_STATS)
    if options.get("to_rgb") and len(mean) == 1:
        mean, std = mean * 3, std * 3
    steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps), eval_t


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


def instantiate(arch, script, num_classes, width):
    import importlib.util
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, arch)
    kwargs = {"num_classes": num_classes, "large_input": False}
    if width is not None:
        kwargs["width"] = width
    return fn(**kwargs)


def append_json(path: Path, ckpt_path: str, row: list):
    data = {}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    data[ckpt_path] = row
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"appended {ckpt_path} -> {path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("train_pretrained_checkpoint.py needs a CUDA GPU (BGU SLURM), not CPU.")

    canonical = utils.canonical_dataset_name(args.dataset)
    train_t, eval_t = train_transform(canonical)
    train_set, _ = utils.DATASET_BUILDERS[canonical](train_t)
    _, test_set = utils.DATASET_BUILDERS[canonical](eval_t)

    num_classes = 10
    if hasattr(train_set, "classes") and train_set.classes:
        num_classes = len(train_set.classes)
    elif canonical == "cifar-100":
        num_classes = 100
    model = instantiate(args.arch, args.script, num_classes, args.width).to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            opt.step()
            running += loss.item() * targets.size(0)
            n += targets.size(0)
        sched.step()
        acc = accuracy(model, test_loader, device)
        print(f"epoch {epoch+1:03d}/{args.epochs}  loss={running/max(n,1):.4f}  test_acc={acc:.2f}  "
              f"lr={sched.get_last_lr()[0]:.5f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    final_acc = accuracy(model, test_loader, device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops_m = utils.calc_flops(model, (3, 32, 32), device) / 1e6

    arch_token = args.arch.replace("_", "-")
    if args.width is not None:
        arch_token = f"{arch_token}-width{args.width}"
    fname = (f"{arch_token}_{dataset_token(canonical)}_{args.source}_"
             f"{final_acc:.2f}_{params_m:.3f}_{flops_m:.2f}.pt")
    args.ckpt_root.mkdir(parents=True, exist_ok=True)
    out_path = args.ckpt_root / fname
    torch.save(best_state, out_path)
    print(f"saved {out_path}  acc={final_acc:.2f} paramsM={params_m:.3f} flopsM={flops_m:.2f}")

    spec = dataset_spec(canonical)
    row = [args.arch, str(args.script), spec]
    if args.width is not None:
        row.append({"width": args.width})

    manifest = {}
    if args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest[str(out_path)] = {
        "row": row, "acc": round(final_acc, 2), "params_m": round(params_m, 3),
        "flops_m": round(flops_m, 2), "epochs": args.epochs, "seed": args.seed,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for catalog in args.append_catalogs or []:
        append_json(catalog, str(out_path), row)


if __name__ == "__main__":
    main()
