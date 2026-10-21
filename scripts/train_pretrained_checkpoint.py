#!/usr/bin/env python3
"""
Train a CNN and save it with SPECTRA's checkpoint filename convention.

    {arch}[-widthW]_{dataset}_{source}_{acc}_{paramsM}_{flopsM}.pt

Intended for the missing thesis-pool cells (DenseNet-CIFAR, SVHN, Fashion-MNIST) and, since v29 (21 Oct),
the wave-3 Tiny-ImageNet checkpoints. Runs on a SLURM GPU node; do not use the Windows laptop.

Example:

    python scripts/train_pretrained_checkpoint.py \\
        --arch densenet40 --script /home/paretsky/spectra_models_instantiation/densenet_cifar.py \\
        --dataset cifar-10 --source densenet-cifar --epochs 200

Tiny-ImageNet (wave 3; ``--dataset tiny-imagenet``, factories in ``spectra_models_instantiation/tiny_imagenet_w3.py``:
``resnet18`` / ``vgg16_bn`` / ``mobilenet_v2`` at 64 px, 200 classes). The recipe proposed for science's GO, every
knob explicit so the catalog row records it:

    python scripts/train_pretrained_checkpoint.py \\
        --arch resnet18 --script /home/paretsky/spectra_models_instantiation/tiny_imagenet_w3.py \\
        --dataset tiny-imagenet --source spectra-w3 --select last \\
        --epochs 100 --batch-size 256 --lr 0.1 --schedule cosine --warmup-epochs 2 --weight-decay 5e-4 --workers 8

SGD (momentum 0.9, Nesterov), lr 0.1 with a 2-epoch linear warmup then cosine to 0 over 100 epochs, weight decay
5e-4, batch 256, label smoothing off, train aug RandomCrop(64, pad 8) + flip (exactly ``utils.build_transform``'s
Tiny recipe under ``SPECTRA_FT_AUG=1``, so the eval line fine-tunes as the net was trained), the reported accuracy on
the labelled val split (SPECTRA's TEST for this dataset) with the plain eval transform, ``--select last`` because every
wave-3 net may be a held-out test network. Defaults keep the old behaviour (multistep 100/150, wd 1e-4, batch 128).

``--smoke-steps N --allow-cpu`` runs N optimizer steps (batch ``--batch-size``) and stops before any epoch, eval or
save: the login-node check that a factory, the loader and the recipe fit together (no GPU, nothing written).
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
                   help="cifar-10 / cifar-100 / svhn / fashion-mnist / tiny-imagenet")
    p.add_argument("--source", required=True,
                   help="Filename source tag, e.g. densenet-cifar or thin-res-net")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--milestones", type=int, nargs="*", default=[100, 150])
    p.add_argument("--schedule", choices=("multistep", "cosine"), default="multistep",
                   help="multistep (the old default: x0.1 at --milestones) or cosine to 0 over --epochs after "
                        "--warmup-epochs of linear warmup (the Tiny-ImageNet recipe)")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear lr warmup epochs before the cosine schedule (ignored under multistep)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Cross-entropy label smoothing (default 0 = off, as before)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="Run this many optimizer steps and stop before any epoch, eval or save (a loader / recipe check)")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Let --smoke-steps run without a GPU (the login-node check); training still needs CUDA")
    p.add_argument("--select", choices=("best", "last"), default="best",
                   help="Which epoch's weights to save. 'best' (the old default) picks the epoch with the best TEST "
                        "accuracy, which selects on the test set; 'last' keeps the final epoch (G2's keep-last), for "
                        "any network that may later be a held-out test network.")
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
        "tiny-imagenet": "tinyimagenet",
    }.get(name, name.replace("-", ""))


def crop_recipe(canonical: str):
    """``(crop, pad)`` of the train-time RandomCrop: 32 / 4 for the 32-px datasets, 64 / 8 for Tiny-ImageNet (the
    same numbers ``utils.build_transform`` uses under SPECTRA_FT_AUG / SPECTRA_FT_AUG_HOLDOUT)."""
    if canonical in utils._FT_AUG_64PX:
        return utils.TinyImageNet.SIZE, 8
    return 32, 4


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
    crop, pad = crop_recipe(utils.canonical_dataset_name(name_or_path))
    steps.append(transforms.RandomCrop(crop, padding=pad))
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
    fn, _ = utils.resolve_instantiation_func(module, arch)
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


def make_scheduler(opt, args):
    """``MultiStepLR`` (the old default) or linear warmup then cosine to 0, stepped once per epoch."""
    if args.schedule == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=0.1)
    import math
    warmup, total = max(0, int(args.warmup_epochs)), max(1, int(args.epochs))

    def factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / float(warmup)
        progress = (epoch - warmup) / float(max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke = int(args.smoke_steps) > 0
    if device.type != "cuda" and not (smoke and args.allow_cpu):
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
    # The dataset's own sample shape: (3, 32, 32) for the 32-px datasets as before, (3, 64, 64) for Tiny-ImageNet.
    input_shape = tuple(int(d) for d in test_set[0][0].shape)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = make_scheduler(opt, args)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))

    best_acc = -1.0
    best_state = None
    steps_done = 0
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
            steps_done += 1
            if smoke and steps_done >= int(args.smoke_steps):
                params_m = sum(p.numel() for p in model.parameters()) / 1e6
                flops_m = utils.calc_flops(model, input_shape, device) / 1e6
                print(f"smoke: {steps_done} steps on {device.type} | {args.arch} on {canonical} "
                      f"(n_train={len(train_set)}, n_test={len(test_set)}, classes={num_classes}) | "
                      f"batch {tuple(images.shape)} -> logits {tuple(logits.shape)} | loss={loss.item():.4f} | "
                      f"paramsM={params_m:.3f} flopsM={flops_m:.2f} | {args.schedule} lr={args.lr} "
                      f"wd={args.weight_decay} ls={args.label_smoothing} | nothing saved", flush=True)
                return
        sched.step()
        acc = accuracy(model, test_loader, device)
        print(f"epoch {epoch+1:03d}/{args.epochs}  loss={running/max(n,1):.4f}  test_acc={acc:.2f}  "
              f"lr={sched.get_last_lr()[0]:.5f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    chosen = best_state if args.select == "best" else last_state
    print(f"select={args.select}: best test-epoch acc {best_acc:.2f}"
          f"{' (logged only; the saved weights are the last epoch)' if args.select == 'last' else ''}", flush=True)
    model.load_state_dict(chosen)
    model.to(device)
    final_acc = accuracy(model, test_loader, device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops_m = utils.calc_flops(model, input_shape, device) / 1e6

    arch_token = args.arch.replace("_", "-")
    if args.width is not None:
        arch_token = f"{arch_token}-width{args.width}"
    fname = (f"{arch_token}_{dataset_token(canonical)}_{args.source}_"
             f"{final_acc:.2f}_{params_m:.3f}_{flops_m:.2f}.pt")
    args.ckpt_root.mkdir(parents=True, exist_ok=True)
    out_path = args.ckpt_root / fname
    torch.save(chosen, out_path)
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
        "flops_m": round(flops_m, 2), "epochs": args.epochs, "seed": args.seed, "select": args.select,
    }
    if args.schedule != "multistep" or args.label_smoothing or canonical in utils._FT_AUG_64PX:
        # v29: the wave-3 recipe is recorded beside the row (the earlier rows keep their exact keys)
        manifest[str(out_path)]["recipe"] = {
            "schedule": args.schedule, "warmup_epochs": args.warmup_epochs, "lr": args.lr,
            "weight_decay": args.weight_decay, "batch_size": args.batch_size,
            "label_smoothing": args.label_smoothing, "input_shape": list(input_shape),
            "train_aug": f"RandomCrop({crop_recipe(canonical)[0]}, pad {crop_recipe(canonical)[1]})"
                         + ("" if canonical == "svhn" else "+flip"),
        }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for catalog in args.append_catalogs or []:
        append_json(catalog, str(out_path), row)


if __name__ == "__main__":
    main()
