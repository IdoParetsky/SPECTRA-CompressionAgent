#!/usr/bin/env python3
"""
Build SPECTRA train/eval JSON catalogs from the pickled checkpoint folder.

The 280-file folder is *not* the agent database. SPECTRA only trains on networks listed
in ``--database`` JSON. This script writes thesis-shaped catalogs that:

* keep the original 6-net / 3-net ``initial_*`` files untouched (running jobs)
* add a multi-family CIFAR catalog (ResNet + VGG + MobileNet train; ShuffleNet held out)
* add C10-only / C100-only generic slices for the recovery track
* optionally emit a "full competent" catalog (every well-trained thin-ResNet + chenyaofo)

Filename convention (already used in ``spectra_pretrained_networks``):

    {arch}_{dataset}_{source}_{acc}_{paramsM}_{flopsM}.{pt,pth,th}

Run on the cluster:

    python scripts/build_network_catalog.py --write-cluster
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CKPT_ROOT = "/home/paretsky/spectra_pretrained_networks"
SCRIPT_ROOT = "/home/paretsky/spectra_models_instantiation"
REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

DATASET_JSON = {"cifar10": "cifar-10", "cifar100": "cifar-100", "svhn": "svhn",
                "fashionmnist": "fashion-mnist", "mnist": "mnist"}


def _entry(fname: str, arch: str, script: str, dataset: str, kwargs=None) -> tuple:
    path = f"{CKPT_ROOT}/{fname}"
    script_path = f"{SCRIPT_ROOT}/{script}"
    row = [arch, script_path, dataset]
    if kwargs:
        row.append(kwargs)
    return path, row


def _must_exist(fname: str) -> str:
    if not Path(f"{CKPT_ROOT}/{fname}").is_file():
        raise FileNotFoundError(f"missing checkpoint: {CKPT_ROOT}/{fname}")
    return fname


# Hand-curated: competent accuracy, instantiable, structurally distinct.
# Train = families the agent is allowed to memorise. Eval = unseen width / depth / family.
GENERIC_TRAIN = [
    _entry("resnet20-width10_cifar10_thin-res-net_91.90_0.107_16.55.pt",
           "resnet20", "thin_res_net.py", "cifar-10"),
    _entry("resnet56-width6_cifar10_thin-res-net_92.88_0.122_18.60.pt",
           "resnet56", "thin_res_net.py", "cifar-10"),
    _entry("resnet32_cifar10_chenyaofo_93.53_047_138.24.pt",
           "resnet32", "resnet_chenyaofo.py", "cifar-10"),
    _entry("vgg16_bn_cifar10_chenyaofo_94.16_15.25_627.46.pt",
           "vgg16_bn", "vgg_chenyaofo.py", "cifar-10"),
    _entry("mobilenet-v2x1_cifar10_chenyaofo_93.79_2.24_175.96.pt",
           "mobilenet_v2x1", "mobilenetv2_chenyaofo.py", "cifar-10"),
    _entry("resnet20-width13_cifar100_thin-res-net_69.95_0.185_27.67.pt",
           "resnet20", "thin_res_net.py", "cifar-100"),
    _entry("resnet56-width9_cifar100_thin-res-net_73.05_0.275_41.13.pt",
           "resnet56", "thin_res_net.py", "cifar-100"),
    _entry("resnet56-width15_cifar100_thin-res-net_78.46_0.758_112.59.pt",
           "resnet56", "thin_res_net.py", "cifar-100"),
    _entry("resnet44_cifar100_chenyaofo_71.63_0.67_184.88.pt",
           "resnet44", "resnet_chenyaofo.py", "cifar-100"),
    _entry("vgg11_bn_cifar100_chenyaofo_70.78_9.8_306.68.pt",
           "vgg11_bn", "vgg_chenyaofo.py", "cifar-100"),
    _entry("mobilenet-v2x1_cifar100_chenyaofo_74.2_2.35_176.18.pt",
           "mobilenet_v2x1", "mobilenetv2_chenyaofo.py", "cifar-100"),
]

GENERIC_EVAL = [
    _entry("resnet20-width2_cifar10_thin-res-net_64.79_0.005_0.79.pt",
           "resnet20", "thin_res_net.py", "cifar-10"),
    _entry("resnet56-width4_cifar10_thin-res-net_88.80_0.054_8.49.pt",
           "resnet56", "thin_res_net.py", "cifar-10"),
    _entry("resnet20-width8_cifar100_thin-res-net_60.09_0.072_10.72.pt",
           "resnet20", "thin_res_net.py", "cifar-100"),
    _entry("shufflenetv2x1_cifar10_chenyaofo_92.98_1.26_90.pt",
           "shufflenetv2x1", "shufflenetv2_chenyaofo.py", "cifar-10"),
    _entry("vgg19_bn_cifar10_chenyaofo_93.91_20.57_797.32.pt",
           "vgg19_bn", "vgg_chenyaofo.py", "cifar-10"),
    _entry("shufflenetv2x1_cifar100_chenyaofo_72.39_1.36_90.18.pt",
           "shufflenetv2x1", "shufflenetv2_chenyaofo.py", "cifar-100"),
]

CIFAR10_GENERIC_TRAIN = [e for e in GENERIC_TRAIN if e[1][2] == "cifar-10"]
CIFAR10_GENERIC_EVAL = [e for e in GENERIC_EVAL if e[1][2] == "cifar-10"]
CIFAR100_GENERIC_TRAIN = [e for e in GENERIC_TRAIN if e[1][2] == "cifar-100"]
CIFAR100_GENERIC_EVAL = [e for e in GENERIC_EVAL if e[1][2] == "cifar-100"]

# Exact 6-net recovery slice, split by dataset. --datasets used to lazy-load the
# other half of initial_* so "C10-only" jobs were still mixed.
THIN_C10_TRAIN = [
    _entry("resnet20-width10_cifar10_thin-res-net_91.90_0.107_16.55.pt",
           "resnet20", "thin_res_net.py", "cifar-10"),
    _entry("resnet56-width6_cifar10_thin-res-net_92.88_0.122_18.60.pt",
           "resnet56", "thin_res_net.py", "cifar-10"),
    _entry("resnet20-width3_cifar10_thin-res-net_76.18_0.010_1.65.pt",
           "resnet20", "thin_res_net.py", "cifar-10"),
]
THIN_C10_EVAL = [
    _entry("resnet20-width2_cifar10_thin-res-net_64.79_0.005_0.79.pt",
           "resnet20", "thin_res_net.py", "cifar-10"),
    _entry("resnet56-width4_cifar10_thin-res-net_88.80_0.054_8.49.pt",
           "resnet56", "thin_res_net.py", "cifar-10"),
]
THIN_C100_TRAIN = [
    _entry("resnet20-width13_cifar100_thin-res-net_69.95_0.185_27.67.pt",
           "resnet20", "thin_res_net.py", "cifar-100"),
    _entry("resnet56-width9_cifar100_thin-res-net_73.05_0.275_41.13.pt",
           "resnet56", "thin_res_net.py", "cifar-100"),
    _entry("resnet56-width15_cifar100_thin-res-net_78.46_0.758_112.59.pt",
           "resnet56", "thin_res_net.py", "cifar-100"),
]
THIN_C100_EVAL = [
    _entry("resnet20-width8_cifar100_thin-res-net_60.09_0.072_10.72.pt",
           "resnet20", "thin_res_net.py", "cifar-100"),
]


def _to_dict(pairs):
    out = {}
    for path, row in pairs:
        out[path] = row
    return out


def _scale_token(raw: str) -> str:
    """'0.5' -> '05', '0.75' -> '075', '1' -> '1', '1.4' -> '14', '1.5' -> '15', '2' -> '2'."""
    if "." in raw:
        return raw.replace(".", "")
    return raw


def discover_competent(ckpt_root: Path):
    """Every thin-ResNet / chenyaofo CIFAR net that looks well-trained and instantiable."""
    entries = []
    for path in sorted(ckpt_root.iterdir()):
        name = path.name
        acc_m = re.search(r"_(?:thin-res-net|chenyaofo)_(\d+(?:\.\d+)?)_", name)
        acc = float(acc_m.group(1)) if acc_m else None

        if "thin-res-net" in name:
            m = re.match(r"(resnet\d+)-width(\d+)_(cifar10|cifar100)_", name)
            if not m:
                continue
            arch, width, ds = m.group(1), int(m.group(2)), m.group(3)
            if ds == "cifar10" and (acc is None or acc < 88 or width < 6):
                continue
            if ds == "cifar100" and (acc is None or acc < 65 or width < 8):
                continue
            entries.append(_entry(name, arch, "thin_res_net.py", DATASET_JSON[ds]))
            continue

        if "_chenyaofo_" not in name:
            continue
        m = re.match(r"(resnet\d+)_(cifar10|cifar100)_chenyaofo_", name)
        if m:
            if (m.group(2) == "cifar10" and acc is not None and acc < 88) or (
                    m.group(2) == "cifar100" and acc is not None and acc < 65):
                continue
            entries.append(_entry(name, m.group(1), "resnet_chenyaofo.py", DATASET_JSON[m.group(2)]))
            continue
        m = re.match(r"(vgg\d+_bn)_(cifar10|cifar100)_chenyaofo_", name)
        if m:
            entries.append(_entry(name, m.group(1), "vgg_chenyaofo.py", DATASET_JSON[m.group(2)]))
            continue
        m = re.match(r"mobilenet-v2x([0-9.]+)_(cifar10|cifar100)_chenyaofo_", name)
        if m:
            arch = "mobilenet_v2x" + _scale_token(m.group(1))
            entries.append(_entry(name, arch, "mobilenetv2_chenyaofo.py", DATASET_JSON[m.group(2)]))
            continue
        m = re.match(r"shufflenetv2x([0-9.]+)_(cifar10|cifar100)_chenyaofo_", name)
        if m:
            arch = "shufflenetv2x" + _scale_token(m.group(1))
            entries.append(_entry(name, arch, "shufflenetv2_chenyaofo.py", DATASET_JSON[m.group(2)]))
            continue
    return entries


def write_json(path: Path, pairs, check_files: bool):
    if check_files:
        missing = [p for p, _ in pairs if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError("catalog references missing checkpoints:\n  " + "\n  ".join(missing))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_dict(pairs), fh, indent=2)
        fh.write("\n")
    print(f"wrote {path}  n={len(pairs)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", type=str, default=CKPT_ROOT)
    parser.add_argument("--write-repo", action="store_true", help="Write configs/ in this repo")
    parser.add_argument("--write-cluster", action="store_true",
                        help="Write /home/paretsky/*.json (does not touch initial_*)")
    parser.add_argument("--include-full", action="store_true",
                        help="Also emit the large competent catalog")
    args = parser.parse_args()

    catalogs = {
        "database_generic_cifar.json": GENERIC_TRAIN,
        "input_generic_heldout.json": GENERIC_EVAL,
        "database_cifar10_generic.json": CIFAR10_GENERIC_TRAIN,
        "input_cifar10_generic.json": CIFAR10_GENERIC_EVAL,
        "database_cifar100_generic.json": CIFAR100_GENERIC_TRAIN,
        "input_cifar100_generic.json": CIFAR100_GENERIC_EVAL,
        "database_c10_thin.json": THIN_C10_TRAIN,
        "input_c10_thin.json": THIN_C10_EVAL,
        "database_c100_thin.json": THIN_C100_TRAIN,
        "input_c100_thin.json": THIN_C100_EVAL,
    }
    if args.include_full:
        catalogs["database_generic_cifar_full.json"] = discover_competent(Path(args.ckpt_root))

    check = Path(args.ckpt_root).is_dir()
    if args.write_repo:
        for name, pairs in catalogs.items():
            write_json(REPO_CONFIGS / name, pairs, check_files=False)
    if args.write_cluster:
        cluster_root = Path("/home/paretsky")
        for name, pairs in catalogs.items():
            write_json(cluster_root / name, pairs, check_files=check)
    if not args.write_repo and not args.write_cluster:
        for name, pairs in catalogs.items():
            print(f"{name}: {len(pairs)} nets")
            for path, row in pairs:
                print(f"  {Path(path).name}  {row[0]}  {row[2]}")


if __name__ == "__main__":
    main()
