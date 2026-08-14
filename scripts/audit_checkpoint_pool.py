#!/usr/bin/env python3
"""CPU smoke-load of RepVGG / akamaster (and optional catalog JSONs).

The 287-file folder is not an agent database. This script only checks that
weights load into the matching instantiation script so a future train/eval
split can cite every CNN checkpoint.

    python scripts/audit_checkpoint_pool.py
    python scripts/audit_checkpoint_pool.py --catalogs configs/database_offline_train.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.test_pruning import _init_static_conf  # noqa: E402
_init_static_conf()

import src.utils as utils  # noqa: E402

CKPT_ROOT = Path("/home/paretsky/spectra_pretrained_networks")
SCRIPT_ROOT = Path("/home/paretsky/spectra_models_instantiation")

AKAMASTER = [
    ("resnet20_cifar10_akamaster_91.73_0.27.th", "resnet20"),
    ("resnet32_cifar10_akamaster_92.63_0.46.th", "resnet32"),
    ("resnet44_cifar10_akamaster_93.1_0.66.th", "resnet44"),
    ("resnet56_cifar10_akamaster_93.39_0.85.th", "resnet56"),
    ("resnet110_cifar10_akamaster_93.68_1.7.th", "resnet110"),
]
REPVGG = [
    ("repvgga0_cifar10_chenyaofo_94.39_7.84_978.16.pt", "repvgg_a0", 10),
    ("repvgga1_cifar10_chenyaofo_94.89_12.82_1702.66.pt", "repvgg_a1", 10),
    ("repvgga2_cifar10_chenyaofo_94.98_26.82_3700.2.pt", "repvgg_a2", 10),
    ("repvgga0_cifar100_chenyaofo_75.22_7.96_978.38.pt", "repvgg_a0", 100),
    ("repvgga1_cifar100_chenyaofo_76.12_12.94_1702.88.pt", "repvgg_a1", 100),
    ("repvgga2_cifar100_chenyaofo_77.18_26.94_3700.44.pt", "repvgg_a2", 100),
]


def _try_load(arch, script, ckpt, dataset, ncls, shape):
    model = utils.load_model_from_script(
        arch, dataset, str(script), str(ckpt), {}, num_classes=ncls, input_shape=shape)
    model = model.cpu().eval()
    y = model(torch.randn(1, *shape))
    nparam = sum(p.numel() for p in model.parameters())
    return tuple(y.shape), nparam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT)
    parser.add_argument("--script-root", type=Path, default=SCRIPT_ROOT)
    parser.add_argument("--catalogs", nargs="*", type=Path, default=[])
    parser.add_argument("--include-1202", action="store_true")
    args = parser.parse_args()

    failures = []

    print("=== akamaster option-A ===")
    script = args.script_root / "resnet_akamaster.py"
    files = list(AKAMASTER)
    if args.include_1202:
        files.append(("resnet1202_cifar10_akamaster_93.82_19.4.th", "resnet1202"))
    for fname, arch in files:
        ckpt = args.ckpt_root / fname
        print(f"  {fname}")
        try:
            shape, nparam = _try_load(arch, script, ckpt, "cifar-10", 10, (3, 32, 32))
            print(f"    ok  out={shape}  params={nparam}")
        except Exception as exc:
            print(f"    FAIL {type(exc).__name__}: {exc}")
            failures.append((fname, str(exc)))

    print("=== RepVGG chenyaofo ===")
    script = args.script_root / "repvgg_chenyaofo.py"
    for fname, arch, ncls in REPVGG:
        ckpt = args.ckpt_root / fname
        dataset = "cifar-10" if ncls == 10 else "cifar-100"
        print(f"  {fname}")
        try:
            shape, nparam = _try_load(arch, script, ckpt, dataset, ncls, (3, 32, 32))
            print(f"    ok  out={shape}  params={nparam}")
        except Exception as exc:
            print(f"    FAIL {type(exc).__name__}: {exc}")
            failures.append((fname, str(exc)))

    for catalog in args.catalogs:
        data = json.loads(catalog.read_text(encoding="utf-8"))
        print(f"=== catalog {catalog} n={len(data)} ===")
        for ckpt, row in data.items():
            arch, script, dataset = row[0], row[1], row[2]
            print(f"  {Path(ckpt).name}")
            try:
                name, _ = utils.parse_dataset_spec(dataset)
                canonical = utils.canonical_dataset_name(name)
                ncls = {"cifar-100": 100, "places365": 365,
                        "imagenet1k": 1000, "imagenet1kv2": 1000}.get(canonical, 10)
                shape = (3, 224, 224) if "imagenet" in canonical else (3, 32, 32)
                out, nparam = _try_load(arch, script, ckpt, dataset, ncls, shape)
                print(f"    ok  out={out}  params={nparam}")
            except Exception as exc:
                print(f"    FAIL {type(exc).__name__}: {exc}")
                failures.append((ckpt, str(exc)))

    if failures:
        print(f"\n{len(failures)} failures")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        sys.exit(1)
    print("audit ok")


if __name__ == "__main__":
    main()
