#!/usr/bin/env python3
"""CPU instantiate + dummy forward for V5 extra families.

Run on the cluster (spectra env). Does not prune, does not overlay src.

    python scripts/v5_catalog_compat_check.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
INST = REPO / "spectra_models_instantiation"

CANDIDATES = [
    ("wide_resnet.py", ["wrn_16_4", "wrn_16_8", "wrn_28_2", "wrn_28_10"]),
    ("preact_resnet.py", ["preact_resnet20", "preact_resnet32", "preact_resnet56"]),
    ("densenet_cifar.py", ["densenet40", "densenet100"]),
    ("resnet_chenyaofo.py", ["resnet20", "resnet32", "resnet44", "resnet56"]),
    ("vgg_chenyaofo.py", ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]),
    ("mobilenetv2_chenyaofo.py", ["mobilenet_v2x05", "mobilenet_v2x075", "mobilenet_v2x1", "mobilenet_v2x14"]),
    ("shufflenetv2_chenyaofo.py", ["shufflenetv2x05", "shufflenetv2x1", "shufflenetv2x15", "shufflenetv2x2"]),
    ("repvgg_chenyaofo.py", ["repvgg_a0", "repvgg_a1", "repvgg_a2"]),
]


def load_mod(name: str):
    path = INST / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    x = torch.randn(2, 3, 32, 32)
    failed = []
    for script, arches in CANDIDATES:
        mod = load_mod(script)
        for arch in arches:
            for ncls in (10, 100):
                try:
                    net = getattr(mod, arch)(num_classes=ncls, large_input=False)
                    net.eval()
                    with torch.no_grad():
                        y = net(x)
                    if tuple(y.shape) != (2, ncls):
                        raise RuntimeError(f"logits {tuple(y.shape)}")
                    nconv = sum(1 for m in net.modules() if isinstance(m, torch.nn.Conv2d))
                    print(f"ok {script}:{arch} c{ncls} convs={nconv} params={sum(p.numel() for p in net.parameters())}")
                except Exception as exc:
                    print(f"FAIL {script}:{arch} c{ncls}: {exc}")
                    failed.append((script, arch, ncls, str(exc)))
    if failed:
        print(f"{len(failed)} failures")
        sys.exit(1)
    print("all factories forward on 32x32")


if __name__ == "__main__":
    main()
