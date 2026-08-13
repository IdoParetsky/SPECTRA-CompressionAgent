#!/usr/bin/env python3
"""Instantiate every network in a SPECTRA catalog JSON and run a 0.8 prune on one layer."""
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

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src.NetworkEnv import prune_current_model  # noqa: E402
import src.utils as utils  # noqa: E402


def num_classes_for(dataset_spec) -> int:
    name, _ = utils.parse_dataset_spec(dataset_spec)
    canonical = utils.canonical_dataset_name(name)
    return {"cifar-100": 100, "places365": 365, "imagenet1k": 1000, "imagenet1kv2": 1000}.get(canonical, 10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("catalogs", nargs="+", type=Path)
    parser.add_argument("--prune", action="store_true", help="Also prune row 0 at rate 0.8")
    args = parser.parse_args()

    failures = []
    for catalog in args.catalogs:
        data = json.loads(catalog.read_text(encoding="utf-8"))
        print(f"=== {catalog} n={len(data)} ===")
        for ckpt, row in data.items():
            arch, script, dataset = row[0], row[1], row[2]
            kwargs = dict(row[3]) if len(row) > 3 and isinstance(row[3], dict) else {}
            print(f"  {Path(ckpt).name}  {arch}")
            try:
                ncls = kwargs.pop("num_classes", num_classes_for(dataset))
                model = utils.load_model_from_script(
                    arch, dataset, script, ckpt, kwargs, num_classes=ncls, input_shape=(3, 32, 32))
                model = model.cpu().eval()
                y = model(torch.randn(1, 3, 32, 32))
                print(f"    load+fwd ok  out={tuple(y.shape)}  params={sum(p.numel() for p in model.parameters())}")
                if args.prune:
                    rows = ModelWithRows(model)
                    prune_current_model(rows, 0.8, 0)
                    out = rows.model.eval()(torch.randn(1, 3, 32, 32))
                    print(f"    prune row0@0.8 ok  out={tuple(out.shape)}")
            except Exception as exc:
                print(f"    FAIL {type(exc).__name__}: {exc}")
                failures.append((ckpt, str(exc)))
    if failures:
        print(f"\n{len(failures)} failures")
        for ckpt, msg in failures:
            print(f"  {Path(ckpt).name}: {msg}")
        sys.exit(1)
    print("all catalogs ok")


if __name__ == "__main__":
    main()
