#!/usr/bin/env python3
"""Catalog L initiation: verify literature-home ckpts, print SPECTRA rows.

CPU-only. Does not train. Does not overlay src/. Does not fetch ImageNet-stem
nets onto CIFAR. Chenyaofo leftovers and VGG-19 / R56 / R110 / DenseNet-100
are already on leap — this script refuses to GPU-fetch them.

    python scripts/init_catalog_l.py
    python scripts/init_catalog_l.py --ckpt-root /home/paretsky/spectra_pretrained_networks

Missing CIFAR-native GoogLeNet / WRN / PreAct are reported, not downloaded.
WRN/PreAct pretrain: scripts/pretrain_v5_diversity.sh on a QOS hole.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAP_PATH = REPO / "configs" / "catalog_l_map.json"
DEFAULT_ROOT = Path("/home/paretsky/spectra_pretrained_networks")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--map", type=Path, default=MAP_PATH)
    args = p.parse_args()
    plan = json.loads(args.map.read_text(encoding="utf-8"))
    root: Path = args.ckpt_root
    missing = 0
    pending = 0
    print(f"ckpt_root={root} exists={root.is_dir()}")
    print("cell\tstatus\tfile")
    for row in plan["rows"]:
        cell = row["cell"]
        prefer = row.get("prefer")
        role = row.get("role")
        if role == "ckpt_pending_pretrain":
            print(f"{cell}\tPENDING_PRETRAIN\t(QOS hole; do not steal v3/V4)")
            pending += 1
            continue
        if role == "missing_cifar_native" or prefer in (None, "frozen probe only"):
            print(f"{cell}\t{role or 'probe_or_missing'}\t{row.get('note', '')}")
            if role == "missing_cifar_native":
                missing += 1
            continue
        path = root / prefer
        also = [root / n for n in row.get("also_on_disk", [])]
        ok = path.is_file()
        extras = sum(1 for a in also if a.is_file())
        status = "HAVE" if ok else "MISSING"
        if not ok:
            missing += 1
        print(f"{cell}\t{status}\t{prefer} extras={extras}")
    print(f"missing={missing} pending_pretrain={pending}")
    if missing:
        print("Do not ImageNet-stem a missing CIFAR cell. Factories first.")
        return 1
    print("Catalog L CIFAR homes with a prefer-file are on disk. No GPU fetch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
