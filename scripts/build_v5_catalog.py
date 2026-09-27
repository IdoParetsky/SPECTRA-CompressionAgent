#!/usr/bin/env python3
"""Turn configs/v5_diversity_plan.json + the checkpoint folder into SPECTRA catalogs.

Does **not** invent weights. Skips a slot when no matching .pt exists.
Never writes a live --database unless --write and every *required* train
prefix resolves.

    python scripts/build_v5_catalog.py --dry-run
    python scripts/build_v5_catalog.py --write --ckpt-root /home/paretsky/spectra_pretrained_networks

P5-B3 (18 Sep, Fable): the train catalog that the GPU sees is gate-driven.

    configs/database_offline_v5_p5b3.json          intended: C10 core + every CIFAR-100 *candidate*
    configs/v5_p5b3_c100_gate.json                 per-candidate status (pending_probe / admitted / rejected)
    python scripts/build_v5_catalog.py --emit-admitted
        -> configs/database_offline_v5_p5b3_admitted.json  (core + admitted C100 only; the --database)
    python scripts/build_v5_catalog.py --check-admitted configs/database_offline_v5_p5b3_admitted.json
        -> exit 1 if a CIFAR-100 row is not admitted (spectra.sbatch runs this before offline_train_v5_p5b3)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO / "configs" / "v5_diversity_plan.json"
SCRIPT_ROOT = Path("/home/paretsky/spectra_models_instantiation")
CKPT_ROOT = Path("/home/paretsky/spectra_pretrained_networks")

SCRIPT_FOR_ARCH = {
    "resnet20": "thin_res_net.py",
    "resnet32": "resnet_chenyaofo.py",
    "resnet44": "resnet_chenyaofo.py",
    "resnet56": "thin_res_net.py",
    "vgg11_bn": "vgg_chenyaofo.py",
    "vgg13_bn": "vgg_chenyaofo.py",
    "vgg16_bn": "vgg_chenyaofo.py",
    "vgg19_bn": "vgg_chenyaofo.py",
    "mobilenet_v2x05": "mobilenetv2_chenyaofo.py",
    "mobilenet_v2x075": "mobilenetv2_chenyaofo.py",
    "mobilenet_v2x1": "mobilenetv2_chenyaofo.py",
    "mobilenet_v2x14": "mobilenetv2_chenyaofo.py",
    "shufflenetv2x05": "shufflenetv2_chenyaofo.py",
    "shufflenetv2x1": "shufflenetv2_chenyaofo.py",
    "shufflenetv2x15": "shufflenetv2_chenyaofo.py",
    "shufflenetv2x2": "shufflenetv2_chenyaofo.py",
    "repvgg_a0": "repvgg_chenyaofo.py",
    "repvgg_a1": "repvgg_chenyaofo.py",
    "repvgg_a2": "repvgg_chenyaofo.py",
    "densenet40": "densenet_cifar.py",
    "densenet100": "densenet_cifar.py",
    "wrn_16_4": "wide_resnet.py",
    "wrn_16_8": "wide_resnet.py",
    "wrn_28_2": "wide_resnet.py",
    "wrn_28_10": "wide_resnet.py",
    "preact_resnet20": "preact_resnet.py",
    "preact_resnet32": "preact_resnet.py",
    "preact_resnet56": "preact_resnet.py",
}

DATASET_JSON = {
    "cifar10": "cifar-10",
    "cifar100": "cifar-100",
    "svhn": "svhn",
    "fashionmnist": {"name": "fashion-mnist", "image_size": 32, "to_rgb": True},
}


def dataset_from_name(name: str):
    lower = name.lower()
    for token, spec in DATASET_JSON.items():
        if f"_{token}_" in lower or lower.startswith(f"{token}_"):
            return spec
    raise ValueError(f"cannot parse dataset from {name}")


def arch_from_name(name: str) -> str:
    stem = name.split("_cifar")[0].split("_svhn")[0].split("_fashion")[0]
    stem = stem.replace("mobilenet-v2", "mobilenet_v2").replace("repvgga", "repvgg_a")
    stem = stem.replace("vgg11-bn", "vgg11_bn").replace("vgg13-bn", "vgg13_bn")
    stem = stem.replace("vgg16-bn", "vgg16_bn").replace("vgg19-bn", "vgg19_bn")
    if stem.startswith("resnet") and "-width" in stem:
        return stem.split("-width")[0]
    return stem.replace("-", "_")


def find_prefix(ckpt_root: Path, prefix: str) -> Path | None:
    hits = sorted(ckpt_root.glob(f"{prefix}*.pt")) + sorted(ckpt_root.glob(f"{prefix}*.pth"))
    return hits[0] if hits else None


def row_for(ckpt: Path) -> list:
    name = ckpt.name
    arch = arch_from_name(name)
    if "thin-res-net" in name:
        script = "thin_res_net.py"
    elif "wide-resnet" in name or arch.startswith("wrn"):
        script = "wide_resnet.py"
    elif "preact-resnet" in name or arch.startswith("preact"):
        script = "preact_resnet.py"
    elif "densenet-cifar" in name or arch.startswith("densenet"):
        script = "densenet_cifar.py"
    elif "chenyaofo" in name:
        if arch.startswith("resnet"):
            script = "resnet_chenyaofo.py"
        elif arch.startswith("vgg"):
            script = "vgg_chenyaofo.py"
        elif "mobilenet" in arch:
            script = "mobilenetv2_chenyaofo.py"
        elif "shufflenet" in arch:
            script = "shufflenetv2_chenyaofo.py"
        elif "repvgg" in arch:
            script = "repvgg_chenyaofo.py"
        else:
            raise KeyError(f"no chenyaofo instantiation for arch={arch} file={name}")
    else:
        script = SCRIPT_FOR_ARCH.get(arch)
        if script is None:
            raise KeyError(f"no instantiation for arch={arch} file={name}")
    return [arch, str(SCRIPT_ROOT / script), dataset_from_name(name)]


GATE_PATH = REPO / "configs" / "v5_p5b3_c100_gate.json"
INTENDED_PATH = REPO / "configs" / "database_offline_v5_p5b3.json"
ADMITTED_PATH = REPO / "configs" / "database_offline_v5_p5b3_admitted.json"


def load_gate(path: Path = GATE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def admitted_catalog(intended: dict, gate: dict) -> dict:
    """C10 core plus only the CIFAR-100 rows whose gate status is 'admitted'."""
    c100 = gate.get("c100", {})
    out = {}
    for path, row in intended.items():
        name = Path(path).name
        dataset = row[2] if isinstance(row[2], str) else row[2].get("name")
        if dataset == "cifar-100":
            if c100.get(name, {}).get("status") == "admitted":
                out[path] = row
        else:
            out[path] = row
    return out


def check_admitted(database: Path, gate: dict) -> list[str]:
    """Names of CIFAR-100 rows in ``database`` that the gate table does not mark admitted."""
    db = json.loads(database.read_text(encoding="utf-8"))
    bad = []
    for path, row in db.items():
        dataset = row[2] if isinstance(row[2], str) else row[2].get("name")
        if dataset == "cifar-100":
            status = gate.get("c100", {}).get(Path(path).name, {}).get("status")
            if status != "admitted":
                bad.append(f"{Path(path).name} (gate status: {status})")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=PLAN_PATH)
    p.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT)
    p.add_argument("--write", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--emit-admitted", action="store_true",
                   help="write --out = non-C100 rows of --intended + CIFAR-100 rows marked 'admitted' in --gate "
                        "(defaults: the P5-B3 trio of files)")
    p.add_argument("--check-admitted", type=Path, metavar="DATABASE",
                   help="gate: exit 1 if DATABASE holds a CIFAR-100 row not marked 'admitted' in --gate")
    p.add_argument("--intended", type=Path, default=INTENDED_PATH,
                   help="intended catalog (V7: configs/database_offline_v7_diverse.json)")
    p.add_argument("--gate", type=Path, default=GATE_PATH,
                   help="gate table (V7: configs/v7_c100_gate.json)")
    p.add_argument("--out", type=Path, default=ADMITTED_PATH,
                   help="admitted catalog to write (V7: configs/database_offline_v7_diverse_admitted.json)")
    p.add_argument("--min-c100", type=int, default=0,
                   help="--emit-admitted exits 2 unless at least this many CIFAR-100 rows are admitted")
    args = p.parse_args()
    if args.check_admitted:
        bad = check_admitted(args.check_admitted, load_gate(args.gate))
        if bad:
            print("C100 gate FAILED — unadmitted CIFAR-100 rows in --database:\n  " + "\n  ".join(bad),
                  file=sys.stderr)
            sys.exit(1)
        print(f"C100 gate ok: {args.check_admitted}")
        return
    if args.emit_admitted:
        intended = json.loads(args.intended.read_text(encoding="utf-8"))
        out = admitted_catalog(intended, load_gate(args.gate))
        n_c100 = sum(1 for r in out.values() if r[2] == "cifar-100")
        args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out} n={len(out)} (cifar-100 admitted: {n_c100})")
        if n_c100 < args.min_c100:
            print(f"only {n_c100} CIFAR-100 net(s) admitted (need {args.min_c100})", file=sys.stderr)
            sys.exit(2)
        if n_c100 == 0:
            print("no CIFAR-100 net is admitted yet — this file is a C10-only control catalog",
                  file=sys.stderr)
        return
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    train = {}
    missing = []
    for prefix in plan["train_p5b_intended"]["keep_existing"]:
        hit = find_prefix(args.ckpt_root, prefix)
        if hit is None:
            missing.append(prefix)
            print(f"MISSING train {prefix}")
            continue
        train[str(hit)] = row_for(hit)
        print(f"train {hit.name}")
    unlike = {}
    for spec in plan["test_holdouts"]["unlike_new_after_p5c"]:
        prefix = f"{spec['arch']}_{spec['dataset'].replace('-', '')}_"
        hit = find_prefix(args.ckpt_root, prefix)
        if hit is None:
            print(f"PENDING unlike {prefix}")
            continue
        unlike[str(hit)] = row_for(hit)
        print(f"unlike-new {hit.name}")

    print(f"resolved train {len(train)}/{len(plan['train_p5b_intended']['keep_existing'])}  unlike-new {len(unlike)}")
    if args.dry_run or not args.write:
        return
    if missing:
        print("refusing --write; required train prefixes missing", file=sys.stderr)
        sys.exit(1)
    out_train = REPO / "configs" / "database_v5_p5b.json"
    out_unlike = REPO / "configs" / "input_v5_unlike_new.json"
    out_train.write_text(json.dumps(train, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_train} n={len(train)}")
    if unlike:
        out_unlike.write_text(json.dumps(unlike, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_unlike} n={len(unlike)}")


if __name__ == "__main__":
    main()
