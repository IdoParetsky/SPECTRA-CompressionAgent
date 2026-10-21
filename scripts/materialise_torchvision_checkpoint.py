#!/usr/bin/env python3
"""
Materialise (or verify) a torchvision ImageNet checkpoint in SPECTRA's pipeline format (v29, the ImageNet line).

The pipeline reads a checkpoint as a plain ``state_dict`` whose keys match the factory in
``spectra_models_instantiation/torchvision_instantiation.py`` (``resnet50`` / ``mobilenet_v2``, 1000 classes), named

    {arch-token}_imagenet1kv1_torchvision_{top1}_{paramsM}_{flopsM}.pth

and referenced from an input JSON as ``[arch, torchvision_instantiation.py, "imagenet1k"]``. The hub weights are
downloaded once into ``--cache`` (``TORCH_HOME``; runs on the GPU nodes are offline and never download), so run this on
the login node. ``--verify <existing .pth>`` instead compares an existing pipeline file with the hub weights tensor by
tensor (and checks that it loads strictly into the factory) without writing anything.

    python scripts/materialise_torchvision_checkpoint.py --arch resnet50 \\
        --verify /home/paretsky/spectra_pretrained_networks/resnet50_imagenet1kv1_torchvision_76.13_25.56.pth
    python scripts/materialise_torchvision_checkpoint.py --arch mobilenet_v2 --out-dir /home/paretsky/spectra_pretrained_networks \\
        --catalog configs/input_imagenet_tv_mobilenet_v2_v1.json

The MACs are ``utils.calc_flops`` at (3, 224, 224): Conv2d / Linear multiply-accumulates plus the element-wise ops the
pipeline counts (BatchNorm, activations), the same number every TRAJ ``FLOPs x`` ratio is read against.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ARCHES = {
    # pipeline arch -> (torchvision weights enum, filename token)
    "resnet50": ("ResNet50_Weights", "resnet50"),
    "mobilenet_v2": ("MobileNet_V2_Weights", "mobilenet-v2"),
}
INPUT_SHAPE = (3, 224, 224)
DEFAULT_SCRIPT = "/home/paretsky/spectra_models_instantiation/torchvision_instantiation.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=sorted(ARCHES))
    p.add_argument("--weights", default="IMAGENET1K_V1", help="torchvision weights name (default IMAGENET1K_V1)")
    p.add_argument("--cache", type=Path, default=Path("/home/paretsky/torch_cache"),
                   help="TORCH_HOME for the one-time hub download (default /home/paretsky/torch_cache)")
    p.add_argument("--out-dir", type=Path, default=Path("/home/paretsky/spectra_pretrained_networks"))
    p.add_argument("--script", default=DEFAULT_SCRIPT, help="instantiation script path written into the catalog row")
    p.add_argument("--catalog", type=Path, nargs="*", default=None, help="input JSON(s) to append the row to")
    p.add_argument("--verify", type=Path, default=None,
                   help="an existing pipeline checkpoint to compare with the hub weights (nothing is written)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hub_state_dict(arch: str, weights_name: str, cache: Path):
    """The hub weights (downloaded into ``cache`` once; a later call reads the cached file) and their meta."""
    os.environ["TORCH_HOME"] = str(cache)
    cache.mkdir(parents=True, exist_ok=True)
    import torchvision
    enum_name, _token = ARCHES[arch]
    weights = torchvision.models.get_weight(f"{enum_name}.{weights_name}")
    filename = os.path.basename(weights.url)
    cached = cache / "hub" / "checkpoints" / filename
    if not cached.is_file():
        print(f"downloading {weights.url} -> {cached}", flush=True)
    try:
        state = weights.get_state_dict(progress=False, check_hash=True)
    except Exception as error:  # noqa: BLE001 - offline node: say where to put the file
        raise SystemExit(f"could not obtain {weights.url} ({type(error).__name__}: {error}); copy the file into "
                         f"{cached} on a node with network access and rerun") from error
    return state, weights, cached


def _factory_model(arch: str, script: str):
    import src.utils as utils
    import importlib.util
    path = script if os.path.isfile(script) else str(REPO / "spectra_models_instantiation" / "torchvision_instantiation.py")
    spec = importlib.util.spec_from_file_location("torchvision_instantiation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn, _name = utils.resolve_instantiation_func(module, arch)
    return fn(num_classes=1000, large_input=True)


def _strict_load(model, state) -> None:
    import torch
    import src.utils as utils
    tensors = {k: v for k, v in state.items() if torch.is_tensor(v) and not utils.ignorable_state_key(k)}
    aligned = utils.apply_key_aliases(tensors, list(model.state_dict().keys()))
    missing, unexpected = utils.load_state_dict_compatible(model, aligned)
    if missing or unexpected:
        raise SystemExit(f"strict load failed: missing={missing[:5]} unexpected={unexpected[:5]}")


def main():
    args = parse_args()
    import torch
    import src.utils as utils
    hub, weights, cached = _hub_state_dict(args.arch, args.weights, args.cache)
    metrics = (weights.meta.get("_metrics") or {}).get("ImageNet-1K", {})
    top1, top5 = metrics.get("acc@1"), metrics.get("acc@5")
    model = _factory_model(args.arch, args.script)
    _strict_load(model, hub)
    model.eval()
    params = int(sum(p.numel() for p in model.parameters()))
    flops = float(utils.calc_flops(model, INPUT_SHAPE, torch.device("cpu")))
    print(f"{args.arch} {args.weights}: hub file {cached} (sha256 {_sha256(cached)[:12]}...) | "
          f"top-1 {top1} top-5 {top5} | params {params} ({params / 1e6:.3f}M) | MACs@224 {flops / 1e6:.3f}M "
          f"(calc_flops, incl. element-wise)", flush=True)

    if args.verify is not None:
        existing = torch.load(args.verify, map_location="cpu", weights_only=False)
        existing = utils.unwrap_checkpoint_state_dict(existing)
        extras = sorted(k for k in existing if utils.ignorable_state_key(k))
        compare = {k: v for k, v in existing.items() if k not in extras}
        # BatchNorm's num_batches_tracked (ignorable on both sides) carries no weight: compare the real tensors
        hub_real = {k: v for k, v in hub.items() if not utils.ignorable_state_key(k)}
        missing = sorted(set(hub_real) - set(compare))
        unexpected = sorted(set(compare) - set(hub_real))
        differ = [k for k in hub_real if k in compare and not torch.equal(hub_real[k].cpu(), compare[k].cpu())]
        verdict = "IDENTICAL to the hub weights" if not (missing or unexpected or differ) else "DIFFERS"
        print(f"verify {args.verify}: {verdict} | {len(compare)} tensors compared, {len(extras)} ignorable extras "
              f"({extras[:3]}{'...' if len(extras) > 3 else ''}) | missing {len(missing)} unexpected {len(unexpected)} "
              f"differing {len(differ)}{(' ' + str(differ[:5])) if differ else ''}", flush=True)
        model2 = _factory_model(args.arch, args.script)
        _strict_load(model2, existing)
        print(f"verify {args.verify}: loads strictly into the pipeline factory ({args.arch})", flush=True)
        return

    token = ARCHES[args.arch][1]
    fname = f"{token}_imagenet1kv1_torchvision_{top1:.2f}_{params / 1e6:.3f}_{flops / 1e6:.3f}.pth"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / fname
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")
    torch.save({k: v.cpu() for k, v in hub.items()}, out)
    row = [args.arch, args.script, "imagenet1k"]
    print(f"saved {out}\nrow: {json.dumps({str(out): row})}", flush=True)
    for catalog in args.catalog or []:
        data = {}
        if catalog.is_file():
            data = json.loads(catalog.read_text(encoding="utf-8"))
        data[str(out)] = row
        tmp = catalog.with_suffix(catalog.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, catalog)
        print(f"appended {out} -> {catalog}", flush=True)


if __name__ == "__main__":
    main()
