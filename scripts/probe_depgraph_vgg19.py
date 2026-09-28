"""
CPU check of the DepGraph VGG-19 CIFAR-100 hold-out before any GPU walk: the released
state_dict loads strict into ``vgg_depgraph.vgg19_bn`` through SPECTRA's own loader, CIFAR-100
test accuracy through SPECTRA's own dataset registry (DepGraph reports 73.50), parameter / MAC
counts next to the zoo twin, channel groups, and one structural cut through the live pruner.

    python scripts/probe_depgraph_vgg19.py --limit 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Configuration.ConfigurationValues import ConfigurationValues  # noqa: E402
from src.Configuration.StaticConf import StaticConf  # noqa: E402

if StaticConf.get_instance() is None:
    StaticConf(ConfigurationValues(
        device=torch.device("cpu"), test_name="depgraph-vgg19-probe", input_dict={},
        compression_rates_dict={0: 1.0, 1: 0.9, 2: 0.8},
        runtime_limit=60, num_epochs=0, train_compressed_layer_only=False,
        allowed_acc_reduction=10, discount_factor=0.99, learning_rate=1e-3,
        rollout_limit=None, passes=1, prune=True, seed=0, n_splits=0,
        train_split=0.7, val_split=0.2, database_dict={},
        actor_checkpoint_path=None, critic_checkpoint_path=None,
        save_pruned_checkpoints=False, test_ts="probe",
    ))

import src.channel_groups as channel_groups  # noqa: E402
import src.utils as utils  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src.NetworkEnv import prune_current_model  # noqa: E402

CKPT = "/home/paretsky/spectra_pretrained_networks/vgg19_cifar100_dep_graph_73.5.pth"
SCRIPT = "/home/paretsky/spectra_models_instantiation/vgg_depgraph.py"
TWIN = ("/home/paretsky/spectra_pretrained_networks/vgg19_bn_cifar100_chenyaofo_73.87_20.61_797.42.pt",
        "/home/paretsky/spectra_models_instantiation/vgg_chenyaofo.py", "vgg19_bn")


@torch.no_grad()
def accuracy(model, loader, limit: int) -> tuple:
    model.eval()
    correct = seen = 0
    for x, y in loader:
        if limit and seen >= limit:
            break
        if limit:
            x, y = x[: limit - seen], y[: limit - seen]
        correct += int((model(x).argmax(1) == y).sum())
        seen += int(y.numel())
    return correct / max(seen, 1), seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--script", default=SCRIPT)
    parser.add_argument("--limit", type=int, default=3000, help="test images (0 = all)")
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    registry = utils.preload_datasets(["cifar-100"], 0.7, 0.2)
    shape = registry.input_shape("cifar-100")
    model = utils.load_model_from_script(
        "vgg19_bn", "cifar-100", args.script, args.checkpoint, {},
        num_classes=registry.num_classes("cifar-100"), input_shape=shape).eval()
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw = raw.get("state_dict", raw) if isinstance(raw, dict) else raw.state_dict()
    own = model.state_dict()
    strict = sorted(set(raw) ^ set(own))
    out = {"strict_key_mismatch": strict[:10], "n_keys": len(own)}

    out["params"] = int(utils.calc_num_parameters(model))
    out["macs"] = int(utils.calc_flops(model, shape))
    groups = channel_groups.build_channel_groups(model) or []
    out["groups"] = len(groups)
    out["groups_prunable"] = sum(1 for g in groups if g.prunable)
    out["group_widths"] = [int(g.width) for g in groups if g.prunable]
    mwr = ModelWithRows(model)
    out["rows"] = len(mwr.row_to_main_layer)

    _, _, test_loader = registry.loaders("cifar-100")
    acc, seen = accuracy(model, test_loader, args.limit)
    out["test_acc"] = round(acc, 4)
    out["test_images"] = seen

    twin_path, twin_script, twin_arch = TWIN
    if Path(twin_path).is_file():
        twin = utils.load_model_from_script(
            twin_arch, "cifar-100", twin_script, twin_path, {},
            num_classes=registry.num_classes("cifar-100"), input_shape=shape).eval()
        out["twin_params"] = int(utils.calc_num_parameters(twin))
        out["twin_macs"] = int(utils.calc_flops(twin, shape))

    cut = ModelWithRows(model)
    cut = prune_current_model(cut, 0.9, 1, quiet=True, record=False, input_shape=shape)
    outcome = dict(getattr(cut, "last_prune_outcome", {}) or {})
    out["cut_row1_mode"] = outcome.get("mode")
    out["cut_row1_param_ratio"] = round(utils.calc_num_parameters(cut.model) / out["params"], 4)
    with torch.no_grad():
        out["cut_forward_shape"] = list(cut.model(torch.zeros(2, *shape)).shape)
    print("DEPGRAPH_VGG19 " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
