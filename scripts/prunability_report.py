"""
Report how much of each architecture SPECTRA can compress structurally.

For every prunable layer the channel-dependency analysis either resolves a coupled group
(the layer physically shrinks) or blocks it (the layer is masked, keeping its shape). This
script prints the split per architecture, plus the parameter reduction achieved by applying
one uniform pass, so the coverage of the pruning implementation is measurable rather than
assumed.

    python scripts/prunability_report.py [--rate 0.8]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf

if StaticConf.get_instance() is None:
    StaticConf(ConfigurationValues(
        device=torch.device("cpu"), test_name="report", input_dict={},
        compression_rates_dict={0: 1.0}, runtime_limit=1, num_epochs=0,
        train_compressed_layer_only=True, allowed_acc_reduction=5, discount_factor=0.99,
        learning_rate=1e-3, rollout_limit=None, passes=1, prune=True, seed=0, n_splits=0,
        train_split=0.7, val_split=0.2, database_dict={}, actor_checkpoint_path=None,
        critic_checkpoint_path=None, save_pruned_checkpoints=False, test_ts="ts",
    ))

import src.channel_groups as channel_groups  # noqa: E402
import src.utils as utils  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src.NetworkEnv import prune_current_model  # noqa: E402


def build_models():
    from torchvision import models
    return {
        "vgg16": (models.vgg16(weights=None), (3, 64, 64)),
        "resnet18": (models.resnet18(weights=None), (3, 64, 64)),
        "resnet50": (models.resnet50(weights=None), (3, 64, 64)),
        "densenet121": (models.densenet121(weights=None), (3, 64, 64)),
        "mobilenet_v2": (models.mobilenet_v2(weights=None), (3, 64, 64)),
        "googlenet": (models.googlenet(weights=None, init_weights=False, aux_logits=False), (3, 64, 64)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=0.8, help="Compression rate per layer")
    args = parser.parse_args()

    header = f"{'architecture':<14}{'rows':>6}{'shrunk':>8}{'masked':>8}{'params before':>16}{'params after':>15}{'reduction':>11}"
    print(header)
    print("-" * len(header))

    for name, (model, shape) in build_models().items():
        model = model.eval()
        params_before = sum(p.numel() for p in model.parameters())
        flops_before = utils.calc_flops(model, shape)

        groups = channel_groups.build_channel_groups(model)
        if groups is None:
            print(f"{name:<14}{'-':>6}{'-':>8}{'-':>8}{'not symbolically traceable':>50}")
            continue

        num_rows = len(ModelWithRows(model).row_to_main_layer)
        outcomes = Counter()
        blocked_reasons = Counter()

        for row_idx in range(num_rows):
            model_with_rows = ModelWithRows(model)
            layer = model_with_rows.all_layers[model_with_rows.row_to_main_layer[row_idx]]
            group = channel_groups.group_of(channel_groups.build_channel_groups(model) or [], layer)

            before = sum(p.numel() for p in model.parameters())
            prune_current_model(model_with_rows, args.rate, row_idx)
            model = model_with_rows.model
            after = sum(p.numel() for p in model.parameters())

            if after < before:
                outcomes["shrunk"] += 1
            else:
                outcomes["masked"] += 1
                if group is not None and not group.prunable:
                    blocked_reasons[group.reason] += 1

        model = model.eval()
        with torch.no_grad():
            model(torch.zeros(1, *shape))  # the compressed network must still run

        params_after = sum(p.numel() for p in model.parameters())
        reduction = 100 * (1 - params_after / params_before)
        print(f"{name:<14}{num_rows:>6}{outcomes['shrunk']:>8}{outcomes['masked']:>8}"
              f"{params_before:>16,}{params_after:>15,}{reduction:>10.1f}%")

        for reason, count in blocked_reasons.most_common():
            print(f"{'':<14}  blocked x{count}: {reason}")


if __name__ == "__main__":
    main()
