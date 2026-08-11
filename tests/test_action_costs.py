"""
Per-action cost features: what each compression rate would remove from the network.

    python -m pytest tests/test_action_costs.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf, SmallVGG, ResidualNet  # noqa: E402

_init_static_conf()

import src.action_costs as action_costs  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402

RATES = [1.0, 0.9, 0.8, 0.7, 0.6]
SHAPE = (3, 32, 32)


def _costs(model, layer, rates=RATES, shape=SHAPE):
    return action_costs.estimate_action_costs(model, layer, rates, shape)


def test_costs_have_one_row_per_action():
    model = SmallVGG().eval()
    costs = _costs(model, model.features[0])

    assert costs.shape == (len(RATES), 3)
    assert torch.allclose(costs[:, 0], torch.tensor(RATES))


def test_no_compression_costs_nothing():
    model = SmallVGG().eval()
    costs = _costs(model, model.features[0])

    assert costs[0, 1].item() == pytest.approx(0.0, abs=1e-9)
    assert costs[0, 2].item() == pytest.approx(0.0, abs=1e-9)


def test_more_aggressive_rates_remove_more():
    model = SmallVGG().eval()
    costs = _costs(model, model.features[3])

    params = costs[:, 1].tolist()
    flops = costs[:, 2].tolist()
    assert params == sorted(params), "parameter cost must grow as the rate falls"
    assert flops == sorted(flops), "MAC cost must grow as the rate falls"
    assert params[-1] > 0


def test_predicted_parameter_cost_matches_actual_pruning():
    """The advertised price must be what the environment actually charges."""
    from src.NetworkEnv import prune_current_model

    model = SmallVGG().eval()
    total_before = sum(p.numel() for p in model.parameters())
    predicted = _costs(model, model.features[3])[2, 1].item()  # rate 0.8

    model_with_rows = ModelWithRows(model)
    row_idx = next(row for row, main in model_with_rows.row_to_main_layer.items()
                   if model_with_rows.all_layers[main] is model.features[3])
    prune_current_model(model_with_rows, 0.8, row_idx)

    actual = 1 - sum(p.numel() for p in model_with_rows.model.parameters()) / total_before
    assert actual == pytest.approx(predicted, rel=1e-3)


def test_cost_reflects_the_whole_coupled_group():
    """
    Pruning a residual-coupled layer also shrinks its group partners, so its advertised cost
    must exceed what the layer alone accounts for.
    """
    model = ResidualNet().eval()
    groups = channel_groups.build_channel_groups(model)
    group = channel_groups.group_of(groups, model.block.conv2)
    assert group is not None and group.prunable
    assert len(group.producers) > 1, "conv2 and the shortcut producer share a dimension"

    costs = _costs(model, model.block.conv2)
    predicted = costs[2, 1].item()  # rate 0.8

    conv2_only = sum(p.numel() for p in model.block.conv2.parameters()) * 0.2
    assert predicted * sum(p.numel() for p in model.parameters()) > conv2_only


def test_masked_layers_report_no_mac_saving():
    """A layer that can only be masked keeps its shape, so its MAC count does not change."""

    class Dynamic(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(8, 2)

        def forward(self, x):
            out = self.conv(x)
            if out.sum() > 0:  # defeats symbolic tracing
                out = out * 2
            return self.fc(self.flatten(self.pool(out)))

    model = Dynamic().eval()
    costs = _costs(model, model.conv)

    assert costs[-1, 2].item() == pytest.approx(0.0, abs=1e-9)
    assert costs[-1, 1].item() > 0  # weights still go away


def test_costs_are_comparable_across_architectures():
    """Fractions of the whole network, so a generic agent can compare unrelated CNNs."""
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models import resnet18

    model = resnet18(weights=None).eval()
    costs = _costs(model, model.layer1[0].conv2, shape=(3, 64, 64))

    assert (costs[:, 1] >= 0).all() and (costs[:, 1] <= 1).all()
    assert (costs[:, 2] >= 0).all() and (costs[:, 2] <= 1).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
