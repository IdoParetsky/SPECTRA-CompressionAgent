"""
Shape-level checks for SPECTRA's structured pruning.

These exercise the compression path only (no training, no GPU), so they can be run on a
laptop before submitting a SLURM job:

    python -m pytest tests/test_pruning.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf


def _init_static_conf():
    """Minimal StaticConf so ModelWithRows can resolve a device."""
    if StaticConf.get_instance() is not None:
        return
    StaticConf(ConfigurationValues(
        device=torch.device("cpu"), test_name="unit-test", input_dict={},
        compression_rates_dict={0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6},
        runtime_limit=60, num_epochs=0, train_compressed_layer_only=True,
        allowed_acc_reduction=5, discount_factor=0.99, learning_rate=1e-3,
        rollout_limit=10, passes=1, prune=True, seed=42, n_splits=0,
        train_split=0.7, val_split=0.2, database_dict={},
        actor_checkpoint_path=None, critic_checkpoint_path=None,
        save_pruned_checkpoints=False, test_ts="ts",
    ))


_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
import src.pruning as pruning  # noqa: E402


class SmallVGG(nn.Module):
    """Conv -> BN -> ReLU stack followed by a flattened classifier head."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 16 * 16, 64), nn.ReLU(), nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualNet(nn.Module):
    """Carries a skip connection, so the second conv cannot be resized independently."""

    class Block(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            return self.relu(x + self.bn2(self.conv2(out)))

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.block = self.Block(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        return self.fc(self.flatten(self.pool(self.block(self.stem(x)))))


def _row_index_of_layer(model_with_rows, layer_idx):
    for row_idx, main_idx in model_with_rows.row_to_main_layer.items():
        if main_idx == layer_idx:
            return row_idx
    raise AssertionError(f"layer {layer_idx} does not start a row")


def test_structural_pruning_shrinks_and_stays_runnable():
    from src.NetworkEnv import prune_current_model

    model = SmallVGG().eval()
    sample = torch.randn(2, 3, 32, 32)
    reference = model(sample)

    params_before = sum(p.numel() for p in model.parameters())

    model_with_rows = ModelWithRows(model)
    first_conv_idx = model_with_rows.row_to_main_layer[0]
    assert isinstance(model_with_rows.all_layers[first_conv_idx], nn.Conv2d)

    prune_current_model(model_with_rows, compression_rate=0.5, row_to_prune_idx=0)

    pruned = model_with_rows.model.eval()
    output = pruned(sample)

    assert output.shape == reference.shape
    assert sum(p.numel() for p in pruned.parameters()) < params_before

    # The first conv keeps half of its filters and the consumers follow suit
    conv1 = pruned.features[0]
    assert conv1.out_channels == 8
    assert pruned.features[1].num_features == 8
    assert pruned.features[3].in_channels == 8


def test_pruning_propagates_through_flatten_into_linear():
    from src.NetworkEnv import prune_current_model

    model = SmallVGG().eval()
    sample = torch.randn(2, 3, 32, 32)

    model_with_rows = ModelWithRows(model)
    second_conv_idx = model_with_rows.row_to_main_layer[1]
    assert isinstance(model_with_rows.all_layers[second_conv_idx], nn.Conv2d)

    prune_current_model(model_with_rows, compression_rate=0.5, row_to_prune_idx=1)
    pruned = model_with_rows.model.eval()

    assert pruned.features[3].out_channels == 16
    # Flatten spreads each channel over 16x16 spatial positions
    assert pruned.classifier[1].in_features == 16 * 16 * 16
    assert pruned(sample).shape == (2, 10)


def test_repeated_pruning_compounds():
    from src.NetworkEnv import prune_current_model

    model = SmallVGG().eval()
    for _ in range(2):
        model_with_rows = ModelWithRows(model)
        prune_current_model(model_with_rows, compression_rate=0.5, row_to_prune_idx=0)
        model = model_with_rows.model

    # 16 -> 8 -> 4 rather than stalling at 8
    assert model.features[0].out_channels == 4
    assert model.eval()(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_residual_coupled_layers_shrink_together():
    """conv2 and the layer feeding the shortcut are one group and must shrink in lockstep."""
    from src.NetworkEnv import prune_current_model

    model = ResidualNet().eval()
    sample = torch.randn(2, 3, 32, 32)
    params_before = sum(p.numel() for p in model.parameters())

    model_with_rows = ModelWithRows(model)
    conv2_idx = next(idx for idx, layer in enumerate(model_with_rows.all_layers)
                     if layer is model.block.conv2)
    row_idx = _row_index_of_layer(model_with_rows, conv2_idx)

    prune_current_model(model_with_rows, compression_rate=0.5, row_to_prune_idx=row_idx)
    pruned = model_with_rows.model.eval()

    # The stem feeds the identity shortcut, so it is part of conv2's group
    assert pruned.block.conv2.out_channels == 8
    assert pruned.stem.out_channels == 8
    assert pruned.block.bn2.num_features == 8
    # Consumers of the group follow
    assert pruned.block.conv1.in_channels == 8
    assert pruned.fc.in_features == 8

    assert pruned(sample).shape == (2, 10)
    assert sum(p.numel() for p in pruned.parameters()) < params_before


class ConcatNet(nn.Module):
    """Two branches merged by torch.cat, as in an Inception block or a DenseNet bank."""

    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 8, 3, padding=1)
        self.b = nn.Conv2d(3, 6, 3, padding=1)
        self.norm = nn.BatchNorm2d(14)
        self.head = nn.Conv2d(14, 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        merged = torch.cat([self.a(x), self.b(x)], dim=1)
        return self.fc(self.flatten(self.pool(self.head(self.norm(merged)))))


def test_concatenated_branch_is_pruned_at_its_offset():
    """
    Only the pruned branch's slice of the concatenated tensor may be removed.

    `a` occupies channels 0..7 and `b` channels 8..13, so halving `a` must leave `b`'s six
    channels intact and shift them down, not delete channels by position.
    """
    from src.NetworkEnv import prune_current_model

    model = ConcatNet().eval()
    sample = torch.randn(2, 3, 16, 16)

    model_with_rows = ModelWithRows(model)
    conv_idx = next(idx for idx, layer in enumerate(model_with_rows.all_layers) if layer is model.a)
    row_idx = _row_index_of_layer(model_with_rows, conv_idx)

    prune_current_model(model_with_rows, compression_rate=0.5, row_to_prune_idx=row_idx)
    pruned = model_with_rows.model.eval()

    assert pruned.a.out_channels == 4
    assert pruned.b.out_channels == 6  # the other branch is untouched
    assert pruned.norm.num_features == 10
    assert pruned.head.in_channels == 10
    assert pruned(sample).shape == (2, 2)


def test_second_concat_branch_prunes_at_a_nonzero_offset():
    from src.NetworkEnv import prune_current_model

    model = ConcatNet().eval()
    sample = torch.randn(2, 3, 16, 16)

    model_with_rows = ModelWithRows(model)
    conv_idx = next(idx for idx, layer in enumerate(model_with_rows.all_layers) if layer is model.b)
    row_idx = _row_index_of_layer(model_with_rows, conv_idx)

    prune_current_model(model_with_rows, compression_rate=0.5, row_to_prune_idx=row_idx)
    pruned = model_with_rows.model.eval()

    assert pruned.a.out_channels == 8
    assert pruned.b.out_channels == 3
    assert pruned.head.in_channels == 11
    assert pruned(sample).shape == (2, 2)


def test_untraceable_model_falls_back_to_masking():
    """Data-dependent control flow defeats symbolic tracing; masking must still apply."""
    from src.NetworkEnv import prune_current_model

    class Dynamic(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(8, 2)

        def forward(self, x):
            out = self.conv(x)
            if out.sum() > 0:  # not symbolically traceable
                out = out * 2
            return self.fc(self.flatten(self.pool(out)))

    model = Dynamic().eval()
    model_with_rows = ModelWithRows(model)
    conv_idx = next(idx for idx, layer in enumerate(model_with_rows.all_layers) if layer is model.conv)

    prune_current_model(model_with_rows, 0.5, _row_index_of_layer(model_with_rows, conv_idx))
    pruned = model_with_rows.model.eval()

    assert pruned.conv.out_channels == 8  # shape preserved
    assert pruning.alive_filters(pruned.conv).numel() == 4
    assert pruned(torch.randn(2, 3, 16, 16)).shape == (2, 2)


def test_replace_layer_mutates_the_model_not_just_the_list():
    model = SmallVGG()
    model_with_rows = ModelWithRows(model)
    replacement = nn.Conv2d(3, 4, 3, padding=1)

    model_with_rows.replace_layer(model_with_rows.row_to_main_layer[0], replacement)

    assert model.features[0] is replacement


def test_flops_counter_reflects_pruning():
    import src.utils as utils
    from src.NetworkEnv import prune_current_model

    model = SmallVGG().eval()
    before = utils.calc_flops(model, (3, 32, 32))

    model_with_rows = ModelWithRows(model)
    prune_current_model(model_with_rows, compression_rate=0.5, row_to_prune_idx=0)

    after = utils.calc_flops(model_with_rows.model.eval(), (3, 32, 32))
    assert after < before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
