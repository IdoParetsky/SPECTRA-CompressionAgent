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
import src.channel_groups as channel_groups  # noqa: E402
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


def test_any_compression_rate_removes_at_least_one_channel():
    """
    Regression: `ceil(0.9 * 6) == 6` kept every filter, so the action was a no-op that the
    agent was still rewarded for. On the thin ResNets in the database (widths of 6-16) this
    fired on most narrow layers.
    """
    for width in range(2, 33):
        for rate in (0.6, 0.7, 0.8, 0.9, 0.95):
            kept = pruning.target_width(width, rate)
            assert 1 <= kept < width, f"rate {rate} on width {width} kept {kept}"


def test_rate_one_is_the_only_no_op():
    assert pruning.target_width(16, 1.0) == 16
    assert pruning.target_width(1, 0.5) == 1  # a single channel cannot be removed


def test_target_width_tracks_the_requested_rate():
    """Away from the rounding edges the realised width should match the request closely."""
    for width in (32, 64, 128):
        for rate in (0.5, 0.75, 0.9):
            assert abs(pruning.target_width(width, rate) - rate * width) <= 1


def test_narrow_layer_actually_shrinks_end_to_end():
    """The rounding fix must reach the model, not just the helper."""
    from src.NetworkEnv import prune_current_model

    model = ResidualNet().eval()
    model_with_rows = ModelWithRows(model)
    before = pruning.layer_width(model.stem)

    prune_current_model(model_with_rows, compression_rate=0.9, row_to_prune_idx=0)

    assert pruning.layer_width(model_with_rows.model.stem) < before
    assert model_with_rows.last_prune_outcome["mode"] == "structural"
    model_with_rows.model.eval()(torch.randn(2, 3, 32, 32))


class MeanPoolNet(nn.Module):
    """Global average pooling written as `x.mean(dim=(2, 3))`, as the thin-ResNets do."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 24, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.fc = nn.Linear(24, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return self.fc(x.mean(dim=(2, 3)))


def test_spatial_mean_does_not_block_pruning():
    """
    Regression: `mean` was rejected outright as an unsupported op, so every layer feeding the
    global-average-pool of a thin ResNet fell back to masking and shed no FLOPs.
    """
    from src.NetworkEnv import prune_current_model
    import src.channel_groups as channel_groups

    model = MeanPoolNet().eval()
    groups = channel_groups.build_channel_groups(model)
    group = channel_groups.group_of(groups, model.conv2)
    assert group is not None and group.prunable, (
        f"conv2 blocked: {group.reason if group else 'no group'}")

    model_with_rows = ModelWithRows(model)
    row = _row_index_of_layer(model_with_rows, model_with_rows.all_layers.index(model.conv2))
    prune_current_model(model_with_rows, compression_rate=0.5, row_to_prune_idx=row)

    assert model_with_rows.last_prune_outcome["mode"] == "structural"
    assert model_with_rows.model.conv2.out_channels == 12
    assert model_with_rows.model.fc.in_features == 12
    model_with_rows.model.eval()(torch.randn(2, 3, 32, 32))


def test_channel_reducing_mean_still_blocks():
    """A reduction over the channel axis genuinely destroys the layout and must block."""
    import src.channel_groups as channel_groups

    class ChannelMean(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(1024, 10)

        def forward(self, x):
            # Averaging over channels collapses the dimension the group would resize
            return self.fc(self.conv(x).mean(dim=1).flatten(1))

    groups = channel_groups.build_channel_groups(ChannelMean().eval())
    assert groups is not None
    assert any(not g.prunable and "reduces the channel dimension" in g.reason for g in groups)


def test_fallback_reason_distinguishes_its_causes():
    """
    'no dependency group resolved' previously covered four unrelated failures, so the logs
    could not say whether the library needed a new trace rule or nothing at all.
    """
    from src.NetworkEnv import prune_current_model

    class Untraceable(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            x = self.conv(x)
            if x.mean() > 0:  # data-dependent control flow defeats symbolic tracing
                x = x * 2
            return self.fc(self.pool(x).flatten(1))

    model_with_rows = ModelWithRows(Untraceable().eval())
    prune_current_model(model_with_rows, compression_rate=0.5, row_to_prune_idx=0)

    outcome = model_with_rows.last_prune_outcome
    assert outcome["mode"] == "masked"
    assert "traceable" in outcome["reason"] or "fx trace raised" in outcome["reason"]


class MiniShuffle(nn.Module):
    """ShuffleNet-V2 stride-1 block: stem, chunk, cat, channel_shuffle."""

    def __init__(self, channels=8):
        super().__init__()
        half = channels // 2
        self.stem = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.pw = nn.Conv2d(half, half, 1, bias=False)
        self.bn = nn.BatchNorm2d(half)
        self.dw = nn.Conv2d(half, half, 3, padding=1, groups=half, bias=False)
        self.bn_dw = nn.BatchNorm2d(half)
        self.head = nn.Conv2d(channels, 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.stem(x)
        x1, x2 = x.chunk(2, dim=1)
        y = torch.nn.functional.relu(self.bn(self.pw(x2)))
        y = torch.nn.functional.relu(self.bn_dw(self.dw(y)))
        out = torch.cat((x1, y), dim=1)
        b, c, h, w = out.size()
        out = out.view(b, 2, c // 2, h, w).transpose(1, 2).contiguous().view(b, -1, h, w)
        return self.fc(self.pool(self.head(out)).flatten(1))


def test_shufflenet_chunk_cat_shuffle_is_structural():
    from src.NetworkEnv import prune_current_model

    model = MiniShuffle(8).eval()
    sample = torch.randn(2, 3, 8, 8)
    params_before = sum(p.numel() for p in model.parameters())
    groups = channel_groups.build_channel_groups(model)
    assert groups is not None
    pw_group = channel_groups.group_of(groups, model.pw)
    assert pw_group is not None
    assert pw_group.prunable, pw_group.reason

    model_with_rows = ModelWithRows(model)
    conv_idx = next(idx for idx, layer in enumerate(model_with_rows.all_layers) if layer is model.pw)
    prune_current_model(model_with_rows, 0.5, _row_index_of_layer(model_with_rows, conv_idx))
    pruned = model_with_rows.model.eval()
    assert model_with_rows.last_prune_outcome["mode"] == "structural", model_with_rows.last_prune_outcome
    assert pruned.pw.out_channels == 2
    assert sum(p.numel() for p in pruned.parameters()) < params_before
    assert pruned(sample).shape == (2, 2)


def test_chenyaofo_shufflenetv2_prunes_without_masking_every_layer():
    from src.NetworkEnv import prune_current_model
    from spectra_models_instantiation.shufflenetv2_chenyaofo import shufflenetv2x1

    sample = torch.randn(1, 3, 32, 32)
    template = shufflenetv2x1(num_classes=10, large_input=False).eval()
    groups = channel_groups.build_channel_groups(template)
    assert groups is not None
    blocked = [g.reason for g in groups if not g.prunable and (g.producers or g.depthwise)]
    prunable = [g for g in groups if g.prunable]
    assert prunable, f"no prunable groups; blocked={blocked[:8]}"

    structural = 0
    masked = 0
    num_rows = len(ModelWithRows(template).row_to_main_layer)
    for row_idx in range(min(num_rows, 8)):
        model = shufflenetv2x1(num_classes=10, large_input=False).eval()
        params_before = sum(p.numel() for p in model.parameters())
        model_with_rows = ModelWithRows(model)
        prune_current_model(model_with_rows, 0.8, row_idx)
        model = model_with_rows.model.eval()
        mode = model_with_rows.last_prune_outcome["mode"]
        if mode == "structural":
            structural += 1
            assert sum(p.numel() for p in model.parameters()) < params_before
        elif mode == "masked":
            masked += 1
        assert model(sample).shape == (1, 10)
    assert structural >= 1, f"structural={structural} masked={masked} blocked={blocked[:8]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
