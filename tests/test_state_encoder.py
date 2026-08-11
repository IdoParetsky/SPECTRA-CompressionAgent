"""
State-encoder checks that do not need pretrained weights downloaded.

    python -m pytest tests/test_state_encoder.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf, SmallVGG, ResidualNet  # noqa: E402

_init_static_conf()

from src.Model.StateEncoder import SpectraStateEncoder, NUM_LAYER_TYPES  # noqa: E402
from src.BERTInputModeler import token_feature_dim, TOKEN_BASE_DIM  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402

FEATURE_DIM = token_feature_dim()  # base + 2 * num_actions action-cost slots


def _fake_state(num_layers=12, target=3, blocks=None, action_costs=None):
    coupling = blocks if blocks is not None else torch.arange(num_layers) // 3
    state = {
        "layer_features": torch.randn(num_layers, FEATURE_DIM),
        "layer_types": torch.randint(0, NUM_LAYER_TYPES, (num_layers,)),
        "coupling_ids": coupling,
        "block_ids": coupling,
        "target_index": target,
    }
    if action_costs is not None:
        state["action_costs"] = action_costs
    return state


def _costs(param_fractions):
    rates = torch.linspace(1.0, 0.6, len(param_fractions))
    return torch.stack([rates, torch.tensor(param_fractions), torch.tensor(param_fractions)], dim=1)


def test_token_layout_matches_moments_and_action_slots():
    """Guard against silent drift between BaseFE moment count and TOKEN_BASE_DIM."""
    from NetworkFeatureExtraction.src.FeatureExtractors.BaseFE import BaseFE
    assert len(BaseFE.MOMENT_NAMES) == 12
    assert len(BaseFE.SHAPE_NAMES) == 7
    assert TOKEN_BASE_DIM == 7 + 12 + 12 + 7
    assert FEATURE_DIM == TOKEN_BASE_DIM + 2 * 5  # StaticConf has 5 rates


def test_encoder_returns_a_fixed_width_state():
    encoder = SpectraStateEncoder(FEATURE_DIM, d_model=64, nhead=4, num_layers=2)
    out = encoder(_fake_state())
    assert out.shape == (1, 64)
    assert torch.isfinite(out).all()


def test_encoder_handles_networks_of_any_depth():
    """No 512-position ceiling on the trainable baseline."""
    encoder = SpectraStateEncoder(FEATURE_DIM, d_model=64, nhead=4, num_layers=2)
    for depth in (1, 7, 200, 700):
        out = encoder(_fake_state(num_layers=depth, target=min(depth - 1, 5)))
        assert out.shape == (1, 64)


def test_target_marker_changes_the_state():
    """Entity marker: the layer under consideration must be distinguishable."""
    encoder = SpectraStateEncoder(FEATURE_DIM, d_model=64, nhead=4, num_layers=2).eval()
    with torch.no_grad():
        encoder.target_marker.normal_()
        state = _fake_state(target=2)
        first = encoder(state)
        second = encoder({**state, "target_index": 7})
    assert not torch.allclose(first, second)


def test_encoder_is_trainable():
    """Unlike the frozen encoder, gradients must reach the representation itself."""
    encoder = SpectraStateEncoder(FEATURE_DIM, d_model=64, nhead=4, num_layers=2)
    encoder(_fake_state()).sum().backward()

    assert encoder.input_proj[0].weight.grad is not None
    assert encoder.block_affinity.grad is not None
    assert any(p.grad is not None for p in encoder.encoder.parameters())


def test_parameter_count_is_a_small_fraction_of_bert_base():
    encoder = SpectraStateEncoder(FEATURE_DIM)
    total = sum(p.numel() for p in encoder.parameters())
    assert total < 15_000_000, f"encoder unexpectedly large: {total:,}"


def test_action_costs_change_the_state():
    encoder = SpectraStateEncoder(FEATURE_DIM, d_model=64, nhead=4, num_layers=2).eval()
    base = _fake_state()

    with torch.no_grad():
        cheap = encoder({**base, "action_costs": _costs([0.0, 0.01, 0.02, 0.03, 0.04])})
        costly = encoder({**base, "action_costs": _costs([0.0, 0.20, 0.35, 0.50, 0.65])})

    assert not torch.allclose(cheap, costly)


def test_state_is_still_produced_without_action_costs():
    encoder = SpectraStateEncoder(FEATURE_DIM, d_model=64, nhead=4, num_layers=2)
    assert encoder(_fake_state()).shape == (1, 64)


def test_action_cost_projection_receives_gradient():
    encoder = SpectraStateEncoder(FEATURE_DIM, d_model=64, nhead=4, num_layers=2)
    encoder(_fake_state(action_costs=_costs([0.0, 0.1, 0.2, 0.3, 0.4]))).sum().backward()
    assert encoder.action_proj[0].weight.grad is not None


def test_coupling_ids_share_residual_producers():
    """Channel-group coupling, not parent-module heuristics, drives the structural bias."""
    model = ResidualNet().eval()
    model_with_rows = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    assert groups is not None

    ids = channel_groups.coupling_ids_for_layers(model_with_rows.all_layers, groups)
    assert ids.numel() == len(model_with_rows.all_layers)

    # Identity path and block.conv2 are merged by the residual add → same coupling id
    # among the producers of that group (see ResidualNet in test_pruning.py)
    group = channel_groups.group_of(groups, model.block.conv2)
    assert group is not None
    producer_ids = []
    for layer, cid in zip(model_with_rows.all_layers, ids.tolist()):
        if any(layer is p for p in group.producers) or any(layer is d for d in group.depthwise):
            producer_ids.append(cid)
    assert len(set(producer_ids)) == 1


def test_weight_statistics_are_fixed_width():
    """No per-filter token explosion — every layer emits the same vector width."""
    from NetworkFeatureExtraction.src.FeatureExtractors.WeightStatisticsFE import WeightStatisticsFE
    from NetworkFeatureExtraction.src.FeatureExtractors.BaseFE import BaseFE

    fe = WeightStatisticsFE(torch.device("cpu"))
    model_with_rows = ModelWithRows(SmallVGG())
    maps = fe.extract_feature_map(model_with_rows)
    expected = len(BaseFE.MOMENT_NAMES) + len(BaseFE.SHAPE_NAMES)
    assert all(len(row) == expected for row in maps)


def test_scale_exponent_replaced_by_abs_p10():
    from NetworkFeatureExtraction.src.FeatureExtractors.BaseFE import BaseFE
    assert "scale_exponent" not in BaseFE.MOMENT_NAMES
    assert "abs_p10" in BaseFE.MOMENT_NAMES
    assert "p25" in BaseFE.MOMENT_NAMES and "median" in BaseFE.MOMENT_NAMES


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
