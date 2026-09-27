"""
``SPECTRA_GROUP_ONCE_PER_PASS``: a coupled group is structurally cut at most once per pass.

CPU only, no datasets. Uses the real thin CIFAR ResNet family and the real prune path
(``NetworkEnv.prune_current_model``) so the recorded owner indices are the ones the
environment will lock.

    python -m pytest tests/test_group_once.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import fortify  # noqa: E402
from src.Configuration.StaticConf import StaticConf  # noqa: E402
from src.NetworkEnv import NetworkEnv, group_owner_indices, prune_current_model  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402

RATES = {0: 1.0, 1: 0.9, 2: 0.8}


def _layer_index(mwr, module):
    return next(i for i, layer in enumerate(mwr.all_layers) if layer is module)


def _row_of_layer(mwr, layer_idx):
    return next(row for row, main in mwr.row_to_main_layer.items() if main == layer_idx)


def _bare_env(model):
    """A NetworkEnv with only what legal_action_mask / lock bookkeeping touch."""
    env = NetworkEnv.__new__(NetworkEnv)
    env.conf = SimpleNamespace(
        compression_rates_dict=RATES,
        device=StaticConf.get_instance().conf_values.device)
    env.current_model = model
    env.row_idx = 1
    env._pass_locked_layers = set()
    return env


def test_group_once_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SPECTRA_GROUP_ONCE_PER_PASS", raising=False)
    assert fortify.group_once_per_pass() is False
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
    assert fortify.group_once_per_pass() is True


def test_legal_mask_force_identity_kwarg(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    free = fortify.legal_action_mask(RATES, row_index=5, alive_count=16, device="cpu")
    assert free.tolist() == [True, True, True]
    forced = fortify.legal_action_mask(RATES, row_index=5, alive_count=16, device="cpu",
                                       force_identity=True)
    assert forced.tolist() == [True, False, False]


def test_structural_group_prune_reports_every_owner_row():
    """Cutting layer1[0].conv2 resizes the whole stage-1 stream; owners = stem + 3x conv2."""
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = ModelWithRows(model)
    stem = model.embed[0]
    conv2s = [blk.conv2 for blk in model.layer1]
    conv1s = [blk.conv1 for blk in model.layer1]
    expected_owners = sorted(_layer_index(mwr, m) for m in [stem] + conv2s)
    not_owners = {_layer_index(mwr, m) for m in conv1s}

    groups = channel_groups.build_channel_groups(model)
    group = channel_groups.group_of(groups, conv2s[0])
    assert group is not None and group.prunable
    assert group_owner_indices(mwr, group) == expected_owners

    row = _row_of_layer(mwr, _layer_index(mwr, conv2s[0]))
    prune_current_model(mwr, 0.8, row, quiet=True, record=False, input_shape=(3, 32, 32))
    outcome = mwr.last_prune_outcome
    assert outcome["mode"] == "structural"
    assert outcome["group_layer_indices"] == expected_owners
    assert not (set(outcome["group_layer_indices"]) & not_owners)
    assert mwr.model.layer1[2].conv2.out_channels == 3  # 4 -> round(3.2) = 3 on every owner


def test_group_lock_forces_identity_on_later_owner_rows_only_when_on(monkeypatch):
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    env = _bare_env(model)
    mwr = ModelWithRows(model)
    conv2_b0 = model.layer1[0].conv2
    conv2_b1 = model.layer1[1].conv2
    conv1_b1 = model.layer1[1].conv1
    row_b0 = _row_of_layer(mwr, _layer_index(mwr, conv2_b0))

    # Cut the stage-1 stream once through the real prune path.
    prune_current_model(mwr, 0.8, row_b0, quiet=True, record=False, input_shape=(3, 32, 32))
    env.current_model = mwr.model
    outcome = dict(mwr.last_prune_outcome)

    # Switch off: nothing is remembered, later owner rows stay fully legal.
    monkeypatch.delenv("SPECTRA_GROUP_ONCE_PER_PASS", raising=False)
    env._register_group_lock(outcome)
    assert env._pass_locked_layers == set()
    fresh = ModelWithRows(env.current_model)
    env.row_idx = _row_of_layer(fresh, _layer_index(fresh, fresh.model.layer1[1].conv2)) + 1
    assert env.legal_action_mask(device="cpu").tolist() == [True, True, True]  # width 3: 0.9/0.8 -> 2

    # Switch on: later conv2 rows of the same stream are identity-only; conv1 rows are not.
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
    env._register_group_lock(outcome)
    assert set(outcome["group_layer_indices"]) <= env._pass_locked_layers
    env.row_idx = _row_of_layer(fresh, _layer_index(fresh, fresh.model.layer1[1].conv2)) + 1
    assert env.legal_action_mask(device="cpu").tolist() == [True, False, False]
    env.row_idx = _row_of_layer(fresh, _layer_index(fresh, fresh.model.layer1[1].conv1)) + 1
    assert env.legal_action_mask(device="cpu").tolist() == [True, True, True]
    # A block-internal conv1 is a singleton group: its cut locks only itself.
    del conv2_b1, conv1_b1


def test_masked_fallback_does_not_lock_anything(monkeypatch):
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    env = _bare_env(model)
    env._register_group_lock({"mode": "masked", "group_layer_indices": [3, 5]})
    env._register_group_lock({"mode": "identity"})
    env._register_group_lock(None)
    assert env._pass_locked_layers == set()


def test_locks_release_at_pass_boundary(monkeypatch):
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    env = _bare_env(model)
    env._register_group_lock({"mode": "structural", "group_layer_indices": [0, 5, 9, 13]})
    assert env._pass_locked_layers == {0, 5, 9, 13}
    assert env._end_of_pass_reset(num_actions=7, num_rows=21) is False
    assert env._pass_locked_layers == {0, 5, 9, 13}
    assert env._end_of_pass_reset(num_actions=21, num_rows=21) is True
    assert env._pass_locked_layers == set()
    assert env._end_of_pass_reset(num_actions=0, num_rows=0) is False


def test_group_once_walk_cuts_each_stream_once_per_pass(monkeypatch):
    """Full row walk over thin r20-w4 with the greedy 0.8 picker, group-once on vs off."""
    def run(group_once: bool):
        if group_once:
            monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
        else:
            monkeypatch.delenv("SPECTRA_GROUP_ONCE_PER_PASS", raising=False)
        model = resnet20(num_classes=10, large_input=False, width=4).eval()
        env = _bare_env(model)
        mwr = ModelWithRows(model)
        num_rows = len(mwr.all_rows) - 1
        for row in range(num_rows):
            env.row_idx = row + 1
            legal = env.legal_action_mask(device="cpu")
            prune_idx = [i for i in legal.nonzero(as_tuple=False).flatten().tolist() if RATES[i] < 1.0]
            if not prune_idx:
                continue
            rate = min(RATES[i] for i in prune_idx)
            mwr = ModelWithRows(env.current_model)
            prune_current_model(mwr, rate, row, quiet=True, record=False, input_shape=(3, 32, 32))
            env.current_model = mwr.model
            env._register_group_lock(dict(mwr.last_prune_outcome or {}))
        env._end_of_pass_reset(num_rows, num_rows)
        m = env.current_model
        streams = [m.layer1[0].conv2.out_channels, m.layer2[0].conv2.out_channels,
                   m.layer3[0].conv2.out_channels]
        assert m(torch.zeros(1, 3, 32, 32)).shape == (1, 10)
        return streams, sum(p.numel() for p in m.parameters())

    origin = sum(p.numel() for p in resnet20(num_classes=10, large_input=False, width=4).parameters())
    streams_once, params_once = run(True)
    streams_walk, params_walk = run(False)
    # r20-w4 streams are 4 / 8 / 16. One 0.8 cut each: 3 / 6 / 13.
    assert streams_once == [3, 6, 13]
    # The plain walk re-cuts the stream on every owning row (3 conv2 + downsample).
    assert streams_walk[0] <= 2 and streams_walk[1] <= 4 and streams_walk[2] <= 10
    assert params_once > params_walk
    assert params_once / origin < 0.80
