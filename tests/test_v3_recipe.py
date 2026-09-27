"""
16 Sep v3 last-train recipe: learning governor (probe / min lifetime / rewind), per-layer
group-cost state, train-only tau, passes replay pin.

CPU only.  python -m pytest tests/test_v3_recipe.py -v
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf, ResidualNet  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import fortify  # noqa: E402
from src.fortify import LearningGovernor  # noqa: E402
import src.action_costs as action_costs  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.utils as utils  # noqa: E402
from src.BERTInputModeler import (BERTInputModeler, TOKEN_BASE_DIM, token_feature_dim,  # noqa: E402
                                  action_cost_slot_dim)
from src.NetworkEnv import NetworkEnv, prune_current_model  # noqa: E402


# ------------------------------------------------------------------ governor

def test_governor_defaults_reproduce_v2_stop_rule():
    """No probe, no rewind: stop only after patience episodes without a new batch max."""
    g = LearningGovernor(min_episodes=100, patience=100, rewind=False, rewind_patience=50, rewind_max=3)
    assert g.observe(0.20, 4)["new_best"] is True
    for _ in range(24):                       # 96 stale episodes
        assert g.observe(0.19, 4)["new_best"] is False
    assert g.should_stop(100) is False        # 96 < 100
    g.observe(0.10, 4)                        # 100 stale
    assert g.should_stop(100) is True
    assert g.rewinds == 0


def test_governor_min_lifetime_blocks_early_stop():
    g = LearningGovernor(min_episodes=250, patience=100, rewind=False, rewind_patience=50, rewind_max=3)
    g.observe(0.30, 4)
    for _ in range(40):
        g.observe(0.10, 4)                    # 160 stale episodes > patience
    assert g.since_improvement == 160
    assert g.should_stop(164) is False        # lifetime not reached
    assert g.should_stop(250) is True


def test_governor_none_score_only_advances_counters():
    g = LearningGovernor(min_episodes=0, patience=8, rewind=False, rewind_patience=50, rewind_max=3)
    g.observe(0.5, 4)
    v = g.observe(None, 4)
    assert v["new_best"] is False and v["since"] == 4
    g.observe(None, 4)
    assert g.should_stop(12) is True


def test_governor_rewind_fires_after_probe_stale_then_caps():
    g = LearningGovernor(min_episodes=0, patience=1000, rewind=True, rewind_patience=50, rewind_max=2)
    assert g.observe(0.30, 4)["rewind"] is False          # first elite
    fired = []
    for i in range(60):                                    # 240 stale episodes
        v = g.observe(0.25, 4)
        if v["rewind"]:
            fired.append((i + 1) * 4)
    # first rewind once 50 stale episodes accrued, second 50 later, then capped at rewind_max=2
    assert fired[:2] == [52, 104]
    assert len(fired) == 2
    assert g.rewinds == 2
    # overall patience keeps counting through rewinds (stop is still governed by patience)
    assert g.since_improvement == 240


def test_governor_no_rewind_without_elite_or_when_improving():
    g = LearningGovernor(min_episodes=0, patience=1000, rewind=True, rewind_patience=8, rewind_max=3)
    for _ in range(5):
        assert g.observe(None, 4)["rewind"] is False       # nothing to return to
    assert g.observe(0.1, 4)["new_best"] is True
    v = g.observe(0.2, 4)                                  # improving: never rewinds
    assert v["new_best"] and not v["rewind"]


def test_governor_resumes_from_saved_state():
    g = LearningGovernor(min_episodes=0, patience=10, rewind=True, rewind_patience=50, rewind_max=3,
                         best_score=0.31, since_improvement=8)
    assert g.has_elite and g.best_score == 0.31
    g.observe(0.30, 4)
    assert g.since_improvement == 12 and g.should_stop(0) is True


# ------------------------------------------------------------------ v3 flags

def test_v3_flags_default_off(monkeypatch):
    for k in ("SPECTRA_STATE_GROUPCOST", "SPECTRA_PROBE_EVERY", "SPECTRA_REWIND_BEST",
              "SPECTRA_TRAIN_TAU", "SPECTRA_MIN_EPISODES", "SPECTRA_PATIENCE_EPISODES"):
        monkeypatch.delenv(k, raising=False)
    assert fortify.state_groupcost() is False
    assert fortify.probe_every() == 0
    assert fortify.rewind_best() is False
    assert fortify.train_tau(10.0) == 10.0
    assert fortify.min_episodes(100) == 100
    assert fortify.patience_episodes(100) == 100
    monkeypatch.setenv("SPECTRA_TRAIN_TAU", "5")
    assert fortify.train_tau(10.0) == 5.0


def test_groupcost_adds_four_token_channels(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    for k in ("SPECTRA_STATE_GROUPCOST", "SPECTRA_STATE_SLACK", "SPECTRA_BUDGET_IN_STATE"):
        monkeypatch.delenv(k, raising=False)
    base = token_feature_dim(5)
    monkeypatch.setenv("SPECTRA_STATE_GROUPCOST", "1")
    assert token_feature_dim(5) == base + fortify.STATE_GROUPCOST_DIM


# ------------------------------------------------------------------ group-cost features

def test_group_cost_features_on_residual_net():
    torch.manual_seed(0)
    model = ResidualNet().eval()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    feats = action_costs.group_cost_features(model, mwr.all_layers, (3, 8, 8), groups=groups)
    L = len(mwr.all_layers)
    assert feats.shape == (L, action_costs.GROUPCOST_FEATURE_DIM)
    idx = {id(m): i for i, m in enumerate(mwr.all_layers)}
    stem, conv1, conv2 = idx[id(model.stem)], idx[id(model.block.conv1)], idx[id(model.block.conv2)]
    bn1 = idx[id(model.block.bn1)]
    # stem and block.conv2 own the same residual stream: identical rows, 2 owners
    assert torch.allclose(feats[stem], feats[conv2])
    assert feats[stem, 2] == pytest.approx(1.0)          # 2 owners / max 2
    assert 0.0 < feats[stem, 0] < 1.0 and 0.0 < feats[stem, 1] < 1.0
    # conv1 is a singleton group: one owner, smaller param share than the stream
    assert feats[conv1, 2] == pytest.approx(0.5)
    assert feats[conv1, 0] < feats[stem, 0]
    # norms/activations do not own a dimension
    assert torch.count_nonzero(feats[bn1]) == 0
    # no cuts yet
    assert torch.count_nonzero(feats[:, 3]) == 0
    # A conv weight scales with both its input and its output width, so group shares
    # legitimately overlap (each is "what removing this whole group would take away").
    assert 0.0 < feats[conv1, 0] <= 1.0 and 0.0 < feats[conv1, 1] <= 1.0


def test_group_cost_cut_counter_follows_structural_prunes(monkeypatch):
    monkeypatch.delenv("SPECTRA_GROUP_ONCE_PER_PASS", raising=False)
    torch.manual_seed(0)
    model = ResidualNet().eval()
    env = NetworkEnv.__new__(NetworkEnv)
    env._episode_group_cuts = {}
    env._pass_locked_layers = set()
    mwr = ModelWithRows(model)
    idx = {id(m): i for i, m in enumerate(mwr.all_layers)}
    stem_idx = idx[id(model.stem)]                       # resolve before the prune replaces modules
    row = next(r for r, main in mwr.row_to_main_layer.items() if main == idx[id(model.block.conv2)])
    prune_current_model(mwr, 0.8, row, quiet=True, record=False, input_shape=(3, 8, 8))
    env._register_group_lock(dict(mwr.last_prune_outcome))
    cuts = env.episode_group_cuts()
    assert len(cuts) == 1 and list(cuts.values()) == [1]
    key = next(iter(cuts))
    assert stem_idx in key                               # stem is an owner of the stream
    # a second cut of the same stream counts 2; group-once stayed off (no locks)
    env._register_group_lock({"mode": "structural", "group_layer_indices": sorted(key)})
    assert env.episode_group_cuts()[key] == 2
    assert env._pass_locked_layers == set()
    # masked / identity outcomes do not count
    env._register_group_lock({"mode": "masked", "group_layer_indices": [1]})
    assert len(env.episode_group_cuts()) == 1
    # the feature column reflects the count (min(1, n/2))
    groups = channel_groups.build_channel_groups(mwr.model)
    feats = action_costs.group_cost_features(mwr.model, mwr.all_layers, (3, 8, 8), groups=groups,
                                             episode_cuts=env.episode_group_cuts())
    stream_rows = [i for i in key if i < feats.shape[0]]
    assert all(feats[i, 3] == pytest.approx(1.0) for i in stream_rows)


def test_layer_extras_land_in_tokens_when_flag_on(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_STATE_GROUPCOST", "1")
    monkeypatch.delenv("SPECTRA_STATE_SLACK", raising=False)
    monkeypatch.delenv("SPECTRA_BUDGET_IN_STATE", raising=False)
    builder = BERTInputModeler()
    L = 3
    maps = {"Topology": [[2, 3, 8, 3, 1, 1, 1]] * L, "Activations": [[0.1] * 12] * L,
            "Weights": [[0.2] * 19] * L}
    extras = torch.tensor([[0.4, 0.3, 1.0, 0.5], [0.1, 0.05, 0.5, 0.0], [0, 0, 0, 0]], dtype=torch.float32)
    tokens = builder._build_layer_tokens(maps, 0, action_costs=None, coupling_ids=torch.arange(L),
                                         layer_extras=extras)
    slot = action_cost_slot_dim()
    assert tokens.shape[1] == TOKEN_BASE_DIM + fortify.FORTIFY_TOKEN_DIM + 4 + slot
    assert torch.allclose(tokens[:, -slot - 4: -slot], extras)
    # missing extras -> zeros, same width
    tokens0 = builder._build_layer_tokens(maps, 0, action_costs=None, coupling_ids=torch.arange(L))
    assert tokens0.shape == tokens.shape
    assert torch.count_nonzero(tokens0[:, -slot - 4: -slot]) == 0


# ------------------------------------------------------------------ train tau / reward

def test_compute_reward_honours_explicit_tau(monkeypatch):
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "neon")
    monkeypatch.delenv("SPECTRA_REWARD_SCALE", raising=False)
    # -7 pp drop: inside the configured tau=5? No -> over budget with tau=5, in-band with tau=10
    over = utils.compute_reward(0.83, 0.90, 0.8, tau=5.0)
    inband = utils.compute_reward(0.83, 0.90, 0.8, tau=10.0)
    assert over == pytest.approx(-(20.0 ** 3))
    assert inband == pytest.approx(20.0)


def test_env_tau_is_train_only(monkeypatch):
    from src.NetworkEnv import AGENT_TRAIN, EVAL_TEST
    env = NetworkEnv.__new__(NetworkEnv)
    env.conf = SimpleNamespace(allowed_acc_reduction=10)
    monkeypatch.setenv("SPECTRA_TRAIN_TAU", "5")
    env.mode = AGENT_TRAIN
    assert env.tau() == 5.0
    env.mode = EVAL_TEST
    assert env.tau() == 10.0
    monkeypatch.delenv("SPECTRA_TRAIN_TAU", raising=False)
    env.mode = AGENT_TRAIN
    assert env.tau() == 10.0


# ------------------------------------------------------------------ probe nets / passes pin

def test_probe_nets_resolution(monkeypatch):
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    agent = A2CAgentReinforce.__new__(A2CAgentReinforce)
    paths = ["/x/resnet20-width10_cifar10.pt", "/x/resnet56-width6_cifar10.pt", "/x/vgg16_bn.pt"]
    agent.env = SimpleNamespace(data_dict={p: (None, None) for p in paths})
    monkeypatch.delenv("SPECTRA_PROBE_EVERY", raising=False)
    assert agent.probe_nets() == []
    monkeypatch.setenv("SPECTRA_PROBE_EVERY", "12")
    monkeypatch.setenv("SPECTRA_PROBE_NETS", "resnet56-width6,resnet20-width10")
    assert agent.probe_nets() == [paths[1], paths[0]]
    monkeypatch.setenv("SPECTRA_PROBE_NETS", "nomatch")
    assert agent.probe_nets() == []


def test_apply_policy_config_pins_passes_softly(tmp_path, monkeypatch):
    import a2c_agent_reinforce_runner as runner
    ckpt = tmp_path / "agent_checkpoints"
    ckpt.mkdir()
    (ckpt / "latest_best_actor.pt").write_bytes(b"x")
    (ckpt / "policy_config.json").write_text(json.dumps({
        "compression_rates": [1.0, 0.9, 0.8], "action_rankings": [None, None, None],
        "env": {"SPECTRA_STATE_GROUPCOST": "1"}, "passes": 2}), encoding="utf-8")
    monkeypatch.delenv("SPECTRA_POLICY_CONFIG", raising=False)
    monkeypatch.delenv("SPECTRA_EVAL_PASSES", raising=False)
    # setenv (not delenv) so monkeypatch restores the pre-test state after the runner writes it
    monkeypatch.setenv("SPECTRA_STATE_GROUPCOST", "0")
    args = SimpleNamespace(actor_checkpoint_path=str(ckpt / "latest_best_actor.pt"),
                           compression_rates=[1.0, 0.9, 0.8], action_rankings=None, passes=1)
    runner.apply_policy_config(args)
    assert args.passes == 2
    assert os.environ["SPECTRA_STATE_GROUPCOST"] == "1"
    # explicit SPECTRA_EVAL_PASSES wins
    monkeypatch.setenv("SPECTRA_EVAL_PASSES", "1")
    args2 = SimpleNamespace(actor_checkpoint_path=str(ckpt / "latest_best_actor.pt"),
                            compression_rates=[1.0, 0.9, 0.8], action_rankings=None, passes=1)
    runner.apply_policy_config(args2)
    assert args2.passes == 1
