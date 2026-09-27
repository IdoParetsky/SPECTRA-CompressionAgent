"""
13 Sep v2 training recipe: slack state, (rate, ranking) actions, PPO update, policy contract.

CPU only. The PPO test is the property the A2C loop never had: on a synthetic reward that
prefers one action, the update must move an exactly-uniform policy off uniform.

    python -m pytest tests/test_v2_recipe.py -v
"""

import json
import math
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
import src.channel_groups as channel_groups  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.utils as utils  # noqa: E402
from src.BERTInputModeler import (BERTInputModeler, TOKEN_BASE_DIM, token_feature_dim,  # noqa: E402
                                  action_cost_slot_dim)
from src.feature_standardizer import cache_path_from_actor  # noqa: E402

RATES3 = {0: 1.0, 1: 0.9, 2: 0.8}


# ------------------------------------------------------------------ state: slack + progress

def test_state_slack_off_by_default_and_adds_two_channels(monkeypatch):
    monkeypatch.delenv("SPECTRA_STATE_SLACK", raising=False)
    monkeypatch.delenv("SPECTRA_BUDGET_IN_STATE", raising=False)
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    base = token_feature_dim(3)
    monkeypatch.setenv("SPECTRA_STATE_SLACK", "1")
    assert fortify.state_slack() is True
    assert token_feature_dim(3) == base + fortify.STATE_SLACK_DIM


def test_accuracy_slack_semantics():
    assert fortify.accuracy_slack(0.0, 10.0) == 1.0          # untouched band
    assert fortify.accuracy_slack(-10.0, 10.0) == 0.0        # exactly at tau
    assert fortify.accuracy_slack(-5.0, 10.0) == pytest.approx(0.5)
    assert fortify.accuracy_slack(-25.0, 10.0) == -1.0       # clipped over-budget
    assert fortify.accuracy_slack(+8.0, 10.0) == 1.0         # gains clip at 1


def test_slack_columns_are_broadcast_to_every_token(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_STATE_SLACK", "1")
    monkeypatch.delenv("SPECTRA_BUDGET_IN_STATE", raising=False)
    builder = BERTInputModeler()
    L = 4
    maps = {
        "Topology": [[2, 3, 8, 3, 1, 1, 1]] * L,
        "Activations": [[0.1] * 12] * L,
        "Weights": [[0.2] * 19] * L,
    }
    tokens = builder._build_layer_tokens(maps, 1, action_costs=None, coupling_ids=torch.arange(L),
                                         param_ratio=0.8, extras=[0.35, 0.6])
    slot_dim = action_cost_slot_dim()  # sized from the StaticConf menu installed by the test conf
    assert tokens.shape[1] == TOKEN_BASE_DIM + fortify.FORTIFY_TOKEN_DIM + 2 + slot_dim
    slack_cols = tokens[:, -slot_dim - 2: -slot_dim]
    assert slack_cols.shape == (L, 2)
    assert torch.allclose(slack_cols[:, 0], torch.full((L,), 0.35))
    assert torch.allclose(slack_cols[:, 1], torch.full((L,), 0.6))
    # default extras = untouched band, no progress
    tokens0 = builder._build_layer_tokens(maps, 1, action_costs=None, coupling_ids=torch.arange(L))
    assert torch.allclose(tokens0[:, -slot_dim - 2: -slot_dim], torch.tensor([[1.0, 0.0]] * L))


# ------------------------------------------------------------------ action = (rate, ranking)

def test_parse_action_rankings_default_and_validation():
    rates = utils.parse_compression_rates([1.0, 0.9, 0.8])
    assert utils.parse_action_rankings(None, rates) == {0: None, 1: None, 2: None}
    with pytest.raises(ValueError):
        utils.parse_action_rankings(["l1", "fpgm"], rates)
    rates5 = utils.parse_compression_rates([1.0, 0.9, 0.8, 0.9, 0.8])
    parsed = utils.parse_action_rankings(["l1", "l1", "l1", "fpgm", "fpgm"], rates5)
    assert parsed == {0: None, 1: "l1", 2: "l1", 3: "fpgm", 4: "fpgm"}
    parsed_none = utils.parse_action_rankings(["none", "none", "none", "geometric_median", "FPGM"], rates5)
    assert parsed_none == {0: None, 1: None, 2: None, 3: "fpgm", 4: "fpgm"}


def _conv_with_duplicates():
    """12 near-identical large filters + 4 distinct small ones: L1 and FPGM must disagree."""
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(8, 16, 3)
    with torch.no_grad():
        base = torch.randn(8, 3, 3)
        for i in range(12):
            conv.weight[i] = 3.0 * base + 1e-3 * torch.randn(8, 3, 3)
        for i in range(12, 16):
            conv.weight[i] = torch.randn(8, 3, 3)
    return conv


def test_explicit_ranking_overrides_env_default(monkeypatch):
    monkeypatch.delenv("SPECTRA_FILTER_IMPORTANCE", raising=False)
    conv = _conv_with_duplicates()
    keep_l1 = set(pruning.select_surviving_filters(conv, 0.5).tolist())
    keep_fpgm = set(pruning.select_surviving_filters(conv, 0.5, mode="fpgm").tolist())
    assert len(keep_l1) == len(keep_fpgm) == 8
    assert keep_l1 <= set(range(12))                    # L1 keeps the big duplicates
    assert {12, 13, 14, 15} <= keep_fpgm                # FPGM keeps the distinct ones
    assert keep_l1 != keep_fpgm
    l1 = pruning.filter_importance(conv)
    assert torch.allclose(pruning.filter_importance(conv, mode=None), l1)
    assert torch.allclose(pruning.filter_importance(conv, mode="l1"), l1)


def test_group_survivors_follow_the_action_ranking(monkeypatch):
    monkeypatch.delenv("SPECTRA_FILTER_IMPORTANCE", raising=False)
    torch.manual_seed(1)
    model = ResidualNet().eval()
    groups = channel_groups.build_channel_groups(model)
    group = channel_groups.group_of(groups, model.block.conv2)
    assert group is not None and group.prunable
    keep_l1 = pruning.select_group_survivors(group, 0.5)
    keep_l1_explicit = pruning.select_group_survivors(group, 0.5, mode="l1")
    keep_fpgm = pruning.select_group_survivors(group, 0.5, mode="fpgm")
    assert keep_l1.numel() == keep_fpgm.numel() == 8
    assert torch.equal(keep_l1, keep_l1_explicit)
    votes_l1 = pruning.group_importance(group)
    votes_fpgm = pruning.group_importance(group, mode="fpgm")
    assert votes_l1.shape == votes_fpgm.shape == (16,)
    assert not torch.allclose(votes_l1 / votes_l1.max(), votes_fpgm / votes_fpgm.max())


def test_masks_and_identity_index_tolerate_duplicate_rates(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    rates5 = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.9, 4: 0.8}
    mask = fortify.legal_action_mask(rates5, row_index=3, alive_count=16, device="cpu")
    assert mask.tolist() == [True] * 5
    # width 4: both 0.9 and 0.8 map to 3 -> all legal; width 2: identity only
    assert fortify.legal_action_mask(rates5, row_index=3, alive_count=4, device="cpu").tolist() == [True] * 5
    assert fortify.legal_action_mask(rates5, row_index=3, alive_count=2, device="cpu").tolist() == [True] + [False] * 4
    assert fortify.identity_action_index(rates5) == 0
    legal = torch.tensor([True] * 5)
    assert int(fortify.heuristic_eval_action(legal, rates5, policy="mild", device="cpu").item()) == 1
    assert int(fortify.heuristic_eval_action(legal, rates5, policy="l1", device="cpu").item()) == 2


# ------------------------------------------------------------------ PPO machinery

def test_gae_matches_hand_computation():
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    rewards = [1.0, 0.0, 2.0]
    values = [0.5, 0.5, 0.5]
    gamma, lam = 0.9, 0.8
    adv, rets = A2CAgentReinforce.gae(rewards, values, bootstrap=0.0, gamma=gamma, lam=lam)
    d2 = 2.0 + 0.0 - 0.5
    d1 = 0.0 + gamma * 0.5 - 0.5
    d0 = 1.0 + gamma * 0.5 - 0.5
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    assert adv == pytest.approx([a0, a1, a2])
    assert rets == pytest.approx([a0 + 0.5, a1 + 0.5, a2 + 0.5])


def _tiny_agent(monkeypatch, n_actions=3):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_POLICY_HEAD_ZERO_INIT", "1")
    monkeypatch.setenv("SPECTRA_ENCODER_DROPOUT", "0")
    monkeypatch.delenv("SPECTRA_STATE_SLACK", raising=False)
    monkeypatch.delenv("SPECTRA_BUDGET_IN_STATE", raising=False)
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    from src.Model.Actor import Actor
    from src.Model.Critic import Critic
    torch.manual_seed(0)
    agent = A2CAgentReinforce.__new__(A2CAgentReinforce)
    agent.conf = SimpleNamespace(
        device=torch.device("cpu"), discount_factor=0.99,
        compression_rates_dict={i: r for i, r in enumerate([1.0, 0.9, 0.8][:n_actions])},
        action_rankings_dict={i: None for i in range(n_actions)}, rollout_limit=None, passes=1)
    agent.actor_model = Actor("cpu", n_actions)
    agent.critic_model = Critic("cpu", n_actions)
    agent.actor_optimizer = torch.optim.Adam(agent.actor_model.parameters(), 3e-3)
    agent.critic_optimizer = torch.optim.Adam(agent.critic_model.parameters(), 3e-3)
    agent.agent_lr = 3e-3
    return agent


def _random_state(n_actions=3, L=6):
    fd = token_feature_dim(n_actions)
    feats = torch.randn(L, fd)
    return {
        "layer_features": feats,
        "layer_types": torch.randint(1, 3, (L,)),
        "coupling_ids": torch.arange(L),
        "block_ids": torch.arange(L),
        "target_index": int(torch.randint(0, L, (1,))),
        "action_costs": torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.05, 0.04], [0.8, 0.1, 0.08]][:n_actions]),
    }


def test_zero_init_head_starts_exactly_uniform(monkeypatch):
    agent = _tiny_agent(monkeypatch)
    with torch.no_grad():
        p = agent.actor_model(_random_state()).probs.flatten()
    assert torch.allclose(p, torch.full((3,), 1 / 3), atol=1e-6)


def test_ppo_update_moves_a_uniform_policy_toward_the_rewarded_action(monkeypatch):
    """Reward +1 for action 2, else 0. A few PPO batches must make action 2 dominant."""
    agent = _tiny_agent(monkeypatch)
    legal = torch.tensor([True, True, True])
    states = [_random_state() for _ in range(24)]

    def collect():
        batch = []
        for ep_i in range(4):
            steps = []
            for s in states[ep_i * 6:(ep_i + 1) * 6]:
                with torch.no_grad():
                    dist = fortify.apply_action_mask(agent.actor_model(s), legal)
                    a = dist.sample()
                    logp = float(dist.log_prob(a).item())
                    v = float(agent.critic_model(s).reshape(-1)[0].item())
                steps.append({"state": s, "legal": legal, "action": int(a.item()), "logp": logp,
                              "value": v, "reward": 1.0 if int(a.item()) == 2 else 0.0})
            batch.append({"steps": steps, "bootstrap": 0.0})
        return batch

    def p_action2():
        with torch.no_grad():
            return float(torch.stack([agent.actor_model(s).probs.flatten()[2] for s in states]).mean())

    before = p_action2()
    assert before == pytest.approx(1 / 3, abs=1e-5)
    for _ in range(12):
        stats = agent._ppo_update(collect(), ret_scale=1.0, ent_coef=0.0)
        assert stats["updated"] and stats["epochs_run"] >= 1
    after = p_action2()
    assert after > 0.6, (before, after)


def test_ppo_update_respects_the_legal_mask(monkeypatch):
    agent = _tiny_agent(monkeypatch)
    legal = torch.tensor([True, False, True])
    s = _random_state()
    good = {"state": s, "legal": legal, "action": 2, "logp": math.log(0.5), "value": 0.0, "reward": 1.0}
    bad = {"state": s, "legal": legal, "action": 0, "logp": math.log(0.5), "value": 0.0, "reward": 0.0}
    batch = [{"steps": [good, good, bad, good, bad, good], "bootstrap": 0.0}]
    for _ in range(3):
        agent._ppo_update(batch, ret_scale=1.0, ent_coef=0.0)
    with torch.no_grad():
        p = fortify.apply_action_mask(agent.actor_model(s), legal).probs.flatten()
    assert p[1].item() == 0.0 and p[2].item() > p[0].item()


# ------------------------------------------------------------------ checkpoint criterion & contract

def test_checkpoint_criterion_values(monkeypatch):
    monkeypatch.delenv("SPECTRA_CHECKPOINT", raising=False)
    monkeypatch.delenv("SPECTRA_REWARD_MODE", raising=False)
    monkeypatch.delenv("SPECTRA_ACTOR_SKIP_OVERBUDGET", raising=False)
    assert fortify.checkpoint_criterion() == "return"
    monkeypatch.setenv("SPECTRA_CHECKPOINT", "val_best")
    assert fortify.checkpoint_criterion() == "val_best"
    monkeypatch.setenv("SPECTRA_CHECKPOINT", "inbudget_compression")
    assert fortify.checkpoint_criterion() == "inbudget"


def test_episode_val_best_compression_tracks_deepest_in_band_point():
    from src.NetworkEnv import NetworkEnv
    env = NetworkEnv.__new__(NetworkEnv)
    env._reset_episode_reward_stats()
    assert env.episode_val_best_compression() == 0.0
    env.episode_best_inband_kept = 0.72
    assert env.episode_val_best_compression() == pytest.approx(0.28)


def test_standardizer_cache_prefers_the_checkpoint_dir(tmp_path):
    run = tmp_path / "job1"
    ckpt = run / "agent_checkpoints"
    ckpt.mkdir(parents=True)
    actor = ckpt / "latest_best_actor.pt"
    actor.write_bytes(b"x")
    # nothing on disk yet -> historical run-level path (also the write target)
    assert cache_path_from_actor(str(actor)) == str(run / "standardizer.pt")
    (run / "standardizer.pt").write_bytes(b"y")
    assert cache_path_from_actor(str(actor)) == str(run / "standardizer.pt")
    # a copy next to the checkpoint wins once it exists (snapshots carry it)
    (ckpt / "standardizer.pt").write_bytes(b"z")
    assert cache_path_from_actor(str(actor)) == str(ckpt / "standardizer.pt")
    snap = tmp_path / "snapshots" / "ep0007"
    snap.mkdir(parents=True)
    (snap / "latest_best_actor.pt").write_bytes(b"x")
    (snap / "standardizer.pt").write_bytes(b"z")
    assert cache_path_from_actor(str(snap / "latest_best_actor.pt")) == str(snap / "standardizer.pt")


def test_apply_policy_config_pins_contract_and_menu(tmp_path, monkeypatch):
    import a2c_agent_reinforce_runner as runner
    ckpt = tmp_path / "agent_checkpoints"
    ckpt.mkdir()
    (ckpt / "latest_best_actor.pt").write_bytes(b"x")
    cfg = {
        "compression_rates": [1.0, 0.9, 0.8, 0.9, 0.8],
        "action_rankings": [None, "l1", "l1", "fpgm", "fpgm"],
        "env": {"SPECTRA_STATE_ALIGN": "next", "SPECTRA_GROUP_ONCE_PER_PASS": "1",
                "SPECTRA_STATE_SLACK": "1", "SPECTRA_ENCODER_DROPOUT": "0"},
    }
    (ckpt / "policy_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    for k in cfg["env"]:
        # setenv (not delenv): monkeypatch only restores keys it recorded, and the runner
        # writes these keys itself — a delenv on an absent key would leak them to later tests.
        monkeypatch.setenv(k, "unset-by-test")
    monkeypatch.delenv("SPECTRA_POLICY_CONFIG", raising=False)
    args = SimpleNamespace(actor_checkpoint_path=str(ckpt / "latest_best_actor.pt"),
                           compression_rates=[1.0, 0.9, 0.8], action_rankings=None)
    assert runner.apply_policy_config(args) is not None
    assert os.environ["SPECTRA_STATE_ALIGN"] == "next"
    assert os.environ["SPECTRA_GROUP_ONCE_PER_PASS"] == "1"
    assert os.environ["SPECTRA_STATE_SLACK"] == "1"
    assert args.compression_rates == [1.0, 0.9, 0.8, 0.9, 0.8]
    assert args.action_rankings == ["none", "l1", "l1", "fpgm", "fpgm"]
    # opt-out
    monkeypatch.setenv("SPECTRA_POLICY_CONFIG", "0")
    args2 = SimpleNamespace(actor_checkpoint_path=str(ckpt / "latest_best_actor.pt"),
                            compression_rates=[1.0, 0.9, 0.8], action_rankings=None)
    assert runner.apply_policy_config(args2) is None
    assert args2.compression_rates == [1.0, 0.9, 0.8]
    # legacy actor without a config file: untouched
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "latest_best_actor.pt").write_bytes(b"x")
    monkeypatch.delenv("SPECTRA_POLICY_CONFIG", raising=False)
    args3 = SimpleNamespace(actor_checkpoint_path=str(legacy / "latest_best_actor.pt"),
                            compression_rates=[1.0, 0.9, 0.8], action_rankings=None)
    assert runner.apply_policy_config(args3) is None
