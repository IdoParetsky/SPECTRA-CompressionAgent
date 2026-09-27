"""
V4-1: factored (rate x ranking) policy head + Taylor criterion. CPU only.

    python -m pytest tests/test_v4_factored.py -v
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf, ResidualNet  # noqa: E402

_init_static_conf()

from src import fortify  # noqa: E402
from src.Configuration.StaticConf import StaticConf  # noqa: E402
from src.Model.Agent import FactoredCategorical  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.utils as utils  # noqa: E402
from src.BERTInputModeler import token_feature_dim  # noqa: E402

MENU = ["l1", "fpgm", "bn_scale", "svd", "taylor"]


# ------------------------------------------------------------------ distribution semantics

def test_factored_log_prob_ignores_ranking_on_identity():
    rate = Categorical(probs=torch.tensor([[0.5, 0.3, 0.2]]))
    rank = Categorical(probs=torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    d = FactoredCategorical(rate, rank, identity_index=0)
    assert d.log_prob(0, 3).item() == pytest.approx(math.log(0.5))          # identity: rank inactive
    assert d.log_prob(2, 3).item() == pytest.approx(math.log(0.2) + math.log(0.4))
    assert d.entropy().item() == pytest.approx(rate.entropy().item() + rank.entropy().item())
    r, k = d.argmax()
    assert (int(r), int(k)) == (0, 3)
    assert torch.allclose(d.probs, rate.probs)


def test_mask_policy_masks_only_the_rate_head():
    rate = Categorical(probs=torch.tensor([[0.2, 0.5, 0.3]]))
    rank = Categorical(probs=torch.tensor([[0.25, 0.75]]))
    d = FactoredCategorical(rate, rank, identity_index=0)
    legal = torch.tensor([True, False, True])
    m = fortify.mask_policy(d, legal)
    assert fortify.is_factored_dist(m)
    assert m.rate.probs.flatten()[1].item() == 0.0
    assert torch.allclose(m.rank.probs, rank.probs)
    # plain Categorical still works through the same helper
    plain = fortify.mask_policy(rate, legal)
    assert not fortify.is_factored_dist(plain) and plain.probs.flatten()[1].item() == 0.0


def test_pick_action_plain_and_factored(monkeypatch):
    rate = Categorical(probs=torch.tensor([[0.1, 0.6, 0.3]]))
    rank = Categorical(probs=torch.tensor([[0.2, 0.8]]))
    legal = torch.tensor([True, True, True])
    r, k, lp = fortify.pick_action(rate, legal, deterministic=True, device="cpu")
    assert (r, k) == (1, None) and lp == pytest.approx(math.log(0.6))
    d = FactoredCategorical(rate, rank, identity_index=0)
    r, k, lp = fortify.pick_action(d, legal, deterministic=True, device="cpu")
    assert (r, k) == (1, 1) and lp == pytest.approx(math.log(0.6) + math.log(0.8))
    # identity argmax -> ranking None
    d0 = FactoredCategorical(Categorical(probs=torch.tensor([[0.8, 0.1, 0.1]])), rank, 0)
    r, k, lp = fortify.pick_action(d0, legal, deterministic=True, device="cpu")
    assert (r, k) == (0, None) and lp == pytest.approx(math.log(0.8))


# ------------------------------------------------------------------ actor construction

def _with_menu(monkeypatch, menu):
    conf = StaticConf.get_instance().conf_values
    monkeypatch.setattr(conf, "ranking_menu", list(menu), raising=False)
    monkeypatch.setattr(conf, "compression_rates_dict", {0: 1.0, 1: 0.9, 2: 0.8}, raising=False)
    monkeypatch.setattr(conf, "num_actions", 3, raising=False)


def test_actor_is_single_head_unless_flag_and_menu(monkeypatch):
    from src.Model.Actor import Actor
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.delenv("SPECTRA_FACTORED_HEAD", raising=False)
    _with_menu(monkeypatch, MENU)
    a = Actor("cpu", 3)
    assert not a.is_factored and a.ranker is None
    monkeypatch.setenv("SPECTRA_FACTORED_HEAD", "1")
    _with_menu(monkeypatch, [])
    a2 = Actor("cpu", 3)
    assert not a2.is_factored          # flag without a menu: still single head
    _with_menu(monkeypatch, MENU)
    a3 = Actor("cpu", 3)
    assert a3.is_factored and a3.ranker[4].out_features == 5


def _state(n_actions=3, L=6):
    fd = token_feature_dim(n_actions)
    return {
        "layer_features": torch.randn(L, fd), "layer_types": torch.randint(1, 3, (L,)),
        "coupling_ids": torch.arange(L), "block_ids": torch.arange(L),
        "target_index": int(torch.randint(0, L, (1,))),
        "action_costs": torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.05, 0.04], [0.8, 0.1, 0.08]]),
    }


def test_factored_actor_forward_and_zero_init(monkeypatch):
    from src.Model.Actor import Actor
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_FACTORED_HEAD", "1")
    monkeypatch.setenv("SPECTRA_POLICY_HEAD_ZERO_INIT", "1")
    monkeypatch.setenv("SPECTRA_ENCODER_DROPOUT", "0")
    for k in ("SPECTRA_STATE_SLACK", "SPECTRA_BUDGET_IN_STATE", "SPECTRA_STATE_GROUPCOST"):
        monkeypatch.delenv(k, raising=False)
    _with_menu(monkeypatch, MENU)
    torch.manual_seed(0)
    actor = Actor("cpu", 3)
    with torch.no_grad():
        d = actor(_state())
    assert fortify.is_factored_dist(d)
    assert torch.allclose(d.rate.probs.flatten(), torch.full((3,), 1 / 3), atol=1e-6)
    assert torch.allclose(d.rank.probs.flatten(), torch.full((5,), 0.2), atol=1e-6)


def _tiny_factored_agent(monkeypatch):
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    from src.Model.Actor import Actor
    from src.Model.Critic import Critic
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_FACTORED_HEAD", "1")
    monkeypatch.setenv("SPECTRA_POLICY_HEAD_ZERO_INIT", "1")
    monkeypatch.setenv("SPECTRA_ENCODER_DROPOUT", "0")
    for k in ("SPECTRA_STATE_SLACK", "SPECTRA_BUDGET_IN_STATE", "SPECTRA_STATE_GROUPCOST"):
        monkeypatch.delenv(k, raising=False)
    _with_menu(monkeypatch, MENU)
    torch.manual_seed(0)
    agent = A2CAgentReinforce.__new__(A2CAgentReinforce)
    agent.conf = SimpleNamespace(
        device=torch.device("cpu"), discount_factor=0.99,
        compression_rates_dict={0: 1.0, 1: 0.9, 2: 0.8},
        action_rankings_dict={0: None, 1: None, 2: None}, ranking_menu=list(MENU),
        rollout_limit=None, passes=1)
    agent.actor_model = Actor("cpu", 3)
    agent.critic_model = Critic("cpu", 3)
    agent.actor_optimizer = torch.optim.Adam(agent.actor_model.parameters(), 3e-3)
    agent.critic_optimizer = torch.optim.Adam(agent.critic_model.parameters(), 3e-3)
    agent.agent_lr = 3e-3
    return agent


def test_action_ranking_maps_factored_index_to_menu(monkeypatch):
    agent = _tiny_factored_agent(monkeypatch)
    assert agent._action_ranking(2, 4) == "taylor"
    assert agent._action_ranking(2, 1) == "fpgm"
    assert agent._action_ranking(0, None) is None
    monkeypatch.delenv("SPECTRA_FACTORED_HEAD", raising=False)
    agent.conf.action_rankings_dict = {0: None, 1: "l1", 2: "fpgm"}
    assert agent._action_ranking(2, 4) == "fpgm"       # single-head table wins when flag is off


def test_factored_ppo_learns_rate_and_ranking_jointly(monkeypatch):
    """Reward 1 only for (rate 0.8, ranking taylor). Both heads must move toward it."""
    agent = _tiny_factored_agent(monkeypatch)
    legal = torch.tensor([True, True, True])
    states = [_state() for _ in range(24)]

    def collect():
        batch = []
        for e in range(4):
            steps = []
            for s in states[e * 6:(e + 1) * 6]:
                with torch.no_grad():
                    d = fortify.mask_policy(agent.actor_model(s), legal)
                    r, k, lp = fortify.pick_action(d, legal, deterministic=False, device="cpu")
                    v = float(agent.critic_model(s).reshape(-1)[0].item())
                # rate head: +1 for 0.8; ranking head: +1 for taylor on any non-identity step
                reward = (1.0 if r == 2 else 0.0) + (1.0 if (r != 0 and k == 4) else 0.0)
                steps.append({"state": s, "legal": legal, "action": r, "rank": k, "logp": lp,
                              "value": v, "reward": reward})
            batch.append({"steps": steps, "bootstrap": 0.0})
        return batch

    def probs():
        with torch.no_grad():
            ds = [agent.actor_model(s) for s in states]
        p_rate = torch.stack([d.rate.probs.flatten()[2] for d in ds]).mean().item()
        p_rank = torch.stack([d.rank.probs.flatten()[4] for d in ds]).mean().item()
        return p_rate, p_rank

    r0, k0 = probs()
    assert r0 == pytest.approx(1 / 3, abs=1e-4) and k0 == pytest.approx(0.2, abs=1e-4)
    for _ in range(20):
        stats = agent._ppo_update(collect(), ret_scale=1.0, ent_coef=0.0)
        assert stats["updated"]
    r1, k1 = probs()
    assert r1 > 0.5 and k1 > 0.4, (r0, k0, r1, k1)


# ------------------------------------------------------------------ Taylor criterion

def test_taylor_scores_bind_and_rank(monkeypatch):
    monkeypatch.delenv("SPECTRA_FILTER_IMPORTANCE", raising=False)
    torch.manual_seed(0)
    model = ResidualNet().eval()
    x = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 10, (8,))
    loader = [(x, y)]
    n = pruning.bind_taylor_scores(model, loader, torch.device("cpu"), n_batches=1)
    assert n == 4  # stem, conv1, conv2, fc
    conv = model.block.conv1
    t = pruning.filter_importance(conv, mode="taylor")
    l1 = pruning.filter_importance(conv)
    assert t.shape == l1.shape == (16,)
    assert torch.all(t >= 0) and t.sum() > 0
    assert not torch.allclose(t / t.max(), l1 / l1.max())
    keep = pruning.select_surviving_filters(conv, 0.5, mode="taylor")
    assert keep.numel() == 8
    # gradients were cleared and the model is back in eval mode without grad requirements leaking
    assert all(p.grad is None for p in model.parameters())
    assert not model.training
    # without bound scores the criterion falls back to L1 (no crash)
    pruning._TAYLOR_SCORES.clear()
    assert torch.allclose(pruning.filter_importance(conv, mode="taylor"), l1)
    assert pruning.normalize_importance_mode("Taylor") == "taylor"


def test_ranking_menu_parsing():
    assert utils.parse_ranking_menu(None) == []
    assert utils.parse_ranking_menu(["l1", "FPGM", "bn", "svd", "taylor", "l1"]) == \
        ["l1", "fpgm", "bn_scale", "svd", "taylor"]


def test_policy_config_pins_factored_menu(tmp_path, monkeypatch):
    import a2c_agent_reinforce_runner as runner
    ckpt = tmp_path / "agent_checkpoints"
    ckpt.mkdir()
    (ckpt / "latest_best_actor.pt").write_bytes(b"x")
    (ckpt / "policy_config.json").write_text(json.dumps({
        "compression_rates": [1.0, 0.9, 0.8], "action_rankings": [None, None, None],
        "factored_head": True, "ranking_menu": MENU,
        "env": {"SPECTRA_FACTORED_HEAD": "1"}, "passes": 2}), encoding="utf-8")
    monkeypatch.delenv("SPECTRA_POLICY_CONFIG", raising=False)
    monkeypatch.delenv("SPECTRA_EVAL_PASSES", raising=False)
    monkeypatch.setenv("SPECTRA_FACTORED_HEAD", "0")
    args = SimpleNamespace(actor_checkpoint_path=str(ckpt / "latest_best_actor.pt"),
                           compression_rates=[1.0, 0.9, 0.8], action_rankings=None,
                           ranking_menu=None, passes=1)
    runner.apply_policy_config(args)
    assert args.ranking_menu == MENU and args.passes == 2
    import os
    assert os.environ["SPECTRA_FACTORED_HEAD"] == "1"


def test_ranking_for_helper_in_runner(monkeypatch):
    import a2c_agent_reinforce_runner as runner
    conf = SimpleNamespace(ranking_menu=MENU, action_rankings_dict={0: None, 1: "l1", 2: "fpgm"})
    monkeypatch.setenv("SPECTRA_FACTORED_HEAD", "1")
    assert runner._ranking_for(conf, fortify, 2, 4) == "taylor"
    assert runner._ranking_for(conf, fortify, 0, None) is None
    monkeypatch.delenv("SPECTRA_FACTORED_HEAD", raising=False)
    assert runner._ranking_for(conf, fortify, 2, 4) == "fpgm"
