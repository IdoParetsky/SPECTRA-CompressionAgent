"""
V6 (21 Sep): counterfactual state probe for frozen actors (``SPECTRA_EVAL_COUNTERFACTUAL``).

At each eval step the runner asks the actor what it would do on content-perturbed copies of
the state (layer features zeroed / shuffled / everything-but-position zeroed). The fraction of
steps whose argmax changes is the cheapest identification of "does the policy read the
state" — required before any encoder GPU (ledger §16 measured encoders under a uniform actor).

CPU only, default off.

    python -m pytest tests/test_v6_counterfactual.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from src import fortify  # noqa: E402
from src.BERTInputModeler import token_feature_dim  # noqa: E402
import a2c_agent_reinforce_runner as runner  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("SPECTRA_EVAL_COUNTERFACTUAL", "SPECTRA_FACTORED_HEAD", "SPECTRA_STATE_SLACK",
                "SPECTRA_BUDGET_IN_STATE", "SPECTRA_STATE_GROUPCOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_ENCODER_DROPOUT", "0")
    yield


def _state(n_actions=3, L=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    fd = token_feature_dim(n_actions)
    return {
        "layer_features": torch.randn(L, fd, generator=g),
        "layer_types": torch.randint(1, 3, (L,), generator=g),
        "coupling_ids": torch.arange(L), "block_ids": torch.arange(L),
        "target_index": 2,
        "action_costs": torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.05, 0.04], [0.8, 0.1, 0.08]]),
    }


def _agent(seed=0):
    from src.Model.Actor import Actor
    torch.manual_seed(seed)
    agent = SimpleNamespace()
    agent.actor_model = Actor("cpu", 3).eval()
    return agent


def test_counterfactual_states_perturb_content_only():
    state = _state()
    before = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}
    variants = runner._counterfactual_states(state)
    assert set(variants) == {"zero_layers", "shuffle_layers", "blind"}
    # original untouched
    for k, v in before.items():
        if torch.is_tensor(v):
            assert torch.equal(state[k], v)
    z = variants["zero_layers"]
    assert torch.count_nonzero(z["layer_features"]) == 0
    assert torch.equal(z["action_costs"], state["action_costs"]) and z["target_index"] == state["target_index"]
    s = variants["shuffle_layers"]
    assert s["layer_features"].shape == state["layer_features"].shape
    assert not torch.equal(s["layer_features"], state["layer_features"])
    # same rows, different order
    a = sorted(tuple(r) for r in s["layer_features"].tolist())
    b = sorted(tuple(r) for r in state["layer_features"].tolist())
    assert a == b
    bl = variants["blind"]
    assert torch.count_nonzero(bl["layer_features"]) == 0 and torch.count_nonzero(bl["action_costs"]) == 0
    # deterministic permutation for a given length
    again = runner._counterfactual_states(state)["shuffle_layers"]["layer_features"]
    assert torch.equal(again, s["layer_features"])


def test_probe_is_off_by_default_and_skips_non_dict_states(monkeypatch):
    conf = SimpleNamespace(device=torch.device("cpu"))
    legal = torch.tensor([True, True, True])
    agent = _agent()
    assert runner.counterfactual_probe(agent, _state(), legal, 1, conf, fortify) is None
    monkeypatch.setenv("SPECTRA_EVAL_COUNTERFACTUAL", "1")
    assert runner.counterfactual_probe(agent, torch.zeros(4), legal, 1, conf, fortify) is None


def test_probe_reports_legal_argmaxes_and_flags(monkeypatch):
    monkeypatch.setenv("SPECTRA_EVAL_COUNTERFACTUAL", "1")
    import src.run_recorder as recorder
    seen = []
    monkeypatch.setattr(recorder, "record", lambda kind, **kw: seen.append((kind, kw)))
    conf = SimpleNamespace(device=torch.device("cpu"))
    legal = torch.tensor([True, False, True])          # 0.9 illegal on this row
    agent = _agent()
    state = _state()
    with torch.no_grad():
        dist = agent.actor_model(state)
    real, _, _ = fortify.pick_action(dist, legal, deterministic=True, device=conf.device)
    out = runner.counterfactual_probe(agent, state, legal, real, conf, fortify)
    assert out["real"] == int(real)
    for key in ("zero_layers", "shuffle_layers", "blind"):
        assert out[key] in (0, 2)                          # never the masked action
    assert out["content_used"] in (0, 1) and out["state_used"] in (0, 1)
    assert 0.0 < out["pmax"] <= 1.0
    assert seen and seen[-1][0] == "counterfactual" and seen[-1][1]["real"] == int(real)


def test_content_blind_actor_shows_state_unused(monkeypatch):
    """An actor whose input projection is zeroed cannot read content: every variant agrees."""
    monkeypatch.setenv("SPECTRA_EVAL_COUNTERFACTUAL", "1")
    conf = SimpleNamespace(device=torch.device("cpu"))
    legal = torch.tensor([True, True, True])
    agent = _agent(seed=3)
    front = agent.actor_model.state_encoder
    with torch.no_grad():
        for module in (front.input_proj[0], front.action_proj[0]):
            module.weight.zero_()
            module.bias.zero_()
    state = _state(seed=5)
    with torch.no_grad():
        dist = agent.actor_model(state)
    real, _, _ = fortify.pick_action(dist, legal, deterministic=True, device=conf.device)
    out = runner.counterfactual_probe(agent, state, legal, real, conf, fortify)
    assert out["content_used"] == 0 and out["state_used"] == 0
    assert out["zero_layers"] == out["shuffle_layers"] == out["blind"] == out["real"]


def test_actor_action_calls_probe_when_on(monkeypatch):
    monkeypatch.setenv("SPECTRA_EVAL_COUNTERFACTUAL", "1")
    monkeypatch.setenv("SPECTRA_EVAL_DETERMINISTIC", "1")
    calls = []
    monkeypatch.setattr(runner, "counterfactual_probe", lambda *a, **k: calls.append(a) or None)
    conf = SimpleNamespace(device=torch.device("cpu"))
    legal = torch.tensor([True, True, True])
    action, rank = runner._actor_action(_agent(), _state(), legal, conf, fortify)
    assert action.shape == (1,) and rank is None and len(calls) == 1
