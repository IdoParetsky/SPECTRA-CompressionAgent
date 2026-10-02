"""
``SPECTRA_TIME_DECIDE`` (default off): a ``step.decide`` stage around each eval-walk decision, the
frozen actor's forward and pick or the heuristic's pick, and its readout in
``scripts/cost_readout.py`` (EFFICIENCY_AND_TRANSFER §11 item 3).

CPU only.

    python -m pytest tests/test_decide_timer.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from src import fortify  # noqa: E402
import src.logging_utils as logging_utils  # noqa: E402
from src.BERTInputModeler import token_feature_dim  # noqa: E402
import a2c_agent_reinforce_runner as runner  # noqa: E402

RATES = {0: 1.0, 1: 0.9, 2: 0.8}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("SPECTRA_TIME_DECIDE", "SPECTRA_EVAL_COUNTERFACTUAL", "SPECTRA_FACTORED_HEAD",
                "SPECTRA_STATE_SLACK", "SPECTRA_BUDGET_IN_STATE", "SPECTRA_STATE_GROUPCOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_ENCODER_DROPOUT", "0")
    monkeypatch.setenv("SPECTRA_EVAL_DETERMINISTIC", "1")
    yield


@pytest.fixture
def stages(monkeypatch):
    seen = []
    monkeypatch.setattr(logging_utils, "_record_stage",
                        lambda name, seconds, ok, error, enabled: seen.append((name, seconds, ok)))
    return seen


def _state(seed=0, L=6):
    g = torch.Generator().manual_seed(seed)
    return {
        "layer_features": torch.randn(L, token_feature_dim(3), generator=g),
        "layer_types": torch.randint(1, 3, (L,), generator=g),
        "coupling_ids": torch.arange(L), "block_ids": torch.arange(L),
        "target_index": 2,
        "action_costs": torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.05, 0.04], [0.8, 0.1, 0.08]]),
    }


def _agent(seed=0):
    from src.Model.Actor import Actor
    torch.manual_seed(seed)
    return SimpleNamespace(actor_model=Actor("cpu", 3).eval())


def _conf():
    return SimpleNamespace(device=torch.device("cpu"), compression_rates_dict=RATES)


def test_flag_defaults_off(monkeypatch):
    assert fortify.time_decide() is False
    monkeypatch.setenv("SPECTRA_TIME_DECIDE", "1")
    assert fortify.time_decide() is True


def test_off_records_nothing_and_on_times_the_actor_decision(monkeypatch, stages):
    legal = torch.tensor([True, True, True])
    agent, state = _agent(), _state()
    off = runner._actor_action(agent, state, legal, _conf(), fortify)
    assert stages == []
    monkeypatch.setenv("SPECTRA_TIME_DECIDE", "1")
    on = runner._actor_action(agent, state, legal, _conf(), fortify)
    assert torch.equal(off[0], on[0]) and off[1] == on[1]
    assert [(name, ok) for name, _, ok in stages] == [("step.decide", True)]
    assert stages[0][1] >= 0.0


def test_counterfactual_probe_stays_outside_the_timed_decision(monkeypatch, stages):
    monkeypatch.setenv("SPECTRA_TIME_DECIDE", "1")
    order = []
    monkeypatch.setattr(runner, "counterfactual_probe", lambda *a, **k: order.append("probe"))
    monkeypatch.setattr(logging_utils, "_record_stage",
                        lambda name, seconds, ok, error, enabled: order.append(name))
    runner._actor_action(_agent(), _state(), torch.tensor([True, True, True]), _conf(), fortify)
    assert order == ["step.decide", "probe"]


@pytest.mark.parametrize("policy", ["mild", "l1", "mildest", "random"])
def test_heuristic_decision_is_timed_the_same_way(monkeypatch, stages, policy):
    legal = torch.tensor([True, True, True])
    torch.manual_seed(0)
    off = runner._heuristic_action(legal, _conf(), fortify, policy)
    assert stages == []
    monkeypatch.setenv("SPECTRA_TIME_DECIDE", "1")
    torch.manual_seed(0)
    on = runner._heuristic_action(legal, _conf(), fortify, policy)
    assert torch.equal(off, on)
    assert [name for name, _, _ in stages] == ["step.decide"]


def _readout():
    spec = importlib.util.spec_from_file_location("cost_readout", REPO / "scripts" / "cost_readout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(root, decide):
    run = root / "job9"
    (run / "events").mkdir(parents=True)
    events = []
    t = 0.0
    for i in range(3):
        if decide:
            events.append({"event": "stage", "stage": "step.decide", "seconds": 0.01 * (i + 1), "t": t + 0.5})
        events.append({"event": "stage", "stage": "step.finetune", "seconds": 9.0, "t": t + 9.5})
        t += 10.0
        events.append({"event": "step", "network": "/x/r56.pth", "compression_rate": 0.9, "seconds": 9.5, "t": t})
    (run / "events" / "rank0.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return run


def test_cost_readout_reports_mean_decision_only_when_timed(tmp_path, capsys):
    readout = _readout()
    (net,) = readout.summarize(str(_run(tmp_path / "on", decide=True)))["networks"]
    assert net["decisions"] == 3 and net["decide_ms"] == pytest.approx(20.0)
    assert net["per_step_s"]["ft_s"] == pytest.approx(9.0)
    (old,) = readout.summarize(str(_run(tmp_path / "off", decide=False)))["networks"]
    assert old["decisions"] == 0 and old["decide_ms"] is None
    readout._print(readout.summarize(str(_run(tmp_path / "off2", decide=False))))
    assert "decide - ms" in capsys.readouterr().out
