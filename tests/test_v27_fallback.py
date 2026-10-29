"""
v27 ``SPECTRA_ALLOC_KIND=fallback`` (default off): plan with the frozen agent AND with a hand rule (the prior), score both
plans on the val half with a cheap proxy (``plan_proxies.PlanScorer``), and walk the agent's plan unless the prior's
scores higher on the first proxy by more than a margin. The contract is exactness: the picked arm's ``(widths, info)`` is
returned unchanged (``info["kind"]`` too, one key ``fallback`` added) and the global RNG is left where the picked arm
alone leaves it, so the walk and the final fine-tune run as under that arm alone. These tests fake the plans, the
scorer and the env (a tiny real ``torch.nn`` model stands for the origin), and make the fakes draw different amounts of
Python, NumPy and torch RNG so a wrong restore shows.

CPU only, no GPU, no datasets.  python -m pytest tests/test_v27_fallback.py -v
"""

import os
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from src import alloc_walk, plan_agent, plan_proxies  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_EVAL_PROXY",
                 "SPECTRA_MIN_WIDTH", "SPECTRA_GROUP_ONCE", "SPECTRA_SEED", "SPECTRA_FT_")
OLD_KINDS = ("uniform", "sens", "inner", "widths", "sample", "agent", "agent_sample", "sens_cost", "randbest")
NET = "toy.pt"
PRIOR_WIDTHS = {1: 3, 2: 4}
AGENT_WIDTHS = {1: 5, 2: 6}
DECISION_LINE = re.compile(r"^\[alloc\] fallback (\S+): pick (agent|prior) on (\S+) \(margin (\S+); (\d+\.\d) min\)$")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


def _burn(n):
    """Draw ``n`` numbers from each of the Python, NumPy and torch CPU generators."""
    return [random.random() for _ in range(n)], np.random.rand(n).tolist(), torch.rand(n).tolist()


def _seed_all(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draw():
    """Three numbers from each global generator: equal draws mean equal states."""
    return [random.random() for _ in range(3)], np.random.rand(3).tolist(), torch.rand(3).tolist()


def _origin():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))


def _env(model=None):
    return types.SimpleNamespace(
        selected_net_path=f"/nets/{NET}", current_model=_origin() if model is None else model, target_keep=0.6,
        conf=types.SimpleNamespace(device="cpu"), train_loader=object(), val_loader=object(),
        _input_shape=lambda: (4,), _dependency_groups=lambda mwr: [])


class _Fakes:
    """The fake plans and scorer ``fallback_plan`` calls: the calibration batches and ``plan_targets`` (the prior) draw
    3 numbers from each generator in all, ``plan_for_env`` (the agent) draws 7, ``PlanScorer.batches`` 1 and each
    ``score`` 2. ``scores`` maps a proxy to ``{"agent": s, "prior": s}``."""

    def __init__(self, monkeypatch, scores, mutate=None):
        self.scores, self.mutate, self.calls = scores, mutate, []
        self.prior_info = {"kind": "sens", "alpha": 0.5, "target": 0.58, "kept": 0.59, "keeps": {1: 0.5, 2: 0.6},
                           "origin_widths": {1: 6, 2: 8}, "sens": {1: 1.0, 2: 2.0}, "held": 0}
        self.agent_info = {"kind": "agent", "alpha": 0.0, "target": 0.58, "kept": 0.57, "keeps": {1: 0.8, 2: 0.7},
                           "origin_widths": {1: 6, 2: 8}, "sens": {1: 0.1, 2: 0.2}, "held": 0, "policy": "p/agent.pt"}
        fakes = self

        class Scorer:
            def __init__(self, model, plan, val_loader, device, train_loader=None, input_shape=None,
                         create_handler=None, name=""):
                fakes.calls.append(("scorer", name, val_loader))

            def batches(self, proxy, seed):
                fakes.calls.append(("batches", proxy, seed))
                _burn(1)
                return ("batches", proxy, seed)

            def score(self, widths, proxy, batches):
                arm = "agent" if widths is AGENT_WIDTHS else "prior"
                fakes.calls.append(("score", arm, proxy, batches))
                _burn(2)
                if fakes.mutate == "scoring":
                    with torch.no_grad():
                        fakes.model[0].weight.add_(1.0)
                return fakes.scores[proxy][arm]

        monkeypatch.setattr(group_sensitivity, "calibration_batches",
                            lambda loader, n, device: self._calibrate())
        monkeypatch.setattr(alloc_walk, "plan_targets", self._plan_targets)
        monkeypatch.setattr(plan_agent, "plan_for_env", self._plan_for_env)
        monkeypatch.setattr(alloc_walk, "ModelWithRows", lambda model: types.SimpleNamespace(model=model))
        monkeypatch.setattr(group_sensitivity, "group_plan", lambda mwr, groups: [])
        monkeypatch.setattr(plan_proxies, "PlanScorer", Scorer)

    def _calibrate(self):
        self.calls.append(("calibration",))
        _burn(1)
        return []

    def _plan_targets(self, model, batches, shape, kind, target, a=0.5, keep_floor=0.1, **kw):
        self.model = model
        self.calls.append(("plan_targets", kind, target, a, keep_floor, dict(kw)))
        _burn(2)
        if self.mutate == "prior":
            with torch.no_grad():
                model[0].weight.add_(1.0)
        return PRIOR_WIDTHS, self.prior_info

    def _plan_for_env(self, env, target, path, k_min, **kw):
        self.model = env.current_model
        self.calls.append(("plan_for_env", target, path, k_min, dict(kw)))
        _burn(7)
        if self.mutate == "agent":
            with torch.no_grad():
                env.current_model[0].weight.add_(1.0)
        return AGENT_WIDTHS, self.agent_info


def _configure(monkeypatch, proxy="cut", margin=None, prior="sens"):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "fallback")
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", prior)
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PROXY", proxy)
    monkeypatch.setenv("SPECTRA_PLAN_AGENT", "/x/plan_agent/policy_latest.pt")
    if margin is not None:
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_MARGIN", str(margin))


def _run_plan(monkeypatch, fakes, env=None, flops=False):
    """``fallback_plan`` from a fixed RNG start: ``(widths, info, lines, records, draw_after)``."""
    lines, records = [], []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
    env = _env() if env is None else env                              # built first: _origin() seeds torch
    _seed_all()
    kw = {"budget": "flops", "kappa": 0.6} if flops else {}
    widths, info = alloc_walk.fallback_plan(env, 0.58, flops, kw, {"budget": "flops"} if flops else {})
    return widths, info, lines, records, _draw()


def _reference_draw(pick):
    """What the global generators hold after ONLY the picked arm's fake calls, from the same start."""
    _seed_all()
    if pick == "prior":
        _burn(1)
        _burn(2)
    else:
        _burn(7)
    return _draw()


# ------------------------------------------------------------------ the pure helpers


def test_fallback_pick_prior_wins_only_strictly_above_agent_plus_margin():
    assert alloc_walk.fallback_pick(0.80, 0.81, 0.0) == "prior"
    assert alloc_walk.fallback_pick(0.80, 0.80, 0.0) == "agent"       # a tie goes to the agent
    assert alloc_walk.fallback_pick(0.81, 0.80, 0.0) == "agent"
    assert alloc_walk.fallback_pick(0.40, 0.99, 0.5) == "prior"       # 0.99 > 0.40 + 0.5
    assert alloc_walk.fallback_pick(0.50, 1.00, 0.5) == "agent"       # exactly agent + margin: not strictly above
    assert alloc_walk.fallback_pick(0.50, 0.90, 0.5) == "agent"
    assert alloc_walk.fallback_pick(0.25, 0.75, 0.25) == "prior"
    assert alloc_walk.fallback_pick(0.25, 0.50, 0.25) == "agent"      # exactly representable: a tie at the margin


def test_rng_snapshot_and_restore_round_trip():
    _seed_all(7)
    snap = alloc_walk.rng_snapshot()
    first = _draw()
    _burn(5)
    alloc_walk.rng_restore(snap)
    assert _draw() == first
    alloc_walk.rng_restore(snap)
    _burn(2)
    assert _draw() != first                                            # the draws do move the state
    assert {"py", "np", "torch"} <= set(snap)
    assert ("cuda" in snap) == torch.cuda.is_available()


def test_state_dict_equal_sees_a_changed_weight_buffer_or_key():
    model = _origin()
    snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert alloc_walk._state_dict_equal(model, snap)
    with torch.no_grad():
        model[0].weight.add_(1e-6)
    assert not alloc_walk._state_dict_equal(model, snap)
    model = _origin()
    model[1].running_mean.add_(1.0)
    assert not alloc_walk._state_dict_equal(model, snap)               # buffers count
    model = _origin()
    assert not alloc_walk._state_dict_equal(model, {k: v for k, v in snap.items() if k != "0.bias"})
    assert not alloc_walk._state_dict_equal(model, {**snap, "extra": torch.zeros(1)})


# ------------------------------------------------------------------ the env vars and the kinds


def test_env_helpers_refuse_what_they_cannot_read(monkeypatch):
    with pytest.raises(ValueError, match="SPECTRA_ALLOC_FALLBACK_PRIOR"):
        alloc_walk.fallback_prior()                                    # no default
    for bad in ("uniform", "agent", "randbest", "inner"):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", bad)
        with pytest.raises(ValueError, match="SPECTRA_ALLOC_FALLBACK_PRIOR"):
            alloc_walk.fallback_prior()
    for raw, want in (("sens", "sens"), (" SENS_COST ", "sens_cost")):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", raw)
        assert alloc_walk.fallback_prior() == want
    with pytest.raises(ValueError, match="SPECTRA_ALLOC_FALLBACK_PROXY"):
        alloc_walk.fallback_proxies()                                  # missing
    for bad in ("", " , ,", "ft12_4", "fs0", "bn", "nonsense", "cut,nonsense"):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PROXY", bad)
        with pytest.raises(ValueError):
            alloc_walk.fallback_proxies()
    for bad in ("cut,cut", "fs32, FS32", "ft1,cut,ft1"):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PROXY", bad)
        with pytest.raises(ValueError, match="duplicate"):
            alloc_walk.fallback_proxies()
    for raw, want in (("ft1, fs32", ["ft1", "fs32"]), (" BN8 ", ["bn8"]), ("fs32,,ft1,", ["fs32", "ft1"]),
                      ("cut", ["cut"])):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PROXY", raw)
        assert alloc_walk.fallback_proxies() == want                   # the order is kept: the first decides
    assert alloc_walk.fallback_margin() == 0.0
    for raw, want in (("0", 0.0), ("0.02", 0.02), ("0.5", 0.5)):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_MARGIN", raw)
        assert alloc_walk.fallback_margin() == want
    for bad in ("-0.01", "-1", "nan"):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_MARGIN", bad)
        with pytest.raises(ValueError, match="SPECTRA_ALLOC_FALLBACK_MARGIN"):
            alloc_walk.fallback_margin()


def test_kinds_end_with_fallback_the_old_ones_stand_and_the_default_is_sens(monkeypatch):
    # v30 appends its kinds after fallback; the old kinds and fallback keep their places
    assert alloc_walk.KINDS[:len(OLD_KINDS) + 1] == OLD_KINDS + ("fallback",)
    assert alloc_walk.KINDS[len(OLD_KINDS)] == "fallback"
    assert alloc_walk.kind() == "sens"
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "Fallback")
    assert alloc_walk.kind() == "fallback"


# ------------------------------------------------------------------ fallback_plan with fakes


@pytest.mark.parametrize("pick", ["agent", "prior"])
def test_the_picked_plan_comes_back_unchanged_and_the_rng_stands_as_under_that_arm_alone(monkeypatch, pick):
    _configure(monkeypatch, proxy="cut")
    scores = {"cut": {"agent": 0.80, "prior": 0.90 if pick == "prior" else 0.70}}
    fakes = _Fakes(monkeypatch, scores)
    before = {k: dict(v) if isinstance(v, dict) else v for k, v in fakes.prior_info.items()}
    env = _env()
    widths, info, lines, records, after = _run_plan(monkeypatch, fakes, env=env)
    picked = fakes.prior_info if pick == "prior" else fakes.agent_info
    other = fakes.agent_info if pick == "prior" else fakes.prior_info
    # (a) the picked fake's widths and info, the kind unchanged, a shallow copy that gains one key
    assert widths is (PRIOR_WIDTHS if pick == "prior" else AGENT_WIDTHS)
    assert info is not picked and "fallback" not in picked and "fallback" not in other
    assert {k: v for k, v in info.items() if k != "fallback"} == picked
    assert info["kind"] == ("sens" if pick == "prior" else "agent")
    assert info["keeps"] is picked["keeps"] and fakes.prior_info == before
    # (b) the RNG stands where the picked arm alone leaves it
    assert after == _reference_draw(pick)
    # (c) the summary carries the pick, the scores per proxy, the prior and the margin
    summary = info["fallback"]
    assert summary["pick"] == pick and summary["prior"] == "sens" and summary["proxies"] == ["cut"]
    assert summary["margin"] == 0.0 and summary["scores"] == scores and summary["minutes"] >= 0.0
    assert set(summary) == {"pick", "prior", "proxies", "margin", "scores", "minutes"}
    # the calls: the rule arm's, then the agent arm's, exactly as _state makes them
    assert fakes.calls[0] == ("calibration",)
    assert fakes.calls[1] == ("plan_targets", "sens", 0.58, alloc_walk.alpha(), alloc_walk.min_keep(), {})
    assert fakes.calls[2] == ("plan_for_env", 0.58, "/x/plan_agent/policy_latest.pt", alloc_walk.min_keep(), {})
    assert fakes.calls[3] == ("scorer", NET, env.val_loader)             # the val half only
    assert [c[:3] for c in fakes.calls[4:]] == [("batches", "cut", 0), ("score", "agent", "cut"),
                                               ("score", "prior", "cut")]   # the agent is scored first
    assert fakes.calls[5][3] == fakes.calls[6][3] == ("batches", "cut", 0)  # both on the same batches
    # the lines and the record
    p = scores["cut"]["prior"]
    assert lines[0] == (f"[alloc] fallback {NET} cut: agent 0.8000 prior sens {p:.4f} "
                        f"(prior − agent {p - 0.80:+.4f})")
    decision = DECISION_LINE.match(lines[1])
    assert decision and decision.groups()[:4] == (NET, pick, "cut", "0") and len(lines) == 2
    event, rec = records[0]
    assert event == "alloc_fallback" and len(records) == 1
    assert (rec["prior"], rec["proxies"], rec["margin"], rec["pick"], rec["budget"]) == ("sens", ["cut"], 0.0, pick, "params")
    assert rec["target"] == 0.58 and rec["proxy_seed"] == 0 and rec["scores"] == scores and "fs_lr" not in rec
    assert rec["agent_widths"] == {"1": 5, "2": 6} and rec["prior_widths"] == {"1": 3, "2": 4}
    assert rec["network"] == f"/nets/{NET}" and rec["minutes"] >= 0.0


def test_the_margin_and_a_tie_decide_and_the_first_proxy_alone_decides(monkeypatch):
    _configure(monkeypatch, proxy="cut", margin=0.05)
    fakes = _Fakes(monkeypatch, {"cut": {"agent": 0.80, "prior": 0.84}})
    assert _run_plan(monkeypatch, fakes)[1]["fallback"]["pick"] == "agent"   # +0.04 is under the 0.05 margin
    fakes = _Fakes(monkeypatch, {"cut": {"agent": 0.80, "prior": 0.86}})
    assert _run_plan(monkeypatch, fakes)[1]["fallback"]["pick"] == "prior"
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_MARGIN", "0")
    fakes = _Fakes(monkeypatch, {"cut": {"agent": 0.80, "prior": 0.80}})
    assert _run_plan(monkeypatch, fakes)[1]["fallback"]["pick"] == "agent"   # a tie
    # (d) two proxies: only the first decides; the second prefers the other arm; both are scored and reported
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PROXY", "cut,fs2")
    for pick in ("agent", "prior"):
        scores = {"cut": {"agent": 0.85, "prior": 0.86 if pick == "prior" else 0.84},
                  "fs2": {"agent": 0.9, "prior": 0.8} if pick == "prior" else {"agent": 0.8, "prior": 0.9}}
        fakes = _Fakes(monkeypatch, scores)
        widths, info, lines, records, after = _run_plan(monkeypatch, fakes)
        assert info["fallback"]["pick"] == pick and info["fallback"]["proxies"] == ["cut", "fs2"]
        assert info["fallback"]["scores"] == scores and records[0][1]["fs_lr"] == plan_agent.fs_lr()
        assert [line.split(": ")[0] for line in lines[:2]] == [f"[alloc] fallback {NET} cut", f"[alloc] fallback {NET} fs2"]
        assert DECISION_LINE.match(lines[2]).groups()[1:3] == (pick, "cut")
        assert [c[:3] for c in fakes.calls[4:]] == [("batches", "cut", 0), ("score", "agent", "cut"), ("score", "prior", "cut"),
                                                   ("batches", "fs2", 0), ("score", "agent", "fs2"),
                                                   ("score", "prior", "fs2")]
        assert after == _reference_draw(pick)


def test_a_flops_budget_passes_the_state_kwargs_through_and_a_residual_is_not_refused(monkeypatch):
    _configure(monkeypatch, proxy="cut", prior="sens_cost")
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "sens")
    fakes = _Fakes(monkeypatch, {"cut": {"agent": 0.8, "prior": 0.7}})
    widths, info, lines, records, _after = _run_plan(monkeypatch, fakes, flops=True)
    assert widths is AGENT_WIDTHS and records[0][1]["budget"] == "flops" and records[0][1]["prior"] == "sens_cost"
    assert fakes.calls[1][1] == "sens_cost" and fakes.calls[1][5] == {"budget": "flops"}
    assert fakes.calls[2][4] == {"budget": "flops", "kappa": 0.6}
    assert "prior sens_cost 0.7000" in lines[0]


@pytest.mark.parametrize("step, stage", [("prior", "sens plan"), ("agent", "agent plan"), ("scoring", "proxy scoring")])
def test_a_plan_that_changes_the_origins_weights_raises(monkeypatch, step, stage):
    _configure(monkeypatch, proxy="cut")
    fakes = _Fakes(monkeypatch, {"cut": {"agent": 0.8, "prior": 0.7}}, mutate=step)
    env = _env()
    with pytest.raises(RuntimeError, match=f"{NET}: the {stage} changed the origin's weights"):
        _run_plan(monkeypatch, fakes, env=env)


def test_the_missing_flags_raise_before_any_plan(monkeypatch):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "fallback")
    fakes = _Fakes(monkeypatch, {"cut": {"agent": 0.8, "prior": 0.7}})
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    with pytest.raises(ValueError, match="SPECTRA_ALLOC_FALLBACK_PRIOR"):
        alloc_walk.fallback_plan(_env(), 0.58, False, {}, {})
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", "sens")
    with pytest.raises(ValueError, match="SPECTRA_ALLOC_FALLBACK_PROXY"):
        alloc_walk.fallback_plan(_env(), 0.58, False, {}, {})
    assert fakes.calls == [] and lines == []


def test_state_routes_the_fallback_kind_and_leaves_the_other_kinds_alone(monkeypatch):
    calls = []
    info = {"kind": "agent", "alpha": 0.0, "target": 0.58, "kept": 0.57, "keeps": {1: 0.8}, "origin_widths": {1: 6},
            "sens": {1: 0.1}, "held": 0, "policy": "p/agent.pt", "fallback": {"pick": "agent"}}
    monkeypatch.setattr(alloc_walk, "fallback_plan",
                        lambda *a: calls.append(a) or ({1: 5}, info))
    monkeypatch.setattr(alloc_walk, "ModelWithRows", lambda model: types.SimpleNamespace(row_to_main_layer=[0, 1, 2]))
    lines, records = [], []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "fallback")
    env = _env()
    entry = alloc_walk._state(env)
    assert len(calls) == 1 and calls[0][0] is env and calls[0][1:] == (pytest.approx(0.58), False, {}, {})
    assert entry["widths"] == {1: 5}
    assert lines == [lines[0]] and lines[0].startswith(f"[alloc] {NET}: agent p/agent.pt plan keeps x0.570 of the params")
    assert [event for event, _k in records] == ["alloc_plan"]
    assert alloc_walk._state(env) is entry and len(calls) == 1         # once per network
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "randbest")
    monkeypatch.setattr(alloc_walk, "fallback_plan", lambda *a: pytest.fail("randbest entered fallback"))
    monkeypatch.setattr(alloc_walk, "randbest_plan",
                        lambda env, target, budget="params": ({1: 4}, {**info, "kind": "randbest", "randbest": {
                            "pick": 0, "n": 1, "score": 0.5, "proxy": "cut", "sigma": 1.0, "seed": 0}}))
    assert alloc_walk._state(_env())["widths"] == {1: 4}
