"""
v32 extensions of the val-picked fallback (``SPECTRA_ALLOC_KIND=fallback``), both default off:

* FB-BENCH: ``SPECTRA_ALLOC_FALLBACK_PRIOR`` takes a comma list (``sens_cost,uniform``). Each prior is planned from the
  same starting RNG state and origin as its own row, then the agent; every proxy scores the agent first, then the priors in
  the listed order; ``fallback_pick_multi`` picks the best prior above the agent plus the margin, and the two-way pick of
  the agent against the first prior is reported beside it. One name is v27 exactly.
* JUDGE: the proxy grammar gains ``rc<E>`` (E epochs of the FINAL recovery recipe, cosine compressed to E epochs) and the
  repeat suffix ``x<k>`` (ft / rc only); ``SPECTRA_ALLOC_FALLBACK_DETAIL=1`` logs the val loss, the per-image correctness
  bits and each draw's accuracy per proxy and arm.

CPU only, no GPU, no datasets.  python -m pytest tests/test_v32_fallback_judge.py -v
"""

import base64
import math
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

from tests.test_plan_agent import SHAPE, _instance, _loader  # noqa: E402
from src import alloc_walk, plan_agent, plan_proxies  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_EVAL_PROXY",
                 "SPECTRA_EVAL_FINAL_FT", "SPECTRA_MIN_WIDTH", "SPECTRA_GROUP_ONCE", "SPECTRA_SEED", "SPECTRA_FT_")
NET = "toy.pt"
AGENT_W = {1: 5, 2: 6}
PRIOR_W = {"sens": {1: 3, 2: 4}, "sens_cost": {1: 2, 2: 5}, "uniform": {1: 4, 2: 4}}
BURN = {"sens": 2, "sens_cost": 3, "uniform": 4}                    # RNG draws of each fake prior plan
TRAIN_AFTER = {"sens": True, "sens_cost": False, "uniform": True}  # the train / eval flag each fake plan leaves
V27_KEYS = {"network", "prior", "proxies", "margin", "pick", "budget", "target", "proxy_seed", "scores", "agent_widths",
            "prior_widths", "minutes"}
DECISION_MULTI = re.compile(r"^\[alloc\] fallback (\S+): pick (\S+) on (\S+) \(margin (\S+); (\d+\.\d) min\) \| "
                            r"two-way pick (agent|prior) \((\S+)\)$")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


# ------------------------------------------------------------------ helpers shared by the fakes


def _burn(n):
    return [random.random() for _ in range(n)], np.random.rand(n).tolist(), torch.rand(n).tolist()


def _seed_all(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draw():
    return [random.random() for _ in range(3)], np.random.rand(3).tolist(), torch.rand(3).tolist()


def _peek():
    """The next draw of every global generator, with the state put back."""
    state = alloc_walk.rng_snapshot()
    out = _draw()
    alloc_walk.rng_restore(state)
    return out


def _origin():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).eval()


def _env():
    return types.SimpleNamespace(
        selected_net_path=f"/nets/{NET}", current_model=_origin(), target_keep=0.6,
        conf=types.SimpleNamespace(device="cpu"), train_loader=object(), val_loader=object(),
        _input_shape=lambda: (4,), _dependency_groups=lambda mwr: [])


class _Fakes:
    """The fake plans and scorer ``fallback_plan`` calls. The calibration draws 1 number from each generator, a prior's
    ``plan_targets`` ``BURN[name]``, ``plan_for_env`` (the agent) 7, ``batches`` 1 and each score 2. ``scores`` maps a proxy to
    ``{arm: score}`` (arm ``agent`` or a prior's name). Every plan entry records the next draw of the generators and the
    origin's train flag, and leaves the flag where ``TRAIN_AFTER`` says (the agent leaves it off)."""

    def __init__(self, monkeypatch, scores, detail_scores=None):
        self.scores, self.detail_scores, self.calls, self.entries = scores, detail_scores, [], []
        self.agent_info = {"kind": "agent", "alpha": 0.0, "target": 0.58, "kept": 0.57, "keeps": {1: 0.8, 2: 0.7},
                           "origin_widths": {1: 6, 2: 8}, "sens": {1: 0.1, 2: 0.2}, "held": 0, "policy": "p/agent.pt"}
        self.prior_infos = {name: {"kind": name, "alpha": 0.5, "target": 0.58, "kept": 0.59, "keeps": {1: 0.5, 2: 0.6},
                                   "origin_widths": {1: 6, 2: 8}, "sens": {1: 1.0, 2: 2.0}, "held": 0}
                            for name in PRIOR_W}
        arm_of = {id(AGENT_W): "agent", **{id(w): name for name, w in PRIOR_W.items()}}
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
                arm = arm_of[id(widths)]
                fakes.calls.append(("score", arm, proxy, batches))
                _burn(2)
                return fakes.scores[proxy][arm]

            def score_detail(self, widths, proxy, batches):
                arm = arm_of[id(widths)]
                fakes.calls.append(("score_detail", arm, proxy, batches))
                _burn(2)
                if fakes.detail_scores is None:
                    pytest.fail("score_detail called with the detail off")
                return dict(fakes.detail_scores[proxy][arm])

        monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda loader, n, device: self._calibrate())
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
        self.calls.append(("plan_targets", kind, target, a, keep_floor, dict(kw)))
        self.entries.append((kind, _peek(), model.training))
        _burn(BURN[kind])
        model.train(TRAIN_AFTER[kind])
        return PRIOR_W[kind], self.prior_infos[kind]

    def _plan_for_env(self, env, target, path, k_min, **kw):
        self.calls.append(("plan_for_env", target, path, k_min, dict(kw)))
        self.entries.append(("agent", _peek(), env.current_model.training))
        _burn(7)
        env.current_model.train(True)
        return AGENT_W, self.agent_info


def _configure(monkeypatch, priors="sens", proxy="cut", margin=None):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "fallback")
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", priors)
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PROXY", proxy)
    monkeypatch.setenv("SPECTRA_PLAN_AGENT", "/x/plan_agent/policy_latest.pt")
    if margin is not None:
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_MARGIN", str(margin))


def _run_plan(monkeypatch, env=None):
    """``fallback_plan`` from a fixed RNG start: ``(widths, info, lines, records, draw_after, env)``."""
    lines, records = [], []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
    env = _env() if env is None else env                                # built first: _origin() seeds torch
    _seed_all()
    widths, info = alloc_walk.fallback_plan(env, 0.58, False, {}, {})
    return widths, info, lines, records, _draw(), env


def _reference_draw(arm):
    """What the generators hold after ONLY the picked arm's fake calls, from the same start."""
    _seed_all()
    if arm == "agent":
        _burn(7)
    else:
        _burn(1)
        _burn(BURN[arm])
    return _draw()


# ------------------------------------------------------------------ the env helpers and the pure picks


def test_fallback_priors_parse_a_list_and_one_name_is_v27(monkeypatch):
    with pytest.raises(ValueError, match="SPECTRA_ALLOC_FALLBACK_PRIOR"):
        alloc_walk.fallback_priors()                                    # no default
    for raw, want in (("sens", ["sens"]), (" SENS_COST ", ["sens_cost"]), ("sens_cost, uniform", ["sens_cost", "uniform"]),
                      ("uniform,,sens,", ["uniform", "sens"]), ("sens,sens_cost,uniform", ["sens", "sens_cost", "uniform"])):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", raw)
        assert alloc_walk.fallback_priors() == want
        assert alloc_walk.fallback_prior() == want[0]                   # the first
    for bad in ("", " , ", "uniform", "agent", "sens,agent", "randbest,sens", "sens,inner"):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", bad)
        with pytest.raises(ValueError, match="SPECTRA_ALLOC_FALLBACK_PRIOR"):
            alloc_walk.fallback_priors()                                # a lone uniform stays refused, as in v27
    for bad in ("sens,sens", "sens_cost, uniform,SENS_COST"):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", bad)
        with pytest.raises(ValueError, match="duplicate"):
            alloc_walk.fallback_priors()
    assert alloc_walk.FALLBACK_PRIORS == ("sens", "sens_cost")          # the v27 tuple stands
    assert alloc_walk.fallback_detail() is False
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_DETAIL", "1")
    assert alloc_walk.fallback_detail() is True


def test_fallback_proxies_accept_the_judge_names_and_refuse_the_rest(monkeypatch):
    for raw, want in (("ft1,ft1x3,rc2,rc5", ["ft1", "ft1x3", "rc2", "rc5"]), ("RC2X2, cut", ["rc2x2", "cut"]),
                      ("bn8,fs32,ft1", ["bn8", "fs32", "ft1"])):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PROXY", raw)
        assert alloc_walk.fallback_proxies() == want
    for bad in ("rc0", "rc", "ft1x1", "ft1x", "ft1x0", "bn8x2", "fs32x3", "cutx2", "ft12_4", "rc2_3", "ft1x3x2", "xx"):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PROXY", bad)
        with pytest.raises(ValueError):
            alloc_walk.fallback_proxies()
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PROXY", "ft1,ft1x3,ft1")
    with pytest.raises(ValueError, match="duplicate"):
        alloc_walk.fallback_proxies()


def test_parse_proxy_judge_grammar_and_the_trainers_grammar_untouched():
    # the plan trainer's grammar (two-tuple) is exactly v23's: rc<E> and x<k> are refused there
    assert [plan_agent.parse_proxy(n) for n in ("cut", "bn8", "fs32", "ft1", " FT12 ")] == [
        ("cut", 0), ("bn", 8), ("fs", 32), ("ft", 1), ("ft", 12)]
    for bad in ("rc2", "ft1x3", "rc2x2", "ft12_4", "fs0", "bn", "", "nonsense"):
        with pytest.raises(ValueError):
            plan_agent.parse_proxy(bad)
    j = lambda n: plan_agent.parse_proxy(n, judge=True)  # noqa: E731
    assert [j(n) for n in ("cut", "bn8", "fs32", "ft1", "rc2", "rc5", "ft1x3", "rc2x2", "ft10x12")] == [
        ("cut", 0, 1), ("bn", 8, 1), ("fs", 32, 1), ("ft", 1, 1), ("rc", 2, 1), ("rc", 5, 1), ("ft", 1, 3),
        ("rc", 2, 2), ("ft", 10, 12)]
    for bad in ("rc0", "rc", "rcx2", "ft1x1", "ft1x0", "ft1x", "ftx3", "bn8x2", "fs32x2", "cutx2", "ft12_4", "ft1x3x2",
                "ft-1", "ft1.5", "", None):
        with pytest.raises(ValueError):
            j(bad)
    with pytest.raises(ValueError, match="ft<E> and rc<E> only"):
        j("bn8x2")
    with pytest.raises(ValueError, match="k >= 2"):
        j("rc2x1")


def test_pick_multi_best_prior_above_the_margin_ties_to_the_first_listed():
    pick = alloc_walk.fallback_pick_multi
    assert pick(0.80, [("a", 0.81), ("b", 0.79)], 0.0) == "a"
    assert pick(0.80, [("a", 0.79), ("b", 0.82)], 0.0) == "b"           # the best prior, not the first above the bar
    assert pick(0.80, [("a", 0.82), ("b", 0.82)], 0.0) == "a"           # a tie between priors: the first listed
    assert pick(0.80, [("b", 0.82), ("a", 0.82)], 0.0) == "b"
    assert pick(0.80, [("a", 0.80), ("b", 0.80)], 0.0) == "agent"       # a tie with the agent goes to the agent
    assert pick(0.80, [("a", 0.79), ("b", 0.70)], 0.0) == "agent"
    assert pick(0.25, [("a", 0.50), ("b", 0.40)], 0.25) == "agent"      # exactly agent + margin: not strictly above
    assert pick(0.25, [("a", 0.50), ("b", 0.75)], 0.25) == "b"
    assert pick(0.40, [("a", 0.99)], 0.5) == "a"
    assert pick(0.5, [], 0.0) == "agent"
    for agent, scores, margin in ((0.8, [0.81], 0.0), (0.8, [0.8], 0.0), (0.4, [0.99], 0.5), (0.5, [1.0], 0.5),
                                  (0.25, [0.5], 0.25), (0.25, [0.75], 0.25)):
        want = alloc_walk.fallback_pick(agent, scores[0], margin)       # one prior: the two-way pick
        assert ("agent" if pick(agent, [("p", scores[0])], margin) == "agent" else "prior") == want


# ------------------------------------------------------------------ fallback_plan with fakes: one prior is v27


@pytest.mark.parametrize("arm", ["agent", "sens"])
def test_a_single_prior_keeps_v27s_lines_record_info_and_rng(monkeypatch, arm):
    _configure(monkeypatch, priors="sens", proxy="cut,fs2")
    scores = {"cut": {"agent": 0.80, "sens": 0.90 if arm == "sens" else 0.70},
              "fs2": {"agent": 0.60, "sens": 0.50}}
    fakes = _Fakes(monkeypatch, scores)
    widths, info, lines, records, after, env = _run_plan(monkeypatch)
    picked = fakes.prior_infos["sens"] if arm == "sens" else fakes.agent_info
    assert widths is (PRIOR_W["sens"] if arm == "sens" else AGENT_W)
    assert {k: v for k, v in info.items() if k != "fallback"} == picked and info["kind"] == picked["kind"]
    assert after == _reference_draw(arm)
    want_scores = {p: {"agent": s["agent"], "prior": s["sens"]} for p, s in scores.items()}
    assert set(info["fallback"]) == {"pick", "prior", "proxies", "margin", "scores", "minutes"}
    assert info["fallback"]["pick"] == ("prior" if arm == "sens" else "agent") and info["fallback"]["scores"] == want_scores
    assert info["fallback"]["prior"] == "sens"
    # exactly v27's lines: one per proxy, then the decision line
    assert lines == [
        f"[alloc] fallback {NET} cut: agent 0.8000 prior sens {scores['cut']['sens']:.4f} "
        f"(prior − agent {scores['cut']['sens'] - 0.80:+.4f})",
        f"[alloc] fallback {NET} fs2: agent 0.6000 prior sens 0.5000 (prior − agent -0.1000)",
        lines[2]]
    assert re.match(r"^\[alloc\] fallback toy.pt: pick (agent|prior) on cut \(margin 0; \d+\.\d min\)$", lines[2])
    event, rec = records[0]
    assert event == "alloc_fallback" and len(records) == 1
    assert set(rec) == V27_KEYS | {"fs_lr"} and rec["scores"] == want_scores and rec["prior_widths"] == {"1": 3, "2": 4}
    # the calls: the prior's own row, the agent's, the scorer; the agent scored first, on the same batches
    assert fakes.calls[:3] == [("calibration",), ("plan_targets", "sens", 0.58, alloc_walk.alpha(), alloc_walk.min_keep(), {}),
                               ("plan_for_env", 0.58, "/x/plan_agent/policy_latest.pt", alloc_walk.min_keep(), {})]
    assert [c[:3] for c in fakes.calls[4:]] == [("batches", "cut", 0), ("score", "agent", "cut"), ("score", "sens", "cut"),
                                               ("batches", "fs2", 0), ("score", "agent", "fs2"), ("score", "sens", "fs2")]


# ------------------------------------------------------------------ several priors


def _multi_scores(agent, **priors):
    return {"agent": agent, **priors}


@pytest.mark.parametrize("want", ["agent", "sens_cost", "uniform"])
def test_several_priors_plan_from_the_same_start_score_agent_first_and_pick_the_best(monkeypatch, want):
    _configure(monkeypatch, priors="sens_cost,uniform", proxy="ft1,cut")
    table = {"agent": {"agent": 0.80, "sens_cost": 0.79, "uniform": 0.79},
             "sens_cost": {"agent": 0.80, "sens_cost": 0.83, "uniform": 0.81},
             "uniform": {"agent": 0.80, "sens_cost": 0.81, "uniform": 0.84}}[want]
    scores = {"ft1": dict(table), "cut": {"agent": 0.5, "sens_cost": 0.9, "uniform": 0.1}}   # only the first proxy decides
    fakes = _Fakes(monkeypatch, scores)
    widths, info, lines, records, after, env = _run_plan(monkeypatch)
    # every plan starts from the same RNG state and train flag: the prior i's calibration burns 1, then plan_targets
    # sees what ``_burn(1)`` leaves from the start; the agent sees the start itself
    _seed_all()
    start = _peek()
    _burn(1)
    after_calibration = _peek()
    assert [(k, f) for k, _p, f in fakes.entries] == [
        ("sens_cost", False), ("uniform", False), ("agent", False)]       # the origin's flag put back before each plan
    assert [k for k, _p, _f in fakes.entries] == ["sens_cost", "uniform", "agent"]
    assert [p for _k, p, _f in fakes.entries] == [after_calibration, after_calibration, start]
    assert [c[:2] for c in fakes.calls if c[0] in ("calibration", "plan_targets", "plan_for_env")] == [
        ("calibration",), ("plan_targets", "sens_cost"), ("calibration",), ("plan_targets", "uniform"), ("plan_for_env", 0.58)]
    # the pick, and what is returned is that arm's own plan with the RNG and the flag left as that arm alone leaves them
    pick = "agent" if want == "agent" else "prior"
    assert widths is (AGENT_W if want == "agent" else PRIOR_W[want])
    expected_info = fakes.agent_info if want == "agent" else fakes.prior_infos[want]
    assert {k: v for k, v in info.items() if k != "fallback"} == expected_info and info["kind"] == expected_info["kind"]
    assert after == _reference_draw(want)
    assert env.current_model.training is (True if want == "agent" else TRAIN_AFTER[want])
    summary = info["fallback"]
    assert set(summary) == {"pick", "prior", "proxies", "margin", "scores", "minutes", "priors", "pick_name", "pick2"}
    assert (summary["pick"], summary["pick_name"], summary["prior"], summary["priors"]) == (
        pick, want, "sens_cost", ["sens_cost", "uniform"])
    assert summary["pick2"] == alloc_walk.fallback_pick(table["agent"], table["sens_cost"], 0.0)   # agent vs the FIRST prior
    assert summary["scores"]["ft1"] == {"agent": table["agent"], "prior": table["sens_cost"],
                                        "priors": {"sens_cost": table["sens_cost"], "uniform": table["uniform"]}}
    assert summary["scores"]["cut"]["prior"] == 0.9 and summary["scores"]["cut"]["priors"] == {"sens_cost": 0.9, "uniform": 0.1}
    # the scoring order: per proxy one batches draw, the agent, then the priors in the listed order
    assert [c[:3] for c in fakes.calls if c[0] in ("batches", "score")] == [
        ("batches", "ft1", 0), ("score", "agent", "ft1"), ("score", "sens_cost", "ft1"), ("score", "uniform", "ft1"),
        ("batches", "cut", 0), ("score", "agent", "cut"), ("score", "sens_cost", "cut"), ("score", "uniform", "cut")]
    assert len({c[3] for c in fakes.calls if c[0] == "score" and c[2] == "ft1"}) == 1     # the same batches
    # the lines: the first prior's v27 line, one line per further prior, then the decision line
    a, p1, p2 = table["agent"], table["sens_cost"], table["uniform"]
    assert lines[:2] == [f"[alloc] fallback {NET} ft1: agent {a:.4f} prior sens_cost {p1:.4f} (prior − agent {p1 - a:+.4f})",
                         f"[alloc] fallback {NET} ft1: agent {a:.4f} prior2 uniform {p2:.4f} (prior2 − agent {p2 - a:+.4f})"]
    assert lines[2].startswith(f"[alloc] fallback {NET} cut: agent 0.5000 prior sens_cost 0.9000 ")
    assert lines[3] == f"[alloc] fallback {NET} cut: agent 0.5000 prior2 uniform 0.1000 (prior2 − agent -0.4000)"
    decision = DECISION_MULTI.match(lines[4])
    assert decision and len(lines) == 5
    assert decision.groups()[:3] == (NET, want, "ft1") and decision.groups()[5:] == (summary["pick2"], "sens_cost")
    event, rec = records[0]
    assert set(rec) == V27_KEYS | {"priors", "pick_name", "pick2", "prior_widths_all"}
    assert rec["pick"] == pick and rec["pick_name"] == want and rec["pick2"] == summary["pick2"]
    assert rec["priors"] == ["sens_cost", "uniform"] and rec["prior"] == "sens_cost"
    assert rec["prior_widths"] == {"1": 2, "2": 5}                       # the first prior's, as in v27
    assert rec["prior_widths_all"] == {"sens_cost": {"1": 2, "2": 5}, "uniform": {"1": 4, "2": 4}}
    assert rec["scores"] == summary["scores"]


def test_the_margin_a_tie_between_priors_and_the_two_way_pick_can_differ(monkeypatch):
    _configure(monkeypatch, priors="sens_cost,uniform,sens", proxy="cut", margin=0.05)
    # uniform (second) is the best prior and clears the margin; sens_cost (the FB2 prior) does not
    scores = {"cut": {"agent": 0.80, "sens_cost": 0.84, "uniform": 0.86, "sens": 0.85}}
    fakes = _Fakes(monkeypatch, scores)
    _w, info, lines, records, after, _env_ = _run_plan(monkeypatch)
    assert (info["fallback"]["pick"], info["fallback"]["pick_name"], info["fallback"]["pick2"]) == ("prior", "uniform", "agent")
    assert after == _reference_draw("uniform")
    assert DECISION_MULTI.match(lines[3]).groups()[1:] == ("uniform", "cut", "0.05", lines[3].split("(margin 0.05; ")[1][:3],
                                                           "agent", "sens_cost")
    assert lines[1].split("prior2 ")[1].startswith("uniform 0.8600") and lines[2].split("prior3 ")[1].startswith("sens 0.8500")
    # a tie between the two best priors goes to the first listed; the third prior's plan is the one NOT picked
    scores = {"cut": {"agent": 0.80, "sens_cost": 0.86, "uniform": 0.86, "sens": 0.85}}
    fakes = _Fakes(monkeypatch, scores)
    _w, info, _l, records, after, _e = _run_plan(monkeypatch)
    assert info["fallback"]["pick_name"] == "sens_cost" and info["fallback"]["pick2"] == "prior"
    assert after == _reference_draw("sens_cost") and records[0][1]["prior_widths_all"].keys() == {"sens_cost", "uniform", "sens"}
    # nobody clears the margin: the agent
    scores = {"cut": {"agent": 0.80, "sens_cost": 0.84, "uniform": 0.85, "sens": 0.85}}
    fakes = _Fakes(monkeypatch, scores)
    _w, info, _l, _r, after, _e = _run_plan(monkeypatch)
    assert info["fallback"]["pick_name"] == "agent" and info["fallback"]["pick"] == "agent" and after == _reference_draw("agent")


def test_a_prior_plan_that_changes_the_origin_raises_naming_the_prior(monkeypatch):
    _configure(monkeypatch, priors="sens_cost,uniform", proxy="cut")
    fakes = _Fakes(monkeypatch, {"cut": {"agent": 0.8, "sens_cost": 0.7, "uniform": 0.7}})
    plan = fakes._plan_targets

    def mutating(model, batches, shape, kind, *a, **k):
        out = plan(model, batches, shape, kind, *a, **k)
        if kind == "uniform":
            with torch.no_grad():
                model[0].weight.add_(1.0)
        return out

    monkeypatch.setattr(alloc_walk, "plan_targets", mutating)
    with pytest.raises(RuntimeError, match=f"{NET}: the uniform plan changed the origin's weights"):
        _run_plan(monkeypatch)


# ------------------------------------------------------------------ the detail mode with fakes


def test_detail_adds_one_line_per_proxy_and_arm_and_a_record_field_and_the_decision_still_uses_the_accuracy(monkeypatch):
    _configure(monkeypatch, priors="sens_cost,uniform", proxy="ft1,rc2")
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_DETAIL", "1")

    def read(acc, loss):
        return {"acc": acc, "loss": loss, "bits": "AAAA", "n": 20, "draws": [acc]}

    details = {"ft1": {"agent": read(0.70, 1.1), "sens_cost": read(0.72, 1.0), "uniform": read(0.68, 1.2)},
               "rc2": {"agent": read(0.75, 0.9), "sens_cost": read(0.74, 0.95), "uniform": read(0.80, 0.8)}}
    scores = {p: {arm: d["acc"] for arm, d in arms.items()} for p, arms in details.items()}
    fakes = _Fakes(monkeypatch, scores, detail_scores=details)
    _w, info, lines, records, after, _e = _run_plan(monkeypatch)
    assert info["fallback"]["pick_name"] == "sens_cost"                  # ft1 decides: rc2 prefers uniform
    assert after == _reference_draw("sens_cost")
    assert not [c for c in fakes.calls if c[0] == "score"]               # the detail path scores through score_detail
    assert [c[:3] for c in fakes.calls if c[0] == "score_detail"][:3] == [
        ("score_detail", "agent", "ft1"), ("score_detail", "sens_cost", "ft1"), ("score_detail", "uniform", "ft1")]
    expect = [f"[alloc] fallback {NET} ft1 agent: acc 0.7000 loss 1.1000", f"[alloc] fallback {NET} ft1 sens_cost: acc 0.7200 loss 1.0000",
              f"[alloc] fallback {NET} ft1 uniform: acc 0.6800 loss 1.2000", f"[alloc] fallback {NET} rc2 agent: acc 0.7500 loss 0.9000",
              f"[alloc] fallback {NET} rc2 sens_cost: acc 0.7400 loss 0.9500", f"[alloc] fallback {NET} rc2 uniform: acc 0.8000 loss 0.8000"]
    assert lines[4:10] == expect and DECISION_MULTI.match(lines[10]) and len(lines) == 11
    rec = records[0][1]
    assert rec["details"] == details and rec["scores"]["rc2"]["priors"] == {"sens_cost": 0.74, "uniform": 0.80}
    # one prior: the detail arms are named by the prior, the other record keys are v27's
    _configure(monkeypatch, priors="sens_cost", proxy="ft1")
    fakes = _Fakes(monkeypatch, {"ft1": scores["ft1"]}, detail_scores={"ft1": details["ft1"]})
    _w, info, lines, records, _a, _e = _run_plan(monkeypatch)
    assert set(records[0][1]) == V27_KEYS | {"details"} and list(records[0][1]["details"]["ft1"]) == ["agent", "sens_cost"]
    assert len(lines) == 4 and "priors" not in info["fallback"]


# ------------------------------------------------------------------ PlanScorer: rc<E>, x<k>, detail, the cache


KEYS = ("SPECTRA_FT_OPTIM", "SPECTRA_FT_SGD_LR", "SPECTRA_FT_MOMENTUM", "SPECTRA_FT_WD", "SPECTRA_FT_COSINE",
        "SPECTRA_FT_SCHEDULE", "SPECTRA_FT_MIXUP", "SPECTRA_FT_LABEL_SMOOTH", "SPECTRA_FT_KD", "SPECTRA_FT_WARMUP_EPOCHS")


class _Spy(ClassificationHandler):
    """The real handler, recording the recipe environment and the arguments of every ``train_model`` call."""
    calls = []

    def train_model(self, loader, **kw):
        _Spy.calls.append({"env": {k: os.environ.get(k) for k in KEYS}, "loader": loader, "kw": dict(kw)})
        return super().train_model(loader, **kw)


def _spy(model):
    return _Spy(model, nn.CrossEntropyLoss())


class _Fake:
    """A handler that trains nothing but prints a recipe line, and records what it was called with."""
    calls = []

    def __init__(self, model):
        self.model = model

    def train_model(self, loader, **kw):
        _Fake.calls.append({"env": {k: os.environ.get(k) for k in KEYS}, "loader": loader, "kw": dict(kw)})
        utils.print_flush("Fine-tune recipe: optim=sgd lr=0.01 trainable=3")


def _scorer(create_handler=_spy, name=NET):
    inst = _instance()
    scorer = plan_proxies.PlanScorer(inst.model, inst.plan, _loader(seed=1, shuffle=False), "cpu",
                                     train_loader=inst.train_loader, input_shape=SHAPE, create_handler=create_handler,
                                     name=name)
    widths, _info = plan_agent.decode([0.1 * k for k in range(inst.n_groups)], inst.pm, 0.6)
    return scorer, widths, inst


def _rng_state():
    npy = np.random.get_state()
    return torch.get_rng_state().tolist(), random.getstate(), (npy[0], npy[1].tolist(), tuple(npy[2:]))


def test_final_recipe_keys_equal_the_runners():
    runner = pytest.importorskip("a2c_agent_reinforce_runner")
    assert plan_proxies._FINAL_FT_ENV_KEYS == runner._FINAL_FT_ENV_KEYS


def test_rc_uses_the_final_recipe_on_the_final_loader_with_kd_off_and_puts_the_environment_back(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_KD", "1")                             # a walk's KD must not reach the judge
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_BATCH", "8")
    _Spy.calls = []
    scorer, widths, inst = _scorer()
    shown = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: shown.append(str(s)))
    before_env = {k: os.environ.get(k) for k in KEYS}
    before_rng = _rng_state()
    s = scorer.score(widths, "rc2", scorer.batches("rc2", 5))
    assert 0.0 <= s <= 1.0 and _rng_state() == before_rng
    assert {k: os.environ.get(k) for k in KEYS} == before_env           # put back
    call = _Spy.calls[0]
    assert call["env"] == {"SPECTRA_FT_OPTIM": "sgd", "SPECTRA_FT_SGD_LR": "0.01", "SPECTRA_FT_MOMENTUM": "0.9",
                           "SPECTRA_FT_WD": "5e-4", "SPECTRA_FT_COSINE": "1", "SPECTRA_FT_SCHEDULE": "",
                           "SPECTRA_FT_MIXUP": "0", "SPECTRA_FT_LABEL_SMOOTH": "0", "SPECTRA_FT_KD": "0",
                           "SPECTRA_FT_WARMUP_EPOCHS": None}
    assert call["kw"]["max_epochs"] == 2 and call["kw"]["patience"] == 3 and call["kw"]["allow_reinit_retry"] is False
    assert "keep_last" not in call["kw"] and "cuda_graph" not in call["kw"]
    loader, aug = utils.final_ft_train_loader(inst.train_loader, 8)
    assert call["loader"].batch_size == 8 and call["loader"] is not inst.train_loader
    assert type(call["loader"].dataset) is type(inst.train_loader.dataset)
    # the handler's own recipe line, through the real handler: sgd, the eval lr, cosine over 2 epochs, no kd
    line = scorer.recipe_line
    for part in ("optim=sgd", "lr=0.01", "cosine=1", "schedule=cosine", "wd=0.0005", "kd=0", "epochs=2", "select=train_loss",
                 "bs=8", f"aug={aug}"):
        assert part in line, (part, line)
    assert not any("Fine-tune recipe" in line for line in shown)         # the handler's lines stay captured
    # the knobs the runner reads: lr, warmcos + warmup, keep_last, cuda_graph
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_LR", "0.05")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCHEDULE", "warmcos")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_WARMUP", "0.5")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SELECT", "last")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH", "1")
    _Fake.calls = []
    scorer, widths, inst = _scorer(create_handler=_Fake)
    scorer.score(widths, "rc3", 7)
    call = _Fake.calls[0]
    assert call["env"]["SPECTRA_FT_SGD_LR"] == "0.05" and call["env"]["SPECTRA_FT_SCHEDULE"] == "warmcos"
    assert call["env"]["SPECTRA_FT_WARMUP_EPOCHS"] == "0.5" and call["env"]["SPECTRA_FT_KD"] == "0"
    assert call["kw"]["keep_last"] is True and call["kw"]["cuda_graph"] is True and call["kw"]["max_epochs"] == 3
    assert os.environ.get("SPECTRA_FT_WARMUP_EPOCHS") is None            # put back


def test_ft_keeps_the_handlers_own_recipe_and_the_train_loader(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_OPTIM", "sgd")                         # a walk key: ft clears it, rc sets its own
    _Spy.calls = []
    scorer, widths, inst = _scorer()
    scorer.score(widths, "ft1", 5)
    call = _Spy.calls[0]
    assert set(call["env"].values()) == {None} and call["loader"] is inst.train_loader and call["kw"]["max_epochs"] == 1
    assert scorer.recipe_line.startswith("optim=adam ") and "bs=" not in scorer.recipe_line


def test_rc_and_ft_are_deterministic_under_their_seed_and_leave_the_origin_and_rng_alone(monkeypatch):
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    scorer, widths, inst = _scorer()
    origin = {k: v.clone() for k, v in inst.model.state_dict().items()}
    before = _rng_state()
    for proxy in ("ft1", "rc1", "rc2"):
        first = scorer.score(widths, proxy, 5)
        scorer._cache.clear()
        assert scorer.score(widths, proxy, 5) == first                    # trained again, same seed: same score
    assert _rng_state() == before
    for key, value in inst.model.state_dict().items():
        assert torch.equal(value, origin[key]), key


def test_x_k_is_the_mean_of_k_draws_at_seeds_s_to_s_plus_k_minus_1(monkeypatch):
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    scorer, widths, _inst = _scorer()
    singles = [_scorer()[0].score(widths, "ft1", 5 + j) for j in range(3)]
    seen = []
    original = scorer._train_read
    monkeypatch.setattr(scorer, "_train_read", lambda w, kind, e, seed, detail: seen.append((kind, e, seed)) or original(
        w, kind, e, seed, detail))
    assert scorer.batches("ft1x3", 5) == 5 and scorer.batches("rc2x2", 5) == 5
    x3 = scorer.score(widths, "ft1x3", 5)
    assert seen == [("ft", 1, 5), ("ft", 1, 6), ("ft", 1, 7)]
    assert x3 == pytest.approx(sum(singles) / 3, abs=1e-12)
    rc = [_scorer()[0].score(widths, "rc1", 9 + j) for j in range(2)]
    seen.clear()
    assert scorer.score(widths, "rc1x2", 9) == pytest.approx(sum(rc) / 2, abs=1e-12)
    assert seen == [("rc", 1, 9), ("rc", 1, 10)]
    with pytest.raises(ValueError, match="seed"):
        scorer.score(widths, "ft1x3", [])
    with pytest.raises(ValueError):
        scorer.score(widths, "bn3x2", [])


def test_ft1_alone_equals_ft1_in_a_list_and_a_shared_draw_is_trained_once(monkeypatch):
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    _Spy.calls = []
    alone, widths, _i = _scorer()
    ft1 = alone.score(widths, "ft1", alone.batches("ft1", 5))
    assert len(_Spy.calls) == 1
    _Spy.calls = []
    many, widths, _i = _scorer()
    got = {}
    for proxy in ("cut", "ft1x3", "rc2", "ft1", "bn3", "fs2"):            # ft1x3 first: its first draw is ft1's
        b = many.batches(proxy, 5)
        got[proxy] = many.score(widths, proxy, b)
    assert got["ft1"] == ft1                                              # identical: the cache returns ft1x3's first draw
    trains = [(c["kw"]["max_epochs"], c["loader"].batch_size) for c in _Spy.calls]
    assert len(trains) == 3 + 1 and trains.count((2, 128)) == 1 and trains.count((1, 16)) == 3   # ft1 was not trained again
    # a different plan is a different key
    other = {r: max(1, w - 1) for r, w in widths.items()}
    many.score(other, "ft1", 5)
    assert len(_Spy.calls) == 5
    # and the cache never changes a score: every proxy reads the same alone as in the list
    for proxy in ("cut", "bn3", "fs2", "rc2", "ft1x3"):
        fresh, w2, _i = _scorer()
        assert fresh.score(w2, proxy, fresh.batches(proxy, 5)) == got[proxy], proxy


def test_detail_fields_one_pass_loss_bits_and_draws(monkeypatch):
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    scorer, widths, _inst = _scorer()
    n_val = sum(len(y) for _x, y in scorer.val)
    assert n_val == 64
    for proxy in ("cut", "bn3", "fs2", "ft1", "rc1", "ft1x3", "rc1x2"):
        b = scorer.batches(proxy, 5)
        d = scorer.score_detail(widths, proxy, b)
        assert d["acc"] == scorer.score(widths, proxy, b), proxy             # the decision's number is unchanged
        assert math.isfinite(d["loss"]) and d["loss"] > 0.0 and d["n"] == n_val
        raw = base64.b64decode(d["bits"])
        assert len(raw) == (n_val + 7) // 8
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n_val]
        k = plan_agent.parse_proxy(proxy, judge=True)[2]
        assert len(d["draws"]) == k and d["acc"] == pytest.approx(sum(d["draws"]) / k, abs=1e-12)
        if k == 1:
            assert d["draws"] == [d["acc"]] and bits.sum() / n_val == pytest.approx(d["acc"]) and "bits_draws" not in d
        else:
            assert len(d["bits_draws"]) == k and d["bits_draws"][0] == d["bits"]
            counts = [np.unpackbits(np.frombuffer(base64.b64decode(s), dtype=np.uint8))[:n_val].sum() / n_val
                      for s in d["bits_draws"]]
            assert counts == pytest.approx(d["draws"])                         # each draw's bits give its accuracy
    # ft1x3's first draw is ft1's: the same bits
    one = scorer.score_detail(widths, "ft1", 5)
    three = scorer.score_detail(widths, "ft1x3", 5)
    assert three["bits"] == one["bits"] and three["draws"][0] == one["acc"]
    # the cross-entropy is the val mean of the logits' loss: check on the cut proxy against a direct pass
    d = scorer.score_detail(widths, "cut", [])
    work = scorer.cutter.cut(plan_agent.rates_of(widths, scorer.widths0)).eval()
    with torch.no_grad():
        direct = sum(float(nn.functional.cross_entropy(work(x), y, reduction="sum")) for x, y in scorer.val) / n_val
    assert d["loss"] == pytest.approx(direct, rel=1e-6)


# ------------------------------------------------------------------ the real plans and scorer through fallback_plan


def test_fallback_plan_end_to_end_on_a_toy_net_priors_equal_their_own_rows_and_ft1_is_the_same_alone(monkeypatch):
    from tests.test_v23_randbest import _env as toy_env
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    target = 0.58
    agent_widths = {}

    def agent(env, tgt, path, k_min, **kw):
        _burn(5)
        widths, info = alloc_walk.plan_targets(
            env.current_model, [], SHAPE, "uniform", 0.8, alloc_walk.alpha(), alloc_walk.min_keep())
        agent_widths.update(widths)
        return widths, {**info, "kind": "agent", "policy": "p/agent.pt"}

    monkeypatch.setattr(plan_agent, "plan_for_env", agent)
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "fallback")
    monkeypatch.setenv("SPECTRA_PLAN_AGENT", "/x/plan_agent/policy_latest.pt")

    def run(priors, proxies, detail=False):
        records = []
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", priors)
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PROXY", proxies)
        if detail:
            monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_DETAIL", "1")
        else:
            monkeypatch.delenv("SPECTRA_ALLOC_FALLBACK_DETAIL", raising=False)
        monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
        env = toy_env()
        _seed_all()
        widths, info = alloc_walk.fallback_plan(env, target, False, {}, {})
        found = [k for event, k in records if event == "alloc_fallback"]
        assert len(found) == 1, [event for event, _k in records]
        return widths, info, found[0]

    def own_row(kind):
        env = toy_env()
        _seed_all()
        batches = group_sensitivity.calibration_batches(env.train_loader, group_sensitivity.CALIB_BATCHES, "cpu")
        return alloc_walk.plan_targets(env.current_model, batches, SHAPE, kind, target, alloc_walk.alpha(),
                                       alloc_walk.min_keep())[0]

    widths, info, rec = run("sens_cost,uniform,sens", "ft1,ft1x2,rc1,cut", detail=True)
    for name in ("sens_cost", "uniform", "sens"):
        assert rec["prior_widths_all"][name] == {str(r): int(w) for r, w in own_row(name).items()}, name
    assert rec["priors"] == ["sens_cost", "uniform", "sens"] and rec["agent_widths"] == {str(r): int(w) for r, w in agent_widths.items()}
    assert set(rec["details"]) == {"ft1", "ft1x2", "rc1", "cut"} and set(rec["details"]["ft1"]) == {"agent", "sens_cost", "uniform", "sens"}
    for proxy, arms in rec["details"].items():
        for arm, d in arms.items():
            assert d["acc"] == (rec["scores"][proxy]["agent"] if arm == "agent" else rec["scores"][proxy]["priors"][arm])
            assert math.isfinite(d["loss"]) and d["n"] == 64
    # ft1 scored in this list equals ft1 listed alone with one prior (the first prior's score, per arm)
    _w1, info1, rec1 = run("sens_cost", "ft1")
    assert rec1["scores"]["ft1"]["agent"] == rec["scores"]["ft1"]["agent"]
    assert rec1["scores"]["ft1"]["prior"] == rec["scores"]["ft1"]["prior"]
    assert set(rec1) == V27_KEYS and set(info1["fallback"]) == {"pick", "prior", "proxies", "margin", "scores", "minutes"}
    # and with the detail on, one prior, the decision and the scores are the same
    _w2, info2, rec2 = run("sens_cost", "ft1", detail=True)
    assert rec2["scores"] == rec1["scores"] and info2["fallback"]["pick"] == info1["fallback"]["pick"]
