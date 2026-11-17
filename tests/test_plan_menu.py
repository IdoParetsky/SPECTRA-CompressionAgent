"""
v36 ``SPECTRA_PLAN_MENU`` (stage 1 of the agent-development road; default off): the plan agent chooses and blends
several no-agent rules per network, then corrects the blend with the residual's scores. Each rule of the menu gives a
weight per group, the one the alloc walk's own row of that kind hands its landing (``plan_agent.menu_rules``); it is
log w_g = Σ_k π_k log ŵ_k,g + the residual's score, π = softmax(u) (``menu_decode``, ``menu_shares``); u comes from a
network-level head of the policy (``PlanPolicy.menu_logits``) and is sampled beside the scores in training. A pure share
is that rule's ``scale_decode`` plan exactly, and the zero-init agent's mean plan is the anchor's, i.e. the residual
agent's prior. With the flag unset every decoder, draw, record, log line and checkpoint is as in tree_v35.

The letters in the section headers are the brief's: (a) flag off, (b) pure shares, (c) init, (d) log-prob, (e) gradient,
(f) landing, (g) save / load, (h) eval against trainer, (i) anchor.

CPU only, no datasets.  python -m pytest tests/test_plan_menu.py -v
"""

import json
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

from tests.test_plan_agent import RATES, SHAPE, ZOO, _loader, _plan, _randomize_norms, toy_state  # noqa: E402
from tests.test_plan_agent import _instance as _toy_instance  # noqa: E402
from tests.test_v16_sens_cost import _batches  # noqa: E402
from tests.test_v17_plan_min_width import _floor_ok  # noqa: E402
from tests.test_v25_plan_resume import _equal, _steps, _synthetic_reward, _weights  # noqa: E402
from tests.test_v25_plan_resume import _run as _job  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, plan_agent, plan_trainer, state_dump  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_MIN_WIDTH")
TOL = {"params": 0.05, "flops": 0.1}  # one stage-1 channel of thin_r20_w4 is ~10 % of its MACs
MENU = ("sens", "uniform", "sens_cost", "sens_thresh")
MENU_FLAG = ",".join(MENU)
# T1's FLOPs line: the residual on sens, the trust region, the walk's width floor
T1 = {"SPECTRA_PLAN_RESIDUAL": "sens", "SPECTRA_PLAN_RESIDUAL_ZMAX": "0.3", "SPECTRA_PLAN_MIN_WIDTH": "2"}
T1_TOKENS = "} min_width=2 residual=sens alpha=0.5 sigma=0.25 zmax=0.3"
MENU_META = {"menu": list(MENU), "menu_anchor": "sens", "menu_init": 6.0, "menu_sigma": 0.5, "menu_share_floor": 0.005}
RESIDUAL_META = {"residual": "sens", "residual_alpha": 0.5, "residual_sigma": 0.25, "residual_zmax": 0.3,
                 "min_width": 2}
SETTINGS = {"kinds": MENU, "init": 6.0, "sigma": 0.5}  # plan_trainer.menu_settings under the default flags
RES = ("sens", 0.5)

_NETS, _TOY = {}, {}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


@pytest.fixture
def v10_env(monkeypatch):
    from tests.test_plan_agent import V10_FLAGS
    for key, value in V10_FLAGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SPECTRA_STATE_TOKENS", raising=False)


# ------------------------------------------------------------------ helpers


def _net(name="thin_r20_w4"):
    """One measured origin per model, shared by the tests (no decoder changes it): the model, its plan, the calibration
    batches, and on them the sensitivities, the costs and the damage curves as ``NetInstance.from_env`` and
    ``plan_for_env`` measure them, with both cost models."""
    if name not in _NETS:
        torch.manual_seed(0)
        model = ZOO[name]().eval()
        _randomize_norms(model)
        plan, batches, costs = _plan(model), _batches(), {}
        sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, costs=costs)
        qs, loss, raw = alloc_walk.curve_probes(model, plan, batches, SHAPE)
        _NETS[name] = types.SimpleNamespace(
            model=model, plan=plan, batches=batches, sens=sens, costs=costs,
            curves={"qs": qs, "loss": loss, "raw": raw, "seconds": 0.0},
            cm={"params": plan_agent.ParamModel(model, plan), "flops": plan_agent.FlopModel(model, plan, SHAPE)})
    return _NETS[name]


def _rules(net, budget, kappa, min_width=2, kinds=MENU, curves=None):
    return plan_agent.menu_rules(kinds, net.cm[budget], kappa, net.sens, net.costs, curves or net.curves, budget, 0.5,
                                 0.1, min_width)


def _hand_curves(net):
    """Damage curves written by hand: the loss rises with the depth of the cut at a rate that differs by group and every
    group saves an equal share of the budget, so the threshold, and sens_thresh's weights with it, move with the target
    (the random-weight test nets lose nothing to most cuts, and their own threshold sits at 0 for every target)."""
    rows, qs = list(net.cm["params"].rows), (0.75, 0.5, 0.25, 0.1)
    p0, f0 = float(net.cm["params"].total0), float(net.cm["flops"].total0)
    return {"qs": qs, "seconds": 0.0,
            "loss": {row: {q: (1.0 - q) * 0.01 * (k + 1) for q in qs} for k, row in enumerate(rows)},
            "raw": {row: {q: ((1.0 - q) * p0 / len(rows), (1.0 - q) * f0 / len(rows)) for q in qs} for row in rows}}


def _instance():
    """The toy instance of ``tests/test_plan_agent.py`` (synthetic sens) with the costs and the damage curves measured,
    once for all tests: the model is rebuilt from the same seed each time."""
    inst = _toy_instance()
    if not _TOY:
        costs = {}
        group_sensitivity.group_sensitivity(inst.model, inst.plan, _batches(), SHAPE, costs=costs)
        qs, loss, raw = alloc_walk.curve_probes(inst.model, inst.plan, _batches(), SHAPE)
        _TOY.update(costs=costs, curves={"qs": qs, "loss": loss, "raw": raw, "seconds": 1.5})
    inst.cost, inst.curves = _TOY["costs"], _TOY["curves"]
    return inst


def _reward(inst):
    """A reward the K plans never tie on and that draws nothing: a fixed weighted sum of the plan's keeps."""
    rows, widths0 = inst.pm.rows, inst.pm.widths0
    return lambda widths, proxy, batches: 10.0 * sum((k + 1) * widths[row] / widths0[row]
                                                     for k, row in enumerate(rows)) / len(rows)


def _spy_draws(monkeypatch):
    """Every ``plan_agent.sample_scores`` call of the trainer, in order: (the mean it drew around, sigma, the draw)."""
    draws, real = [], plan_agent.sample_scores

    def draw(mean, sigma, k, generator=None):
        out = real(mean, sigma, k, generator)
        draws.append((mean.detach().clone(), float(sigma), out.clone()))
        return out

    monkeypatch.setattr(plan_agent, "sample_scores", draw)
    return draws


def _train(monkeypatch, folder, inst, budget="flops", instances=5, calls=None, **flags):
    """One trainer job on ``inst`` under ``flags``: ``(lines, records, blob)``; the blob is read without building it."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(plan_agent.NetInstance, "from_env",
                        classmethod(lambda cls, env, p, m, l, **kw: (calls if calls is not None else []).append(kw)
                                    or inst))
    monkeypatch.setattr(plan_trainer, "out_dir", lambda: str(folder))
    for key, value in {"SPECTRA_PLAN_INSTANCES": str(instances), "SPECTRA_PLAN_K": "3", "SPECTRA_PLAN_BATCH": "2",
                       "SPECTRA_PLAN_PROXY": "cut", "SPECTRA_PLAN_REF_EVERY": "2", "SPECTRA_PLAN_SAVE_EVERY": "4",
                       "SPECTRA_PLAN_NETS": "toy", "SPECTRA_PLAN_BUDGET": budget, **flags}.items():
        monkeypatch.setenv(key, value)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    plan_trainer.run(None, [("toy.pt", (None, None))])
    with open(Path(folder) / "plan_train.jsonl", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    return lines, records, plan_agent.read_blob(str(Path(folder) / "policy_latest.pt"))


def _seed_all(seed=0):
    """Every global generator at one state, as a fresh job starts (the trainer seeds only torch's and its own)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _menu_text(line):
    """The menu's own tokens in a trainer line (the run folder, named after the test, says menu too)."""
    return " menu=" in line or line.startswith("[plan] menu") or " menu_" in line


def _hot_policy(seed=1, menu=len(MENU)):
    """A menu policy far from its init: a random score head, a menu head that reads the tokens a little and the
    network-level inputs a lot, and a bias that leaves every rule a live share."""
    policy = plan_agent.PlanPolicy(63, menu=menu)
    g, d = torch.Generator().manual_seed(seed), policy.encoder.output_dim
    with torch.no_grad():
        policy.head.weight.normal_(std=0.5, generator=g)
        policy.menu_head.weight[:, :d].normal_(std=0.02, generator=g)
        policy.menu_head.weight[:, d:].normal_(std=1.0, generator=g)
        policy.menu_head.bias.copy_(torch.tensor([1.0, 0.4, -0.3, 0.6][:menu]))
    return policy


def _origin(monkeypatch):
    """One origin for both sides of (h): a fake env that ``NetInstance.from_env`` and ``plan_for_env`` both read, with
    one state and an unshuffled train loader, so both draw the same calibration batches."""
    torch.manual_seed(0)
    model = ZOO["thin_r20_w4"]().eval()
    _randomize_norms(model)
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    plan = group_sensitivity.group_plan(mwr, groups)
    state = toy_state(len(state_dump.token_rows(mwr, groups, plan)))
    monkeypatch.setattr(state_dump, "encode_origin", lambda env, m, g: dict(state))
    env = types.SimpleNamespace(conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES),
                                current_model=model, target_keep=0.6, _input_shape=lambda: SHAPE,
                                train_loader=_loader(seed=1, shuffle=False), val_loader=_loader(seed=2, shuffle=False),
                                _dependency_groups=lambda m: channel_groups.build_channel_groups(m.model),
                                create_learning_handler=None, reset=lambda **kw: dict(state))
    return env, model, plan


def _save(tmp_path, policy, name, budget="mixed", **meta):
    path = tmp_path / "plan_agent" / f"policy_{name}.pt"
    plan_agent.save_policy(policy, str(path), tokens="layer", zero=["sens"], k_min=0.1, budget=budget, **meta)
    return str(path)


# ------------------------------------------------------------------ the flags


def test_menu_flags_default_off_parse_and_errors(monkeypatch):
    assert plan_agent.menu_kinds() is None and plan_agent.menu_kinds(default=()) == ()
    assert plan_agent.menu_init() == 6.0 == plan_agent.MENU_INIT
    assert plan_agent.menu_sigma() == 0.5 == plan_agent.MENU_SIGMA
    assert plan_trainer.menu_settings("sens", ("flops",)) is None and plan_trainer.menu_settings() is None
    assert plan_agent.MENU_KINDS == MENU
    for raw in ("", "   "):
        monkeypatch.setenv("SPECTRA_PLAN_MENU", raw)
        assert plan_agent.menu_kinds() is None
    monkeypatch.setenv("SPECTRA_PLAN_MENU", " Sens, UNIFORM ,sens_cost,sens_thresh ")
    assert plan_agent.menu_kinds() == MENU
    monkeypatch.setenv("SPECTRA_PLAN_MENU", "sens_thresh,sens")
    assert plan_agent.menu_kinds() == ("sens_thresh", "sens")                      # the order is the menu's
    for bad in ("sens", "sens,inner", "sens,sens", "sens,uniform,sens", "sens,agent", "sens,sens_curve", "sens,lamp",
                "sens,widths", ","):
        monkeypatch.setenv("SPECTRA_PLAN_MENU", bad)
        with pytest.raises(ValueError, match="SPECTRA_PLAN_MENU"):
            plan_agent.menu_kinds()
    monkeypatch.setenv("SPECTRA_PLAN_MENU_INIT", " -2 ")
    monkeypatch.setenv("SPECTRA_PLAN_MENU_SIGMA", "0.25")
    assert plan_agent.menu_init() == -2.0 and plan_agent.menu_sigma() == 0.25
    for bad in ("six", "nan", "inf", "-inf"):
        monkeypatch.setenv("SPECTRA_PLAN_MENU_INIT", bad)
        with pytest.raises(ValueError, match="SPECTRA_PLAN_MENU_INIT"):
            plan_agent.menu_init()
    for bad in ("0", "-0.5", "nan", "inf", "wide"):
        monkeypatch.setenv("SPECTRA_PLAN_MENU_SIGMA", bad)
        with pytest.raises(ValueError, match="SPECTRA_PLAN_MENU_SIGMA"):
            plan_agent.menu_sigma()


# ------------------------------------------------------------------ the rules' weights


@pytest.mark.parametrize("budget", ["params", "flops"])
def test_menu_rules_are_the_weights_each_alloc_row_hands_its_landing(budget):
    net = _net()
    pm = net.cm[budget]
    rows = list(pm.rows)
    rules = _rules(net, budget, 0.5)
    assert len(rules) == 4 and all(list(w) == rows for w in rules)
    cost = {row: net.costs[row][1 if budget == "flops" else 0] for row in rows}
    assert rules[0] == alloc_walk.weights("sens", net.sens, 0.5) == plan_agent.prior_weights("sens", rows, net.sens)
    assert rules[1] == {row: 1.0 for row in rows} == alloc_walk.weights("uniform", net.sens)
    assert rules[2] == alloc_walk.weights("sens_cost", net.sens, 0.5, cost=cost)
    assert rules[2] == plan_agent.prior_weights("sens_cost", rows, net.sens, net.costs, budget)
    # sens_thresh: what plan_targets hands _land at the same target, floor and budget, measured on the same batches
    for kappa, floor in ((0.5, 2), (0.3, 1)):
        _w, info = alloc_walk.plan_targets(net.model, net.batches, SHAPE, "sens_thresh", kappa, budget=budget,
                                           min_width=floor)
        assert _rules(net, budget, kappa, floor)[3] == info["weights"]
        curve = alloc_walk.probe_curves(rows, net.curves["qs"], net.curves["loss"], net.curves["raw"], budget)
        assert info["curve"] == {row: [list(point) for point in curve[row]] for row in rows}
        assert info["curve_keeps"] == list(net.curves["qs"]) == list(alloc_walk.curve_keeps())
    # the anchor's power is the residual's alpha, the rules follow the list's order, and a missing measurement is named
    other = plan_agent.menu_rules(("uniform", "sens"), pm, 0.5, net.sens, None, None, budget, 0.7, 0.1, 2)
    assert other == [rules[1], alloc_walk.weights("sens", net.sens, 0.7)] and other[1] != rules[0]
    with pytest.raises(RuntimeError, match="sens_cost needs the sensitivities and costs"):
        plan_agent.menu_rules(MENU, pm, 0.5, net.sens, None, net.curves, budget)
    with pytest.raises(RuntimeError, match="sens_thresh needs the origin's damage curves"):
        plan_agent.menu_rules(MENU, pm, 0.5, net.sens, net.costs, None, budget)
    with pytest.raises(RuntimeError, match="toy: the menu rule sens needs the sensitivities"):
        plan_agent.menu_rules(MENU, pm, 0.5, None, net.costs, net.curves, budget, name="toy")
    with pytest.raises(ValueError, match="not a menu rule"):
        plan_agent.menu_rules(("sens", "inner"), pm, 0.5, net.sens)


@pytest.mark.parametrize("budget", ["params", "flops"])
def test_sens_thresh_weights_follow_the_target_and_keep_the_floors(budget):
    net = _net()
    pm = net.cm[budget]
    curves = _hand_curves(net)
    w = {kappa: _rules(net, budget, kappa, 2, ("sens", "sens_thresh"), curves)[1] for kappa in (0.3, 0.5, 0.8)}
    assert w[0.3] != w[0.5] and w[0.5] != w[0.8]
    assert all(w[0.3][row] <= w[0.5][row] <= w[0.8][row] for row in pm.rows)     # a deeper target, a deeper pick
    floor = {row: max(0.1, min(1.0, 2 / pm.widths0[row])) for row in pm.rows}
    assert all(floor[row] <= w[kappa][row] <= 1.0 for kappa in w for row in pm.rows)
    assert any(w[0.3][row] == floor[row] > 0.1 for row in pm.rows)               # the width floor binds on a thin group
    curve = alloc_walk.probe_curves(pm.rows, curves["qs"], curves["loss"], curves["raw"], budget)
    for kappa in w:
        knob, tau, est, weights = alloc_walk.curve_weights("sens_thresh", curve, floor, pm.total0, kappa, pm.rows)
        assert knob == "tau" and tau > 0 and weights == w[kappa] and abs(est - kappa) < 0.1
        assert (tau, est) == alloc_walk._curve_threshold(curve, floor, pm.total0, kappa)[:2]
    lam = alloc_walk.curve_weights("sens_curve", curve, floor, pm.total0, 0.5)     # the other knob, as plan_targets
    assert lam[0] == "lambda" and lam[1:3] == alloc_walk._curve_lambda(curve, floor, pm.total0, 0.5)[:2]
    # an instance hands the same weights: its curves, its cost model, the decoders' floors
    inst = types.SimpleNamespace(pm=pm, sens=net.sens, cost=net.costs, curves=curves, name="x",
                                 cost_model=lambda b: net.cm[b])
    assert plan_agent.NetInstance.rules_at(inst, ("sens", "sens_thresh"), 0.5, budget, 0.5, 0.1, 2)[1] == w[0.5]


# ------------------------------------------------------------------ the shares


def test_menu_shares_are_the_softmax_and_the_mean_plans_floor_drops_small_shares():
    u = [0.3, -1.2, 2.0, 0.0]
    pi = plan_agent.menu_shares(u)
    assert pi == pytest.approx(torch.softmax(torch.tensor(u, dtype=torch.float64), dim=0).tolist(), rel=1e-12)
    assert sum(pi) == pytest.approx(1.0) and plan_agent.menu_shares([v + 40.0 for v in u]) == pytest.approx(pi)
    assert plan_agent.menu_shares([1000.0, 0.0]) == [1.0, 0.0] and plan_agent.menu_shares([-1000.0, 0.0]) == [0.0, 1.0]
    assert plan_agent.menu_shares([5.0, 5.0]) == [0.5, 0.5]
    # the default init: 0.74 % off the anchor on a menu of four, a pure share once the mean plan's floor drops it
    init = plan_agent.menu_shares([6.0, 0.0, 0.0, 0.0])
    assert init[0] == pytest.approx(0.99262, abs=1e-5)
    assert init[1] == init[2] == init[3] == pytest.approx(0.00246, abs=1e-5)
    assert plan_agent.MENU_SHARE_FLOOR == 0.005
    for k in (2, 3, 4, 6, 10):
        assert plan_agent.menu_shares([6.0] + [0.0] * (k - 1), plan_agent.MENU_SHARE_FLOOR) == [1.0] + [0.0] * (k - 1)
    # a larger share stays, a smaller one goes and the rest keep their proportions
    full = plan_agent.menu_shares([2.0, 1.0, -4.0, -9.0])
    assert full[2] < 0.005 and full[3] < 0.005 < full[1]
    kept = plan_agent.menu_shares([2.0, 1.0, -4.0, -9.0], 0.005)
    assert kept[2:] == [0.0, 0.0] and sum(kept) == pytest.approx(1.0)
    assert kept[0] / kept[1] == pytest.approx(full[0] / full[1])
    assert plan_agent.menu_shares([2.0, 1.0, 0.5, 0.0], 0.005) == plan_agent.menu_shares([2.0, 1.0, 0.5, 0.0])
    assert plan_agent.menu_shares([4.0, 0.0], 0.0) == plan_agent.menu_shares([4.0, 0.0])        # 0 = no floor


# ------------------------------------------------------------------ (b) a pure share is the rule's own plan


@pytest.mark.parametrize("budget", ["params", "flops"])
@pytest.mark.parametrize("min_width", [1, 2])
def test_a_pure_share_at_zero_scores_is_that_rules_scale_decode_plan_exactly(budget, min_width):
    net = _net()
    pm = net.cm[budget]
    zero = [0.0] * len(pm.rows)
    for kappa, curves in ((0.4, None), (0.6, None), (0.5, _hand_curves(net))):
        rules = _rules(net, budget, kappa, min_width, curves=curves)
        for k in range(len(MENU)):
            pi = [1.0 if j == k else 0.0 for j in range(len(MENU))]
            want, want_info = plan_agent.scale_decode(rules[k], pm, kappa, 0.1, min_width=min_width)
            for zmax in (0.0, 0.3):
                widths, info = plan_agent.menu_decode(zero, pi, rules, pm, kappa, 0.1, min_width=min_width, zmax=zmax)
                assert widths == want and {key: info[key] for key in want_info} == want_info
                assert info["base"] == rules[k] == info["w"] and info["pi"] == pi
                assert _floor_ok(widths, pm.widths0, min_width) and info["kept"] == pytest.approx(pm.kept(widths))
    assert plan_agent.menu_mix([0.0, 1.0, 0.0, 0.0], rules, pm.rows) == rules[1]
    assert plan_agent.menu_mix([0, 0, 1, 0], rules, pm.rows) == rules[2]            # integer shares are shares too


@pytest.mark.parametrize("budget", ["params", "flops"])
def test_on_the_anchor_alone_the_menu_decode_is_the_residual_agents_decode_bit_for_bit(budget):
    net = _net()
    pm = net.cm[budget]
    rules = _rules(net, budget, 0.6)
    g = torch.Generator().manual_seed(2)
    for scale in (0.25, 3.0, 1e4):
        z = (scale * torch.randn(len(pm.rows), generator=g)).tolist()
        for zmax in (0.0, 0.3):
            widths, info = plan_agent.menu_decode(z, [1.0, 0.0, 0.0, 0.0], rules, pm, 0.6, 0.1, min_width=2, zmax=zmax)
            want, want_info = plan_agent.residual_decode(z, rules[0], pm, 0.6, 0.1, min_width=2, zmax=zmax)
            assert widths == want and {key: info[key] for key in want_info} == want_info
            assert set(info) == set(want_info) | {"pi", "base"}


# ------------------------------------------------------------------ the blend


def test_menu_decode_blends_the_normalised_log_weights_under_the_scores_and_the_trust_region():
    net = _net()
    pm = net.cm["flops"]
    rows = list(pm.rows)
    rules = _rules(net, "flops", 0.6, curves=_hand_curves(net))
    z = (0.4 * torch.randn(len(rows), generator=torch.Generator().manual_seed(5))).tolist()
    pi = plan_agent.menu_shares([0.4, -0.3, 0.1, 0.8])
    assert min(pi) > 0.1
    logs = []
    for w in rules:                                                          # log of each rule over its geometric mean
        raw = [math.log(w[row]) for row in rows]
        logs.append([v - sum(raw) / len(raw) for v in raw])
    assert all(abs(sum(lw)) < 1e-9 for lw in logs)
    base = plan_agent.scale_decode(rules[0], pm, 0.6, 0.1, min_width=2)[0]
    for zmax in (0.0, 0.3):
        widths, info = plan_agent.menu_decode(z, pi, rules, pm, 0.6, 0.1, min_width=2, zmax=zmax)
        for n, row in enumerate(rows):
            mix = sum(pi[k] * logs[k][n] for k in range(4))
            eff = plan_agent.trust_score(z[n], zmax) if zmax > 0 else z[n]
            assert info["base"][row] == pytest.approx(math.exp(mix), rel=1e-12)
            assert info["w"][row] == pytest.approx(math.exp(mix + eff), rel=1e-12)
            if zmax > 0:
                assert math.exp(-zmax) - 1e-12 <= info["w"][row] / info["base"][row] <= math.exp(zmax) + 1e-12
        assert widths == plan_agent.scale_decode(info["w"], pm, 0.6, 0.1, min_width=2)[0]
        assert widths == plan_agent.residual_decode(z, info["base"], pm, 0.6, 0.1, min_width=2, zmax=zmax)[0]
        assert info["pi"] == pi and info["z"] == {row: float(v) for row, v in zip(rows, z)}
        assert abs(info["kept"] - 0.6) < TOL["flops"] and info["kept"] == pytest.approx(pm.kept(widths))
        assert _floor_ok(widths, pm.widths0, 2) and widths != base
    # a rule's own scale never reaches the blend: its geometric mean is divided out
    scaled = [rules[0], {r: 7.3 * v for r, v in rules[1].items()}, rules[2], {r: 0.01 * v for r, v in rules[3].items()}]
    mixed = plan_agent.menu_mix(pi, rules, rows)
    assert plan_agent.menu_mix(pi, scaled, rows) == pytest.approx(mixed, rel=1e-12)
    # a share of zero drops its rule, and the shares must match the rules
    half = plan_agent.menu_mix([0.5, 0.5, 0.0, 0.0], rules, rows)
    assert half == pytest.approx({row: math.exp(0.5 * logs[0][n] + 0.5 * logs[1][n]) for n, row in enumerate(rows)})
    assert half == plan_agent.menu_mix([0.5, 0.5], rules[:2], rows)
    with pytest.raises(ValueError, match="2 shares for 4 rules"):
        plan_agent.menu_mix([0.5, 0.5], rules, rows)


# ------------------------------------------------------------------ the policy's menu head


def test_policy_without_a_menu_is_built_as_before_and_the_menu_head_draws_nothing(v10_env):
    from src.Model.StateEncoder import build_state_encoder
    torch.manual_seed(3)
    encoder = build_state_encoder("transformer", 63)
    nn.Linear(encoder.output_dim, 1)                                             # the score head's constructor draw
    after = torch.get_rng_state()
    torch.manual_seed(3)
    plain = plan_agent.PlanPolicy(63)
    assert torch.equal(torch.get_rng_state(), after)                          # the encoder's and the head's, no more
    assert plain.menu == 0 and not hasattr(plain, "menu_head")
    plain_state = plain.state_dict()
    assert set(plain_state) == {f"encoder.{key}" for key in encoder.state_dict()} | {"head.weight", "head.bias"}
    assert all(torch.equal(plain_state[f"encoder.{key}"], value) for key, value in encoder.state_dict().items())
    torch.manual_seed(3)
    menu = plan_agent.PlanPolicy(63, menu=4)
    assert torch.equal(torch.get_rng_state(), after)                              # the menu head's draw is put back
    menu_state = menu.state_dict()
    assert set(menu_state) == set(plain_state) | {"menu_head.weight", "menu_head.bias"}
    assert all(torch.equal(menu_state[key], plain_state[key]) for key in plain_state)
    d = menu.encoder.output_dim
    assert menu.menu == 4 and menu.menu_head.weight.shape == (4, d + 4) and not bool(menu.menu_head.weight.any())
    assert menu.menu_head.bias.tolist() == [6.0, 0.0, 0.0, 0.0]
    assert plan_agent.PlanPolicy(63, menu=3, menu_init=2.5).menu_head.bias.tolist() == [2.5, 0.0, 0.0]
    # the scores never see the menu head: forward is the module's before, to the bit
    state = toy_state(8)
    mask, k = torch.tensor([True] * 7 + [False]), torch.tensor([0, 0, 1, 1, 2, 2, 3])
    with torch.no_grad():
        plain.head.weight.normal_()
        menu.head.load_state_dict(plain.head.state_dict())
        menu.menu_head.weight.normal_()
        mu = plain(state, mask, k, 4)
        assert float(mu.abs().sum()) > 0 and torch.equal(menu(state, mask, k, 4), mu)
        # psi = the menu head on [the mean of the encoder's outputs over the net's tokens ; x_net]
        x = [0.2, -0.1, 0.05, 0.6]
        encoded = menu.encoder(state)
        assert encoded.shape == (8, d)                                            # one output per layer token
        want = menu.menu_head(torch.cat([encoded.mean(dim=0), torch.tensor(x)]))
        psi = menu.menu_logits(state, x)
        assert psi.shape == (4,) and torch.equal(psi, want)
        assert not torch.equal(menu.menu_logits(state, [0.2, -0.1, 0.05, 0.4]), psi)   # it reads x_net
        menu.menu_head.weight.zero_()
        assert menu.menu_logits(state, x).tolist() == [6.0, 0.0, 0.0, 0.0]   # at the init: the bias, whatever it reads
    with pytest.raises(ValueError, match="3 network-level inputs for a menu of 4 rules"):
        menu.menu_logits(state, [0.1, 0.2, 0.6])


def test_menu_features_are_the_rules_plan_costs_and_kappa():
    net = _net()
    for budget in ("flops", "params"):
        pm = net.cm[budget]
        rules = _rules(net, budget, 0.45)
        x, plans = plan_agent.menu_features(rules, pm, 0.45, 0.1, 2)
        assert len(x) == len(MENU) == len(plans) and x[-1] == 0.45
        for k in range(len(MENU)):
            assert plans[k] == plan_agent.scale_decode(rules[k], pm, 0.45, 0.1, min_width=2)
        params = [pm.params(widths) for widths, _info in plans]
        assert x[:-1] == [math.log(p / params[0]) for p in params[1:]]
        assert plan_agent.menu_features(rules, pm, 0.45, 0.1, 2, kappa=0.47)[0] == x[:-1] + [0.47]
        if budget == "flops":                                                 # at equal FLOPs the rules' params differ
            assert max(abs(v) for v in x[:-1]) > 0.01
        else:                                                                     # at equal params they land together
            assert max(abs(v) for v in x[:-1]) < 2 * TOL["params"]


# ------------------------------------------------------------------ (c) the init


@pytest.mark.parametrize("budget", ["params", "flops"])
def test_the_zero_init_agents_mean_plan_is_the_anchors_plan_exactly(v10_env, budget):
    inst = _instance()
    torch.manual_seed(0)
    policy, plain = plan_agent.PlanPolicy(63, menu=len(MENU)), plan_agent.PlanPolicy(63)
    cfg = plan_trainer.config()
    for kappa in (0.35, 0.5, 0.6, 0.85):
        for floor in (1, 2):
            prior = plan_trainer.prior_of(inst, RES, budget)
            at = plan_trainer.menu_at(inst, SETTINGS, RES, kappa, cfg, budget, floor)
            assert at["rules"][0] == prior and len(at["x"]) == len(MENU) and at["x"][-1] == kappa
            widths, info = plan_trainer.mean_plan(policy, inst, kappa, cfg, budget, floor, prior, 0.3, at)
            want, want_info = plan_agent.scale_decode(prior, inst.cost_model(budget), kappa, cfg.k_min, min_width=floor)
            assert widths == want and info["kept"] == want_info["kept"] and info["keeps"] == want_info["keeps"]
            assert info["pi"] == [1.0, 0.0, 0.0, 0.0] and info["base"] == prior
            # ... which is the mean plan of the residual agent this menu starts from
            w_plain, i_plain = plan_trainer.mean_plan(plain, inst, kappa, cfg, budget, floor, prior, 0.3)
            assert (widths, info["kept"]) == (w_plain, i_plain["kept"]) and "pi" not in i_plain
    assert plan_trainer.menu_at(inst, None, RES, 0.6, cfg, budget, 2) is None


def test_without_the_floor_the_default_inits_small_shares_move_a_width(v10_env):
    """Why the mean plan drops the shares under ``MENU_SHARE_FLOOR``: softmax(6, 0, 0, 0) leaves 0.74 % off the anchor,
    and decoding that blend is not the anchor's plan at every target."""
    net = _net()
    pm = net.cm["flops"]
    zero, moved = [0.0] * len(pm.rows), 0
    for step in range(11):
        kappa = 0.35 + 0.05 * step
        rules = _rules(net, "flops", kappa)
        anchor = plan_agent.scale_decode(rules[0], pm, kappa, 0.1, min_width=2)[0]
        soft = plan_agent.menu_decode(zero, plan_agent.menu_shares([6.0, 0.0, 0.0, 0.0]), rules, pm, kappa, 0.1,
                                      min_width=2, zmax=0.3)[0]
        dropped = plan_agent.menu_shares([6.0, 0.0, 0.0, 0.0], plan_agent.MENU_SHARE_FLOOR)
        floored = plan_agent.menu_decode(zero, dropped, rules, pm, kappa, 0.1, min_width=2, zmax=0.3)[0]
        assert floored == anchor
        moved += soft != anchor
    assert moved > 0


# ------------------------------------------------------------------ (d) the log-prob


def test_the_menu_log_prob_is_the_two_gaussian_terms_summed_by_hand():
    g = torch.Generator().manual_seed(11)
    mu, psi = torch.randn(7, generator=g), torch.randn(4, generator=g)
    z = mu + 0.25 * torch.randn(5, 7, generator=g)
    u = psi + 0.5 * torch.randn(5, 4, generator=g)
    got = plan_agent.menu_log_prob(z, mu, 0.25, u, psi, 0.5)
    assert got.shape == (5,)
    for k in range(5):
        want = (-sum((float(z[k, j]) - float(mu[j])) ** 2 for j in range(7)) / (2 * 0.25 ** 2)
                - sum((float(u[k, j]) - float(psi[j])) ** 2 for j in range(4)) / (2 * 0.5 ** 2))
        assert float(got[k]) == pytest.approx(want, rel=1e-5)
    assert torch.equal(got, plan_agent.log_prob(z, mu, 0.25) + plan_agent.log_prob(u, psi, 0.5))
    # torch's own Normal agrees, up to the constants both terms drop
    const = 7 * math.log(0.25 * math.sqrt(2 * math.pi)) + 4 * math.log(0.5 * math.sqrt(2 * math.pi))
    full = (torch.distributions.Normal(mu, 0.25).log_prob(z).sum(-1)
            + torch.distributions.Normal(psi, 0.5).log_prob(u).sum(-1))
    torch.testing.assert_close(got, full + const, rtol=1e-5, atol=1e-4)


# ------------------------------------------------------------------ (e) the gradient


def test_reinforce_gradient_reaches_the_menu_heads_weights_and_bias_with_the_right_sign(v10_env):
    inst = _instance()
    policy = plan_agent.PlanPolicy(63, menu=len(MENU))
    state, x = inst.state_at(0.6, "flops"), [0.2, -0.1, 0.05, 0.6]
    mu = policy(state, inst.token_mask, inst.token_k, inst.n_groups)
    psi = policy.menu_logits(state, x)
    assert psi.requires_grad and psi.tolist() == [6.0, 0.0, 0.0, 0.0]
    g = torch.Generator().manual_seed(0)
    z = plan_agent.sample_scores(mu, 0.25, 64, g)
    u = plan_agent.sample_scores(psi, 0.5, 64, g)
    assert u.shape == (64, 4) and not u.requires_grad
    reward = u[:, 1] - u[:, 0]                       # plans that lean from the anchor to the second rule score higher
    adv = (reward - reward.mean()) / (reward.std() + 1e-6)
    loss = -(adv * plan_agent.menu_log_prob(z, mu, 0.25, u, psi, 0.5)).mean()
    loss.backward()
    grad_b, grad_w, d = policy.menu_head.bias.grad, policy.menu_head.weight.grad, policy.encoder.output_dim
    assert float(grad_b[1]) < 0 < float(grad_b[0])   # a descent step lifts the second rule's logit, lowers the anchor's
    assert float(grad_w[:, :d].abs().sum()) > 0 and float(grad_w[:, d:].abs().sum()) > 0   # through tokens and x_net
    torch.testing.assert_close(grad_w[:, d:], grad_b.unsqueeze(1) * torch.tensor(x).unsqueeze(0), rtol=1e-4, atol=1e-6)
    assert float(policy.head.weight.grad.abs().sum()) > 0                                # the scores' term, as before
    # once the head reads the tokens, the menu's term reaches the encoder too
    policy.zero_grad()
    with torch.no_grad():
        policy.menu_head.weight.normal_(std=0.1)
    psi = policy.menu_logits(state, x)
    (-(adv * plan_agent.log_prob(u, psi, 0.5)).mean()).backward()
    assert policy.head.weight.grad is None or not bool(policy.head.weight.grad.any())
    assert any(p.grad is not None and bool(p.grad.any()) for p in policy.encoder.parameters())


# ------------------------------------------------------------------ (f) the landing


@pytest.mark.parametrize("budget", ["flops", "params"])
def test_random_shares_and_scores_land_the_budget_within_the_decoders_tolerance(budget):
    net = _net()
    pm = net.cm[budget]
    g = torch.Generator().manual_seed(7)
    plans = set()
    for trial in range(16):
        kappa = (0.35, 0.5, 0.65, 0.85)[trial % 4]
        rules = _rules(net, budget, kappa, curves=_hand_curves(net) if trial % 3 == 0 else None)
        centre, spread = ([6.0, 0.0, 0.0, 0.0], 0.5) if trial % 2 else ([0.0] * 4, 3.0)   # the trainer's, and far wider
        pi = plan_agent.menu_shares([c + spread * float(e) for c, e in zip(centre, torch.randn(4, generator=g))])
        z = (0.25 * torch.randn(len(pm.rows), generator=g)).tolist()
        widths, info = plan_agent.menu_decode(z, pi, rules, pm, kappa, 0.1, min_width=2, zmax=0.3)
        assert abs(info["kept"] - kappa) < TOL[budget], (trial, kappa, info["kept"], pi)
        assert info["kept"] == pytest.approx(pm.kept(widths)) and set(widths) == set(pm.rows)
        assert _floor_ok(widths, pm.widths0, 2) and min(info["keeps"].values()) >= 0.1 - 1e-12
        plans.add(tuple(widths[row] for row in pm.rows))
    assert len(plans) > 8
    real = alloc_walk.cut_to(net.model, net.plan, plan_agent.rates_of(widths, pm.widths0), SHAPE)   # the last plan, cut
    assert alloc_walk.group_widths(real, pm.rows) == widths
    size = utils.calc_flops(real, SHAPE) if budget == "flops" else utils.calc_num_parameters(real)
    assert size / pm.total0 == pytest.approx(info["kept"], rel=1e-9)


# ------------------------------------------------------------------ (a) the flag off


def test_flag_off_the_residual_trainer_draws_decodes_and_scores_as_before(monkeypatch, tmp_path, v10_env):
    """T1's FLOPs line on the toy net with the menu flag unset: nothing of the menu is called or written, and every
    step is the loop tree_v35 runs, rebuilt here from its own parts: the generator's only draw is the scores', each plan
    is ``residual_decode`` of its row, and the loss is the scores' log-density alone."""
    inst, again = _instance(), _instance()                                         # measured before the traps are set
    for name in ("menu_decode", "menu_shares", "menu_mix", "menu_rules", "menu_features", "menu_log_prob"):
        monkeypatch.setattr(plan_agent, name, lambda *a, _n=name, **k: pytest.fail(f"flag off must not call {_n}"))
    monkeypatch.setattr(plan_agent.PlanPolicy, "menu_logits", lambda *a, **k: pytest.fail("flag off: no menu head"))
    monkeypatch.setattr(alloc_walk, "curve_probes", lambda *a, **k: pytest.fail("flag off: no curves"))
    draws = _spy_draws(monkeypatch)
    inst.reward = _reward(inst)
    calls = []
    lines, records, blob = _train(monkeypatch, tmp_path, inst, calls=calls, **T1)
    assert calls == [{"tokens": "layer", "zero": (), "with_cost": False}]
    assert lines[0].endswith(T1_TOKENS) and not any(_menu_text(line) for line in lines)
    assert not any(key.startswith("menu") for key in blob) and "menu_head.weight" not in blob["state_dict"]
    assert not any("menu_pi" in r for r in records)
    steps = [r for r in records if "it" in r]
    cfg = plan_trainer.config()
    prior, fm = inst.prior("sens", "flops", 0.5), inst.cost_model("flops")
    assert len(steps) == 5 == len(draws)                                           # one draw per instance
    assert torch.equal(draws[0][0], torch.zeros(inst.n_groups))
    moving = 0
    for it, (r, (mu, sigma, z)) in enumerate(zip(steps, draws)):
        assert sigma == r["sigma"] == pytest.approx(0.5 * plan_trainer.sigma_at(cfg, it))
        eps = torch.randn((cfg.k, inst.n_groups), generator=torch.Generator().manual_seed(cfg.seed * 100003 + it))
        assert torch.equal(z, mu.unsqueeze(0) + sigma * eps)                       # the generator's first draw
        kept = [plan_agent.residual_decode(z[k].tolist(), prior, fm, r["kappa"], cfg.k_min, min_width=2, zmax=0.3)
                for k in range(cfg.k)]
        assert r["kept"] == [info["kept"] for _w, info in kept]
        assert r["rewards"] == [inst.reward(widths, "cut", []) for widths, _i in kept]
        rewards = torch.tensor(r["rewards"], dtype=mu.dtype)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        assert r["loss"] == pytest.approx(float(-(adv * plan_agent.log_prob(z, mu, sigma)).mean()), rel=1e-4, abs=1e-5)
        moving += abs(r["loss"]) > 1e-3
    assert moving >= 3                                                              # the check is not of zeros
    assert steps[0]["mean_plan"] == steps[0]["refs"]["prior"]
    # an explicit empty flag is the same run
    again.reward = _reward(again)
    lines2, records2, blob2 = _train(monkeypatch, tmp_path / "empty", again, SPECTRA_PLAN_MENU=" ", **T1)
    assert lines2[0] == lines[0] and _steps(records2) == _steps(records) and set(blob2) == set(blob)
    assert _equal(blob2["state_dict"], blob["state_dict"])


# ------------------------------------------------------------------ the trainer with the menu


def test_trainer_with_the_menu_draws_the_shares_after_the_scores_and_starts_on_the_prior(monkeypatch, tmp_path,
                                                                                         v10_env):
    draws, inits = _spy_draws(monkeypatch), []

    class Spy(plan_agent.PlanPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            inits.append({key: value.clone() for key, value in self.state_dict().items()})

    monkeypatch.setattr(plan_agent, "PlanPolicy", Spy)
    plain = _instance()
    plain.reward = _reward(plain)
    _seed_all()
    _off_lines, off_records, _off_blob = _train(monkeypatch, tmp_path / "off", plain, **T1)
    off, rng_off = list(draws), alloc_walk.rng_snapshot()
    del draws[:]
    inst = _instance()
    inst.reward = _reward(inst)
    calls = []
    _seed_all()
    lines, records, blob = _train(monkeypatch, tmp_path / "on", inst, calls=calls, SPECTRA_PLAN_MENU=MENU_FLAG, **T1)
    # the same seed without the menu is the same job up to the menu: the policy starts from the same weights, every
    # generator ends where it ended, and the instances, their targets and their sigmas are the same
    assert _equal(alloc_walk.rng_snapshot(), rng_off)
    assert set(inits[1]) == set(inits[0]) | {"menu_head.weight", "menu_head.bias"}
    assert all(torch.equal(inits[1][key], inits[0][key]) for key in inits[0])
    off_steps = [r for r in off_records if "it" in r]
    assert [(r["kappa"], r["sigma"]) for r in records if "it" in r] == [(r["kappa"], r["sigma"]) for r in off_steps]
    assert calls == [{"tokens": "layer", "zero": (), "with_cost": True, "with_curves": True}]
    assert lines[0].endswith(T1_TOKENS + " menu=sens,uniform,sens_cost,sens_thresh menu_init=6 menu_sigma=0.5")
    assert not any("WARNING" in line for line in lines if line.startswith("[plan] menu"))
    setup = [line for line in lines if line.startswith("[plan] menu setup ")]
    assert len(setup) == 1 and re.fullmatch(
        r"\[plan\] menu setup toy\.pt: sens,uniform,sens_cost,sens_thresh \| k=0\.60 b=flops rule plans' params kept "
        r"x\d\.\d{3} x\d\.\d{3} x\d\.\d{3} x\d\.\d{3} \| \d+\.\ds", setup[0]), setup
    assert float(setup[0].rsplit(" | ", 1)[1][:-1]) >= 1.5                          # the curves' seconds are counted
    steps = [r for r in records if "it" in r]
    cfg = plan_trainer.config()
    fm = inst.cost_model("flops")
    assert len(steps) == 5 and len(draws) == 10                             # per instance: the scores, then the logits
    assert torch.equal(draws[0][2], off[0][2])                              # the scores' noise is the run's without it
    assert draws[1][0].tolist() == [6.0, 0.0, 0.0, 0.0]                             # psi at the init
    differs = 0
    for it, r in enumerate(steps):
        (mu, sigma, z), (psi, sigma_u, u) = draws[2 * it], draws[2 * it + 1]
        assert sigma == r["sigma"] and sigma_u == 0.5 and u.shape == (cfg.k, len(MENU))
        g = torch.Generator().manual_seed(cfg.seed * 100003 + it)
        assert torch.equal(z, mu.unsqueeze(0) + sigma * torch.randn((cfg.k, inst.n_groups), generator=g))
        assert torch.equal(u, psi.unsqueeze(0) + 0.5 * torch.randn((cfg.k, len(MENU)), generator=g))
        rules = inst.rules_at(MENU, r["kappa"], "flops", 0.5, cfg.k_min, 2)
        shares = [plan_agent.menu_shares(row) for row in u.tolist()]
        plans = [plan_agent.menu_decode(z[k].tolist(), shares[k], rules, fm, r["kappa"], cfg.k_min, min_width=2,
                                        zmax=0.3) for k in range(cfg.k)]
        assert r["kept"] == [info["kept"] for _w, info in plans]
        assert all(abs(kept - r["kappa"]) < TOL["flops"] for kept in r["kept"])
        assert r["menu_pi"] == [sum(s[j] for s in shares) / cfg.k for j in range(len(MENU))]
        assert sum(r["menu_pi"]) == pytest.approx(1.0) and r["menu_pi"][0] > 0.9
        rewards = torch.tensor(r["rewards"], dtype=mu.dtype)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        both = float(-(adv * plan_agent.menu_log_prob(z, mu, sigma, u, psi, 0.5)).mean())
        assert r["loss"] == pytest.approx(both, rel=1e-4, abs=1e-5)
        differs += abs(both - float(-(adv * plan_agent.log_prob(z, mu, sigma)).mean())) > 1e-3
        assert r["residual"] == "sens" and r["prior"] == "sens"
    assert differs >= 3                                                              # the logits' term is in the loss
    first = steps[0]                                                            # the zero-init agent: the prior's plan
    assert set(first["refs"]) == {"uniform", "sens", "inner", "prior"} and first["mean_plan"] == first["refs"]["prior"]
    summaries = [r for r in records if "summary" in r]
    menu_lines = [line for line in lines if line.startswith("[plan] menu summary ")]
    assert len(summaries) == 3 == len(menu_lines) == len([line for line in lines if line.startswith("[plan] summary ")])
    for s, line in zip(summaries, menu_lines):
        assert len(s["menu_pi"]) == 4 and sum(s["menu_pi"]) == pytest.approx(1.0)
        assert line == (f"[plan] menu summary toy.pt k={s['kappa']:.2f} b=flops: pi "
                        + " ".join(f"{kind} {p:.3f}" for kind, p in zip(MENU, s["menu_pi"])))
        assert lines.index(line) == lines.index(next(x for x in lines if x.startswith(
            f"[plan] summary toy.pt k={s['kappa']:.2f} b=flops: "))) + 1
    assert all(s["menu_pi"] == [1.0, 0.0, 0.0, 0.0] for s in summaries)             # five small steps from B = 6
    assert {key: blob[key] for key in MENU_META} == MENU_META and blob["residual"] == "sens"
    assert not any(key.startswith("menu") for key in blob["config"])
    assert blob["state_dict"]["menu_head.weight"].shape == (4, 256 + 4)
    assert float(blob["state_dict"]["menu_head.weight"].abs().sum()) > 0             # the head has moved off zero
    monkeypatch.setenv("SPECTRA_PLAN_MENU", MENU_FLAG)
    policy, loaded = plan_agent.load_policy(str(tmp_path / "on" / "policy_latest.pt"))
    assert policy.menu == 4 and not policy.training and loaded["menu"] == list(MENU)


def test_trainer_menu_reads_only_what_its_rules_need_and_warns_when_the_init_is_off_the_anchor(monkeypatch, tmp_path,
                                                                                               v10_env):
    inst = _instance()
    inst.reward = _reward(inst)
    inst.cost = inst.curves = None                                                   # sens,uniform needs neither
    calls = []
    lines, records, blob = _train(monkeypatch, tmp_path / "two", inst, instances=3, calls=calls,
                                  SPECTRA_PLAN_MENU="sens,uniform", SPECTRA_PLAN_MENU_INIT="0",
                                  SPECTRA_PLAN_MENU_SIGMA="0.3", **T1)
    assert calls == [{"tokens": "layer", "zero": (), "with_cost": False}]
    assert lines[0].endswith(T1_TOKENS + " menu=sens,uniform menu_init=0 menu_sigma=0.3")
    warn = [line for line in lines if line.startswith("[plan] menu WARNING")]
    assert len(warn) == 1 and "SPECTRA_PLAN_MENU_INIT=0 starts the anchor at a share of 0.5000" in warn[0]
    assert (blob["menu"], blob["menu_anchor"]) == (["sens", "uniform"], "sens")
    assert (blob["menu_init"], blob["menu_sigma"]) == (0.0, 0.3)
    assert blob["state_dict"]["menu_head.weight"].shape == (2, 256 + 2)
    steps = [r for r in records if "it" in r]
    assert all(len(r["menu_pi"]) == 2 and 0.2 < r["menu_pi"][0] < 0.8 for r in steps)   # an even start, both rules live
    summaries = [r for r in records if "summary" in r]
    assert all(0.3 < s["menu_pi"][0] < 0.7 for s in summaries)
    setup = next(line for line in lines if line.startswith("[plan] menu setup "))
    assert re.fullmatch(r"\[plan\] menu setup toy\.pt: sens,uniform \| k=0\.60 b=flops rule plans' params kept "
                        r"x\d\.\d{3} x\d\.\d{3} \| \d+\.\ds", setup), setup
    # a rule whose measurement the origin lacks stops the run at that net's setup, by name
    bare = _instance()
    bare.curves = None
    with pytest.raises(RuntimeError, match="toy.pt: the menu rule sens_thresh needs the origin's damage curves"):
        _train(monkeypatch, tmp_path / "bare", bare, SPECTRA_PLAN_MENU=MENU_FLAG, SPECTRA_PLAN_MENU_INIT="6", **T1)


def test_trainer_menu_under_a_mixed_budget_prices_its_rules_per_instance(monkeypatch, tmp_path, v10_env):
    """``sens`` is the residual's prior on both budgets, so one menu serves a mixed run; sens_cost and sens_thresh are
    priced in each instance's own budget."""
    draws = _spy_draws(monkeypatch)
    inst = _instance()
    inst.reward = _reward(inst)
    lines, records, blob = _train(monkeypatch, tmp_path, inst, budget="mixed", instances=8,
                                  SPECTRA_PLAN_MENU=MENU_FLAG, **T1)
    steps = [r for r in records if "it" in r]
    cfg = plan_trainer.config()
    assert {r["budget"] for r in steps} == {"params", "flops"} and blob["budget"] == "mixed"
    setup = [line for line in lines if line.startswith("[plan] menu setup ")]
    assert len(setup) == 1 and re.fullmatch(
        r"\[plan\] menu setup toy\.pt: sens,uniform,sens_cost,sens_thresh \| k=0\.60 rule plans' params kept "
        r"(x\d\.\d{3} ?){4} \| k=0\.60 b=flops rule plans' params kept (x\d\.\d{3} ?){4} \| \d+\.\ds", setup[0]), setup
    for it, r in enumerate(steps):
        budget, u, z = r["budget"], draws[2 * it + 1][2], draws[2 * it][2]
        rules = inst.rules_at(MENU, r["kappa"], budget, 0.5, cfg.k_min, 2)
        cost = {row: inst.cost[row][1 if budget == "flops" else 0] for row in inst.pm.rows}
        assert rules[2] == alloc_walk.weights("sens_cost", {row: float(inst.sens[row]) for row in inst.pm.rows}, 0.5,
                                              cost=cost)
        kept = [plan_agent.menu_decode(z[k].tolist(), plan_agent.menu_shares(u[k].tolist()), rules,
                                       inst.cost_model(budget), r["kappa"], cfg.k_min, min_width=2, zmax=0.3)[1]["kept"]
                for k in range(cfg.k)]
        assert r["kept"] == kept and all(abs(v - r["kappa"]) < TOL[budget] for v in kept)
    assert inst.rules_at(MENU, 0.5, "params")[2] != inst.rules_at(MENU, 0.5, "flops")[2]
    menu_lines = [line for line in lines if line.startswith("[plan] menu summary ")]
    assert len(menu_lines) == 6 and sum(" b=flops: pi " in line for line in menu_lines) == 3
    assert len([r for r in records if "summary" in r and len(r["menu_pi"]) == 4]) == 6


def test_a_menu_run_stopped_and_resumed_ends_where_the_uninterrupted_run_does(monkeypatch, tmp_path, v10_env):
    inst = _instance()
    inst.reward = _synthetic_reward(inst)
    inst.cost_model("flops")  # built before the first job: the three jobs share the instance, and building it draws
    knobs = {**T1, "SPECTRA_PLAN_MENU": MENU_FLAG, "SPECTRA_PLAN_MENU_INIT": "1", "SPECTRA_PLAN_BUDGET": "flops",
             "SPECTRA_PLAN_STATE": "1"}
    _full_lines, full_records = _job(monkeypatch, tmp_path / "full", inst, **knobs)
    a_lines, a_records = _job(monkeypatch, tmp_path / "a", inst, SPECTRA_PLAN_MAX_MINUTES="1e-9", **knobs)
    assert "[plan] MAX_MINUTES=1e-09 reached after 2 instances" in a_lines
    state = tmp_path / "a" / "trainer_state.pt"
    kept = plan_agent.read_blob(str(state))
    assert kept["done"] == 2 and kept["meta"]["menu"] == list(MENU) and kept["meta"]["menu_init"] == 1.0
    assert "menu_head.weight" in kept["policy"]
    b_lines, b_records = _job(monkeypatch, tmp_path / "b", inst, SPECTRA_PLAN_RESUME=str(state), **knobs)
    assert f"[plan] resumed from {state} at 2 instances" in b_lines
    assert _steps(a_records) + _steps(b_records) == _steps(full_records)
    assert all("menu_pi" in r for r in _steps(full_records))
    full = _weights(tmp_path / "full" / "policy_latest.pt")
    assert _equal(_weights(tmp_path / "b" / "policy_latest.pt"), full)
    assert not _equal(_weights(tmp_path / "a" / "policy_latest.pt"), full)
    assert float(full["menu_head.weight"].abs().sum()) > 0 and full["menu_head.bias"].tolist() != [1.0, 0.0, 0.0, 0.0]
    summary = lambda rs: [r for r in rs if "summary" in r]  # noqa: E731
    assert summary(b_records) == summary(full_records) and summary(full_records)[0]["menu_pi"][0] < 0.9
    # a chained job under another menu setting, or without the menu, is another run
    with pytest.raises(ValueError, match=r"menu_sigma=0.5 \(here 0.3\)"):
        _job(monkeypatch, tmp_path / "c", inst, SPECTRA_PLAN_RESUME=str(state),
             **{**knobs, "SPECTRA_PLAN_MENU_SIGMA": "0.3"})
    with pytest.raises(ValueError, match=r"menu=\['sens', 'uniform', 'sens_cost', 'sens_thresh'\] \(here None\)"):
        _job(monkeypatch, tmp_path / "d", inst, SPECTRA_PLAN_RESUME=str(state),
             **{**knobs, "SPECTRA_PLAN_MENU": ""})


# ------------------------------------------------------------------ (g) save and load


def test_menu_checkpoint_round_trips_and_the_three_mismatches_are_refused(monkeypatch, tmp_path, v10_env):
    policy = _hot_policy()
    path = _save(tmp_path, policy, "menu", **RESIDUAL_META, **MENU_META)
    plain_path = _save(tmp_path, plan_agent.PlanPolicy(63), "plain", **RESIDUAL_META)
    blob = plan_agent.read_blob(path)                                               # reading never builds or checks
    assert {key: blob[key] for key in MENU_META} == MENU_META and "menu" not in plan_agent.read_blob(plain_path)
    assert plan_agent.check_menu(plan_agent.read_blob(plain_path)) is None
    monkeypatch.setenv("SPECTRA_PLAN_MENU", MENU_FLAG)
    loaded, again = plan_agent.load_policy(path)
    assert loaded.menu == 4 and not loaded.training and again["menu"] == list(MENU)
    assert plan_agent.check_menu(blob) == MENU
    assert _equal(dict(loaded.state_dict()), dict(policy.state_dict()))
    state = toy_state(8)
    mask, k, x = torch.ones(8, dtype=torch.bool), torch.arange(8) // 2, [0.2, -0.1, 0.05, 0.6]
    with torch.no_grad():
        policy.eval()
        assert torch.equal(loaded(state, mask, k, 4), policy(state, mask, k, 4))
        assert torch.equal(loaded.menu_logits(state, x), policy.menu_logits(state, x))
    # 1. a menu policy without the flag
    monkeypatch.delenv("SPECTRA_PLAN_MENU")
    with pytest.raises(ValueError, match="is a menu policy; set SPECTRA_PLAN_MENU=sens,uniform,sens_cost,sens_thresh"):
        plan_agent.load_policy(path)
    assert plan_agent.load_policy(plain_path)[0].menu == 0                           # no menu anywhere: as before
    # 2. a menu policy under another menu: other rules, fewer rules, another order
    for other in ("sens,uniform,sens_cost", "sens,uniform", "sens,sens_cost,uniform,sens_thresh"):
        monkeypatch.setenv("SPECTRA_PLAN_MENU", other)
        with pytest.raises(ValueError, match="is another menu"):
            plan_agent.load_policy(path)
        with pytest.raises(ValueError, match="trained with the menu sens,uniform,sens_cost,sens_thresh"):
            plan_agent.policy_of(blob)
    # 3. a policy without a menu under the flag
    monkeypatch.setenv("SPECTRA_PLAN_MENU", MENU_FLAG)
    with pytest.raises(ValueError, match=f"has no menu, but SPECTRA_PLAN_MENU={MENU_FLAG} is set"):
        plan_agent.load_policy(plain_path)
    # the checkpoint and its policy must agree on the menu when they are written
    with pytest.raises(ValueError, match="menu head for 4 rules"):
        _save(tmp_path, policy, "bad", **RESIDUAL_META)
    with pytest.raises(ValueError, match="menu head for 0 rules"):
        _save(tmp_path, plan_agent.PlanPolicy(63), "bad", **RESIDUAL_META, **MENU_META)
    with pytest.raises(ValueError, match="menu head for 4 rules"):
        _save(tmp_path, policy, "bad", **RESIDUAL_META, **{**MENU_META, "menu": ["sens", "uniform"]})
    assert not (tmp_path / "plan_agent" / "policy_bad.pt").exists()


def test_warm_start_and_resume_refuse_another_menu(monkeypatch, v10_env):
    assert "menu" in plan_agent.ENSEMBLE_KEYS
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    base = {"tokens": "layer", "zero": [], "k_min": 0.1, "budget": "flops", "residual": "sens", "residual_alpha": 0.5,
            "residual_zmax": 0.3, "min_width": 2}
    menu_policy, plain_policy = plan_agent.PlanPolicy(63, menu=4), plan_agent.PlanPolicy(63)
    menu_blob = {**base, **MENU_META, "feature_dim": 63, "encoder": "transformer",
                 "state_dict": _hot_policy().state_dict()}
    plain_blob = {**base, "feature_dim": 63, "encoder": "transformer", "state_dict": plain_policy.state_dict()}
    plan_trainer.warm_start(menu_policy, menu_blob, {**base, **MENU_META}, "/ckpt.pt")
    assert _equal(dict(menu_policy.state_dict()), dict(menu_blob["state_dict"]))
    plan_trainer.warm_start(menu_policy, menu_blob, {**base, **MENU_META, "menu_sigma": 0.2, "menu_init": 3.0},
                            "/ckpt.pt")                                       # the run's own sigma and init may differ
    with pytest.raises(ValueError, match="trained with menu=None"):
        plan_trainer.warm_start(menu_policy, plain_blob, {**base, **MENU_META}, "/ckpt.pt")
    with pytest.raises(ValueError, match=r"trained with menu=\['sens', 'uniform', 'sens_cost', 'sens_thresh'\]"):
        plan_trainer.warm_start(plain_policy, menu_blob, dict(base), "/ckpt.pt")
    with pytest.raises(ValueError, match="trained with menu="):
        plan_trainer.warm_start(menu_policy, menu_blob, {**base, **MENU_META, "menu": ["sens", "sens_thresh"]},
                                "/ckpt.pt")
    stored = {**base, **MENU_META, "config": {}}
    plan_trainer.check_resume(stored, dict(stored), "/state.pt")
    with pytest.raises(ValueError, match=r"menu_init=6.0 \(here 2.0\)"):
        plan_trainer.check_resume(stored, {**stored, "menu_init": 2.0}, "/state.pt")
    with pytest.raises(ValueError, match="menu_anchor="):
        plan_trainer.check_resume(stored, {**base, "config": {}}, "/state.pt")


# ------------------------------------------------------------------ (h) the eval's mean plan is the trainer's


@pytest.mark.parametrize("budget", ["flops", "params"])
def test_plan_for_env_plans_the_trainers_mean_plan_for_the_same_net_and_kappa(monkeypatch, tmp_path, v10_env, budget):
    env, model, plan = _origin(monkeypatch)
    path = _save(tmp_path, _hot_policy(), "menu", **RESIDUAL_META, **MENU_META)
    monkeypatch.setenv("SPECTRA_PLAN_MENU", MENU_FLAG)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    # the trainer's side: the origin read by from_env (sens, costs and curves on the calibration batches), the loaded
    # policy, the mean plan
    _seed_all(5)
    plan_agent.NetInstance.from_env(env, "toy.pt", None, None, zero=("sens",), with_cost=True)
    without = alloc_walk.rng_snapshot()
    _seed_all(5)
    inst = plan_agent.NetInstance.from_env(env, "toy.pt", None, None, zero=("sens",), with_cost=True, with_curves=True)
    assert _equal(alloc_walk.rng_snapshot(), without)                              # the curves' cuts leave no trace ...
    batches = group_sensitivity.calibration_batches(env.train_loader, group_sensitivity.CALIB_BATCHES, "cpu")
    qs, loss, raw = alloc_walk.curve_probes(model, plan, batches, SHAPE)
    assert not _equal(alloc_walk.rng_snapshot(), without)                          # ... though they do draw
    assert (inst.curves["qs"], inst.curves["loss"], inst.curves["raw"]) == (qs, loss, raw)
    assert inst.curves["seconds"] > 0
    assert set(inst.cost) == set(inst.sens) == set(inst.pm.rows)
    policy, _blob = plan_agent.load_policy(path)
    cfg = plan_trainer.config()
    cm = inst.cost_model(budget)
    kw = {"budget": "flops"} if budget == "flops" else {}
    for kappa in (0.45, 0.6):
        prior = plan_trainer.prior_of(inst, RES, budget)
        at = plan_trainer.menu_at(inst, SETTINGS, RES, kappa, cfg, budget, 2)
        want, want_info = plan_trainer.mean_plan(policy, inst, kappa, cfg, budget, 2, prior, 0.3, at)
        del lines[:]
        widths, info = plan_agent.plan_for_env(env, kappa, path, kappa=kappa, **kw)
        assert widths == want and info["kept"] == want_info["kept"] and abs(info["kept"] - kappa) < TOL[budget]
        menu = info["menu"]
        assert menu["kinds"] == list(MENU) and menu["pi"] == want_info["pi"] and menu["x"] == at["x"]
        assert sum(p > 0 for p in menu["pi"]) >= 2 and sum(menu["pi"]) == pytest.approx(1.0)   # a real blend
        assert menu["pi"] == plan_agent.menu_shares(menu["psi"], 0.005) and menu["seconds"] > 0
        assert menu["kept"] == [rule_info["kept"] for _w, rule_info in at["plans"]]
        assert menu["params"] == [cm.params(w) / float(inst.pm.total0) for w, _i in at["plans"]]
        assert widths == plan_agent.menu_decode([info["sens"][row] for row in cm.rows], menu["pi"], at["rules"], cm,
                                                kappa, 0.1, min_width=2, zmax=0.3)[0]
        assert widths != at["plans"][0][0] and info["min_width"] == 2 and info["kind"] == "agent"
        # the prior entry: the anchor's own plan, and the blend as the weights the scores multiplied
        assert info["residual"] == "sens" and info["prior"]["kind"] == "sens" and info["prior"]["zmax"] == 0.3
        assert info["prior"]["widths"] == at["plans"][0][0] and info["prior"]["kept"] == at["plans"][0][1]["kept"]
        assert info["prior"]["weights"] == want_info["base"] == plan_agent.menu_mix(menu["pi"], at["rules"], cm.rows)
        assert info["prior"]["weights"] != prior and info["prior"]["moved"] > 0
        assert alloc_walk.grid_priority(info, cm.rows) == {
            row: math.log(max(1e-12, want_info["base"][row])) + plan_agent.trust_score(info["sens"][row], 0.3)
            for row in cm.rows}
        line = [x for x in lines if x.startswith("[alloc] plan agent residual on ")]
        assert len(line) == 1
        assert line[0].startswith("[alloc] plan agent residual on sens alpha=0.5 zmax 0.3: prior keeps x")
        assert line[0].endswith(f"dist {info['prior']['dist']:.3f}; menu pi "
                                + " ".join(f"{kind} {p:.3f}" for kind, p in zip(MENU, menu["pi"])))
        assert not any("WARNING" in x for x in lines)
    # the walk's own call: the decode target under the target the state carries; kappa is the head's, the rules are
    # the target's
    widths, info = plan_agent.plan_for_env(env, 0.58, path, kappa=0.6, **kw)
    at58 = plan_trainer.menu_at(inst, SETTINGS, RES, 0.58, cfg, budget, 2)
    assert info["menu"]["x"] == at58["x"][:-1] + [0.6]
    assert info["menu"]["kept"] == [i["kept"] for _w, i in at58["plans"]]
    with torch.no_grad():
        psi = policy.menu_logits(inst.state_at(0.6, budget), info["menu"]["x"]).tolist()
    assert info["menu"]["psi"] == psi and abs(info["kept"] - 0.58) < TOL[budget]


def test_plan_for_env_at_the_zero_init_menu_is_the_prior_and_a_plain_agent_is_as_before(monkeypatch, tmp_path, v10_env):
    env, model, plan = _origin(monkeypatch)
    pm = plan_agent.ParamModel(model, plan)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    plain = _save(tmp_path, plan_agent.PlanPolicy(63), "plain", **RESIDUAL_META)
    _seed_all(7)
    w0, i0 = plan_agent.plan_for_env(env, 0.58, plain)                               # no menu anywhere: tree_v35's plan
    rng_plain = alloc_walk.rng_snapshot()
    assert "menu" not in i0 and not any("menu" in line for line in lines)
    assert set(i0) == {"kind", "alpha", "target", "kept", "budget", "keeps", "origin_widths", "sens", "held", "policy",
                       "min_width", "residual", "prior"}
    prior = i0["prior"]["weights"]
    assert w0 == plan_agent.scale_decode(prior, pm, 0.58, 0.1, min_width=2)[0] == i0["prior"]["widths"]
    menu = _save(tmp_path, plan_agent.PlanPolicy(63, menu=4), "menu", **RESIDUAL_META, **MENU_META)
    monkeypatch.setenv("SPECTRA_PLAN_MENU", MENU_FLAG)
    del lines[:]
    _seed_all(7)
    w1, i1 = plan_agent.plan_for_env(env, 0.58, menu)                                # the zero-init menu: the same plan
    assert _equal(alloc_walk.rng_snapshot(), rng_plain)     # and the generators the recovery draws from, as it left
    assert w1 == w0 and i1["kept"] == i0["kept"] and i1["menu"]["pi"] == [1.0, 0.0, 0.0, 0.0]
    assert i1["menu"]["psi"] == [6.0, 0.0, 0.0, 0.0] and i1["prior"] == i0["prior"]
    assert {key: i1[key] for key in i0 if key != "policy"} == {key: i0[key] for key in i0 if key != "policy"}
    assert set(i1) == set(i0) | {"menu"} and i1["policy"] == "plan_agent/policy_menu.pt"
    assert lines[-1].endswith("0/12 groups moved, max |Δkeep| 0.00, dist 0.000; menu pi sens 1.000 uniform 0.000 "
                              "sens_cost 0.000 sens_thresh 0.000")
    # the walk's plan line and its grid rounding read the menu agent as they read every agent
    assert alloc_walk.grid_priority(i1, pm.rows) == alloc_walk.grid_priority(i0, pm.rows)
    # not defined for a menu policy yet: a sampled plan, an ensemble
    with pytest.raises(ValueError, match="a sampled plan"):
        plan_agent.plan_for_env(env, 0.58, menu, sample=(0.2, 3))
    with pytest.raises(ValueError, match="an ensemble"):
        plan_agent.plan_for_env(env, 0.58, f"{menu},{menu}")
    # the contract at the eval: the flag and the checkpoint agree, before anything is measured
    monkeypatch.setattr(group_sensitivity, "calibration_batches",
                        lambda *a, **k: pytest.fail("refused before measuring"))
    with pytest.raises(ValueError, match="has no menu, but SPECTRA_PLAN_MENU"):
        plan_agent.plan_for_env(env, 0.58, plain)
    monkeypatch.setenv("SPECTRA_PLAN_MENU", "sens,uniform")
    with pytest.raises(ValueError, match="is another menu"):
        plan_agent.plan_for_env(env, 0.58, menu)
    monkeypatch.delenv("SPECTRA_PLAN_MENU")
    with pytest.raises(ValueError, match="is a menu policy; set SPECTRA_PLAN_MENU="):
        plan_agent.plan_for_env(env, 0.58, menu)


# ------------------------------------------------------------------ (i) the anchor


def test_a_menu_whose_anchor_is_not_the_residuals_prior_is_refused(monkeypatch, tmp_path, v10_env):
    plan_agent.check_menu_anchor(MENU, "sens", "flops")
    plan_agent.check_menu_anchor(MENU, "sens", "params")
    plan_agent.check_menu_anchor(MENU, "auto", "flops")                              # auto on FLOPs starts from sens
    plan_agent.check_menu_anchor(("sens_cost", "sens"), "auto", "params")
    with pytest.raises(ValueError,
                       match="anchor .* is 'sens', but the residual's prior on a params instance is 'sens_cost'"):
        plan_agent.check_menu_anchor(MENU, "auto", "params")
    with pytest.raises(ValueError, match="list 'sens' first"):
        plan_agent.check_menu_anchor(("uniform", "sens"), "sens", "flops")
    for off in ("off", None):
        with pytest.raises(ValueError, match="the menu needs the residual"):
            plan_agent.check_menu_anchor(MENU, off, "flops")
    # the trainer: refused from the flags alone, before any origin is read
    monkeypatch.setattr(plan_agent.NetInstance, "from_env",
                        classmethod(lambda cls, *a, **k: pytest.fail("refused before the origins are read")))
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    shard = [("toy.pt", (None, None))]
    monkeypatch.setenv("SPECTRA_PLAN_NETS", "toy")
    monkeypatch.setenv("SPECTRA_PLAN_BUDGET", "flops")
    monkeypatch.setenv("SPECTRA_PLAN_MENU", "uniform,sens")
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "sens")
    with pytest.raises(ValueError, match="SPECTRA_PLAN_MENU=uniform,sens: the menu's anchor .* is 'uniform'"):
        plan_trainer.run(None, shard)
    monkeypatch.setenv("SPECTRA_PLAN_MENU", MENU_FLAG)
    monkeypatch.delenv("SPECTRA_PLAN_RESIDUAL")
    with pytest.raises(ValueError, match="the menu needs the residual"):
        plan_trainer.run(None, shard)
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "sens_cost")
    with pytest.raises(ValueError, match="the residual's prior on a flops instance is 'sens_cost'"):
        plan_trainer.run(None, shard)
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "auto")                               # two priors under a mixed budget
    monkeypatch.setenv("SPECTRA_PLAN_BUDGET", "mixed")
    with pytest.raises(ValueError, match="on a params instance is 'sens_cost'"):
        plan_trainer.run(None, shard)
    assert plan_trainer.menu_settings("sens", ("params", "flops")) == SETTINGS         # sens on both: one anchor
    assert plan_trainer.menu_settings("auto", ("flops",)) == SETTINGS
    # the eval: the checkpoint's anchor against the rule this job's residual starts from on its budget
    monkeypatch.delenv("SPECTRA_PLAN_RESIDUAL")
    monkeypatch.delenv("SPECTRA_PLAN_BUDGET")
    env = _origin(monkeypatch)[0]
    monkeypatch.setattr(group_sensitivity, "calibration_batches",
                        lambda *a, **k: pytest.fail("refused before measuring"))
    auto = _save(tmp_path, plan_agent.PlanPolicy(63, menu=4), "auto", **{**RESIDUAL_META, "residual": "auto"},
                 **MENU_META)
    with pytest.raises(ValueError, match="the plan agent at .*policy_auto.pt: the menu's anchor .* is 'sens', but the "
                                         "residual's prior on a params instance is 'sens_cost'"):
        plan_agent.plan_for_env(env, 0.58, auto)
    bare = _save(tmp_path, plan_agent.PlanPolicy(63, menu=4), "bare", **MENU_META)   # no residual in the checkpoint
    with pytest.raises(ValueError, match="the menu needs the residual"):
        plan_agent.plan_for_env(env, 0.58, bare)
