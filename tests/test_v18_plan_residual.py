"""
v18 ``SPECTRA_PLAN_RESIDUAL`` (T2): the plan agent as a residual on a no-agent rule. ``plan_agent.residual_decode``
plans keep_g ∝ w_g · exp(z_g) on the sens / sens_cost weights of ``prior_weights`` (``auto`` = sens_cost on params,
sens on FLOPs), so the zero-init agent's mean plan is ``scale_decode`` of the prior, bit for bit; the trainer caches
each net's prior per budget, decodes every sampled and mean plan on it, scores the prior as a reference, scales its σ
schedule by RESIDUAL_SIGMA / SIGMA and writes the mode into the checkpoint; ``plan_for_env`` reads it back (the env
overrides it with a WARNING), measures the target's prior through the alloc walk's calibration path and reports the
agent's distance from it. Default off: with the flag unset and no ``residual`` in the blob every decoder, record, log
line and checkpoint is as in tree_v17.

CPU only, no datasets.  python -m pytest tests/test_v18_plan_residual.py -v
"""

import inspect
import json
import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import RATES, SHAPE, ZOO, _loader, _plan, _randomize_norms, toy_state  # noqa: E402
from tests.test_plan_agent import _instance as _toy_instance  # noqa: E402
from tests.test_v15_flops_budget import POLICY  # noqa: E402
from tests.test_v16_sens_cost import _batches  # noqa: E402
from tests.test_v17_plan_min_width import _floor_ok  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, fortify, plan_agent, plan_trainer, state_dump  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_MIN_WIDTH")
TOL = {"params": 0.05, "flops": 0.1}  # one stage-1 channel of thin_r20_w4 is ~10 % of its MACs


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


def _model(name="thin_r20_w4"):
    torch.manual_seed(0)
    model = ZOO[name]().eval()
    _randomize_norms(model)
    return model


def _measured(model, plan, batches=None):
    """``(sens, costs)`` of the planned groups, as ``group_sensitivity(costs={})`` measures them."""
    costs = {}
    sens, _base = group_sensitivity.group_sensitivity(model, plan, batches if batches is not None else _batches(),
                                                      SHAPE, costs=costs)
    return sens, costs


def _cost_model(model, plan, budget):
    return plan_agent.ParamModel(model, plan) if budget == "params" else plan_agent.FlopModel(model, plan, SHAPE)


def _instance(with_cost=True):
    """The toy instance of ``tests/test_plan_agent.py`` (synthetic sens), priced for the sens_cost prior."""
    inst = _toy_instance()
    if with_cost:
        _sens, inst.cost = _measured(inst.model, inst.plan)
    return inst


def _strip(records):
    return [{k: v for k, v in r.items() if k != "seconds"} for r in records]


# ------------------------------------------------------------------ the flags


def test_plan_residual_flags_default_off_parse_and_errors(monkeypatch):
    assert fortify.plan_residual() == "off" and fortify.plan_residual(default=None) is None
    assert fortify.plan_residual_alpha() == 0.5 and fortify.plan_residual_alpha(default=None) is None
    assert fortify.plan_residual_sigma() == 0.25
    assert plan_trainer.residual() == ("off", 0.5, 0.25)
    for raw, want in (("sens", "sens"), ("SENS_COST", "sens_cost"), (" auto ", "auto"), ("off", "off")):
        monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", raw)
        assert fortify.plan_residual() == want and fortify.plan_residual(default=None) == want
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "prior")
    with pytest.raises(ValueError):
        fortify.plan_residual()
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "")
    assert fortify.plan_residual() == "off" and fortify.plan_residual(default=None) is None
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL_ALPHA", "0.7")
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL_SIGMA", "0.1")
    assert fortify.plan_residual_alpha() == 0.7 and fortify.plan_residual_alpha(default=None) == 0.7
    assert fortify.plan_residual_sigma() == 0.1 and plan_trainer.residual() == ("off", 0.7, 0.1)
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL_SIGMA", "-1")
    assert fortify.plan_residual_sigma() == 0.0


def test_prior_kind_auto_picks_sens_cost_on_params_and_sens_on_flops():
    assert plan_agent.prior_kind("auto", "params") == "sens_cost" and plan_agent.prior_kind("auto", "flops") == "sens"
    assert plan_agent.prior_kind("auto") == "sens_cost"
    for kind in plan_agent.PRIOR_KINDS:
        assert plan_agent.prior_kind(kind, "params") == kind == plan_agent.prior_kind(kind, "flops")
    for bad in ("uniform", "inner", "off", ""):
        with pytest.raises(ValueError):
            plan_agent.prior_kind(bad)


def test_sigma_schedule_scales_and_is_unchanged_at_scale_one():
    cfg = plan_trainer.config()
    for it in (0, 90, 180, 300):
        assert plan_trainer.sigma_at(cfg, it, 1.0) == plan_trainer.sigma_at(cfg, it)
        assert plan_trainer.sigma_at(cfg, it, 0.5) == pytest.approx(0.5 * plan_trainer.sigma_at(cfg, it))
    assert plan_trainer.sigma_at(cfg, 0, 0.5) == 0.25 and plan_trainer.sigma_at(cfg, 300, 0.5) == pytest.approx(0.1)


# ------------------------------------------------------------------ prior weights and the decoder


def test_prior_weights_are_the_alloc_walk_rules_and_need_costs_for_sens_cost():
    model = _model()
    plan = _plan(model)
    rows = [row for _g, row in plan]
    sens, costs = _measured(model, plan)
    assert plan_agent.prior_weights("sens", rows, sens) == alloc_walk.weights("sens", sens, 0.5)
    assert plan_agent.prior_weights("sens", rows, sens, costs, "flops", 0.7) == alloc_walk.weights("sens", sens, 0.7)
    p_cost = {row: costs[row][0] for row in rows}
    f_cost = {row: costs[row][1] for row in rows}
    assert plan_agent.prior_weights("sens_cost", rows, sens, costs) == alloc_walk.weights("sens_cost", sens, 0.5,
                                                                                            cost=p_cost)
    assert plan_agent.prior_weights("sens_cost", rows, sens, costs, "flops") == alloc_walk.weights(
        "sens_cost", sens, 0.5, cost=f_cost)
    assert plan_agent.prior_weights("sens_cost", rows, sens, costs) != plan_agent.prior_weights(
        "sens_cost", rows, sens, costs, "flops")
    with pytest.raises(ValueError):
        plan_agent.prior_weights("sens_cost", rows, sens)
    with pytest.raises(ValueError):
        plan_agent.prior_weights("uniform", rows, sens)
    with pytest.raises(ValueError):
        plan_agent.prior_weights("sens", rows, sens, budget="macs")


@pytest.mark.parametrize("budget", ["params", "flops"])
@pytest.mark.parametrize("kind", ["sens", "sens_cost"])
@pytest.mark.parametrize("min_width", [1, 2])
def test_residual_decode_at_zero_is_the_prior_plan_exactly(budget, kind, min_width):
    model = _model()
    plan = _plan(model)
    pm = _cost_model(model, plan, budget)
    sens, costs = _measured(model, plan)
    prior = plan_agent.prior_weights(kind, pm.rows, sens, costs, budget)
    for kappa in (0.4, 0.6):
        w_r, i_r = plan_agent.residual_decode([0.0] * len(pm.rows), prior, pm, kappa, min_width=min_width)
        w_s, i_s = plan_agent.scale_decode(prior, pm, kappa, min_width=min_width)
        assert w_r == w_s and i_r["kept"] == i_s["kept"] and i_r["keeps"] == i_s["keeps"] and i_r["c"] == i_s["c"]
        assert i_r["w"] == prior and i_r["z"] == {row: 0.0 for row in pm.rows}
        assert _floor_ok(w_r, pm.widths0, min_width) and abs(i_r["kept"] - kappa) < TOL[budget]
        assert i_r["kept"] == pytest.approx(pm.kept(w_r))
    if min_width == 1:  # the prior's weights are the comparator rule's: what plan_targets computes for that kind
        _w, info = alloc_walk.plan_targets(model, _batches(), SHAPE, kind, 0.6, budget=budget)
        assert prior == alloc_walk.weights(kind, info["sens"], info["alpha"], cost=info.get("cost"))


def test_residual_decode_moves_with_z_lands_the_budget_ignores_a_shift_and_holds():
    model = _model()
    plan = _plan(model)
    sens, costs = _measured(model, plan)
    held = {row for group, row in plan if len(group.producers) > 1}
    for budget in ("params", "flops"):
        pm = _cost_model(model, plan, budget)
        prior = plan_agent.prior_weights(plan_agent.prior_kind("auto", budget), pm.rows, sens, costs, budget)
        base, _i = plan_agent.scale_decode(prior, pm, 0.6)
        eps = torch.randn(len(pm.rows), generator=torch.Generator().manual_seed(1)).tolist()
        z = [0.25 * e for e in eps]
        widths, info = plan_agent.residual_decode(z, prior, pm, 0.6)
        assert widths != base and set(widths) == set(pm.rows)
        assert abs(info["kept"] - 0.6) < TOL[budget] and info["kept"] == pytest.approx(pm.kept(widths))
        assert min(info["keeps"].values()) >= 0.1 - 1e-12
        assert info["w"] == pytest.approx({row: prior[row] * math.exp(z[k]) for k, row in enumerate(pm.rows)})
        assert plan_agent.residual_decode([v + 3.0 for v in z], prior, pm, 0.6)[0] == widths  # c absorbs a shift
        real = alloc_walk.cut_to(model, plan, plan_agent.rates_of(widths, pm.widths0), SHAPE)
        assert alloc_walk.group_widths(real, pm.rows) == widths
        del real
        kept, k_info = plan_agent.residual_decode(z, prior, pm, 0.6, held=held)
        assert all(kept[row] == pm.widths0[row] and k_info["keeps"][row] == 1.0 for row in held)
    pm = plan_agent.ParamModel(model, plan)
    prior = plan_agent.prior_weights("sens", pm.rows, sens)
    huge, _i = plan_agent.residual_decode([1e4 if k == 0 else -1e4 for k in range(len(pm.rows))], prior, pm, 0.6)
    assert set(huge) == set(pm.rows)                                   # exp stays finite: the scores are clipped


def test_instance_prior_is_cached_per_budget_and_reference_sens_cost_uses_it(v10_env):
    inst = _instance()
    sens = {row: float(inst.sens[row]) for row in inst.pm.rows}
    assert inst.prior("sens") is inst.prior("sens", "params", 0.5) and inst.prior("sens") == alloc_walk.weights(
        "sens", sens, 0.5)
    assert inst.prior("sens", "flops") == inst.prior("sens") and inst.prior("sens", "params", 1.0) != inst.prior("sens")
    p, f = inst.prior("sens_cost", "params"), inst.prior("sens_cost", "flops")
    assert p == alloc_walk.weights("sens_cost", sens, 0.5, cost={r: inst.cost[r][0] for r in inst.pm.rows})
    assert f == alloc_walk.weights("sens_cost", sens, 0.5, cost={r: inst.cost[r][1] for r in inst.pm.rows}) and p != f
    assert inst.prior("sens_cost", "flops") is f
    for budget in ("params", "flops"):
        for kind in plan_agent.PRIOR_KINDS:
            ref = inst.reference(kind, 0.4, budget=budget, min_width=2)
            assert ref == plan_agent.scale_decode(inst.prior(kind, budget), inst.cost_model(budget), 0.4, min_width=2)
    with pytest.raises(ValueError):
        inst.prior("uniform")
    plain = _instance(with_cost=False)
    assert plain.cost is None and plain.prior("sens_cost") is None and plain.reference("sens_cost", 0.4) is None
    assert plain.prior("sens") == inst.prior("sens") and plain.reference("sens", 0.4) == inst.reference("sens", 0.4)
    plain.sens = None
    assert plain.prior("sens") is None and plain.reference("sens", 0.4) is None
    assert plain.reference("uniform", 0.4) is not None


# ------------------------------------------------------------------ trainer


def _run(monkeypatch, tmp_path, inst, budget, instances=5, calls=None):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(plan_agent.NetInstance, "from_env",
                        classmethod(lambda cls, env, p, m, l, **kw: (calls if calls is not None else []).append(kw)
                                    or inst))
    monkeypatch.setattr(plan_trainer, "out_dir", lambda: str(tmp_path))
    for key, value in {"SPECTRA_PLAN_INSTANCES": str(instances), "SPECTRA_PLAN_K": "3", "SPECTRA_PLAN_BATCH": "2",
                       "SPECTRA_PLAN_PROXY": "bn2", "SPECTRA_PLAN_REF_EVERY": "2", "SPECTRA_PLAN_SAVE_EVERY": "4",
                       "SPECTRA_PLAN_NETS": "toy", "SPECTRA_PLAN_BUDGET": budget}.items():
        monkeypatch.setenv(key, value)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    plan_trainer.run(None, [("toy.pt", (None, None))])
    with open(tmp_path / "plan_train.jsonl", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    _policy, blob = plan_agent.load_policy(str(tmp_path / "policy_latest.pt"))
    return lines, records, blob


def test_trainer_off_is_as_before_and_never_touches_the_residual_decoder(monkeypatch, tmp_path, v10_env):
    monkeypatch.setattr(plan_agent, "residual_decode", lambda *a, **k: pytest.fail("off must decode as before"))
    calls = []
    off = tmp_path / "off"
    off.mkdir()
    lines, records, blob = _run(monkeypatch, off, _instance(with_cost=False), "params", calls=calls)
    assert calls == [{"tokens": "layer", "zero": (), "with_cost": False}]
    assert not any(key in blob for key in ("residual", "residual_alpha", "residual_sigma"))
    assert not any("residual" in line or line.startswith("[plan] prior") for line in lines)
    steps = [r for r in records if "it" in r]
    cfg = plan_trainer.config()
    assert [r["sigma"] for r in steps] == [plan_trainer.sigma_at(cfg, it) for it in range(5)] and steps[0]["sigma"] == 0.5
    assert not any("residual" in r or "prior" in r for r in records)
    assert all(set(r["refs"]) == {"uniform", "sens", "inner"} for r in records if "refs" in r)
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "off")                      # an explicit off is the same run
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    lines2, records2, blob2 = _run(monkeypatch, explicit, _instance(with_cost=False), "params")
    assert lines2[0] == lines[0] and _strip(records2) == _strip(records) and set(blob2) == set(blob)


def test_trainer_residual_starts_at_the_prior_scales_sigma_and_records_it(monkeypatch, tmp_path, v10_env):
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "sens_cost")
    inst = _instance()
    calls = []
    lines, records, blob = _run(monkeypatch, tmp_path, inst, "params", calls=calls)
    assert calls == [{"tokens": "layer", "zero": (), "with_cost": True}]
    trainer_lines = [line for line in lines if line.startswith("[plan] trainer {")]
    assert len(trainer_lines) == 1 and trainer_lines[0].endswith("} residual=sens_cost alpha=0.5 sigma=0.25")
    prior_lines = [line for line in lines if line.startswith("[plan] prior ")]
    assert len(prior_lines) == 1 and prior_lines[0].startswith("[plan] prior toy.pt: sens_cost alpha=0.5 weights min ")
    steps = [r for r in records if "it" in r]
    cfg = plan_trainer.config()
    assert len(steps) == 5 and steps[0]["sigma"] == 0.25
    assert [r["sigma"] for r in steps] == [pytest.approx(0.5 * plan_trainer.sigma_at(cfg, it)) for it in range(5)]
    assert all(r["residual"] == "sens_cost" and r["prior"] == "sens_cost" for r in steps)
    assert all(abs(k - r["kappa"]) < TOL["params"] for r in steps for k in r["kept"])
    first = steps[0]                                                          # mu = 0 before the first update
    assert set(first["refs"]) == {"uniform", "sens", "inner", "prior"}
    assert first["mean_plan"] == first["refs"]["prior"]
    it1 = next(line for line in lines if line.startswith("[plan] it=1/5 toy.pt k="))
    assert " sigma=0.250 | " in it1 and " | mean plan " in it1 and " prior " in it1
    assert blob["residual"] == "sens_cost" and blob["residual_alpha"] == 0.5 and blob["residual_sigma"] == 0.25
    assert not any(key in blob["config"] for key in ("residual", "residual_alpha", "residual_sigma"))
    summaries = [r for r in records if "summary" in r]
    assert len(summaries) == 3 and all(set(s["refs"]) == {"uniform", "sens", "inner", "prior"} for s in summaries)
    assert all("| prior " in line for line in lines if line.startswith("[plan] summary "))
    # the recorded prior is the sens_cost rule on the params cost, decoded with the same floor as the mean plan
    kappa = steps[0]["kappa"]
    widths, info = plan_agent.scale_decode(inst.prior("sens_cost", "params", 0.5), inst.pm, kappa, 0.1)
    assert first["refs"]["prior"]["kept"] == pytest.approx(info["kept"])


def test_trainer_auto_under_a_mixed_budget_takes_sens_cost_on_params_and_sens_on_flops(monkeypatch, tmp_path, v10_env):
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "auto")
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL_ALPHA", "0.7")
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL_SIGMA", "0.3")
    inst = _instance()
    calls = []
    lines, records, blob = _run(monkeypatch, tmp_path, inst, "mixed", instances=8, calls=calls)
    assert calls[0]["with_cost"] is True
    steps = [r for r in records if "it" in r]
    assert {r["budget"] for r in steps} == {"params", "flops"}
    assert all(r["residual"] == "auto" and r["prior"] == ("sens" if r["budget"] == "flops" else "sens_cost")
               for r in steps)
    assert steps[0]["sigma"] == pytest.approx(0.3) and all(r["sigma"] <= 0.3 + 1e-12 for r in steps)
    prior_lines = [line for line in lines if line.startswith("[plan] prior ")]
    assert len(prior_lines) == 2                                               # one per budget, params first
    assert prior_lines[0].startswith("[plan] prior toy.pt: sens_cost alpha=0.7 weights min ")
    assert prior_lines[1].startswith("[plan] prior toy.pt b=flops: sens alpha=0.7 weights min ")
    assert lines[0].endswith("} residual=auto alpha=0.7 sigma=0.3")
    assert (blob["residual"], blob["residual_alpha"], blob["residual_sigma"]) == ("auto", 0.7, 0.3)
    assert inst.prior("sens", "flops", 0.7) is not None and inst.prior("sens_cost", "params", 0.7) is not None
    summaries = [r for r in records if "summary" in r]
    assert len(summaries) == 6 and all("prior" in s["refs"] for s in summaries)


def test_trainer_sens_residual_needs_no_costs_and_sens_cost_refuses_without_them(monkeypatch, tmp_path, v10_env):
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "sens")
    calls = []
    lines, records, blob = _run(monkeypatch, tmp_path / "sens", _instance(with_cost=False), "flops", calls=calls)
    assert calls == [{"tokens": "layer", "zero": (), "with_cost": False}] and blob["residual"] == "sens"
    steps = [r for r in records if "it" in r]
    assert all(r["prior"] == "sens" and r["budget"] == "flops" for r in steps)
    assert all(abs(k - r["kappa"]) < TOL["flops"] for r in steps for k in r["kept"])
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "auto")                     # auto on flops alone: sens, no costs
    calls = []
    _run(monkeypatch, tmp_path / "auto", _instance(with_cost=False), "flops", calls=calls)
    assert calls == [{"tokens": "layer", "zero": (), "with_cost": False}]
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "sens_cost")
    with pytest.raises(RuntimeError, match="sens_cost prior needs"):
        _run(monkeypatch, tmp_path / "nocost", _instance(with_cost=False), "params")


# ------------------------------------------------------------------ plan_for_env and the alloc walk


def _origin(monkeypatch):
    torch.manual_seed(0)
    model = ZOO["thin_r20_w4"]().eval()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    plan = group_sensitivity.group_plan(mwr, groups)
    rows = state_dump.token_rows(mwr, groups, plan)
    state = toy_state(len(rows))
    monkeypatch.setattr(state_dump, "encode_origin", lambda env, m, g: dict(state))
    return model, plan, rows, state


def _env(model, loader=None):
    return types.SimpleNamespace(conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES),
                                 current_model=model, target_keep=0.6, _input_shape=lambda: SHAPE,
                                 train_loader=loader if loader is not None else _loader(seed=1, shuffle=False),
                                 _dependency_groups=lambda m: channel_groups.build_channel_groups(m.model))


def _save(tmp_path, policy, name, **meta):
    path = tmp_path / "plan_agent" / f"policy_{name}.pt"
    plan_agent.save_policy(policy, str(path), tokens="layer", zero=["sens"], k_min=0.1, budget="mixed", **meta)
    return str(path)


def _expected_prior(env, model, plan, kind, budget, a=0.5):
    """The prior ``plan_for_env`` should build: the rule's weights from the batches the alloc walk's sens kind draws."""
    rows = [row for _g, row in plan]
    batches = group_sensitivity.calibration_batches(env.train_loader, group_sensitivity.CALIB_BATCHES, "cpu")
    sens, costs = _measured(model, plan, batches)
    return plan_agent.prior_weights(kind, rows, sens, costs, budget, a)


@pytest.mark.parametrize("budget", ["params", "flops"])
def test_plan_for_env_with_a_residual_blob_reproduces_the_prior_at_mu_zero(monkeypatch, tmp_path, v10_env, budget):
    model, plan, rows, state = _origin(monkeypatch)
    policy = plan_agent.PlanPolicy(63)                                        # zero-init head: mu = 0
    path = _save(tmp_path, policy, "t2", residual="auto", residual_alpha=0.5, residual_sigma=0.25)
    env = _env(model)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    kw = {"budget": "flops", "kappa": 0.6} if budget == "flops" else {}
    widths, info = plan_agent.plan_for_env(env, 0.58, path, **kw)
    kind = plan_agent.prior_kind("auto", budget)
    pm = _cost_model(model, plan, budget)
    prior = _expected_prior(env, model, plan, kind, budget)
    want, want_info = plan_agent.scale_decode(prior, pm, 0.58, 0.1)
    assert widths == want and info["kept"] == want_info["kept"] == pytest.approx(pm.kept(widths))
    assert info["kind"] == "agent" and info["residual"] == "auto" and info["budget"] == budget
    assert info["prior"]["kind"] == kind and info["prior"]["alpha"] == 0.5 and info["prior"]["weights"] == prior
    assert info["prior"]["widths"] == want and info["prior"]["kept"] == want_info["kept"]
    assert (info["prior"]["moved"], info["prior"]["max_abs"], info["prior"]["dist"]) == (0, 0.0, 0.0)
    assert info["keeps"] == {row: widths[row] / pm.widths0[row] for row in pm.rows}
    assert info["sens"] == {row: 0.0 for row in pm.rows} and "min_width" not in info
    assert not any("WARNING" in line for line in lines)
    assert [line for line in lines if line.startswith("[alloc] plan agent residual on ")] == [
        f"[alloc] plan agent residual on {kind} alpha=0.5: prior keeps x{want_info['kept']:.3f}, agent "
        f"x{want_info['kept']:.3f}; 0/{len(pm.rows)} groups moved, max |Δkeep| 0.00, dist 0.000"]
    # the comparator the eval runs beside: plan_targets of the same kind on the same batches has the same weights
    batches = group_sensitivity.calibration_batches(env.train_loader, group_sensitivity.CALIB_BATCHES, "cpu")
    _w, t_info = alloc_walk.plan_targets(model, batches, SHAPE, kind, 0.58, budget=budget)
    assert prior == alloc_walk.weights(kind, t_info["sens"], 0.5, cost=t_info.get("cost"))
    # the checkpoint's floor applies to the prior and the plan alike
    floored = _save(tmp_path, policy, "t2f", residual="auto", residual_alpha=0.5, residual_sigma=0.25, min_width=2)
    w2, i2 = plan_agent.plan_for_env(env, 0.58, floored, **kw)
    assert w2 == plan_agent.scale_decode(prior, pm, 0.58, 0.1, min_width=2)[0] and i2["min_width"] == 2
    assert _floor_ok(w2, pm.widths0, 2) and i2["prior"]["widths"] == w2
    # a draw around the mean: the sampled plan is residual_decode of mu + sigma * eps, still on the prior
    ws, si = plan_agent.plan_for_env(env, 0.58, path, sample=(0.2, 3), **kw)
    eps = torch.randn(len(pm.rows), generator=torch.Generator().manual_seed(3), dtype=torch.float64).tolist()
    assert ws == plan_agent.residual_decode([0.2 * e for e in eps], prior, pm, 0.58, 0.1)[0]
    assert si["kind"] == "agent_sample" and si["residual"] == "auto" and si["sample"]["seed"] == 3
    assert abs(si["kept"] - 0.58) < TOL[budget] and si["prior"]["widths"] == want
    assert ws != want or si["sens"] != info["sens"]


def test_plan_for_env_env_flag_overrides_the_blob_with_a_warning(monkeypatch, tmp_path, v10_env):
    model, plan, rows, state = _origin(monkeypatch)
    policy = plan_agent.PlanPolicy(63)
    t2 = _save(tmp_path, policy, "t2", residual="sens_cost", residual_alpha=0.5, residual_sigma=0.25)
    plain = _save(tmp_path, policy, "plain")
    env = _env(model)
    pm = plan_agent.ParamModel(model, plan)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    w0, i0 = plan_agent.plan_for_env(env, 0.58, plain)                        # no residual anywhere: as before
    assert w0 == plan_agent.decode([0.0] * len(pm.rows), pm, 0.58)[0] and "residual" not in i0 and "prior" not in i0
    assert not lines
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "off")                          # an explicit off on a plain blob: silent
    assert plan_agent.plan_for_env(env, 0.58, plain) == (w0, i0) and not lines
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "sens")                         # the env turns the plain blob into T2
    w1, i1 = plan_agent.plan_for_env(env, 0.58, plain)
    sens_prior = _expected_prior(env, model, plan, "sens", "params")
    assert w1 == plan_agent.scale_decode(sens_prior, pm, 0.58, 0.1)[0] and i1["residual"] == "sens"
    assert lines[0] == "[alloc] WARNING: the plan agent trained with residual=off; SPECTRA_PLAN_RESIDUAL=sens overrides it"
    lines.clear()
    w2, i2 = plan_agent.plan_for_env(env, 0.58, t2)                             # ... and a sens_cost blob into sens
    assert w2 == w1 and i2["residual"] == "sens" and i2["prior"]["kind"] == "sens"
    assert lines[0] == ("[alloc] WARNING: the plan agent trained with residual=sens_cost; SPECTRA_PLAN_RESIDUAL=sens "
                        "overrides it")
    lines.clear()
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "off")                          # the env switches T2 off: a WARNING
    w3, i3 = plan_agent.plan_for_env(env, 0.58, t2)
    assert w3 == w0 and "residual" not in i3
    assert lines == ["[alloc] WARNING: the plan agent trained with residual=sens_cost; SPECTRA_PLAN_RESIDUAL=off "
                     "overrides it"]
    lines.clear()
    monkeypatch.delenv("SPECTRA_PLAN_RESIDUAL")
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL_ALPHA", "1")                      # alpha from the env, with a WARNING
    w4, i4 = plan_agent.plan_for_env(env, 0.58, t2)
    cost_prior = _expected_prior(env, model, plan, "sens_cost", "params", a=1.0)
    assert w4 == plan_agent.scale_decode(cost_prior, pm, 0.58, 0.1)[0] and i4["prior"]["alpha"] == 1.0
    assert lines[0] == ("[alloc] WARNING: the plan agent trained with residual_alpha=0.5; "
                        "SPECTRA_PLAN_RESIDUAL_ALPHA=1 overrides it")
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL_ALPHA", "0.5")                    # the same value: no WARNING
    lines.clear()
    w5, i5 = plan_agent.plan_for_env(env, 0.58, t2)
    assert i5["prior"]["alpha"] == 0.5 and not any("WARNING" in line for line in lines)
    assert w5 == plan_agent.scale_decode(_expected_prior(env, model, plan, "sens_cost", "params"), pm, 0.58, 0.1)[0]


def test_plan_for_env_draws_the_calibration_batches_as_the_sens_kind_would(monkeypatch, tmp_path, v10_env):
    """The prior's sensitivities come from the batches the alloc walk's sens kind would draw at that point of
    ``_state``: the shuffled train loader samples with the global torch RNG, and building the policy module moves it,
    so ``plan_for_env`` draws before it builds the policy."""
    model, plan, rows, state = _origin(monkeypatch)
    policy = plan_agent.PlanPolicy(63)
    path = _save(tmp_path, policy, "t2", residual="sens", residual_alpha=0.5, residual_sigma=0.25)
    loader = _loader(n=256, batch=16, seed=5, shuffle=True)
    env = _env(model, loader)
    pm = plan_agent.ParamModel(model, plan)
    torch.manual_seed(123)
    sens_kind_batches = group_sensitivity.calibration_batches(loader, group_sensitivity.CALIB_BATCHES, "cpu")
    sens, _c = _measured(model, plan, sens_kind_batches)
    want = plan_agent.prior_weights("sens", pm.rows, sens)
    torch.manual_seed(123)
    widths, info = plan_agent.plan_for_env(env, 0.58, path)
    assert info["prior"]["weights"] == want and widths == plan_agent.scale_decode(want, pm, 0.58, 0.1)[0]
    torch.manual_seed(124)                                                      # another state: other batches
    other = group_sensitivity.calibration_batches(loader, group_sensitivity.CALIB_BATCHES, "cpu")
    assert not all(torch.equal(a[0], b[0]) for a, b in zip(other, sens_kind_batches))
    o_sens, _c = _measured(model, plan, other)
    assert plan_agent.prior_weights("sens", pm.rows, o_sens) != want


def test_plan_for_env_with_a_trained_head_moves_off_the_prior_and_lands_the_budget(monkeypatch, tmp_path, v10_env):
    model, plan, rows, state = _origin(monkeypatch)
    policy = plan_agent.PlanPolicy(63)
    with torch.no_grad():
        torch.manual_seed(1)
        policy.head.weight.normal_(std=2.0)
    path = _save(tmp_path, policy, "t2", residual="sens_cost", residual_alpha=0.5, residual_sigma=0.25)
    env = _env(model)
    pm = plan_agent.ParamModel(model, plan)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    widths, info = plan_agent.plan_for_env(env, 0.58, path)
    prior = _expected_prior(env, model, plan, "sens_cost", "params")
    base, _i = plan_agent.scale_decode(prior, pm, 0.58, 0.1)
    assert widths != base and info["prior"]["widths"] == base and info["prior"]["moved"] > 0
    assert info["prior"]["dist"] > 0 and info["prior"]["max_abs"] > 0 and abs(info["kept"] - 0.58) < TOL["params"]
    z = [info["sens"][row] for row in pm.rows]
    assert any(abs(v) > 1e-6 for v in z) and abs(sum(z)) < 1e-4                # centred scores, as the policy emits
    assert widths == plan_agent.residual_decode(z, prior, pm, 0.58, 0.1)[0]
    assert lines[-1].startswith("[alloc] plan agent residual on sens_cost alpha=0.5: prior keeps x")
    assert f"; {info['prior']['moved']}/{len(pm.rows)} groups moved, max |Δkeep| {info['prior']['max_abs']:.2f}" in lines[-1]


def test_alloc_agent_with_the_residual_calls_plan_for_env_by_its_real_signature(monkeypatch):
    """As ``tests/test_v17_plan_min_width.py`` binds the floor: ``_state`` passes nothing new, ``plan_for_env`` reads
    ``SPECTRA_PLAN_RESIDUAL`` (or the blob) itself."""
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "agent")
    monkeypatch.setenv("SPECTRA_PLAN_AGENT", POLICY)
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "auto")
    real = inspect.signature(plan_agent.plan_for_env)
    calls = []

    def fake(*args, **kwargs):
        real.bind(*args, **kwargs)
        calls.append(kwargs)
        return {5: 3}, {"kind": "agent", "alpha": 0.0, "target": args[1], "kept": 0.57, "keeps": {5: 0.5},
                        "origin_widths": {5: 6}, "sens": {5: 0.1}, "held": 0, "policy": "plan_agent/policy_latest.pt",
                        "residual": "auto", "prior": {"kind": "sens_cost", "alpha": 0.5, "kept": 0.57, "widths": {5: 3},
                                                      "weights": {5: 1.0}, "moved": 0, "max_abs": 0.0, "dist": 0.0}}

    monkeypatch.setattr(plan_agent, "plan_for_env", fake)
    env = types.SimpleNamespace(selected_net_path="net.pt", current_model=ZOO["thin_r20_w4"]().eval(),
                                conf=types.SimpleNamespace(device="cpu"), target_keep=0.6, train_loader=None)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    cache = alloc_walk._state(env)
    assert cache["widths"] == {5: 3} and calls == [{}]
    assert lines and lines[0].startswith("[alloc] net.pt: agent plan_agent/policy_latest.pt plan keeps x0.570")
    monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "flops")
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "flop:0.6")
    alloc_walk._state(types.SimpleNamespace(selected_net_path="other.pt", current_model=env.current_model,
                                            conf=env.conf, target_keep=0.6, train_loader=None))
    assert calls[-1] == {"budget": "flops", "kappa": 0.6}
