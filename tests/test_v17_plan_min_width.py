"""
v17 ``SPECTRA_PLAN_MIN_WIDTH``: one width floor for every plan decoder (``plan_agent.decode`` / ``scale_decode`` /
``polish`` / ``widths_of``, ``NetInstance.reference``, ``plan_for_env``, ``alloc_walk.plan_targets``), so the trainer
rewards and the eval decodes only plans the alloc walk can realize: its legal mask refuses every cut of a group at or
below ``fortify.min_width_for_prune`` (2 by default). Ledger §350–§351: T0-F's plans on r56-w4 put six groups at width
1; the walk floored them at 2, declared the plan complete 5 % of the FLOPs above its budget and the stall fallback cut
a residual stream to the chance floor. Default off: with the flag unset every decoder, log line, record and checkpoint
is as in tree_v16.

CPU only, no datasets.  python -m pytest tests/test_v17_plan_min_width.py -v
"""

import os
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import LAYOUT, RATES, SHAPE, ZOO, _instance, _plan, _randomize_norms, toy_state  # noqa: E402
from tests.test_v15_flops_budget import POLICY, _alloc_env, _run_trainer  # noqa: E402
from tests.test_v16_sens_cost import _batches  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, fortify, plan_agent, plan_trainer, state_dump  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_MIN_WIDTH")


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


def _floor_ok(widths, widths0, floor):
    return all(widths[row] >= min(floor, w0) for row, w0 in widths0.items())


def _walked(w0, target):
    """The width the alloc walk reaches from ``w0`` heading for ``target`` under the eval's legal mask (fortify on,
    a non-stem row): ``choose`` plays the legal cut closest to the target until none is legal or none gets closer."""
    width = int(w0)
    for _ in range(12):
        legal = fortify.legal_action_mask(RATES, row_index=1, alive_count=width, device="cpu")
        legal_idx = [i for i in range(len(RATES)) if bool(legal[i])]
        pick = alloc_walk.choose(width, int(target), RATES, legal_idx, 0)
        if pick == 0:
            break
        width = pruning.target_width(width, RATES[pick])
    return width


# ------------------------------------------------------------------ the flag


def test_plan_min_width_flag_default_off_walk_alias_clamp_and_errors(monkeypatch):
    assert fortify.plan_min_width() == 1 and plan_trainer.min_width() == 1
    for raw, want in (("", 1), ("1", 1), ("0", 1), ("-4", 1), ("2", 2), (" 3 ", 3)):
        monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", raw)
        assert fortify.plan_min_width() == want and plan_trainer.min_width() == want
    monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "walk")
    assert fortify.min_width_for_prune() == 2 and fortify.plan_min_width() == 2
    monkeypatch.setenv("SPECTRA_MIN_WIDTH_FOR_PRUNE", "3")
    assert fortify.plan_min_width() == 3
    monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "WALK")
    assert fortify.plan_min_width() == 3
    monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "two")
    with pytest.raises(ValueError):
        fortify.plan_min_width()


# ------------------------------------------------------------------ the decoders


def test_widths_of_and_polish_honour_the_floor_and_default_to_the_old_values():
    widths0 = {0: 16, 5: 32, 9: 7, 11: 1, 13: 2}
    keeps = {0: 1.0, 5: 0.34, 9: 0.0, 11: 0.4, 13: 0.3}
    assert plan_agent.widths_of(keeps, widths0) == {0: 16, 5: 11, 9: 1, 11: 1, 13: 1}
    assert plan_agent.widths_of(keeps, widths0, 1) == plan_agent.widths_of(keeps, widths0)
    assert plan_agent.widths_of(keeps, widths0, 2) == {0: 16, 5: 11, 9: 2, 11: 1, 13: 2}
    assert plan_agent.widths_of(keeps, widths0, 3) == {0: 16, 5: 11, 9: 3, 11: 1, 13: 2}
    model = ZOO["thin_r20_w4"]().eval()
    pm = plan_agent.ParamModel(model, _plan(model))
    low = {row: 0.1 for row in pm.rows}
    target = 0.0                                                     # below any plan: nothing may move down
    old, p_old = plan_agent.polish(low, plan_agent.widths_of(low, pm.widths0), pm, target, 0.1)
    assert (old, p_old) == plan_agent.polish(low, plan_agent.widths_of(low, pm.widths0), pm, target, 0.1, min_width=1)
    assert old == plan_agent.widths_of(low, pm.widths0) and any(w == 1 for w in old.values())
    new, p_new = plan_agent.polish(low, plan_agent.widths_of(low, pm.widths0, 2), pm, target, 0.1, min_width=2)
    assert new == plan_agent.widths_of(low, pm.widths0, 2) and _floor_ok(new, pm.widths0, 2)
    assert p_new == pm.params(new) > p_old == pm.params(old)
    # a plan above the target still comes down, but never through the floor
    high = {row: 0.9 for row in pm.rows}
    down, _p = plan_agent.polish(high, plan_agent.widths_of(high, pm.widths0, 2), pm, 0.3 * pm.total0, 0.1, min_width=2)
    assert _floor_ok(down, pm.widths0, 2) and any(down[row] < plan_agent.widths_of(high, pm.widths0)[row] for row in pm.rows)


@pytest.mark.parametrize("budget", ["params", "flops"])
def test_decode_floors_every_group_meets_the_budget_and_the_walk_can_realize_it(budget):
    model = _model()
    plan = _plan(model)
    pm = plan_agent.ParamModel(model, plan) if budget == "params" else plan_agent.FlopModel(model, plan, SHAPE)
    z = [-10.0 if pm.widths0[row] <= 8 else 0.0 for row in pm.rows]  # the 4- and 8-wide groups at the k_min floor
    kappa = 0.4
    old, old_info = plan_agent.decode(z, pm, kappa)
    assert (old, old_info) == plan_agent.decode(z, pm, kappa, min_width=1)
    ones = [row for row in pm.rows if old[row] == 1]
    assert ones                                                       # the old decoder plans width-1 groups ...
    assert all(_walked(pm.widths0[row], 1) == 2 for row in ones)     # ... which the walk floors at 2
    tol = 0.1 if budget == "flops" else 0.05                         # one stage-1 channel is ~10 % of this net's MACs
    new, info = plan_agent.decode(z, pm, kappa, min_width=2)
    assert _floor_ok(new, pm.widths0, 2) and abs(info["kept"] - kappa) < tol and new != old
    assert info["kept"] == pytest.approx(pm.kept(new)) and set(new) == set(pm.rows)
    assert all(new[row] == 2 for row in ones)                         # lifted to the floor, not beyond
    real = alloc_walk.cut_to(model, plan, plan_agent.rates_of(new, pm.widths0), SHAPE)
    assert alloc_walk.group_widths(real, pm.rows) == new
    del real
    for row, w0 in pm.widths0.items():
        walked = _walked(w0, new[row])
        assert walked == new[row] or walked == new[row] + 1           # +1: a target no single rate reaches (16 -> 15)
    three, info3 = plan_agent.decode(z, pm, kappa, min_width=3)
    assert _floor_ok(three, pm.widths0, 3) and abs(info3["kept"] - kappa) < tol + 0.02


def test_scale_decode_floor_keeps_held_groups_whole_and_defaults_to_the_old_plan():
    model = _model()
    plan = _plan(model)
    pm = plan_agent.ParamModel(model, plan)
    held = {row for group, row in plan if len(group.producers) > 1}
    uniform = {row: 1.0 for row in pm.rows}
    old, old_info = plan_agent.scale_decode(uniform, pm, 0.35, held=held)
    assert (old, old_info) == plan_agent.scale_decode(uniform, pm, 0.35, held=held, min_width=1)
    assert any(old[row] == 1 for row in pm.rows)                      # the inner family cuts a 4-wide group to 1
    new, info = plan_agent.scale_decode(uniform, pm, 0.35, held=held, min_width=2)
    assert _floor_ok(new, pm.widths0, 2) and all(new[row] == pm.widths0[row] for row in held) and new != old
    assert info["kept"] == pytest.approx(pm.kept(new)) and abs(info["kept"] - 0.35) < 0.06
    assert all(info["keeps"][row] == 1.0 for row in held)


def test_instance_reference_passes_the_floor(v10_env):
    inst = _instance()
    changed = 0
    for kind in ("uniform", "sens", "inner"):
        w1, i1 = inst.reference(kind, 0.35)
        assert (w1, i1) == inst.reference(kind, 0.35, min_width=1)
        w2, i2 = inst.reference(kind, 0.35, min_width=2)
        assert _floor_ok(w2, inst.pm.widths0, 2) and i2["kept"] == pytest.approx(inst.pm.kept(w2))
        changed += w1 != w2
    assert changed
    w3, _i3 = inst.reference("inner", 0.35, budget="flops", min_width=2)
    assert _floor_ok(w3, inst.pm.widths0, 2) and all(w3[row] == inst.pm.widths0[row] for row in inst.held)


# ------------------------------------------------------------------ trainer


def test_trainer_reads_the_floor_records_it_and_is_silent_when_off(monkeypatch, tmp_path, v10_env):
    inst = _instance()
    on = tmp_path / "on"
    on.mkdir()
    monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "2")
    lines, records, blob = _run_trainer(monkeypatch, on, inst, "params")
    assert blob["min_width"] == 2 and "min_width" not in blob["config"]
    trainer_lines = [line for line in lines if line.startswith("[plan] trainer {")]
    assert len(trainer_lines) == 1 and trainer_lines[0].endswith("} min_width=2")
    summaries = [r for r in records if "summary" in r]
    assert len(summaries) == 3
    assert all(int(w) >= min(2, inst.pm.widths0[int(r)]) for s in summaries for r, w in s["widths"].items())
    monkeypatch.delenv("SPECTRA_PLAN_MIN_WIDTH")
    off = tmp_path / "off"
    off.mkdir()
    lines_off, records_off, blob_off = _run_trainer(monkeypatch, off, _instance(), "params")
    assert "min_width" not in blob_off and not any("min_width" in line for line in lines_off)
    monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "1")                       # 1 is off, byte for byte
    one = tmp_path / "one"
    one.mkdir()
    lines_one, records_one, blob_one = _run_trainer(monkeypatch, one, _instance(), "params")
    assert "min_width" not in blob_one and lines_one[0] == lines_off[0]
    strip = lambda rs: [{k: v for k, v in r.items() if k != "seconds"} for r in rs]  # noqa: E731
    assert strip(records_one) == strip(records_off)
    assert [r["widths"] for r in records_one if "summary" in r] == [r["widths"] for r in records_off if "summary" in r]


# ------------------------------------------------------------------ plan_for_env and the alloc walk


def _policy_with_a_width_one_plan(pm, state, rows, target):
    """A random head whose unfloored plan at ``target`` puts some group at width 1 (so the floor is exercised)."""
    mask, k = plan_agent.token_index(rows, pm.rows, "cpu")
    prepared = plan_agent.plan_state(state, plan_agent.spans_of(LAYOUT), 0.6, ("sens",))
    for seed in range(40):
        torch.manual_seed(seed)
        policy = plan_agent.PlanPolicy(63)
        with torch.no_grad():
            policy.head.weight.normal_(std=3.0)
            mu = policy.eval()(prepared, mask, k, len(pm.rows)).tolist()
        widths, _info = plan_agent.decode(mu, pm, target)
        if any(w == 1 for w in widths.values()):
            return policy, mu
    pytest.fail("no random head planned a width-1 group")


def test_plan_for_env_takes_the_floor_from_the_checkpoint_then_from_the_env(monkeypatch, tmp_path, v10_env):
    torch.manual_seed(0)
    model = ZOO["thin_r20_w4"]().eval()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    plan = group_sensitivity.group_plan(mwr, groups)
    rows = state_dump.token_rows(mwr, groups, plan)
    state = toy_state(len(rows))
    monkeypatch.setattr(state_dump, "encode_origin", lambda env, m, g: dict(state))
    pm = plan_agent.ParamModel(model, plan)
    policy, mu = _policy_with_a_width_one_plan(pm, state, rows, 0.4)
    env = types.SimpleNamespace(conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES),
                                current_model=model, target_keep=0.6,
                                _dependency_groups=lambda m: channel_groups.build_channel_groups(m.model))
    plain = tmp_path / "plan_agent" / "policy_plain.pt"
    plan_agent.save_policy(policy, str(plain), tokens="layer", zero=["sens"], k_min=0.1)
    floored = tmp_path / "plan_agent" / "policy_floored.pt"
    plan_agent.save_policy(policy, str(floored), tokens="layer", zero=["sens"], k_min=0.1, min_width=2)
    w0, i0 = plan_agent.plan_for_env(env, 0.4, str(plain))
    assert w0 == plan_agent.decode(mu, pm, 0.4)[0] and "min_width" not in i0 and any(w == 1 for w in w0.values())
    w2, i2 = plan_agent.plan_for_env(env, 0.4, str(floored))
    assert w2 == plan_agent.decode(mu, pm, 0.4, min_width=2)[0] and i2["min_width"] == 2 and _floor_ok(w2, pm.widths0, 2)
    assert i2["kept"] == pytest.approx(pm.kept(w2)) and i2["keeps"] == {row: w2[row] / pm.widths0[row] for row in pm.rows}
    monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "3")                       # the env flag wins over the checkpoint
    w3, i3 = plan_agent.plan_for_env(env, 0.4, str(floored))
    assert w3 == plan_agent.decode(mu, pm, 0.4, min_width=3)[0] and i3["min_width"] == 3 and _floor_ok(w3, pm.widths0, 3)
    w3p, i3p = plan_agent.plan_for_env(env, 0.4, str(plain))
    assert w3p == w3 and i3p["min_width"] == 3
    monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "1")                       # 1 is off: the checkpoint's floor stands
    assert plan_for_env_widths(env, floored) == w2 and plan_for_env_widths(env, plain) == w0
    ws, si = plan_agent.plan_for_env(env, 0.4, str(floored), sample=(0.2, 3))
    assert si["min_width"] == 2 and _floor_ok(ws, pm.widths0, 2) and si["kind"] == "agent_sample"


def plan_for_env_widths(env, path):
    return plan_agent.plan_for_env(env, 0.4, str(path))[0]


def _fake_agent(calls):
    def fake(env, target, path, k_min, **kw):
        calls.append((target, path, k_min, kw))
        return {5: 3}, {"kind": "agent", "alpha": 0.0, "target": target, "kept": 0.57, "keeps": {5: 0.5},
                        "origin_widths": {5: 6}, "sens": {5: 0.1}, "held": 0, "policy": "plan_agent/policy_latest.pt"}
    return fake


def test_alloc_state_passes_the_floor_to_the_agent_and_the_heuristics_only_when_set(monkeypatch):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "agent")
    monkeypatch.setenv("SPECTRA_PLAN_AGENT", POLICY)
    calls, lines, recorded = [], [], []
    monkeypatch.setattr(plan_agent, "plan_for_env", _fake_agent(calls))
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda *a, **k: recorded.append(k))
    alloc_walk._state(_alloc_env())
    assert calls[-1] == (pytest.approx(0.58), POLICY, 0.1, {}) and "width floor" not in lines[-1]
    assert "min_width" not in recorded[-1]
    monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "2")
    alloc_walk._state(_alloc_env())
    assert calls[-1] == (pytest.approx(0.58), POLICY, 0.1, {"min_width": 2})
    assert lines[-1].endswith("; group width floor 2") and recorded[-1]["min_width"] == 2
    monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "flops")
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "flop:0.6")
    alloc_walk._state(_alloc_env())
    assert calls[-1] == (pytest.approx(0.58), POLICY, 0.1, {"budget": "flops", "kappa": 0.6, "min_width": 2})
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "uniform")
    planned = []
    monkeypatch.setattr(alloc_walk, "plan_targets",
                        lambda *a, **k: planned.append(k) or _fake_agent([])(None, a[4], "", 0.1))
    monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: [])
    env = _alloc_env()
    env._input_shape = lambda: SHAPE
    alloc_walk._state(env)
    assert planned[-1] == {"budget": "flops", "min_width": 2}
    monkeypatch.delenv("SPECTRA_PLAN_MIN_WIDTH")
    monkeypatch.delenv("SPECTRA_ALLOC_BUDGET")
    env = _alloc_env()
    env._input_shape = lambda: SHAPE
    alloc_walk._state(env)
    assert planned[-1] == {} and "width floor" not in lines[-1] and "min_width" not in recorded[-1]


@pytest.mark.parametrize("budget", ["params", "flops"])
def test_plan_targets_floor_under_both_budgets_and_default_unchanged(budget):
    model = _model()
    plan = _plan(model)
    rows = [row for _g, row in plan]
    size = (lambda m: utils.calc_flops(m, SHAPE)) if budget == "flops" else utils.calc_num_parameters
    size0 = size(model)
    lifted = 0
    for kind_name, batches in (("uniform", []), ("inner", []), ("sens", _batches())):
        w1, i1 = alloc_walk.plan_targets(model, batches, SHAPE, kind_name, 0.3, budget=budget)
        w1b, i1b = alloc_walk.plan_targets(model, batches, SHAPE, kind_name, 0.3, budget=budget, min_width=1)
        assert w1 == w1b and i1["keeps"] == i1b["keeps"] and i1["kept"] == i1b["kept"]
        w2, i2 = alloc_walk.plan_targets(model, batches, SHAPE, kind_name, 0.3, budget=budget, min_width=2)
        origin = i2["origin_widths"]
        assert _floor_ok(w2, origin, 2) and set(w2) == set(rows)
        assert all(i2["keeps"][row] >= min(1.0, 2.0 / origin[row]) - 1e-12 for row in rows)
        assert 0.2 < i2["kept"] < 0.5 and i2["held"] == i1["held"]
        cut = alloc_walk.cut_to(model, plan, i2["keeps"], SHAPE)
        assert alloc_walk.group_widths(cut, rows) == w2 and i2["kept"] == pytest.approx(size(cut) / size0)
        del cut
        lifted += any(w1[row] < 2 <= w2[row] for row in rows)
    assert lifted >= 1                                                # at 0.3 the uniform plan cuts a 4-wide group to 1
