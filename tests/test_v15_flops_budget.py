"""
v15 T0-F: the plan-as-action agent under a FLOPs budget. ``plan_agent.FlopModel`` prices a plan in MACs the way
``ParamModel`` prices it in parameters; ``SPECTRA_PLAN_BUDGET`` trains on it, ``SPECTRA_ALLOC_BUDGET`` walks on it,
and ``SPECTRA_FIXED_TARGET_METRIC=flop`` lands and ends the eval walk on kept MACs. Every knob defaults off: with
nothing set the params paths are the ones ``tests/test_plan_agent.py`` and ``tests/test_v10_fixed_target.py`` pin.

CPU only, no datasets.  python -m pytest tests/test_v15_flops_budget.py -v
"""

import json
import os
import random
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf, _row_index_of_layer  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import (LAYOUT, RATES, SHAPE, V10_FLAGS, ZOO, _instance, _plan,  # noqa: E402
                                   _randomize_norms, toy_state)
from tests.test_v10_fixed_target import _layer_index, _target_env  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, fortify, plan_agent, plan_trainer, state_dump  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402
from src.NetworkEnv import NetworkEnv, EVAL_TEST  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_")
POLICY = "/x/runs/job1/plan_agent/policy_latest.pt"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


@pytest.fixture
def v10_env(monkeypatch):
    for key, value in V10_FLAGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SPECTRA_STATE_TOKENS", raising=False)


def _random_widths(cm, seed):
    """One width per planned group, uniform in [max(1, round(0.1 w0)), w0]."""
    rng = random.Random(seed)
    return {row: rng.randint(max(1, round(0.1 * w0)), w0) for row, w0 in cm.widths0.items()}


# ------------------------------------------------------------------ FlopModel


@pytest.mark.parametrize("name", list(ZOO))
def test_flop_model_matches_calc_flops_of_the_real_cut(name):
    torch.manual_seed(0)
    model = ZOO[name]().eval()
    _randomize_norms(model)
    plan = _plan(model)
    fm = plan_agent.FlopModel(model, plan, SHAPE)
    assert fm.rows == fm.pm.rows and fm.widths0 == fm.pm.widths0 and not model.training
    assert fm.total0 == utils.calc_flops(model, SHAPE) and fm.struct(fm.widths0) == fm.struct0 < fm.total0
    assert fm.probes == sum(w0 > 1 for w0 in fm.widths0.values()) and fm.seconds > 0
    assert fm.cost(fm.widths0) == pytest.approx(fm.total0, rel=1e-12) and fm.kept(fm.widths0) == pytest.approx(1.0)
    uniform, _info = plan_agent.scale_decode({row: 1.0 for row in fm.rows}, fm, 0.6)
    for widths in [uniform] + [_random_widths(fm, seed) for seed in (1, 2, 3)]:
        rates = plan_agent.rates_of(widths, fm.widths0)
        real = alloc_walk.cut_to(model, plan, rates, SHAPE).eval()
        assert alloc_walk.group_widths(real, fm.rows) == widths
        assert fm.cost(widths) == pytest.approx(utils.calc_flops(real, SHAPE), rel=1e-9)
        assert fm.params(widths) == utils.calc_num_parameters(real)
        assert fm.kept(widths) == pytest.approx(fm.cost(widths) / fm.total0)
        del real


def test_cost_is_params_on_the_param_model_and_decode_there_is_unchanged():
    torch.manual_seed(0)
    model = ZOO["thin_r20_w4"]().eval()
    pm = plan_agent.ParamModel(model, _plan(model))
    for seed in (1, 2, 3):
        widths = _random_widths(pm, seed)
        assert pm.cost(widths) == pm.params(widths) and pm.kept(widths) == pm.params(widths) / pm.total0
    z = [0.3 * k for k in range(len(pm.rows))]
    widths, info = plan_agent.decode(z, pm, 0.6)
    assert info["kept"] == pm.params(widths) / pm.total0 and abs(info["kept"] - 0.6) < 0.02
    keeps = [info["keeps"][row] for row in pm.rows]
    assert all(a <= b + 1e-12 for a, b in zip(keeps, keeps[1:])) and keeps[0] < keeps[-1]
    assert plan_agent.decode([v + 5.0 for v in z], pm, 0.6)[0] == widths
    rounded = plan_agent.widths_of({row: 0.62 for row in pm.rows}, pm.widths0)
    polished, p = plan_agent.polish({row: 0.62 for row in pm.rows}, rounded, pm, 0.6 * pm.total0, 0.1)
    assert p == pm.params(polished) and abs(p - 0.6 * pm.total0) <= abs(pm.params(rounded) - 0.6 * pm.total0)


@pytest.mark.parametrize("name", ["r20_chenyaofo", "vgg11_bn"])
def test_decode_and_scale_decode_on_the_flop_model_land_the_kept_flops(name):
    torch.manual_seed(0)
    model = ZOO[name]().eval()
    plan = _plan(model)
    fm = plan_agent.FlopModel(model, plan, SHAPE)
    held = {row for group, row in plan if len(group.producers) > 1}
    z = [0.3 * k for k in range(len(fm.rows))]
    for kappa in (0.6, 0.45):
        widths, info = plan_agent.decode(z, fm, kappa)
        assert abs(info["kept"] - kappa) < 0.02 and min(info["keeps"].values()) >= 0.1 - 1e-12
        real = alloc_walk.cut_to(model, plan, plan_agent.rates_of(widths, fm.widths0), SHAPE)
        assert utils.calc_flops(real, SHAPE) / fm.total0 == pytest.approx(info["kept"], rel=1e-9)
        del real
        s_widths, s_info = plan_agent.scale_decode({row: 1.0 for row in fm.rows}, fm, kappa, held=held)
        assert abs(s_info["kept"] - kappa) < 0.02 and all(s_widths[row] == fm.widths0[row] for row in held)
        assert s_info["kept"] == pytest.approx(fm.kept(s_widths))


# ------------------------------------------------------------------ state and instance


def test_plan_state_flags_the_flops_budget_in_the_first_action_slot(v10_env):
    spans = plan_agent.spans_of(LAYOUT)
    state = toy_state(6)
    base = plan_agent.plan_state(state, spans, 0.45, zero=("sens",))["layer_features"]
    same = plan_agent.plan_state(state, spans, 0.45, zero=("sens",), budget="params")["layer_features"]
    assert torch.equal(same, base)
    flops = plan_agent.plan_state(state, spans, 0.45, zero=("sens",), budget="flops")["layer_features"]
    assert torch.all(flops[:, 53] == 1.0) and torch.count_nonzero(flops[:, 54:63]) == 0
    assert torch.equal(flops[:, :53], base[:, :53]) and torch.count_nonzero(base[:, 53:63]) == 0
    with pytest.raises(ValueError):
        plan_agent.plan_state(state, spans, 0.45, budget="macs")


def test_instance_builds_the_flop_model_lazily_and_check_flops_agrees_with_the_real_cut(v10_env):
    inst = _instance()
    assert inst.fm is None and inst.cost_model("params") is inst.pm and inst.fm is None
    chk = inst.check_flops(0.6)
    assert inst.fm is not None and inst.cost_model("flops") is inst.fm
    assert chk["analytic"] == pytest.approx(chk["real"], abs=1e-9)
    assert chk["probes"] == inst.fm.probes > 0 and chk["seconds"] == inst.fm.seconds > 0
    _w, inner = inst.reference("inner", 0.6, budget="flops")
    assert all(inner["keeps"][row] == 1.0 for row in inst.held) and abs(inner["kept"] - 0.6) < 0.04
    assert inner["kept"] != pytest.approx(inst.reference("inner", 0.6)[1]["kept"], abs=1e-6)
    feats = inst.state_at(0.5, "flops")["layer_features"]
    assert torch.all(feats[:, 53] == 1.0) and torch.count_nonzero(inst.state_at(0.5)["layer_features"][:, 53:63]) == 0
    with pytest.raises(ValueError):
        inst.cost_model("macs")


# ------------------------------------------------------------------ trainer


def _run_trainer(monkeypatch, tmp_path, inst, budget, instances=5):
    monkeypatch.setattr(plan_agent.NetInstance, "from_env", classmethod(lambda cls, env, p, m, l, **kw: inst))
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


def test_trainer_config_reads_and_validates_the_budget(monkeypatch):
    assert plan_trainer.config().budget == "params"
    monkeypatch.setenv("SPECTRA_PLAN_BUDGET", "Mixed")
    assert plan_trainer.config().budget == "mixed"
    monkeypatch.setenv("SPECTRA_PLAN_BUDGET", "macs")
    with pytest.raises(ValueError):
        plan_trainer.config()


def test_trainer_runs_end_to_end_under_a_flops_budget(monkeypatch, tmp_path, v10_env):
    inst = _instance()
    lines, records, blob = _run_trainer(monkeypatch, tmp_path, inst, "flops")
    steps = [r for r in records if "it" in r]
    assert len(steps) == 5 and all(r["budget"] == "flops" and len(r["rewards"]) == 3 for r in steps)
    assert blob["budget"] == "flops" and blob["config"]["budget"] == "flops" and blob["nets"] == ["toy.pt"]
    assert inst.fm is not None and inst.fm.probes > 0
    check = [line for line in lines if line.startswith("[plan] check-flops toy.pt: uniform k=0.60 FLOPs analytic x")]
    assert len(check) == 1 and "WARNING" not in check[0] and f"| {inst.fm.probes} probes in " in check[0]
    assert sum(line.startswith("[plan] check toy.pt:") for line in lines) == 1
    it_lines = [line for line in lines if line.startswith("[plan] it=")]
    assert len(it_lines) == 5 and all(" b=flops sigma=" in line for line in it_lines)
    assert all(line.startswith(f"[plan] it={r['it'] + 1}/5 toy.pt k={r['kappa']:.3f} b=flops sigma=")
               for r, line in zip(steps, it_lines))
    summaries = [r for r in records if "summary" in r]
    assert len(summaries) == 3 and all(r["budget"] == "flops" for r in summaries)
    assert all(" b=flops: mean plan " in line for line in lines if line.startswith("[plan] summary "))
    assert any(line.startswith("[plan] DONE 5 instances") for line in lines)


def test_trainer_mixed_budget_draws_both_and_tags_only_flops(monkeypatch, tmp_path, v10_env):
    inst = _instance()
    lines, records, blob = _run_trainer(monkeypatch, tmp_path, inst, "mixed", instances=8)
    steps = [r for r in records if "it" in r]
    assert len(steps) == 8 and {r["budget"] for r in steps} == {"params", "flops"}
    assert blob["budget"] == "mixed" and blob["config"]["budget"] == "mixed"
    it_lines = [line for line in lines if line.startswith("[plan] it=")]
    assert len(it_lines) == 8
    for r, line in zip(steps, it_lines):
        tag = " b=flops" if r["budget"] == "flops" else ""
        assert line.startswith(f"[plan] it={r['it'] + 1}/8 toy.pt k={r['kappa']:.3f}{tag} sigma=")
    summaries = [r for r in records if "summary" in r]
    assert len(summaries) == 6 and {r["budget"] for r in summaries} == {"params", "flops"}
    summary_lines = [line for line in lines if line.startswith("[plan] summary ")]
    assert len(summary_lines) == 6 and sum(" b=flops: " in line for line in summary_lines) == 3
    assert sum(line.startswith("[plan] check-flops toy.pt:") for line in lines) == 1


def test_trainer_params_budget_logs_and_records_as_before(monkeypatch, tmp_path, v10_env):
    inst = _instance()
    lines, records, blob = _run_trainer(monkeypatch, tmp_path, inst, "params")
    steps = [r for r in records if "it" in r]
    assert all(r["budget"] == "params" for r in steps) and blob["budget"] == "params" and inst.fm is None
    assert not any("b=" in line for line in lines if line.startswith(("[plan] it=", "[plan] summary ")))
    assert not any(line.startswith("[plan] check-flops") for line in lines)
    assert len([r for r in records if "summary" in r]) == 3


# ------------------------------------------------------------------ alloc walk


def test_alloc_budget_default_and_parse(monkeypatch):
    assert alloc_walk.budget() == "params"
    monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "FLOPs")
    assert alloc_walk.budget() == "flops"
    monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "macs")
    with pytest.raises(ValueError):
        alloc_walk.budget()


@pytest.mark.parametrize("kind_name", ["uniform", "inner"])
def test_plan_targets_under_a_flops_budget_bisects_on_kept_flops(monkeypatch, kind_name):
    torch.manual_seed(0)
    model = ZOO["thin_r20_w4"]().eval()
    monkeypatch.setattr(group_sensitivity, "group_sensitivity",
                        lambda *a, **k: pytest.fail("uniform / inner must not measure sensitivity"))
    plan = _plan(model)
    flops0 = utils.calc_flops(model, SHAPE)
    widths, info = alloc_walk.plan_targets(model, [], SHAPE, kind_name, 0.58, budget="flops")
    cut = alloc_walk.cut_to(model, plan, info["keeps"], SHAPE)
    tol = 0.05 if kind_name == "inner" else 0.04                                    # groups of 4-16: coarse steps
    assert info["kept"] == pytest.approx(utils.calc_flops(cut, SHAPE) / flops0)   # kept is a FLOPs fraction
    assert info["kept"] == pytest.approx(0.58, abs=tol)
    assert alloc_walk.group_widths(cut, list(widths)) == widths
    assert info["held"] == (3 if kind_name == "inner" else 0)
    del cut
    _w, params = alloc_walk.plan_targets(model, [], SHAPE, kind_name, 0.58)          # the default: still params
    cut = alloc_walk.cut_to(model, plan, params["keeps"], SHAPE)
    assert params["kept"] == pytest.approx(utils.calc_num_parameters(cut) / utils.calc_num_parameters(model))
    assert params["kept"] == pytest.approx(0.58, abs=tol)


def test_flops_walk_target_prefers_the_flop_fixed_target_then_the_flop_size_match(monkeypatch):
    env = types.SimpleNamespace(target_keep=0.7)
    assert alloc_walk.flops_walk_target(env) == pytest.approx(0.6)             # a params target is not a FLOPs one
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "flop:0.8,0.5")
    assert alloc_walk.flops_walk_target(env) == pytest.approx(0.5)
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "flop:0.55")
    assert alloc_walk.flops_walk_target(env) == pytest.approx(0.55)
    monkeypatch.setenv("SPECTRA_FIXED_TARGET_METRIC", "flop")
    assert alloc_walk.flops_walk_target(env) == pytest.approx(0.7)
    assert alloc_walk.flops_walk_target(types.SimpleNamespace(target_keep=None)) == pytest.approx(0.55)


def _fake_agent(calls, kind="agent"):
    def fake(env, target, path, k_min, sample=None, budget="params", kappa=None):
        calls.append((target, path, k_min, sample, budget, kappa))
        return {5: 3}, {"kind": kind, "alpha": 0.0, "target": target, "kept": 0.57, "keeps": {5: 0.5},
                        "origin_widths": {5: 6}, "sens": {5: 0.1}, "held": 0, "policy": "plan_agent/policy_latest.pt",
                        "budget": budget, **({"sample": {"around": "agent", "sigma": 0.2, "seed": 7, "dist": 0.5}}
                                             if sample else {})}
    return fake


def _alloc_env(target_keep=0.6):
    return types.SimpleNamespace(selected_net_path="net.pt", current_model=ZOO["thin_r20_w4"]().eval(),
                                 conf=types.SimpleNamespace(device="cpu"), target_keep=target_keep, train_loader=None)


def test_alloc_state_under_a_flops_budget_plans_the_agent_on_flops(monkeypatch):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "agent")
    monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "flops")
    monkeypatch.setenv("SPECTRA_PLAN_AGENT", POLICY)
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "flop:0.6")
    calls, lines = [], []
    monkeypatch.setattr(plan_agent, "plan_for_env", _fake_agent(calls))
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    cache = alloc_walk._state(_alloc_env(target_keep=0.7))              # 0.7 is a params target: ignored
    assert cache["widths"] == {5: 3}
    assert calls == [(pytest.approx(0.58), POLICY, 0.1, None, "flops", 0.6)]
    assert "agent plan_agent/policy_latest.pt plan keeps x0.570 of the FLOPs (target x0.580 = walk target" in lines[0]
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "agent_sample")
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SIGMA", "0.2")
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SEED", "7")
    monkeypatch.setenv("SPECTRA_FIXED_TARGET_METRIC", "flop")
    monkeypatch.setattr(plan_agent, "plan_for_env", _fake_agent(calls, "agent_sample"))
    alloc_walk._state(_alloc_env(target_keep=0.7))                      # now a FLOPs target: it wins
    assert calls[-1] == (pytest.approx(0.68), POLICY, 0.1, (0.2, 7), "flops", 0.7)
    assert "agent_sample sigma=0.2 seed=7 around the mean of plan_agent/policy_latest.pt plan keeps x0.570 of the " \
           "FLOPs (target x0.680" in lines[-1]


def test_alloc_state_without_the_budget_flag_calls_the_params_paths_as_before(monkeypatch):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "agent")
    monkeypatch.setenv("SPECTRA_PLAN_AGENT", POLICY)
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "flop:0.5")
    calls, lines = [], []

    def fake(env, target, path, k_min):
        calls.append((target, path, k_min))
        return {5: 3}, {"kind": "agent", "alpha": 0.0, "target": target, "kept": 0.57, "keeps": {5: 0.5},
                        "origin_widths": {5: 6}, "sens": {5: 0.1}, "held": 0, "policy": "plan_agent/policy_latest.pt"}

    monkeypatch.setattr(plan_agent, "plan_for_env", fake)
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    assert alloc_walk._state(_alloc_env())["widths"] == {5: 3}
    assert calls == [(pytest.approx(0.58), POLICY, 0.1)]
    assert " plan keeps x0.570 of the params (target x0.580 = walk target" in lines[0]
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "uniform")
    planned = []
    monkeypatch.setattr(alloc_walk, "plan_targets", lambda *a, **k: planned.append((a, k)) or fake(None, a[4], "", 0.1))
    monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: [])
    env = _alloc_env()
    env._input_shape = lambda: SHAPE
    alloc_walk._state(env)
    assert planned[0][1] == {} and planned[0][0][3:] == ("uniform", pytest.approx(0.58), 0.5, 0.1)


def test_alloc_action_tracks_kept_flops_under_the_flops_budget(monkeypatch):
    monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "flops")
    env = _alloc_env()
    env.row_idx = 1
    env.flops_ratio = lambda: 0.9
    env.param_ratio = lambda: pytest.fail("the flops walk must not read the params ratio")
    env._alloc_walk = {"net.pt": {"widths": {}, "n_rows": 99, "idle": 0, "fallback": False, "last_kept": 1.0}}
    legal = torch.ones(len(RATES), dtype=torch.bool)
    assert int(alloc_walk.action(env, legal, RATES, "cpu").item()) == 0
    assert env._alloc_walk["net.pt"]["last_kept"] == pytest.approx(0.9)


def test_plan_for_env_decodes_on_the_flop_model_under_a_flops_budget(monkeypatch, tmp_path, v10_env):
    torch.manual_seed(0)
    model = ZOO["thin_r20_w4"]().eval()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    plan = group_sensitivity.group_plan(mwr, groups)
    rows = state_dump.token_rows(mwr, groups, plan)
    state = toy_state(len(rows))
    monkeypatch.setattr(state_dump, "encode_origin", lambda env, m, g: dict(state))
    policy = plan_agent.PlanPolicy(63)
    with torch.no_grad():
        policy.head.weight.normal_()
    path = tmp_path / "plan_agent" / "policy_it00300.pt"
    plan_agent.save_policy(policy, str(path), tokens="layer", zero=["sens"], k_min=0.1, budget="flops")
    env = types.SimpleNamespace(conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES),
                                current_model=model, target_keep=0.6, _input_shape=lambda: SHAPE,
                                _dependency_groups=lambda m: channel_groups.build_channel_groups(m.model))
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    widths, info = plan_agent.plan_for_env(env, 0.58, str(path), budget="flops", kappa=0.6)
    fm = plan_agent.FlopModel(model, plan, SHAPE)
    assert info["budget"] == "flops" and info["kind"] == "agent" and set(widths) == set(fm.rows)
    # One channel of this net's stage-1 stream is ~10 % of its MACs, so a random head's plan can miss by up to ~0.05.
    assert abs(info["kept"] - 0.58) < 0.05 and info["kept"] == pytest.approx(fm.kept(widths))
    real = alloc_walk.cut_to(model, plan, plan_agent.rates_of(widths, fm.widths0), SHAPE)
    assert utils.calc_flops(real, SHAPE) / fm.total0 == pytest.approx(info["kept"], rel=1e-9)
    prepared = plan_agent.plan_state(state, plan_agent.spans_of(LAYOUT), 0.6, ("sens",), budget="flops")
    mask, k = plan_agent.token_index(rows, fm.rows, "cpu")
    with torch.no_grad():
        mu = policy.eval()(prepared, mask, k, len(fm.rows)).tolist()
    assert widths == plan_agent.decode(mu, fm, 0.58)[0]
    assert not any("WARNING" in line for line in lines)                 # trained under flops, decoded on flops
    p_widths, p_info = plan_agent.plan_for_env(env, 0.58, str(path))
    assert p_info["budget"] == "params" and p_info["kept"] == pytest.approx(fm.pm.kept(p_widths))
    assert any(line == "[alloc] WARNING: the plan agent trained under a flops budget; this job decodes on params"
               for line in lines)


# ------------------------------------------------------------------ env: the FLOPs fixed-target metric


def test_fixed_target_metric_default_aliases_and_errors(monkeypatch):
    assert fortify.fixed_target_metric() == "param"
    for raw, want in (("flop", "flop"), ("FLOPs", "flop"), ("macs", "flop"), ("param", "param"), ("params", "param")):
        monkeypatch.setenv("SPECTRA_FIXED_TARGET_METRIC", raw)
        assert fortify.fixed_target_metric() == want
    monkeypatch.setenv("SPECTRA_FIXED_TARGET_METRIC", "bytes")
    with pytest.raises(ValueError):
        fortify.fixed_target_metric()
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    assert "SPECTRA_FIXED_TARGET_METRIC" not in A2CAgentReinforce.POLICY_CONTRACT_KEYS
    env = NetworkEnv.__new__(NetworkEnv)
    assert env._target_metric() == "param"                              # no target: the knob is not even read
    env.target_keep = 0.5
    with pytest.raises(ValueError):
        env._target_metric()


def test_episode_target_in_eval_reads_the_entries_of_the_metric(monkeypatch):
    env = NetworkEnv.__new__(NetworkEnv)
    env.mode = EVAL_TEST
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "flop:0.55")
    assert env._episode_target() == pytest.approx(0.6) and len(lines) == 1 and "param:<keep>" in lines[-1]
    monkeypatch.setenv("SPECTRA_FIXED_TARGET_METRIC", "flop")
    assert env._episode_target() == pytest.approx(0.55) and len(lines) == 1
    assert env._episode_target(0.7) == 0.7
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "param:0.5")
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "flop:0.8,0.45")
    assert env._episode_target() == pytest.approx(0.45)
    monkeypatch.delenv("SPECTRA_EVAL_SIZE_POINTS")
    assert env._episode_target() == pytest.approx(0.6) and len(lines) == 2 and "flop:<keep>" in lines[-1]


def _flop_target_env(monkeypatch, accs):
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    monkeypatch.setenv("SPECTRA_FIXED_TARGET_METRIC", "flop")
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    env, seen = _target_env(model, target=1.0, accs=accs)
    env.original_flops = utils.calc_flops(model, SHAPE)
    mwr = ModelWithRows(model)
    env.row_idx = _row_index_of_layer(mwr, _layer_index(mwr, model.layer3[0].conv1)) + 1
    return env, seen


def test_a_cut_past_the_flop_target_lands_on_kept_macs_and_ends_the_episode(monkeypatch):
    env, seen = _flop_target_env(monkeypatch, accs=[0.80])
    assert env._target_metric() == "flop"
    half_f, half_p = env.preview_flops_ratio(0.5), env.preview_param_ratio(0.5)
    assert half_f < 1.0 and abs(half_f - half_p) > 0.01               # a stage-3 cut: unequal MAC and param shares
    target = 0.5 * (1.0 + half_f)                                      # a 0.5 cut of this layer passes it in MACs
    env.target_keep = target
    rate, landed = env._land_on_target(0.5)
    assert 0.5 < rate < 1.0 and landed <= target + 1e-9
    assert env.preview_flops_ratio(min(1.0 - 1e-6, rate + 1.0 / 1024)) > target   # the mildest such cut
    assert env._land_on_target(1.0 - 1e-6) == (1.0 - 1e-6, None)
    _, reward, done = env.step(0.5)
    kept = env.flops_ratio()
    assert done and kept <= target + 1e-9 and kept > half_f
    assert env._target_final["kept"] == pytest.approx(kept) and env.param_ratio() != pytest.approx(kept, abs=0.01)
    assert reward == pytest.approx(-10.0)                               # 0.90 -> 0.80, landed: no penalty
    assert env.episode_target_score() == pytest.approx(-10.0)
    assert seen[-1]["target_extras"] == pytest.approx(fortify.target_channels(kept, target))


def test_a_cut_just_above_the_flop_target_does_not_end_the_walk(monkeypatch):
    env, _ = _flop_target_env(monkeypatch, accs=[0.85])
    after = env.preview_flops_ratio(0.5)
    env.target_keep = after - 0.0025
    assert env._land_on_target(0.5) == (0.5, None)
    _, reward, done = env.step(0.5)
    assert env.flops_ratio() == pytest.approx(after) and not done and env._target_final is None
    assert env.param_ratio() < env.target_keep                          # on params this walk would have ended
    assert reward == pytest.approx(-5.0)
    assert env.episode_target_score() == pytest.approx(fortify.target_score(-5.0, after, env.target_keep))
