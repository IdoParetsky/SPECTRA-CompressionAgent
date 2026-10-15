"""
v22 ENS (default off): ``plan_agent.plan_for_env`` given several checkpoints, comma-separated, decodes the mean of
their scores once. One path decodes exactly as before. The members must agree on ``plan_agent.ENSEMBLE_KEYS``, and
the global RNG after the call is the one a single checkpoint leaves, so a recovery downstream draws the same.

CPU only, no datasets.  python -m pytest tests/test_v22_plan_ensemble.py -v
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

from tests.test_plan_agent import LAYOUT, RATES, ZOO, toy_state  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import plan_agent, state_dump  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
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


def _setup(monkeypatch):
    torch.manual_seed(0)
    model = ZOO["thin_r20_w4"]().eval()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    plan = group_sensitivity.group_plan(mwr, groups)
    rows = state_dump.token_rows(mwr, groups, plan)
    state = toy_state(len(rows))
    monkeypatch.setattr(state_dump, "encode_origin", lambda env, m, g: dict(state))
    env = types.SimpleNamespace(conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES),
                                current_model=model, target_keep=0.6,
                                _dependency_groups=lambda m: channel_groups.build_channel_groups(m.model))
    pm = plan_agent.ParamModel(model, plan)
    prepared = plan_agent.plan_state(state, plan_agent.spans_of(LAYOUT), 0.6, ("sens",))
    mask, k = plan_agent.token_index(rows, pm.rows, "cpu")
    return env, pm, prepared, mask, k


def _save(tmp_path, run, seed, **meta):
    torch.manual_seed(seed)
    policy = plan_agent.PlanPolicy(63)
    with torch.no_grad():
        policy.head.weight.normal_()
    path = tmp_path / run / "plan_agent" / "policy_latest.pt"
    plan_agent.save_policy(policy, str(path), tokens="layer", zero=["sens"], k_min=0.1, **meta)
    return policy, str(path)


def _mu(policy, pm, prepared, mask, k):
    with torch.no_grad():
        return policy.eval()(prepared, mask, k, len(pm.rows)).tolist()


def test_one_path_decodes_as_before_and_the_same_path_twice_is_the_same_plan(monkeypatch, tmp_path, v10_env):
    env, pm, prepared, mask, k = _setup(monkeypatch)
    policy, path = _save(tmp_path, "job1", 1)
    w1, i1 = plan_agent.plan_for_env(env, 0.58, path)
    assert "ensemble" not in i1 and i1["policy"] == "plan_agent/policy_latest.pt"
    assert w1 == plan_agent.decode(_mu(policy, pm, prepared, mask, k), pm, 0.58)[0]
    w2, i2 = plan_agent.plan_for_env(env, 0.58, f"{path}, {path}")
    assert w2 == w1 and i2["ensemble"]["n"] == 2 and i2["ensemble"]["spread"] == pytest.approx(0.0, abs=1e-12)
    assert i2["policy"] == "job1/plan_agent/policy_latest.pt+job1/plan_agent/policy_latest.pt"


def test_three_members_decode_the_mean_score(monkeypatch, tmp_path, v10_env):
    env, pm, prepared, mask, k = _setup(monkeypatch)
    members = [_save(tmp_path, f"job{s}", s) for s in (1, 2, 3)]
    mus = [_mu(p, pm, prepared, mask, k) for p, _ in members]
    mean = [sum(v) / 3 for v in zip(*mus)]
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **kw: lines.append(str(s)))
    widths, info = plan_agent.plan_for_env(env, 0.58, ",".join(path for _, path in members))
    assert widths == plan_agent.decode(mean, pm, 0.58)[0]
    assert info["sens"] == pytest.approx({row: m for row, m in zip(pm.rows, mean)})
    assert info["ensemble"]["n"] == 3 and info["ensemble"]["spread"] > 0
    assert abs(info["kept"] - 0.58) < 0.02
    assert any("plan agent ensemble of 3 policies" in l and "job2/plan_agent/policy_latest.pt" in l for l in lines)
    singles = [plan_agent.plan_for_env(env, 0.58, path)[0] for _, path in members]
    assert any(widths != w for w in singles)  # the mean is not simply one member's plan


def test_the_global_rng_is_left_as_one_checkpoint_leaves_it(monkeypatch, tmp_path, v10_env):
    env, _pm, _prepared, _mask, _k = _setup(monkeypatch)
    (_p1, a), (_p2, b) = _save(tmp_path, "job1", 1), _save(tmp_path, "job2", 2)
    torch.manual_seed(123)
    plan_agent.plan_for_env(env, 0.58, a)
    after_one = torch.rand(4)
    torch.manual_seed(123)
    plan_agent.plan_for_env(env, 0.58, f"{a},{b}")
    after_two = torch.rand(4)
    assert torch.equal(after_one, after_two)


@pytest.mark.parametrize("meta", [{"k_min": 0.2}, {"min_width": 2}, {"budget": "flops"}, {"tokens": "group"}])
def test_members_that_trained_differently_are_refused(monkeypatch, tmp_path, v10_env, meta):
    env, _pm, _prepared, _mask, _k = _setup(monkeypatch)
    _p1, a = _save(tmp_path, "job1", 1)
    torch.manual_seed(2)
    policy = plan_agent.PlanPolicy(63)
    path = tmp_path / "job2" / "plan_agent" / "policy_latest.pt"
    saved = {"tokens": "layer", "zero": ["sens"], "k_min": 0.1}
    saved.update(meta)
    plan_agent.save_policy(policy, str(path), **saved)
    with pytest.raises(ValueError, match="plan-agent ensemble"):
        plan_agent.plan_for_env(env, 0.58, f"{a},{path}")


def test_an_empty_path_list_is_refused(monkeypatch, tmp_path, v10_env):
    env, *_rest = _setup(monkeypatch)
    with pytest.raises(ValueError):
        plan_agent.plan_for_env(env, 0.58, " , ")
