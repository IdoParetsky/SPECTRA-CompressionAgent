"""
tree_v11 D-PROXY pieces: the ``sample`` allocation kind (src/alloc_walk.py) and the plan proxies
(src/plan_proxies.py). CPU only, no datasets.  python -m pytest tests/test_v11_plan_sample.py -v
"""

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from src import alloc_walk, plan_proxies  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402
from tests.test_alloc_walk import RATES, _rising_sens  # noqa: E402
from tests.test_v10_fixed_target import _target_env  # noqa: E402

SAMPLE_ENV = ("SPECTRA_ALLOC_KIND", "SPECTRA_ALLOC_SAMPLE_AROUND", "SPECTRA_ALLOC_SAMPLE_SIGMA",
              "SPECTRA_ALLOC_SAMPLE_SEED", "SPECTRA_ALLOC_UNDERSHOOT", "SPECTRA_FIXED_TARGET",
              "SPECTRA_GROUP_ONCE_PER_PASS", "SPECTRA_EVAL_PROXIES", "SPECTRA_EVAL_PROXY_SEED") + plan_proxies._RECIPE_KEYS


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in SAMPLE_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


def _model():
    torch.manual_seed(0)
    return resnet20(num_classes=10, large_input=False, width=4).eval()


def test_sample_flags_default_and_parse(monkeypatch):
    assert alloc_walk.sample_around() == "sens" and alloc_walk.sample_sigma() == 0.5 and alloc_walk.sample_seed() == 0
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "sample")
    assert alloc_walk.kind() == "sample"
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_AROUND", "Uniform")
    assert alloc_walk.sample_around() == "uniform"
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_AROUND", "inner")
    with pytest.raises(ValueError):
        alloc_walk.sample_around()
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SIGMA", "-1")
    assert alloc_walk.sample_sigma() == 0.0


def test_sample_noise_is_seeded_private_and_sigma_zero_is_identity():
    rows = [3, 7, 11, 15]
    state = random.getstate()
    a, b = alloc_walk.sample_noise(rows, 0.5, 4), alloc_walk.sample_noise(rows, 0.5, 4)
    assert random.getstate() == state                         # its own RNG: the walk's draws do not move
    assert a == b and a != alloc_walk.sample_noise(rows, 0.5, 5)
    assert alloc_walk.sample_noise(rows, 0.0, 4) == {row: 1.0 for row in rows}


@pytest.mark.parametrize("around", ["uniform", "sens"])
def test_a_sample_plan_keeps_the_target_varies_by_seed_and_reduces_to_its_base(monkeypatch, around):
    model = _model()
    sens = _rising_sens(model)
    calls = []

    def fake(m, plan, b, shape, keep=0.5):
        calls.append(1)
        return sens, 1.0

    monkeypatch.setattr(group_sensitivity, "group_sensitivity", fake)
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_AROUND", around)
    plans = []
    for seed in (1, 2):
        monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SEED", str(seed))
        widths, info = alloc_walk.plan_targets(model, [], (3, 32, 32), "sample", 0.6)
        assert info["kept"] == pytest.approx(0.6, abs=0.04)
        assert info["sample"] == {"around": around, "sigma": 0.5, "seed": seed}
        assert set(widths) == set(sens)
        plans.append(info["keeps"])
    assert plans[0] != plans[1]
    assert len(calls) == (2 if around == "sens" else 0)        # a uniform base never measures
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SIGMA", "0")
    _, base = alloc_walk.plan_targets(model, [], (3, 32, 32), around, 0.6)
    _, flat = alloc_walk.plan_targets(model, [], (3, 32, 32), "sample", 0.6)
    assert flat["keeps"] == base["keeps"]


def test_a_sample_walk_lands_on_the_target(monkeypatch):
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "sample")
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SEED", "3")
    model = _model()
    sens = _rising_sens(model)
    monkeypatch.setattr(group_sensitivity, "group_sensitivity", lambda m, plan, b, shape, keep=0.5: (sens, 1.0))
    env, _ = _target_env(model, target=0.6, accs=[0.9] * 400, passes=6)
    env.conf.compression_rates_dict = RATES
    origin = alloc_walk.group_widths(model, list(sens))
    done, steps = False, 0
    while not done and steps < 300:
        legal = env.legal_action_mask(device="cpu")
        pick = int(alloc_walk.action(env, legal, RATES, "cpu").item())
        assert bool(legal[pick])
        _, _, done = env.step(RATES[pick])
        steps += 1
    assert done and 0.6 - 0.03 <= env.param_ratio() <= 0.6 + 1e-9
    state = env._alloc_walk[env.selected_net_path]
    now = alloc_walk.group_widths(env.current_model, list(sens))
    off_plan = [row for row in sens if now[row] / origin[row] < state["widths"][row] / origin[row] - 0.15]
    assert len(off_plan) <= (1 if state["fallback"] else 0)     # 4-16 channel groups: only the landing cut leaves the plan


def test_proxy_names_parse(monkeypatch):
    assert plan_proxies.proxies() == [] and plan_proxies.proxy_seed() == 0
    monkeypatch.setenv("SPECTRA_EVAL_PROXIES", "cut, bn8,BN32,ft1,ft12_4,ft40_10")
    assert plan_proxies.proxies() == ["cut", "bn8", "bn32", "ft1", "ft12_4", "ft40_10"]
    monkeypatch.setenv("SPECTRA_EVAL_PROXIES", "bn")
    with pytest.raises(ValueError):
        plan_proxies.proxies()


def test_seeded_gives_common_draws_and_restores_the_global_rng():
    torch.manual_seed(123)
    without = torch.rand(3)
    torch.manual_seed(123)
    with plan_proxies.seeded(7):
        inside = torch.rand(3)
    assert torch.equal(torch.rand(3), without)                  # the block left the global stream where it was
    with plan_proxies.seeded(7):
        assert torch.equal(torch.rand(3), inside)               # same seed, same draws in every job


def test_handler_recipe_clears_and_restores(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_OPTIM", "sgd")
    monkeypatch.setenv("SPECTRA_FT_SGD_LR", "0.1")
    with plan_proxies.handler_recipe():
        assert "SPECTRA_FT_OPTIM" not in os.environ and "SPECTRA_FT_SGD_LR" not in os.environ
    assert os.environ["SPECTRA_FT_OPTIM"] == "sgd" and os.environ["SPECTRA_FT_SGD_LR"] == "0.1"


class _Handler:
    def __init__(self, model, log):
        self.model, self.log = model, log

    def evaluate_model(self, loader):
        self.model.eval()
        right = total = 0
        with torch.no_grad():
            for x, y in loader:
                right += int((self.model(x).argmax(1) == y).sum())
                total += int(y.numel())
        return right / total

    def train_model(self, loader, **kw):
        kw["recipe_keys"] = sorted(k for k in plan_proxies._RECIPE_KEYS if k in os.environ)
        self.log.append(kw)


def _fake_env():
    torch.manual_seed(1)
    batches = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(4)]
    log = []
    env = SimpleNamespace(conf=SimpleNamespace(device="cpu"), train_loader=batches, val_loader=batches[:2],
                          test_loader=batches[2:], create_learning_handler=lambda m: _Handler(m, log))
    return env, log


def _norm_means(model):
    return [m.running_mean.clone() for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def test_proxies_work_on_a_copy_and_use_the_handler_recipe(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_OPTIM", "sgd")
    env, log = _fake_env()
    model = _model()
    before = _norm_means(model)
    out = plan_proxies.measure_one(env, model, "bn8", "size_param")
    assert set(out) == {"val", "test", "minutes"} and 0.0 <= out["val"] <= 1.0
    assert all(torch.equal(a, b) for a, b in zip(before, _norm_means(model)))    # the candidate is untouched
    plan_proxies.measure_one(env, model, "ft12_4", "size_param")
    plan_proxies.measure_one(env, model, "ft1", "size_param")
    assert [(c["max_epochs"], c["patience"]) for c in log] == [(12, 4), (1, 2)]
    assert all(c["recipe_keys"] == [] and c["allow_reinit_retry"] is False for c in log)
    assert os.environ["SPECTRA_FT_OPTIM"] == "sgd"


def test_measure_reads_cut_from_the_point_and_keeps_going_past_a_failure(monkeypatch):
    env, _ = _fake_env()
    point = {"val_acc": 0.5, "test_acc": 0.4, "val_origin": 0.9, "test_origin": 0.9, "param": 0.6, "flop": 0.5,
             "step": 12}
    cand = {"point": point, "model": _model()}

    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(plan_proxies, "measure_one", broken)
    out = plan_proxies.measure(env, "/nets/r56w4.pt", "size_param", cand, ["cut", "bn8"])
    assert out["cut"] == {"val": 0.5, "test": 0.4, "minutes": 0.0} and out["bn8"] is None


def _point(step):
    return {"step": step, "param": 0.6, "flop": 0.5, "val_acc": 0.8, "test_acc": 0.8, "val_origin": 0.9,
            "test_origin": 0.9}


def test_the_runner_measures_each_non_origin_step_once_and_only_when_asked(monkeypatch):
    import a2c_agent_reinforce_runner as runner
    monkeypatch.setattr(runner.utils, "print_flush", lambda msg: None)
    calls = []
    monkeypatch.setattr(plan_proxies, "measure",
                        lambda env, net, label, cand, names: calls.append((label, cand["point"]["step"], names)))
    env, _ = _fake_env()
    env.data_dict = {"net.pt": (_model(), None)}
    env.selected_net_path = "net.pt"

    def candidates():
        return {label: {"point": _point(step), "model": None, "key": None}
                for label, step in (("val_best", 5), ("size_param0.60", 5), ("terminal", 9))}

    picked = {"val_best": _point(5), "origin": _point(0)}
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_ORIGIN", "1")
    runner._run_final_ft(env, "net.pt", picked, candidates(), 0)
    assert calls == []                                                   # off by default
    monkeypatch.setenv("SPECTRA_EVAL_PROXIES", "cut,bn8")
    runner._run_final_ft(env, "net.pt", picked, candidates(), 0)
    assert calls == [("val_best", 5, ["cut", "bn8"]), ("terminal", 9, ["cut", "bn8"])]
