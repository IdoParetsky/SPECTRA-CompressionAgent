"""
Allocation-following eval walk (``SPECTRA_EVAL_POLICY=alloc``, src/alloc_walk.py).

CPU only, no datasets.  python -m pytest tests/test_alloc_walk.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402
from tests.test_v10_fixed_target import _target_env  # noqa: E402

RATES = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
ALLOC_ENV = ("SPECTRA_ALLOC_KIND", "SPECTRA_ALLOC_ALPHA", "SPECTRA_ALLOC_UNDERSHOOT", "SPECTRA_ALLOC_MIN_KEEP",
             "SPECTRA_FIXED_TARGET", "SPECTRA_GROUP_ONCE_PER_PASS", "SPECTRA_EVAL_SIZE_MATCH")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ALLOC_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


def _rising_sens(model):
    """A fake measurement: later groups hurt more, so a sens plan keeps more of them."""
    rows = [row for _g, row in group_sensitivity.group_plan(ModelWithRows(model))]
    return {row: 0.01 * (1 + k) for k, row in enumerate(rows)}


def test_flags_default_and_parse(monkeypatch):
    assert alloc_walk.kind() == "sens" and alloc_walk.alpha() == 0.5
    assert alloc_walk.undershoot() == 0.02 and alloc_walk.min_keep() == 0.1
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "Uniform")
    assert alloc_walk.kind() == "uniform"
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "anti")
    with pytest.raises(ValueError):
        alloc_walk.kind()


def test_weights_follow_a0():
    sens = {"a": 1.0, "b": 4.0, "c": 0.25, "d": -0.1}
    assert alloc_walk.weights("uniform", sens) == {k: 1.0 for k in sens}
    w = alloc_walk.weights("sens", sens, 0.5)
    assert w["b"] > w["a"] > w["c"] > w["d"] > 0          # the negative rise is floored, not undefined
    mid = (0.25 + 1.0) / 2                                 # median of the floored values
    assert w["b"] == pytest.approx((4.0 / mid) ** 0.5)


def test_choose_takes_the_closest_width_and_ties_to_the_milder_cut():
    legal = [0, 1, 2, 3, 4]
    assert alloc_walk.choose(64, 64, RATES, legal, 0) == 0                 # at its target: identity
    assert alloc_walk.choose(40, 48, RATES, legal, 0) == 0                 # below it: never cut further
    pick = alloc_walk.choose(64, 40, RATES, legal, 0)
    best = min((abs(pruning.target_width(64, RATES[i]) - 40), -RATES[i], i) for i in legal[1:])
    assert pick == best[2] == 4
    assert alloc_walk.choose(10, 8, RATES, [0, 1, 3], 0) == 1             # 9 and 7 tie at 1 channel: milder
    assert alloc_walk.choose(64, 40, RATES, [0, 1], 0) == 1                # strongest legal is still closer


@pytest.mark.parametrize("kind_name", ["uniform", "sens"])
def test_plan_keeps_the_target_and_orders_keeps_by_sensitivity(monkeypatch, kind_name):
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    sens = _rising_sens(model)
    monkeypatch.setattr(group_sensitivity, "group_sensitivity", lambda m, plan, b, shape, keep=0.5: (sens, 1.0))
    widths, info = alloc_walk.plan_targets(model, [], (3, 32, 32), kind_name, 0.6)
    assert info["kept"] == pytest.approx(0.6, abs=0.04)          # groups of 4-16 channels: coarse steps
    assert set(widths) == set(sens)
    keeps = [info["keeps"][row] for row in sorted(sens)]
    if kind_name == "uniform":
        assert max(keeps) - min(keeps) < 1e-9
    else:
        assert all(b >= a for a, b in zip(keeps, keeps[1:])) and keeps[-1] > keeps[0]
    origin = info["origin_widths"]
    assert all(widths[row] <= origin[row] for row in widths)
    cut = alloc_walk.cut_to(model, group_sensitivity.group_plan(ModelWithRows(model)), info["keeps"], (3, 32, 32))
    assert utils.calc_num_parameters(cut) / utils.calc_num_parameters(model) == pytest.approx(info["kept"])


@pytest.mark.parametrize("kind_name", ["uniform", "sens"])
def test_the_walk_lands_on_the_target_and_follows_the_plan(monkeypatch, kind_name):
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", kind_name)
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
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
    kept = env.param_ratio()
    assert done and 0.6 - 0.03 <= kept <= 0.6 + 1e-9
    state = env._alloc_walk[env.selected_net_path]
    assert not state["fallback"]
    now = alloc_walk.group_widths(env.current_model, list(sens))
    realised = [now[row] / origin[row] for row in sorted(sens)]
    planned = [state["widths"][row] / origin[row] for row in sorted(sens)]
    assert all(r >= p - 0.15 for r, p in zip(realised, planned))     # no group is driven far past its target
    if kind_name == "sens":
        half = len(realised) // 2
        assert sum(realised[half:]) / (len(realised) - half) > sum(realised[:half]) / half


def test_a_stalled_walk_falls_back_to_the_strongest_cut(monkeypatch):
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    env, _ = _target_env(model, target=0.6, accs=[0.9] * 50)
    env.conf.compression_rates_dict = RATES
    widths = alloc_walk.group_widths(model, [r for _g, r in group_sensitivity.group_plan(ModelWithRows(model))])
    env._alloc_walk = {env.selected_net_path: {"widths": widths, "n_rows": 3, "idle": 0,
                                               "fallback": False, "last_kept": 1.0}}
    legal = torch.ones(len(RATES), dtype=torch.bool)
    picks = [int(alloc_walk.action(env, legal, RATES, "cpu").item()) for _ in range(3)]
    assert picks == [0, 0, 4]                                          # every group at its plan; a pass idle
    assert env._alloc_walk[env.selected_net_path]["fallback"]
