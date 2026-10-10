"""
v16 ``SPECTRA_ALLOC_KIND=sens_cost``: ``sens`` weights divided by what the same half cut saves (params, or MACs under
``SPECTRA_ALLOC_BUDGET=flops``). ``group_sensitivity(costs={})`` measures the savings on the cut it already makes; the
default (``costs=None``) and every other kind are untouched.

CPU only, no datasets.  python -m pytest tests/test_v16_sens_cost.py -v
"""

import copy
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import SHAPE, ZOO, _plan, _randomize_norms  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402
from src.NetworkEnv import prune_current_model  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


def _model(name="thin_r20_w4"):
    torch.manual_seed(0)
    model = ZOO[name]().eval()
    _randomize_norms(model)
    return model


def _batches(n=2, size=8):
    gen = torch.Generator().manual_seed(3)
    return [(torch.randn(size, *SHAPE, generator=gen), torch.randint(0, 10, (size,), generator=gen))
            for _ in range(n)]


# ------------------------------------------------------------------ the kind


def test_sens_cost_is_a_kind_and_selected_by_the_env_var(monkeypatch):
    assert "sens_cost" in alloc_walk.KINDS
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "sens_cost")
    assert alloc_walk.kind() == "sens_cost"
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "SENS_COST")
    assert alloc_walk.kind() == "sens_cost"
    assert "sens_cost" not in alloc_walk.SAMPLE_AROUND


def test_default_kind_is_still_sens(monkeypatch):
    assert "SPECTRA_ALLOC_KIND" not in os.environ and alloc_walk.kind() == "sens"
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "")
    assert alloc_walk.kind() == "sens"


# ------------------------------------------------------------------ weights


def test_weights_sens_cost_divides_by_the_cost_and_floors_both():
    w = alloc_walk.weights("sens_cost", {1: 1.0, 2: 1.0}, a=1.0, cost={1: 1.0, 2: 4.0})
    assert w == pytest.approx({1: 1.6, 2: 0.4})
    # a zero cost is floored at 5 % of the median cost (median of [0, 2, 2] is 2, floor 0.1), not divided by:
    # v = [1 / 0.1, 0.5, 0.5], median 0.5
    w = alloc_walk.weights("sens_cost", {1: 1.0, 2: 1.0, 3: 1.0}, a=1.0, cost={1: 0.0, 2: 2.0, 3: 2.0})
    assert w == pytest.approx({1: 20.0, 2: 1.0, 3: 1.0})
    # an all-zero cost falls back to the 1e-6 floor on every row: still finite, all weights equal
    w = alloc_walk.weights("sens_cost", {1: 1.0, 2: 1.0}, a=1.0, cost={1: 0.0, 2: 0.0})
    assert w == pytest.approx({1: 1.0, 2: 1.0})
    # a negative sensitivity is floored the way ``sens`` floors it (positives [0, 1, 1]: floor 0.05)
    w = alloc_walk.weights("sens_cost", {1: -1.0, 2: 1.0, 3: 1.0}, a=1.0, cost={1: 1.0, 2: 1.0, 3: 1.0})
    assert w == pytest.approx({1: 0.05, 2: 1.0, 3: 1.0})
    # alpha is the power
    w = alloc_walk.weights("sens_cost", {1: 1.0, 2: 1.0}, a=0.5, cost={1: 1.0, 2: 4.0})
    assert w == pytest.approx({1: 1.6 ** 0.5, 2: 0.4 ** 0.5})


def test_weights_sens_cost_needs_a_cost_for_every_row():
    with pytest.raises(ValueError):
        alloc_walk.weights("sens_cost", {1: 1.0, 2: 1.0})
    with pytest.raises(ValueError):
        alloc_walk.weights("sens_cost", {1: 1.0, 2: 1.0}, cost={1: 1.0})
    with pytest.raises(ValueError):
        alloc_walk.weights("sens_cost", {1: 1.0, 2: 1.0}, cost={1: 1.0, 3: 1.0})


def test_weights_of_the_other_kinds_are_unchanged():
    sens = {1: 4.0, 2: 1.0, 3: 0.25, 4: -1.0}
    # positives [4, 1, 0.25, 0]: median 0.625, floor 0.03125; floored s = [4, 1, 0.25, 0.03125], median 0.625
    want = {1: 2.529822128134704, 2: 1.2649110640673518, 3: 0.6324555320336759, 4: 0.22360679774997896}
    assert alloc_walk.weights("sens", sens) == pytest.approx(want, rel=1e-12)
    assert alloc_walk.weights("sens", sens, 0.5) == pytest.approx(want, rel=1e-12)
    linear = {1: 6.4, 2: 1.6, 3: 0.4, 4: 0.05}
    assert alloc_walk.weights("sens", sens, a=1.0) == pytest.approx(linear, rel=1e-12)
    assert alloc_walk.weights("sens", sens, 1.0, cost={1: 1, 2: 9, 3: 1, 4: 1}) == pytest.approx(linear, rel=1e-12)
    assert alloc_walk.weights("sample", sens, 1.0) == pytest.approx(linear, rel=1e-12)
    for name in ("uniform", "inner"):
        assert alloc_walk.weights(name, sens, 0.7) == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}


# ------------------------------------------------------------------ group_sensitivity(costs=...)


def test_group_sensitivity_costs_do_not_change_sens_and_are_the_savings_of_the_same_cut():
    model = _model()
    plan, batches = _plan(model), _batches()
    sens0, base0 = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE)
    sens1, base1 = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, costs=None)
    costs = {}
    sens2, base2 = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, costs=costs)
    assert sens0 == sens1 == sens2 and base0 == base1 == base2
    rows = [row for _g, row in plan]
    assert list(costs) == rows and list(sens2) == rows
    for row in rows:
        p_saved, f_saved = costs[row]
        assert isinstance(p_saved, float) and isinstance(f_saved, float) and p_saved > 0 and f_saved > 0
    p0, f0 = utils.calc_num_parameters(model), utils.calc_flops(model, SHAPE)
    for row in (rows[0], rows[len(rows) // 2], rows[-1]):
        cut = prune_current_model(ModelWithRows(copy.deepcopy(model)), group_sensitivity.SENS_KEEP, row, quiet=True,
                                  record=False, input_shape=SHAPE, importance="l1")
        assert costs[row][0] == p0 - utils.calc_num_parameters(cut.model)
        assert costs[row][1] == pytest.approx(f0 - utils.calc_flops(cut.model, SHAPE), rel=1e-12)
        del cut


def test_group_sensitivity_without_costs_measures_no_cost(monkeypatch):
    model = _model()
    plan = _plan(model)[:2]
    monkeypatch.setattr(utils, "calc_flops", lambda *a, **k: pytest.fail("costs=None must not price the cuts"))
    monkeypatch.setattr(utils, "calc_num_parameters", lambda *a, **k: pytest.fail("costs=None must not price the cuts"))
    sens, _base = group_sensitivity.group_sensitivity(model, plan, _batches(1), SHAPE)
    assert len(sens) == 2


# ------------------------------------------------------------------ plan_targets


@pytest.mark.parametrize("budget", ["flops", "params"])
def test_plan_targets_sens_cost_lands_the_budget(budget):
    model = _model("r20_chenyaofo")
    plan, batches = _plan(model), _batches()
    rows = [row for _g, row in plan]
    widths, info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_cost", 0.6, budget=budget)
    assert info["kind"] == "sens_cost" and info["cost_budget"] == budget and set(info["cost"]) == set(rows)
    assert set(info["sens"]) == set(rows) and set(widths) == set(rows)
    size = (lambda m: utils.calc_flops(m, SHAPE)) if budget == "flops" else utils.calc_num_parameters
    cut = alloc_walk.cut_to(model, plan, info["keeps"], SHAPE)
    kept = size(cut) / size(model)
    assert kept == pytest.approx(info["kept"]) and abs(kept - 0.6) < 0.02
    assert alloc_walk.group_widths(cut, rows) == widths
    # the cost handed to the weights is the one of the budget
    raw = {}
    group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, costs=raw)
    assert info["cost"] == {row: raw[row][1 if budget == "flops" else 0] for row in rows}
    del cut


def test_plan_targets_sens_and_uniform_info_has_no_cost():
    model = _model()
    batches = _batches()
    for kind_name in ("sens", "uniform"):
        _w, info = alloc_walk.plan_targets(model, batches, SHAPE, kind_name, 0.6)
        assert "cost" not in info and "cost_budget" not in info and info["kind"] == kind_name
