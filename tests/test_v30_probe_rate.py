"""
v30 ``SPECTRA_ALLOC_KIND=sens_cost_at`` and ``sens_curve``: two probe-rate variants of the cost-aware rule.
``sens_cost_at`` measures every group again at the depth the ``sens_cost`` plan cuts it (``min(planned keep, 0.5)``);
``sens_curve`` measures each group's loss rise and saving at several keeps and allocates along the monotone curves with
one multiplier. ``group_sensitivity(keep=)`` may name a keep per row. Every other kind, and a float ``keep``, are as in
tree_v29.

CPU only, no datasets.  python -m pytest tests/test_v30_probe_rate.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import SHAPE  # noqa: E402
from tests.test_v16_sens_cost import _batches, _model  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_")
NEW_KINDS = ("sens_cost_at", "sens_curve")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


def _rows(model):
    return [row for _g, row in group_sensitivity.group_plan(ModelWithRows(model))]


def _lands(info, target, floor=0.1, tol=0.04):
    """The plan keeps ``target`` to within the coarse steps of a 4-16 channel net, or every group sits at its floor."""
    return abs(info["kept"] - target) <= tol or all(k <= floor + 1e-9 for k in info["keeps"].values())


# ------------------------------------------------------------------ flags


def test_kinds_are_registered_last_and_selected_by_the_env_var(monkeypatch):
    assert alloc_walk.KINDS[-2:] == NEW_KINDS and "sens_cost" in alloc_walk.KINDS
    for name in NEW_KINDS:
        monkeypatch.setenv("SPECTRA_ALLOC_KIND", name.upper())
        assert alloc_walk.kind() == name
    assert not set(NEW_KINDS) & set(alloc_walk.SAMPLE_AROUND)


def test_curve_keeps_and_probe_rounds_parse_defaults_and_reject_bad_values(monkeypatch):
    assert alloc_walk.curve_keeps() == (0.75, 0.5, 0.25, 0.1) and alloc_walk.probe_rounds() == 1
    monkeypatch.setenv("SPECTRA_ALLOC_CURVE_KEEPS", " 0.25, 0.5 ,0.5,0.9 ")
    assert alloc_walk.curve_keeps() == (0.9, 0.5, 0.25)                    # sorted descending, no duplicates
    monkeypatch.setenv("SPECTRA_ALLOC_CURVE_KEEPS", "0.3")
    assert alloc_walk.curve_keeps() == (0.3,)
    for bad in ("0", "1", "1.5", "-0.2", "a,b", "0.5,x", "nan", ","):
        monkeypatch.setenv("SPECTRA_ALLOC_CURVE_KEEPS", bad)
        with pytest.raises(ValueError):
            alloc_walk.curve_keeps()
    monkeypatch.setenv("SPECTRA_ALLOC_PROBE_ROUNDS", "3")
    assert alloc_walk.probe_rounds() == 3
    for bad in ("0", "-1", "x", "1.5"):
        monkeypatch.setenv("SPECTRA_ALLOC_PROBE_ROUNDS", bad)
        with pytest.raises(ValueError):
            alloc_walk.probe_rounds()


# ------------------------------------------------------------------ group_sensitivity


def test_a_dict_keep_equal_to_the_float_gives_the_same_sensitivities_and_costs():
    model, batches = _model(), _batches()
    plan = group_sensitivity.group_plan(ModelWithRows(model))
    rows = [row for _g, row in plan]
    flat_costs, dict_costs = {}, {}
    flat, base_flat = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, keep=0.5, costs=flat_costs)
    by_row, base_dict = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, keep={r: 0.5 for r in rows},
                                                            costs=dict_costs)
    assert flat == by_row and base_flat == base_dict and flat_costs == dict_costs


def test_a_dict_keep_cuts_each_row_to_its_own_fraction():
    model, batches = _model(), _batches()
    plan = group_sensitivity.group_plan(ModelWithRows(model))
    rows = [row for _g, row in plan]
    mixed = {row: (0.75 if k % 2 else 0.25) for k, row in enumerate(rows)}
    costs = {}
    group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, keep=mixed, costs=costs)
    for q in (0.25, 0.75):
        ref = {}
        group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, keep=q, costs=ref)
        assert all(costs[row] == ref[row] for row in rows if mixed[row] == q)


# ------------------------------------------------------------------ _land


def test_land_with_the_sens_cost_weights_is_the_sens_cost_plan():
    model, batches = _model(), _batches()
    plan = group_sensitivity.group_plan(ModelWithRows(model))
    rows = [row for _g, row in plan]
    raw = {}
    sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, costs=raw)
    w = alloc_walk.weights("sens_cost", sens, 0.5, cost={row: raw[row][0] for row in rows})
    floor = {row: 0.1 for row in rows}
    size0 = utils.calc_num_parameters(model)
    keeps, frac, widths = alloc_walk._land(model, plan, rows, w, floor, 0.45, utils.calc_num_parameters, size0, SHAPE,
                                           16, 0.003)
    want_widths, want = alloc_walk.plan_targets(model, batches, SHAPE, "sens_cost", 0.45)
    assert keeps == want["keeps"] and widths == want_widths and frac == want["kept"]


# ------------------------------------------------------------------ the curve helpers


def test_monotone_curve_takes_the_running_maximum_of_the_positive_parts():
    points = [(1.0, 0.0, 0.0), (0.75, 0.5, 10.0), (0.5, 0.2, 8.0), (0.25, -0.1, 30.0), (0.1, 0.9, 25.0)]
    curve = alloc_walk._monotone_curve(points)
    assert curve == [(1.0, 0.0, 0.0), (0.75, 0.5, 10.0), (0.5, 0.5, 10.0), (0.25, 0.5, 30.0), (0.1, 0.9, 30.0)]


def test_curve_at_interpolates_linearly_and_holds_the_ends():
    curve = [(1.0, 0.0, 0.0), (0.5, 1.0, 10.0), (0.1, 5.0, 20.0)]
    assert alloc_walk._curve_at(curve, 1.0) == (0.0, 0.0) and alloc_walk._curve_at(curve, 0.75) == (0.5, 5.0)
    assert alloc_walk._curve_at(curve, 0.5) == (1.0, 10.0)
    assert alloc_walk._curve_at(curve, 0.05) == (5.0, 20.0)


def test_curve_lambda_cuts_the_cheap_group_deeper_and_meets_the_target():
    curve = {1: [(1.0, 0.0, 0.0), (0.5, 0.01, 40.0), (0.1, 0.05, 80.0)],       # cheap to cut
             2: [(1.0, 0.0, 0.0), (0.5, 1.0, 40.0), (0.1, 5.0, 80.0)]}         # costly to cut
    floor = {1: 0.1, 2: 0.1}
    lam, kept, picks = alloc_walk._curve_lambda(curve, floor, 200.0, 0.6)       # the cheap group's deepest cut saves 40 %
    assert lam > 0 and picks == {1: 0.1, 2: 1.0} and abs(kept - 0.6) < 1e-9
    assert all(floor[row] <= k <= 1.0 for row, k in picks.items())
    lam0, kept0, picks0 = alloc_walk._curve_lambda(curve, floor, 200.0, 1.0)     # nothing to save: no multiplier
    assert lam0 == 0.0 and kept0 == 1.0 and picks0 == {1: 1.0, 2: 1.0}


# ------------------------------------------------------------------ sens_cost_at


@pytest.mark.parametrize("target", [0.45, 0.25])
def test_sens_cost_at_probes_each_group_at_the_depth_the_plan_cuts_it(target):
    model, batches = _model(), _batches()
    widths, info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_cost_at", target)
    rows = _rows(model)
    assert info["kind"] == "sens_cost_at" and info["rounds"] == 1 and info["held"] == 0 and set(widths) == set(rows)
    assert _lands(info, target)
    half, probe = info["keeps_half"], info["probe_keeps"]
    assert all(probe[row] == min(half[row], 0.5) for row in rows)
    shallow = [row for row in rows if half[row] >= 0.5]
    assert all(info["sens"][row] == info["sens_half"][row] for row in shallow)
    cut = alloc_walk.cut_to(model, group_sensitivity.group_plan(ModelWithRows(model)), info["keeps"], SHAPE)
    assert utils.calc_num_parameters(cut) / utils.calc_num_parameters(model) == pytest.approx(info["kept"])
    assert set(info["cost"]) == set(rows) and info["cost_budget"] == "params"
    origin = info["origin_widths"]
    assert all(widths[row] <= origin[row] for row in rows)


def test_sens_cost_at_with_more_rounds_probes_at_the_last_plan(monkeypatch):
    monkeypatch.setenv("SPECTRA_ALLOC_PROBE_ROUNDS", "2")
    model, batches = _model(), _batches()
    _widths, info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_cost_at", 0.35)
    assert info["rounds"] == 2 and _lands(info, 0.35)
    assert all(0.0 < k <= 0.5 for k in info["probe_keeps"].values())


# ------------------------------------------------------------------ sens_curve


def test_sens_curve_lands_on_the_target_keeps_the_floor_and_cuts_deeper_at_a_deeper_budget():
    model, batches = _model(), _batches()
    sums = {}
    for target in (0.45, 0.25):
        widths, info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_curve", target)
        rows = _rows(model)
        assert info["kind"] == "sens_curve" and info["held"] == 0 and set(widths) == set(rows)
        assert _lands(info, target)
        assert all(k >= 0.1 - 1e-12 for k in info["keeps"].values())
        for row in rows:
            curve = info["curve"][row]
            assert curve[0] == [1.0, 0.0, 0.0] and [p[0] for p in curve] == [1.0, 0.75, 0.5, 0.25, 0.1]
            assert all(b[1] >= a[1] and b[2] >= a[2] for a, b in zip(curve, curve[1:]))
            assert all(isinstance(v, float) for p in curve for v in p)
        assert info["curve_keeps"] == [0.75, 0.5, 0.25, 0.1] and info["lambda"] >= 0.0
        assert set(info["sens"]) == set(rows) and info["cost_budget"] == "params"
        sums[target] = sum(info["keeps"].values())
    assert sums[0.25] < sums[0.45]


def test_sens_curve_info_is_json_serialisable():
    import json
    model, batches = _model(), _batches()
    _widths, info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_curve", 0.45)
    json.dumps({key: info[key] for key in ("curve", "curve_keeps", "lambda", "est_kept")})


def test_sens_curve_with_a_one_probe_curve_runs_and_lands(monkeypatch):
    monkeypatch.setenv("SPECTRA_ALLOC_CURVE_KEEPS", "0.5")
    model, batches = _model(), _batches()
    _widths, info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_curve", 0.45)
    assert info["curve_keeps"] == [0.5] and _lands(info, 0.45)
    assert all(len(curve) == 2 for curve in info["curve"].values())
    assert all(max(0.0, info["sens"][row]) == curve[1][1] for row, curve in info["curve"].items())


# ------------------------------------------------------------------ the width floor


@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_the_width_floor_holds_for_both_kinds(kind_name):
    model, batches = _model(), _batches()
    widths, info = alloc_walk.plan_targets(model, batches, SHAPE, kind_name, 0.3, min_width=2)
    origin = info["origin_widths"]
    for row, width in widths.items():
        assert width >= min(2, origin[row])
        assert info["keeps"][row] >= min(1.0, 2.0 / origin[row]) - 1e-12


# ------------------------------------------------------------------ grid rounding priority


def test_grid_priority_ranks_the_new_kinds_by_their_own_decoder_weights():
    model, batches = _model(), _batches()
    rows = _rows(model)
    _w, at = alloc_walk.plan_targets(model, batches, SHAPE, "sens_cost_at", 0.3)
    want = alloc_walk.weights("sens_cost", at["sens"], 0.5, cost=at["cost"])
    assert alloc_walk.grid_priority(at, rows) == {row: want[row] for row in rows}
    _w, cv = alloc_walk.plan_targets(model, batches, SHAPE, "sens_curve", 0.3)
    assert alloc_walk.grid_priority(cv, rows) == {row: float(cv["weights"][row]) for row in rows}
    _w, sc = alloc_walk.plan_targets(model, batches, SHAPE, "sens_cost", 0.3)
    assert alloc_walk.grid_priority(sc, rows) == alloc_walk.weights("sens_cost", sc["sens"], 0.5, cost=sc["cost"])
