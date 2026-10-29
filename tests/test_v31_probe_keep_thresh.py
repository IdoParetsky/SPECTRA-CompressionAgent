"""
v31 ``SPECTRA_ALLOC_SENS_KEEP`` (the scalar keep at which ``sens`` and ``sens_cost`` measure; unset = as in tree_v30) and
``SPECTRA_ALLOC_KIND=sens_thresh`` (one global threshold on the monotone damage curves, Li et al. 2017's per-layer
sensitivity rule). Every other kind, and an unset flag, are as in tree_v30.

CPU only, no datasets.  python -m pytest tests/test_v31_probe_keep_thresh.py -v
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import SHAPE  # noqa: E402
from tests.test_v16_sens_cost import _batches, _model  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_")
V30_KINDS = ("uniform", "sens", "inner", "widths", "sample", "agent", "agent_sample", "sens_cost", "randbest", "fallback",
             "sens_cost_at", "sens_curve")
GRID = [round(1.0 - 0.01 * i, 2) for i in range(100)]


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


def _spy(monkeypatch):
    calls, real = [], group_sensitivity.group_sensitivity

    def wrapper(*args, **kwargs):
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(group_sensitivity, "group_sensitivity", wrapper)
    return calls


def _curves():
    """Three hand-built monotone curves (loss, saving) over size0 = 200; row 3 is free to cut to 0.75."""
    pts = {1: [(1.0, 0.0, 0.0), (0.5, 0.01, 40.0), (0.1, 0.05, 80.0)],
           2: [(1.0, 0.0, 0.0), (0.5, 1.0, 40.0), (0.1, 5.0, 80.0)],
           3: [(1.0, 0.0, 0.0), (0.75, 0.0, 10.0), (0.5, 0.2, 30.0), (0.1, 2.0, 60.0)]}
    return {row: alloc_walk._monotone_curve(p) for row, p in pts.items()}


# ------------------------------------------------------------------ flags


def test_sens_keep_is_none_when_unset_or_blank_and_parses_a_float_in_the_open_unit_interval(monkeypatch):
    assert alloc_walk.sens_keep() is None
    monkeypatch.setenv("SPECTRA_ALLOC_SENS_KEEP", "  ")
    assert alloc_walk.sens_keep() is None
    monkeypatch.setenv("SPECTRA_ALLOC_SENS_KEEP", "0.387")
    assert alloc_walk.sens_keep() == 0.387
    for bad in ("0", "1", "1.5", "-0.2", "abc"):
        monkeypatch.setenv("SPECTRA_ALLOC_SENS_KEEP", bad)
        with pytest.raises(ValueError):
            alloc_walk.sens_keep()


def test_sens_thresh_is_the_last_kind_and_every_earlier_kind_keeps_its_index(monkeypatch):
    assert alloc_walk.KINDS[-3:] == ("sens_cost_at", "sens_curve", "sens_thresh")
    assert alloc_walk.KINDS[:len(V30_KINDS)] == V30_KINDS and len(alloc_walk.KINDS) == len(V30_KINDS) + 1
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "SENS_THRESH")
    assert alloc_walk.kind() == "sens_thresh" and "sens_thresh" not in alloc_walk.SAMPLE_AROUND


# ------------------------------------------------------------------ SPECTRA_ALLOC_SENS_KEEP in plan_targets


@pytest.mark.parametrize("kind_name", ["sens_cost", "sens"])
def test_unset_flag_calls_group_sensitivity_without_a_keep_and_adds_nothing_to_info(monkeypatch, kind_name):
    model, batches = _model(), _batches()
    want_widths, want = alloc_walk.plan_targets(model, batches, SHAPE, kind_name, 0.45)
    calls = _spy(monkeypatch)
    widths, info = alloc_walk.plan_targets(model, batches, SHAPE, kind_name, 0.45)
    assert widths == want_widths and info == want and "probe_keep" not in info
    assert len(calls) == 1 and all("keep" not in kw for kw in calls)


@pytest.mark.parametrize("kind_name", ["sens_cost", "sens"])
def test_the_flag_sets_the_measuring_keep_lands_and_is_reported(monkeypatch, kind_name):
    model, batches = _model(), _batches()
    monkeypatch.setenv("SPECTRA_ALLOC_SENS_KEEP", "0.387")
    calls = _spy(monkeypatch)
    _widths, info = alloc_walk.plan_targets(model, batches, SHAPE, kind_name, 0.45)
    assert len(calls) == 1 and calls[0]["keep"] == 0.387
    assert _lands(info, 0.45) and info["probe_keep"] == 0.387


@pytest.mark.parametrize("kind_name", ["sens_cost", "sens"])
def test_the_flag_at_the_default_keep_gives_the_unset_plan(monkeypatch, kind_name):
    model, batches = _model(), _batches()
    want_widths, want = alloc_walk.plan_targets(model, batches, SHAPE, kind_name, 0.35)
    monkeypatch.setenv("SPECTRA_ALLOC_SENS_KEEP", "0.5")
    widths, info = alloc_walk.plan_targets(model, batches, SHAPE, kind_name, 0.35)
    assert widths == want_widths and info["keeps"] == want["keeps"] and info["kept"] == want["kept"]


# ------------------------------------------------------------------ _curve_threshold


def _brute(curve, floor, tau):
    ks = sorted({g for g in GRID if g >= floor} | {floor})
    return min(k for k in ks if alloc_walk._curve_at(curve, k)[0] <= tau + 1e-12)


def test_curve_threshold_picks_the_smallest_grid_keep_within_tau_and_kept_falls_as_tau_rises():
    curve, floor = _curves(), {1: 0.1, 2: 0.1, 3: 0.1}
    def kept_of(tau):
        return 1.0 - sum(alloc_walk._curve_at(curve[r], _brute(curve[r], floor[r], tau))[1] for r in curve) / 200.0

    for target in (0.9, 0.8, 0.7, 0.6, 0.5):
        tau, kept, picks = alloc_walk._curve_threshold(curve, floor, 200.0, target)
        assert tau >= 0.0 and set(picks) == set(curve) and abs(kept - target) < 0.05
        for row, k in picks.items():
            assert k == _brute(curve[row], floor[row], tau)
        assert kept == pytest.approx(kept_of(tau))
    sweep = [kept_of(tau) for tau in (0.0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0, 5.0)]
    assert all(b <= a + 1e-12 for a, b in zip(sweep, sweep[1:])) and sweep[0] > sweep[-1]


def test_curve_threshold_is_zero_when_the_target_needs_no_saving():
    curve, floor = _curves(), {1: 0.1, 2: 0.1, 3: 0.1}
    tau, kept, picks = alloc_walk._curve_threshold(curve, floor, 200.0, 0.99)      # row 3 saves 10 at no loss
    assert tau == 0.0 and picks[3] == 0.75 and picks[1] == picks[2] == 1.0 and kept == pytest.approx(0.95)
    assert alloc_walk._curve_threshold(curve, floor, 200.0, kept)[0] == 0.0


def test_curve_threshold_with_an_unreachable_target_sits_every_group_at_its_deepest_legal_point():
    curve, floor = _curves(), {1: 0.5, 2: 0.5, 3: 0.5}
    tau, kept, picks = alloc_walk._curve_threshold(curve, floor, 200.0, 0.0)       # the floors alone keep 0.5
    assert tau == max(alloc_walk._curve_at(curve[row], 0.5)[0] for row in curve) == 1.0
    assert picks == {1: 0.5, 2: 0.5, 3: 0.5} and kept > 0.0


# ------------------------------------------------------------------ sens_thresh


@pytest.mark.parametrize("target,budget", [(0.5, "params"), (0.3, "params"), (0.5, "flops")])
def test_sens_thresh_lands_on_the_target_and_reports_its_curves(target, budget):
    model, batches = _model(), _batches()
    widths, info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_thresh", target, budget=budget)
    rows = _rows(model)
    assert info["kind"] == "sens_thresh" and info["held"] == 0 and set(widths) == set(rows)
    assert _lands(info, target) and info["tau"] >= 0.0 and isinstance(info["tau"], float) and "lambda" not in info
    assert info["curve_keeps"] == list(alloc_walk.curve_keeps()) and info["cost_budget"] == budget
    assert set(info["curve"]) == set(rows) and set(info["weights"]) == set(rows)
    assert all(k >= 0.1 - 1e-12 for k in info["keeps"].values())


def test_sens_thresh_keeps_the_width_floor():
    model, batches = _model(), _batches()
    widths, info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_thresh", 0.3, min_width=2)
    for row, width in widths.items():
        assert width >= min(2, info["origin_widths"][row])


def test_sens_curve_is_unchanged_it_reports_lambda_and_no_tau():
    model, batches = _model(), _batches()
    _widths, info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_curve", 0.45)
    assert list(info) == ["kind", "alpha", "target", "kept", "keeps", "origin_widths", "sens", "held", "curve",
                          "curve_keeps", "lambda", "est_kept", "cost_budget", "weights"]
    assert info["kind"] == "sens_curve" and "tau" not in info and _lands(info, 0.45)


def test_grid_priority_of_sens_thresh_is_its_weights():
    model, batches = _model(), _batches()
    _w, info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_thresh", 0.3)
    rows = _rows(model)
    assert alloc_walk.grid_priority(info, rows) == {row: float(info["weights"][row]) for row in rows}
