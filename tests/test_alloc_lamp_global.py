"""
``SPECTRA_ALLOC_KIND=global_mag`` and ``lamp``: two agent-free allocation rules read off the weights alone, the
comparators a reviewer may ask for (``docs/paper/LIT_WILCOXON_COMPETITORS_16NOV.md``). ``global_mag`` is the global
magnitude ranking (one importance threshold for every layer; Torch-Pruning's ``global_pruning=True``), ``lamp`` is LAMP
(Lee et al., ICLR 2021) in its structured form. A channel's magnitude is the score the plan's own L1 cut ranks it by
(``pruning.group_importance``); ``global_mag`` divides it by the group's mean, ``lamp`` squares it and rescales it
within the group (score_u = w_u / sum of the w_v at least as large). One threshold then serves every group, chosen on
the budget's cost model through the unchanged ``_curve_threshold``, and the plan lands through ``_land`` like every
rule. No batch is read and no probe runs. Every other kind is as before.

CPU only, no datasets.  python -m pytest tests/test_alloc_lamp_global.py -v
"""

import copy
import json
import os
import random
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import SmallVGG, _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import SHAPE, _plan  # noqa: E402
from tests.test_v16_sens_cost import _model  # noqa: E402
from src import alloc_walk, plan_agent  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_ALLOC_", "SPECTRA_PLAN_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_MIN_WIDTH",
                 "SPECTRA_GROUP_ONCE", "SPECTRA_STEM_ROWS", "SPECTRA_WIDTH_LADDER", "SPECTRA_ACTION_",
                 "SPECTRA_PROTECT_STREAMS", "SPECTRA_FORTIFY", "SPECTRA_EVAL_ROLLBACK", "SPECTRA_CUT_RANK",
                 "SPECTRA_FILTER_IMPORTANCE")
OLD_KINDS = ("uniform", "sens", "inner", "widths", "sample", "agent", "agent_sample", "sens_cost", "randbest",
             "fallback", "sens_cost_at", "sens_curve", "sens_thresh")
NEW_KINDS = ("global_mag", "lamp")
NEW_INFO = ["kind", "alpha", "target", "kept", "keeps", "origin_widths", "sens", "held", "tau", "est_kept",
            "cost_budget", "weights"]
GRID = [round(1.0 - 0.01 * i, 2) for i in range(100)]
RATES = {0: 1.0, 1: 0.9, 2: 0.8}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


def _rows(model):
    return [row for _group, row in _plan(model)]


def _scores(model, kind_name):
    """``{row: the kind's score per channel}`` of every planned group of ``model``."""
    return {row: alloc_walk._channel_scores(kind_name, alloc_walk._group_magnitudes(group))
            for group, row in _plan(model)}


def _steps(model, kind_name, floor):
    scores = _scores(model, kind_name)
    return {row: alloc_walk._threshold_steps(scores[row], alloc_walk._threshold_keeps(floor[row])) for row in scores}


def _picked(row_steps, tau):
    """The step ``_curve_threshold`` picks at ``tau``: the deepest one before the first whose loss is above it."""
    best = row_steps[0]
    for step in row_steps[1:]:
        if step[2] > tau + 1e-12:
            break
        best = step
    return best


def _lands(info, target, floor=0.1, tol=0.04):
    """The plan keeps ``target`` to within the coarse steps of a 4-16 channel net, or every group sits at its floor."""
    return abs(info["kept"] - target) <= tol or all(k <= floor + 1e-9 for k in info["keeps"].values())


def _vgg():
    torch.manual_seed(0)
    return SmallVGG().eval()


def _alloc_env(model, target_keep, passes=6):
    return types.SimpleNamespace(selected_net_path="net.pt", current_model=model, target_keep=target_keep,
                                 conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES, passes=passes),
                                 train_loader=None, _input_shape=lambda: SHAPE)


# ------------------------------------------------------------------ the kinds


def test_the_two_kinds_come_after_every_old_kind_and_an_unknown_kind_still_raises(monkeypatch):
    assert alloc_walk.KINDS[:len(OLD_KINDS)] == OLD_KINDS                  # every old name, at its old index
    assert alloc_walk.KINDS[len(OLD_KINDS):] == NEW_KINDS == alloc_walk.MAGNITUDE_KINDS
    assert alloc_walk.kind() == "sens"                                     # the default is still sens
    for name in NEW_KINDS:
        monkeypatch.setenv("SPECTRA_ALLOC_KIND", name.upper())
        assert alloc_walk.kind() == name
    assert not set(NEW_KINDS) & set(alloc_walk.SAMPLE_AROUND) and not set(NEW_KINDS) & set(alloc_walk.FALLBACK_PRIORS)
    for bad in ("magnitude", "global", "lamp2", "global_mag,lamp"):
        monkeypatch.setenv("SPECTRA_ALLOC_KIND", bad)
        with pytest.raises(ValueError):
            alloc_walk.kind()


# ------------------------------------------------------------------ the scores


def test_lamp_scores_on_a_hand_computed_four_channel_example():
    # sorted ascending 1, 2, 3, 4: 1 / (1 + 2 + 3 + 4), 2 / (2 + 3 + 4), 3 / (3 + 4), 4 / 4
    assert alloc_walk._lamp_scores([1.0, 2.0, 3.0, 4.0]) == pytest.approx([0.1, 2 / 9, 3 / 7, 1.0], rel=1e-12)
    assert alloc_walk._lamp_scores([4.0, 1.0, 3.0, 2.0]) == pytest.approx([1.0, 0.1, 3 / 7, 2 / 9], rel=1e-12)
    assert alloc_walk._lamp_scores([2.0, 2.0]) == [1.0, 0.5]              # equal values: the earlier ranks larger
    assert alloc_walk._lamp_scores([0.0, 0.0]) == [0.0, 0.0] and alloc_walk._lamp_scores([]) == []
    # the kind squares the magnitudes first: 1, 4, 9, 16 give 1 / 30, 4 / 29, 9 / 25, 16 / 16
    assert alloc_walk._channel_scores("lamp", [1.0, 2.0, 3.0, 4.0]) == pytest.approx([1 / 30, 4 / 29, 9 / 25, 1.0],
                                                                                    rel=1e-12)
    # global_mag: over the group's mean (3)
    assert alloc_walk._channel_scores("global_mag", [1.0, 2.0, 3.0, 6.0]) == pytest.approx([1 / 3, 2 / 3, 1.0, 2.0],
                                                                                          rel=1e-12)


@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_scores_ignore_a_groups_scale_keep_its_order_and_give_a_dead_channel_zero(kind_name):
    magnitudes = [0.3, 1.7, 0.9, 2.2, 0.0]
    scores = alloc_walk._channel_scores(kind_name, magnitudes)
    assert scores[4] == 0.0 and sorted(range(5), key=scores.__getitem__) == sorted(range(5), key=magnitudes.__getitem__)
    assert alloc_walk._channel_scores(kind_name, [8.0 * m for m in magnitudes]) == scores      # a power of two: exact
    assert alloc_walk._channel_scores(kind_name, [10.0 * m for m in magnitudes]) == pytest.approx(scores, rel=1e-12)
    assert alloc_walk._channel_scores(kind_name, [0.0, 0.0]) == [0.0, 0.0]


# ------------------------------------------------------------------ the threshold on one group and across groups


def test_threshold_keeps_are_the_grid_curve_threshold_reads_from_the_floor_up():
    keeps = alloc_walk._threshold_keeps(0.1)
    assert keeps == [k for k in GRID if k >= 0.1] and keeps[0] == 1.0 and keeps[-1] == 0.1
    keeps = alloc_walk._threshold_keeps(0.125)                             # a width floor off the grid: 2 of 16
    assert keeps[-1] == 0.125 and keeps[:-1] == [k for k in GRID if k >= 0.125]
    assert alloc_walk._threshold_keeps(1.0) == [1.0]


def test_threshold_steps_give_the_cuts_width_and_the_largest_score_it_removes():
    steps = alloc_walk._threshold_steps([0.2, 1.0, 0.5, 2.3], [1.0, 0.75, 0.5, 0.25])
    assert steps == [(1.0, 4, 0.0), (0.75, 3, 0.2), (0.5, 2, 0.5), (0.25, 1, 1.0)]
    # the cut counts the channels that are not dead (3 here) and removes one of them at any keep below 1
    assert alloc_walk._threshold_steps([0.0, 1.0, 0.0, 2.0, 3.0], [1.0, 0.9, 0.5]) == [(1.0, 5, 0.0), (0.9, 2, 1.0),
                                                                                         (0.5, 2, 1.0)]
    # one channel alive: the dead ones go at no loss
    assert alloc_walk._threshold_steps([0.0, 0.0, 5.0], [1.0, 0.5]) == [(1.0, 3, 0.0), (0.5, 1, 0.0)]


def test_threshold_costs_add_up_to_the_saving_of_the_picked_widths_together():
    # two groups around one layer of a * b weights, 4 channels each
    steps = {"a": [(1.0, 4, 0.0), (0.75, 3, 0.1), (0.5, 2, 0.4)], "b": [(1.0, 4, 0.0), (0.75, 3, 0.2), (0.5, 2, 0.3)]}
    saved = alloc_walk._threshold_costs(steps, lambda widths: widths.get("a", 4) * widths.get("b", 4), 16)
    assert saved == {"a": [0.0, 4.0, 6.0], "b": [0.0, 3.0, 6.0]}
    assert saved["a"][1] + saved["b"][1] == 16 - 3 * 3                     # tau 0.25: both at 3 channels
    assert saved["a"][1] + saved["b"][2] == 16 - 3 * 2                     # tau 0.35
    assert saved["a"][2] + saved["b"][2] == 16 - 2 * 2                     # tau 0.5
    assert (16 - 3 * 4) + (16 - 4 * 3) == 8 != 16 - 3 * 3                  # one group at a time counts the layer twice


def test_the_curves_give_the_threshold_rule_and_the_joint_saving_through_the_unchanged_curve_threshold(monkeypatch):
    monkeypatch.setattr(alloc_walk, "_group_magnitudes", lambda group: group)          # a "group" is its magnitudes
    rng = random.Random(0)
    for _trial in range(12):
        widths0 = {row: rng.choice([4, 8, 16, 33, 64, 150]) for row in range(rng.randint(2, 5))}
        plan = [([rng.random() ** 2 for _ in range(n)], row) for row, n in widths0.items()]
        plan[0][0][0] = 0.0                                                # a dead channel in the first group
        floor = {row: max(0.1, min(1.0, 2.0 / n)) for row, n in widths0.items()}

        def cost_of(widths, widths0=widths0):          # a chain of 3x3 layers with biases and a 10-way head
            ws = [3] + [widths.get(row, n) for row, n in widths0.items()] + [10]
            return sum(9 * a * b + b for a, b in zip(ws, ws[1:]))

        total0 = cost_of({})
        for kind_name in NEW_KINDS:
            curve, half = alloc_walk._magnitude_curves(plan, kind_name, floor, cost_of, total0)
            steps = {row: alloc_walk._threshold_steps(alloc_walk._channel_scores(kind_name, magnitudes),
                                                      alloc_walk._threshold_keeps(floor[row]))
                     for magnitudes, row in plan}
            for row, row_steps in steps.items():       # each step's own loss comes back, nothing is interpolated
                assert [alloc_walk._curve_at(curve[row], k)[0] for k, _w, _l in row_steps] == [s[2] for s in row_steps]
                assert all(b[2] >= a[2] and b[1] <= a[1] for a, b in zip(row_steps, row_steps[1:]))
            assert set(half) == set(steps) and all(v >= 0.0 for v in half.values())
            for target in (0.8, 0.5, 0.2):
                tau, est, picks = alloc_walk._curve_threshold(curve, floor, total0, target)
                picked = {row: _picked(row_steps, tau) for row, row_steps in steps.items()}
                assert picks == {row: step[0] for row, step in picked.items()}
                assert est == pytest.approx(cost_of({row: step[1] for row, step in picked.items()}) / total0, abs=1e-9)


@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_a_channels_magnitude_is_the_cuts_own_score_and_the_threshold_keeps_what_the_cut_keeps(kind_name):
    model = _model()
    model.layer1[0].conv1.weight.data[1].zero_()                          # a dead filter: the cut counts the others
    for group, _row in _plan(model):
        magnitudes = alloc_walk._group_magnitudes(group)
        assert magnitudes == pruning.group_importance(group, "l1").tolist() and len(magnitudes) == group.width
        scores = alloc_walk._channel_scores(kind_name, magnitudes)
        for keep in (0.75, 0.5):
            kept = set(pruning.select_group_survivors(group, keep, mode="l1").tolist())
            _keep, width, loss = alloc_walk._threshold_steps(scores, [1.0, keep])[1]
            removed = [scores[c] for c in range(group.width) if c not in kept]
            assert width == len(kept) and max(removed) == loss           # the cut's width, the strongest it removes
            assert min(scores[c] for c in kept) >= loss                  # every channel it keeps scores at least that


@pytest.mark.parametrize("budget", ["params", "flops"])
@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_the_kept_fraction_never_rises_with_tau_and_is_the_cost_models_at_the_picked_widths(kind_name, budget):
    model = _model()
    plan = _plan(model)
    floor = {row: 0.1 for _group, row in plan}
    cost = plan_agent.FlopModel(model, plan, SHAPE) if budget == "flops" else plan_agent.ParamModel(model, plan)
    curve, _half = alloc_walk._magnitude_curves(plan, kind_name, floor, cost.cost, cost.total0)
    steps = _steps(model, kind_name, floor)
    assert set(curve) == set(steps) == set(floor)
    kept = []
    for tau in sorted({0.0} | {loss for row_steps in steps.values() for _k, _w, loss in row_steps}):
        picked = {row: _picked(row_steps, tau) for row, row_steps in steps.items()}
        est = 1.0 - sum(alloc_walk._curve_at(curve[row], step[0])[1] for row, step in picked.items()) / cost.total0
        assert est == pytest.approx(cost.kept({row: step[1] for row, step in picked.items()}), abs=1e-9)
        kept.append(est)
    assert kept[0] == 1.0 and kept[-1] < 0.2                               # nothing cut at tau 0, every floor at last
    assert all(b <= a + 1e-12 for a, b in zip(kept, kept[1:]))


# ------------------------------------------------------------------ the plan


@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_the_plan_reads_no_batch_runs_no_probe_and_reports_tau_the_estimate_and_its_picks(monkeypatch, kind_name):
    monkeypatch.setattr(group_sensitivity, "group_sensitivity",
                        lambda *a, **k: pytest.fail("a magnitude kind must not run the sensitivity probe"))
    monkeypatch.setattr(group_sensitivity, "calib_loss", lambda *a, **k: pytest.fail("a magnitude kind reads no batch"))
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    model = _model()
    rows = _rows(model)
    widths, info = alloc_walk.plan_targets(model, None, SHAPE, kind_name, 0.5)
    assert list(info) == NEW_INFO
    assert info["kind"] == kind_name and info["held"] == 0 and info["cost_budget"] == "params" and info["target"] == 0.5
    assert isinstance(info["tau"], float) and info["tau"] > 0.0 and 0.0 < info["est_kept"] < 1.0
    assert set(widths) == set(info["keeps"]) == set(info["weights"]) == set(info["sens"]) == set(rows)
    assert all(isinstance(v, float) and v > 0.0 for v in info["sens"].values())
    assert _lands(info, 0.5) and abs(info["est_kept"] - 0.5) < 0.05       # the cost model at the picks; the landing
    assert alloc_walk.grid_priority(info, rows) == {row: float(info["weights"][row]) for row in rows}
    json.dumps({key: info[key] for key in ("tau", "est_kept", "weights", "sens", "keeps")})
    assert len(lines) == 1 and lines[0].startswith(f"[alloc] {kind_name}: one threshold tau=")
    assert f"of {len(rows)} of {len(rows)} groups" in lines[0] and "of the params" in lines[0]


@pytest.mark.parametrize("budget", ["params", "flops"])
@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_the_plan_lands_on_the_target_under_both_budgets(kind_name, budget):
    model = _model("r20_chenyaofo")
    plan = _plan(model)
    rows = [row for _group, row in plan]
    widths, info = alloc_walk.plan_targets(model, [], SHAPE, kind_name, 0.6, budget=budget)
    size = (lambda m: utils.calc_flops(m, SHAPE)) if budget == "flops" else utils.calc_num_parameters
    cut = alloc_walk.cut_to(model, plan, info["keeps"], SHAPE)
    kept = size(cut) / size(model)
    assert kept == pytest.approx(info["kept"]) and abs(kept - 0.6) < 0.02
    assert alloc_walk.group_widths(cut, rows) == widths and info["cost_budget"] == budget
    assert abs(info["est_kept"] - 0.6) < 0.02                              # the threshold alone is already close
    del cut


@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_a_deeper_target_lands_on_the_thin_net(kind_name):
    model = _model()
    widths, info = alloc_walk.plan_targets(model, [], SHAPE, kind_name, 0.3)
    assert _lands(info, 0.3) and info["cost_budget"] == "params" and set(widths) == set(_rows(model))
    assert all(k >= 0.1 - 1e-12 for k in info["keeps"].values())
    origin = info["origin_widths"]
    assert all(1 <= widths[row] <= origin[row] for row in widths)


@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_the_keep_floor_the_width_floor_and_the_keep_grid_hold(kind_name):
    model = _model()
    widths, info = alloc_walk.plan_targets(model, [], SHAPE, kind_name, 0.3, min_width=2)
    origin = info["origin_widths"]
    for row, width in widths.items():
        floor = max(0.1, min(1.0, 2.0 / origin[row]))                      # keep_floor and the walk's width floor
        assert width >= min(2, origin[row])
        assert info["keeps"][row] >= floor - 1e-12
        pick = info["weights"][row]                                        # the threshold's pick: the grid or the floor
        assert pick >= floor - 1e-12 and (pick in GRID or pick == floor)


# ------------------------------------------------------------------ what the rules respond to


@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_scaling_every_filter_of_one_group_changes_neither_its_scores_nor_the_plan(kind_name):
    model = _vgg()
    scores = _scores(model, kind_name)
    widths, info = alloc_walk.plan_targets(model, [], SHAPE, kind_name, 0.6)
    tenfold = copy.deepcopy(model)
    tenfold.features[3].weight.data.mul_(10.0)                             # all 32 filters of the second conv, 10x
    after = _scores(tenfold, kind_name)
    assert all(after[row] == pytest.approx(scores[row], rel=1e-5) for row in scores)
    eightfold = copy.deepcopy(model)
    eightfold.features[3].weight.data.mul_(8.0)                            # a power of two: every float scales exactly
    assert _scores(eightfold, kind_name) == scores
    widths8, info8 = alloc_walk.plan_targets(eightfold, [], SHAPE, kind_name, 0.6)
    assert widths8 == widths and info8["keeps"] == info["keeps"] and info8["tau"] == info["tau"]


@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_the_group_whose_small_filters_are_scaled_up_10x_is_kept_wider_than_uniform_keeps_it(kind_name):
    model = _vgg()
    for layer in (model.features[0], model.features[3], model.classifier[1]):
        layer.weight.data[::2] *= 0.1                                      # every group: half its filters are small
    model.features[3].weight.data[::2] *= 10.0                             # the second conv's small filters, 10x
    rows = _rows(model)
    conv2, linear = rows[1], rows[2]
    assert [alloc_walk.group_widths(model, rows)[row] for row in rows] == [16, 32, 64]
    uniform, uniform_info = alloc_walk.plan_targets(model, [], SHAPE, "uniform", 0.75)
    widths, info = alloc_walk.plan_targets(model, [], SHAPE, kind_name, 0.75)
    assert abs(info["kept"] - 0.75) < 0.02 and abs(uniform_info["kept"] - 0.75) < 0.02
    assert widths[conv2] > uniform[conv2] and info["keeps"][conv2] > uniform_info["keeps"][conv2] + 0.05
    assert widths[linear] < uniform[linear]                                # the small half of the linear layer pays


@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_with_no_group_score_at_all_the_plan_is_the_uniform_plan(monkeypatch, kind_name):
    model = _vgg()
    uniform, uniform_info = alloc_walk.plan_targets(model, [], SHAPE, "uniform", 0.6)
    monkeypatch.setattr(alloc_walk, "_group_magnitudes", lambda group: None)
    widths, info = alloc_walk.plan_targets(model, [], SHAPE, kind_name, 0.6)
    assert widths == uniform and info["keeps"] == uniform_info["keeps"] and info["kept"] == uniform_info["kept"]
    assert info["tau"] == 0.0 and info["est_kept"] == 1.0
    assert set(info["weights"].values()) == {1.0} and set(info["sens"].values()) == {0.0}


def test_a_group_the_cut_has_no_score_for_is_left_out_of_the_threshold(monkeypatch):
    model = _vgg()
    rows = _rows(model)
    real = alloc_walk._group_magnitudes
    monkeypatch.setattr(alloc_walk, "_group_magnitudes", lambda group: None if group.width == 16 else real(group))
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    widths, info = alloc_walk.plan_targets(model, [], SHAPE, "lamp", 0.6)
    assert info["weights"][rows[0]] == 1.0 and info["sens"][rows[0]] == 0.0 and "of 2 of 3 groups" in lines[0]
    assert set(widths) == set(rows) and abs(info["kept"] - 0.6) < 0.04


# ------------------------------------------------------------------ the walk's state, and the old kinds


@pytest.mark.parametrize("grid_round", [False, True])
@pytest.mark.parametrize("kind_name", NEW_KINDS)
def test_the_walks_state_plans_with_the_kind_logs_it_and_records_it(monkeypatch, kind_name, grid_round):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", kind_name)
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")                 # the registered alloc cells walk group-once
    if grid_round:
        monkeypatch.setenv("SPECTRA_ALLOC_GRID_ROUND", "1")
    monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: [])
    monkeypatch.setattr(group_sensitivity, "group_sensitivity",
                        lambda *a, **k: pytest.fail("a magnitude kind must not run the sensitivity probe"))
    lines, records = [], []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
    model = _model()
    rows = _rows(model)
    origin = alloc_walk.group_widths(model, rows)
    entry = alloc_walk._state(_alloc_env(model, 0.6))
    assert set(entry["widths"]) == set(rows) and not entry["fallback"] and ("grid" in entry) == grid_round
    assert all(1 <= entry["widths"][row] <= origin[row] for row in rows)
    assert lines[0].startswith(f"[alloc] {kind_name}: one threshold tau=") and "(target x0.580)" in lines[0]
    assert lines[1].startswith(f"[alloc] net.pt: {kind_name} alpha=0.5 plan keeps x")
    plans = [fields for event, fields in records if event == "alloc_plan"]
    assert len(plans) == 1 and plans[0]["kind"] == kind_name and set(plans[0]["rows"]) == {str(row) for row in rows}
    rounds = [fields for event, fields in records if event == "alloc_grid_round"]
    if grid_round:                                                         # the rounding ranks the groups by the picks
        assert any(line.startswith("[alloc] grid round: ") for line in lines)
        assert [fields["kind"] for fields in rounds] == [kind_name]
    else:
        assert rounds == [] and len(lines) == 2


def test_an_old_kind_plans_as_before_with_its_own_info_and_no_threshold_line(monkeypatch):
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    model = _vgg()
    _widths, info = alloc_walk.plan_targets(model, [], SHAPE, "uniform", 0.6)
    assert lines == [] and list(info) == ["kind", "alpha", "target", "kept", "keeps", "origin_widths", "sens", "held"]
    assert alloc_walk.grid_priority(info, list(info["keeps"])) is None
