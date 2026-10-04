"""CPU tests for the allocation headroom probe (scripts/allocation_probe.py).

    python -m pytest tests/test_allocation_probe.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import allocation_probe as ap  # noqa: E402
import selection_probe as sp  # noqa: E402
import src.utils as utils  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402

SHAPE = (3, 32, 32)


def _model(seed=0):
    torch.manual_seed(seed)
    model = resnet20(10, False, 4).eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.1, 0.1)
                m.running_mean.uniform_(-0.1, 0.1)
                m.running_var.uniform_(0.5, 1.5)
    return model


def _batches(n=2, size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(size, 3, 32, 32, generator=g), torch.randint(0, 10, (size,), generator=g))
            for _ in range(n)]


def test_cut_alloc_leaves_keep_one_alone_and_a_uniform_keep_matches_the_s0_cut():
    model = _model()
    plan = sp.cut_plan(model)
    same, modes = ap.cut_alloc(model, plan, {key: 1.0 for key, _, _ in plan}, SHAPE)
    assert modes == ["identity"] * len(plan)
    assert sp.shape_signature(same) == sp.shape_signature(model)
    half, modes = ap.cut_alloc(model, plan, {key: 0.5 for key, _, _ in plan}, SHAPE)
    assert modes == ["structural"] * len(plan)
    with sp.RankingOverride() as override:
        reference, _, _ = sp.cut(model, plan, 0.5, None, override, SHAPE)
    assert sp.shape_signature(half) == sp.shape_signature(reference) != sp.shape_signature(model)


def test_each_group_is_cut_to_its_own_keep():
    model = _model(1)
    plan = sp.cut_plan(model)
    keeps = {key: (0.5 if i % 2 else 1.0) for i, (key, _, _) in enumerate(plan)}
    cut_model, _ = ap.cut_alloc(model, plan, keeps, SHAPE)
    widths = ap.group_widths(cut_model)
    for i, (key, group, _) in enumerate(plan):
        if i % 2:
            assert widths[key] < group.width
        else:
            assert widths[key] == group.width


def test_match_params_reaches_the_target_for_uniform_and_random_weights():
    model = _model(2)
    plan = sp.cut_plan(model)
    params0 = utils.calc_num_parameters(model)
    keeps, frac, cut_model, _ = ap.match_params(model, plan, ap.allocation_weights("uniform", plan), 0.6,
                                                SHAPE, params0)
    assert abs(frac - 0.6) < 0.06
    assert utils.calc_num_parameters(cut_model) / params0 == pytest.approx(frac)
    assert len({round(v, 9) for v in keeps.values()}) == 1
    weights = ap.allocation_weights("random", plan, rng=np.random.default_rng([0, 1]), sigma=0.35)
    keeps, frac, _, _ = ap.match_params(model, plan, weights, 0.6, SHAPE, params0)
    assert abs(frac - 0.6) < 0.06
    assert len({round(v, 9) for v in keeps.values()}) > 1
    assert all(0.1 <= v <= 1.0 for v in keeps.values())


def test_match_params_bisects_on_a_given_measure():
    model = _model(2)
    plan = sp.cut_plan(model)
    params0 = utils.calc_num_parameters(model)
    cpu = torch.device("cpu")
    flops0 = utils.calc_flops(model, SHAPE, cpu)

    def measure(cut):
        return utils.calc_flops(cut, SHAPE, cpu) / flops0

    weights = ap.allocation_weights("random", plan, rng=np.random.default_rng([0, 1]), sigma=0.35)
    keeps, frac, cut_model, _ = ap.match_params(model, plan, weights, 0.5, SHAPE, params0, measure=measure)
    assert abs(frac - 0.5) < 0.06
    assert measure(cut_model) == pytest.approx(frac)


def test_sensitivity_is_the_loss_after_cutting_that_group_alone():
    model = _model(3)
    plan = sp.cut_plan(model)
    batches = _batches(seed=4)
    sens, base = ap.group_sensitivity(model, plan, batches, 0.5, SHAPE)
    assert set(sens) == {key for key, _, _ in plan}
    assert all(math.isfinite(v) for v in sens.values())
    assert base == pytest.approx(ap.calib_loss(model, batches))
    entry = plan[1]
    alone, _ = ap.cut_alloc(model, [entry], {entry[0]: 0.5}, SHAPE)
    assert ap.calib_loss(alone, batches) - base == pytest.approx(sens[entry[0]], abs=1e-6)


def test_weights_follow_sensitivity_and_random_draws_are_reproducible():
    plan = sp.cut_plan(_model(4))
    keys = [key for key, _, _ in plan]
    sens = {key: float(i + 1) for i, key in enumerate(keys)}
    up = ap.allocation_weights("sens", plan, sens, alpha=0.5)
    down = ap.allocation_weights("anti", plan, sens, alpha=0.5)
    twice = ap.allocation_weights("sens2", plan, sens, alpha=0.5)
    assert [up[k] for k in keys] == sorted(up[k] for k in keys)
    assert [down[k] for k in keys] == sorted((down[k] for k in keys), reverse=True)
    assert all(twice[k] == pytest.approx(up[k] ** 2) for k in keys)
    sens[keys[0]] = -0.3
    floored = ap.allocation_weights("sens", plan, sens, alpha=0.5)
    assert all(math.isfinite(v) and v > 0 for v in floored.values())
    a = ap.allocation_weights("random", plan, rng=np.random.default_rng([0, 1]))
    b = ap.allocation_weights("random", plan, rng=np.random.default_rng([0, 1]))
    c = ap.allocation_weights("random", plan, rng=np.random.default_rng([0, 2]))
    assert a == b != c
    with pytest.raises(ValueError):
        ap.allocation_weights("greedy", plan, sens)


def _row(alloc, dval, dtest, draw=0, seed=0, params=0.6):
    return {"keep": 0.6, "budget": "40", "alloc": alloc, "draw": draw, "ft_seed": seed,
            "d_val_pp": dval, "d_test_pp": dtest, "params_kept": params, "flops_kept": params}


def test_summarize_applies_the_registered_calls():
    uniform = [_row("uniform", v, v, seed=i) for i, v in enumerate((-2.0, -2.2, -1.8))]  # sd 0.2 -> bar 0.5
    headroom = uniform + [_row("sens", -1.2, -1.5), _row("anti", -3.0, -3.0), _row("random", -2.5, -2.4)]
    cell = ap.summarize(headroom, 0.6, 40)
    assert cell["bar"] == pytest.approx(0.5) and cell["call"] == "HEADROOM"
    assert ap.summarize(uniform + [_row("sens", -1.8, -1.0)], 0.6, 40)["call"] == "FLAT"
    harm = uniform + [_row("sens", -2.6, -2.0), _row("anti", -3.0, -3.0), _row("random", -2.9, -1.0)]
    assert ap.summarize(harm, 0.6, 40)["call"] == "HARM"
    unmatched = uniform + [_row("sens", -0.5, -0.5, params=0.65), _row("anti", -2.1, -2.1)]
    cell = ap.summarize(unmatched, 0.6, 40)
    assert cell["call"] == "FLAT" and cell["allocs"]["sens"]["matched"] is False
    best = uniform + [_row("random", -2.5, -1.0, draw=0), _row("random", -1.1, -1.8, draw=1)]
    cell = ap.summarize(best, 0.6, 40)
    assert cell["random_valbest"] == "random1" and cell["call"] == "HEADROOM"
    best_but_test_worse = uniform + [_row("random", -1.1, -2.5, draw=1)]
    assert ap.summarize(best_but_test_worse, 0.6, 40)["call"] == "FLAT"
    assert ap.summarize(uniform[:1], 0.6, 40) is None


def test_summarize_reads_the_matched_check_on_flops_when_asked():
    uniform = [dict(_row("uniform", v, v, seed=i), flops_kept=0.5) for i, v in enumerate((-2.0, -2.2, -1.8))]
    sens = dict(_row("sens", -1.2, -1.5, params=0.7), flops_kept=0.5)
    assert ap.summarize(uniform + [sens], 0.6, 40)["allocs"]["sens"]["matched"] is False
    cell = ap.summarize(uniform + [sens], 0.6, 40, match_key="flops_kept")
    assert cell["allocs"]["sens"]["matched"] is True
    assert cell["call"] == "HEADROOM" and cell["match"] == "flops"
