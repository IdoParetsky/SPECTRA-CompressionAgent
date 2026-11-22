"""
v39 ``SPECTRA_SENS_BN_RECAL=N`` (default 0 = off): the threshold rule's probes re-estimate every BatchNorm's running
statistics over ``N`` calibration batches before the loss is read (the base on a copy of the model, each cut on the cut
itself), so a group is scored on the network its cut leaves after recalibration. Flag off, ``group_sensitivity`` is
exactly what it was.

CPU only, no datasets.  python -m pytest tests/test_v39_sens_bn_recal.py -v
"""

import copy
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import SHAPE, ZOO, _plan, _randomize_norms  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.recovery_edits as recovery_edits  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG = "SPECTRA_SENS_BN_RECAL"
FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", FLAG)


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
    _randomize_norms(model)  # running stats far from what the batches give, so a recalibration moves them
    return model


def _batches(n=2, size=8):
    gen = torch.Generator().manual_seed(3)
    return [(torch.randn(size, *SHAPE, generator=gen), torch.randint(0, 10, (size,), generator=gen))
            for _ in range(n)]


def _norms(model):
    return [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]


def _same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


# ------------------------------------------------------------------ the flag


def test_flag_is_off_by_default():
    assert FLAG not in os.environ and group_sensitivity.sens_bn_recal() == 0


@pytest.mark.parametrize("value, want", [("", 0), ("0", 0), ("8", 8), (" 3 ", 3), ("00", 0)])
def test_flag_reads_a_non_negative_int(monkeypatch, value, want):
    monkeypatch.setenv(FLAG, value)
    assert group_sensitivity.sens_bn_recal() == want


@pytest.mark.parametrize("value", ["-1", "abc", "2.5", "1e1", "true"])
def test_an_invalid_value_raises(monkeypatch, value):
    monkeypatch.setenv(FLAG, value)
    with pytest.raises(ValueError, match="SPECTRA_SENS_BN_RECAL"):
        group_sensitivity.sens_bn_recal()


def test_group_sensitivity_raises_on_an_invalid_value_before_probing(monkeypatch):
    model = _model()
    plan = _plan(model)[:1]
    monkeypatch.setenv(FLAG, "many")
    with pytest.raises(ValueError, match="SPECTRA_SENS_BN_RECAL"):
        group_sensitivity.group_sensitivity(model, plan, _batches(1), SHAPE)


def test_the_flag_is_not_a_policy_contract_key():
    # read as text: importing the agent module pulls in more than a CPU test needs
    text = (Path(__file__).resolve().parents[1] / "src" / "A2C_Agent_Reinforce.py").read_text(encoding="utf-8")
    start = text.index("POLICY_CONTRACT_KEYS = (")
    block = text[start:text.index("ACTION_GEOMETRY_KEYS", start)]
    assert "SPECTRA_STATE_SENS" in block and FLAG not in block


# ------------------------------------------------------------------ recalibrate_batchnorm takes the calibration list


def test_recalibrate_batchnorm_takes_a_list_of_tuples_and_stops_after_n_batches():
    model = _model().train()
    batches = _batches(3)
    moved = recovery_edits.recalibrate_batchnorm(model, batches, torch.device("cpu"), n_batches=1)
    assert moved == len(_norms(model)) > 0 and model.training
    assert all(int(m.num_batches_tracked) == 1 for m in _norms(model))


# ------------------------------------------------------------------ flag off


def test_flag_off_calls_no_recalibration_prints_nothing_and_is_the_unset_result(monkeypatch):
    model, batches = _model(), _batches()
    plan = _plan(model)
    before = copy.deepcopy(model.state_dict())
    ref_sens, ref_base = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE)
    calls, lines = [], []
    monkeypatch.setattr(recovery_edits, "recalibrate_batchnorm", lambda *a, **k: calls.append(1) or 0)
    monkeypatch.setattr(utils, "print_flush", lambda msg: lines.append(str(msg)))
    for value in (None, "", "0"):
        if value is None:
            monkeypatch.delenv(FLAG, raising=False)
        else:
            monkeypatch.setenv(FLAG, value)
        sens, base = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE)
        assert sens == ref_sens and base == ref_base and list(sens) == [row for _g, row in plan]
    assert calls == [] and not [l for l in lines if "bn recal" in l]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


# ------------------------------------------------------------------ flag on


def test_flag_on_is_finite_leaves_the_input_model_alone_and_logs_once(monkeypatch):
    model, batches = _model(), _batches()
    plan = _plan(model)[:4]
    ref = copy.deepcopy(model)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda msg: lines.append(str(msg)))
    monkeypatch.setenv(FLAG, "2")
    sens, base = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE)
    assert torch.isfinite(torch.tensor(base)) and list(sens) == [row for _g, row in plan]
    assert all(torch.isfinite(torch.tensor(v)) for v in sens.values())
    assert _same_state(model, ref) and not model.training  # weights, BN running stats, num_batches_tracked, mode
    assert [l for l in lines if "bn recal" in l] == ["[sens] bn recal 2 batches on every probe"]


def test_flag_on_recalibrates_the_base_copy_and_every_probed_cut_and_moves_the_stats(monkeypatch):
    model, batches = _model(), _batches()
    plan = _plan(model)[:3]
    real = recovery_edits.recalibrate_batchnorm
    seen = []

    def counting(m, loader, device, n_batches=8):
        before = [n.running_mean.clone() for n in _norms(m)]
        out = real(m, loader, device, n_batches=n_batches)
        moved = any(not torch.equal(b, n.running_mean) for b, n in zip(before, _norms(m)))
        seen.append((m, loader, n_batches, moved))
        return out

    monkeypatch.setattr(recovery_edits, "recalibrate_batchnorm", counting)
    monkeypatch.setenv(FLAG, "2")
    ref = copy.deepcopy(model)
    group_sensitivity.group_sensitivity(model, plan, batches, SHAPE)
    assert len(seen) == len(plan) + 1                      # the base, then once per probed group
    assert all(s[0] is not model for s in seen)            # never the caller's model
    assert all(s[1] is batches and s[2] == 2 for s in seen)
    assert all(s[3] for s in seen)                         # the running stats moved inside every probe
    assert _same_state(model, ref)


def test_flag_on_changes_the_scores_and_reads_the_base_on_the_recalibrated_copy(monkeypatch):
    model, batches = _model(), _batches()
    plan = _plan(model)[:3]
    off_sens, off_base = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE)
    monkeypatch.setenv(FLAG, "2")
    on_sens, on_base = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE)
    copy_model = copy.deepcopy(model)
    recovery_edits.recalibrate_batchnorm(copy_model, batches, torch.device("cpu"), n_batches=2)
    assert on_base == group_sensitivity.calib_loss(copy_model, batches)
    assert on_base != off_base and on_sens != off_sens


def test_flag_on_still_fills_costs_like_flag_off(monkeypatch):
    model, batches = _model(), _batches()
    plan = _plan(model)[:3]
    off_costs = {}
    group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, costs=off_costs)
    monkeypatch.setenv(FLAG, "2")
    on_costs = {}
    group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, costs=on_costs)
    assert on_costs == off_costs and list(on_costs) == [row for _g, row in plan]
