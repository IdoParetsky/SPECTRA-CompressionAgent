"""
v34 (5 Nov): two default-off flags that shorten an ImageNet eval decision, CPU only.

* ``SPECTRA_EVAL_LEAN_WALK=1``: on the TEST walk a cut step that reaches no size target skips the TEST read and the
  top-5 capture (``test_acc`` NaN); the first point reaching each size target is read in full; flag off, the stock
  read and capture;
* ``SPECTRA_EVAL_FINAL_FT_SIZE_ONLY=1``: the final fine-tune recovers only the size-point candidates.

    OMP_NUM_THREADS=4 python -m pytest tests/test_v34_lean_walk.py -v
"""

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from src import fortify  # noqa: E402
import src.utils as utils  # noqa: E402
import a2c_agent_reinforce_runner as runner  # noqa: E402

PREFIXES = ("SPECTRA_EVAL_", "SPECTRA_IMN_", "SPECTRA_FT_")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for key in list(os.environ):
        if key.startswith(PREFIXES):
            monkeypatch.delenv(key, raising=False)
    yield


# ------------------------------------------------------------------------------------------------ fixtures


class FakeEnv:
    """The few members ``_traj_cut_point`` touches; ``score_test_loader`` counts its calls."""

    def __init__(self, param=0.9, flop=0.9):
        self.param = param
        self.flop = flop
        self.last_val_acc = 0.74
        self.original_acc = 0.76
        self._origin_test_acc = 0.77
        self.reads = 0

    def score_test_loader(self):
        self.reads += 1
        return 0.5, 0.76, -26.0

    def param_ratio(self):
        return self.param

    def flops_ratio(self):
        return self.flop


@pytest.fixture
def captures(monkeypatch):
    seen = []

    def fake_capture(env, point):
        seen.append(dict(point))
        return point

    monkeypatch.setattr(runner.eval_topk, "capture", fake_capture)
    return seen


# ------------------------------------------------------------------------------------------------ flags


def test_flags_default_off_and_on(monkeypatch):
    assert fortify.eval_lean_walk() is False
    assert fortify.eval_final_ft_size_only() is False
    monkeypatch.setenv("SPECTRA_EVAL_LEAN_WALK", "1")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SIZE_ONLY", "1")
    assert fortify.eval_lean_walk() is True
    assert fortify.eval_final_ft_size_only() is True


# ------------------------------------------------------------------------------------------------ _traj_cut_point


def test_cut_point_flag_off_reads_and_captures(captures):
    env = FakeEnv(flop=0.70)
    reached = set()
    point, last_test = runner._traj_cut_point(env, 3, 0.9, None, (("flop", 0.48),), reached)
    assert env.reads == 1
    assert len(captures) == 1
    assert last_test == (0.5, 0.76)
    assert point["test_acc"] == 0.5
    assert point["test_origin"] == 0.76
    assert reached == set()  # the stock path tracks nothing


def test_cut_point_lean_skips_read_before_target(monkeypatch, captures):
    monkeypatch.setenv("SPECTRA_EVAL_LEAN_WALK", "1")
    env = FakeEnv(flop=0.70)
    reached = set()
    carried = (0.55, 0.78)
    point, last_test = runner._traj_cut_point(env, 3, 0.9, carried, (("flop", 0.48),), reached)
    assert env.reads == 0
    assert captures == []
    assert math.isnan(point["test_acc"])
    assert math.isnan(point["test_dacc_pp"])
    assert point["test_origin"] == carried[1]
    assert point["flop"] == 0.70
    assert point["step"] == 3
    assert last_test is carried
    assert reached == set()


def test_cut_point_lean_without_any_test_origin_is_nan(monkeypatch, captures):
    monkeypatch.setenv("SPECTRA_EVAL_LEAN_WALK", "1")
    env = FakeEnv(flop=0.70)
    env._origin_test_acc = None
    point, last_test = runner._traj_cut_point(env, 0, 0.9, None, (("flop", 0.48),), set())
    assert env.reads == 0 and last_test is None
    assert math.isnan(point["test_origin"]) and math.isnan(point["test_acc"])
    env._origin_test_acc = 0.77
    point, _ = runner._traj_cut_point(env, 0, 0.9, None, (("flop", 0.48),), set())
    assert point["test_origin"] == 0.77


def test_cut_point_lean_reads_once_on_reaching_target(monkeypatch, captures):
    monkeypatch.setenv("SPECTRA_EVAL_LEAN_WALK", "1")
    env = FakeEnv(flop=0.479)
    reached = set()
    targets = (("flop", 0.48),)
    point, last_test = runner._traj_cut_point(env, 10, 0.9, None, targets, reached)
    assert env.reads == 1
    assert len(captures) == 1
    assert reached == {("flop", 0.48)}
    assert last_test == (0.5, 0.76)
    assert point["test_acc"] == 0.5
    env.flop = 0.47  # already reached: no second read
    point, last_test2 = runner._traj_cut_point(env, 11, 0.9, last_test, targets, reached)
    assert env.reads == 1
    assert len(captures) == 1
    assert math.isnan(point["test_acc"])
    assert last_test2 is last_test


def test_cut_point_lean_two_targets(monkeypatch, captures):
    monkeypatch.setenv("SPECTRA_EVAL_LEAN_WALK", "1")
    env = FakeEnv()
    reached = set()
    targets = (("flop", 0.56), ("flop", 0.48))
    last = None
    reads = []
    for flop in (0.55, 0.50, 0.47):
        env.flop = flop
        before = env.reads
        point, last = runner._traj_cut_point(env, 0, 0.9, last, targets, reached)
        reads.append(env.reads - before)
    assert reads == [1, 0, 1]
    assert reached == {("flop", 0.56), ("flop", 0.48)}
    assert len(captures) == 2


def test_cut_point_lean_param_target_uses_param_ratio(monkeypatch, captures):
    monkeypatch.setenv("SPECTRA_EVAL_LEAN_WALK", "1")
    env = FakeEnv(param=0.40, flop=0.90)
    reached = set()
    runner._traj_cut_point(env, 0, 0.9, None, (("flop", 0.5), ("param", 0.42)), reached)
    assert env.reads == 1
    assert reached == {("param", 0.42)}


# ------------------------------------------------------------------------------------------------ _run_final_ft


def _candidate(step, param, flop):
    point = {"step": step, "param": param, "flop": flop, "val_dacc_pp": -1.0, "test_acc": 0.5}
    return {"point": point, "model": object(), "key": (param, flop, 1.0)}


@pytest.fixture
def logged(monkeypatch):
    """``utils.print_flush`` goes through the logger, so capture its messages directly."""
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda msg: lines.append(str(msg)))
    return lines


@pytest.fixture
def ft_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_final_ft", lambda env, net_path, tag, cand, epochs, **kw: calls.append(tag))
    return calls


def _final_ft_inputs():
    size_label = fortify.size_point_label("flop", 0.48)
    candidates = {"val_best": _candidate(9, 0.8, 0.8), size_label: _candidate(40, 0.4, 0.47)}
    picked = {"val_best": {"step": 9}}
    return size_label, candidates, picked


def test_run_final_ft_size_only_drops_val_best(monkeypatch, ft_calls, logged):
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "flop:0.48")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SIZE_ONLY", "1")
    size_label, candidates, picked = _final_ft_inputs()
    runner._run_final_ft(SimpleNamespace(), "/x/net.pt", picked, candidates, 1)
    assert ft_calls == [size_label]
    assert list(candidates) == [size_label]
    assert any("SIZE_ONLY drops val_best (step 9)" in line for line in logged)


def test_run_final_ft_size_only_with_no_size_candidate_returns(monkeypatch, ft_calls, logged):
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "flop:0.48")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SIZE_ONLY", "1")
    candidates = {"val_best": _candidate(9, 0.8, 0.8)}
    runner._run_final_ft(SimpleNamespace(), "/x/net.pt", {"val_best": {"step": 9}}, candidates, 1)
    assert ft_calls == []
    assert any("SIZE_ONLY found no size-point candidate; nothing recovered" in line for line in logged)


def test_run_final_ft_flag_off_keeps_both(monkeypatch, ft_calls):
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "flop:0.48")
    size_label, candidates, picked = _final_ft_inputs()
    runner._run_final_ft(SimpleNamespace(), "/x/net.pt", picked, candidates, 1)
    assert ft_calls == ["val_best", size_label]


# ------------------------------------------------------------------------------------------------ printing


def test_print_traj_point_with_nan_test_does_not_raise(monkeypatch):
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda msg: lines.append(str(msg)))
    nan = float("nan")
    point = {"step": 12, "rate": 0.9, "param": 0.6, "flop": 0.7, "val_acc": 0.74, "val_origin": 0.76,
             "val_dacc_pp": -2.0, "test_acc": nan, "test_origin": 0.76, "test_dacc_pp": nan}
    runner._print_traj_point("val_best", "net", point)
    assert len(lines) == 1 and "nan" in lines[0]
