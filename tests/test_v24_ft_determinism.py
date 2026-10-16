"""
SPECTRA_EVAL_FT_DETERMINISTIC (v24, default off): deterministic GPU algorithms for the evaluation's fine-tunes.
CPU only, no datasets. The flag is applied by ``fortify.apply_ft_determinism``; the runner calls it once, before the
first CUDA use.

    python -m pytest tests/test_v24_ft_determinism.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from src import fortify  # noqa: E402

FLAG = "SPECTRA_EVAL_FT_DETERMINISTIC"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Flag unset, CUBLAS_WORKSPACE_CONFIG unset, the idempotency latch reset; torch's global state restored after."""
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "placeholder")      # records "was absent" so teardown removes it
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    monkeypatch.setattr(fortify, "_FT_DETERMINISM_APPLIED", False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    restore = torch.use_deterministic_algorithms  # the real one: a test may patch it, and this teardown runs first
    yield
    restore(enabled, warn_only=warn_only)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def test_flag_defaults_off_and_apply_touches_nothing(monkeypatch):
    assert not fortify.eval_ft_deterministic()

    def boom(*a, **k):
        raise AssertionError("torch.use_deterministic_algorithms called with SPECTRA_EVAL_FT_DETERMINISTIC unset")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", boom)
    before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
              torch.are_deterministic_algorithms_enabled())
    lines = []
    assert fortify.apply_ft_determinism(lines.append) is False
    assert (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.are_deterministic_algorithms_enabled()) == before
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ and lines == []
    assert fortify._FT_DETERMINISM_APPLIED is False


def test_flag_parses(monkeypatch):
    for raw, expected in (("1", True), ("true", True), ("on", True), ("0", False), ("", False)):
        monkeypatch.setenv(FLAG, raw)
        assert fortify.eval_ft_deterministic() is expected


def test_on_calls_use_deterministic_algorithms_with_warn_only(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: calls.append((a, k)))
    assert fortify.apply_ft_determinism(lambda s: None) is True
    assert calls == [((True,), {"warn_only": True})]


def test_on_sets_torch_state_for_real(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    assert fortify.apply_ft_determinism(lambda s: None) is True
    assert torch.are_deterministic_algorithms_enabled() and torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False


def test_on_sets_cublas_workspace_config_when_unset(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    fortify.apply_ft_determinism(lambda s: None)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_on_keeps_an_existing_cublas_workspace_config(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    lines = []
    fortify.apply_ft_determinism(lines.append)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert any("CUBLAS_WORKSPACE_CONFIG=:16:8" in ln for ln in lines)


def test_on_logs_one_line(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    lines = []
    fortify.apply_ft_determinism(lines.append)
    assert lines == ["[eval] deterministic GPU algorithms ON: cudnn.deterministic=1 benchmark=0 "
                     "CUBLAS_WORKSPACE_CONFIG=:4096:8 (warn_only)"]


def test_second_call_is_idempotent(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    lines = []
    assert fortify.apply_ft_determinism(lines.append) is True
    n = len(lines)
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: calls.append((a, k)))
    assert fortify.apply_ft_determinism(lines.append) is True
    assert len(lines) == n and calls == []


def test_warns_when_cuda_was_initialised_first(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    lines = []
    assert fortify.apply_ft_determinism(lines.append) is True
    warnings_ = [ln for ln in lines if "WARNING" in ln]
    assert len(warnings_) == 1 and "CUDA was initialised" in warnings_[0] and "cuBLAS determinism may not hold" in warnings_[0]
    assert any("deterministic GPU algorithms ON" in ln for ln in lines)        # still applied


def test_no_warning_when_cuda_not_initialised(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    lines = []
    fortify.apply_ft_determinism(lines.append)
    assert not any("WARNING" in ln for ln in lines)


def _tiny_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(4, 2))


@pytest.mark.parametrize("benchmark", [False, True])
def test_measure_throughput_with_flag_on_leaves_cudnn_benchmark_alone(monkeypatch, benchmark):
    monkeypatch.setenv(FLAG, "1")
    torch.backends.cudnn.benchmark = benchmark
    out = fortify.measure_throughput(_tiny_net(), (3, 8, 8), "cpu", batches=(2, 1), warmup_iters=1, timed_iters=2,
                                     repeats=2)
    assert set(out) == {2, 1}
    assert torch.backends.cudnn.benchmark is benchmark
