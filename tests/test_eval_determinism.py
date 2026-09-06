"""Frozen-policy eval determinism and the within-step size-probe cache (no GPU)."""

import sys
import types
from pathlib import Path

import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.fortify as fortify

CPU = torch.device("cpu")


def _dist(probs):
    return Categorical(probs=torch.tensor([probs], dtype=torch.float32))


def test_eval_is_stochastic_by_default(monkeypatch):
    """Already-quoted TEST rows were produced by sampling; do not change them silently."""
    monkeypatch.delenv("SPECTRA_EVAL_DETERMINISTIC", raising=False)
    assert fortify.eval_deterministic() is False

    legal = torch.tensor([True, True, True])
    seen = {int(fortify.policy_action(_dist([0.34, 0.33, 0.33]), legal, device=CPU).item())
            for _ in range(200)}
    assert len(seen) > 1


def test_deterministic_eval_takes_the_argmax(monkeypatch):
    monkeypatch.setenv("SPECTRA_EVAL_DETERMINISTIC", "1")
    legal = torch.tensor([True, True, True])
    for _ in range(20):
        action = fortify.policy_action(_dist([0.2, 0.5, 0.3]), legal, device=CPU)
        assert action.shape == (1,)
        assert int(action.item()) == 1


def test_deterministic_eval_respects_the_legal_mask(monkeypatch):
    """Argmax must be taken *after* masking, or a fortified stem row could be pruned."""
    monkeypatch.setenv("SPECTRA_EVAL_DETERMINISTIC", "1")
    legal = torch.tensor([True, False, True])
    action = fortify.policy_action(_dist([0.2, 0.5, 0.3]), legal, device=CPU)
    assert int(action.item()) == 2


def test_set_policy_eval_mode_only_fires_when_deterministic(monkeypatch):
    """The state encoder carries dropout; leaving it in train mode perturbs TEST logits."""
    model = torch.nn.Sequential(torch.nn.Dropout(0.1))
    model.train()

    monkeypatch.delenv("SPECTRA_EVAL_DETERMINISTIC", raising=False)
    fortify.set_policy_eval_mode(model, None)
    assert model.training is True

    monkeypatch.setenv("SPECTRA_EVAL_DETERMINISTIC", "1")
    fortify.set_policy_eval_mode(model, None)
    assert model.training is False


def test_size_probe_cache_hits_within_a_step(monkeypatch):
    """Floor check, look-ahead and the Δparams/ΔFLOPs preference all ask for the same probe."""
    monkeypatch.delenv("SPECTRA_PREVIEW_CACHE", raising=False)
    from src.NetworkEnv import NetworkEnv

    env = types.SimpleNamespace(_ratio_cache={})
    calls = []

    def compute():
        calls.append(1)
        return (0.82, 0.71)

    for _ in range(5):
        assert NetworkEnv._cached_ratio(env, ("preview", 3, 0.8), compute) == (0.82, 0.71)
    assert len(calls) == 1

    # A different rate on the same row is a separate probe
    NetworkEnv._cached_ratio(env, ("preview", 3, 0.9), compute)
    assert len(calls) == 2

    # The model changed: NetworkEnv.step drops the cache
    env._ratio_cache = {}
    NetworkEnv._cached_ratio(env, ("preview", 3, 0.8), compute)
    assert len(calls) == 3


def test_size_probe_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SPECTRA_PREVIEW_CACHE", "0")
    from src.NetworkEnv import NetworkEnv

    env = types.SimpleNamespace(_ratio_cache={})
    calls = []

    def compute():
        calls.append(1)
        return 1.0

    for _ in range(3):
        NetworkEnv._cached_ratio(env, "param_ratio", compute)
    assert len(calls) == 3
