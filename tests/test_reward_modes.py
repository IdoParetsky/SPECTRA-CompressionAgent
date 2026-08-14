"""Reward-mode unit tests (no GPU)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf


def _init_static_conf(tau=5):
    if StaticConf.get_instance() is None:
        StaticConf(ConfigurationValues(
            device=torch.device("cpu"), test_name="unit-test", input_dict={},
            compression_rates_dict={0: 1.0, 1: 0.9, 2: 0.8},
            runtime_limit=60, num_epochs=0, train_compressed_layer_only=True,
            allowed_acc_reduction=tau, discount_factor=0.99, learning_rate=1e-3,
            rollout_limit=10, passes=1, prune=True, seed=42, n_splits=0,
            train_split=0.7, val_split=0.2, database_dict={},
            actor_checkpoint_path=None, critic_checkpoint_path=None,
            save_pruned_checkpoints=False, test_ts="ts",
        ))
    else:
        StaticConf.get_instance().conf_values.allowed_acc_reduction = tau


def test_neon_matches_legacy_trichotomy(monkeypatch):
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "neon")
    _init_static_conf(5)
    from src.utils import compute_reward
    assert abs(compute_reward(0.97, 1.0, 0.9) - 10.0) < 1e-9
    assert abs(compute_reward(0.90, 1.0, 0.9) - (-(10.0 ** 3))) < 1e-6
    assert abs(compute_reward(1.01, 1.0, 0.9) - (10.0 ** 3)) < 1e-6


def test_structural_uses_realized_params(monkeypatch):
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural")
    _init_static_conf(10)
    from src.utils import compute_reward
    r = compute_reward(0.99, 1.0, 0.9, params_before=1000, params_after=950)
    assert abs(r - 5.0) < 1e-9
    r2 = compute_reward(0.80, 1.0, 0.9, params_before=1000, params_after=950)
    assert abs(r2 - (-(5.0 ** 3))) < 1e-6


def test_shaped_softens_near_cliff(monkeypatch):
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_shaped")
    _init_static_conf(10)
    from src.utils import compute_reward
    r = compute_reward(0.95, 1.0, 0.9, params_before=100, params_after=90)
    assert abs(r - 2.5) < 1e-9


def test_structural_guard_uses_nominal_on_violation(monkeypatch):
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_guard")
    _init_static_conf(10)
    from src.utils import compute_reward
    # Tiny realized prune but large Δacc violation → penalty uses nominal 10%
    r = compute_reward(0.80, 1.0, 0.9, params_before=1000, params_after=995)
    assert abs(r - (-(10.0 ** 3))) < 1e-6
    # In-budget mild loss → realized 0.5% credit
    r2 = compute_reward(0.95, 1.0, 0.9, params_before=1000, params_after=995)
    assert abs(r2 - 0.5) < 1e-9


def test_masked_noop_does_not_get_neon_compression_credit():
    from src.NetworkEnv import reward_compression_rate

    tau = 10
    # In-budget mask, numel unchanged → identity rate (zero NEON compression credit)
    assert reward_compression_rate({"mode": "masked"}, 0.8, 1000, 1000, 0.95, 1.0, tau) == 1.0
    # Over-budget still uses the nominal rate so wrecking an unprunable layer is punished
    assert reward_compression_rate({"mode": "masked"}, 0.8, 1000, 1000, 0.80, 1.0, tau) == 0.8
    # Structural shrink keeps the action's rate
    assert reward_compression_rate({"mode": "structural"}, 0.8, 1000, 800, 0.95, 1.0, tau) == 0.8
