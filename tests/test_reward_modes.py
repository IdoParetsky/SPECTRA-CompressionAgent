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


def test_structural_band_grades_violations_by_overshoot_not_cut_size(monkeypatch):
    """
    §52.1: under -reduction**3 every over-budget cut is punished in proportion to its
    size, so the only signal is "cut less" and a net whose band is empty contributes
    nothing about *where* to cut. structural_band penalises the accuracy overshoot.
    """
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_band")
    monkeypatch.delenv("SPECTRA_REWARD_SCALE", raising=False)
    _init_static_conf(10)
    from src.utils import compute_reward

    # Same 15 pp drop (5 pp past tau) from a small and a large cut -> same penalty
    small = compute_reward(0.85, 1.0, 0.9, params_before=1000, params_after=990)
    large = compute_reward(0.85, 1.0, 0.9, params_before=1000, params_after=600)
    assert abs(small - (-(5.0 ** 3))) < 1e-6
    assert small == large

    # A worse drop at the same cut size is penalised harder
    worse = compute_reward(0.75, 1.0, 0.9, params_before=1000, params_after=990)
    assert worse < small

    # Any violation is still strictly worse than a no-op, and in-budget beats both
    identity = compute_reward(0.99, 1.0, 1.0, params_before=1000, params_after=1000)
    in_budget = compute_reward(0.95, 1.0, 0.9, params_before=1000, params_after=900)
    assert worse < small < 0 <= identity < in_budget


def test_structural_band_keeps_the_neon_arms(monkeypatch):
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_band")
    monkeypatch.delenv("SPECTRA_REWARD_SCALE", raising=False)
    _init_static_conf(10)
    from src.utils import compute_reward

    # In-budget: realized reduction, exactly like structural
    assert abs(compute_reward(0.95, 1.0, 0.9, params_before=1000, params_after=900)
               - 10.0) < 1e-9
    # Accuracy gain: cubed realized reduction, exactly like structural
    assert abs(compute_reward(1.01, 1.0, 0.9, params_before=1000, params_after=900)
               - (10.0 ** 3)) < 1e-6


def test_cbrt_scale_is_monotone_and_off_by_default(monkeypatch):
    _init_static_conf(10)
    from src.utils import apply_reward_scale, compute_reward, reward_scale_name

    monkeypatch.delenv("SPECTRA_REWARD_SCALE", raising=False)
    assert reward_scale_name() == "raw"
    assert apply_reward_scale(-8000.0) == -8000.0

    monkeypatch.setenv("SPECTRA_REWARD_SCALE", "cbrt")
    assert abs(apply_reward_scale(8000.0) - 20.0) < 1e-9
    assert abs(apply_reward_scale(-8000.0) + 20.0) < 1e-9
    assert apply_reward_scale(0.0) == 0.0

    # Strictly monotone: the per-step preference ordering is unchanged
    raw = [-8000.0, -125.0, -1e-3, 0.0, 5.0, 1000.0]
    scaled = [apply_reward_scale(v) for v in raw]
    assert scaled == sorted(scaled)

    # Applied through compute_reward for every mode
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "neon")
    assert abs(compute_reward(0.85, 1.0, 0.9) - (-10.0)) < 1e-6


def test_reward_branch_labels_the_trichotomy():
    from src.utils import reward_branch

    assert reward_branch(-12.0, 10) == "over_budget"
    assert reward_branch(-3.0, 10) == "in_budget"
    assert reward_branch(0.0, 10) == "in_budget"
    assert reward_branch(0.4, 10) == "gain"


def test_reward_trace_records_branch_per_step(monkeypatch, tmp_path):
    """C9 Adam-40 landed VGG-16 C100 at -7.8 pp (in budget) and thin r20 at -17.1 (cliff)."""
    import json

    monkeypatch.setenv("SPECTRA_REWARD_MODE", "neon")
    monkeypatch.setenv("SPECTRA_REWARD_TRACE", "1")
    monkeypatch.setenv("SPECTRA_RUN_DIR", str(tmp_path))
    _init_static_conf(10)
    from src.utils import compute_reward, trace_reward

    for net, new_acc in (("vgg16_bn_cifar100_chenyaofo_74.pt", 0.662),
                         ("resnet20-width16_cifar100_thin-res-net_72.98.pt", 0.569)):
        reward = compute_reward(new_acc, 0.740, 0.9)
        trace_reward(net, 0.9, new_acc, 0.740, reward)

    rows = [json.loads(l) for l in (tmp_path / "reward_trace.jsonl").read_text().splitlines()]
    assert [r["dataset"] for r in rows] == ["cifar-100", "cifar-100"]
    assert [r["branch"] for r in rows] == ["in_budget", "over_budget"]
    assert rows[0]["reward"] > 0 > rows[1]["reward"]


def test_reward_trace_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SPECTRA_REWARD_TRACE", raising=False)
    monkeypatch.setenv("SPECTRA_RUN_DIR", str(tmp_path))
    _init_static_conf(10)
    from src.utils import trace_reward

    trace_reward("vgg16_bn_cifar10_chenyaofo.pt", 0.9, 0.93, 0.94, 10.0)
    assert not (tmp_path / "reward_trace.jsonl").exists()


def test_masked_noop_does_not_get_neon_compression_credit():
    from src.NetworkEnv import reward_compression_rate

    tau = 10
    # In-budget mask, numel unchanged → identity rate (zero NEON compression credit)
    assert reward_compression_rate({"mode": "masked"}, 0.8, 1000, 1000, 0.95, 1.0, tau) == 1.0
    # Over-budget still uses the nominal rate so wrecking an unprunable layer is punished
    assert reward_compression_rate({"mode": "masked"}, 0.8, 1000, 1000, 0.80, 1.0, tau) == 0.8
    # Structural shrink keeps the action's rate
    assert reward_compression_rate({"mode": "structural"}, 0.8, 1000, 800, 0.95, 1.0, tau) == 0.8
