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


def test_cbrt_cubes_keeps_inband_linear(monkeypatch):
    """Fable V5 / Ido 19 Sep: cbrt on cubed arms only. ρ=20 → in-band +20, miss −20, gain +20."""
    _init_static_conf(10)
    from src.utils import compute_reward

    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural")
    monkeypatch.setenv("SPECTRA_REWARD_SCALE", "cbrt_cubes")
    kwargs = dict(params_before=1000, params_after=800)

    in_band = compute_reward(0.95, 1.0, 0.8, **kwargs)   # Δacc −5, ρ=20
    miss = compute_reward(0.80, 1.0, 0.8, **kwargs)      # Δacc −20
    gain = compute_reward(1.01, 1.0, 0.8, **kwargs)      # Δacc +1
    assert abs(in_band - 20.0) < 1e-9
    assert abs(miss + 20.0) < 1e-6
    assert abs(gain - 20.0) < 1e-6
    # One legal cut pays for one miss (live cbrt: ~2.7 vs −20)
    assert abs(in_band + miss) < 1e-6

    monkeypatch.setenv("SPECTRA_REWARD_SCALE", "cbrt")
    in_band_cbrt = compute_reward(0.95, 1.0, 0.8, **kwargs)
    assert abs(in_band_cbrt - (20.0 ** (1.0 / 3.0))) < 1e-9
    assert in_band_cbrt < 3.0 < in_band


def test_cbrt_miss_keeps_the_raw_gain_cube(monkeypatch):
    """G2 cubic-gain train (Ido 1 Oct): miss −ρ, in-band +ρ, gain raw +ρ³; the other scales are unchanged."""
    _init_static_conf(10)
    from src.utils import apply_reward_scale, compute_reward

    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural")
    monkeypatch.setenv("SPECTRA_REWARD_SCALE", "cbrt_miss")
    big = dict(params_before=1000, params_after=800)      # ρ = 20
    assert abs(compute_reward(0.95, 1.0, 0.8, **big) - 20.0) < 1e-9
    assert abs(compute_reward(0.80, 1.0, 0.8, **big) + 20.0) < 1e-6
    assert abs(compute_reward(1.01, 1.0, 0.8, **big) - 20.0 ** 3) < 1e-6
    # Below ρ = 1 the cube pays a gain less than an in-band cut
    small = dict(params_before=1000, params_after=995)    # ρ = 0.5
    assert compute_reward(1.01, 1.0, 0.9, **small) < compute_reward(0.99, 1.0, 0.9, **small)
    # Only a cubed negative is rooted; an uncubed negative (shaped taper) passes through
    assert apply_reward_scale(-8.0, cubed=False) == -8.0
    assert apply_reward_scale(8000.0, cubed=True) == 8000.0
    assert abs(apply_reward_scale(-8000.0, cubed=True) + 20.0) < 1e-9
    # structural_band's miss arm is rooted as under cbrt_cubes
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_band")
    assert abs(compute_reward(0.85, 1.0, 0.8, **big) + 5.0) < 1e-6


def test_cbrt_default_unchanged_on_inband(monkeypatch):
    """Live v3/V4 keep shrinking the in-band arm. Do not change the default."""
    _init_static_conf(10)
    from src.utils import compute_reward, reward_scale_name

    monkeypatch.delenv("SPECTRA_REWARD_SCALE", raising=False)
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural")
    assert reward_scale_name() == "raw"
    assert abs(compute_reward(0.95, 1.0, 0.8, params_before=1000, params_after=800) - 20.0) < 1e-9


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


def test_structural_unified_identity_and_in_budget(monkeypatch):
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_unified")
    monkeypatch.setenv("SPECTRA_REWARD_SCALE", "cbrt")  # must be ignored
    monkeypatch.setenv("SPECTRA_UNIFIED_EPS", "1")
    _init_static_conf(10)
    from src.utils import compute_reward

    identity = compute_reward(0.99, 1.0, 1.0, params_before=1000, params_after=1000)
    assert abs(identity) < 1e-9

    # Δ = -5 pp, ρ_w = 10, u = 5 → R = 10 * 5/10 = 5 (not cubed, not cbrt)
    in_budget = compute_reward(0.95, 1.0, 0.9, params_before=1000, params_after=900)
    assert abs(in_budget - 5.0) < 1e-9

    # Continuity at τ: Δ = -10, u = 0, o = 0 → R = 0
    at_wall = compute_reward(0.90, 1.0, 0.9, params_before=1000, params_after=900)
    assert abs(at_wall) < 1e-9

    # Accuracy gain: u = τ+Δ = 11, R = 10 * 11/10 = 11
    gain = compute_reward(1.01, 1.0, 0.9, params_before=1000, params_after=900)
    assert abs(gain - 11.0) < 1e-9


def test_structural_unified_overshoot_ranks_harm_per_byte(monkeypatch):
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_unified")
    monkeypatch.delenv("SPECTRA_REWARD_SCALE", raising=False)
    monkeypatch.setenv("SPECTRA_UNIFIED_EPS", "1")
    _init_static_conf(10)
    from src.utils import compute_reward

    # Same 15 pp drop (o = 5). Larger cut is less negative: -o²/(ρ+ε)
    small = compute_reward(0.85, 1.0, 0.9, params_before=1000, params_after=990)  # ρ=1
    large = compute_reward(0.85, 1.0, 0.9, params_before=1000, params_after=600)  # ρ=40
    assert abs(small - (-25.0 / 2.0)) < 1e-9
    assert abs(large - (-25.0 / 41.0)) < 1e-9
    assert large > small

    worse = compute_reward(0.75, 1.0, 0.9, params_before=1000, params_after=990)  # o=15
    assert worse < small

    identity = compute_reward(0.99, 1.0, 1.0, params_before=1000, params_after=1000)
    in_budget = compute_reward(0.95, 1.0, 0.9, params_before=1000, params_after=900)
    assert worse < small < 0 <= identity < in_budget
    assert large < 0 <= identity


def test_structural_unified_mixes_weights_and_flops(monkeypatch):
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_unified")
    monkeypatch.setenv("SPECTRA_UNIFIED_EPS", "1")
    _init_static_conf(10)
    from src.utils import compute_reward, unified_rho

    rho = unified_rho(10.0, 1000, 900, 200, 100)
    assert abs(rho - 30.0) < 1e-9  # ½·10 + ½·50
    # Δ = -5, u = 5 → R = 30 * 0.5 = 15
    r = compute_reward(
        0.95, 1.0, 0.9,
        params_before=1000, params_after=900,
        flops_before=200, flops_after=100)
    assert abs(r - 15.0) < 1e-9


def test_structural_prefer_credits_full_rho_inside_tau(monkeypatch):
    """F1 slack taper prefers timid cuts; prefer pays ρ whenever the cut is legal."""
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_prefer")
    monkeypatch.setenv("SPECTRA_REWARD_SCALE", "cbrt")  # must be ignored
    monkeypatch.setenv("SPECTRA_UNIFIED_EPS", "1")
    _init_static_conf(10)
    from src.utils import compute_reward

    identity = compute_reward(0.99, 1.0, 1.0, params_before=1000, params_after=1000)
    assert abs(identity) < 1e-9

    # Δ = -5, ρ = 10 → R = 10 (F1 would have paid 10 * 5/10 = 5)
    in_budget = compute_reward(0.95, 1.0, 0.9, params_before=1000, params_after=900)
    assert abs(in_budget - 10.0) < 1e-9

    # Near the wall a large legal cut still gets full ρ (F1 tapers to ~0)
    near_wall = compute_reward(0.91, 1.0, 0.8, params_before=1000, params_after=800)
    assert abs(near_wall - 20.0) < 1e-9
    timid = compute_reward(0.99, 1.0, 0.9, params_before=1000, params_after=950)
    assert near_wall > timid

    # Over-budget arm matches F1 (harm per byte)
    small = compute_reward(0.85, 1.0, 0.9, params_before=1000, params_after=990)
    large = compute_reward(0.85, 1.0, 0.9, params_before=1000, params_after=600)
    assert large > small
    assert small < 0 <= identity < in_budget


def test_train_floor_flag_and_truncated_inbudget(monkeypatch):
    from src.fortify import train_respects_size_floor, inbudget_checkpoint_score, INBUDGET_OVER_PENALTY

    monkeypatch.delenv("SPECTRA_TRAIN_RESPECT_FLOOR", raising=False)
    assert train_respects_size_floor() is False
    monkeypatch.setenv("SPECTRA_TRAIN_RESPECT_FLOOR", "1")
    assert train_respects_size_floor() is True
    assert -INBUDGET_OVER_PENALTY < inbudget_checkpoint_score(12.0, 0.0, False)


def test_state_align_and_skip_eval_flags(monkeypatch):
    from src.fortify import state_align_next, skip_eval, reward_needs_flops

    monkeypatch.delenv("SPECTRA_STATE_ALIGN", raising=False)
    assert state_align_next() is False
    monkeypatch.setenv("SPECTRA_STATE_ALIGN", "next")
    assert state_align_next() is True

    monkeypatch.delenv("SPECTRA_SKIP_EVAL", raising=False)
    assert skip_eval() is False
    monkeypatch.setenv("SPECTRA_SKIP_EVAL", "1")
    assert skip_eval() is True

    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_prefer")
    assert reward_needs_flops() is True
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "neon")
    assert reward_needs_flops() is False


def test_actor_skip_overbudget_flag(monkeypatch):
    from src.fortify import actor_skip_overbudget, policy_gradient_advantages

    monkeypatch.delenv("SPECTRA_ACTOR_SKIP_OVERBUDGET", raising=False)
    assert actor_skip_overbudget() is False
    monkeypatch.setenv("SPECTRA_ACTOR_SKIP_OVERBUDGET", "1")
    assert actor_skip_overbudget() is True

    adv = torch.tensor([[2.0], [4.0], [-10.0]])
    out, kind = policy_gradient_advantages(adv, [False, False, True], True)
    assert kind == "masked"
    assert float(out[2]) == 0.0
    assert abs(float(out[0]) + float(out[1])) < 1e-5  # kept-only standardize, mean 0
    _, skip_kind = policy_gradient_advantages(adv, [True, True, True], True)
    assert skip_kind == "skip"
    _, full_kind = policy_gradient_advantages(adv, [False, True, False], False)
    assert full_kind == "full"


def test_truncated_returns_bootstrap(monkeypatch):
    import torch
    from src.utils import compute_returns

    rewards = [torch.tensor([[1.0]]), torch.tensor([[1.0]])]
    masks = [torch.tensor([[1.0]]), torch.tensor([[1.0]])]  # truncated, not done
    boot = torch.tensor([[10.0]])
    ret = compute_returns(boot, rewards, masks, 0.5)
    # R1 = 1 + 0.5*10 = 6; R0 = 1 + 0.5*6 = 4
    assert abs(float(ret[1]) - 6.0) < 1e-6
    assert abs(float(ret[0]) - 4.0) < 1e-6
    dead = compute_returns(boot, rewards, [torch.tensor([[1.0]]), torch.tensor([[0.0]])], 0.5)
    assert abs(float(dead[1]) - 1.0) < 1e-6


def test_inbudget_checkpoint_prefers_compression_over_return(monkeypatch):
    from src.fortify import inbudget_checkpoint_score, inbudget_checkpointing

    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_unified")
    monkeypatch.delenv("SPECTRA_CHECKPOINT", raising=False)
    assert inbudget_checkpointing() is True
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural_prefer")
    assert inbudget_checkpointing() is True
    monkeypatch.setenv("SPECTRA_CHECKPOINT", "return")
    assert inbudget_checkpointing() is False
    monkeypatch.delenv("SPECTRA_CHECKPOINT", raising=False)
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "structural")
    monkeypatch.setenv("SPECTRA_ACTOR_SKIP_OVERBUDGET", "1")
    assert inbudget_checkpointing() is True

    identity = inbudget_checkpoint_score(0.0, 0.0, False)
    mild = inbudget_checkpoint_score(12.0, 0.0, False)
    over = inbudget_checkpoint_score(40.0, 8.0, True)
    assert over < identity < mild
    assert inbudget_checkpoint_score(1.0, 0.0, False) > over


def test_masked_noop_does_not_get_neon_compression_credit():
    from src.NetworkEnv import reward_compression_rate

    tau = 10
    # In-budget mask, numel unchanged → identity rate (zero NEON compression credit)
    assert reward_compression_rate({"mode": "masked"}, 0.8, 1000, 1000, 0.95, 1.0, tau) == 1.0
    # Over-budget still uses the nominal rate so wrecking an unprunable layer is punished
    assert reward_compression_rate({"mode": "masked"}, 0.8, 1000, 1000, 0.80, 1.0, tau) == 0.8
    # Structural shrink keeps the action's rate
    assert reward_compression_rate({"mode": "structural"}, 0.8, 1000, 800, 0.95, 1.0, tau) == 0.8
