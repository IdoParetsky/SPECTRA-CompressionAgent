"""Unit tests for SPECTRA fortify helpers (no GPU / no pretrained nets)."""

import os
import sys
from pathlib import Path

import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
from src.BERTInputModeler import token_feature_dim, TOKEN_BASE_DIM  # noqa: E402


def test_fortify_on_by_default(monkeypatch):
    monkeypatch.delenv("SPECTRA_FORTIFY", raising=False)
    assert fortify.fortify_enabled() is True
    assert fortify.fortify_token_dim() == fortify.FORTIFY_TOKEN_DIM


def test_fortify_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "0")
    assert fortify.fortify_enabled() is False
    assert fortify.fortify_token_dim() == 0
    assert token_feature_dim(5) == TOKEN_BASE_DIM + 10


def test_fortify_token_dim_when_on(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    assert fortify.fortify_token_dim() == fortify.FORTIFY_TOKEN_DIM
    assert token_feature_dim(5) == TOKEN_BASE_DIM + fortify.FORTIFY_TOKEN_DIM + 10


def test_noop_mask_always_on(monkeypatch):
    monkeypatch.delenv("SPECTRA_FORTIFY", raising=False)
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    # alive=1 → identity only even without fortify
    mask = fortify.legal_action_mask(rates, row_index=5, alive_count=1, device="cpu")
    assert mask.tolist() == [True, False, False]


def test_stem_forces_identity(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_STEM_ROWS", "1")
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    mask = fortify.legal_action_mask(rates, row_index=0, alive_count=16, device="cpu")
    assert mask.tolist() == [True, False, False]


def test_narrow_width_forces_identity(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_MIN_WIDTH_FOR_PRUNE", "2")
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    mask = fortify.legal_action_mask(rates, row_index=5, alive_count=2, device="cpu")
    assert mask.tolist() == [True, False, False]


def test_entropy_anneal_decays_after_warmup(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_ENTROPY_ANNEAL_HORIZON", "10")
    monkeypatch.setenv("SPECTRA_ENTROPY_MIN", "0.01")
    base = 0.05
    assert fortify.entropy_coef(0, warmup_len=5, base=base) == base
    mid = fortify.entropy_coef(10, warmup_len=5, base=base)  # 5/10 through anneal
    assert 0.01 < mid < base
    end = fortify.entropy_coef(100, warmup_len=5, base=base)
    assert abs(end - 0.01) < 1e-9


def test_apply_action_mask_renorms(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    probs = torch.tensor([[0.2, 0.5, 0.3]])
    dist = Categorical(probs=probs)
    legal = torch.tensor([True, False, True])
    masked = fortify.apply_action_mask(dist, legal)
    assert torch.allclose(masked.probs[0, 1], torch.tensor(0.0))
    assert torch.allclose(masked.probs.sum(), torch.tensor(1.0))


def test_budget_in_state_adds_one_channel(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    monkeypatch.setenv("SPECTRA_BUDGET_IN_STATE", "1")
    assert fortify.budget_in_state() is True
    assert fortify.fortify_token_dim() == fortify.FORTIFY_TOKEN_DIM + 1
    assert token_feature_dim(5) == TOKEN_BASE_DIM + fortify.FORTIFY_TOKEN_DIM + 1 + 10


def test_heuristic_eval_l1_picks_strongest_prune(monkeypatch):
    monkeypatch.setenv("SPECTRA_EVAL_POLICY", "l1")
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, True, True])
    action = fortify.heuristic_eval_action(legal, rates, policy="l1", device="cpu")
    assert int(action.item()) == 2


def test_heuristic_eval_mild_prefers_0_9(monkeypatch):
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, True, True])
    action = fortify.heuristic_eval_action(legal, rates, policy="mild", device="cpu")
    assert int(action.item()) == 1


def test_heuristic_eval_identity_when_no_prune_legal():
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, False, False])
    action = fortify.heuristic_eval_action(legal, rates, policy="l1", device="cpu")
    assert int(action.item()) == 0


def test_eval_lookahead_off_by_default(monkeypatch):
    monkeypatch.delenv("SPECTRA_EVAL_LOOKAHEAD", raising=False)
    monkeypatch.delenv("SPECTRA_EVAL_MIN_FLOP_RATIO", raising=False)
    assert fortify.eval_min_flop_ratio() == 0.0
    assert fortify.eval_lookahead_enabled() is False
    monkeypatch.setenv("SPECTRA_EVAL_LOOKAHEAD", "1")
    assert fortify.eval_lookahead_enabled() is True


def test_flop_floor_auto_enables_lookahead(monkeypatch):
    monkeypatch.delenv("SPECTRA_EVAL_LOOKAHEAD", raising=False)
    monkeypatch.setenv("SPECTRA_EVAL_MIN_FLOP_RATIO", "0.70")
    assert fortify.eval_min_flop_ratio() == 0.70
    assert fortify.eval_lookahead_enabled() is True


class _FakePreviewEnv:
    def __init__(self, previews):
        self.previews = previews

    def preview_param_ratio(self, rate):
        return self.previews[float(rate)]


def test_lookahead_keeps_action_when_preview_stays_on_floor(monkeypatch):
    monkeypatch.delenv("SPECTRA_EVAL_MIN_FLOP_RATIO", raising=False)
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, True, True])
    env = _FakePreviewEnv({1.0: 0.80, 0.9: 0.75, 0.8: 0.71})
    action = torch.tensor([2])
    out = fortify.action_respecting_param_floor(env, action, legal, rates, 0.70, "cpu")
    assert int(out.item()) == 2


def test_lookahead_falls_back_to_milder_cut_then_identity(monkeypatch):
    monkeypatch.delenv("SPECTRA_EVAL_MIN_FLOP_RATIO", raising=False)
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, True, True])
    env = _FakePreviewEnv({1.0: 0.83, 0.9: 0.72, 0.8: 0.66})
    action = torch.tensor([2])
    out = fortify.action_respecting_param_floor(env, action, legal, rates, 0.70, "cpu")
    assert int(out.item()) == 1
    env_both_low = _FakePreviewEnv({1.0: 0.83, 0.9: 0.65, 0.8: 0.60})
    out_id = fortify.action_respecting_param_floor(
        env_both_low, action, legal, rates, 0.70, "cpu")
    assert int(out_id.item()) == 0


class _FakeRatioEnv:
    def __init__(self, param, flop, param_previews, flop_previews):
        self._param = param
        self._flop = flop
        self.param_previews = param_previews
        self.flop_previews = flop_previews

    def param_ratio(self):
        return self._param

    def flops_ratio(self):
        return self._flop

    def preview_param_ratio(self, rate):
        return self.param_previews[float(rate)]

    def preview_flops_ratio(self, rate):
        return self.flop_previews[float(rate)]

    def preview_ratios(self, rate):
        return self.preview_param_ratio(rate), self.preview_flops_ratio(rate)


def test_eval_at_size_floor_param_then_flop(monkeypatch):
    monkeypatch.delenv("SPECTRA_EVAL_MIN_FLOP_RATIO", raising=False)
    env = _FakeRatioEnv(0.69, 0.80, {}, {})
    stop, reason = fortify.eval_at_size_floor(env)
    assert stop is True and reason == "param"
    monkeypatch.setenv("SPECTRA_EVAL_MIN_FLOP_RATIO", "0.70")
    env_flop = _FakeRatioEnv(0.80, 0.55, {}, {})
    stop, reason = fortify.eval_at_size_floor(env_flop)
    assert stop is True and reason == "flop"
    env_ok = _FakeRatioEnv(0.80, 0.75, {}, {})
    stop, reason = fortify.eval_at_size_floor(env_ok)
    assert stop is False and reason == ""


def test_eval_at_size_floor_skips_flops_when_floor_off(monkeypatch):
    monkeypatch.delenv("SPECTRA_EVAL_MIN_FLOP_RATIO", raising=False)

    class _ParamOnly:
        def param_ratio(self):
            return 0.80

    stop, reason = fortify.eval_at_size_floor(_ParamOnly())
    assert stop is False and reason == ""


def test_flop_lookahead_falls_back_when_params_ok_flops_breach(monkeypatch):
    monkeypatch.setenv("SPECTRA_EVAL_MIN_FLOP_RATIO", "0.70")
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, True, True])
    action = torch.tensor([2])
    env = _FakeRatioEnv(
        0.80, 0.72,
        {1.0: 0.80, 0.9: 0.80, 0.8: 0.80},
        {1.0: 0.72, 0.9: 0.71, 0.8: 0.55},
    )
    out = fortify.action_respecting_param_floor(env, action, legal, rates, 0.70, "cpu")
    assert int(out.item()) == 1
    env_both = _FakeRatioEnv(
        0.80, 0.72,
        {1.0: 0.80, 0.9: 0.80, 0.8: 0.80},
        {1.0: 0.72, 0.9: 0.60, 0.8: 0.55},
    )
    out_id = fortify.action_respecting_param_floor(
        env_both, action, legal, rates, 0.70, "cpu")
    assert int(out_id.item()) == 0


def test_flop_lookahead_keeps_0_8_when_both_floors_ok(monkeypatch):
    monkeypatch.setenv("SPECTRA_EVAL_MIN_FLOP_RATIO", "0.70")
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, True, True])
    env = _FakeRatioEnv(
        0.85, 0.80,
        {1.0: 0.85, 0.9: 0.82, 0.8: 0.80},
        {1.0: 0.80, 0.9: 0.76, 0.8: 0.72},
    )
    out = fortify.action_respecting_param_floor(
        env, torch.tensor([2]), legal, rates, 0.70, "cpu")
    assert int(out.item()) == 2


def test_prefer_param_per_flop_picks_efficient_cut(monkeypatch):
    monkeypatch.setenv("SPECTRA_EVAL_MIN_FLOP_RATIO", "0.70")
    monkeypatch.setenv("SPECTRA_EVAL_PREFER_PARAM_PER_FLOP", "1")
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, True, True])
    env = _FakeRatioEnv(
        0.90, 0.85,
        {1.0: 0.90, 0.9: 0.86, 0.8: 0.88},
        {1.0: 0.85, 0.9: 0.82, 0.8: 0.77},
    )
    out = fortify.action_preferring_param_per_flop(
        env, torch.tensor([2]), legal, rates, 0.70, "cpu")
    assert int(out.item()) == 1


def test_prefer_param_per_flop_skips_when_all_cuts_flop_heavy(monkeypatch):
    monkeypatch.setenv("SPECTRA_EVAL_MIN_FLOP_RATIO", "0.70")
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, True, True])
    env = _FakeRatioEnv(
        0.92, 0.80,
        {1.0: 0.92, 0.9: 0.91, 0.8: 0.90},
        {1.0: 0.80, 0.9: 0.74, 0.8: 0.71},
    )
    out = fortify.action_preferring_param_per_flop(
        env, torch.tensor([2]), legal, rates, 0.70, "cpu")
    assert int(out.item()) == 0


def test_prefer_param_per_flop_noop_without_flop_floor(monkeypatch):
    monkeypatch.delenv("SPECTRA_EVAL_MIN_FLOP_RATIO", raising=False)
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, True, True])
    env = _FakeRatioEnv(0.90, 0.85, {0.8: 0.80}, {0.8: 0.80})
    out = fortify.action_preferring_param_per_flop(
        env, torch.tensor([2]), legal, rates, 0.70, "cpu")
    assert int(out.item()) == 2


def test_heuristic_eval_random_stays_in_prune_set():
    rates = {0: 1.0, 1: 0.9, 2: 0.8}
    legal = torch.tensor([True, True, True])
    torch.manual_seed(0)
    seen = {int(fortify.heuristic_eval_action(legal, rates, policy="random", device="cpu").item())
            for _ in range(20)}
    assert seen <= {1, 2}
    assert 0 not in seen


def test_fortify_features_shape():
    coupling = torch.tensor([0, 0, 1, 2, 2])
    topo = [[2, 3, 16, 3, 1, 1, 1]] * 5
    feats = fortify.build_fortify_features(5, coupling, topo, device="cpu")
    assert feats.shape == (5, fortify.FORTIFY_TOKEN_DIM)
    assert feats[0, 1] == 1.0  # stem
    assert feats[0, 2] == 1.0 and feats[1, 2] == 1.0  # coupled pair
    assert feats[2, 2] == 0.0  # singleton coupling
