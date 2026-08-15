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
