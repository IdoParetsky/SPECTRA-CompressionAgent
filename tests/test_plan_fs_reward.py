"""
``fs<S>`` reward proxy (default off): S SGD steps on S fixed train batches, then val accuracy. The plan trainer runs
them on its masked cut, re-masking after every step (``plan_agent.short_finetune`` with ``MaskedCut.remask``); the
eval side (``plan_proxies.measure_one``) runs the same steps on a real cut. Nothing changes unless
``SPECTRA_PLAN_PROXY=fs<S>`` or ``SPECTRA_EVAL_PROXIES`` names ``fs<S>``.

CPU only, no datasets.  python -m pytest tests/test_plan_fs_reward.py -v
"""

import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import SHAPE, _instance, _plan, _randomize_norms  # noqa: E402
from src import alloc_walk, plan_agent, plan_proxies, pruning  # noqa: E402
import src.run_recorder as recorder  # noqa: E402

TINY_SHAPE = (3, 8, 8)
LR = 0.05


class TinyChain(nn.Module):
    """conv-bn-relu x2, pool, then a flattened linear head (the flatten expansion is part of the cut)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU(),
            nn.Conv2d(6, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8 * 4 * 4, 5))

    def forward(self, x):
        return self.classifier(self.features(x))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SPECTRA_PLAN_") or key == "SPECTRA_EVAL_PROXIES":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


def _tiny():
    torch.manual_seed(0)
    model = TinyChain().eval()
    _randomize_norms(model)
    return model


def _loader(n=40, batch=8, seed=0, onehot=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, *TINY_SHAPE, generator=g)
    y = torch.randint(0, 5, (n,), generator=g)
    y = torch.nn.functional.one_hot(y, 5).float() if onehot else y
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch, shuffle=True)


def _fixed_batches(n=3, seed=2):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(8, *TINY_SHAPE, generator=g), torch.randint(0, 5, (8,), generator=g)) for _ in range(n)]


# ------------------------------------------------------------------ the names

def test_parse_proxy_accepts_fs_and_keeps_cut_and_bn():
    assert plan_agent.parse_proxy("fs16") == ("fs", 16)
    assert plan_agent.parse_proxy(" FS1 ") == ("fs", 1)
    assert plan_agent.parse_proxy("cut") == ("cut", 0)
    assert plan_agent.parse_proxy("bn8") == ("bn", 8)
    for bad in ("fs0", "fs", "fsx", "fs-3", "fs1.5", "ft0", "bn0", ""):
        with pytest.raises(ValueError, match="fs<S>"):
            plan_agent.parse_proxy(bad)


def test_eval_proxies_accept_fs_reject_fs0_and_keep_the_other_groups(monkeypatch):
    monkeypatch.setenv("SPECTRA_EVAL_PROXIES", "cut,bn8,fs16,ft1")
    assert plan_proxies.proxies() == ["cut", "bn8", "fs16", "ft1"]
    for bad in ("fs0", "fs", "fsx"):
        monkeypatch.setenv("SPECTRA_EVAL_PROXIES", f"cut,{bad}")
        with pytest.raises(ValueError, match="fs<S>"):
            plan_proxies.proxies()
    m = plan_proxies._NAME.match("bn8")
    assert (m.group(2), m.group(3), m.group(4), m.group(5)) == ("8", None, None, None)
    m = plan_proxies._NAME.match("ft3")
    assert (m.group(2), m.group(3), m.group(4), m.group(5)) == (None, "3", None, None)
    m = plan_proxies._NAME.match("ft12_4")
    assert (m.group(2), m.group(3), m.group(4), m.group(5)) == (None, "12", "4", None)
    m = plan_proxies._NAME.match("fs16")
    assert (m.group(2), m.group(3), m.group(4), m.group(5)) == (None, None, None, "16")


def test_fs_recipe_defaults_and_lr_knob(monkeypatch):
    assert (plan_agent.FS_MOMENTUM, plan_agent.FS_WD) == (0.9, 5e-4)
    assert plan_agent.fs_lr() == 0.01
    monkeypatch.setenv("SPECTRA_PLAN_FS_LR", "0.003")
    assert plan_agent.fs_lr() == 0.003


# ------------------------------------------------------------------ the batches

def test_ft_batches_are_n_index_labelled_pairs_deterministic_in_the_seed():
    loader = _loader()
    state = torch.get_rng_state()
    a = plan_agent.ft_batches(loader, 3, 7, "cpu")
    assert torch.equal(torch.get_rng_state(), state)
    b = plan_agent.ft_batches(loader, 3, 7, "cpu")
    assert len(a) == 3 and all(isinstance(t, tuple) and len(t) == 2 for t in a)
    for (xa, ya), (xb, yb) in zip(a, b):
        assert ya.dtype == torch.long and ya.dim() == 1 and int(ya.max()) < 5
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    c = plan_agent.ft_batches(loader, 3, 8, "cpu")
    assert not all(torch.equal(xa, xc) for (xa, _), (xc, _) in zip(a, c))
    assert len(plan_agent.ft_batches(loader, 99, 7, "cpu")) == len(loader)


def test_ft_batches_turn_one_hot_labels_into_the_same_class_indices():
    idx = plan_agent.ft_batches(_loader(), 2, 5, "cpu")
    hot = plan_agent.ft_batches(_loader(onehot=True), 2, 5, "cpu")
    for (xi, yi), (xh, yh) in zip(idx, hot):
        assert yh.dtype == torch.long and torch.equal(xi, xh) and torch.equal(yi, yh)


# ------------------------------------------------------------------ the short fine-tune and the masks

def test_short_finetune_trains_every_parameter_restores_the_flags_and_ends_in_eval():
    model = _tiny()
    frozen = model.features[0].weight
    frozen.requires_grad = False
    before = frozen.detach().clone()
    seen = []
    with torch.no_grad():  # the caller's grad mode does not matter
        plan_agent.short_finetune(model, _fixed_batches(2), LR, after_step=lambda: seen.append(torch.is_grad_enabled()))
    assert seen == [False, False]
    assert frozen.requires_grad is False and all(p.requires_grad for p in model.parameters() if p is not frozen)
    assert not torch.equal(frozen.detach(), before)
    assert not model.training


def test_remask_zeroes_every_recorded_entry_after_a_step_and_a_drift():
    model = _tiny()
    cutter = plan_agent.MaskedCut(model)
    rates = {row: 0.5 for row in cutter.rows}
    work = cutter.cut(rates)
    assert cutter._masks and any(idx.numel() > 0 for _p, idx, _dim in cutter._masks)
    conv = work.features[0].weight
    alive = conv.detach().flatten(1).abs().sum(1) > 0
    before = conv.detach().clone()
    plan_agent.short_finetune(work, _fixed_batches(1), LR)
    assert not torch.equal(conv.detach()[alive], before[alive])
    with torch.no_grad():  # drift: whatever a step may have put back into the cut entries
        for param, idx, dim in cutter._masks:
            if dim == 0:
                param[idx] = 1.0
            else:
                param[:, idx] = 1.0
    assert any(torch.count_nonzero(p.index_select(dim, idx)) > 0 for p, idx, dim in cutter._masks)
    cutter.remask()
    for param, idx, dim in cutter._masks:
        assert param.requires_grad and idx.device == param.device
        assert torch.count_nonzero(param.index_select(dim, idx)) == 0
    cutter.cut({})
    assert cutter._masks == []


def test_masked_cut_with_a_record_is_the_cut_without_one():
    model = _tiny()
    cutter = plan_agent.MaskedCut(model)
    rates = {row: 0.5 for row in cutter.rows}
    recorded = {k: v.clone() for k, v in cutter.cut(rates).state_dict().items()}
    cutter.work.load_state_dict(cutter.origin)
    for group, row in cutter.plan:
        plan_agent._zero_group(group, pruning.select_group_survivors(group, rates[row], mode="l1"))
    for key, value in cutter.work.state_dict().items():
        assert torch.equal(value, recorded[key]), key


def test_the_masked_fine_tune_is_the_real_cuts_fine_tune():
    model = _tiny()
    plan = _plan(model)
    cutter = plan_agent.MaskedCut(model)
    assert cutter.rows == [row for _g, row in plan] and len(plan) >= 1
    rates = {row: 0.5 for row in cutter.rows}
    batches = _fixed_batches(3)
    x_eval = torch.randn(6, *TINY_SHAPE, generator=torch.Generator().manual_seed(9))
    real = alloc_walk.cut_to(model, plan, rates, TINY_SHAPE)
    masked = cutter.cut(rates)
    real.eval(), masked.eval()
    with torch.no_grad():
        real0, masked0 = real(x_eval), masked(x_eval)
    torch.testing.assert_close(masked0, real0, atol=1e-4, rtol=1e-4)
    plan_agent.short_finetune(real, batches, LR)
    plan_agent.short_finetune(masked, batches, LR, after_step=cutter.remask)
    with torch.no_grad():
        real1, masked1 = real(x_eval), masked(x_eval)
    assert not torch.allclose(real1, real0, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(masked1, masked0, atol=1e-4, rtol=1e-4)
    assert torch.allclose(real1, masked1, atol=1e-4, rtol=1e-4), float((real1 - masked1).abs().max())


# ------------------------------------------------------------------ the eval-side proxy

def test_measure_one_fs_fine_tunes_a_copy_of_the_real_cut_on_the_seeded_batches(monkeypatch):
    model = _tiny()
    plan = _plan(model)
    cut = alloc_walk.cut_to(model, plan, {row: 0.5 for row in plan_agent.MaskedCut(model).rows}, TINY_SHAPE)
    before = {k: v.clone() for k, v in cut.state_dict().items()}
    loader = _loader()
    seen = []

    class Handler:
        def __init__(self, net):
            seen.append(net)

        def evaluate_model(self, _loader):
            return 0.5

    env = types.SimpleNamespace(train_loader=loader, conf=types.SimpleNamespace(device="cpu"),
                                val_loader=None, test_loader=None, create_learning_handler=Handler)
    monkeypatch.setenv("SPECTRA_EVAL_PROXY_SEED", "4")
    out = plan_proxies.measure_one(env, cut, "fs3")
    assert out["val"] == 0.5 and out["test"] == 0.5 and len(seen) == 1
    for key, value in cut.state_dict().items():
        assert torch.equal(value, before[key]), key  # the candidate itself is untouched
    expect = alloc_walk.cut_to(model, plan, {row: 0.5 for row in plan_agent.MaskedCut(model).rows}, TINY_SHAPE)
    plan_agent.short_finetune(expect, plan_agent.ft_batches(loader, 3, 4, "cpu"), plan_agent.fs_lr())
    for key, value in seen[0].state_dict().items():
        torch.testing.assert_close(value, expect.state_dict()[key], atol=1e-6, rtol=1e-6)
    assert not torch.equal(seen[0].features[0].weight, before["features.0.weight"])


# ------------------------------------------------------------------ the trainer's reward

def test_instance_fs_reward_is_the_real_cut_fine_tuned_and_never_leaks_into_the_next_plan():
    inst = _instance()
    batches = inst.batches("fs3", seed=3)
    again = inst.batches("fs3", seed=3)
    assert len(batches) == 3 and all(y.dtype == torch.long for _x, y in batches)
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(batches, again))
    assert inst.batches("cut", seed=3) == [] and len(inst.batches("bn4", seed=3)) == 4
    widths, _info = plan_agent.decode([0.1 * k for k in range(inst.n_groups)], inst.pm, 0.6)
    rates = plan_agent.rates_of(widths, inst.pm.widths0)
    r_cut = inst.reward(widths, "cut", [])
    r_fs = inst.reward(widths, "fs3", batches)
    real = alloc_walk.cut_to(inst.model, inst.plan, rates, SHAPE)
    plan_agent.short_finetune(real, batches, plan_agent.fs_lr())
    assert r_fs == pytest.approx(100.0 * (plan_agent.accuracy(real, inst.val) - inst.origin_val))
    assert inst.reward(widths, "cut", []) == pytest.approx(r_cut)
    assert inst.reward(widths, "fs3", batches) == pytest.approx(r_fs)
    with pytest.raises(ValueError):
        inst.reward(widths, "fs0", [])
