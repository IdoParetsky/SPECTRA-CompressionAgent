"""
v23 (default off): the ``ft<E>`` training reward (``SPECTRA_PLAN_PROXY=ft<E>``) and the warm start
(``SPECTRA_PLAN_INIT``). ``plan_agent.NetInstance.finetune_reward`` cuts the net for real and fine-tunes that copy E
epochs with the learning handler's own recipe under the instance's seed, as the eval-side proxy
(``plan_proxies.measure_one``) does; ``plan_trainer.warm_start`` loads a checkpoint's policy weights into a run that
trains under the same settings. Nothing changes unless one of the two is set.

CPU only, no datasets.  python -m pytest tests/test_v23_plan_ft_reward.py -v
"""

import json
import math
import os
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import (RATES, SHAPE, V10_FLAGS, ZOO, _instance, _loader, _randomize_norms,  # noqa: E402
                                   toy_state)
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, plan_agent, plan_proxies, plan_trainer, state_dump  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FT_", "SPECTRA_EVAL_PROXY", "SPECTRA_FINETUNE_PATIENCE")
STEP_LINE = re.compile(r"^\[plan\] it=\d+/\d+ toy\.pt k=\d\.\d{3} sigma=\d\.\d{3} \| plans mean [+-]\d+\.\d{2} "
                       r"max [+-]\d+\.\d{2} min [+-]\d+\.\d{2} kept x\d\.\d{3} \| mu std \d+\.\d{3}"
                       r"( \| mean plan [+-]\d+\.\d{2}( \w+ [+-]\d+\.\d{2})+)? \| \d+\.\d+s$")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


@pytest.fixture
def v10_env(monkeypatch):
    for key, value in V10_FLAGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SPECTRA_STATE_TOKENS", raising=False)


def _plan_widths(inst, kappa=0.6, slope=0.1):
    widths, _info = plan_agent.decode([slope * k for k in range(inst.n_groups)], inst.pm, kappa)
    return widths


def _rng_state():
    """The global torch / random / numpy state as plain Python values (comparable with ``==``)."""
    npy = np.random.get_state()
    return torch.get_rng_state().tolist(), random.getstate(), (npy[0], npy[1].tolist(), tuple(npy[2:]))


def _handler(model):
    return ClassificationHandler(model, nn.CrossEntropyLoss())


# ------------------------------------------------------------------ the name and the batches

def test_parse_proxy_accepts_ft_and_keeps_the_other_names():
    assert plan_agent.parse_proxy("ft1") == ("ft", 1)
    assert plan_agent.parse_proxy(" FT3 ") == ("ft", 3)
    assert plan_agent.parse_proxy("cut") == ("cut", 0)
    assert plan_agent.parse_proxy("bn8") == ("bn", 8)
    assert plan_agent.parse_proxy("fs16") == ("fs", 16)
    assert plan_agent.PROXIES == ("cut", "bn", "fs", "ft")
    for bad in ("ft0", "ft", "ftx", "ft-1", "ft1.5", "ft12_4", "fs0", "bn0", "", "epoch1"):
        with pytest.raises(ValueError, match="ft<E>"):
            plan_agent.parse_proxy(bad)


def test_trainer_config_accepts_ft_and_keeps_its_default(monkeypatch):
    assert plan_trainer.config().proxy == "bn32"
    monkeypatch.setenv("SPECTRA_PLAN_PROXY", "ft1")
    assert plan_trainer.config().proxy == "ft1"
    monkeypatch.setenv("SPECTRA_PLAN_PROXY", "ft0")
    with pytest.raises(ValueError):
        plan_trainer.config()


def test_batches_for_ft_is_the_seed_and_the_reward_refuses_anything_else():
    inst = _instance()
    assert inst.batches("ft1", 7) == 7 and inst.batches("ft2", seed=3) == 3
    assert inst.batches("cut", 7) == [] and len(inst.batches("bn2", 7)) == 2 and len(inst.batches("fs2", 7)) == 2
    widths = _plan_widths(inst)
    for bad in ([], None, 1.5, True, "7"):
        with pytest.raises(ValueError, match="seed"):
            inst.reward(widths, "ft1", bad)
    assert inst.create_handler is plan_agent.default_handler and inst.recipe_line is None
    handler = plan_agent.default_handler(ZOO["thin_r20_w4"]())
    assert isinstance(handler, ClassificationHandler) and isinstance(handler.loss_func, nn.CrossEntropyLoss)


def test_from_env_hands_the_instance_the_envs_own_handler(v10_env):
    torch.manual_seed(0)
    model = ZOO["thin_r20_w4"]().eval()
    _randomize_norms(model)
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    plan = group_sensitivity.group_plan(mwr, groups)
    rows = state_dump.token_rows(mwr, groups, plan)
    made = []

    def create(net):
        made.append(net)
        return _handler(net)

    env = types.SimpleNamespace(conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES),
                                current_model=model, train_loader=_loader(), val_loader=_loader(seed=1, shuffle=False),
                                reset=lambda **kw: toy_state(len(rows)), _input_shape=lambda: SHAPE,
                                _dependency_groups=lambda m: channel_groups.build_channel_groups(m.model),
                                create_learning_handler=create)
    inst = plan_agent.NetInstance.from_env(env, "/zoo/toy.pt", None, None, with_sens=False)
    assert inst.create_handler is create and inst.name == "toy.pt"
    r = inst.reward(_plan_widths(inst), "ft1", 2)
    assert math.isfinite(r) and len(made) == 1 and made[0] is not inst.model


# ------------------------------------------------------------------ the ft<E> reward

def test_ft1_reward_is_the_real_cut_fine_tuned_by_the_handlers_own_recipe(monkeypatch):
    inst = _instance()
    seen = []

    def create(net):
        seen.append(net)
        return _handler(net)

    inst.create_handler = create
    origin = {k: v.clone() for k, v in inst.model.state_dict().items()}
    widths = _plan_widths(inst)
    rates = plan_agent.rates_of(widths, inst.pm.widths0)
    bare = alloc_walk.cut_to(inst.model, inst.plan, rates, SHAPE)
    monkeypatch.setenv("SPECTRA_FT_OPTIM", "sgd")  # a recipe override: the reward must hide it from the handler ...
    monkeypatch.setenv("SPECTRA_FT_LR", "0.5")
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    before = _rng_state()
    r = inst.reward(widths, "ft1", 5)
    assert _rng_state() == before  # ... and put the global RNG back
    assert (os.environ["SPECTRA_FT_OPTIM"], os.environ["SPECTRA_FT_LR"]) == ("sgd", "0.5")
    assert isinstance(r, float) and math.isfinite(r)
    assert inst.recipe_line.startswith("optim=adam lr=0.001 ") and "patience=2 epochs=1" in inst.recipe_line
    assert inst.recipe_line.endswith(" select=train_loss")
    # a structural cut to the plan's widths, every parameter trainable, never the masked copy nor the origin
    assert len(seen) == 1 and seen[0] is not inst.cutter.work and seen[0] is not inst.model
    assert alloc_walk.group_widths(seen[0], inst.pm.rows) == widths
    assert utils.calc_num_parameters(seen[0]) == inst.pm.params(widths) < inst.pm.total0
    assert all(p.requires_grad for p in seen[0].parameters())
    for key, value in inst.model.state_dict().items():
        assert torch.equal(value, origin[key]), key
    moved = [k for k, v in seen[0].state_dict().items()
             if v.dtype.is_floating_point and not torch.equal(v, bare.state_dict()[k])]
    assert moved  # the epoch ran
    # the same value by hand: the eval-side recipe (plan_proxies.measure_one) on the real cut under the same seed
    expect = alloc_walk.cut_to(inst.model, inst.plan, rates, SHAPE)
    for param in expect.parameters():
        param.requires_grad = True
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    with plan_proxies.seeded(5), plan_proxies.handler_recipe():
        _handler(expect).train_model(inst.train_loader, allow_reinit_retry=False, max_epochs=1, patience=2)
    assert any("Fine-tune recipe: optim=adam" in line for line in lines)
    assert r == pytest.approx(100.0 * (plan_agent.accuracy(expect, inst.val) - inst.origin_val))
    for key, value in seen[0].state_dict().items():
        torch.testing.assert_close(value, expect.state_dict()[key], atol=1e-6, rtol=1e-6)


def test_ft1_reward_equals_the_eval_side_proxy_on_the_same_cut_and_seed(monkeypatch):
    inst = _instance()
    widths = _plan_widths(inst)
    cut = alloc_walk.cut_to(inst.model, inst.plan, plan_agent.rates_of(widths, inst.pm.widths0), SHAPE)
    env = types.SimpleNamespace(train_loader=inst.train_loader, val_loader=_loader(seed=1, shuffle=False),
                                test_loader=_loader(seed=1, shuffle=False), conf=types.SimpleNamespace(device="cpu"),
                                create_learning_handler=_handler)
    monkeypatch.setenv("SPECTRA_EVAL_PROXY_SEED", "5")
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    out = plan_proxies.measure_one(env, cut, "ft1")
    assert inst.reward(widths, "ft1", 5) == pytest.approx(100.0 * (out["val"] - inst.origin_val))


def test_ft1_reward_is_deterministic_and_every_plan_of_an_instance_sees_the_same_epoch():
    inst = _instance()

    class Spy:
        """The train loader, recording the labels of each epoch it yields in order."""

        def __init__(self, loader):
            self.loader, self.epochs = loader, []

        def __iter__(self):
            labels = []
            for x, y in self.loader:
                labels.append(y.clone())
                yield x, y
            self.epochs.append(torch.cat(labels))

        def __len__(self):
            return len(self.loader)

    spy = inst.train_loader = Spy(inst.train_loader)
    w1, w2 = _plan_widths(inst), _plan_widths(inst, slope=-0.1)
    assert w1 != w2
    a = inst.reward(w1, "ft1", 3)
    recipe = inst.recipe_line
    b = inst.reward(dict(w1), "ft1", 3)  # the same plan again: the same value
    c = inst.reward(w2, "ft1", 3)  # another plan of the same instance: the same shuffle (common random numbers)
    d = inst.reward(w1, "ft1", 4)  # another instance seed: another shuffle
    assert a == b and all(math.isfinite(v) for v in (a, c, d))
    assert inst.recipe_line == recipe and "trainable=" not in recipe  # the recipe, not the plan's parameter count
    assert len(spy.epochs) == 4
    assert torch.equal(spy.epochs[0], spy.epochs[1]) and torch.equal(spy.epochs[0], spy.epochs[2])
    assert not torch.equal(spy.epochs[0], spy.epochs[3])
    assert sorted(spy.epochs[0].tolist()) == sorted(spy.epochs[3].tolist())  # the same samples, another order
    assert math.isfinite(inst.reward(w1, "ft2", 3))  # E = 2: two epochs, patience 3
    assert "epochs=2" in inst.recipe_line and "patience=3" in inst.recipe_line and len(spy.epochs) == 6


def test_a_fine_tune_that_did_not_run_is_an_error_and_the_log_function_is_put_back():
    inst = _instance()
    widths = _plan_widths(inst)
    before = utils.print_flush
    inst.train_loader = []  # the handler aborts on an empty loader and keeps the bare cut: not a reward
    with pytest.raises(RuntimeError, match="did not run"):
        inst.reward(widths, "ft1", 1)
    assert utils.print_flush is before

    class Broken:
        def __init__(self, net):
            pass

        def train_model(self, *a, **k):
            raise RuntimeError("boom")

    inst.train_loader, inst.create_handler = _loader(), Broken
    with pytest.raises(RuntimeError, match="boom"):
        inst.reward(widths, "ft1", 1)
    assert utils.print_flush is before and inst.recipe_line is None

    class Silent(Broken):
        def train_model(self, *a, **k):
            return None  # no recipe line: it never trained

    inst.create_handler = Silent
    with pytest.raises(RuntimeError, match="did not run"):
        inst.reward(widths, "ft1", 1)


def test_a_fine_tune_whose_loss_went_nan_is_flagged_in_the_log(monkeypatch):
    inst = _instance()
    widths = _plan_widths(inst)

    class Diverged:
        def __init__(self, net):
            pass

        def train_model(self, *a, **k):
            utils.print_flush("Fine-tune recipe: optim=adam lr=0.001 cosine=0 schedule=plateau trainable=10")
            utils.print_flush("Epoch 1/1: Loss = nan, LR = 0.00100")

    inst.create_handler = Diverged
    shown = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: shown.append(str(s)))
    assert math.isfinite(inst.reward(widths, "ft1", 1))
    assert any("WARNING" in s and "NaN" in s and inst.name in s for s in shown)
    assert not any("Epoch 1/1" in s for s in shown)  # the handler's own lines stay captured
    assert inst.recipe_line == "optim=adam lr=0.001 cosine=0 schedule=plateau"


def test_the_other_proxies_are_untouched():
    inst = _instance()
    widths = _plan_widths(inst)
    rates = plan_agent.rates_of(widths, inst.pm.widths0)
    real = alloc_walk.cut_to(inst.model, inst.plan, rates, SHAPE).eval()
    r_cut = 100.0 * (plan_agent.accuracy(real, inst.val) - inst.origin_val)
    assert inst.reward(widths, "cut", []) == pytest.approx(r_cut)
    batches = inst.batches("fs2", 3)
    r_fs = inst.reward(widths, "fs2", batches)
    plan_agent.short_finetune(real, batches, plan_agent.fs_lr())
    assert r_fs == pytest.approx(100.0 * (plan_agent.accuracy(real, inst.val) - inst.origin_val))
    assert inst.recipe_line is None


# ------------------------------------------------------------------ the trainer

def _prepare(monkeypatch, tmp_path, inst, **knobs):
    """The trainer on the toy instance; returns the log lines (``run`` is the caller's)."""
    monkeypatch.setattr(plan_agent.NetInstance, "from_env", classmethod(lambda cls, env, p, m, l, **kw: inst))
    monkeypatch.setattr(plan_trainer, "out_dir", lambda: str(tmp_path))
    env = {"SPECTRA_PLAN_INSTANCES": "2", "SPECTRA_PLAN_K": "2", "SPECTRA_PLAN_BATCH": "1", "SPECTRA_PLAN_PROXY": "bn2",
           "SPECTRA_PLAN_REF_EVERY": "5", "SPECTRA_PLAN_SAVE_EVERY": "2", "SPECTRA_PLAN_SUMMARY_KAPPAS": "0.6",
           "SPECTRA_PLAN_NETS": "toy"}
    env.update(knobs)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    return lines


def _records(tmp_path):
    with open(tmp_path / "plan_train.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_trainer_runs_with_the_ft1_reward_and_logs_the_recipe_once(monkeypatch, tmp_path, v10_env):
    inst = _instance()
    lines = _prepare(monkeypatch, tmp_path, inst, SPECTRA_PLAN_INSTANCES="3", SPECTRA_PLAN_PROXY="ft1",
                     SPECTRA_PLAN_REF_EVERY="2", SPECTRA_FT_OPTIM="sgd")
    plan_trainer.run(None, [("toy.pt", (None, None)), ("other.pt", (None, None))])
    assert os.environ["SPECTRA_FT_OPTIM"] == "sgd"
    head = [line for line in lines if line.startswith("[plan] trainer ")]
    assert len(head) == 1 and '"proxy": "ft1"' in head[0] and " init=" not in head[0] and "fs_lr" not in head[0]
    recipe = [line for line in lines if line.startswith("[plan] ft recipe toy.pt: ")]
    assert len(recipe) == 1 and "optim=adam" in recipe[0] and "patience=2 epochs=1" in recipe[0]
    assert recipe[0] == f"[plan] ft recipe toy.pt: {inst.recipe_line}"
    assert not any("Fine-tune recipe:" in line or "Epoch 1/1" in line for line in lines)  # the handler's lines stay out
    steps = [line for line in lines if line.startswith("[plan] it=")]
    assert len(steps) == 3 and all(STEP_LINE.match(line) for line in steps), steps
    assert lines.index(recipe[0]) < lines.index(steps[0])
    records = _records(tmp_path)
    its = [r for r in records if "it" in r]
    assert len(its) == 3 and all(len(r["rewards"]) == 2 and all(math.isfinite(v) for v in r["rewards"]) for r in its)
    assert "mean_plan" in its[0] and set(its[1]["refs"]) == {"uniform", "sens", "inner"} and "refs" not in its[2]
    assert len([r for r in records if "summary" in r]) == 1
    _policy, blob = plan_agent.load_policy(str(tmp_path / "policy_latest.pt"))
    assert blob["proxy"] == "ft1" and blob["recipe_line"] == inst.recipe_line
    assert "fs_lr" not in blob and "init" not in blob
    assert any(line.startswith("[plan] DONE 3 instances") for line in lines)


def test_trainer_without_the_flags_writes_neither_recipe_nor_init(monkeypatch, tmp_path, v10_env):
    inst = _instance()
    lines = _prepare(monkeypatch, tmp_path, inst)
    mus = []
    draw = plan_agent.sample_scores
    monkeypatch.setattr(plan_agent, "sample_scores",
                        lambda mu, sigma, k, g=None: mus.append(mu.detach().clone()) or draw(mu, sigma, k, g))
    plan_trainer.run(None, [("toy.pt", (None, None))])
    assert torch.equal(mus[0], torch.zeros(inst.n_groups))  # the zero start
    head = [line for line in lines if line.startswith("[plan] trainer ")][0]
    assert '"proxy": "bn2"' in head and " init=" not in head
    assert not any(line.startswith(("[plan] ft recipe", "[plan] init ")) for line in lines)
    assert inst.recipe_line is None
    _policy, blob = plan_agent.load_policy(str(tmp_path / "policy_latest.pt"))
    assert "init" not in blob and "recipe_line" not in blob and blob["proxy"] == "bn2"


# ------------------------------------------------------------------ the warm start

def _checkpoint(tmp_path, run, seed, **meta):
    torch.manual_seed(seed)
    policy = plan_agent.PlanPolicy(meta.pop("feature_dim", 63))
    with torch.no_grad():
        policy.head.weight.normal_()
        policy.head.bias.fill_(0.3)
    path = tmp_path / run / "plan_agent" / "policy_latest.pt"
    saved = {"tokens": "layer", "zero": [], "k_min": 0.1, "budget": "params", "proxy": "bn8", "nets": ["a.pt", "b.pt"]}
    saved.update(meta)
    plan_agent.save_policy(policy, str(path), **saved)
    return policy.eval(), str(path)


def test_init_loads_the_checkpoints_weights_and_records_the_path(monkeypatch, tmp_path, v10_env):
    inst = _instance()
    policy, path = _checkpoint(tmp_path, "job1", 1)
    lines = _prepare(monkeypatch, tmp_path, inst, SPECTRA_PLAN_INIT=path)
    mus = []
    draw = plan_agent.sample_scores
    monkeypatch.setattr(plan_agent, "sample_scores",
                        lambda mu, sigma, k, g=None: mus.append(mu.detach().clone()) or draw(mu, sigma, k, g))
    plan_trainer.run(None, [("toy.pt", (None, None))])
    records = _records(tmp_path)
    steps = [r for r in records if "it" in r]
    with torch.no_grad():
        want = policy(inst.state_at(steps[0]["kappa"], "params"), inst.token_mask, inst.token_k, inst.n_groups)
    assert float(want.abs().sum()) > 0
    torch.testing.assert_close(mus[0], want)  # the first mu is the checkpoint's on the same state
    cfg = plan_trainer.config()
    assert [r["sigma"] for r in steps] == [plan_trainer.sigma_at(cfg, it) for it in range(2)]  # the run's own schedule
    head = [line for line in lines if line.startswith("[plan] trainer ")][0]
    assert head.endswith(f" init={path}") and '"proxy": "bn2"' in head
    assert any(line == f"[plan] init {path}: policy weights loaded (trained with proxy='bn8' budget='params' on 2 nets)"
               for line in lines)
    assert not any("WARNING" in line for line in lines)
    for tag in ("policy_latest.pt", "policy_it00002.pt"):
        blob = plan_agent.read_blob(str(tmp_path / tag))
        assert blob["init"] == path and blob["proxy"] == "bn2" and blob["tokens"] == "layer"
    assert "init" not in plan_agent.read_blob(path)  # the source is untouched
    assert len([r for r in records if "summary" in r]) == 1


@pytest.mark.parametrize("meta", [{"k_min": 0.2}, {"min_width": 2}, {"budget": "flops"}, {"tokens": "group"},
                                  {"zero": ["sens"]}, {"feature_dim": 64}, {"encoder": "set"},
                                  {"residual": "sens_cost", "residual_alpha": 0.5}])
def test_init_refuses_a_checkpoint_trained_under_other_settings(monkeypatch, tmp_path, v10_env, meta):
    inst = _instance()
    _policy, path = _checkpoint(tmp_path, "job1", 1, **meta)
    lines = _prepare(monkeypatch, tmp_path, inst, SPECTRA_PLAN_INIT=path)
    with pytest.raises(ValueError, match="SPECTRA_PLAN_INIT"):
        plan_trainer.run(None, [("toy.pt", (None, None))])
    assert not any(line.startswith(("[plan] init ", "[plan] it=")) for line in lines)
    assert not (tmp_path / "policy_latest.pt").exists()


def test_init_warns_on_another_standardizer_and_fails_fast_on_a_missing_file(monkeypatch, tmp_path, v10_env):
    inst = _instance()
    _policy, path = _checkpoint(tmp_path, "job1", 1, standardizer="/elsewhere/standardizer.json")
    lines = _prepare(monkeypatch, tmp_path, inst, SPECTRA_PLAN_INIT=path)
    plan_trainer.run(None, [("toy.pt", (None, None))])
    warned = [line for line in lines if line.startswith("[plan] WARNING: SPECTRA_PLAN_INIT trained under standardizer")]
    assert len(warned) == 1 and "/elsewhere/standardizer.json" in warned[0]
    assert any(line.startswith(f"[plan] init {path}: policy weights loaded") for line in lines)
    missing = str(tmp_path / "nowhere" / "policy_latest.pt")
    lines = _prepare(monkeypatch, tmp_path, inst, SPECTRA_PLAN_INIT=missing)
    with pytest.raises(FileNotFoundError):
        plan_trainer.run(None, [("toy.pt", (None, None))])
    assert lines == []  # before the config line and the origin reads


def test_warm_start_checks_the_keys_and_loads_in_place(monkeypatch):
    policy = plan_agent.PlanPolicy(63)
    torch.manual_seed(3)
    source = plan_agent.PlanPolicy(63)
    with torch.no_grad():
        source.head.weight.normal_()
    blob = {"state_dict": source.state_dict(), "feature_dim": 63, "encoder": "transformer", "tokens": "layer",
            "zero": [], "k_min": 0.1, "budget": "params", "standardizer": ""}
    meta = {"tokens": "layer", "zero": [], "k_min": 0.1, "budget": "params", "standardizer": ""}
    printed = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: printed.append(str(s)))
    params = list(policy.parameters())
    plan_trainer.warm_start(policy, blob, meta, "/ckpt.pt")
    assert all(p is q for p, q in zip(params, policy.parameters()))  # the optimizer's tensors, filled in place
    torch.testing.assert_close(policy.head.weight, source.head.weight)
    with pytest.raises(ValueError, match="k_min"):
        plan_trainer.warm_start(policy, {**blob, "k_min": 0.2}, meta, "/ckpt.pt")
    with pytest.raises(ValueError, match="residual_zmax"):
        plan_trainer.warm_start(policy, {**blob, "residual_zmax": 0.3}, meta, "/ckpt.pt")
    plan_trainer.warm_start(policy, {**blob, "standardizer": "/a/b.json"}, meta, "/ckpt.pt")  # a warning only
    warned = [line for line in printed if line.startswith("[plan] WARNING: SPECTRA_PLAN_INIT trained under")]
    assert len(warned) == 1 and "standardizer /a/b.json" in warned[0]
    assert sum(line.startswith("[plan] init /ckpt.pt: policy weights loaded") for line in printed) == 2
