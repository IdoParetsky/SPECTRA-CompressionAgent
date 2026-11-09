"""
E1 (9 Nov): a faster cut walk with exactly the same output. Three default-off flags and a read-only diagnostic, CPU only.

* ``SPECTRA_EVAL_IDENTITY_MEMO=1``: an identity step reuses the last val read of the unchanged model and prints the two
  lines ``evaluate_model`` prints; cut steps always read; cleared at reset and rollback; a non-persistent loader's one
  seed draw is burned;
* ``SPECTRA_EVAL_SKIP_STATE=1``: ``NetworkEnv.step`` skips the next-state encoding when the runner says the policy
  never reads it (non-actor ``EVAL_TEST``, importance not random);
* ``SPECTRA_EVAL_CACHE_READS`` is tested in ``tests/test_e1_eval_cache.py``;
* ``SPECTRA_EVAL_E1_DIAG=1``: ``[e1] diag`` lines with state_dict and RNG hashes; nothing altered, no RNG drawn.

Each walk below is a real ``NetworkEnv.reset`` and ``step`` over a thin ResNet-20 and TensorDataset loaders, with the
per-step fine-tune off (``is_to_train=False``).

    OMP_NUM_THREADS=4 python -m pytest tests/test_e1_fast_walk.py -v
"""

import contextlib
import copy
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from src import fortify  # noqa: E402
import src.logging_utils as logging_utils  # noqa: E402
import src.NetworkEnv as netenv  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
from src.NetworkEnv import NetworkEnv, AGENT_TRAIN, EVAL_TEST, EVAL_TRAIN  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402
import a2c_agent_reinforce_runner as runner  # noqa: E402

NET = "/x/tiny_r20w4.pt"
_REAL_STAGE = logging_utils.stage
_REAL_EVALUATE = ClassificationHandler.evaluate_model
RATES = [1.0, 0.6, 1.0, 1.0, 0.7, 1.0]  # four identity steps, two cuts


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SPECTRA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    monkeypatch.setattr(recorder, "issue", lambda *a, **k: None)
    yield


# ------------------------------------------------------------------------------------------------ fixtures


def _data(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return TensorDataset(x, y)


def _loaders(workers):
    """(train, val, test); val and test are persistent multi-worker loaders when ``workers`` > 0."""
    kw = {"batch_size": 8, "shuffle": False}
    if workers:
        kw.update(num_workers=workers, persistent_workers=True)
    return DataLoader(_data(16, 0), batch_size=8), DataLoader(_data(16, 1), **kw), DataLoader(_data(16, 2), **kw)


def _workers_or_skip():
    try:
        loader = DataLoader(_data(4, 0), batch_size=2, num_workers=2, persistent_workers=True)
        assert len(list(loader)) == 2
        del loader
    except Exception as error:  # noqa: BLE001 - no worker processes here: the burn path is still tested
        pytest.skip(f"DataLoader workers unavailable: {type(error).__name__}: {error}")


class _StubFE:
    """FeatureExtractor stand-in that only counts its calls."""
    calls = []

    def __init__(self, loader, device):
        pass

    def encode_to_bert_input(self, *args, **kwargs):
        _StubFE.calls.append(1)
        return {"state": len(_StubFE.calls)}


def _spy_reads(monkeypatch):
    """Every ``evaluate_model`` call goes to the returned list (the loader it was given)."""
    reads = []

    def spy(self, loader):
        reads.append(loader)
        return _REAL_EVALUATE(self, loader)

    monkeypatch.setattr(ClassificationHandler, "evaluate_model", spy)
    return reads


def _count_stages(monkeypatch):
    names = []

    @contextlib.contextmanager
    def counting(name, *args, **kwargs):
        names.append(name)
        with _REAL_STAGE(name, *args, **kwargs):
            yield

    monkeypatch.setattr(logging_utils, "stage", counting)
    return names


def _env(monkeypatch, *, memo=True, workers=0, real_fe=False):
    """A reset ``NetworkEnv`` (EVAL_TEST) around a thin ResNet-20, and its loaders."""
    if memo:
        monkeypatch.setenv("SPECTRA_EVAL_IDENTITY_MEMO", "1")
    else:
        monkeypatch.delenv("SPECTRA_EVAL_IDENTITY_MEMO", raising=False)
    if not real_fe:
        monkeypatch.setattr(netenv, "FeatureExtractor", _StubFE)
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    loaders = _loaders(workers)
    env = NetworkEnv(networks=[NET], mode=EVAL_TEST)
    torch.manual_seed(1234)
    env.reset(test_net_path=NET, test_model=model, test_loaders=loaders)
    return env, loaders


def _mask(line):
    return re.sub(r"done in [0-9.]+s", "done in Xs", line)


def _walk(monkeypatch, rates, *, memo=False, skip=False, workers=0, real_fe=False):
    """One scripted walk; what a quoted run would carry, with the timing masked."""
    lines, records = [], []
    monkeypatch.setattr(utils, "print_flush", lambda msg: lines.append(_mask(str(msg))))
    monkeypatch.setattr(recorder, "record", lambda kind, **kw: records.append(
        (kind, {k: v for k, v in kw.items() if k != "seconds"})))
    stages = _count_stages(monkeypatch)
    reads = _spy_reads(monkeypatch)
    _StubFE.calls = []
    env, loaders = _env(monkeypatch, memo=memo, workers=workers, real_fe=real_fe)
    env.skip_state_encoding = skip
    states = [env.step(rate, is_to_train=False)[0] for rate in rates]
    out = {
        "actions": list(env.actions_history),
        "records": records,
        "lines": lines,
        "state_dict": {k: v.detach().clone() for k, v in env.current_model.state_dict().items()},
        "rng": torch.get_rng_state(),
        "stages": stages,
        "reads": len(reads),
        "states": states,
        "hits": getattr(env, "_memo_hits", 0),
        "identity_steps": getattr(env, "_memo_identity_steps", 0),
        "skips": getattr(env, "_state_skips", 0),
        "last_val_acc": env.last_val_acc,
        "params": utils.calc_num_parameters(env.current_model),
        "origin_params": env.original_params,
    }
    del env, loaders
    return out


def _assert_same_output(a, b):
    assert a["actions"] == b["actions"]
    assert a["records"] == b["records"]
    assert a["lines"] == b["lines"]
    assert a["last_val_acc"] == b["last_val_acc"]
    assert a["state_dict"].keys() == b["state_dict"].keys()
    for key, value in a["state_dict"].items():
        assert torch.equal(value, b["state_dict"][key]), key
    assert torch.equal(a["rng"], b["rng"])


def _refresh(env):
    """A real val read of the current model, remembered the way ``step`` remembers it."""
    handler = env.create_learning_handler(env.current_model)
    handler.evaluate_model(env.val_loader)
    env._val_memo_store(handler)


# ------------------------------------------------------------------------------------------------ flags


def test_flags_default_off_and_parse(monkeypatch):
    assert not fortify.eval_identity_memo() and not fortify.eval_skip_state()
    assert not fortify.eval_cache_reads() and not fortify.eval_e1_diag()
    assert fortify.eval_cache_reads_mb() == 1024
    for name, getter in (("SPECTRA_EVAL_IDENTITY_MEMO", fortify.eval_identity_memo),
                         ("SPECTRA_EVAL_SKIP_STATE", fortify.eval_skip_state),
                         ("SPECTRA_EVAL_CACHE_READS", fortify.eval_cache_reads),
                         ("SPECTRA_EVAL_E1_DIAG", fortify.eval_e1_diag)):
        monkeypatch.setenv(name, "1")
        assert getter() is True
        monkeypatch.setenv(name, "0")
        assert getter() is False
    monkeypatch.setenv("SPECTRA_EVAL_CACHE_READS_MB", "256")
    assert fortify.eval_cache_reads_mb() == 256
    monkeypatch.setenv("SPECTRA_EVAL_CACHE_READS_MB", "lots")
    assert fortify.eval_cache_reads_mb() == 1024


# ------------------------------------------------------------------------------------------------ (a) flags unset


def test_a_flags_unset_every_step_reads_and_encodes(monkeypatch):
    out = _walk(monkeypatch, RATES)
    assert out["reads"] == 1 + len(RATES)                      # the reset baseline and every step, identity or cut
    assert len(_StubFE.calls) == 1 + len(RATES)                # the reset state and every step's next state
    assert all(state is not None for state in out["states"])
    assert (out["hits"], out["identity_steps"], out["skips"]) == (0, 0, 0)
    assert out["stages"].count("step.evaluate") == len(RATES)
    assert out["stages"].count("step.feature_extraction") == len(RATES)


def test_evaluate_model_keeps_its_lines_and_return_and_records_last_eval(monkeypatch):
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda msg: lines.append(str(msg)))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)).eval()
    handler = ClassificationHandler(model, nn.CrossEntropyLoss())
    _, val, _ = _loaders(0)
    acc = handler.evaluate_model(val)
    assert isinstance(acc, float) and handler.last_eval[0] == acc
    assert lines == [f"Accuracy: {acc:.3f}", f"Average Loss: {handler.last_eval[1]:.3f}"]


# ------------------------------------------------------------------------------------------------ (b) memo on = off


@pytest.mark.parametrize("workers", [0, 2])
def test_b_memo_walk_is_identical_to_the_stock_walk(monkeypatch, workers):
    if workers:
        _workers_or_skip()
    off = _walk(monkeypatch, RATES, memo=False, workers=workers)
    on = _walk(monkeypatch, RATES, memo=True, workers=workers)
    _assert_same_output(off, on)
    assert any(line.startswith("Accuracy: ") for line in on["lines"])
    assert on["identity_steps"] == 4 and on["hits"] == 4       # the reset seeds step 0; each cut refreshes the memo
    assert on["reads"] == off["reads"] - 4                      # the four identity reads are the only ones gone
    assert off["hits"] == 0 and off["identity_steps"] == 0
    assert on["stages"] == off["stages"]                        # the same stages, the same counts
    assert on["params"] < on["origin_params"]                   # the walk really cut something


def test_b_the_burn_is_exactly_the_draw_a_new_iterator_makes(monkeypatch):
    env, loaders = _env(monkeypatch, workers=0)
    assert netenv._memo_loader_draws(env.val_loader) == (1, None)
    start = torch.get_rng_state()
    assert env._val_memo_read(env.current_model) is not None
    after_read = torch.get_rng_state()
    torch.set_rng_state(start)
    iter(env.val_loader)                                        # what the skipped read would have done
    assert torch.equal(after_read, torch.get_rng_state())


def test_b_persistent_workers_burn_nothing(monkeypatch):
    _workers_or_skip()
    env, loaders = _env(monkeypatch, workers=2)
    assert netenv._memo_loader_draws(env.val_loader) == (0, None)
    start = torch.get_rng_state()
    assert env._val_memo_read(env.current_model) is not None
    assert torch.equal(start, torch.get_rng_state())


def test_b_only_plain_sequential_loaders_may_skip(monkeypatch):
    data = _data(16, 1)
    assert netenv._memo_loader_draws(DataLoader(data, batch_size=8)) == (1, None)
    gen = torch.Generator().manual_seed(3)
    assert netenv._memo_loader_draws(DataLoader(data, batch_size=8, generator=gen)) == (1, gen)
    assert netenv._memo_loader_draws(DataLoader(data, batch_size=8, shuffle=True)) is None
    assert netenv._memo_loader_draws(DataLoader(data, batch_size=None)) is None
    assert netenv._memo_loader_draws([(torch.zeros(2, 3), torch.zeros(2))]) is None
    assert netenv._memo_loader_draws(
        DataLoader(data, batch_sampler=torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(data), 8, False))) is None


# ------------------------------------------------------------------------------------------------ (c) invalidation


def test_c_the_memo_is_invalidated_by_anything_that_changes_what_a_read_sees(monkeypatch):
    env, loaders = _env(monkeypatch, workers=0)
    model = env.current_model
    assert env._val_memo_read(model) is not None                # seeded by the reset's baseline read

    # an in-place BN running_mean edit
    bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    bn.running_mean.add_(0.5)
    assert env._val_memo_read(model) is None
    _refresh(env)
    assert env._val_memo_read(model) is not None

    # an in-place weight edit that leaves every value as it was: the version counter still moves
    with torch.no_grad():
        next(model.parameters()).mul_(1.0)
    assert env._val_memo_read(model) is None
    _refresh(env)

    # train mode (a step calls .eval() first, so only a direct read sees it)
    model.train()
    assert env._val_memo_read(model) is None
    model.eval()
    assert env._val_memo_read(model) is not None

    # a different loader object
    other_val = _loaders(0)[1]
    own_val, env.val_loader = env.val_loader, other_val
    assert env._val_memo_read(model) is None
    env.val_loader = own_val
    assert env._val_memo_read(model) is not None

    # SPECTRA_AMP and SPECTRA_CHANNELS_LAST
    for name in ("SPECTRA_AMP", "SPECTRA_CHANNELS_LAST"):
        monkeypatch.setenv(name, "1")
        assert env._val_memo_read(model) is None, name
        monkeypatch.delenv(name)
        assert env._val_memo_read(model) is not None, name

    # rollback_to
    snapshot = env.rollback_snapshot()
    env.rollback_to(snapshot)
    assert env._val_memo is None
    assert env._val_memo_read(env.current_model) is None

    # a new model object (same weights)
    _refresh(env)
    assert env._val_memo_read(env.current_model) is not None
    env.current_model = copy.deepcopy(env.current_model)
    assert env._val_memo_read(env.current_model) is None

    # reset clears and re-seeds
    env.reset(test_net_path=NET, test_model=resnet20(num_classes=10, large_input=False, width=4).eval(),
              test_loaders=loaders)
    assert env._val_memo_read(env.current_model) is not None
    assert (env._memo_hits, env._memo_identity_steps) == (0, 0)


def test_c_agent_train_never_uses_the_memo(monkeypatch):
    env, _ = _env(monkeypatch, workers=0)
    assert env._val_memo_enabled()
    env.mode = AGENT_TRAIN
    assert not env._val_memo_enabled()
    env.mode = EVAL_TRAIN
    assert env._val_memo_enabled()


def test_c_the_memo_is_off_without_the_flag(monkeypatch):
    env, _ = _env(monkeypatch, memo=False, workers=0)
    assert env._val_memo is None
    assert env._val_memo_read(env.current_model) is None


# ------------------------------------------------------------------------------------------------ (d) cut steps


def test_d_cut_steps_always_evaluate(monkeypatch):
    reads = _spy_reads(monkeypatch)
    env, _ = _env(monkeypatch, workers=0)
    base = len(reads)
    assert base == 1
    params = utils.calc_num_parameters(env.current_model)
    env.step(1.0, is_to_train=False)                            # identity: the memo answers
    assert len(reads) == base
    env.step(0.6, is_to_train=False)                            # cut: a real read
    assert len(reads) == base + 1
    env.step(1.0, is_to_train=False)                            # identity after a cut: answered by the cut's read
    assert len(reads) == base + 1
    env.step(0.7, is_to_train=False)
    env.step(0.7, is_to_train=False)
    assert len(reads) == base + 3
    assert utils.calc_num_parameters(env.current_model) < params
    assert env._memo_identity_steps == 2 and env._memo_hits == 2
    assert all(loader is env.val_loader for loader in reads)


# ------------------------------------------------------------------------------------------------ (e) skip the state


@pytest.mark.parametrize("flag,mode,policy,importance,expected", [
    (True, EVAL_TEST, "alloc", None, True),
    (True, EVAL_TEST, "mild", "l2", True),
    (True, EVAL_TEST, "actor", None, False),
    (True, EVAL_TEST, "alloc", "random", False),
    (True, EVAL_TRAIN, "alloc", None, False),
    (True, AGENT_TRAIN, "alloc", None, False),
    (False, EVAL_TEST, "alloc", None, False),
])
def test_e_skip_state_gate(monkeypatch, flag, mode, policy, importance, expected):
    if flag:
        monkeypatch.setenv("SPECTRA_EVAL_SKIP_STATE", "1")
    if importance:
        monkeypatch.setenv("SPECTRA_FILTER_IMPORTANCE", importance)
    assert runner._e1_skip_state(fortify, mode, policy) is expected


def test_e_skipped_state_walk_is_identical_and_keeps_the_stages(monkeypatch):
    stock = _walk(monkeypatch, [1.0, 0.6, 1.0], real_fe=True)
    skipped = _walk(monkeypatch, [1.0, 0.6, 1.0], skip=True, real_fe=True)
    _assert_same_output(stock, skipped)
    assert all(state is not None for state in stock["states"])
    assert all(state is None for state in skipped["states"])
    assert skipped["skips"] == 3 and stock["skips"] == 0
    assert skipped["stages"] == stock["stages"]                 # the stage is still entered on every step
    assert skipped["stages"].count("step.feature_extraction") == 3


def test_e_skip_and_memo_together_are_identical_too(monkeypatch):
    stock = _walk(monkeypatch, RATES)
    both = _walk(monkeypatch, RATES, memo=True, skip=True)
    _assert_same_output(stock, both)
    assert both["hits"] == 4 and both["skips"] == len(RATES)
    assert len(_StubFE.calls) == 1                              # only the reset encoded a state


# ------------------------------------------------------------------------------------------------ DIAG


def test_diag_hash_is_read_only_and_sensitive():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Flatten(), nn.Linear(4 * 30 * 30, 2))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    digest = runner._e1_state_hash(model)
    assert len(digest) == 64 and digest == runner._e1_state_hash(model)
    assert torch.equal(rng, torch.get_rng_state())              # no RNG drawn
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    with torch.no_grad():
        model[0].weight[0, 0, 0, 0] += 1.0
    assert runner._e1_state_hash(model) != digest
    assert runner._e1_state_hash(copy.deepcopy(model).double()) != runner._e1_state_hash(model)   # dtype counts
    cpu, cuda = runner._e1_rng_hashes()
    assert len(cpu) == 64 and (cuda == "none" or len(cuda) == 64)
    assert torch.equal(rng, torch.get_rng_state())


def _candidate(step=9):
    point = {"step": step, "param": 0.8, "flop": 0.8, "val_dacc_pp": -1.0, "test_acc": 0.5}
    return {"point": point, "model": nn.Linear(2, 2), "key": (0.8, 0.8, 1.0)}


@pytest.mark.parametrize("diag", [False, True])
def test_diag_lines_only_with_the_flag(monkeypatch, diag):
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda msg: lines.append(str(msg)))
    monkeypatch.setattr(runner, "_final_ft", lambda *args, **kwargs: None)
    if diag:
        monkeypatch.setenv("SPECTRA_EVAL_E1_DIAG", "1")
    runner._run_final_ft(SimpleNamespace(), "/x/net.pt", {"val_best": {"step": 9}}, {"val_best": _candidate()}, 1)
    e1 = [line for line in lines if "[e1]" in line]
    if not diag:
        assert e1 == []
    else:
        assert len(e1) == 1
        assert re.fullmatch(r"\[e1\] diag pre val_best sd=[0-9a-f]{64} cpu_rng=[0-9a-f]{64} cuda_rng=([0-9a-f]{64}|none)",
                            e1[0])


def test_summary_line_only_when_a_flag_is_set(monkeypatch):
    assert runner._e1_flags_on(fortify) is False
    for name in ("SPECTRA_EVAL_IDENTITY_MEMO", "SPECTRA_EVAL_SKIP_STATE", "SPECTRA_EVAL_CACHE_READS",
                 "SPECTRA_EVAL_E1_DIAG"):
        monkeypatch.setenv(name, "1")
        assert runner._e1_flags_on(fortify) is True
        monkeypatch.delenv(name)
