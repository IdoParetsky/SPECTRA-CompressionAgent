"""
E1 (9 Nov): ``SPECTRA_EVAL_CACHE_READS=1``, the walk's val and test reads replayed from host memory, CPU only.

``utils.HostCachedEvalLoader`` keeps the batches a persistent multi-worker DataLoader yields on its first complete
pass and replays them; a partial pass fills nothing; ``utils.wrap_eval_loader_cached`` refuses what it cannot wrap
without changing the RNG stream or the memory budget; the runner restores the original loaders before the final
fine-tune.

    OMP_NUM_THREADS=4 python -m pytest tests/test_e1_eval_cache.py -v
"""

import copy
import gc
import os
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
import src.utils as utils  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
import a2c_agent_reinforce_runner as runner  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SPECTRA_"):
            monkeypatch.delenv(key, raising=False)
    yield


# ------------------------------------------------------------------------------------------------ fixtures


def _dataset(n=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, 3, 8, 8, generator=g), torch.randint(0, 10, (n,), generator=g))


def _persistent(n=20, batch=6, seed=0):
    """Four batches of 6, 6, 6 and 2, read by two persistent workers."""
    return DataLoader(_dataset(n, seed), batch_size=batch, shuffle=False, num_workers=2, persistent_workers=True)


def _workers_or_skip():
    try:
        loader = _persistent()
        assert len(list(loader)) == 4
        del loader
    except Exception as error:  # noqa: BLE001 - no worker processes here
        pytest.skip(f"DataLoader workers unavailable: {type(error).__name__}: {error}")


def _same_batches(a, b):
    assert len(a) == len(b)
    for (xa, ya), (xb, yb) in zip(a, b):
        assert xa.dtype == xb.dtype and xa.shape == xb.shape and ya.dtype == yb.dtype and ya.shape == yb.shape
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10)).eval()


# ------------------------------------------------------------------------------------------------ (f) the batches


def test_f_cached_batches_equal_the_inner_loaders_on_passes_1_to_3():
    _workers_or_skip()
    inner = _persistent()
    reference = list(inner)
    wrapper = utils.HostCachedEvalLoader(inner)
    assert wrapper._batches is None
    passes = [list(wrapper) for _ in range(3)]
    for got in passes:
        _same_batches(got, reference)
    assert wrapper._batches is not None
    # passes 2 and 3 are replays of pass 1's own tensors: no copy, no transform
    for a, b in zip(passes[0], passes[2]):
        assert a[0] is b[0] and a[1] is b[1]
    assert [x.shape[0] for x, _ in passes[1]] == [6, 6, 6, 2]


# ------------------------------------------------------------------------------------------------ (g) evaluate_model


def test_g_evaluate_model_gives_the_same_accuracy_and_lines(monkeypatch):
    _workers_or_skip()
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda msg: lines.append(str(msg)))
    model = _model()

    bare = _persistent()
    stock = []
    for _ in range(3):
        del lines[:]
        acc = ClassificationHandler(model, nn.CrossEntropyLoss()).evaluate_model(bare)
        stock.append((acc, list(lines)))

    wrapper = utils.HostCachedEvalLoader(_persistent())
    for expected_acc, expected_lines in stock:                  # pass 1 fills the cache, passes 2 and 3 replay it
        del lines[:]
        handler = ClassificationHandler(model, nn.CrossEntropyLoss())
        acc = handler.evaluate_model(wrapper)
        assert acc == expected_acc
        assert lines == expected_lines
        assert handler.last_eval[0] == acc
    assert wrapper._batches is not None


# ------------------------------------------------------------------------------------------------ (h) partial passes


def test_h_a_partial_pass_does_not_fill_the_cache():
    _workers_or_skip()
    wrapper = utils.HostCachedEvalLoader(_persistent())
    first = next(iter(wrapper))                                  # get_input_shape's next(iter(...))
    gc.collect()
    assert wrapper._batches is None
    iterator = iter(wrapper)
    next(iterator)
    iterator.close()
    assert wrapper._batches is None
    for i, _ in enumerate(wrapper):                              # a loop that breaks out early
        if i == 1:
            break
    gc.collect()
    assert wrapper._batches is None
    assert first[0].shape == (6, 3, 8, 8)
    full = list(wrapper)                                         # the first complete pass commits
    assert wrapper._batches is not None and len(wrapper._batches) == 4
    _same_batches(full, wrapper._batches)
    kept = wrapper._batches
    next(iter(wrapper))                                          # a partial pass now replays and changes nothing
    assert wrapper._batches is kept
    _same_batches(list(wrapper), full)


# ------------------------------------------------------------------------------------------------ (i) the RNG


@pytest.mark.parametrize("passes", [1, 2, 5])
def test_i_the_rng_state_matches_the_inner_loaders_after_k_passes(passes):
    _workers_or_skip()
    bare = _persistent()
    torch.manual_seed(7)
    for _ in range(passes):
        list(bare)
    after_bare = torch.get_rng_state()

    wrapper = utils.HostCachedEvalLoader(_persistent())
    torch.manual_seed(7)
    for _ in range(passes):
        list(wrapper)
    assert torch.equal(after_bare, torch.get_rng_state())


# ------------------------------------------------------------------------------------------------ (j) refusal, delegation, restore


def test_j_refusals():
    inner = DataLoader(_dataset(), batch_size=6, shuffle=False)             # not persistent
    out, reason = utils.wrap_eval_loader_cached(inner, 1024)
    assert out is inner and "persistent" in reason

    out, reason = utils.wrap_eval_loader_cached([(torch.zeros(2), torch.zeros(2))], 1024)
    assert isinstance(out, list) and "DataLoader" in reason

    _workers_or_skip()
    big = _persistent()
    out, reason = utils.wrap_eval_loader_cached(big, 0)                     # over the cap
    assert out is big and "cap" in reason
    big_n = _persistent(n=4000)                                              # 4000 x (3x8x8 floats + a label)
    out, reason = utils.wrap_eval_loader_cached(big_n, 1)
    assert out is big_n and "1 MB cap" in reason

    out, reason = utils.wrap_eval_loader_cached(big, 1024)
    assert reason is None and isinstance(out, utils.HostCachedEvalLoader) and out.inner is big


def test_j_a_dataset_item_read_leaves_the_global_rng_alone():
    _workers_or_skip()

    class Noisy(TensorDataset):
        def __getitem__(self, i):
            x, y = super().__getitem__(i)
            return x + torch.rand(()), y                                     # a transform that draws

    loader = DataLoader(Noisy(torch.zeros(8, 3), torch.zeros(8, dtype=torch.long)), batch_size=4, num_workers=2,
                        persistent_workers=True)
    torch.manual_seed(3)
    before = torch.get_rng_state()
    utils.wrap_eval_loader_cached(loader, 1024)
    assert torch.equal(before, torch.get_rng_state())


def test_j_attributes_are_delegated():
    _workers_or_skip()
    inner = _persistent()
    wrapper = utils.HostCachedEvalLoader(inner)
    assert len(wrapper) == len(inner) == 4
    assert wrapper.batch_size == 6 and wrapper.num_workers == 2
    assert wrapper.dataset is inner.dataset
    assert wrapper.persistent_workers is True
    with pytest.raises(AttributeError):
        wrapper.no_such_attribute
    clone = copy.copy(wrapper)                                               # __getattr__ must not recurse
    assert clone.inner is inner


def test_j_the_runner_wraps_logs_and_restores_the_originals(monkeypatch):
    _workers_or_skip()
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda msg: lines.append(str(msg)))
    val, test = _persistent(), DataLoader(_dataset(), batch_size=6)          # the test loader is not persistent
    env = SimpleNamespace(val_loader=val, test_loader=test)

    held = runner._e1_wrap_eval_loaders(env, fortify)
    assert isinstance(env.val_loader, utils.HostCachedEvalLoader) and env.val_loader.inner is val
    assert env.test_loader is test                                           # refused: left as it was
    assert lines == ["[e1] cache val=on test=refused: not a persistent multi-worker loader"]

    runner._e1_restore_eval_loaders(env, held)
    assert env.val_loader is val and env.test_loader is test
    runner._e1_restore_eval_loaders(env, held)                               # the finally's second call is a no-op
    runner._e1_restore_eval_loaders(env, None)
    assert env.val_loader is val and env.test_loader is test


def test_j_restore_refuses_a_loader_it_did_not_wrap(monkeypatch):
    _workers_or_skip()
    monkeypatch.setattr(utils, "print_flush", lambda msg: None)
    env = SimpleNamespace(val_loader=_persistent(), test_loader=_persistent(seed=1))
    held = runner._e1_wrap_eval_loaders(env, fortify)
    env.val_loader = _persistent(seed=2)                                     # something else swapped it
    with pytest.raises(AssertionError):
        runner._e1_restore_eval_loaders(env, held)


def test_j_the_cap_comes_from_the_flag(monkeypatch):
    _workers_or_skip()
    monkeypatch.setattr(utils, "print_flush", lambda msg: None)
    monkeypatch.setenv("SPECTRA_EVAL_CACHE_READS_MB", "0")
    env = SimpleNamespace(val_loader=_persistent(), test_loader=_persistent())
    held = runner._e1_wrap_eval_loaders(env, fortify)
    assert not isinstance(env.val_loader, utils.HostCachedEvalLoader)
    runner._e1_restore_eval_loaders(env, held)
