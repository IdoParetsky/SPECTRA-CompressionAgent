"""CPU tests for TRACE (``SPECTRA_EVAL_FT_TRACE``, tree_v40): a val read along the final fine-tune through
``train_model``'s ``step_hook``. The read must leave training exactly as it was: the same final weights, the same
global random state (a ``DataLoader`` iterator draws a base seed from it even without shuffling), the same
module train/eval flags (a frozen BatchNorm stays in eval()).

    python -m pytest tests/test_ft_trace.py -v
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from src.ft_trace import FtTrace  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
from src.Configuration.StaticConf import StaticConf  # noqa: E402

KEYS = ("SPECTRA_EVAL_FT_TRACE", "SPECTRA_AMP", "SPECTRA_CHANNELS_LAST", "SPECTRA_FT_MIXUP", "SPECTRA_FT_KD",
        "SPECTRA_FT_SCHEDULE", "SPECTRA_FT_LABEL_SMOOTH", "SPECTRA_FT_OPTIM", "SPECTRA_FT_SGD_LR",
        "SPECTRA_FT_MOMENTUM", "SPECTRA_FT_WD", "SPECTRA_FT_COSINE")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    # the final fine-tune's recipe: SGD, momentum 0.9, wd 5e-4, cosine
    monkeypatch.setenv("SPECTRA_FT_OPTIM", "sgd")
    monkeypatch.setenv("SPECTRA_FT_SGD_LR", "0.05")
    monkeypatch.setenv("SPECTRA_FT_MOMENTUM", "0.9")
    monkeypatch.setenv("SPECTRA_FT_WD", "5e-4")
    monkeypatch.setenv("SPECTRA_FT_COSINE", "1")
    yield


def _net():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
        nn.Conv2d(4, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(6, 10))


def _data(n, seed):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, 3, 8, 8, generator=g), torch.randint(0, 10, (n,), generator=g))


def _train_loader(n=56):
    # the default generator: the shuffle draws from the global torch RNG
    return DataLoader(_data(n, 1), shuffle=True, batch_size=8)


def _val_loader():
    return DataLoader(_data(24, 2), shuffle=False, batch_size=10)


def _seed_all(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _run(trace_factory=None, n=56, every=2, model=None, epochs=3):
    """Train ``epochs`` epochs of ``n`` samples (batch 8); the model, seeds and loaders are the same every call."""
    model = model if model is not None else _net()
    train, val = _train_loader(n), _val_loader()
    _seed_all()
    handler = ClassificationHandler(model, nn.CrossEntropyLoss())
    trace = trace_factory(val, every) if trace_factory else None
    handler.train_model(train, allow_reinit_retry=False, max_epochs=epochs, patience=epochs + 1, keep_last=True,
                        **({"step_hook": trace} if trace is not None else {}))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model, handler, val, trace, state, torch.get_rng_state()


def _device():
    return StaticConf.get_instance().conf_values.device


def _make(cls=FtTrace, **kw):
    return lambda val, every: cls(every, val, _device(), "val_best", "net.pt", "path/net.pt", **kw)


class _NoRngRestore(FtTrace):
    """The same read without putting the random states back: what the hook must not do."""

    @staticmethod
    def _restore_rng(states):
        pass


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_flag_parses(monkeypatch):
    assert fortify.eval_ft_trace() == 0
    monkeypatch.setenv("SPECTRA_EVAL_FT_TRACE", "25")
    assert fortify.eval_ft_trace() == 25
    monkeypatch.setenv("SPECTRA_EVAL_FT_TRACE", "-3")
    assert fortify.eval_ft_trace() == 0
    monkeypatch.setenv("SPECTRA_EVAL_FT_TRACE", "junk")
    assert fortify.eval_ft_trace() == 0


def test_traced_training_is_byte_identical():
    _, _, _, _, plain, plain_rng = _run()
    _, _, _, trace, traced, traced_rng = _run(_make())
    assert len(trace.records) > 3
    assert _same(plain, traced)
    assert torch.equal(plain_rng, traced_rng)


def test_the_test_has_teeth_without_the_rng_restore():
    # a val loader's iterator draws its base seed from the global torch RNG, so a read that does not restore the
    # random state moves the training shuffle: the final weights and the random state both differ
    _, _, _, _, plain, plain_rng = _run()
    _, _, _, trace, broken, broken_rng = _run(_make(_NoRngRestore))
    assert len(trace.records) > 3
    assert not torch.equal(plain_rng, broken_rng)
    assert not _same(plain, broken)


def test_read_schedule_odd_total_steps():
    _, _, _, trace, _, _ = _run(_make(), n=56, every=2)  # 7 steps per epoch x 3 = 21
    steps = [r["step"] for r in trace.records]
    assert steps == list(range(0, 21, 2)) + [21]
    assert trace.final_done


def test_read_schedule_no_duplicate_at_a_multiple_of_every():
    _, _, _, trace, _, _ = _run(_make(), n=64, every=4)  # 8 steps per epoch x 3 = 24, a multiple of 4
    steps = [r["step"] for r in trace.records]
    assert steps == list(range(0, 25, 4))
    assert trace.final_done


def test_skipped_batches_are_not_counted():
    # 57 samples at batch 8: a last batch of 1 sample, skipped by train_model; 7 steps per epoch
    _, _, _, trace, _, _ = _run(_make(), n=57, every=5, epochs=2)
    assert [r["step"] for r in trace.records] == [0, 5, 10, 14]


def test_last_read_equals_evaluate_model_on_the_final_weights():
    model, handler, val, trace, _, _ = _run(_make())
    assert trace.records[-1]["val_acc"] == float(handler.evaluate_model(val))
    assert 0.0 <= trace.records[-1]["val_acc"] <= 1.0
    assert trace.records[-1]["val_loss"] > 0.0


def test_record_fields_printer_and_recorder():
    lines, events = [], []

    class _Recorder:
        def record(self, event, **fields):
            events.append((event, fields))

    _, _, _, trace, _, _ = _run(_make(recorder=_Recorder(), printer=lambda s: lines.append(s)), every=5)
    assert len(events) == len(lines) == len(trace.records)
    assert {e for e, _ in events} == {"eval_ft_trace"}
    first, last = events[0][1], events[-1][1]
    assert first["network"] == "path/net.pt" and first["label"] == "val_best" and first["step"] == 0
    assert first["train_loss"] is None and first["grad_norm"] is None and first["lr"] is None
    assert first["wmove_mean"] == 0.0 and first["n_conv"] == 2
    assert last["train_loss"] > 0.0 and last["grad_norm"] > 0.0 and last["lr"] is not None
    assert 0.0 < last["wmove_mean"] <= last["wmove_max"] and last["wmove_first"] > 0 and last["wmove_last"] > 0
    assert lines[0].startswith("[eval] FT_TRACE val_best net.pt step=0 | val ")
    assert "train loss - | gnorm - | lr -" in lines[0]
    assert "wmove mean" in lines[-1] and lines[-1].endswith(" s")


def test_a_model_without_convolutions_reports_no_weight_movement():
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
    _, _, _, trace, _, _ = _run(_make(), model=net)
    assert all(r["n_conv"] == 0 and r["wmove_mean"] is None and r["wmove_max"] is None for r in trace.records)


def test_frozen_batchnorm_stays_in_eval_and_flags_are_restored():
    def frozen_net():
        net = _net()
        bn = net[1]
        for p in bn.parameters():
            p.requires_grad = False
        bn.running_mean.fill_(0.5)
        return net

    base = frozen_net()[1].running_mean.clone()
    plain_model, _, _, _, plain, _ = _run(model=frozen_net())
    traced_model, _, _, _, traced, _ = _run(_make(), model=frozen_net())
    assert torch.equal(plain_model[1].running_mean, base)  # train_model holds it in eval()
    assert torch.equal(traced_model[1].running_mean, base)
    assert not torch.equal(traced_model[4].running_mean, torch.zeros(6))  # the trainable one did move
    assert _same(plain, traced)

    # a read leaves every module's flag as it found it, mixed flags included
    net = frozen_net()
    net.train()
    net[1].eval()
    net[7].eval()
    before = [(m, m.training) for m in net.modules()]
    trace = _make()(_val_loader(), 2)
    trace.on_start(net)
    assert [(m, m.training) for m in net.modules()] == before
    trace._read(1)
    assert [(m, m.training) for m in net.modules()] == before
