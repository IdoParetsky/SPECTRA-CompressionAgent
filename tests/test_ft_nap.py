"""CPU tests for NAP (``SPECTRA_EVAL_FT_NAP``, tree_v41, queue row NAPJ): the NAP2-v2 logger that rides TRACE's hook
along the final fine-tune (``src/ft_nap.py``). Training with the logger attached must equal training with TRACE alone,
bit for bit (weights, optimizer state, every random state); the records must follow the registered layout; and the
numbers are checked against direct computations from the raw per-step gradients and weights.

    python -m pytest tests/test_ft_nap.py -v
"""

import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import src.ModelHandlers.ClassificationHandler as handler_mod  # noqa: E402
from src import ft_nap  # noqa: E402
from src.ft_nap import NapLogger  # noqa: E402
from src.ft_trace import FtTrace  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
from src.Configuration.StaticConf import StaticConf  # noqa: E402

KEYS = ("SPECTRA_EVAL_FT_TRACE", "SPECTRA_EVAL_FT_NAP", "SPECTRA_AMP", "SPECTRA_CHANNELS_LAST", "SPECTRA_FT_MIXUP",
        "SPECTRA_FT_KD", "SPECTRA_FT_SCHEDULE", "SPECTRA_FT_LABEL_SMOOTH", "SPECTRA_FT_OPTIM", "SPECTRA_FT_SGD_LR",
        "SPECTRA_FT_MOMENTUM", "SPECTRA_FT_WD", "SPECTRA_FT_COSINE", "SPECTRA_IMN_FT")
LABEL, NAME, NETWORK = "size_flop0.20", "net.pt", "path/net.pt"
W_FAMILIES = ("S", "R", "Y", "V", "K", "J", "F0")


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
    ft_nap._PROBE_CACHE.clear()
    ft_nap._ORIGIN_CACHE.clear()
    yield
    ft_nap._PROBE_CACHE.clear()
    ft_nap._ORIGIN_CACHE.clear()


# --- helpers ------------------------------------------------------------------------------------------------------------

def _device():
    return StaticConf.get_instance().conf_values.device


def _net(classes=10, widths=(4, 6), seed=0, bn=True):
    torch.manual_seed(seed)
    layers = [nn.Conv2d(3, widths[0], 3, padding=1)] + ([nn.BatchNorm2d(widths[0])] if bn else []) + [nn.ReLU()]
    layers += [nn.Conv2d(widths[0], widths[1], 3, padding=1)] + ([nn.BatchNorm2d(widths[1])] if bn else []) + [nn.ReLU()]
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(widths[1], classes))


def _data(n, seed, classes=10):
    g = torch.Generator().manual_seed(seed)
    labels = (torch.arange(n) % classes)[torch.randperm(n, generator=g)]
    images = torch.randn(n, 3, 8, 8, generator=g) + 0.3 * torch.randn(classes, 3, 1, 1, generator=g)[labels]
    return TensorDataset(images, labels)


def _seed_all(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class _Recorder:
    def __init__(self):
        self.events, self.issues = [], []

    def record(self, event, **fields):
        self.events.append((event, fields))

    def issue(self, kind, detail="", **fields):
        self.issues.append((kind, detail, fields))

    def of(self, event):
        return [fields for name, fields in self.events if name == event]


class _Spy(FtTrace):
    """TRACE, plus the ground truth the tests compare against: per step the loss, the pre-clip norm, the lr, the
    clipped gradient, the parameters after the step and the momentum buffer before it; the optimizer state at the end."""

    def __init__(self, *args, handler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler = handler
        self.t_loss, self.t_gnorm, self.t_lr, self.t_grad, self.t_theta, self.t_buf = [], [], [], [], [], []
        self._buf = None
        self.opt_state = None

    def _flat(self, tensors):
        return torch.cat([t.detach().reshape(-1) for t in tensors]).double().clone()

    def on_start(self, model):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.theta0 = self._flat(self.params)
        super().on_start(model)

    def on_step(self, step, loss, grad_norm, lr):
        self.t_loss.append(float(loss))
        self.t_gnorm.append(float(grad_norm))
        self.t_lr.append(float(lr))
        self.t_grad.append(self._flat([p.grad for p in self.params]))
        self.t_theta.append(self._flat(self.params))
        self.t_buf.append(self._buf)
        state = self.handler.optimizer.state
        if all("momentum_buffer" in state[p] and state[p]["momentum_buffer"] is not None for p in self.params):
            self._buf = self._flat([state[p]["momentum_buffer"] for p in self.params])
        super().on_step(step, loss, grad_norm, lr)

    def on_end(self, step):
        super().on_end(step)
        self.opt_state = copy.deepcopy(self.handler.optimizer.state_dict())


class _Run:
    pass


def _origin(classes=10):
    return _net(classes, widths=(6, 8), seed=3)


def _make_nap(train_ds, val_ds, out_dir, classes=10, cls=NapLogger, origin="default", **kwargs):
    recorder, lines = _Recorder(), []
    args = dict(recorder=recorder, printer=lines.append, out_dir=str(out_dir) if out_dir is not None else None,
                batch_size=16, momentum=0.9, weight_decay=5e-4, probe_n=24, calib_n=64)
    args.update(kwargs)
    nap = cls(_origin(classes) if origin == "default" else origin, val_ds, train_ds, classes, _device(), LABEL, NETWORK,
              **args)
    nap.test_recorder, nap.test_lines = recorder, lines
    return nap


def _run(nap_factory=None, n=160, batch=16, epochs=2, classes=10, every=5, model=None, n_val=64, hook_args=None,
         train_args=None):
    """Train ``epochs`` epochs of ``n`` samples; the model, the seeds and the loaders are the same on every call."""
    run = _Run()
    run.model = model if model is not None else _net(classes)
    run.train_ds, run.val_ds = _data(n, 1, classes), _data(n_val, 2, classes)
    train = DataLoader(run.train_ds, shuffle=True, batch_size=batch)
    val = DataLoader(run.val_ds, shuffle=False, batch_size=32)
    run.state0 = {k: v.detach().clone() for k, v in run.model.state_dict().items()}
    run.nap = nap_factory(run.train_ds, run.val_ds) if nap_factory is not None else None  # may seed torch: before the run's seed
    _seed_all()
    handler = ClassificationHandler(run.model, nn.CrossEntropyLoss())
    run.trace = _Spy(every, val, _device(), LABEL, NAME, NETWORK, handler=handler,
                     **(hook_args if hook_args is not None else {"nap": run.nap} if run.nap is not None else {}))
    handler.train_model(train, allow_reinit_retry=False, max_epochs=epochs, patience=epochs + 1, keep_last=True,
                        step_hook=run.trace, **(train_args or {}))
    run.state = {k: v.detach().clone() for k, v in run.model.state_dict().items()}
    run.rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    run.steps = len(run.trace.t_loss)
    return run


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _same_opt(a, b):
    if a["param_groups"] != b["param_groups"] or a["state"].keys() != b["state"].keys():
        return False
    for key in a["state"]:
        if a["state"][key].keys() != b["state"][key].keys():
            return False
        for name, value in a["state"][key].items():
            other = b["state"][key][name]
            if torch.is_tensor(value):
                if not torch.equal(value, other):
                    return False
            elif value != other:
                return False
    return True


def _same_rng(a, b):
    return (torch.equal(a[0], b[0]) and a[1] == b[1] and a[2][0] == b[2][0] and np.array_equal(a[2][1], b[2][1])
            and a[2][2:] == b[2][2:])


def _rng_now():
    return torch.get_rng_state(), random.getstate(), np.random.get_state()


def _nonfinite(obj, path=""):
    """Paths of every non-finite number in a record (the gate's own walk)."""
    if obj is None or isinstance(obj, (bool, str)):
        return []
    if isinstance(obj, (int, float)):
        return [] if math.isfinite(obj) else [path or "/"]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in _nonfinite(v, path + "/" + str(k))]
    if isinstance(obj, (list, tuple)):
        return [p for i, v in enumerate(obj) for p in _nonfinite(v, "%s[%d]" % (path, i))]
    return [path + ":" + type(obj).__name__]


def _expected_reads(last):
    return sorted(set([s for s in ft_nap.READ_STEPS if s <= last] + [last]))


def _expected_ckpts(last):
    return sorted(set([s for s in ft_nap.CKPT_STEPS if s <= last] + [last]))


def _by_step(nap):
    return {rec["step"]: rec for rec in nap.records}


def _load(nap):
    with np.load(nap.file_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["meta"] = json.loads(str(arrays["meta"][()]))
    return arrays


def _close(a, b, rtol=2e-3, atol=1e-9):  # records keep 6 significant digits: never ask them for more than 1e-5
    return np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), rtol=rtol, atol=atol)


# --- the flag and the hook's default ------------------------------------------------------------------------------------

def test_flag_parses(monkeypatch):
    assert fortify.eval_ft_nap() is False
    monkeypatch.setenv("SPECTRA_EVAL_FT_NAP", "1")
    assert fortify.eval_ft_nap() is True
    for value in ("0", "", "junk", "true", "2"):
        monkeypatch.setenv("SPECTRA_EVAL_FT_NAP", value)
        assert fortify.eval_ft_nap() is False


def test_trace_without_nap_is_the_current_hook():
    plain = _run()
    explicit = _run(hook_args={"nap": None})
    assert plain.trace.nap is None and explicit.trace.nap is None
    assert _same(plain.state, explicit.state) and _same_rng(plain.rng, explicit.rng)
    assert _same_opt(plain.trace.opt_state, explicit.trace.opt_state)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]  # noqa: E731
    assert strip(plain.trace.records) == strip(explicit.trace.records)
    assert [r["step"] for r in plain.trace.records] == [0, 5, 10, 15, 20]


# --- invariance -----------------------------------------------------------------------------------------------------------

def test_training_with_nap_is_bit_identical(tmp_path):
    plain = _run()
    logged = _run(lambda tr, va: _make_nap(tr, va, tmp_path))
    nap = logged.nap
    assert nap.failure is None and nap.done and len(nap.records) == len(_expected_reads(20))
    assert _same(plain.state, logged.state)
    assert _same_opt(plain.trace.opt_state, logged.trace.opt_state)
    assert _same_rng(plain.rng, logged.rng)
    assert plain.trace.t_loss == logged.trace.t_loss and plain.trace.t_gnorm == logged.trace.t_gnorm
    strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]  # noqa: E731
    assert strip(plain.trace.records) == strip(logged.trace.records)
    assert not _same(plain.state0, plain.state)  # the run did train


class _NoFlagRestore(NapLogger):
    """The same reads without putting the module flags back: what the logger must not do."""

    def _live_read(self, mode):
        return self._probe_read(self.model, mode, self.convs, self.head)


def test_the_test_has_teeth_without_the_flag_restore(tmp_path):
    plain = _run()
    broken = _run(lambda tr, va: _make_nap(tr, va, tmp_path, cls=_NoFlagRestore))
    assert broken.nap.failure is None and len(broken.nap.records) > 3
    assert not _same(plain.state, broken.state)  # a read that leaves eval() behind changes the BatchNorm updates


def test_nap_under_a_graphed_forward(monkeypatch, tmp_path):
    # torch.cuda.make_graphed_callables replaces model.forward with an instance attribute that replays the graph in the
    # captured train mode (Python hooks never run there) and calls the eager forward otherwise. The stand-in asserts
    # what the logger owes it: no hook at the capture or at a replay, and no deep copy of a model that carries it.
    hooked = lambda net: any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in net.modules())  # noqa: E731
    seen = []

    def make_graphed_callables(module, sample_args, **kwargs):
        assert not hooked(module), "a hook is registered at the capture"
        with torch.no_grad():
            for _ in range(3):
                module(sample_args[0])
        eager, state = module.forward, module.training

        def graphed(*args):
            if module.training == state:
                assert not hooked(module), "a hook is registered at a replay"
                seen.append("replay")
            else:
                seen.append("eager")
            return eager(*args)

        module.forward = graphed
        return module

    real_copy = copy.deepcopy

    def guarded_copy(obj, *args, **kwargs):
        if isinstance(obj, nn.Module):
            assert "forward" not in obj.__dict__, "a model carrying the graphed forward was deep-copied"
        return real_copy(obj, *args, **kwargs)

    monkeypatch.setattr(handler_mod, "cuda_graph_blocker", lambda *args, **kwargs: "")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "make_graphed_callables", make_graphed_callables)
    monkeypatch.setattr(ft_nap.copy, "deepcopy", guarded_copy)
    plain = _run(train_args={"cuda_graph": True})
    replays, eager = seen.count("replay"), seen.count("eager")
    assert replays == 20 and eager == 8  # 20 graphed steps; TRACE reads val (2 batches) in eval mode at steps 5, 10, 15, 20
    del seen[:]
    logged = _run(lambda tr, va: _make_nap(tr, va, tmp_path), train_args={"cuda_graph": True})
    nap = logged.nap
    assert nap.failure is None and nap.done and [r["step"] for r in nap.records] == _expected_reads(20)
    assert seen.count("replay") == replays and seen.count("eager") > eager  # the logger's reads ran the eager forward
    assert _same(plain.state, logged.state) and _same_rng(plain.rng, logged.rng)
    assert _same_opt(plain.trace.opt_state, logged.trace.opt_state)
    assert "forward" not in logged.model.__dict__ and not hooked(logged.model)


def test_flags_hooks_grads_and_origin_are_left_as_found(tmp_path):
    model = _net()
    model.train()
    model[1].eval()
    model[7].eval()
    train_ds, val_ds = _data(160, 1), _data(64, 2)
    origin = _origin()
    origin.train()
    origin_state = {k: v.clone() for k, v in origin.state_dict().items()}
    nap = _make_nap(train_ds, val_ds, tmp_path, origin=origin)
    before = [(m, m.training) for m in model.modules()]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = _rng_now()
    nap.on_start(model)
    assert nap.failure is None and nap.started
    assert [(m, m.training) for m in model.modules()] == before
    assert _same(state, model.state_dict()) and _same_rng(rng, _rng_now())
    assert all(p.grad is None for p in model.parameters())
    hooks = lambda net: any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in net.modules())  # noqa: E731
    assert not hooks(model) and not hooks(origin)
    assert _same(origin_state, origin.state_dict()) and origin.training and all(m.training for m in origin.modules())
    # a read after a step: the gradients the optimizer left are not written to, the flags come back
    g = torch.Generator().manual_seed(5)
    grads = [torch.randn(p.shape, generator=g) * 0.01 for p in model.parameters()]
    for p, value in zip(model.parameters(), grads):
        p.grad = value.clone()
    kept = [p.grad for p in model.parameters()]
    nap.on_step(1, torch.tensor(2.0), torch.tensor(0.5), 0.05)
    assert nap.failure is None and nap._pending["step"] == 1
    assert all(p.grad is k and torch.equal(k, value) for p, k, value in zip(model.parameters(), kept, grads))
    assert [(m, m.training) for m in model.modules()] == before
    assert _same(state, model.state_dict()) and _same_rng(rng, _rng_now()) and not hooks(model)


def test_a_logger_failure_never_stops_the_fine_tune(tmp_path):
    class _NoLabels(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, index):
            return torch.zeros(3, 8, 8), 0

    plain = _run()
    failed = _run(lambda tr, va: _make_nap(tr, _NoLabels(), tmp_path))
    nap = failed.nap
    assert nap.dead and not nap.started and "on_start" in nap.failure
    assert any("stopped at on_start" in line and "the fine-tune continues" in line for line in nap.test_lines)
    assert nap.test_recorder.issues and nap.test_recorder.issues[0][0] == "ft_nap_failed"
    assert not nap.test_recorder.events and nap.file_path is None
    assert _same(plain.state, failed.state) and _same_rng(plain.rng, failed.rng)
    assert _same_opt(plain.trace.opt_state, failed.trace.opt_state)


def test_shapes_are_kept_and_a_changed_shape_stops_nap_only(tmp_path):
    logged = _run(lambda tr, va: _make_nap(tr, va, tmp_path))
    assert logged.nap.failure is None
    assert [tuple(p.shape) for p in logged.model.parameters()] == [tuple(v) for v in logged.nap.shapes]
    model = _net()
    nap = _make_nap(_data(160, 1), _data(64, 2), tmp_path / "b")
    nap.on_start(model)
    assert nap.failure is None
    model[8].weight.data = torch.zeros(10, 5)
    nap.on_step(1, torch.tensor(2.0), torch.tensor(0.5), 0.05)
    assert nap.dead and "changed shape" in nap.failure
    assert any("changed shape" in line for line in nap.test_lines)
    assert nap.test_recorder.issues[0][0] == "ft_nap_failed"
    nap.on_step(2, torch.tensor(2.0), torch.tensor(0.5), 0.05)  # a dead logger ignores every later call
    nap.on_end(2)
    assert len(nap.test_recorder.of("eval_ft_nap")) == 0 and len(nap.test_recorder.of("eval_ft_nap0")) == 1


# --- records --------------------------------------------------------------------------------------------------------------

def _check_records(run, last, classes=10):
    nap = run.nap
    recorder = nap.test_recorder
    assert nap.failure is None and nap.done
    zero = recorder.of("eval_ft_nap0")
    assert len(zero) == 1
    assert set(ft_nap.NAP0_KEYS) <= set(zero[0]) and zero[0]["network"] == NETWORK and zero[0]["label"] == LABEL
    assert zero[0]["step"] == 0 and zero[0]["seconds"] > 0
    for key in ("H5", "Z3", "F7", "F2", "H1", "PDIV"):
        assert set(zero[0][key]) >= {"stale", "recal"}, key
    assert "origin" in zero[0]["H5"] and zero[0]["F7"]["form"] == "F7pre"
    assert "stale_vs_recal_ho_acc" in zero[0]["BN"] and zero[0]["Z4"]["state"] == "recal"
    events = recorder.of("eval_ft_nap")
    reads, ckpts = _expected_reads(last), _expected_ckpts(last)
    assert [e["step"] for e in events] == reads
    for event in events:
        step = event["step"]
        assert set(ft_nap.NAP_KEYS) <= set(event), step
        assert event["network"] == NETWORK and event["label"] == LABEL and event["seconds"] >= 0
        assert event["ckpt"] == (step in ckpts)
        for family in W_FAMILIES:
            assert bool(event[family]) == (step in ckpts), (family, step)
        assert bool(event["E"]) == (step == last) and bool(event["H"]) == (step == last)
        assert bool(event["X"]) == (step > 0)
        if step == 0:
            assert set(event["P"]) == {"stale", "recal"}
            assert event["P"]["stale"].keys() == event["P"]["recal"].keys()
            assert event["F0"]["g"] == {"missing": [["F0", "no training step before step 0"]]}
        else:
            assert 0.0 <= event["P"]["acc_ho"] <= 1.0 and 0.0 <= event["P"]["churn_ho"] <= 1.0
            assert event["P"]["js_prev_ho"] >= 0.0 and event["X"]["n"] == step
        for family in ft_nap.NAP_KEYS:
            for name, value in event[family].items():
                if name.endswith("_L") and value is not None:
                    assert len(value) == 16, (family, name)
    assert not [p for e in zero + events for p in _nonfinite(e)]
    # every None is named, with a reason, in the dict's own missing list
    def named(d):
        gone = {name for name, _why in d.get("missing", [])}
        for name, value in d.items():
            if value is None:
                assert name in gone, name
            elif isinstance(value, dict):
                named(value)
    for event in zero + events:
        named(event)
    # the log lines: the cut, one per read, the end
    lines = nap.test_lines
    assert sum(" cut | " in line for line in lines) == 1 and sum(" end step=" in line for line in lines) == 1
    assert [int(line.split("step=")[1].split(" ")[0]) for line in lines if " | probe acc ho " in line] == reads
    assert all(line.startswith("[eval] FT_NAP %s %s " % (LABEL, NAME)) for line in lines)
    assert " file " in lines[-1] and " MB" in lines[-1] and " mem held " in lines[-1] and " ms" in lines[-1]
    # the array file
    assert nap.file_path == os.path.join(str(nap.out_dir), "%s__%s.npz" % (LABEL, NAME)) and os.path.isfile(nap.file_path)
    assert 0 < os.path.getsize(nap.file_path) == nap.file_bytes < 2 * 1024 * 1024 < 20 * 1024 * 1024
    data = _load(nap)
    meta = data.pop("meta")
    for name, value in data.items():
        if value.dtype.kind == "f":
            assert np.isfinite(value).all(), name
    n_probe = nap.n_ho + nap.n_tr
    assert data["probe_logits"].shape == (len(reads), n_probe, classes) and data["probe_logits"].dtype == np.float16
    assert data["probe_steps"].tolist() == reads and data["ckpt_steps"].tolist() == ckpts
    assert data["probe_labels"].shape == (n_probe,) and data["probe_index_ho"].shape == (nap.n_ho,)
    assert data["step_series"].shape == (last, len(ft_nap.STEP_COLS)) and meta["step_cols"] == list(ft_nap.STEP_COLS)
    assert data["s5_steps"].tolist() == list(range(0, last + 1, 3)) and data["s5_proj"].shape == (last // 3 + 1, 4096)
    assert data["f0_w"].shape == (len(ckpts), 3, 96) and data["f0_g"].shape == (len(ckpts), 3, 96)
    assert data["f0_g__ok"][0].sum() == 0 and data["f0_g__ok"][1:].all() and "f0_w__ok" not in data
    assert data["layer_S1_dist"].shape == (len(ckpts), 3) and data["layer_Y2_cka_stale"].shape == (len(ckpts), 2)
    assert meta["network"] == NETWORK and meta["label"] == LABEL and meta["n_classes"] == classes and meta["steps"] == last
    assert meta["layers"] == ["0", "3", "8"] and meta["layer_shapes"][0] == [4, 3, 3, 3] and "mb_acc" in meta["step_missing"]
    return zero[0], events, data, meta


def test_records_schedule_keys_and_array_file(tmp_path):
    run = _run(lambda tr, va: _make_nap(tr, va, tmp_path))
    zero, events, data, meta = _check_records(run, 20)
    last = events[-1]
    assert set(last["E"]) >= {"ho", "tr", "reads", "E5_el2n_mean", "E7_weight_high"} and last["E"]["reads"] == len(events)
    assert 0.0 <= last["H"]["loo_acc"] <= 1.0 and "R7_loo_acc_stale" in last["H"]
    assert zero["BN"]["n_bn"] == 2 and zero["BN"]["calib_n"] == 64
    assert zero["Z4"]["batches"] == 1 and zero["Z4"]["gnorm"] > 0 and zero["Z4"]["snip"] > 0
    assert zero["Z3"]["jac"]["n_inputs"] == 16 and zero["Z3"]["jac"]["s1_mean"] > 0
    # the probe's own reads agree with the final model on the probe images
    probe = ft_nap.build_probe(run.val_ds, run.train_ds, None, 24, 64, use_cache=False)
    run.model.eval()
    with torch.no_grad():
        acc_ho = (run.model(probe["x_ho"]).argmax(1).numpy() == probe["y_ho"]).mean()
        acc_tr = (run.model(probe["x_cal"][:24]).argmax(1).numpy() == probe["y_cal"][:24]).mean()
    assert _close(last["P"]["acc_ho"], acc_ho, 1e-5) and _close(last["P"]["acc_tr"], acc_tr, 1e-5)
    with torch.no_grad():
        logits = torch.cat([run.model(probe["x_ho"]), run.model(probe["x_cal"][:24])]).numpy()
    assert np.allclose(data["probe_logits"][-1].astype(np.float64), logits, rtol=2e-3, atol=2e-3)
    assert data["probe_labels"].tolist() == probe["y_ho"].tolist() + probe["y_cal"][:24].tolist()
    assert data["probe_index_ho"].tolist() == probe["idx_ho"].tolist()


@pytest.mark.parametrize("n,batch,epochs,last", [(112, 16, 2, 14), (104, 8, 1, 13), (128, 16, 1, 8), (48, 16, 1, 3)])
def test_the_last_step_is_always_a_checkpoint(tmp_path, n, batch, epochs, last):
    # 14: after the last scheduled read; 13: a scheduled read that is no checkpoint; 8: a scheduled checkpoint; 3: early
    plain = _run(n=n, batch=batch, epochs=epochs)
    run = _run(lambda tr, va: _make_nap(tr, va, tmp_path), n=n, batch=batch, epochs=epochs)
    assert run.steps == last
    _check_records(run, last)
    assert _same(plain.state, run.state) and _same_rng(plain.rng, run.rng)
    assert _same_opt(plain.trace.opt_state, run.trace.opt_state)


def test_c100_variant(tmp_path):
    plain = _run(n=200, classes=100, n_val=100)
    run = _run(lambda tr, va: _make_nap(tr, va, tmp_path, classes=100), n=200, classes=100, n_val=100)
    assert run.steps == 26
    zero, events, data, meta = _check_records(run, 26, classes=100)
    assert _same(plain.state, run.state) and _same_rng(plain.rng, run.rng)
    assert zero["Z3"]["jac"]["tenth"] == 10 and zero["Z3"]["jac"]["rank10_inputs"] == 0 and zero["H5"]["stale"]["n"] == 48
    assert events[-1]["P"]["cls_worst_ho"] is not None


def test_a_model_without_convolutions_or_batchnorm_is_logged(tmp_path):
    def net():
        torch.manual_seed(0)
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))

    plain = _run(model=net())
    run = _run(lambda tr, va: _make_nap(tr, va, tmp_path, origin=net()), model=net())
    nap = run.nap
    assert nap.failure is None and nap.done and [r["step"] for r in nap.records] == _expected_reads(20)
    assert _same(plain.state, run.state) and _same_rng(plain.rng, run.rng)
    zero = nap.test_recorder.of("eval_ft_nap0")[0]
    assert zero["BN"]["n_bn"] == 0 and ["shift", "no BatchNorm with running statistics: recal equals stale"] in zero["BN"]["missing"]
    assert zero["H5"]["stale"]["loo_acc"] is not None and zero["F2"]["stale"]["erank_pen"] is not None
    assert not [p for e in [zero] + nap.records for p in _nonfinite(e)]
    assert ["V7", "no BatchNorm"] in nap.records[-1]["V"]["missing"]


# --- the numbers against direct computations -----------------------------------------------------------------------------

def test_series_and_weight_features_match_the_raw_steps(tmp_path):
    run = _run(lambda tr, va: _make_nap(tr, va, tmp_path))
    spy, nap = run.trace, run.nap
    events, data = _by_step(nap), _load(nap)
    steps = run.steps
    theta = [spy.theta0] + spy.t_theta          # theta[s]: the parameters after step s
    grad = torch.stack(spy.t_grad)              # clipped, as the optimizer used it
    gnorm = torch.tensor(spy.t_gnorm, dtype=torch.float64)
    scale = torch.clamp(gnorm + 1e-6, min=1.0)
    pre = grad * scale[:, None]                 # the pre-clip gradient
    upd = torch.stack([theta[s] - theta[s - 1] for s in range(1, steps + 1)])
    series = data["step_series"].astype(np.float64)
    assert _close(series[:, 0], spy.t_loss, 1e-6) and _close(series[:, 1], spy.t_lr, 1e-6)
    assert _close(series[:, 2], spy.t_gnorm, 1e-6) and _close(series[:, 3], upd.norm(dim=1), 1e-4)
    assert _close(series[:, 4], scale, 1e-6)
    assert _close(pre.norm(dim=1), gnorm, 1e-4)  # the recovered gradient has the pre-clip norm
    sizes = [int(p.numel()) for p in run.model.parameters()]
    matrix = torch.cat([torch.full((s,), p.dim() >= 2) for s, p in zip(sizes, run.model.parameters())])
    layer_slices, offset = [], 0
    for size, p in zip(sizes, run.model.parameters()):
        if p.dim() >= 2:
            layer_slices.append(slice(offset, offset + size))
        offset += size
    ckpts = _expected_ckpts(steps)
    for k, step in enumerate(ckpts):
        event = events[step]
        now, first = theta[step], theta[0]
        dist = float((now - first).norm())
        assert _close(event["S"]["S1_dist"], dist, 1e-4, 1e-12)
        assert _close(event["R"]["R3_all"], dist / float(first.norm()), 1e-4, 1e-12)
        assert _close(event["S"]["S3_path"], float(upd[:step].norm(dim=1).sum()), 1e-4, 1e-12)
        assert _close(event["S"]["S7_sign0"], float((torch.sign(now) != torch.sign(first)).double().mean()), 1e-5)
        assert _close(data["layer_S1_dist"][k], [float((now[s] - first[s]).norm()) for s in layer_slices], 1e-3, 1e-9)
        if step > 0:
            rot = [1.0 - float(torch.dot(now[s], first[s]) / (now[s].norm() * first[s].norm())) for s in layer_slices]
            assert _close(data["layer_S2_rot"][k], rot, 2e-2, 1e-9)
            assert _close(event["S"]["S3_direct"], dist / float(upd[:step].norm(dim=1).sum()), 1e-4)
        if k >= 1:
            prev = ckpts[k - 1]
            window = slice(prev, step)          # rows of the steps prev+1 .. step
            n = step - prev
            assert event["K"]["n"] == n and event["J"]["n"] == n and event["V"]["n"] == n
            g = pre[window]
            assert _close(event["J"]["J4emp"], float((g ** 2).sum(dim=1).mean()), 1e-3)
            assert _close(event["V"]["V1_all"], float(upd[window].norm(dim=1).mean() / now.norm()), 1e-3)
            cos = [float(torch.dot(spy.t_buf[t], grad[t]) / (spy.t_buf[t].norm() * grad[t].norm()))
                   for t in range(prev, step) if spy.t_buf[t] is not None]
            assert _close(event["V"]["V2_all"], np.mean(cos), 5e-3, 1e-4)
            mask_now = now.float().abs() > now.float().abs().median()
            before = theta[prev].float().abs()
            assert _close(event["S"]["S8_mask"], float((mask_now != (before > before.median())).double().mean()), 1e-5)
            if n >= 2:
                mean = g.mean(dim=0)
                spread = (n / (n - 1.0)) * float((g ** 2).sum(dim=1).mean() - (mean ** 2).sum())
                assert _close(event["K"]["K1"], float((mean ** 2).sum()) / (spread + 1e-12), 2e-3)
                unit = g / g.norm(dim=1, keepdim=True)
                pairs = torch.triu_indices(n, n, 1)
                cosines = (unit @ unit.t())[pairs[0], pairs[1]]
                assert _close(event["K"]["K3mb_cbar"], float(cosines.mean()), 2e-3, 1e-4)  # exact: no bucket is shared
                assert _close(event["K"]["K4_cos_min"], float(cosines.min()), 2e-3, 1e-4)
                assert _close(event["K"]["K7_disp"], float(torch.pdist(g).mean()), 2e-3)
                assert _close(event["K"]["K4_div"], float((g ** 2).sum() / (g.sum(dim=0) ** 2).sum()), 2e-3)
                assert _close(event["K"]["K8_trace"], float(((g - mean) ** 2).sum() / n), 2e-3)
        if k >= 2:
            d1, d0 = now - theta[ckpts[k - 1]], theta[ckpts[k - 1]] - theta[ckpts[k - 2]]
            for tag, sel in (("w", matrix), ("o", ~matrix)):
                if float(d0[sel].norm()) > 0:
                    cosine = float(torch.dot(d1[sel], d0[sel]) / (d1[sel].norm() * d0[sel].norm()))
                    assert _close(event["S"]["S4_cos_" + tag], cosine, 5e-3, 1e-4)
                    same = float((torch.sign(d1[sel]) == torch.sign(d0[sel])).double().mean())
                    assert _close(event["S"]["S4_same_" + tag], same, 1e-5)
    assert events[0]["S"]["S1_dist"] == 0.0 and events[0]["R"]["R3_all"] == 0.0 and events[0]["S"]["S7_sign0"] == 0.0
    assert all(abs(v) < 1e-9 for v in events[0]["Y"]["Y2_stale_L"][:12]) and events[steps]["Y"]["Y2_stale_L"][8] > 0
    last = events[steps]
    assert _close(last["X"]["loss_all"], np.mean(spy.t_loss), 1e-5)
    assert _close(last["X"]["loss_last"], np.mean(spy.t_loss[13:]), 1e-5) and last["X"]["lr"] == pytest.approx(spy.t_lr[-1])
    ema = spy.t_loss[0]
    for value in spy.t_loss[1:]:
        ema = 0.9 * ema + 0.1 * value
    assert _close(last["X"]["ema09"], ema, 1e-5)
    # the head's weight-decay balance and the per-layer gradient series
    cols = [i for i, p in enumerate(run.model.parameters()) if p.dim() >= 2]
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    per_layer = torch.stack([pre[:, bounds[i]:bounds[i + 1]].norm(dim=1) for i in cols], dim=1)
    assert _close(data["step_layer_gnorm"], per_layer, 1e-4, 1e-9)
    weight_norm = np.asarray([float(theta[steps][bounds[i]:bounds[i + 1]].norm()) for i in cols])
    mean_g = per_layer[ckpts[-2]:steps].mean(dim=0).numpy()
    assert _close(data["layer_V8_wd_ratio"][-1], 5e-4 * weight_norm / mean_g, 1e-3)
    assert _close(data["layer_V5_grad_ratio"][-1], mean_g / weight_norm, 1e-3)


def test_preclip_gradient_is_recovered(tmp_path):
    class _Keep(NapLogger):
        def _step(self, step, loss, grad_norm, lr):
            super()._step(step, loss, grad_norm, lr)
            if step == 1:
                self.seen = (self.S.main.detach().double().clone(), float(grad_norm))

    def loud():
        net = _net()
        with torch.no_grad():
            net[8].weight.mul_(30.0)  # a large loss gradient: the clip fires on the first step
        return net

    run = _run(lambda tr, va: _make_nap(tr, va, tmp_path, cls=_Keep), model=loud())
    recovered, norm = run.nap.seen
    assert norm > 1.5
    # the same first batch, the same starting point, no clipping
    reference = loud()
    reference.train()
    _seed_all()
    x, y = next(iter(DataLoader(run.train_ds, shuffle=True, batch_size=16)))
    nn.CrossEntropyLoss()(reference(x), y.long()).backward()
    unclipped = torch.cat([p.grad.reshape(-1) for p in reference.parameters()]).double()
    assert float(unclipped.norm()) == pytest.approx(norm, rel=1e-5)
    assert torch.allclose(recovered, unclipped, rtol=1e-4, atol=1e-7)
    clipped = run.trace.t_grad[0]
    assert float(clipped.norm()) == pytest.approx(1.0, rel=1e-4) and not torch.allclose(clipped, unclipped, rtol=1e-2)


def _drive(nap, model, grads, lr=0.05):
    """Feed ``grads`` (one list of per-parameter tensors per step) through on_step, the parameters left alone."""
    for step, per_param in enumerate(grads, start=1):
        for p, value in zip(model.parameters(), per_param):
            p.grad = value
        total = math.sqrt(sum(float((v.double() ** 2).sum()) for v in per_param))
        assert total < 1.0  # no clipping: the hook's gradient is the pre-clip one
        nap.on_step(step, torch.tensor(1.0), torch.tensor(total), lr)
        assert nap.failure is None


def test_k1_is_large_for_identical_gradients_and_small_for_noise(tmp_path):
    g = torch.Generator().manual_seed(11)
    model = _net()
    shapes = [p.shape for p in model.parameters()]
    one = [torch.randn(s, generator=g) * 0.02 for s in shapes]
    nap = _make_nap(_data(160, 1), _data(64, 2), tmp_path / "same")
    nap.on_start(model)
    _drive(nap, model, [one] * 20)
    same = nap._pending
    assert same["step"] == 20 and same["K"]["n"] == 12
    assert same["K"]["K1"] > 1e6 and min(same["K"]["K1_L"][:8]) > 1e3
    assert same["K"]["K3mb_alpha"] == pytest.approx(1.0, rel=1e-4) and same["K"]["K3mb_cbar"] == pytest.approx(1.0, rel=1e-4)
    assert same["K"]["K4_div"] == pytest.approx(1.0 / 12, rel=1e-4) and same["K"]["K7_disp"] < 1e-5

    model = _net()
    noise = [[torch.randn(s, generator=g) * 0.02 for s in shapes] for _ in range(20)]
    nap = _make_nap(_data(160, 1), _data(64, 2), tmp_path / "noise")
    nap.on_start(model)
    _drive(nap, model, noise)
    rec = nap._pending
    raw = np.stack([torch.cat([v.reshape(-1) for v in step]).double().numpy() for step in noise[8:20]])  # steps 9..20
    n = raw.shape[0]
    mean = raw.mean(axis=0)
    spread = n / (n - 1.0) * ((raw ** 2).sum(axis=1).mean() - (mean ** 2).sum())
    expected = (mean ** 2).sum() / (spread + 1e-12)
    assert rec["K"]["K1"] == pytest.approx(expected, rel=1e-3) and rec["K"]["K1"] < 0.3
    assert float(ft_nap.k1_snr((raw.sum(axis=0) ** 2).sum(), (raw ** 2).sum(), n)) == pytest.approx(expected, rel=1e-9)
    assert abs(rec["K"]["K3mb_cbar"]) < 0.2 and rec["K"]["K3mb_alpha"] == pytest.approx(1.0 / 12, rel=0.5)
    assert rec["J"]["J4emp"] == pytest.approx((raw ** 2).sum(axis=1).mean(), rel=1e-4)
    # per layer: the first convolution's weight
    first = np.stack([step[0].reshape(-1).double().numpy() for step in noise[8:20]])
    mean = first.mean(axis=0)
    layer = (mean ** 2).sum() / (n / (n - 1.0) * ((first ** 2).sum(axis=1).mean() - (mean ** 2).sum()) + 1e-12)
    assert nap.ck_arrays[-1]["K1_snr"][0] == pytest.approx(layer, rel=1e-3)


def test_family0_matches_the_reference_statistics(tmp_path):
    def reference(weight):  # scripts/selection_probe.py::nap_filter_stats, the per-filter port of nap2/stats.py
        w = weight.detach().double().reshape(weight.shape[0], -1)
        n = w.shape[1]
        mean = w.mean(dim=1)
        centred = w - mean[:, None]
        m2 = centred.pow(2).mean(dim=1)
        flat = m2 <= 0
        safe = torch.where(flat, torch.ones_like(m2), m2)
        skew = torch.where(flat, torch.zeros_like(m2), centred.pow(3).mean(dim=1) / safe.pow(1.5))
        kurt = torch.where(flat, torch.zeros_like(m2), centred.pow(4).mean(dim=1) / safe.pow(2) - 3.0)
        q = torch.quantile(w, torch.tensor([0.0, 0.25, 0.75, 0.5, 1.0], dtype=w.dtype), dim=1)
        cov = centred.pow(2).sum(dim=1) / max(n - 1, 1)
        return torch.stack([mean, m2, q[3], m2.sqrt(), w.max(dim=1).values, w.min(dim=1).values, cov, skew, kurt,
                            q[0], q[1], q[2], q[3], q[4], w.abs().sum(dim=1), w.norm(dim=1)], dim=1).numpy()

    # ragged rows through the padded form
    g = torch.Generator().manual_seed(3)
    rows = [torch.randn(n, generator=g, dtype=torch.float64) for n in (27, 36, 36, 5, 1, 64)]
    rows.append(torch.full((9,), 0.25, dtype=torch.float64))  # a constant row: zero skewness and kurtosis
    width = max(r.numel() for r in rows)
    padded = torch.stack([torch.nn.functional.pad(r, (0, width - r.numel())) for r in rows])
    stats, sums = ft_nap.row_stats(padded, torch.tensor([r.numel() for r in rows]))
    for i, row in enumerate(rows):
        assert np.allclose(stats[i].numpy(), reference(row[None])[0], rtol=1e-9, atol=1e-12), i
        assert np.allclose(sums[i].numpy(), [float(row.sum()), float((row ** 2).sum()), float((row ** 3).sum()),
                                             float((row ** 4).sum())], rtol=1e-9, atol=1e-12)
    assert stats[-1, 7] == 0 and stats[-1, 8] == 0
    # end to end: the last checkpoint's arrays against the final weights and the last step's pre-clip gradient
    run = _run(lambda tr, va: _make_nap(tr, va, tmp_path))
    data = _load(run.nap)
    weights = [p for p in run.model.parameters() if p.dim() >= 2]
    sizes = [int(p.numel()) for p in run.model.parameters()]
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    cols = [i for i, p in enumerate(run.model.parameters()) if p.dim() >= 2]
    scale = max(1.0, run.trace.t_gnorm[-1] + 1e-6)
    for layer, (weight, col) in enumerate(zip(weights, cols)):
        grad = (run.trace.t_grad[-1][bounds[col]:bounds[col + 1]] * scale).reshape(weight.shape)
        for name, tensor, rtol in (("f0_w", weight, 1e-5), ("f0_g", grad, 1e-4)):
            got = data[name][-1, layer].astype(np.float64)
            whole = reference(tensor.reshape(1, -1))[0]
            per_filter = reference(tensor)
            assert np.allclose(got[:16], whole, rtol=rtol, atol=1e-7), (name, layer)
            assert np.allclose(got[16:32], per_filter.mean(axis=0), rtol=rtol, atol=1e-7), (name, layer)
            assert np.allclose(got[32:48], per_filter.std(axis=0), rtol=rtol, atol=1e-7), (name, layer)
            assert np.allclose(got[48:], np.percentile(per_filter, [10, 50, 90], axis=0).reshape(-1), rtol=rtol,
                               atol=1e-7), (name, layer)
    event = _by_step(run.nap)[20]
    form, _bad = ft_nap.fixed_form(data["f0_w"][-1, :, 15])
    assert _close(event["F0"]["w"]["l2_L"], form, 1e-5)


def test_y4_matches_a_direct_svd(tmp_path):
    run = _run(lambda tr, va: _make_nap(tr, va, tmp_path))
    data, nap = _load(run.nap), run.nap
    assert data["meta"]["y4"] == {"max_dim": 256, "by": "origin", "layers": [0, 1, 2]}
    names = ["0.weight", "3.weight", "8.weight"]  # the last is tall: 10 outputs from 6 inputs
    for layer, name in enumerate(names):
        w0 = run.state0[name].double().reshape(run.state0[name].shape[0], -1)
        wt = run.state[name].double().reshape(w0.shape)
        sigma = torch.linalg.svdvals(wt - w0)
        q = sigma / sigma.sum()
        erank = math.exp(float(-(q * q.log()).sum()))
        assert _close(data["layer_Y4_erank"][-1, layer], erank, 1e-3), name
        assert _close(data["layer_Y4_top"][-1, layer], float(sigma[0] / torch.linalg.svdvals(w0)[0]), 1e-3), name
        u0, s0, _ = torch.linalg.svd(w0, full_matrices=False)
        ut, st, _ = torch.linalg.svd(wt, full_matrices=False)
        lead = min(10, int((st > 1e-10 * st[0]).sum()))
        cosine = (u0[:, s0 > 1e-10 * s0[0]].t() @ ut[:, :lead]).abs()
        assert data["layer_Y4_intruders"][-1, layer] == float((cosine.max(dim=0).values < 0.5).sum()), name
    assert data["layer_Y4_top"][0].tolist() == [0.0, 0.0, 0.0] and data["layer_Y4_intruders"][0].tolist() == [0.0, 0.0, 0.0]
    assert data["layer_Y4_erank__ok"][0].sum() == 0 and data["layer_Y4_erank__ok"][1:].all()
    # an intruder is found when a new leading direction is planted
    w0 = torch.linalg.qr(torch.randn(6, 6, generator=torch.Generator().manual_seed(1), dtype=torch.float64))[0][:, :3]
    base = w0 @ torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)) @ w0.t()
    mix = (w0[:, 0] + w0[:, 1] + w0[:, 2]) / math.sqrt(3.0)  # cosine 0.577 with each: not an intruder at 0.5
    fresh = torch.linalg.qr(torch.cat([w0, torch.ones(6, 1, dtype=torch.float64)], dim=1))[0][:, 3]
    nap.y4_layers = [(0, 0, 6, 6, False)]
    nap.Lw = 1
    nap.y4_ref = {0: (np.asarray([3.0, 2.0, 1.0]), w0.numpy())}
    for extra, expected in ((10.0 * torch.outer(fresh, fresh), 1.0), (10.0 * torch.outer(mix, mix), 0.0)):
        moved = base + extra
        flat = torch.cat([(moved @ moved.t()).reshape(-1), (extra @ extra.t()).reshape(-1)]).numpy()
        _erank, _top, intruders = nap._y4_host(flat, True)
        assert intruders[0] == expected


def test_cut_features_match_direct_computations(tmp_path):
    def dulled():
        net = _net()
        with torch.no_grad():
            net[0].weight[0].zero_()  # one silent channel in the first convolution
            net[0].bias[0].zero_()
        return net

    model = dulled()
    train_ds, val_ds = _data(160, 1), _data(64, 2)
    nap = _make_nap(train_ds, val_ds, tmp_path)
    nap.on_start(model)
    assert nap.failure is None
    zero = nap.test_recorder.of("eval_ft_nap0")[0]
    probe = ft_nap.build_probe(val_ds, train_ds, None, 24, 64, use_cache=False)
    x_ho, x_tr, y_tr = probe["x_ho"], probe["x_cal"][:24], torch.as_tensor(probe["y_cal"][:24])
    work = copy.deepcopy(model).eval()
    outs = {}
    handles = [work[i].register_forward_hook(lambda m, a, out, i=i: outs.setdefault(i, []).append(out.detach().double()))
               for i in (0, 3)]
    handles.append(work[8].register_forward_pre_hook(lambda m, a: outs.setdefault(8, []).append(a[0].detach().double())))
    with torch.no_grad():
        logits = torch.cat([work(x_ho), work(x_tr)])
    for handle in handles:
        handle.remove()
    arrays = nap.cut_arrays
    for layer, index in enumerate((0, 3)):
        ho, tr = outs[index]
        both = torch.cat([ho, tr])
        assert _close(arrays["stale_q"][layer], float((both ** 2).mean()), 1e-5)
        pairs = [torch.nn.functional.cosine_similarity(a[:12].flatten(1), a[12:24].flatten(1)).sum() for a in (ho, tr)]
        assert _close(arrays["stale_c"][layer], float(sum(pairs) / 24), 1e-5)
        level = tr.abs().mean(dim=(0, 2, 3))
        assert _close(arrays["stale_d01"][layer], float((level <= 0.01 * level.mean()).double().mean()), 1e-9)
    assert arrays["stale_d01"].tolist() == [0.25, 0.0] and arrays["stale_dead"].tolist() == [0.25, 0.0]
    assert zero["F7"]["stale"]["dormant_all"] == pytest.approx(0.1) and zero["F7"]["stale"]["dead_all"] == pytest.approx(0.1)
    reps = [(outs[0][0].mean(dim=(2, 3)), outs[0][1].mean(dim=(2, 3))), (outs[3][0].mean(dim=(2, 3)), outs[3][1].mean(dim=(2, 3))),
            (outs[8][0], outs[8][1])]
    onehot = torch.nn.functional.one_hot(y_tr, 10).double()
    centre = torch.eye(24, dtype=torch.float64) - 1.0 / 24
    label_kernel = centre @ onehot @ onehot.t() @ centre
    for index, (ho, tr) in enumerate(reps):
        x = torch.cat([ho, tr])
        sigma = torch.linalg.svdvals(x - x.mean(dim=0, keepdim=True))
        q = sigma[sigma > 0] / sigma.sum()
        assert _close(arrays["stale_erank"][index], math.exp(float(-(q * q.log()).sum())), 1e-6)
        lam = sigma ** 2
        assert _close(arrays["stale_pr"][index], float(lam.sum() ** 2 / (lam ** 2).sum()), 1e-6)
        assert arrays["stale_srank"][index] == int(((sigma / sigma.sum()).cumsum(0) < 0.99 - 1e-12).sum()) + 1
        kernel = centre @ tr @ tr.t() @ centre
        assert _close(arrays["stale_align"][index], float((kernel * label_kernel).sum() / (kernel.norm() * label_kernel.norm())), 1e-6)
    assert zero["F2"]["stale"]["erank_pen"] == pytest.approx(arrays["stale_erank"][2], rel=1e-5)
    assert zero["H1"]["stale"]["align_pen"] == pytest.approx(arrays["stale_align"][2], rel=1e-5)
    # P at step 0 (stale) and H5 on the stale penultimate features
    side = ft_nap._Side(logits[:24].numpy(), probe["y_ho"])
    first = nap._pending
    assert first["step"] == 0 and first["P"]["stale"]["acc_ho"] == pytest.approx(side.correct.mean(), rel=1e-5)
    assert first["P"]["stale"]["loss_ho"] == pytest.approx(side.ce.mean(), rel=1e-5)
    labels = torch.as_tensor(np.concatenate([probe["y_ho"], probe["y_cal"][:24]]))
    direct = ft_nap.h5_forecast(torch.cat([outs[8][0], outs[8][1]]), labels, 10, 24)
    assert zero["H5"]["stale"]["loo_acc"] == pytest.approx(direct["loo_acc"], rel=1e-5) and zero["H5"]["stale"]["n"] == 48
    assert zero["H5"]["stale"]["loo_mse"] == pytest.approx(direct["loo_mse"], rel=1e-4)
    # the recalibrated copy: cumulative BatchNorm statistics of the 64 calibration images, train-mode BatchNorm only
    recal = copy.deepcopy(model).eval()
    for m in recal.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
            m.train()
    with torch.no_grad():
        recal(probe["x_cal"])
    recal.eval()
    with torch.no_grad():
        acc = (recal(x_ho).argmax(1).numpy() == probe["y_ho"]).mean()
    assert zero["BN"]["recal_acc_ho"] == pytest.approx(acc, rel=1e-5) and first["P"]["recal"]["acc_ho"] == pytest.approx(acc, rel=1e-5)
    assert zero["BN"]["stale_vs_recal_ho_acc"] == pytest.approx(side.correct.mean() - acc, rel=1e-5, abs=1e-9)
    assert zero["Z3"]["jac"]["s1_over_s10"] is None  # 6 penultimate units: the Jacobian's rank is below 10
    assert ["s1_over_s10", "the Jacobian's rank is below 10 at every input"] in zero["Z3"]["jac"]["missing"]
    assert zero["PDIV"]["stale"]["js_ho"] > 0 and 0 <= zero["PDIV"]["stale"]["agree_ho"] <= 1
    assert zero["PDIV"]["origin_acc_ho"] is not None and zero["H5"]["origin"]["loo_acc"] is not None


def test_cut_gradients_match_direct_computations(tmp_path):
    model = _net(bn=False, widths=(4, 12))  # without BatchNorm the recalibrated copy is the model itself
    train_ds, val_ds = _data(160, 1), _data(64, 2)
    nap = _make_nap(train_ds, val_ds, tmp_path, origin=_net(bn=False, widths=(6, 16), seed=3), calib_n=100)
    nap.on_start(model)
    assert nap.failure is None
    zero = nap.test_recorder.of("eval_ft_nap0")[0]
    z4 = zero["Z4"]
    assert "missing" not in z4 and z4["batches"] == 2 and z4["batch"] == 64
    probe = ft_nap.build_probe(val_ds, train_ds, None, 24, 100, use_cache=False)
    pick = np.random.default_rng(ft_nap.seed_of(probe["dataset"], NAME, "Z4")).permutation(100)
    batches = [np.sort(pick[:64]), np.sort(pick[64:])]
    work = copy.deepcopy(model).double().eval()
    weights = [work[0].weight, work[2].weight, work[6].weight]
    params = list(work.parameters())
    gnorm, layersum, snip, grasp, fisher, profile = [], [], [], [], [], []

    def gradient(x, y):
        work.zero_grad()
        nn.functional.cross_entropy(work(x), y).backward()
        return [p.grad.clone() for p in params]

    for idx in batches:
        x, y = probe["x_cal"][idx].double(), torch.as_tensor(probe["y_cal"][idx])
        acts = []
        handles = [m.register_forward_hook(lambda m, a, out: (out.retain_grad(), acts.append(out))[0]) for m in (work[0], work[2])]
        work.zero_grad()
        nn.functional.cross_entropy(work(x), y).backward()
        for handle in handles:
            handle.remove()
        g = [p.grad.clone() for p in params]
        gnorm.append(math.sqrt(sum(float((v ** 2).sum()) for v in g)))
        wg = [work[0].weight.grad, work[2].weight.grad, work[6].weight.grad]
        layersum.append(sum(float(v.norm()) for v in wg))
        snip.append(sum(float((w * v).abs().sum()) for w, v in zip(weights, wg)))
        fisher.append([float((0.5 * ((a * a.grad).flatten(2).sum(dim=2) ** 2).mean(dim=0)).sum()) for a in acts])
        profile.append([float(a.grad.norm()) for a in acts])
        # H g by central differences along g
        eps = 1e-5
        with torch.no_grad():
            for p, v in zip(params, g):
                p.add_(eps * v)
        up = gradient(x, y)
        with torch.no_grad():
            for p, v in zip(params, g):
                p.sub_(2 * eps * v)
        down = gradient(x, y)
        with torch.no_grad():
            for p, v in zip(params, g):
                p.add_(eps * v)
        hg = {id(p): (a - b) / (2 * eps) for p, a, b in zip(params, up, down)}
        grasp.append(-sum(float((w * hg[id(w)]).sum()) for w in weights))
    assert z4["gnorm"] == pytest.approx(np.mean(gnorm), rel=1e-4) and z4["gnorm_sd"] == pytest.approx(np.std(gnorm), rel=1e-2)
    assert z4["gnorm_layersum"] == pytest.approx(np.mean(layersum), rel=1e-4)
    assert z4["snip"] == pytest.approx(np.mean(snip), rel=1e-4)
    assert z4["grasp"] == pytest.approx(np.mean(grasp), rel=2e-2, abs=1e-6)
    assert z4["fisher_ch"] == pytest.approx(np.sum(np.mean(fisher, axis=0)), rel=1e-3)
    assert _close(nap.cut_arrays["z4_fisher"], np.mean(fisher, axis=0), 1e-3)
    weights_per_channel = np.asarray([27.0, 36.0])
    assert z4["fisher_param"] == pytest.approx(float((np.mean(fisher, axis=0) * weights_per_channel).sum()), rel=1e-3)
    assert _close(nap.cut_arrays["z33_gnorm"], np.mean(profile, axis=0), 1e-3)
    assert zero["Z3"]["grad"]["g_ratio"] == pytest.approx(np.mean(profile, axis=0)[0] / np.mean(profile, axis=0)[1], rel=1e-3)
    # SynFlow: absolute weights, an all-ones input, the sum of the outputs
    flow = copy.deepcopy(model).double().eval()
    with torch.no_grad():
        for p in flow.parameters():
            p.abs_()
    flow.zero_grad()
    total = flow(torch.ones(1, 3, 8, 8, dtype=torch.float64)).sum()
    total.backward()
    score = sum(float((m.weight * m.weight.grad).sum()) for m in (flow[0], flow[2], flow[6]))
    assert z4["synflow_log"] == pytest.approx(math.log(score), rel=1e-6) and z4["synflow_logR"] == pytest.approx(math.log(float(total)), rel=1e-6)
    # Z3.4: the exact Jacobian of 16 probe inputs
    net = copy.deepcopy(model).double().eval()
    xs = torch.cat([probe["x_ho"][:8], probe["x_cal"][:8]]).double()
    top, ratio, msv = [], [], []
    for x in xs:
        jac = torch.autograd.functional.jacobian(lambda v: net(v[None])[0], x).reshape(10, -1)
        sigma = torch.linalg.svdvals(jac)
        top.append(float(sigma[0]))
        ratio.append(float(sigma[0] / sigma[9]))
        msv.append(float((sigma ** 2).sum() / 10))
    jac = zero["Z3"]["jac"]
    assert jac["s1_mean"] == pytest.approx(np.mean(top), rel=1e-4) and jac["msv"] == pytest.approx(np.mean(msv), rel=1e-4)
    assert jac["s1_over_s10"] == pytest.approx(np.mean(ratio), rel=1e-2) and jac["rank10_inputs"] == 16
    assert all(p.grad is None for p in model.parameters())  # the live model's gradients were never written


def test_first_order_proxies_survive_a_network_without_double_backward(monkeypatch, tmp_path):
    real = torch.autograd.grad

    def first_order_only(outputs, inputs, *args, **kwargs):
        if kwargs.get("create_graph"):
            raise RuntimeError("the derivative for this operation is not implemented")
        return real(outputs, inputs, *args, **kwargs)

    plain = _make_nap(_data(160, 1), _data(64, 2), tmp_path / "a")
    plain.on_start(_net())
    monkeypatch.setattr(torch.autograd, "grad", first_order_only)
    model = _net()
    nap = _make_nap(_data(160, 1), _data(64, 2), tmp_path / "b")
    nap.on_start(model)
    assert nap.failure is None and plain.failure is None
    zero, full = nap.test_recorder.of("eval_ft_nap0")[0], plain.test_recorder.of("eval_ft_nap0")[0]
    z4 = zero["Z4"]
    assert "grasp" not in z4 and z4["missing"] == [["grasp", "RuntimeError: the derivative for this operation is not implemented"]]
    assert full["Z4"]["grasp"] is not None and "missing" not in full["Z4"]
    for key in ("gnorm", "gnorm_layersum", "snip", "fisher_ch", "fisher_param", "synflow_log", "snip_L", "fisher_L"):
        assert z4[key] == full["Z4"][key], key  # the first-order proxies are the same numbers
    assert zero["Z3"]["grad"] == full["Z3"]["grad"] and zero["Z3"]["jac"] == full["Z3"]["jac"]
    assert all(p.grad is None for p in model.parameters()) and not _nonfinite(zero)


# --- H5, the probe, the small estimators ---------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [7, 60])
def test_h5_closed_form_loo_equals_refitting(d):
    g = torch.Generator().manual_seed(4)
    n, classes = 40, 5
    labels = torch.arange(n) % classes
    feats = torch.randn(n, d, generator=g, dtype=torch.float64) + 0.5 * torch.randn(classes, d, generator=g, dtype=torch.float64)[labels]
    out, loo = ft_nap.h5_forecast(feats, labels, classes, n_ho=20, return_loo=True)
    z = (feats - feats.mean(dim=0)) / feats.std(dim=0, unbiased=False)
    kernel = z @ z.t()
    lam = 1e-3 * float(kernel.diagonal().sum()) / n
    onehot = torch.nn.functional.one_hot(labels, classes).double()
    ybar = onehot.mean(dim=0)
    refit = torch.zeros_like(onehot)
    for i in range(n):
        keep = torch.arange(n) != i
        alpha = torch.linalg.solve(kernel[keep][:, keep] + lam * torch.eye(n - 1, dtype=torch.float64), onehot[keep] - ybar)
        refit[i] = kernel[i, keep] @ alpha + ybar
    assert float((loo - refit).abs().max()) < 1e-6
    assert out["loo_acc"] == pytest.approx(float((refit.argmax(dim=1) == labels).double().mean()))
    assert out["loo_mse"] == pytest.approx(float(((refit - onehot) ** 2).sum(dim=1).mean()), rel=1e-9)
    assert out["lam"] == pytest.approx(lam) and out["n"] == n and out["d"] == d
    assert 0.0 <= out["loo_acc_ho"] <= 1.0 and 0.0 <= out["fit_acc_tr2ho"] <= 1.0
    # the labels' own kernel forecasts them; an empty kernel gives no forecast
    perfect = ft_nap.h5_forecast(onehot, labels, classes, n_ho=20)
    assert perfect["loo_acc"] == 1.0 and perfect["fit_acc_tr2ho"] == 1.0  # (h5_forecast itself does not round)
    assert ft_nap.h5_forecast(torch.ones(n, 3, dtype=torch.float64), labels, classes) is None


def test_probe_is_stratified_deterministic_and_leaves_the_rng_alone():
    val_ds, train_ds = _data(100, 2), _data(300, 1)
    _seed_all(7)
    before = _rng_now()
    probe = ft_nap.build_probe(val_ds, train_ds, None, 40, 120, use_cache=False)
    assert _same_rng(before, _rng_now())
    again = ft_nap.build_probe(val_ds, train_ds, None, 40, 120, use_cache=False)
    for key in ("idx_ho", "idx_tr", "y_ho", "y_cal"):
        assert np.array_equal(probe[key], again[key]), key
    assert torch.equal(probe["x_ho"], again["x_ho"]) and torch.equal(probe["x_cal"], again["x_cal"])
    assert probe["dataset"] == "tensordataset" and probe["x_ho"].shape == (40, 3, 8, 8) and probe["x_cal"].shape == (120, 3, 8, 8)
    # stratified: every block of 10 holds each class once, on both sides; no image twice
    for labels in (probe["y_ho"], probe["y_cal"]):
        for start in range(0, len(labels), 10):
            assert sorted(labels[start:start + 10].tolist()) == list(range(10))
    assert len(set(probe["idx_ho"].tolist())) == 40 and len(set(probe["idx_tr"].tolist())) == 120
    assert torch.equal(probe["x_ho"], val_ds.tensors[0][torch.as_tensor(probe["idx_ho"])])
    assert np.array_equal(probe["y_cal"], train_ds.tensors[1].numpy()[probe["idx_tr"]])
    # the seed is the dataset's name and the side: another name, another probe; the cache returns the same object
    other = ft_nap.build_probe(val_ds, train_ds, "another", 40, 120, use_cache=False)
    assert not np.array_equal(probe["idx_ho"], other["idx_ho"])
    assert ft_nap.seed_of("tensordataset", "probe_ho") != ft_nap.seed_of("tensordataset", "probe_tr")
    assert ft_nap.build_probe(val_ds, train_ds, None, 40, 120) is ft_nap.build_probe(val_ds, train_ds, None, 40, 120)
    order = ft_nap.stratified_order(np.asarray([0] * 7 + [1] * 2 + [2] * 3), 5)
    assert sorted(order.tolist()) == list(range(12)) and sorted(np.asarray([0] * 7 + [1] * 2 + [2] * 3)[order[:6]].tolist()) == [0, 0, 1, 1, 2, 2]


class _Pixels(torch.utils.data.Dataset):
    """A CIFAR-like dataset: uint8 images, ``targets`` and a ``transform`` applied item by item."""

    def __init__(self, data, targets, transform):
        self.data, self.targets, self.transform = data, targets, transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.targets[index]


class _Normalise:
    def __call__(self, image):
        return (torch.as_tensor(image).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.25


class RandomFlipNoise:
    """A train-time augmentation that draws from the global generator."""

    def __call__(self, image):
        image = torch.as_tensor(image)
        if torch.rand(()) < 0.5:
            image = image.flip(1)
        return image + (torch.rand(image.shape) * 8).to(image.dtype)


class _Compose:
    def __init__(self, steps):
        self.transforms = list(steps)

    def __call__(self, image):
        for step in self.transforms:
            image = step(image)
        return image


def test_probe_reads_the_train_split_without_augmentation():
    g = torch.Generator().manual_seed(9)
    pixels = lambda n: torch.randint(0, 256, (n, 8, 8, 3), generator=g, dtype=torch.uint8).numpy()  # noqa: E731
    labels = lambda n: [int(v) for v in torch.randint(0, 10, (n,), generator=g)]  # noqa: E731
    plain = _Compose([_Normalise()])
    train_base = _Pixels(pixels(300), labels(300), _Compose([RandomFlipNoise(), _Normalise()]))
    test_base = _Pixels(pixels(120), labels(120), plain)
    val_idx = torch.randperm(120, generator=g)[:60].tolist()
    val_ds, train_ds = Subset(test_base, val_idx), Subset(train_base, list(range(20, 300)))
    _seed_all(3)
    before = _rng_now()
    probe = ft_nap.build_probe(val_ds, train_ds, None, 20, 50, use_cache=False)
    assert _same_rng(before, _rng_now())
    assert probe["dataset"] == "_pixels" and "val dataset's transform" in probe["how_tr"]
    norm = _Normalise()
    for j in (0, 7, 19):
        base = int(probe["base_ho"][j])
        assert base == val_idx[int(probe["idx_ho"][j])] and probe["y_ho"][j] == test_base.targets[base]
        assert torch.equal(probe["x_ho"][j], norm(test_base.data[base]))
    for j in (0, 13, 49):
        base = int(probe["base_tr"][j])
        assert base == 20 + int(probe["idx_tr"][j]) and probe["y_cal"][j] == train_base.targets[base]
        assert torch.equal(probe["x_cal"][j], norm(train_base.data[base]))  # the stored image, never an augmented one
    assert train_base.transform.transforms[0].__class__ is RandomFlipNoise  # the train dataset itself is untouched
    # a val transform that augments, or a val dataset that cannot be read un-augmented, is refused
    augmenting = Subset(_Pixels(test_base.data, test_base.targets, _Compose([RandomFlipNoise(), _Normalise()])), val_idx)
    with pytest.raises(ValueError, match="augments"):
        ft_nap.build_probe(augmenting, train_ds, None, 20, 50, use_cache=False)
    with pytest.raises(ValueError, match="un-augmented"):
        ft_nap.build_probe(_data(40, 1), train_ds, None, 20, 50, use_cache=False)


def test_fixed_form():
    form, bad = ft_nap.fixed_form(np.arange(15.0))
    assert bad == 0 and len(form) == 16 == len(ft_nap.FORM16)
    assert form[:8] == pytest.approx(np.linspace(0, 14, 8)) and form[8] == pytest.approx(7.0)
    assert form[10:14] == [0.0, 14.0, 0.0, 14.0] and form[14] == pytest.approx(14.0) and form[15] == 1.0
    form, bad = ft_nap.fixed_form([1.0, float("nan"), 3.0, float("inf")])
    assert bad == 2 and all(math.isfinite(v) for v in form) and form[12:14] == [1.0, 3.0]
    assert ft_nap.fixed_form([]) == (None, 0) and ft_nap.fixed_form([float("nan")]) == (None, 1)
    assert ft_nap.fixed_form([5.0])[0] == [5.0] * 8 + [5.0, 0.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0]
    fam = ft_nap._Fam()
    fam.put("a", 1.23456789)
    fam.put("b", float("nan"))
    fam.put("c", None, "not measured")
    fam.form("d_L", [float("nan"), 2.0])
    fam.form("e_L", None, "no window")
    assert fam["a"] == 1.23457 and fam["b"] is None and fam["c"] is None and fam["e_L"] is None and len(fam["d_L"]) == 16
    assert fam["missing"] == [["b", "non-finite"], ["c", "not measured"],
                              ["d_L", "1 of 2 layers non-finite, left out of the form"], ["e_L", "no window"]]
    bad = []
    assert ft_nap._clean({"x": [1.0, float("inf")], "y": np.float32(2.0), "z": np.int64(3)}, "", bad) == {"x": [1.0, None], "y": 2.0, "z": 3}
    assert bad == ["/x[1]"]


def test_cka_unbiased():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(30, 5, generator=g, dtype=torch.float64)
    y = torch.randn(30, 9, generator=g, dtype=torch.float64) + x[:, :1]

    def hsic(k, l):  # the unbiased estimator, written out
        n = k.shape[0]
        k, l = k.clone(), l.clone()
        k.fill_diagonal_(0)
        l.fill_diagonal_(0)
        one = torch.ones(n, dtype=k.dtype)
        return (float((k * l).sum()) + float(k.sum() * l.sum()) / ((n - 1) * (n - 2))
                - 2.0 / (n - 2) * float((k @ one) @ (l @ one))) / (n * (n - 3))

    kx, ky = x @ x.t(), y @ y.t()
    direct = hsic(kx, ky) / math.sqrt(hsic(kx, kx) * hsic(ky, ky))
    assert float(ft_nap.cka_unbiased(x, y)) == pytest.approx(direct, rel=1e-9)
    assert float(ft_nap.cka_unbiased(x, x)) == pytest.approx(1.0, abs=1e-12)
    rotation = torch.linalg.qr(torch.randn(5, 5, generator=g, dtype=torch.float64))[0]
    assert float(ft_nap.cka_unbiased(x @ rotation * 3.0 + 2.0, y)) == pytest.approx(direct, rel=1e-9)
    padded = torch.zeros(2, 30, 9, dtype=torch.float64)
    padded[0, :, :5], padded[1] = x, y
    batch = ft_nap.cka_unbiased(padded, torch.stack([y, y]))
    assert float(batch[0]) == pytest.approx(direct, rel=1e-9) and float(batch[1]) == pytest.approx(1.0, abs=1e-12)
    assert math.isnan(float(ft_nap.cka_unbiased(x[:3], y[:3])))


def test_probe_estimators():
    g = np.random.default_rng(0)
    logits = g.standard_normal((400, 6)) * 2.0
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = np.asarray([g.choice(6, p=row) for row in p])
    t1 = ft_nap.temperature(logits, labels)
    assert 0.7 < t1 < 1.4
    assert ft_nap.temperature(3.0 * logits, labels) == pytest.approx(3.0 * t1, rel=1e-4)
    assert ft_nap.temperature(logits, logits.argmax(axis=1)) == pytest.approx(0.01)  # separable: the lower bound
    side = ft_nap._Side(logits, labels)
    assert side.ce.mean() == pytest.approx(-np.log(p[np.arange(400), labels]).mean())
    fam = ft_nap._p_static(side, side, (side, side))
    assert fam["js_origin_ho"] == pytest.approx(0.0, abs=1e-12) and fam["agree_origin_tr"] == 1.0
    assert fam["gap_acc"] == 0.0 and 0.0 <= fam["ece_ho"] < 0.15 and "missing" not in fam
    assert fam["brier_ho"] == pytest.approx(((p - np.eye(6)[labels]) ** 2).sum(axis=1).mean())
    sharp = ft_nap._Side(np.eye(6)[labels] * 50.0, labels)
    assert ft_nap._p_static(sharp, sharp)["ece_ho"] == pytest.approx(0.0, abs=1e-9)
    assert ft_nap.js_divergence(side.logp, sharp.logp).min() >= 0.0
    # E7: two clusters, and a constant
    weight, separation = ft_nap.gmm2(np.concatenate([g.normal(0.0, 0.1, 300), g.normal(5.0, 0.1, 100)]))
    assert weight == pytest.approx(0.25, abs=0.01) and separation > 20
    assert ft_nap.gmm2(np.ones(50)) is None
    # E1, E2, E6 on a hand-made history
    steps = [0, 1, 2, 4]
    correct = np.asarray([[1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1]], dtype=bool)
    hist = ft_nap.e_history(steps, correct, np.where(correct, 1.0, -1.0), np.where(correct, 0.75, 0.25))
    assert hist["correct0"] == 0.25 and hist["E1_learned"] == 0.75  # inputs 0, 1, 2; input 3 is never correct twice in a row
    assert hist["E1_tau_p50"] == pytest.approx(0.25) and hist["E1_relearned"] == pytest.approx(2.0 / 3.0)
    assert hist["E2_mean"] == pytest.approx(0.5) and hist["E2_any"] == 0.5 and hist["E2_never"] == 0.5
    assert hist["E6_aum_neg"] == 0.0 and hist["E6_conf_low"] == 0.0 and hist["E6_aum_mean"] == pytest.approx(0.25)
    assert hist["E6_conf_mean"] == pytest.approx(0.5625) and hist["E6_var_mean"] == pytest.approx(0.1875)
    # K9 / K10 on one example: the head gradient is e h^T
    tr = ft_nap._Side(np.asarray([[2.0, 0.0, -1.0]]), np.asarray([1]))
    h, w = np.asarray([[1.0, 2.0]]), np.asarray([[1.0, 0.5], [0.0, 0.0], [-0.5, -0.25]])
    e = tr.p[0] - np.asarray([0.0, 1.0, 0.0])
    el2n, grand, gamma, k10 = ft_nap.head_gradient(tr, h, None, np.linalg.norm(w))
    grad = np.outer(e, h[0])
    assert el2n[0] == pytest.approx(np.linalg.norm(e)) and grand[0] == pytest.approx(np.linalg.norm(grad))
    assert gamma[0] == pytest.approx(-(grad * w).sum() / (np.linalg.norm(grad) * np.linalg.norm(w)))
    assert k10[0] == pytest.approx(np.linalg.norm(grad)) and k10[2] == pytest.approx(np.linalg.norm(grad) / np.linalg.norm(h))


def test_sketch_keeps_inner_products(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
                          nn.Flatten(), nn.Linear(128, 400), nn.ReLU(), nn.Linear(400, 10))
    nap = _make_nap(_data(160, 1), _data(64, 2), tmp_path, origin=None)
    nap.on_start(model)
    assert nap.failure is None and nap.P == 55850 and nap.m == 14 and nap.P_pad == 14 * 4096 and nap.nb == 14 * 64
    assert sorted(nap.bperm.tolist()) == list(range(nap.nb)) and set(nap.sign.reshape(-1).tolist()) == {-1.0, 1.0}
    g = torch.Generator().manual_seed(1)
    x = torch.zeros(nap.P_pad)
    y = torch.zeros(nap.P_pad)
    x[:nap.P] = torch.randn(nap.P, generator=g)
    y[:nap.P] = 0.6 * x[:nap.P] + 0.8 * torch.randn(nap.P, generator=g)
    sx, sy = nap._sketch(x).double(), nap._sketch(y).double()
    assert sx.shape == (4096,)
    scale = float(x.norm() * y.norm())
    assert abs(float(sx @ sy) - float(x.double() @ y.double())) < 0.08 * scale
    assert float(sx @ sx) == pytest.approx(float(x.double() @ x.double()), rel=0.1)
    assert torch.allclose(nap._sketch(2.0 * x - y), 2.0 * nap._sketch(x) - nap._sketch(y), atol=1e-3)  # linear
    # the same network and dataset give the same sketch; another network another one
    again = _make_nap(_data(160, 1), _data(64, 2), tmp_path, origin=None)
    again.on_start(copy.deepcopy(model))
    assert torch.equal(again.bperm, nap.bperm) and torch.equal(again.sign, nap.sign)
    other = NapLogger(None, _data(64, 2), _data(160, 1), 10, _device(), LABEL, "path/other.pt", probe_n=24, calib_n=64)
    other.on_start(copy.deepcopy(model))
    assert other.failure is None and not torch.equal(other.bperm, nap.bperm)
    zero = nap.test_recorder.of("eval_ft_nap0")[0]
    assert ["H5", "no origin model"] in zero["H5"]["origin"]["missing"] and zero["H5"]["stale"]["loo_acc"] is not None
    assert zero["H5"]["stale"]["d"] == 400


def test_nonfinite_values_are_missing_never_numbers(tmp_path):
    g = torch.Generator().manual_seed(2)
    model = _net()
    shapes = [p.shape for p in model.parameters()]
    grads = [[torch.randn(s, generator=g) * 0.02 for s in shapes] for _ in range(8)]
    grads[3][0][0, 0, 0, 0] = float("nan")  # step 4: one entry of the first convolution's gradient is not a number
    nap = _make_nap(_data(160, 1), _data(64, 2), tmp_path)
    nap.on_start(model)
    for step, per_param in enumerate(grads, start=1):
        for p, value in zip(model.parameters(), per_param):
            p.grad = value
        nap.on_step(step, torch.tensor(float("nan") if step == 4 else 1.0), torch.tensor(0.9), 0.05)
        assert nap.failure is None
    nap.on_end(8)
    assert nap.failure is None and nap.done
    events = nap.test_recorder.of("eval_ft_nap")
    assert [e["step"] for e in events] == [0, 1, 2, 3, 5, 8]
    assert not [path for e in events for path in _nonfinite(e)]
    at5, at8 = events[4], events[5]
    gone = {name for name, _why in at5["K"]["missing"]}
    assert {"K1", "K8_top", "K3mb_cbar"} <= gone and at5["K"]["K1"] is None and at5["J"]["J4emp"] is None
    assert at5["X"]["loss_last"] is None and ["loss_last", "non-finite"] in at5["X"]["missing"]
    assert at5["S"]["S1_dist"] == 0.0 and at5["Y"]["Y4_intr_sum"] == 0.0  # the weights never moved: finite, as they are
    assert at8["K"]["K1"] is not None and at8["J"]["J4emp"] is not None  # the next window is clean again
    assert at8["X"]["loss_last"] is not None and at8["X"]["ema09"] is None  # the running average keeps the bad step
    data = _load(nap)
    for name, value in data.items():
        if name != "meta" and value.dtype.kind == "f":
            assert np.isfinite(value).all(), name
    ok = data["step_series__ok"]
    assert ok[3, 0] == 0 and ok.sum() == ok.size - 1
    assert data["step_layer_gnorm__ok"][3].tolist() == [0, 1, 1] and data["step_layer_gnorm__ok"].sum() == 8 * 3 - 1
    assert "step_series" in data["meta"]["masked"]
    # the host solvers give None, never an exception, on a matrix that is not finite
    bad = np.full((3, 3), np.nan)
    assert ft_nap._eigh(bad) is None and ft_nap._eigh(bad, vectors=True) is None and ft_nap.spectrum_ranks(bad) is None
    assert ft_nap._eigh(np.zeros((0, 0))) is None and ft_nap.spectrum_ranks(np.zeros((4, 4))) is None
    values, vectors = ft_nap._eigh(np.diag([3.0, 1.0, 2.0]), vectors=True)
    assert values.tolist() == [1.0, 2.0, 3.0] and vectors.shape == (3, 3)
    feats = torch.randn(12, 4, generator=g, dtype=torch.float64)
    feats[0, 0] = float("nan")
    assert ft_nap.h5_forecast(feats, torch.arange(12) % 3, 3, n_ho=6) is None


def test_synflow_opens_clipped_relus(tmp_path):
    def net():
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU6(inplace=True),
                              nn.Conv2d(4, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU6(inplace=True),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(6, 10))
        with torch.no_grad():
            model[0].weight.mul_(20.0)  # with absolute weights an all-ones input now saturates the clip at 6
        return model

    model = net()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    nap = _make_nap(_data(160, 1), _data(64, 2), tmp_path, origin=None)
    nap.on_start(model)
    assert nap.failure is None and _same(state, model.state_dict())
    z4 = nap.test_recorder.of("eval_ft_nap0")[0]["Z4"]
    assert z4["synflow_relu6_opened"] == 2 and "missing" not in z4 and all(v is not None for v in z4["synflow_L"])
    assert model[2].max_val == 6.0 and model[5].max_val == 6.0 and model[1].eps == 1e-5  # the live model is as it was
    flow = net().double().eval()
    with torch.no_grad():
        for p in flow.parameters():
            p.abs_()
        for m in flow.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1.0)
                m.bias.zero_()
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
                m.eps = 0.0
    ones = torch.ones(1, 3, 8, 8, dtype=torch.float64)
    clipped = flow(ones).sum()
    for index in (2, 5):
        flow[index] = nn.ReLU()
    total = flow(ones).sum()
    total.backward()
    scores = [float((m.weight * m.weight.grad).sum()) for m in (flow[0], flow[3], flow[9])]
    assert float(clipped) < float(total) and min(scores) > 0  # the clip was active, and open every layer scores
    assert z4["synflow_log"] == pytest.approx(math.log(sum(scores)), rel=1e-6)
    assert z4["synflow_logR"] == pytest.approx(math.log(float(total)), rel=1e-6)
    assert nap.cut_arrays["z4_synflow_log"] == pytest.approx(np.log(scores), rel=1e-9)
    h5 = nap.test_recorder.of("eval_ft_nap0")[0]["H5"]["stale"]
    assert h5["n"] == 48 and isinstance(h5["n"], int) and h5["d"] == 6 and isinstance(h5["d"], int)


# --- the runner ---------------------------------------------------------------------------------------------------------

def _runner_env(monkeypatch):
    import a2c_agent_reinforce_runner as runner
    from tests.test_v9c_traj_models import _Env, _candidates, _point, _quiet
    lines, _ = _quiet(monkeypatch, runner)
    seen = []

    class _HookEnv(_Env):
        def create_learning_handler(self, model):
            handler = super().create_learning_handler(model)
            train = handler.train_model

            def train_model(loader, **kwargs):
                seen.append(kwargs.get("step_hook", "absent"))
                return train(loader, **kwargs)

            handler.train_model = train_model
            return handler

    return runner, _HookEnv, _candidates, _point, lines, seen


def test_runner_builds_nap_only_with_both_flags(monkeypatch):
    runner, env_cls, candidates, point, lines, seen = _runner_env(monkeypatch)
    picked = {"val_best": point(5, 0.7, -9.0)}
    runner._run_final_ft(env_cls(), "net.pt", picked, candidates(), 2)
    assert seen == ["absent", "absent"] and not any("FT_NAP" in line for line in lines)
    monkeypatch.setenv("SPECTRA_EVAL_FT_NAP", "1")  # without TRACE: one line per candidate, no hook
    runner._run_final_ft(env_cls(), "net.pt", picked, candidates(), 2)
    assert seen[2:] == ["absent", "absent"]
    assert sum("FT_NAP" in line and "needs SPECTRA_EVAL_FT_TRACE > 0; ignored" in line for line in lines) == 2
    monkeypatch.setenv("SPECTRA_EVAL_FT_TRACE", "25")
    env = env_cls()
    runner._run_final_ft(env, "net.pt", picked, candidates(), 2)
    hooks = seen[4:]
    assert len(hooks) == 2 and all(isinstance(h, FtTrace) and isinstance(h.nap, NapLogger) for h in hooks)
    nap = hooks[1].nap
    assert nap.origin is env.data_dict["net.pt"][0] and nap.val_dataset is env.val_loader.dataset
    assert nap.train_dataset is env.train_loader.dataset and nap.label == "size_param0.80" and nap.network == "net.pt"
    assert nap.batch_size == 128 and nap.momentum == 0.9 and nap.weight_decay == 5e-4 and callable(nap.out_dir)
    assert nap.recorder is runner.run_recorder and nap.probe_n == 500 and nap.calib_n == 2000
    monkeypatch.delenv("SPECTRA_EVAL_FT_NAP")  # TRACE alone: the hook as before
    runner._run_final_ft(env_cls(), "net.pt", picked, candidates(), 2)
    assert all(isinstance(h, FtTrace) and h.nap is None for h in seen[6:]) and len(seen) == 8


def test_runner_ignores_nap_on_the_imagenet_path(monkeypatch):
    runner, env_cls, candidates, point, lines, seen = _runner_env(monkeypatch)
    monkeypatch.setenv("SPECTRA_EVAL_FT_NAP", "1")
    monkeypatch.setenv("SPECTRA_EVAL_FT_TRACE", "25")
    monkeypatch.setattr(runner.imagenet_fast, "enabled", lambda: True)
    monkeypatch.setattr(runner, "_final_ft_imagenet", lambda *args, **kwargs: 0.5)
    cand = candidates()["val_best"]
    assert runner._final_ft(env_cls(), "net.pt", "val_best", cand, 2) == 0.5
    assert any("FT_NAP val_best net.pt: not supported on the ImageNet path; ignored" in line for line in lines)
    assert any("FT_TRACE val_best net.pt: not supported on the ImageNet path; ignored" in line for line in lines)
    assert seen == []
