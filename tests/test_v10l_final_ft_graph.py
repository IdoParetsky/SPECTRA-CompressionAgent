"""CPU tests for tree_v10l (8 Oct): ``SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH`` replays the final fine-tune's forward
and backward as CUDA graphs. The login node has no GPU, so ``torch.cuda.make_graphed_callables`` is stood in
for: the default call is unchanged, the runner passes the flag only when set, the handler falls back to eager
where a graph cannot apply, and TrainGraph captures once, dispatches by shape, puts the BatchNorm buffers back
and releases the graphed forward. Speed and GPU parity are scripts/bench_final_ft_graph.py's job.

    python -m pytest tests/test_v10l_final_ft_graph.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import src.ModelHandlers.ClassificationHandler as handler_mod  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler, TrainGraph, cuda_graph_blocker  # noqa: E402
from tests.test_v9c_traj_models import _Env, _candidates, _point, _quiet  # noqa: E402

KEYS = ("SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH", "SPECTRA_EVAL_FINAL_FT_SELECT", "SPECTRA_EVAL_FINAL_FT_KD",
        "SPECTRA_EVAL_FINAL_FT_SCHEDULE", "SPECTRA_EVAL_FINAL_FT_BATCH", "SPECTRA_AMP", "SPECTRA_CHANNELS_LAST",
        "SPECTRA_FT_MIXUP", "SPECTRA_FT_KD", "SPECTRA_FT_AUG", "SPECTRA_FT_SCHEDULE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(4, 4))


def _loader(n=40, batch=16):
    g = torch.Generator().manual_seed(0)
    return DataLoader(TensorDataset(torch.randn(n, 3, 8, 8, generator=g), torch.randint(0, 4, (n,), generator=g)),
                      batch_size=batch)


def _capture_logs(monkeypatch):
    import src.run_recorder as recorder
    import src.utils as utils
    printed, records = [], []
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))
    monkeypatch.setattr(recorder, "record", lambda kind, **k: records.append(dict(k, kind=kind)))
    return printed, records


def _fake_graphing(calls):
    """Stand-in for make_graphed_callables: the warm-up's forwards move the BatchNorm stats, and the module's
    forward becomes a counted wrapper around the eager one (an instance attribute, as the real one sets)."""
    def make_graphed_callables(module, sample_args, **kwargs):
        calls.append(tuple(sample_args[0].shape))
        assert kwargs == {"allow_unused_input": True}
        with torch.no_grad():
            for _ in range(3):
                module(sample_args[0])
        eager = module.forward

        def graphed(*args):
            graphed.n += 1
            return eager(*args)

        graphed.n = 0
        module.forward = graphed
        return module

    return make_graphed_callables


def _fake_cuda(monkeypatch, calls):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "make_graphed_callables", _fake_graphing(calls))


def test_flag_parses(monkeypatch):
    assert not fortify.eval_final_ft_cuda_graph()
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH", "1")
    assert fortify.eval_final_ft_cuda_graph()


def test_runner_passes_cuda_graph_only_when_set(monkeypatch):
    import a2c_agent_reinforce_runner as runner
    lines, _ = _quiet(monkeypatch, runner)
    seen = []

    class _GraphEnv(_Env):
        def create_learning_handler(self, model):
            handler = super().create_learning_handler(model)
            train = handler.train_model

            def train_model(loader, **kwargs):
                seen.append(kwargs.get("cuda_graph", "absent"))
                return train(loader, **kwargs)

            handler.train_model = train_model
            return handler

    runner._run_final_ft(_GraphEnv(), "net.pt", {"val_best": _point(5, 0.7, -9.0)}, _candidates(), 10)
    assert seen == ["absent", "absent"] and not any("graph=1" in ln for ln in lines)
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH", "1")
    runner._run_final_ft(_GraphEnv(), "net.pt", {"val_best": _point(5, 0.7, -9.0)}, _candidates(), 10)
    assert seen[2:] == [True, True]
    assert sum(" kd=0 graph=1 init=inherit " in ln for ln in lines) == 2


def test_default_call_unchanged_and_cpu_falls_back_to_the_same_arithmetic(monkeypatch):
    printed, records = _capture_logs(monkeypatch)
    states = {}
    for flag in (False, True):
        model = _net()
        torch.manual_seed(1)
        ClassificationHandler(model, nn.CrossEntropyLoss()).train_model(
            _loader(), allow_reinit_retry=False, max_epochs=2, patience=3, keep_last=True, cuda_graph=flag)
        states[flag] = {k: v.clone() for k, v in model.state_dict().items()}
        assert "forward" not in model.__dict__
    assert all(torch.equal(states[False][k], states[True][k]) for k in states[False])
    recipe = [ln for ln in printed if "Fine-tune recipe" in ln]
    assert "graph=" not in recipe[0] and recipe[1].endswith(" graph=0")
    assert sum("CUDA graph off (no CUDA device); eager fine-tune" in ln for ln in printed) == 1
    finetune = [r for r in records if r["kind"] == "finetune"]
    assert "cuda_graph" not in finetune[0] and finetune[1]["cuda_graph"] is False


def test_blocker_reasons(monkeypatch):
    model = _net()
    assert cuda_graph_blocker(model, use_cuda=False) == "no CUDA device"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cuda_graph_blocker(model, True) == ""
    assert cuda_graph_blocker(model, True, use_amp=True) == "AMP"
    assert cuda_graph_blocker(model, True, use_channels_last=True) == "channels_last"
    assert cuda_graph_blocker(model, True, use_kd=True) == "KD"
    assert cuda_graph_blocker(model, True, mixup_alpha=0.2) == "mixup"
    model[0].weight.requires_grad = False
    assert cuda_graph_blocker(model, True) == "frozen parameters"
    model[0].weight.requires_grad = True
    model[2].register_forward_hook(lambda *a: None)
    assert cuda_graph_blocker(model, True) == "module hooks"


def test_train_graph_captures_once_dispatches_by_shape_and_releases(monkeypatch):
    printed, _ = _capture_logs(monkeypatch)
    calls = []
    _fake_cuda(monkeypatch, calls)
    model, ref = _net().train(), _net().train()
    graph = TrainGraph(model, "[t] ")
    x = torch.randn(16, 3, 8, 8)
    out, ref_out = graph(x), ref(x)
    assert calls == [(16, 3, 8, 8)] and graph.shape == (16, 3, 8, 8) and model.forward.n == 1
    # the warm-up's three forwards are undone: the stats are one train step from the start, as eager
    for name, buffer in model[1].named_buffers():
        assert torch.equal(buffer, dict(ref[1].named_buffers())[name]), name
    assert torch.equal(out, ref_out)
    graph(torch.randn(16, 3, 8, 8))
    graph(torch.randn(8, 3, 8, 8))  # a short last batch runs the eager forward
    assert (graph.graphed_steps, graph.eager_steps) == (2, 1) and model.forward.n == 2 and len(calls) == 1
    graph.release()
    assert "forward" not in model.__dict__ and model.forward.__func__ is nn.Sequential.forward
    assert any(ln.startswith("[t] CUDA graph captured: batch 16") for ln in printed)


def test_failed_capture_runs_eager_without_retrying(monkeypatch):
    printed, _ = _capture_logs(monkeypatch)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    def boom(module, sample_args, **kwargs):
        module(sample_args[0])
        module.forward = lambda *a: None
        raise RuntimeError("operation not permitted when stream is capturing")

    monkeypatch.setattr(torch.cuda, "make_graphed_callables", boom)
    model, ref = _net().train(), _net().train()
    graph = TrainGraph(model)
    x = torch.randn(16, 3, 8, 8)
    out, ref_out = graph(x), ref(x)
    assert graph.failed and graph.shape is None and "forward" not in model.__dict__
    assert torch.equal(out, ref_out) and torch.equal(model[1].running_mean, ref[1].running_mean)
    graph(x)
    assert graph.eager_steps == 2 and graph.graphed_steps == 0
    assert sum("CUDA graph capture failed (RuntimeError" in ln for ln in printed) == 1


def test_train_model_through_train_graph_matches_eager(monkeypatch):
    printed, records = _capture_logs(monkeypatch)
    calls = []
    _fake_cuda(monkeypatch, calls)
    states = {}
    for flag in (False, True):
        if flag:
            monkeypatch.setattr(handler_mod, "cuda_graph_blocker", lambda *a, **k: "")
        model = _net()
        torch.manual_seed(1)
        ClassificationHandler(model, nn.CrossEntropyLoss()).train_model(
            _loader(n=40, batch=16), allow_reinit_retry=False, max_epochs=2, patience=3, keep_last=True,
            cuda_graph=flag)
        states[flag] = {k: v.clone() for k, v in model.state_dict().items()}
        assert "forward" not in model.__dict__
    assert calls == [(16, 3, 8, 8)]  # one capture, at the first full batch
    assert all(torch.equal(states[False][k], states[True][k]) for k in states[False])
    assert any(ln.endswith(" graph=1") for ln in printed if "Fine-tune recipe" in ln)
    assert any("CUDA graph: 4 graphed steps, 2 eager" in ln for ln in printed)  # 16 + 16 + 8 per epoch
    assert [r.get("cuda_graph") for r in records if r["kind"] == "finetune"] == [None, True]
