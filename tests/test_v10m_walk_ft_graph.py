"""CPU tests for tree_v10m (8 Oct): ``SPECTRA_FT_CUDA_GRAPH`` replays an eval walk's per-step recipe-A
fine-tune as CUDA graphs. The walk passes ``cuda_graph=True`` only when the flag is set and never in
``AGENT_TRAIN``; ``TrainGraph.release`` collects the dropped graphs, since a walk captures once per step.

    python -m pytest tests/test_v10m_walk_ft_graph.py -v
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import src.ModelHandlers.ClassificationHandler as handler_mod  # noqa: E402
from src.ModelHandlers.ClassificationHandler import TrainGraph  # noqa: E402
from src.NetworkEnv import AGENT_TRAIN, EVAL_TEST, NetworkEnv  # noqa: E402
from tests.test_v10l_final_ft_graph import _capture_logs, _fake_cuda, _net  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in [k for k in os.environ if k.startswith("SPECTRA_")]:
        monkeypatch.delenv(key, raising=False)
    yield


class _Handler:
    def __init__(self):
        self.calls = []

    def unfreeze_all_layers(self):
        pass

    def freeze_all_layers_but_pruned(self, params):
        pass

    def train_model(self, loader, **kwargs):
        self.calls.append(kwargs)


def _env(mode):
    env = NetworkEnv.__new__(NetworkEnv)
    env.conf = SimpleNamespace(train_compressed_layer_only=False)
    env.mode, env.train_loader, env.val_loader = mode, "train-loader", "val-loader"
    return env


def _recover(env):
    handler = _Handler()
    recipe = env._recover_after_prune(handler, SimpleNamespace(), {"mode": "masked"}, is_to_train=True)
    return recipe, handler.calls


def test_flag_parses(monkeypatch):
    assert not fortify.ft_cuda_graph()
    monkeypatch.setenv("SPECTRA_FT_CUDA_GRAPH", "1")
    assert fortify.ft_cuda_graph()


def test_walk_passes_cuda_graph_only_when_set_and_never_in_agent_train(monkeypatch):
    assert _recover(_env(EVAL_TEST)) == ("A", [{}])
    monkeypatch.setenv("SPECTRA_FT_CUDA_GRAPH", "1")
    assert _recover(_env(EVAL_TEST)) == ("A", [{"cuda_graph": True}])
    assert _recover(_env(AGENT_TRAIN)) == ("A", [{}])
    monkeypatch.setenv("SPECTRA_TRAIN_FT_EPOCHS", "3")
    assert _recover(_env(AGENT_TRAIN))[1] == [{"max_epochs": 3, "patience": fortify.train_ft_patience()}]


def test_release_collects_only_when_a_graphed_forward_was_dropped(monkeypatch):
    _capture_logs(monkeypatch)
    _fake_cuda(monkeypatch, [])
    collected = []
    monkeypatch.setattr(handler_mod.gc, "collect", lambda *a, **k: collected.append(1) or 0)
    TrainGraph(_net().train()).release()
    assert collected == []
    model = _net().train()
    graph = TrainGraph(model)
    graph(torch.randn(16, 3, 8, 8))
    graph.release()
    assert collected == [1] and "forward" not in model.__dict__
    graph.release()
    assert collected == [1]
