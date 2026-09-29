"""CPU tests for tree_v9c (29 Sep): TRAJ candidates saved as ``state_dict`` + arch/recipe JSON instead
of pickled modules, a final fine-tune that survives one bad candidate, the scratch control, and a
final fine-tune from a saved walk. Everything defaults off.

    python -m pytest tests/test_v9c_traj_models.py -v
"""

import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src.NetworkEnv import prune_current_model  # noqa: E402
import src.traj_models as traj_models  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
V9C_KEYS = ("SPECTRA_EVAL_SAVE_TRAJ_MODELS", "SPECTRA_EVAL_FINAL_FT_SCRATCH", "SPECTRA_EVAL_FINAL_FT_SCRATCH_EPOCHS",
            "SPECTRA_EVAL_FINAL_FT_SCRATCH_LR", "SPECTRA_EVAL_FINAL_FT_FROM", "SPECTRA_EVAL_FINAL_FT_ORIGIN",
            "SPECTRA_EVAL_FINAL_FT_EPOCHS", "SPECTRA_EVAL_FINAL_FT_BATCH", "SPECTRA_EVAL_FINAL_FT_KD",
            "SPECTRA_FT_AUG", "SPECTRA_FT_AUTOAUG")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in V9C_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


def _point(step, param, val):
    return {"step": step, "rate": 0.9, "param": param, "flop": param, "val_acc": 0.9 + val / 100.0,
            "val_origin": 0.9, "val_dacc_pp": val, "test_acc": 0.8 + val / 100.0, "test_origin": 0.8,
            "test_dacc_pp": val}


def _prune_rows(model, rate=0.5, max_rows=12):
    """Real SPECTRA cuts on the first rows, one per step as the env makes them; returns the pruned
    module and how many cuts were structural."""
    structural = 0
    for row in range(min(len(ModelWithRows(model).row_to_main_layer), max_rows)):
        mwr = ModelWithRows(model)
        prune_current_model(mwr, rate, row, quiet=True, record=False, input_shape=(3, 32, 32))
        model = mwr.model
        if mwr.last_prune_outcome.get("mode") == "structural":
            structural += 1
    return model.eval(), structural


def _file_path_module(name):
    """Import an instantiation file the way the cluster loader does: by path, not by package name."""
    spec = importlib.util.spec_from_file_location(f"{name}_v9c_by_path",
                                                  REPO / "spectra_models_instantiation" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- the 21726337 crash, reproduced

def test_a_file_path_class_cannot_be_pickled_but_its_state_dict_round_trips(tmp_path):
    mod = _file_path_module("thin_res_net")
    pruned, structural = _prune_rows(mod.resnet20(num_classes=10, large_input=False, width=4).eval())
    assert structural >= 1
    with pytest.raises(Exception, match="pickle|import"):
        torch.save(pruned, io.BytesIO())                          # what tree_v9b did after every walk
    stem = traj_models.save_candidate(pruned, str(tmp_path / "net.pt__val_best__step7"), _point(7, 0.8, -3.0))
    assert stem is not None
    assert all(isinstance(v, torch.Tensor) for v in torch.load(stem + ".pt", weights_only=True).values())
    rebuilt, doc = traj_models.load_candidate(mod.resnet20(num_classes=10, large_input=False, width=4), stem)
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(rebuilt.eval()(x), pruned(x), atol=1e-6)
    assert doc["point"]["step"] == 7 and doc["params"] == sum(p.numel() for p in pruned.parameters())


# ---------------------------------------------------------------- the four P-cell architectures

ARCHES = [
    ("thin_res_net", "resnet20", {"num_classes": 10, "large_input": False, "width": 4}),
    ("vgg_chenyaofo", "vgg11_bn", {"num_classes": 100, "large_input": False}),
    ("resnet_chenyaofo", "resnet20", {"num_classes": 10, "large_input": False}),
    ("vgg_depgraph", "vgg11_bn", {"num_classes": 100, "large_input": False}),
]


@pytest.mark.parametrize("module_name,factory,kwargs", ARCHES, ids=[a[0] for a in ARCHES])
def test_pruned_state_dict_rebuilds_into_a_fresh_original(tmp_path, module_name, factory, kwargs):
    mod = _file_path_module(module_name)
    build = getattr(mod, factory)
    torch.manual_seed(0)
    pruned, structural = _prune_rows(build(**kwargs).eval())
    assert structural >= 2, f"{module_name}: expected real structural cuts"
    fresh = build(**kwargs)
    assert sum(p.numel() for p in fresh.parameters()) > sum(p.numel() for p in pruned.parameters())
    stem = traj_models.save_candidate(pruned, str(tmp_path / f"{module_name}.pt__size_param0.80__step3"),
                                      _point(3, 0.8, -2.0), {"network": f"{module_name}.pt"})
    rebuilt, doc = traj_models.load_candidate(fresh, stem)
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(rebuilt.eval()(x), pruned(x), atol=1e-5)
    assert traj_models.module_arch(rebuilt) == traj_models.module_arch(pruned)
    assert sum(p.numel() for p in fresh.parameters()) > doc["params"]   # the template was copied, not edited
    json.dumps(doc)                                                     # arch + recipe stay plain JSON


def test_save_candidate_never_raises(monkeypatch, tmp_path):
    def boom(*_a, **_k):
        raise OSError("disk full")
    lines = []
    monkeypatch.setattr(traj_models.utils, "print_flush", lambda msg: lines.append(str(msg)))
    monkeypatch.setattr(traj_models.torch, "save", boom)
    assert traj_models.save_candidate(nn.Linear(2, 2), str(tmp_path / "x")) is None
    assert lines == ["[eval] TRAJ save failed for x: OSError: disk full; continuing"]


def test_load_candidates_parses_labels_and_skips_fine_tuned_copies(tmp_path):
    template = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
    small = nn.Sequential(nn.Linear(4, 2), nn.BatchNorm1d(2))
    d = str(tmp_path)
    traj_models.save_candidate(small, traj_models.candidate_stem(d, "a.pt", "val_best", 5), _point(5, 0.7, -9.9))
    traj_models.save_candidate(small, traj_models.candidate_stem(d, "a.pt", "size_param0.80", 3), _point(3, 0.8, -8))
    traj_models.save_candidate(template, traj_models.candidate_stem(d, "a.pt", "origin", -1), _point(-1, 1.0, 0.0))
    traj_models.save_candidate(small, traj_models.candidate_stem(d, "a.pt", "val_best", 5, "__ft100"), _point(5, 0.7, 0))
    traj_models.save_candidate(small, traj_models.candidate_stem(d, "b.pt", "val_best", 9), _point(9, 0.5, -1))
    loaded = traj_models.load_candidates(d, "a.pt", template)
    assert sorted(loaded) == ["origin", "size_param0.80", "val_best"]
    assert loaded["val_best"]["point"]["step"] == 5 and loaded["origin"]["point"]["step"] == -1
    assert loaded["val_best"]["model"][0].out_features == 2 and template[0].out_features == 3


def test_rebuild_rejects_an_arch_from_another_network():
    with pytest.raises(KeyError):
        traj_models.rebuild_from_state_dict(nn.Linear(2, 2), {}, {"features.0": {"type": "Linear"}})


def test_reinit_parameters_draws_new_weights_and_keeps_shapes():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Linear(4, 2))
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(7.0)
        net[1].running_mean.fill_(3.0)
    shapes = [p.shape for p in net.parameters()]
    assert traj_models.reinit_parameters(net) == 3
    assert [p.shape for p in net.parameters()] == shapes
    assert not torch.any(net[0].weight == 7.0) and float(net[1].running_mean.abs().max()) == 0.0


def test_scratch_and_from_flags(monkeypatch):
    assert fortify.eval_final_ft_scratch() == "" and fortify.eval_final_ft_from() == ""
    assert fortify.eval_final_ft_scratch_epochs() == 200 and fortify.eval_final_ft_scratch_lr() == 0.1
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH", "1")
    assert fortify.eval_final_ft_scratch() == "both"
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH", "only")
    assert fortify.eval_final_ft_scratch() == "only"
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_FROM", " /tmp/x ")
    assert fortify.eval_final_ft_from() == "/tmp/x"


# ---------------------------------------------------------------- runner

class _Env:
    def __init__(self, fail_label=None):
        self.train_loader = DataLoader(TensorDataset(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long)),
                                       batch_size=2)
        self.val_loader = self.test_loader = self.train_loader
        self.selected_net_path = "net.pt"
        self.data_dict = {"net.pt": (nn.Linear(1, 1, bias=False), None)}
        self.fail_label = fail_label
        self.runs = []

    def create_learning_handler(self, model):
        env = self

        class _Handler:
            def train_model(self, loader, **kwargs):
                if env.fail_label and kwargs.get("tag") == f"final FT {env.fail_label}":
                    raise RuntimeError("CUDA OOM")
                env.runs.append({"tag": kwargs.get("tag"), "start": float(model.weight),
                                 "epochs": kwargs.get("max_epochs"), "lr": os.environ.get("SPECTRA_FT_SGD_LR")})

            def evaluate_model(self, _loader):
                return 0.75

        return _Handler()


def _candidates():
    out = {}
    for label, step, w in (("val_best", 5, 5.0), ("size_param0.80", 3, 3.0)):
        m = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            m.weight.fill_(w)
        out[label] = {"point": _point(step, 0.7 if step == 5 else 0.8, -9.0), "model": m, "key": None}
    return out


def _quiet(monkeypatch, runner, tmp_path=None):
    lines, issues = [], []
    monkeypatch.setattr(runner.utils, "print_flush", lambda msg: lines.append(str(msg)))
    monkeypatch.setattr(runner.run_recorder, "record", lambda *a, **k: None)
    monkeypatch.setattr(runner.run_recorder, "issue", lambda kind, msg, **k: issues.append((kind, msg, k)))
    monkeypatch.setattr(runner.logging_utils, "exception", lambda msg: lines.append(f"EXC {msg}"))
    if tmp_path is not None:
        monkeypatch.setattr(runner.run_recorder, "recorder", lambda: SimpleNamespace(run_dir=str(tmp_path)))
    return lines, issues


def test_scratch_modes_train_a_fresh_copy_with_the_scratch_budget(monkeypatch):
    import a2c_agent_reinforce_runner as runner
    _quiet(monkeypatch, runner)
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH", "1")
    env = _Env()
    runner._run_final_ft(env, "net.pt", {"val_best": _point(5, 0.7, -9.0)}, _candidates(), 100)
    assert [r["tag"] for r in env.runs] == ["final FT val_best", "final FT val_best+scratch",
                                            "final FT size_param0.80", "final FT size_param0.80+scratch"]
    inherit, scratch = env.runs[0], env.runs[1]
    assert inherit["start"] == 5.0 and inherit["epochs"] == 100 and inherit["lr"] == "0.01"
    assert abs(scratch["start"]) <= 1.0 and scratch["epochs"] == 200 and scratch["lr"] == "0.1"
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH", "only")
    env = _Env()
    runner._run_final_ft(env, "net.pt", {"val_best": _point(5, 0.7, -9.0)}, _candidates(), 100)
    assert [r["tag"] for r in env.runs] == ["final FT val_best+scratch", "final FT size_param0.80+scratch"]


def test_one_failing_candidate_does_not_cost_the_others(monkeypatch):
    import a2c_agent_reinforce_runner as runner
    lines, issues = _quiet(monkeypatch, runner)
    env = _Env(fail_label="val_best")
    runner._run_final_ft(env, "net.pt", {"val_best": _point(5, 0.7, -9.0)}, _candidates(), 10)
    assert [r["tag"] for r in env.runs] == ["final FT size_param0.80"]
    assert issues and issues[0][0] == "final_ft_failed" and "CUDA OOM" in issues[0][1]
    assert any(ln.startswith("EXC [eval] TRAJ final_ft val_best net.pt failed") for ln in lines)


def test_saved_files_are_state_dicts_with_arch_json(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    _quiet(monkeypatch, runner, tmp_path)
    monkeypatch.setenv("SPECTRA_EVAL_SAVE_TRAJ_MODELS", "1")
    runner._run_final_ft(_Env(), "/nets/net.pt", {"val_best": _point(5, 0.7, -9.0)}, _candidates(), 10)
    names = sorted(os.listdir(tmp_path / "traj_models"))
    assert "net.pt__val_best__step5.pt" in names and "net.pt__val_best__step5.json" in names
    assert "net.pt__val_best__step5__ft10.pt" in names and "net.pt__size_param0.80__step3__ft10.json" in names
    state = torch.load(tmp_path / "traj_models" / "net.pt__val_best__step5.pt", weights_only=True)
    assert isinstance(state, dict) and set(state) == {"weight"}
    doc = json.loads((tmp_path / "traj_models" / "net.pt__val_best__step5__ft10.json").read_text())
    assert doc["final_ft"]["epochs"] == 10 and doc["arch"][""]["type"] == "Linear" and doc["point"]["step"] == 5


def test_final_ft_from_saved_hands_the_saved_candidates_to_the_final_ft(monkeypatch, tmp_path):
    import a2c_agent_reinforce_runner as runner
    lines, _ = _quiet(monkeypatch, runner)
    d = str(tmp_path)
    for label, cand in _candidates().items():
        traj_models.save_candidate(cand["model"], traj_models.candidate_stem(d, "net.pt", label, cand["point"]["step"]),
                                   cand["point"])
    env = _Env()
    env.score_test_loader = lambda: (0.8, 0.8, None)
    env.last_val_acc = env.original_acc = 0.9
    env.param_ratio = env.flops_ratio = lambda: 1.0
    seen = {}
    monkeypatch.setattr(runner, "_run_final_ft",
                        lambda e, net, picked, cands, epochs: seen.update(picked=picked, cands=cands, epochs=epochs))
    runner._final_ft_from_saved(env, "/nets/net.pt", d, 100)
    assert sorted(seen["cands"]) == ["size_param0.80", "val_best"] and seen["epochs"] == 100
    assert float(seen["cands"]["val_best"]["model"].weight) == 5.0
    assert seen["picked"]["val_best"]["step"] == 5 and seen["picked"]["origin"]["step"] == -1
    assert any("final_ft from" in ln and "val_best" in ln for ln in lines)
