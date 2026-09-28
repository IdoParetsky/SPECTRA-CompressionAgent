"""CPU tests for the V9b protocol cell (28 Sep): val carved from the held-out test split, the
TRAJ final fine-tune and its candidate copies, pre-registered size points, and the offline
selection readout. Everything defaults off.

    python -m pytest tests/test_v9b_protocol.py -v
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src import fortify  # noqa: E402
from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import src.utils as utils  # noqa: E402

V9B_KEYS = ("SPECTRA_VAL_FROM_TEST", "SPECTRA_VAL_TEST_FRACTION", "SPECTRA_SPLIT_SEED",
            "SPECTRA_EVAL_SIZE_POINTS", "SPECTRA_EVAL_FINAL_FT_EPOCHS", "SPECTRA_EVAL_FINAL_FT_LR",
            "SPECTRA_EVAL_FINAL_FT_BATCH", "SPECTRA_EVAL_FINAL_FT_KD", "SPECTRA_EVAL_FINAL_FT_ORIGIN",
            "SPECTRA_EVAL_SAVE_TRAJ_MODELS", "SPECTRA_FT_AUG", "SPECTRA_FT_AUTOAUG")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in V9B_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


def _point(step, param, val, test=None, flop=None):
    return {"step": step, "rate": 0.9, "param": param, "flop": param if flop is None else flop,
            "val_acc": 0.9 + val / 100.0, "val_origin": 0.9, "val_dacc_pp": val,
            "test_acc": 0.8 + (val if test is None else test) / 100.0, "test_origin": 0.8,
            "test_dacc_pp": val if test is None else test}


# A flat, noisy val curve around -10: out of band at steps 3-4, back in at 5, out again after.
CURVE = [_point(-1, 1.0, 0.0), _point(0, 0.95, -3.0), _point(1, 0.9, -8.0), _point(2, 0.85, -9.6),
         _point(3, 0.8, -10.4), _point(4, 0.75, -10.2), _point(5, 0.7, -9.9), _point(6, 0.65, -11.0),
         _point(7, 0.6, -12.5)]


# ---------------------------------------------------------------- val from the held-out split

def test_val_from_test_fraction_parsing(monkeypatch):
    assert utils.val_from_test_fraction() == 0.0
    monkeypatch.setenv("SPECTRA_VAL_FROM_TEST", "1")
    assert utils.val_from_test_fraction() == 0.5
    monkeypatch.setenv("SPECTRA_VAL_TEST_FRACTION", "0.3")
    assert utils.val_from_test_fraction() == pytest.approx(0.3)
    monkeypatch.setenv("SPECTRA_VAL_TEST_FRACTION", "0.99")
    assert utils.val_from_test_fraction() == pytest.approx(0.9)
    monkeypatch.setenv("SPECTRA_VAL_TEST_FRACTION", "half")
    assert utils.val_from_test_fraction() == 0.5


def test_split_held_out_is_disjoint_complete_and_seeded(monkeypatch):
    data = TensorDataset(torch.arange(100))
    val, test = utils._split_held_out(data, 0.5)
    assert len(val) == 50 and len(test) == 50
    assert set(val.indices).isdisjoint(test.indices)
    assert sorted(val.indices + test.indices) == list(range(100))
    torch.manual_seed(123)                                    # the global RNG must not matter
    again, _ = utils._split_held_out(data, 0.5)
    assert again.indices == val.indices
    monkeypatch.setenv("SPECTRA_SPLIT_SEED", "7")
    other, _ = utils._split_held_out(data, 0.5)
    assert other.indices != val.indices


def _fake_builder(n_train=80, n_test=20):
    def build(_transform):
        return (TensorDataset(torch.zeros(n_train, 3), torch.zeros(n_train, dtype=torch.long)),
                TensorDataset(torch.ones(n_test, 3), torch.zeros(n_test, dtype=torch.long)))
    return build


def test_load_cnn_dataset_val_from_test_keeps_the_whole_train_split(monkeypatch):
    monkeypatch.setitem(utils.DATASET_BUILDERS, "cifar-10", _fake_builder())
    monkeypatch.setattr(utils, "DATALOADER_WORKERS", 0)
    train, val, test = utils.load_cnn_dataset("cifar-10", 0.7, 0.2)
    assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (62, 18, 20)
    assert float(val.dataset[0][0].sum()) == 0.0              # legacy val comes from the train split
    monkeypatch.setenv("SPECTRA_VAL_FROM_TEST", "1")
    train, val, test = utils.load_cnn_dataset("cifar-10", 0.7, 0.2)
    assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (80, 10, 10)
    assert all(float(val.dataset[i][0].sum()) == 3.0 for i in range(len(val.dataset)))
    assert set(val.dataset.indices).isdisjoint(test.dataset.indices)


# ---------------------------------------------------------------- final fine-tune loader

def _fake_cifar(n=12, aug=False):
    ds = datasets.CIFAR10.__new__(datasets.CIFAR10)
    ds.data = np.random.RandomState(0).randint(0, 255, size=(n, 32, 32, 3), dtype=np.uint8)
    ds.targets = list(range(n))
    steps = [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.25,) * 3)]
    if aug:
        steps = [transforms.RandomHorizontalFlip()] + steps
    ds.transform = transforms.Compose(steps)
    ds.target_transform = None
    return ds


def test_final_ft_loader_prepends_crop_flip_on_the_same_images():
    base = _fake_cifar()
    inner = Subset(Subset(base, [5, 6, 7, 8, 9]), [0, 2, 4])
    walk = DataLoader(inner, batch_size=4, shuffle=False)
    loader, aug = utils.final_ft_train_loader(walk, batch_size=2)
    assert aug == "crop+flip" and loader.batch_size == 2
    assert isinstance(loader.dataset, Subset) and loader.dataset.indices == [5, 7, 9]
    steps = loader.dataset.dataset.transform.transforms
    assert isinstance(steps[0], transforms.RandomCrop) and isinstance(steps[1], transforms.RandomHorizontalFlip)
    assert not isinstance(base.transform.transforms[0], transforms.RandomCrop)   # walk loader untouched
    x, y = next(iter(loader))
    assert x.shape == (2, 3, 32, 32) and set(y.tolist()) <= {5, 7, 9}


def test_final_ft_loader_keeps_augmenting_or_non_cifar_loaders():
    walk = DataLoader(_fake_cifar(aug=True), batch_size=4)
    assert utils.final_ft_train_loader(walk) == (walk, "loader")
    loader, aug = utils.final_ft_train_loader(walk, batch_size=3)
    assert aug == "loader" and loader.batch_size == 3
    plain = DataLoader(TensorDataset(torch.zeros(6, 3)), batch_size=2)
    assert utils.final_ft_train_loader(plain) == (plain, "none")


# ---------------------------------------------------------------- size points

def test_eval_size_points_parsing(monkeypatch):
    assert fortify.eval_size_points() == ()
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "param:0.6,0.8,bad,0.8,1.5")
    assert fortify.eval_size_points() == (("param", 0.8), ("param", 0.6))
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "flops:0.39")
    assert fortify.eval_size_points() == (("flop", 0.39),)
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "width:0.5")
    assert fortify.eval_size_points() == ()


def test_select_size_points_first_point_at_or_below_target():
    got = fortify.select_size_points(CURVE, (("param", 0.8), ("param", 0.62), ("param", 0.1)))
    assert list(got) == ["size_param0.80", "size_param0.62", "size_param0.10"]
    assert got["size_param0.80"]["step"] == 3                 # quoted even though val left the band
    assert got["size_param0.62"]["step"] == 7
    assert got["size_param0.10"] is None


def test_final_ft_flags_default_off(monkeypatch):
    assert fortify.eval_final_ft_epochs() == 0
    assert fortify.eval_final_ft_lr() == pytest.approx(0.01)
    assert fortify.eval_final_ft_batch() == 128
    assert not (fortify.eval_final_ft_kd() or fortify.eval_final_ft_origin()
                or fortify.eval_save_traj_models())
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_EPOCHS", "100")
    assert fortify.eval_final_ft_epochs() == 100


# ---------------------------------------------------------------- runner: candidates + final FT

class _Env:
    def __init__(self):
        self.current_model = nn.Linear(1, 1, bias=False)
        self.train_loader = DataLoader(TensorDataset(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long)),
                                       batch_size=2)
        self.val_loader = self.test_loader = self.train_loader
        self.selected_net_path = "net.pt"
        self.data_dict = {"net.pt": (nn.Linear(1, 1, bias=False), None)}
        self.handlers = []

    def create_learning_handler(self, model):
        env = self

        class _Handler:
            def train_model(self, loader, **kwargs):
                env.handlers.append({"model": model, "loader": loader, "kwargs": kwargs,
                                     "env": {k: os.environ.get(k) for k in (
                                         "SPECTRA_FT_OPTIM", "SPECTRA_FT_SGD_LR", "SPECTRA_FT_COSINE",
                                         "SPECTRA_FT_KD", "SPECTRA_FT_SCHEDULE")}})
                with torch.no_grad():
                    model.weight.add_(1000.0)

            def evaluate_model(self, _loader):
                return 0.75

        return _Handler()


def _walk(env, runner, points, size_points=()):
    candidates = {}
    for p in points[1:]:
        with torch.no_grad():
            env.current_model.weight.fill_(float(p["step"]))
        runner._keep_final_ft_candidates(env, p, candidates, size_points, 10.0)
    return candidates


def test_kept_val_best_copy_matches_the_selection_rule():
    import a2c_agent_reinforce_runner as runner
    env = _Env()
    candidates = _walk(env, runner, CURVE, (("param", 0.8),))
    picked = fortify.select_trajectory_points(CURVE, min_param=0.7, tau_pp=10.0)
    assert picked["val_best"]["step"] == 5                    # the late in-band pop, not first exit
    assert candidates["val_best"]["point"]["step"] == 5
    assert float(candidates["val_best"]["model"].weight) == 5.0
    assert candidates["size_param0.80"]["point"]["step"] == 3
    assert float(candidates["size_param0.80"]["model"].weight) == 3.0


def test_run_final_ft_uses_the_fixed_recipe_and_restores_the_walk_env(monkeypatch, capsys):
    import a2c_agent_reinforce_runner as runner
    env = _Env()
    candidates = _walk(env, runner, CURVE, (("param", 0.7),))  # size 0.70 is val_best's step: FT once
    monkeypatch.setenv("SPECTRA_FT_OPTIM", "adam")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_ORIGIN", "1")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_BATCH", "4")
    monkeypatch.delenv("SPECTRA_FT_COSINE", raising=False)
    lines = []
    monkeypatch.setattr(runner.utils, "print_flush", lambda msg: lines.append(str(msg)))
    monkeypatch.setattr(runner.run_recorder, "record", lambda *a, **k: None)
    picked = fortify.select_trajectory_points(CURVE, min_param=0.7, tau_pp=10.0)
    runner._run_final_ft(env, "net.pt", picked, candidates, 100)
    assert [h["env"] for h in env.handlers] == [
        {"SPECTRA_FT_OPTIM": "sgd", "SPECTRA_FT_SGD_LR": "0.01", "SPECTRA_FT_COSINE": "1",
         "SPECTRA_FT_KD": "0", "SPECTRA_FT_SCHEDULE": ""}] * 2    # val_best + origin
    assert env.handlers[0]["kwargs"]["max_epochs"] == 100 and env.handlers[0]["kwargs"]["patience"] == 101
    assert env.handlers[0]["loader"].batch_size == 4
    assert os.environ["SPECTRA_FT_OPTIM"] == "adam" and "SPECTRA_FT_COSINE" not in os.environ
    assert float(candidates["val_best"]["model"].weight) == 5.0          # the kept copy is not trained
    final = [ln for ln in lines if ln.startswith("[eval] TRAJ final_ft")]
    assert any(ln.startswith("[eval] TRAJ final_ft val_best net.pt step=5 | acc 0.800 -> 0.750 (-0.050)")
               and "walk acc 0.701" in ln and "bs=4 aug=none" in ln for ln in final)
    assert any("size_param0.70" in ln and "same point as val_best" in ln for ln in final)
    assert any(ln.startswith("[eval] TRAJ final_ft origin net.pt step=-1") for ln in final)


def test_run_final_ft_drops_a_copy_that_disagrees_with_the_selection(monkeypatch):
    import a2c_agent_reinforce_runner as runner
    env = _Env()
    candidates = _walk(env, runner, CURVE)
    lines = []
    monkeypatch.setattr(runner.utils, "print_flush", lambda msg: lines.append(str(msg)))
    runner._run_final_ft(env, "net.pt", {"val_best": CURVE[2], "origin": CURVE[0]}, candidates, 10)
    assert env.handlers == [] and "val_best" not in candidates
    assert any("dropping the copy" in ln for ln in lines)


# ---------------------------------------------------------------- offline readout

def test_readout_rules_on_a_flat_noisy_band_edge():
    import traj_readout as tr
    assert tr.val_best(CURVE, 10)["step"] == 5
    assert tr.first_exit(CURVE, 10)["step"] == 2
    assert tr.smooth(CURVE, 10, 3)["step"] == 2               # medians at 3..5 sit below -10
    assert tr.edge_count(CURVE, 10, 1.5) == 5                 # steps 2-6
    assert tr.val_best(CURVE, 5)["step"] == 0
    got = tr.size_points(CURVE, tr.parse_sizes("param:0.8,0.6"))
    assert got["size_param0.80"]["step"] == 3 and got["size_param0.60"]["step"] == 7


def test_readout_gap_and_load(tmp_path):
    import json
    import traj_readout as tr
    pts = [_point(-1, 1.0, 0.0), _point(0, 0.9, -8.0, test=-3.0), _point(1, 0.8, -9.0, test=-4.0)]
    mean_gap, worst = tr.gap(pts)
    assert mean_gap == pytest.approx(5.0) and worst == pytest.approx(5.0)
    events = tmp_path / "events"
    events.mkdir()
    (events / "rank0.jsonl").write_text(
        json.dumps({"event": "finetune"}) + "\n"
        + json.dumps({"event": "eval_traj_summary", "network": "/x/r20.pt", "points": pts}) + "\n",
        encoding="utf-8")
    (tmp_path / "run_records.jsonl").write_text((events / "rank0.jsonl").read_text(encoding="utf-8"),
                                                encoding="utf-8")
    rows = tr.load_points(str(tmp_path))
    assert [(r[0], len(r[1])) for r in rows] == [("r20.pt", 3)]     # one walk, not one per file
    assert tr.main([str(tmp_path), "--taus", "10,5", "--sizes", "param:0.85"]) == 0
