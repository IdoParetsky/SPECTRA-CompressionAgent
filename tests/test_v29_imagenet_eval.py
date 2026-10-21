"""
v29 (21 Oct): the ImageNet evaluation line, CPU-only checks on torchvision ResNet-50 and MobileNetV2 at 224.

* ``SPECTRA_VAL_FROM_TRAIN=<n>`` (default off): a class-balanced val held out of the TRAIN split, the whole
  official val as TEST; refused together with ``SPECTRA_VAL_FROM_TEST``; the legacy ImageNet carve is untouched.
* ``SPECTRA_EVAL_TOP5=1`` (default off): top-5 beside the TRAJ lines (``src/eval_topk.py``); off, the runner's points,
  lines and records are exactly as before.
* The pruning path on torchvision Bottleneck (downsample coupling) and InvertedResidual: channel groups, sensitivity,
  the width floor (``SPECTRA_PLAN_MIN_WIDTH=walk``), MACs at the dataset's input shape (224), the rules' plans landing
  at params 0.5 and FLOPs 0.5 / 0.7, the fixed-target landing in both metrics, grid rounding
  (``SPECTRA_ALLOC_GRID_ROUND``), and the plan agent's state scored by the T6-fs policy of record (read-only; a fresh
  policy where the file is absent).
* The two pipeline checkpoints load strictly into the factories where present.

CPU only. Slow (minutes): ResNet-50 plans are measured on real cuts at 224.

    OMP_NUM_THREADS=4 python -m pytest tests/test_v29_imagenet_eval.py -v
"""

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_pruning import _init_static_conf, _row_index_of_layer  # noqa: E402

_init_static_conf()

import torchvision.models as tvm  # noqa: E402
from tests.test_plan_agent import RATES, V10_FLAGS, _randomize_norms  # noqa: E402
from tests.test_v10_fixed_target import _layer_index, _target_env  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, eval_topk, fortify, plan_agent, state_dump  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

SHAPE = (3, 224, 224)
RATES3 = {0: 1.0, 1: 0.9, 2: 0.8}
T6FS_POLICY = "/home/paretsky/scratch_audit/tree_v21/runs/job22886409/plan_agent/policy_latest.pt"
T6FS_STANDARDIZER = "/home/paretsky/scratch_audit/tree_v10/runs/job22156116/snapshots/ep0127/standardizer.pt"
CKPTS = {
    "resnet50": "/home/paretsky/spectra_pretrained_networks/resnet50_imagenet1kv1_torchvision_76.13_25.56.pth",
    "mobilenet_v2": "/home/paretsky/spectra_pretrained_networks/mobilenet-v2_imagenet1kv1_torchvision_71.88_3.505_327.487.pth",
}
PARAMS = {"resnet50": 25557032, "mobilenet_v2": 3504872}
FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_", "SPECTRA_MIN_WIDTH",
                 "SPECTRA_VAL_FROM", "SPECTRA_FT_AUG", "SPECTRA_SPLIT_SEED", "SPECTRA_BATCH_SIZE", "SPECTRA_STATE_",
                 "SPECTRA_BUDGET_IN_STATE", "SPECTRA_GROUP_ONCE", "SPECTRA_STEM_ROWS", "SPECTRA_PROBE_BATCHES",
                 "SPECTRA_STANDARDIZER_PATH", "SPECTRA_ENCODER_DROPOUT", "SPECTRA_FORTIFY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    monkeypatch.setattr(recorder, "issue", lambda *a, **k: None)
    monkeypatch.setattr(utils, "DATALOADER_WORKERS", 0)
    yield


# ------------------------------------------------------------------ SPECTRA_VAL_FROM_TRAIN


def test_val_from_train_flag_parse(monkeypatch):
    assert utils.val_from_train_per_class() == 0
    monkeypatch.setenv("SPECTRA_VAL_FROM_TRAIN", "10")
    assert utils.val_from_train_per_class() == 10
    monkeypatch.setenv("SPECTRA_VAL_FROM_TRAIN", "0")
    assert utils.val_from_train_per_class() == 0
    for bad in ("x", "-1", "2.5"):
        monkeypatch.setenv("SPECTRA_VAL_FROM_TRAIN", bad)
        with pytest.raises(ValueError):
            utils.val_from_train_per_class()


def test_class_balanced_holdout_is_balanced_disjoint_and_seeded():
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 7, 400).tolist() + list(range(7)) * 12
    val, train = utils._class_balanced_holdout(targets, 3, seed=0)
    assert val == sorted(val) and train == sorted(train) and not set(val) & set(train)
    assert sorted(val + train) == list(range(len(targets)))
    assert np.bincount(np.array(targets)[val], minlength=7).tolist() == [3] * 7
    assert utils._class_balanced_holdout(targets, 3, seed=0) == (val, train)
    assert utils._class_balanced_holdout(targets, 3, seed=1)[0] != val
    with pytest.raises(ValueError):
        utils._class_balanced_holdout(targets, 500, seed=0)
    assert utils._targets_of(SimpleNamespace(labels=[1, 2])) == [1, 2]
    with pytest.raises(ValueError):
        utils._targets_of(SimpleNamespace())


class _FakeFolder(datasets.ImageFolder):
    """An ImageFolder-shaped dataset of random-size RGB images (no files): what ``datasets.ImageNet`` looks like."""

    def __init__(self, split, transform, classes=5, per_class=6):
        torch.utils.data.Dataset.__init__(self)
        self.split, self.transform, self.target_transform = split, transform, None
        self.classes = [f"n{c:08d}" for c in range(classes)]
        self.targets = [c for c in range(classes) for _ in range(per_class)]
        self.samples = [(f"{split}/{c}/{i}.JPEG", c) for c in range(classes) for i in range(per_class)]
        self.imgs = self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rng = np.random.default_rng(1000 * (self.split == "val") + index)
        w, h = int(rng.integers(40, 300)), int(rng.integers(40, 260))
        img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        target = self.samples[index][1]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _kinds(t):
    return [type(s).__name__ for s in t.transforms]


def test_load_cnn_dataset_val_from_train_on_an_imagenet_shaped_dataset(monkeypatch):
    calls = []

    def fake_split(canonical, split, transform):
        calls.append((canonical, split, _kinds(transform)))
        return _FakeFolder("train" if split == "train" else "val", transform, per_class=6 if split == "train" else 2)

    monkeypatch.setattr(utils, "_large_spatial_split", fake_split)
    monkeypatch.setenv("SPECTRA_BATCH_SIZE", "5")
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    # the legacy carve (flags off): the three constructions and sizes as before
    train, val, test = utils.load_cnn_dataset("imagenet1k", 0.7, 0.2)
    assert [c[1] for c in calls] == ["train", "val", "train"]
    assert calls[0][2] == ["RandomResizedCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert calls[1][2] == ["Resize", "CenterCrop", "ToTensor", "Normalize"] and calls[2][2] == calls[1][2]
    assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (23, 7, 10)
    assert utils.DatasetRegistry.key_for("imagenet1k") == "imagenet1k"
    calls.clear()
    # the ImageNet line: 2 per class out of train, the whole official val as TEST
    monkeypatch.setenv("SPECTRA_VAL_FROM_TRAIN", "2")
    assert utils.DatasetRegistry.key_for("imagenet1k") == "imagenet1k|valtrain=2"
    train, val, test = utils.load_cnn_dataset("imagenet1k", 0.7, 0.2)
    assert [c[1] for c in calls] == ["train", "val"]                            # the train split is listed once
    assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (20, 10, 10)
    assert isinstance(train.dataset, torch.utils.data.Subset) and isinstance(val.dataset, torch.utils.data.Subset)
    assert train.dataset.dataset is not val.dataset.dataset                     # augmented vs plain copies ...
    assert train.dataset.dataset.samples is val.dataset.dataset.samples         # ... of one file list
    assert _kinds(train.dataset.dataset.transform) == ["RandomResizedCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert _kinds(val.dataset.dataset.transform) == ["Resize", "CenterCrop", "ToTensor", "Normalize"]
    assert _kinds(test.dataset.transform) == ["Resize", "CenterCrop", "ToTensor", "Normalize"]
    val_targets = [val.dataset.dataset.targets[i] for i in val.dataset.indices]
    assert sorted(val_targets) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert not set(val.dataset.indices) & set(train.dataset.indices)
    ref_val, _ref_train = utils._class_balanced_holdout(train.dataset.dataset.targets, 2, 0)
    assert list(val.dataset.indices) == ref_val
    x, y = next(iter(val))
    assert tuple(x.shape) == (5, 3, 224, 224) and y.tolist() == val_targets[:5]
    x, _y = next(iter(train))
    assert tuple(x.shape) == (5, 3, 224, 224)
    assert any(line.startswith("Val from train on imagenet1k: 2 per class held out of the train split -> n_train=20, "
                               "n_val=10, n_test=10 (the whole test split; split_seed=0)") for line in lines)
    assert utils.final_ft_train_loader(train, 5)[1] == "loader"                 # RandomResizedCrop is already there
    loader, aug = utils.final_ft_train_loader(train, 4)
    assert aug == "loader" and loader.batch_size == 4
    monkeypatch.setenv("SPECTRA_SPLIT_SEED", "3")
    train2, val2, _t = utils.load_cnn_dataset("imagenet1k", 0.7, 0.2)
    assert list(val2.dataset.indices) != list(val.dataset.indices)
    monkeypatch.setenv("SPECTRA_FT_AUG_GPU", "1")
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")
    train3, _v, _t = utils.load_cnn_dataset("imagenet1k", 0.7, 0.2)
    assert isinstance(train3, torch.utils.data.DataLoader)
    assert any(line.startswith("SPECTRA_FT_AUG_GPU ignored on imagenet1k: not a fixed-size uint8 dataset") for line in lines)
    monkeypatch.setenv("SPECTRA_VAL_FROM_TEST", "1")
    with pytest.raises(ValueError):
        utils.load_cnn_dataset("imagenet1k", 0.7, 0.2)


# ------------------------------------------------------------------ top-5


def test_topk_correct_on_a_toy_batch():
    logits = torch.tensor([
        [9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # target 0: rank 1
        [1.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # target 0: rank 2
        [5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # target 0: rank 5
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0],   # target 0: rank 6
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],   # target 0: rank 10
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0],   # target 9: rank 1
    ])
    targets = torch.tensor([0, 0, 0, 0, 0, 9])
    assert utils.topk_correct(logits, targets, (1, 5)) == {1: 2, 5: 4}
    assert utils.topk_correct(logits, targets, (1, 2, 6, 10)) == {1: 2, 2: 3, 6: 5, 10: 6}
    assert utils.topk_correct(logits, targets, (5, 50)) == {5: 4, 50: 6}        # k past the class count reads all
    two = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    assert utils.topk_correct(two, torch.tensor([1, 1]), (1, 5)) == {1: 1, 5: 2}


def test_evaluate_topk_matches_a_manual_count():
    torch.manual_seed(0)
    model = nn.Linear(8, 12)
    x, y = torch.randn(50, 8), torch.randint(0, 12, (50,))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=16)
    model.train()
    acc = utils.evaluate_topk(model, loader, "cpu", (1, 5))
    assert model.training                                                        # the mode is put back
    with torch.no_grad():
        logits = model(x)
    order = logits.argsort(dim=1, descending=True)
    top1 = float((order[:, 0] == y).float().mean())
    top5 = float((order[:, :5] == y.view(-1, 1)).any(dim=1).float().mean())
    assert acc[1] == pytest.approx(top1) and acc[5] == pytest.approx(top5) and acc[5] >= acc[1]
    onehot = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, nn.functional.one_hot(y, 12).float()),
                                         batch_size=7)
    assert utils.evaluate_topk(model, onehot, "cpu", (1, 5)) == acc


def _topk_env():
    torch.manual_seed(1)
    origin = nn.Linear(4, 6)
    x, y = torch.randn(24, 4), torch.randint(0, 6, (24,))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)
    current = nn.Linear(4, 6)
    return SimpleNamespace(selected_net_path="net.pt", data_dict={"net.pt": (origin, None)}, current_model=current,
                           conf=SimpleNamespace(device="cpu"), test_loader=loader, val_loader=loader)


def test_eval_topk_is_a_no_op_off_and_adds_fields_on(monkeypatch):
    env = _topk_env()
    point = {"step": 3, "test_acc": 0.5, "val_acc": 0.4}
    assert not eval_topk.enabled()
    assert eval_topk.capture(env, dict(point)) == point and eval_topk.line("val_best", "net.pt", point) is None
    assert eval_topk.final(env, env.current_model, "val_best", "net.pt", point) == {}
    monkeypatch.setenv("SPECTRA_EVAL_TOP5", "1")
    calls = []
    real = utils.evaluate_topk

    def counted(model, loader, device, ks=(1, 5)):
        calls.append(model)
        return real(model, loader, device, ks)

    monkeypatch.setattr(utils, "evaluate_topk", counted)
    got = eval_topk.capture(env, dict(point))
    assert set(got) == set(point) | {"test_top5", "test_origin_top5", "val_top5", "val_origin_top5",
                                     "test_top1_pass", "val_top1_pass"}
    assert len(calls) == 4 and calls[:2] == [env.data_dict["net.pt"][0]] * 2 and calls[2:] == [env.current_model] * 2
    want = real(env.current_model, env.test_loader, "cpu")
    assert got["test_top5"] == want[5] and got["test_top1_pass"] == want[1]
    origin_pt = eval_topk.capture(env, {"step": -1, "test_acc": 0.5})
    assert len(calls) == 4 and origin_pt["test_top5"] == origin_pt["test_origin_top5"]   # the origin read is cached
    line = eval_topk.line("val_best", "net.pt", got)
    assert re.fullmatch(r"\[eval\] TRAJ top5 val_best net\.pt step=3 \| top5 \d\.\d{3} -> \d\.\d{3} \([+-]\d\.\d{3}\) \| "
                        r"val top5 Δacc [+-]\d+\.\d{2} pp \| top1 pass \d\.\d{3} vs TRAJ 0\.500", line), line
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    extras = eval_topk.final(env, env.current_model, "val_best", "net.pt", got)
    assert set(extras) == {"test_final_top5", "val_final_top5", "test_origin_top5", "val_origin_top5",
                           "test_final_top1_pass"}
    assert len(lines) == 1 and lines[0].startswith("[eval] TRAJ final_ft_top5 val_best net.pt step=3 | top5 ")


def test_runner_traj_lines_and_record_gain_top5_only_under_the_flag(monkeypatch):
    import a2c_agent_reinforce_runner as runner
    from tests.test_v9c_traj_models import _Env, _candidates, _quiet
    lines, _issues = _quiet(monkeypatch, runner)
    records = []
    monkeypatch.setattr(runner.run_recorder, "record", lambda kind, **k: records.append(dict(k, kind=kind)))

    class _TrajEnv(_Env):
        def __init__(self):
            super().__init__()
            self.conf = SimpleNamespace(device="cpu")
            self.current_model = nn.Linear(1, 1, bias=False)
            self.last_val_acc, self.original_acc = 0.7, 0.8

        def param_ratio(self):
            return 0.6

        def flops_ratio(self):
            return 0.55

    env = _TrajEnv()
    point = runner._traj_capture(env, 5, 0.8, 0.71, 0.75)
    assert set(point) == {"step", "rate", "param", "flop", "val_acc", "val_origin", "val_dacc_pp", "test_acc",
                          "test_origin", "test_dacc_pp"}
    runner._print_traj_point("val_best", "net.pt", point)
    assert len(lines) == 1 and lines[0].startswith("[eval] TRAJ val_best net.pt step=5 | acc 0.750 -> 0.710 (-0.040)")
    runner._final_ft(env, "net.pt", "val_best", _candidates()["val_best"], 2)
    final = [r for r in records if r["kind"] == "eval_traj_final_ft"]
    assert len(final) == 1 and "test_final_top5" not in final[0]
    assert not any("top5" in ln for ln in lines)
    lines.clear(), records.clear()
    monkeypatch.setenv("SPECTRA_EVAL_TOP5", "1")
    env = _TrajEnv()
    point = runner._traj_capture(env, 5, 0.8, 0.71, 0.75)
    assert "test_top5" in point and point["test_acc"] == 0.71
    runner._print_traj_point("val_best", "net.pt", point)
    assert len(lines) == 2 and lines[0].startswith("[eval] TRAJ val_best net.pt step=5 | acc 0.750 -> 0.710 (-0.040)")
    assert lines[1].startswith("[eval] TRAJ top5 val_best net.pt step=5 | top5 ")
    runner._final_ft(env, "net.pt", "val_best", _candidates()["val_best"], 2)
    final = [r for r in records if r["kind"] == "eval_traj_final_ft"]
    assert len(final) == 1 and {"test_final_top5", "val_final_top5"} <= set(final[0])
    assert sum(ln.startswith("[eval] TRAJ final_ft_top5 val_best net.pt step=5 | top5 ") for ln in lines) == 1
    assert sum(ln.startswith("[eval] TRAJ final_ft val_best net.pt step=5 | acc ") for ln in lines) == 1


# ------------------------------------------------------------------ the torchvision nets at 224


class _Net:
    """A torchvision net's origin read once: groups, plan, cost models at 224."""

    def __init__(self, name, model):
        self.name, self.model = name, model
        self.mwr = ModelWithRows(model)
        self.groups = channel_groups.build_channel_groups(model)
        self.plan = group_sensitivity.group_plan(self.mwr, self.groups)
        self.rows = [row for _g, row in self.plan]
        self.pm = plan_agent.ParamModel(model, self.plan)
        self._fm = None
        self.flops = float(utils.calc_flops(model, SHAPE))

    @property
    def fm(self):
        if self._fm is None:
            self._fm = plan_agent.FlopModel(self.model, self.plan, SHAPE)
        return self._fm

    def cost(self, budget):
        return self.fm if budget == "flops" else self.pm

    def kept(self, model, budget):
        if budget == "flops":
            return utils.calc_flops(model, SHAPE) / self.flops
        return utils.calc_num_parameters(model) / self.pm.total0


def _build(name):
    with pytest.MonkeyPatch.context() as mp:
        for key in list(os.environ):
            if key.startswith("SPECTRA_"):
                mp.delenv(key)
        torch.manual_seed(0)
        model = {"resnet50": tvm.resnet50, "mobilenet_v2": tvm.mobilenet_v2}[name](weights=None).eval()
        _randomize_norms(model)
        return _Net(name, model)


@pytest.fixture(scope="module")
def r50():
    return _build("resnet50")


@pytest.fixture(scope="module")
def mbv2():
    return _build("mobilenet_v2")


def _net(request, name):
    return request.getfixturevalue({"resnet50": "r50", "mobilenet_v2": "mbv2"}[name])


def _batches(n=2, size=2, classes=1000, seed=3):
    gen = torch.Generator().manual_seed(seed)
    return [(torch.randn(size, *SHAPE, generator=gen), torch.randint(0, classes, (size,), generator=gen))
            for _ in range(n)]


def _loader224(n=4, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = torch.utils.data.TensorDataset(torch.randn(n, *SHAPE, generator=g), torch.randint(0, 1000, (n,), generator=g))
    return torch.utils.data.DataLoader(data, batch_size=batch)


@pytest.mark.parametrize("name", ["resnet50", "mobilenet_v2"])
def test_origin_params_flops_and_groups(request, name):
    net = _net(request, name)
    assert utils.calc_num_parameters(net.model) == PARAMS[name] == net.pm.total0
    lo, hi = {"resnet50": (4.05e9, 4.15e9), "mobilenet_v2": (0.295e9, 0.32e9)}[name]
    assert lo < net.flops < hi, net.flops                                       # the pipeline's MACs at 224
    struct = net.fm.struct0                                                     # Conv2d / Linear MACs alone
    want = {"resnet50": 4.09e9, "mobilenet_v2": 0.301e9}[name]
    assert abs(struct - want) / want < 0.01
    assert net.fm.total0 == net.flops and net.fm.probes == len(net.rows)
    print(f"[v29] {name}: params {net.pm.total0} MACs@224 {net.flops / 1e9:.4f}G (structural {struct / 1e9:.4f}G) "
          f"groups {len(net.groups)} planned {len(net.rows)} FlopModel {net.fm.seconds:.1f}s")
    assert all(g.prunable and g.width >= 2 for g, _r in net.plan)
    names = {id(m): n for n, m in net.model.named_modules()}
    streams = [g for g, _r in net.plan if len(g.producers) > 1]
    if name == "resnet50":
        assert len(net.groups) == 39 and len(net.rows) == 37 and len(streams) == 4
        for stage in ("layer1", "layer2", "layer3", "layer4"):
            group = next(g for g in streams if f"{stage}.0.downsample.0" in {names[id(p)] for p in g.producers})
            producers = sorted(names[id(p)] for p in group.producers)
            blocks = len(getattr(net.model, stage))
            assert producers == sorted([f"{stage}.0.downsample.0"] + [f"{stage}.{k}.conv3" for k in range(blocks)])
            consumers = {names[id(ref.module)] for ref in group.consumers}
            nxt = {"layer1": "layer2", "layer2": "layer3", "layer3": "layer4", "layer4": None}[stage]
            if nxt:
                assert {f"{nxt}.0.conv1", f"{nxt}.0.downsample.0"} <= consumers
            else:
                assert "fc" in consumers
            assert {f"{stage}.{k}.conv1" for k in range(1, blocks)} <= consumers
    else:
        assert len(net.groups) == 27 and len(net.rows) == 25 and len(streams) == 5
        depthwise = [g for g, _r in net.plan if g.depthwise]
        assert len(depthwise) == 17                                               # one per inverted residual
        for group in depthwise:
            assert len(group.producers) == 1 and len(group.depthwise) == 1
            assert names[id(group.depthwise[0])].endswith("conv.1.0") or names[id(group.depthwise[0])] == "features.1.conv.0.0"
        for group in streams:
            assert all(names[id(p)].endswith("conv.2") or names[id(p)].endswith("conv.1") for p in group.producers)


@pytest.mark.parametrize("name", ["resnet50", "mobilenet_v2"])
def test_floors_and_decoders_at_the_walk_floor(request, name):
    net = _net(request, name)
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "walk")
        assert fortify.plan_min_width() == 2
        for budget in ("params", "flops"):
            cm = net.cost(budget)
            for kappa in (0.5, 0.7):
                widths, info = plan_agent.scale_decode({r: 1.0 for r in cm.rows}, cm, kappa, min_width=fortify.plan_min_width())
                assert abs(info["kept"] - kappa) <= 0.011 and all(w >= 2 for w in widths.values())
                assert info["kept"] == pytest.approx(cm.kept(widths))
    finally:
        monkeypatch.undo()


def _sens_and_costs(net):
    costs = {}
    sens, _base = group_sensitivity.group_sensitivity(net.model, net.plan, _batches(), SHAPE, costs=costs)
    return sens, costs


@pytest.mark.parametrize("kind", ["sens_cost", "uniform"])
@pytest.mark.parametrize("budget,target", [("params", 0.5), ("flops", 0.5), ("flops", 0.7)])
@pytest.mark.parametrize("name", ["mobilenet_v2", "resnet50"])
def test_rule_plans_land_on_the_budget(request, name, kind, budget, target):
    net = _net(request, name)
    import time
    started = time.perf_counter()
    widths, info = alloc_walk.plan_targets(net.model, _batches(), SHAPE, kind, target, 0.5, 0.1, budget=budget,
                                           min_width=2)
    cut = alloc_walk.cut_to(net.model, net.plan, info["keeps"], SHAPE)
    real = net.kept(cut, budget)
    seconds = time.perf_counter() - started
    print(f"[v29] {name} {kind} {budget} {target}: plan kept x{info['kept']:.4f} real x{real:.4f} "
          f"min width {min(widths.values())} in {seconds:.0f}s")
    assert info["kept"] == pytest.approx(real, abs=1e-9)                        # plan_targets measures the real cut
    assert abs(real - target) <= 0.011, (name, kind, budget, target, real)
    assert alloc_walk.group_widths(cut, net.rows) == widths
    assert all(widths[r] >= min(2, net.pm.widths0[r]) for r in net.rows)
    assert set(widths) == set(net.rows) and info["kind"] == kind
    if kind == "sens_cost":
        assert info["cost_budget"] == budget and set(info["cost"]) == set(net.rows)
    with torch.no_grad():
        out = cut.eval()(torch.randn(1, *SHAPE))
    assert out.shape == (1, 1000) and torch.isfinite(out).all()
    del cut


def _stub_env(model, row_layer, accs, metric):
    env, _seen = _target_env(model, target=1.0, accs=accs)
    env.train_loader, env.val_loader = _loader224(), _loader224(n=2, seed=1)
    env._input_shape_cache = None
    env.selected_net_path = "tv.pth"
    env._dependency_groups = lambda mwr: channel_groups.build_channel_groups(mwr.model)
    mwr = ModelWithRows(model)
    env.row_idx = _row_index_of_layer(mwr, _layer_index(mwr, row_layer)) + 1
    if metric == "flop":
        env.original_flops = utils.calc_flops(model, SHAPE)
    return env


@pytest.mark.parametrize("metric", ["param", "flop"])
@pytest.mark.parametrize("name", ["resnet50", "mobilenet_v2"])
def test_fixed_target_landing_in_both_metrics(request, monkeypatch, name, metric):
    net = _net(request, name)
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    monkeypatch.setenv("SPECTRA_FIXED_TARGET_METRIC", metric)
    import copy
    model = copy.deepcopy(net.model)
    layer = model.layer2[0].conv1 if name == "resnet50" else model.features[5].conv[0][0]
    env = _stub_env(model, layer, accs=[0.80], metric=metric)
    assert env._input_shape() == SHAPE and env._target_metric() == metric
    preview = env.preview_flops_ratio if metric == "flop" else env.preview_param_ratio
    half = preview(0.5)
    assert 0.9 < half < 1.0
    target = 0.5 * (1.0 + half)                                                 # a 0.5 cut of this row passes it
    env.target_keep = target
    rate, landed = env._land_on_target(0.5)
    assert landed is not None and landed <= target + 1e-9 and rate > 0.5 and target - landed < 0.011
    assert env._land_on_target(0.99)[1] is None                                 # a cut that stays above it is left alone
    _state, _reward, done = env.step(0.5)
    kept = env._target_kept()
    assert done and env._target_final is not None and kept <= target + 1e-9 and target - kept < 0.011
    assert env.last_step_outcome["mode"] == "structural"
    print(f"[v29] {name} {metric}: a 0.5 cut narrowed to {rate:.4f}, landed x{kept:.4f} on target x{target:.4f}")


@pytest.mark.parametrize("budget", ["params", "flops"])
@pytest.mark.parametrize("name", ["resnet50", "mobilenet_v2"])
def test_grid_rounding_puts_the_plan_on_the_walk_grid(request, monkeypatch, name, budget):
    net = _net(request, name)
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
    cm = net.cost(budget)
    target = 0.5
    planned, _info = plan_agent.scale_decode({r: 1.0 for r in cm.rows}, cm, target - 0.02, min_width=2)
    walks = alloc_walk.walk_grids(net.model, net.plan, RATES3, 6, net.groups, aims=planned)
    assert set(walks) == set(net.rows)
    stem_row = net.rows[0]
    if name == "resnet50":
        assert walks[stem_row]["grid"] == {net.pm.widths0[stem_row]: 0}          # the stem row is identity-only
    else:
        # MobileNetV2's stem group is also owned by the first block's depthwise conv (row 1, not a stem row), so the
        # walk cuts it there: the group has a grid (the stem row itself stays identity-only).
        assert len(walks[stem_row]["grid"]) > 1 and walks[stem_row]["cap"] == 6
    floors = {row: max(1, int(round(0.1 * w0)), min(w0, 2)) for row, w0 in cm.widths0.items()}
    rounded, report = alloc_walk.grid_round_widths(planned, {r: list(walks[r]["grid"]) for r in planned}, cm.kept,
                                                   target, None, floors=floors)
    assert all(rounded[r] in walks[r]["grid"] for r in planned)
    assert report["kept"] <= target + 1e-9 and report["kept"] >= target - 0.02, report
    assert report["kept"] == pytest.approx(cm.kept(rounded))
    print(f"[v29] {name} {budget} grid round: kept x{report['kept']:.4f} (target x{target}), "
          f"{report['down']} down / {report['up']} up steps")


@pytest.mark.parametrize("name", ["resnet50", "mobilenet_v2"])
def test_plan_agent_state_is_built_and_the_t6fs_policy_scores_it(request, monkeypatch, name):
    from NetworkFeatureExtraction.src.FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
    from src.feature_standardizer import FeatureStandardizer
    net = _net(request, name)
    for key, value in V10_FLAGS.items():
        monkeypatch.setenv(key, value)
    for key, value in (("SPECTRA_ENCODER_DROPOUT", "0"), ("SPECTRA_STEM_ROWS", "1"), ("SPECTRA_GROUP_ONCE_PER_PASS", "1"),
                       ("SPECTRA_PLAN_MIN_WIDTH", "walk"), ("SPECTRA_PROBE_BATCHES", "1")):
        monkeypatch.setenv(key, value)
    policy_path = T6FS_POLICY
    tmp = None
    if not os.path.isfile(policy_path):
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "v29_fresh_policy.pt"
        policy = plan_agent.PlanPolicy(63)
        with torch.no_grad():
            policy.head.weight.normal_()
        plan_agent.save_policy(policy, str(tmp), tokens="layer", zero=[], k_min=0.1, budget="params")
        policy_path = str(tmp)
    if os.path.isfile(T6FS_STANDARDIZER):
        monkeypatch.setenv("SPECTRA_STANDARDIZER_PATH", T6FS_STANDARDIZER)
        FeatureStandardizer.reset_instance()
        FeatureStandardizer.instance(38).load(T6FS_STANDARDIZER)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    import copy
    model = copy.deepcopy(net.model)
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    loader = _loader224(n=2, batch=2)
    sens_features, summary = group_sensitivity.layer_features(model, _batches(), SHAPE)
    assert sens_features.shape == (len(mwr.all_layers), 2) and summary["groups"] == len(net.rows)
    env = SimpleNamespace(conf=SimpleNamespace(device="cpu", compression_rates_dict=RATES), current_model=model,
                          target_keep=0.5, row_idx=1, _layer_sens=sens_features, train_loader=loader,
                          _input_shape=lambda: SHAPE,
                          _dependency_groups=lambda m: channel_groups.build_channel_groups(m.model),
                          feature_extractor=FeatureExtractor(loader, "cpu"))
    state = state_dump.encode_origin(env, mwr, groups)
    feats = state["layer_features"]
    assert feats.shape == (len(mwr.all_layers), 63) and torch.isfinite(feats).all()
    assert state_dump.layout(63, len(RATES))[-1] == ("action", 53, 63)
    widths, info = plan_agent.plan_for_env(env, 0.5, policy_path, 0.1)
    print(f"[v29] {name}: {info['policy']} plan keeps x{info['kept']:.4f} over {len(widths)} groups; "
          f"keeps min {min(info['keeps'].values()):.2f} max {max(info['keeps'].values()):.2f}; "
          f"scores std {float(torch.tensor(list(info['sens'].values())).std()):.3f}")
    assert set(widths) == set(net.rows) and info["kind"] == "agent" and info["min_width"] == 2
    assert abs(info["kept"] - 0.5) <= 0.011 and all(w >= 2 for w in widths.values())
    scores = torch.tensor([info["sens"][r] for r in net.rows])
    assert torch.isfinite(scores).all() and abs(float(scores.sum())) < 1e-3  # centred over groups
    if policy_path == T6FS_POLICY:
        assert float(scores.std()) > 0                                          # a trained head differentiates groups
        assert not any("WARNING" in line for line in lines), lines
    cut = alloc_walk.cut_to(model, net.plan, plan_agent.rates_of(widths, net.pm.widths0), SHAPE)
    assert utils.calc_num_parameters(cut) / net.pm.total0 == pytest.approx(info["kept"], abs=1e-9)
    assert alloc_walk.group_widths(cut, net.rows) == widths
    with torch.no_grad():
        out = cut.eval()(torch.randn(1, *SHAPE))
    assert out.shape == (1, 1000) and torch.isfinite(out).all()
    if tmp is not None and tmp.is_file():
        tmp.unlink()


@pytest.mark.parametrize("name", ["resnet50", "mobilenet_v2"])
def test_pipeline_checkpoints_load_strictly_where_present(name):
    path = CKPTS[name]
    if not os.path.isfile(path):
        pytest.skip(f"{path} is not on this machine")
    script = str(ROOT / "spectra_models_instantiation" / "torchvision_instantiation.py")
    lines = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
        model = utils.load_model_from_script(name, "imagenet1k", script, path, {}, num_classes=1000, input_shape=SHAPE)
    assert utils.calc_num_parameters(model) == PARAMS[name] and not model.training
    head = model.fc if name == "resnet50" else model.classifier[-1]
    assert head.out_features == 1000 and float(head.weight.abs().mean()) > 0
    with torch.no_grad():
        out = model(torch.zeros(1, *SHAPE))
    assert out.shape == (1, 1000)
    groups = channel_groups.build_channel_groups(model)
    assert groups is not None and sum(g.prunable for g in groups) == {"resnet50": 37, "mobilenet_v2": 25}[name]
