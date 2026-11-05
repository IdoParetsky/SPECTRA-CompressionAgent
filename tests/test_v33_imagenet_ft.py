"""
v33 (5 Nov): the ImageNet recovery path (``src/imagenet_fast.py``, ``SPECTRA_IMN_FT``, default off), CPU only.

* flags default off; ``_final_ft`` with the flag off takes the stock path (its line and recipe literal unchanged);
* the node-local pack (built from an ImageFolder-shaped tree, read back byte for byte, reused, refused on the home);
* the loaders: PIL and torchvision decode from the pack, the GPU-decode path run on the CPU (the same crops as the
  worker path), the epoch order and a resumed epoch's batches;
* the recovery loop: the step schedule and weight decay as flags, the per-epoch cap, checkpoints, a forced stop at a
  step and an exact resume (the resumed run's weights equal an uninterrupted run's), the origin-control helper, the
  decision save / restore, the decision timing line.

    OMP_NUM_THREADS=4 python -m pytest tests/test_v33_imagenet_ft.py -v
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import src.imagenet_fast as imn  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.traj_models as traj_models  # noqa: E402
import src.utils as utils  # noqa: E402

PREFIXES = ("SPECTRA_IMN_", "SPECTRA_AMP", "SPECTRA_EVAL_", "SPECTRA_SEED", "SPECTRA_FT_", "SPECTRA_TIME_DECIDE",
            "SLURM_CPUS_PER_TASK", "SLURM_JOB_ID")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for key in list(os.environ):
        if key.startswith(PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    monkeypatch.setattr(recorder, "issue", lambda *a, **k: None)
    monkeypatch.setattr(utils, "DATALOADER_WORKERS", 0)
    monkeypatch.setattr(imn, "_PRESTAGE", {})
    yield


# ------------------------------------------------------------------------------------------------ fixtures


def _tree(tmp_path, classes=3, per_class=4):
    """An ImageFolder-shaped tree of random-size JPEGs (one grayscale, one PNG under a .JPEG name)."""
    root = tmp_path / "train"
    rng = np.random.default_rng(0)
    for c in range(classes):
        d = root / f"n{c:08d}"
        d.mkdir(parents=True)
        for i in range(per_class):
            w, h = int(rng.integers(40, 300)), int(rng.integers(40, 260))
            arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            path = d / f"n{c:08d}_{i}.JPEG"
            if c == 1 and i == 0:
                Image.fromarray(arr[:, :, 0]).save(path, format="JPEG", quality=90)        # grayscale
            elif c == 2 and i == 1:
                Image.fromarray(arr).save(path, format="PNG")                               # a PNG named .JPEG
            else:
                Image.fromarray(arr).save(path, format="JPEG", quality=90)
    return root


def _train_tf():
    return utils.build_transform("imagenet1k", {}, train=True)


class _FlipData(Dataset):
    """(3, 8, 8) floats with a p=0.5 flip from torch's global RNG: the augmentation the per-sample seeding must pin."""

    def __init__(self, n=24, classes=4):
        g = torch.Generator().manual_seed(1)
        self.x = torch.randn(n, 3, 8, 8, generator=g)
        self.y = torch.randint(0, classes, (n,), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        if float(torch.rand(1).item()) < 0.5:
            x = x.flip(-1)
        return x, int(self.y[i])


def _tiny(classes=4):
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(4, classes))


def _point(step=0, param=0.5, flop=0.48):
    return {"step": step, "rate": 0.8, "param": param, "flop": flop, "val_acc": 0.7, "val_origin": 0.75,
            "val_dacc_pp": -5.0, "test_acc": 0.68, "test_origin": 0.76, "test_dacc_pp": -8.0}


class _Env:
    def __init__(self, dataset=None, batch=4, model=None, test_acc=0.71, val_acc=0.69):
        dataset = dataset if dataset is not None else _FlipData()
        self.train_loader = DataLoader(dataset, batch_size=batch, shuffle=True)
        self.val_loader = DataLoader(dataset, batch_size=batch)
        self.test_loader = DataLoader(dataset, batch_size=batch)
        self.conf = SimpleNamespace(device=torch.device("cpu"), seed=42)
        self.selected_net_path = "/zoo/net.pt"
        self.data_dict = {"/zoo/net.pt": (model if model is not None else _tiny(), (self.train_loader, self.val_loader, self.test_loader))}
        self.kd_teacher = None
        self.t_start = time.perf_counter() - 12.5
        self.original_acc = 0.75
        self.last_val_acc = 0.75
        self.current_model = self.data_dict["/zoo/net.pt"][0]
        self._accs = (test_acc, val_acc)
        self.runs = []

    def param_ratio(self):
        return 1.0

    def flops_ratio(self):
        return 1.0

    def score_test_loader(self):
        return 0.76, 0.76, 0.0

    def create_learning_handler(self, model):
        env = self

        class _Handler:
            def train_model(self, loader, **kwargs):
                env.runs.append({"tag": kwargs.get("tag"), "wd": os.environ.get("SPECTRA_FT_WD"),
                                 "optim": os.environ.get("SPECTRA_FT_OPTIM"), "cosine": os.environ.get("SPECTRA_FT_COSINE"),
                                 "epochs": kwargs.get("max_epochs"), "keep_last": kwargs.get("keep_last", "absent")})

            def evaluate_model(self, loader):
                return env._accs[0] if loader is env.test_loader else env._accs[1]

        return _Handler()


def _quiet(monkeypatch, runner, records=None, tmp_path=None):
    lines = []
    monkeypatch.setattr(runner.utils, "print_flush", lambda msg: lines.append(str(msg)))
    monkeypatch.setattr(runner.run_recorder, "record",
                        (lambda kind, **k: records.append(dict(k, kind=kind))) if records is not None else (lambda *a, **k: None))
    monkeypatch.setattr(runner.run_recorder, "issue", lambda *a, **k: None)
    monkeypatch.setattr(runner.logging_utils, "exception", lambda msg: lines.append(f"EXC {msg}"))
    if tmp_path is not None:  # no real run directory: the decision timing reads an absent events file
        monkeypatch.setattr(runner.run_recorder, "recorder",
                            lambda: SimpleNamespace(run_dir=str(tmp_path), path=str(tmp_path / "events.jsonl")))
    return lines


# --------------------------------------------------------------------------------------------------- flags


def test_flags_default_off_and_parse(monkeypatch):
    assert imn.enabled() is False and imn.origin_only() is False and imn.ckpt_dir() == ""
    assert imn.data_path() == "nfs" and imn.milestones() == [30, 60] and imn.gamma() == 0.1
    assert imn.weight_decay() == 1e-4 and imn.momentum() == 0.9 and imn.channels_last() is True
    assert imn.grad_clip() == 0.0 and imn.ckpt_every_steps() == 0 and imn.max_steps_per_epoch() == 0
    assert imn.stop_at() is None and imn.pack_limit() == 0 and imn.prestage_enabled() is True
    monkeypatch.setenv("SPECTRA_IMN_DATA", "tar")
    with pytest.raises(ValueError):
        imn.data_path()
    monkeypatch.setenv("SPECTRA_IMN_DATA", "Pack-GPU")
    assert imn.data_path() == "pack-gpu"
    monkeypatch.setenv("SPECTRA_IMN_FT_MILESTONES", " 60, 30 ;80 ")
    assert imn.milestones() == [30, 60, 80]
    monkeypatch.setenv("SPECTRA_IMN_FT_MILESTONES", "")
    assert imn.milestones() == []
    monkeypatch.setenv("SPECTRA_IMN_STOP_AT", "e0s1000")
    assert imn.stop_at() == (0, 1000)
    monkeypatch.setenv("SPECTRA_IMN_STOP_AT", "1000")
    with pytest.raises(ValueError):
        imn.stop_at()
    monkeypatch.setenv("SPECTRA_IMN_STOP_AT", "e1s5")
    monkeypatch.setenv("SPECTRA_IMN_WORKERS", "3")
    assert imn.workers() == 3
    monkeypatch.delenv("SPECTRA_IMN_WORKERS")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    assert imn.workers() == 10
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    assert imn.workers() == 4
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert imn.workers() == 0  # the fixture's DATALOADER_WORKERS
    monkeypatch.setenv("SPECTRA_IMN_FT", "1")
    monkeypatch.setenv("SPECTRA_IMN_ORIGIN_ONLY", "1")
    assert imn.enabled() and imn.origin_only()
    assert "data=pack-gpu" in imn.summary() and "origin_only=1" in imn.summary() and imn.summary().endswith(",stop=e1s5")
    assert imn.recovery_path("size_flop0.48") == ""
    monkeypatch.setenv("SPECTRA_IMN_CKPT_DIR", "/cells/x")
    assert imn.recovery_path("size flop/0.48") == "/cells/x/recovery__size_flop_0.48.pt"
    assert imn.decision_dir() == "/cells/x/decision" and imn.decision_saved("net.pt") is False


# ------------------------------------------------------------------------------------------- the stock path


def test_final_ft_flag_off_is_the_stock_path(monkeypatch):
    import a2c_agent_reinforce_runner as runner
    records = []
    lines = _quiet(monkeypatch, runner, records)
    monkeypatch.setattr(imn, "recover", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recover called with the flag off")))
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_BATCH", "4")
    env = _Env()
    cand = {"point": _point(), "model": _tiny(), "key": None}
    assert runner._final_ft(env, "/zoo/net.pt", "val_best", cand, 2) == 0.71
    assert len(env.runs) == 1 and env.runs[0]["wd"] == "5e-4" and env.runs[0]["optim"] == "sgd" \
        and env.runs[0]["cosine"] == "1" and env.runs[0]["epochs"] == 2 and env.runs[0]["keep_last"] == "absent"
    traj = [ln for ln in lines if ln.startswith("[eval] TRAJ final_ft val_best net.pt")]
    assert len(traj) == 1
    assert traj[0].startswith("[eval] TRAJ final_ft val_best net.pt step=0 | acc 0.760 -> 0.710 (-0.050) | params x0.500 | "
                              "FLOPs x0.480 | val Δacc -6.00 pp | walk acc 0.680 | sgd lr=0.01 m=0.9 wd=5e-4 cos e2 bs=4 "
                              "aug=none kd=0 init=inherit | ")
    assert "SPECTRA_FT_WD" not in os.environ
    rec = [r for r in records if r["kind"] == "eval_traj_final_ft"]
    assert len(rec) == 1 and rec[0]["schedule"] == "cos" and "imn" not in rec[0] and not any(ln.startswith("[imn]") for ln in lines)


# -------------------------------------------------------------------------------------------------- the pack


def test_scan_tree_matches_imagefolder_order(tmp_path):
    root = _tree(tmp_path)
    rel, targets, classes = imn.scan_tree(str(root))
    folder = datasets.ImageFolder(str(root))
    assert rel == imn.rel_paths_of(folder, imn.train_root_of(folder)) and targets == folder.targets
    assert classes == folder.classes and len(rel) == 12
    assert imn.scan_tree(str(root), limit_classes=2)[0] == rel[:8]
    assert imn.scan_tree(str(root), limit_files=5)[0] == rel[:5]


def test_pack_build_read_reuse_and_refusals(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    rel, targets, _c = imn.scan_tree(str(root))
    stem = str(tmp_path / "local" / "train")
    lines = []
    pack = imn.build_pack(str(root), rel, stem, threads=3, log=lines.append, chunk=5)
    assert pack.n == 12 and os.path.exists(stem + ".pack") and os.path.exists(stem + ".idx.npz")
    for i, r in enumerate(rel):
        assert pack.read(i) == (root / r).read_bytes()
    assert pack.read_many([3, 0, 11]) == [pack.read(3), pack.read(0), pack.read(11)]
    assert pack.bytes == sum(len((root / r).read_bytes()) for r in rel)
    assert any(ln.startswith("[imn] staged train pack: 12 files") for ln in lines)
    lines.clear()
    again = imn.build_pack(str(root), rel, stem, threads=3, log=lines.append)
    assert again.n == 12 and any(ln.startswith("[imn] pack reused") for ln in lines)
    meta = json.load(open(imn.Pack.meta_path(stem), encoding="utf-8"))
    assert meta["done"] and meta["n"] == 12 and meta["digest"] == imn.pack_digest(str(root), rel)
    assert not imn.Pack.complete(stem, 11, meta["digest"]) and not imn.Pack.complete(stem, 12, "other")
    # a pack that would sit on the network home is refused unless SPECTRA_IMN_ALLOW_HOME_PACK=1
    monkeypatch.setattr(imn, "_mount_type", lambda path: "nfs")
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))
    with pytest.raises(RuntimeError):
        imn.build_pack(str(root), rel, str(tmp_path / "other" / "train"), threads=2, log=lines.append)
    monkeypatch.setenv("SPECTRA_IMN_ALLOW_HOME_PACK", "1")
    assert imn.build_pack(str(root), rel, str(tmp_path / "other" / "train"), threads=2, log=lines.append).n == 12
    # no room: refused with the need and the free space in the message
    monkeypatch.setattr(imn.shutil, "disk_usage", lambda d: SimpleNamespace(free=10))
    with pytest.raises(RuntimeError, match="not enough room"):
        imn.build_pack(str(root), rel, str(tmp_path / "third" / "train"), threads=2, log=lines.append)


def test_pack_stem_and_plan_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRA_IMN_LOCAL_DIR", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert imn.pack_stem() == os.path.join(str(tmp_path), "spectra_imn_123", "train")
    root = _tree(tmp_path)
    loader = DataLoader(datasets.ImageFolder(str(root), _train_tf()), batch_size=2)
    src, rel, base, indices = imn._pack_plan(loader)
    assert src == str(root) and len(rel) == 12 and indices is None
    monkeypatch.setenv("SPECTRA_IMN_PACK_LIMIT", "7")
    assert len(imn._pack_plan(loader)[1]) == 7
    monkeypatch.setenv("SPECTRA_IMN_PACK_SRC", "/elsewhere")
    assert imn._pack_plan(loader)[0] == "/elsewhere"


# ------------------------------------------------------------------------------------------- augmentation


def test_rrc_params_within_bounds_and_seeded():
    for height, width in ((375, 500), (40, 300), (300, 40), (224, 224), (1, 1)):
        gen = torch.Generator().manual_seed(3)
        for _ in range(200):
            top, left, h, w = imn.rrc_params(height, width, gen)
            assert 0 < h <= height and 0 < w <= width and 0 <= top <= height - h and 0 <= left <= width - w
    a = imn.rrc_params(375, 500, torch.Generator().manual_seed(9))
    assert a == imn.rrc_params(375, 500, torch.Generator().manual_seed(9))
    assert a != imn.rrc_params(375, 500, torch.Generator().manual_seed(10))
    assert imn.sample_seed(42, 0, 5) != imn.sample_seed(42, 1, 5) != imn.sample_seed(43, 1, 5)
    assert imn.sample_seed(42, 0, 5) == imn.sample_seed(42, 0, 5) and 0 <= imn.sample_seed(1, 2, 3) < 2 ** 62


def test_decode_rgb_handles_jpeg_grayscale_and_png(tmp_path):
    root = _tree(tmp_path)
    rel, _t, _c = imn.scan_tree(str(root))
    for r in rel:
        img = imn.decode_rgb((root / r).read_bytes(), "cpu")
        assert img.dtype == torch.uint8 and img.shape[0] == 3 and img.shape[1] > 0
        ref = np.asarray(Image.open(root / r).convert("RGB"))
        assert tuple(img.shape[1:]) == ref.shape[:2]
    gray = next(r for r in rel if r.startswith("n00000001/n00000001_0"))
    g = imn.decode_rgb((root / gray).read_bytes())
    assert torch.equal(g[0], g[1]) and torch.equal(g[1], g[2])
    out = imn.crop_flip_uint8(g, torch.Generator().manual_seed(0))
    assert out.shape == (3, 224, 224) and out.dtype == torch.uint8
    x = imn.normalize_batch(out.unsqueeze(0), torch.tensor(imn.MEAN).view(1, 3, 1, 1), torch.tensor(imn.STD).view(1, 3, 1, 1))
    assert x.dtype == torch.float32 and x.shape == (1, 3, 224, 224)
    assert imn.normalize_batch(x, None, None) is x


def test_packed_images_tv_and_pil_are_seeded_per_sample_and_epoch(tmp_path):
    root = _tree(tmp_path)
    rel, targets, _c = imn.scan_tree(str(root))
    pack = imn.build_pack(str(root), rel, str(tmp_path / "local" / "train"), threads=2, log=lambda s: None)
    tv = imn.PackedImages(pack, targets, None, "tv", None, seed=42)
    pil = imn.PackedImages(pack, targets, None, "pil", _train_tf(), seed=42)
    assert len(tv) == 12 and len(pil) == 12
    x, y = tv[4]
    assert x.shape == (3, 224, 224) and x.dtype == torch.uint8 and y == targets[4]
    assert torch.equal(x, tv[4][0])                                         # the same epoch: the same crop
    tv.set_epoch(1)
    assert not torch.equal(x, tv[4][0])                                     # another epoch: another crop
    p, py = pil[4]
    assert p.shape == (3, 224, 224) and p.dtype == torch.float32 and py == targets[4]
    assert torch.equal(p, pil[4][0])
    pil.set_epoch(2)
    assert not torch.equal(p, pil[4][0])
    sub = imn.PackedImages(pack, targets, [11, 2, 100], "tv", None, seed=1)   # a limited pack drops what it lacks
    assert len(sub) == 2 and sub[0][1] == targets[11]
    with pytest.raises(ValueError):
        imn.PackedImages(pack, targets, None, "gpu")


def test_epoch_order_and_batches_resume_exactly():
    order = imn.epoch_order(10, 42, 0)
    assert sorted(order.tolist()) == list(range(10)) and order.tolist() == imn.epoch_order(10, 42, 0).tolist()
    assert order.tolist() != imn.epoch_order(10, 42, 1).tolist() and order.tolist() != imn.epoch_order(10, 43, 0).tolist()
    full = list(imn._Batches(order, 3, 0))
    assert [len(b) for b in full] == [3, 3, 3, 1] and sum(full, []) == order.tolist()
    assert list(imn._Batches(order, 3, 2)) == full[2:] and len(imn._Batches(order, 3, 2)) == 2

    class _Index(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, i):
            return torch.tensor(i), i

    data = imn.WorkerData(_Index(), 3, 0, 42, "x")
    assert data.steps_per_epoch == 4 and data.images == 10
    seen = [(step, x.tolist()) for step, x, _y in data.iterate(0, 0)]
    assert [s for s, _ in seen] == [0, 1, 2, 3] and sum((b for _, b in seen), []) == order.tolist()
    tail = [(step, x.tolist()) for step, x, _y in data.iterate(0, 2)]
    assert tail == seen[2:]


def test_gpu_decode_path_on_cpu_matches_the_worker_path(tmp_path):
    root = _tree(tmp_path)
    rel, targets, _c = imn.scan_tree(str(root))
    pack = imn.build_pack(str(root), rel, str(tmp_path / "local" / "train"), threads=2, log=lambda s: None)
    positions = list(range(12))
    gpu = imn.GpuDecodeData(pack, targets, positions, 5, 42, "cpu", threads=2, prefetch=2)
    assert gpu.steps_per_epoch == 3 and gpu.images == 12 and gpu.label == "pack-gpu"
    batches = list(gpu.iterate(0, 0))
    assert [s for s, _x, _y in batches] == [0, 1, 2] and [x.shape[0] for _s, x, _y in batches] == [5, 5, 2]
    tv = imn.PackedImages(pack, targets, positions, "tv", None, seed=42)
    order = imn.epoch_order(12, 42, 0)
    for step, x, y in batches:
        assert x.dtype == torch.uint8 and x.shape[1:] == (3, 224, 224)
        for k, pos in enumerate(order[step * 5:(step + 1) * 5]):
            ref, ref_y = tv[int(pos)]
            assert torch.equal(x[k], ref) and int(y[k]) == ref_y
    resumed = list(gpu.iterate(0, 1))
    assert [s for s, _x, _y in resumed] == [1, 2] and all(torch.equal(a[1], b[1]) for a, b in zip(resumed, batches[1:]))
    assert gpu.state_dict() == {} and gpu.load_state_dict({}) is None
    assert imn.dali_available() is False or imn.dali_available() is True


def test_prestage_runs_in_the_background_and_is_joined(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setenv("SPECTRA_IMN_FT", "1")
    monkeypatch.setenv("SPECTRA_IMN_DATA", "pack-tv")
    monkeypatch.setenv("SPECTRA_IMN_LOCAL_DIR", str(tmp_path / "node"))
    loader = DataLoader(datasets.ImageFolder(str(root), _train_tf()), batch_size=2)
    lines = []
    assert imn.prestage((loader, None, None), log=lines.append) is True
    assert imn.prestage((loader, None, None), log=lines.append) is False        # once per job
    pack = imn.staged_pack(loader, log=lines.append)
    assert pack.n == 12 and any(ln.startswith("[imn] prestage started") for ln in lines) \
        and any(ln.startswith("[imn] prestage joined") for ln in lines)
    monkeypatch.setenv("SPECTRA_IMN_DATA", "nfs")
    monkeypatch.setattr(imn, "_PRESTAGE", {})
    assert imn.prestage((loader, None, None), log=lines.append) is False       # not a pack path


# ------------------------------------------------------------------------------------------------ recovery


def _recover(env, model, label, epochs, lines, **kw):
    return imn.recover(env, model, label, _point(), epochs, kw.pop("lr", 0.01), kw.pop("batch", 4), "net.pt", log=lines.append)


def _weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_recover_schedule_weight_decay_cap_and_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRA_IMN_CKPT_DIR", str(tmp_path / "cell"))
    monkeypatch.setenv("SPECTRA_IMN_FT_MILESTONES", "1,3")
    monkeypatch.setenv("SPECTRA_IMN_FT_GAMMA", "0.5")
    monkeypatch.setenv("SPECTRA_IMN_FT_WD", "0.01")
    monkeypatch.setenv("SPECTRA_IMN_MAX_STEPS_PER_EPOCH", "2")
    lines = []
    out = _recover(_Env(), _tiny(), "size_flop0.48", 4, lines, lr=0.1)
    assert [r["lr"] for r in out["timing"]] == [0.1, 0.05, 0.05, 0.025] and out["timing"][-1]["lr_after"] == 0.025
    assert out["recipe"]["wd"] == 0.01 and out["recipe"]["milestones"] == [1, 3] and out["recipe"]["select"] == "last"
    assert out["steps_per_epoch"] == 2 and all(r["images"] == 8 for r in out["timing"]) and out["epochs_ran"] == 4
    assert out["resumed"] == 0 and out["data"] == "nfs" and out["img_s"] > 0
    assert [ln for ln in lines if ln.startswith("[ckpt] saved epoch ")] and lines[-1].startswith("[imn] recovery summary size_flop0.48 net.pt: 4 epoch rows")
    assert any(ln.startswith("[imn] epoch 1/4 size_flop0.48 | loss ") and "| lr 0.1 -> 0.05 |" in ln for ln in lines)
    assert any("capped from 6" in ln and "wd=0.01 milestones=[1, 3] gamma=0.5" in ln for ln in lines if ln.startswith("[imn] recovery "))
    path = tmp_path / "cell" / "recovery__size_flop0.48.pt"
    assert path.exists() and json.load(open(str(path)[:-3] + ".json"))["epoch"] == 4
    assert imn.recipe_text(out) == "sgd lr=0.1 m=0.9 wd=0.01 step1,3 g0.5 e4 keep=last bs=4 aug=imn:nfs amp=0 cl=0 clip=0 kd=0 init=inherit cap=2"
    # the loader of a (3, 8, 8) float dataset reaches the model as floats (no uint8 normalisation)
    assert not any(ln.startswith("[imn] WARNING") for ln in lines)


def test_recover_without_ckpt_dir_warns_and_runs(monkeypatch):
    lines = []
    out = _recover(_Env(), _tiny(), "val_best", 1, lines)
    assert out["epochs_ran"] == 1 and any(ln.startswith("[imn] WARNING: SPECTRA_IMN_CKPT_DIR unset") for ln in lines)


def test_forced_stop_then_resume_equals_an_uninterrupted_run(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRA_IMN_FT_MILESTONES", "1")
    monkeypatch.setenv("SPECTRA_IMN_CKPT_EVERY_STEPS", "0")

    def run(cell, stop=None, epochs=2):
        monkeypatch.setenv("SPECTRA_IMN_CKPT_DIR", str(tmp_path / cell))
        if stop:
            monkeypatch.setenv("SPECTRA_IMN_STOP_AT", stop)
        else:
            monkeypatch.delenv("SPECTRA_IMN_STOP_AT", raising=False)
        lines = []
        model = _tiny()
        torch.manual_seed(1234)
        out = _recover(_Env(), model, "val_best", epochs, lines)
        return out, lines, _weights(model)

    # A: stops right after the checkpoint at epoch 0, step 2 (a 24-image split at batch 4 = 6 steps per epoch)
    monkeypatch.setenv("SPECTRA_IMN_CKPT_DIR", str(tmp_path / "A"))
    monkeypatch.setenv("SPECTRA_IMN_STOP_AT", "e0s2")
    lines_a = []
    torch.manual_seed(1234)
    with pytest.raises(SystemExit) as stop:
        _recover(_Env(), _tiny(), "val_best", 2, lines_a)
    assert stop.value.code == 0
    assert any(ln.startswith("[ckpt] saved epoch 0 step 2 (") for ln in lines_a)
    assert any(ln.startswith("[ckpt] forced stop at epoch 0 step 2") for ln in lines_a)
    side = json.load(open(tmp_path / "A" / "recovery__val_best.json"))
    assert side["epoch"] == 0 and side["step"] == 2 and side["global_step"] == 2 and side["timing"] == []
    # B: the same cell, no stop: resumes at step 2 and finishes the two epochs
    out_b, lines_b, w_b = run("A")
    assert any(ln.startswith("[ckpt] resumed at epoch 0 step 2 (global step 2, lr 0.01,") for ln in lines_b)
    assert out_b["resumed"] == 1 and out_b["epochs_ran"] == 2 and out_b["timing"][0]["resumed_at"] == 2 \
        and out_b["timing"][0]["images"] == 24 and out_b["timing"][1]["resumed_at"] == 0
    assert any("| resumed at step 2" in ln for ln in lines_b if ln.startswith("[imn] epoch 1/2"))
    assert any("resumed 1 time(s)" in ln for ln in lines_b if ln.startswith("[imn] recovery summary"))
    # C: an uninterrupted run with the same seed: the same weights, bit for bit
    out_c, _lines_c, w_c = run("C")
    assert out_c["resumed"] == 0 and _same(w_b, w_c)
    assert [r["lr"] for r in out_b["timing"]] == [r["lr"] for r in out_c["timing"]] == [0.01, 0.001]
    # D: the finished cell again: nothing to train, the checkpoint says epoch 2
    out_d, lines_d, w_d = run("A")
    assert any(ln.startswith("[ckpt] resumed at epoch 2 step 0") for ln in lines_d) and out_d["epochs_ran"] == 2 \
        and out_d["resumed"] == 2 and _same(w_d, w_c)
    # E: a different architecture refuses the checkpoint
    monkeypatch.setenv("SPECTRA_IMN_CKPT_DIR", str(tmp_path / "A"))
    monkeypatch.delenv("SPECTRA_IMN_STOP_AT", raising=False)
    other = nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.BatchNorm2d(6), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(6, 4))
    with pytest.raises(RuntimeError, match="architecture differs"):
        _recover(_Env(), other, "val_best", 2, [])


def test_mid_epoch_checkpoints_every_n_steps_resume_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRA_IMN_FT_MILESTONES", "")
    monkeypatch.setenv("SPECTRA_IMN_CKPT_EVERY_STEPS", "2")
    monkeypatch.setenv("SPECTRA_IMN_CKPT_DIR", str(tmp_path / "A"))
    monkeypatch.setenv("SPECTRA_IMN_STOP_AT", "e1s4")
    lines = []
    torch.manual_seed(7)
    with pytest.raises(SystemExit):
        _recover(_Env(), _tiny(), "x", 3, lines)
    saved = [ln for ln in lines if ln.startswith("[ckpt] saved epoch")]
    assert [ln.split(" (")[0] for ln in saved] == ["[ckpt] saved epoch 0 step 2", "[ckpt] saved epoch 0 step 4",
                                                   "[ckpt] saved epoch 1", "[ckpt] saved epoch 1 step 2", "[ckpt] saved epoch 1 step 4"]
    monkeypatch.delenv("SPECTRA_IMN_STOP_AT")
    model_b = _tiny()
    torch.manual_seed(7)
    out_b = _recover(_Env(), model_b, "x", 3, lines)
    assert out_b["timing"][1]["resumed_at"] == 4 and out_b["epochs_ran"] == 3
    monkeypatch.setenv("SPECTRA_IMN_CKPT_DIR", str(tmp_path / "C"))
    model_c = _tiny()
    torch.manual_seed(7)
    _recover(_Env(), model_c, "x", 3, [])
    assert _same(_weights(model_b), _weights(model_c))


def test_recover_on_the_pack_paths(tmp_path, monkeypatch):
    root = _tree(tmp_path, classes=4, per_class=3)
    monkeypatch.setenv("SPECTRA_IMN_LOCAL_DIR", str(tmp_path / "node"))
    folder = datasets.ImageFolder(str(root), _train_tf())
    for path in ("pack", "pack-tv", "pack-gpu"):
        monkeypatch.setenv("SPECTRA_IMN_DATA", path)
        monkeypatch.setenv("SPECTRA_IMN_LOCAL_DIR", str(tmp_path / "node" / path))
        monkeypatch.setenv("SPECTRA_IMN_CKPT_DIR", str(tmp_path / f"cell-{path}"))
        monkeypatch.setattr(imn, "_PRESTAGE", {})
        env = _Env(dataset=torch.utils.data.Subset(folder, list(range(1, 12))), batch=4)
        lines = []
        out = _recover(env, _tiny(), "size_flop0.48", 1, lines)
        assert out["data"] == path and out["epochs_ran"] == 1 and out["timing"][0]["images"] == 11
        assert any(ln.startswith(f"[imn] train data: {path} | 11 images | 3 steps of 4 per epoch") for ln in lines)
        assert any(ln.startswith("[imn] staged train pack: 12 files") for ln in lines)


def test_dali_path_refuses_without_the_library(tmp_path, monkeypatch):
    if imn.dali_available():
        pytest.skip("nvidia.dali present")
    root = _tree(tmp_path)
    rel, targets, _c = imn.scan_tree(str(root))
    pack = imn.build_pack(str(root), rel, str(tmp_path / "local" / "train"), threads=2, log=lambda s: None)
    with pytest.raises(RuntimeError, match="nvidia.dali"):
        imn.DaliData(pack, targets, None, 4, 42, "cpu")


# ---------------------------------------------------------------------------------- the runner under the flag


def test_final_ft_flag_on_runs_the_registered_recipe_and_saves_the_decision(tmp_path, monkeypatch):
    import a2c_agent_reinforce_runner as runner
    records = []
    lines = _quiet(monkeypatch, runner, records, tmp_path)
    monkeypatch.setenv("SPECTRA_IMN_FT", "1")
    monkeypatch.setenv("SPECTRA_IMN_CKPT_DIR", str(tmp_path / "cell"))
    monkeypatch.setenv("SPECTRA_IMN_FT_MILESTONES", "1")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_BATCH", "4")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_KD", "1")
    env = _Env()
    cand = {"point": _point(), "model": _tiny(), "key": None}
    assert runner._final_ft(env, "/zoo/net.pt", "size_flop0.48", cand, 2) == 0.71
    assert env.runs == []                                                      # the stock train_model never ran
    assert "SPECTRA_FT_WD" not in os.environ
    traj = [ln for ln in lines if ln.startswith("[eval] TRAJ final_ft size_flop0.48 net.pt")]
    assert len(traj) == 1
    assert traj[0].startswith("[eval] TRAJ final_ft size_flop0.48 net.pt step=0 | acc 0.760 -> 0.710 (-0.050) | params x0.500 | "
                              "FLOPs x0.480 | val Δacc -6.00 pp | walk acc 0.680 | sgd lr=0.01 m=0.9 wd=0.0001 step1 g0.1 e2 "
                              "keep=last bs=4 aug=imn:nfs amp=0 cl=0 clip=0 kd=0 init=inherit | train ")
    assert traj[0].rstrip().endswith(" min") and "over 1 job(s)" in traj[0]
    assert any(ln.startswith("[imn] decision timing net.pt: ") and "total 12." in ln for ln in lines)
    assert any(ln.startswith("[ckpt] decision saved net.pt__size_flop0.48__step0: step 0 params x0.500 FLOPs x0.480") for ln in lines)
    assert any(ln.startswith("[imn] SPECTRA_EVAL_FINAL_FT_KD ignored") for ln in lines)
    assert any(ln.startswith("[imn] TEST read net.pt: 24 images in ") for ln in lines) and any(ln.startswith("[imn] val read net.pt: 24 images") for ln in lines)
    assert any(ln.startswith("[ckpt] saved epoch 2 (") for ln in lines)
    rec = [r for r in records if r["kind"] == "eval_traj_final_ft"]
    assert len(rec) == 1 and rec[0]["schedule"] == "step1" and rec[0]["aug"] == "imn:nfs" and rec[0]["imn"]["wd"] == 1e-4 \
        and rec[0]["imn_resumed"] == 0 and len(rec[0]["imn_epochs"]) == 2 and rec[0]["select"] == "last" and rec[0]["epochs"] == 2
    # the decision round-trips through traj_models: the resume hook finds it and the candidate rebuilds
    assert imn.decision_saved("net.pt") and not imn.decision_saved("other.pt")
    assert imn.restored_line("net.pt").startswith("[ckpt] decision restored from ") and "size_flop0.48 step 0 params x0.500 FLOPs x0.480" in imn.restored_line("net.pt")
    loaded = traj_models.load_candidates(imn.decision_dir(), "net.pt", _tiny())
    assert list(loaded) == ["size_flop0.48"] and loaded["size_flop0.48"]["point"]["flop"] == 0.48
    assert _same(_weights(loaded["size_flop0.48"]["model"]), _weights(cand["model"]))
    # a second call (a resumed job) saves nothing new and resumes the finished recovery
    n_saved = len([ln for ln in lines if ln.startswith("[ckpt] decision saved")])
    assert runner._final_ft(env, "/zoo/net.pt", "size_flop0.48", cand, 2) == 0.71
    assert len([ln for ln in lines if ln.startswith("[ckpt] decision saved")]) == n_saved
    assert any(ln.startswith("[ckpt] resumed at epoch 2 step 0") for ln in lines)
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH", "only")
    with pytest.raises(ValueError, match="SCRATCH"):
        runner._final_ft(env, "/zoo/net.pt", "size_flop0.48", cand, 2, scratch=True)


def test_origin_only_helper_recovers_the_unpruned_network(tmp_path, monkeypatch):
    import a2c_agent_reinforce_runner as runner
    lines = _quiet(monkeypatch, runner, None, tmp_path)
    monkeypatch.setenv("SPECTRA_IMN_FT", "1")
    monkeypatch.setenv("SPECTRA_IMN_ORIGIN_ONLY", "1")
    monkeypatch.setenv("SPECTRA_IMN_CKPT_DIR", str(tmp_path / "origin"))
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_BATCH", "4")
    env = _Env()
    runner._final_ft_origin_only(env, "/zoo/net.pt", 1)
    assert any(ln.startswith("[eval] TRAJ origin net.pt step=-1 | acc 0.760 -> 0.760 (+0.000) | params x1.000 | FLOPs x1.000") for ln in lines)
    assert any(ln.startswith("[imn] origin control net.pt: no walk") for ln in lines)
    assert any(ln.startswith("[eval] TRAJ final_ft origin net.pt step=-1 | acc 0.760 -> 0.710 (-0.050) | params x1.000 | FLOPs x1.000") for ln in lines)
    assert not imn.decision_saved("net.pt") and (tmp_path / "origin" / "recovery__origin.pt").exists()
    runner._final_ft_origin_only(env, "/zoo/net.pt", 0)
    assert any(ln.startswith("[imn] SPECTRA_EVAL_FINAL_FT_EPOCHS is 0") for ln in lines)


def test_decision_timing_from_the_recorded_stages(tmp_path):
    events = tmp_path / "rank0.jsonl"
    rows = [{"event": "stage", "stage": "reset.group_sensitivity", "seconds": 90.5},
            {"event": "stage", "stage": "reset.feature_extraction", "seconds": 40.0},
            {"event": "stage", "stage": "step.decide", "seconds": 300.25}, {"event": "stage", "stage": "step.decide", "seconds": 1.0},
            {"event": "stage", "stage": "step.prune", "seconds": 8.0}, {"event": "other", "x": 1}, {"event": "stage"}]
    events.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    env = SimpleNamespace(t_start=time.perf_counter() - 600.0)
    t = imn.decision_timing(env, str(events))
    assert t["sensitivities"] == 90.5 and t["state"] == 40.0 and t["plan"] == 301.25 and t["cut"] == 8.0 and t["origin val"] == 0.0
    assert 599.0 < t["total"] < 602.0 and abs(t["other"] - (t["total"] - 439.75)) < 1e-6
    line = imn.timing_line("net.pt", t)
    assert line.startswith("[imn] decision timing net.pt: sensitivities 90.5 s | state 40.0 s | plan 301.2 s | cut 8.0 s | other ")
    assert line.endswith("min)") and "origin val" not in line
    assert imn.timing_line("net.pt", {}) == "[imn] decision timing net.pt: no stage recorded"
    assert imn.decision_timing(None, str(tmp_path / "missing.jsonl")) == {label: 0.0 for _k, label in imn._STAGE_LABELS}
