"""
v29 (21 Oct, wave 3): Tiny-ImageNet-200 in the pipeline, the three 64-px factories and the pretraining script.

* ``utils.TinyImageNet`` reads the standard layout (train 100k / labelled val 10k = SPECTRA's test), classes in
  sorted ``wnids.txt`` order, a deterministic file order, RGB decoding; the registry, ``SPECTRA_VAL_FROM_TEST`` (the
  same ``_split_held_out`` carve as CIFAR: 5k val / 5k test) and ``SPECTRA_FT_AUG`` (RandomCrop(64, pad 8) + flip,
  the 64-px analogue of CIFAR's recipe and exactly what ``scripts/train_pretrained_checkpoint.py`` trains with) work
  as for CIFAR, and ``SPECTRA_FT_AUG_GPU`` holds the split on the device with pad 8.
* ``spectra_models_instantiation/tiny_imagenet_w3.py``: ResNet-18 (CIFAR stem), VGG-16-BN (2x2 head), MobileNetV2 x1
  (stride-1 stem) forward a 64-px batch, expose prunable channel groups and survive a uniform structural keep.
* CIFAR / SVHN / Fashion-MNIST are unchanged with the new flags off: their transforms, registry keys, splits, the GPU
  loader's pad-4 crops and the final-FT loader labels are pinned here as they were.

CPU only. Real-dataset tests skip where ``/home/paretsky/spectra_datasets/tiny-imagenet-200`` is absent.

    OMP_NUM_THREADS=4 python -m pytest tests/test_v29_tiny_imagenet.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402
from src.NetworkEnv import prune_current_model  # noqa: E402
from spectra_models_instantiation import tiny_imagenet_w3 as w3  # noqa: E402

REAL_ROOT = "/home/paretsky/spectra_datasets"
REAL_TINY = os.path.join(REAL_ROOT, utils.TINY_IMAGENET_DIR, "wnids.txt")
TINY_MEAN, TINY_STD = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
FLAGS = ("SPECTRA_FT_AUG", "SPECTRA_FT_AUTOAUG", "SPECTRA_FT_AUG_GPU", "SPECTRA_FT_AUG_HOLDOUT", "SPECTRA_VAL_FROM_TEST",
         "SPECTRA_VAL_FROM_TRAIN", "SPECTRA_SPLIT_SEED", "SPECTRA_VAL_TEST_FRACTION", "SPECTRA_BATCH_SIZE",
         "SPECTRA_PROBE_BATCHES")
FMNIST = {"name": "fashion-mnist", "image_size": 32, "to_rgb": True}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in FLAGS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    monkeypatch.setattr(recorder, "issue", lambda *a, **k: None)
    monkeypatch.setattr(utils, "DATALOADER_WORKERS", 0)
    yield


def _kinds(t):
    return [type(s).__name__ for s in t.transforms]


def _colour(index):
    return ((37 * index) % 256, (91 * index + 20) % 256, (151 * index + 7) % 256)


def _write_tiny(root: Path, wnids, n_train=4, n_val=6):
    """A synthetic Tiny-ImageNet layout: solid-colour 64x64 JPEGs (colour = sample index), one stored grayscale."""
    base = root / utils.TINY_IMAGENET_DIR
    (base / "test" / "images").mkdir(parents=True)
    (base / "wnids.txt").write_text("\n".join(wnids) + "\n", encoding="utf-8")  # deliberately unsorted
    (base / "words.txt").write_text("\n".join(f"{w}\tthing {w}" for w in wnids) + "\n", encoding="utf-8")
    order = sorted(wnids)
    i = 0
    for wnid in order:
        folder = base / "train" / wnid / "images"
        folder.mkdir(parents=True)
        for j in reversed(range(n_train)):  # written in reverse so the listing order is not the sample order
            index = order.index(wnid) * n_train + j
            img = Image.new("RGB", (64, 64), _colour(index))
            if index == 1:
                img = img.convert("L")  # a grayscale file: the loader must still give 3 channels
            img.save(folder / f"{wnid}_{j}.JPEG", quality=100, subsampling=0)
            i += 1
    folder = base / "val" / "images"
    folder.mkdir(parents=True)
    rows = []
    for k in range(n_val):
        wnid = order[k % len(order)]
        Image.new("RGB", (64, 64), _colour(100 + k)).save(folder / f"val_{k}.JPEG", quality=100, subsampling=0)
        rows.append(f"val_{k}.JPEG\t{wnid}\t0\t0\t63\t63")
    rows.reverse()  # annotations listed backwards: the sample order must still be val_0 .. val_{n-1}
    (base / "val" / "val_annotations.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    return _write_tiny(tmp_path_factory.mktemp("tiny"), ["n02", "n01", "n03"])


def _close_colour(x, colour, tol=6):
    """A normalised (3, 64, 64) tensor from the plain eval transform matches the solid colour within JPEG error."""
    mean = torch.tensor(TINY_MEAN).view(3, 1, 1)
    std = torch.tensor(TINY_STD).view(3, 1, 1)
    pixel = (x * std + mean) * 255.0
    want = torch.tensor(colour, dtype=torch.float32).view(3, 1, 1)
    return bool((pixel - want).abs().max() <= tol)


# ------------------------------------------------------------------ names, stats, builders


def test_aliases_stats_and_builders():
    for alias in ("tiny-imagenet", "tinyimagenet", "tiny_imagenet", "tiny-imagenet-200", "Tiny_ImageNet_200", "TinyImageNet"):
        assert utils.canonical_dataset_name(alias) == "tiny-imagenet"
    assert utils.DATASET_STATS["tiny-imagenet"] == (TINY_MEAN, TINY_STD)
    assert "tiny-imagenet" in utils.DATASET_BUILDERS and utils._FT_AUG_64PX == ("tiny-imagenet",)
    assert "tiny-imagenet" not in utils._FT_AUG_DATASETS and "tiny-imagenet" not in utils._LARGE_SPATIAL
    assert utils._dataset_from_net_path("resnet18_tinyimagenet_spectra-w3_60.00_11.3_2200.pt") == "tiny-imagenet"
    assert utils._dataset_from_net_path("resnet50_imagenet1kv1_torchvision_76.13_25.56.pth") == "imagenet"
    assert utils._dataset_from_net_path("vgg16_bn_cifar100_x.pt") == "cifar-100"
    import train_pretrained_checkpoint as tpc
    assert tpc.dataset_token("tiny-imagenet") == "tinyimagenet" and tpc.dataset_token("cifar-10") == "cifar10"
    assert tpc.crop_recipe("tiny-imagenet") == (64, 8) and tpc.crop_recipe("cifar-10") == (32, 4)


def test_synthetic_layout_order_labels_and_rgb(tiny_root):
    train = utils.TinyImageNet(tiny_root, "train")
    assert train.classes == ["n01", "n02", "n03"] and len(train) == 12
    assert train.targets == [0] * 4 + [1] * 4 + [2] * 4
    assert [os.path.basename(p) for p, _t in train.samples[:4]] == [f"n01_{j}.JPEG" for j in range(4)]
    assert [os.path.basename(p) for p, _t in train.samples[8:]] == [f"n03_{j}.JPEG" for j in range(4)]
    img, target = train[5]
    assert isinstance(img, Image.Image) and img.mode == "RGB" and img.size == (64, 64) and target == 1
    gray, _t = train[1]  # stored grayscale, read as RGB
    assert gray.mode == "RGB" and gray.getbands() == ("R", "G", "B")
    eval_tf = utils.build_transform("tiny-imagenet", {})
    train_eval = utils.TinyImageNet(tiny_root, "train", transform=eval_tf)
    x, _t = train_eval[7]
    assert tuple(x.shape) == (3, 64, 64) and _close_colour(x, _colour(7))
    val = utils.TinyImageNet(tiny_root, "val", transform=eval_tf)
    assert len(val) == 6 and val.targets == [0, 1, 2, 0, 1, 2]
    assert [os.path.basename(p) for p, _t in val.samples] == [f"val_{k}.JPEG" for k in range(6)]
    x, target = val[4]
    assert target == 1 and _close_colour(x, _colour(104))
    data = train.data
    assert data.shape == (12, 64, 64, 3) and data.dtype == np.uint8 and train.data is data
    assert np.abs(data[3].astype(int) - np.array(_colour(3))).max() <= 6
    with pytest.raises(ValueError):
        utils.TinyImageNet(tiny_root, "test")
    with pytest.raises(FileNotFoundError):
        utils.TinyImageNet(tiny_root / "nowhere", "train")


def test_builders_registry_key_and_num_classes(tiny_root, monkeypatch):
    monkeypatch.setattr(utils, "SPECTRA_DATASETS", str(tiny_root))
    train, val = utils.DATASET_BUILDERS["tiny-imagenet"](utils.build_transform("tiny-imagenet", {}))
    assert (len(train), len(val)) == (12, 6) and utils.infer_num_classes(train) == 3
    assert utils.DatasetRegistry.key_for("tiny-imagenet") == "tiny-imagenet"
    assert utils.DatasetRegistry.key_for("tinyimagenet") == "tiny-imagenet"
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")
    assert utils.DatasetRegistry.key_for("tiny-imagenet") == "tiny-imagenet|aug=1"
    assert not utils.gpu_ft_aug("tiny-imagenet")
    monkeypatch.setenv("SPECTRA_FT_AUG_GPU", "1")
    assert utils.gpu_ft_aug("tiny-imagenet") and not utils.gpu_ft_aug("tiny-imagenet", {"image_size": 32})
    assert utils.DatasetRegistry.key_for("tiny-imagenet") == "tiny-imagenet|aug=1|gpuaug=1"
    monkeypatch.setenv("SPECTRA_FT_AUTOAUG", "1")
    assert not utils.gpu_ft_aug("tiny-imagenet")


# ------------------------------------------------------------------ transforms: crop 64 pad 8, parity with pretraining


def test_transforms_crop64_pad8_and_pretraining_parity(monkeypatch):
    plain = utils.build_transform("tiny-imagenet", {}, train=True)
    assert _kinds(plain) == ["ToTensor", "Normalize"]                       # SPECTRA_FT_AUG off: no augmentation
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")
    train_t = utils.build_transform("tiny-imagenet", {}, train=True)
    eval_t = utils.build_transform("tiny-imagenet", {}, train=False)
    assert _kinds(train_t) == ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert _kinds(eval_t) == ["ToTensor", "Normalize"]
    crop = train_t.transforms[0]
    assert crop.size == (64, 64) and crop.padding == 8
    norm = train_t.transforms[-1]
    assert tuple(norm.mean) == TINY_MEAN and tuple(norm.std) == TINY_STD
    import train_pretrained_checkpoint as tpc
    theirs, their_eval = tpc.train_transform("tiny-imagenet")
    assert _kinds(theirs) == _kinds(train_t) and _kinds(their_eval) == _kinds(eval_t)
    for a, b in zip(theirs.transforms, train_t.transforms):
        if isinstance(a, transforms.RandomCrop):
            assert (a.size, a.padding) == (b.size, b.padding)
        if isinstance(a, transforms.Normalize):
            assert list(a.mean) == list(b.mean) and list(a.std) == list(b.std)
    cifar_t = utils.build_transform("cifar-10", {}, train=True)                # CIFAR keeps crop 32 pad 4
    assert (cifar_t.transforms[0].size, cifar_t.transforms[0].padding) == ((32, 32), 4)
    theirs_c, _e = tpc.train_transform("cifar-10")
    assert (theirs_c.transforms[0].size, theirs_c.transforms[0].padding) == ((32, 32), 4)


# ------------------------------------------------------------------ splits and loaders


def _reference_split(n, seed=0, share=0.5):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    k = int(round(n * share))
    return perm[:k], perm[k:]


def test_load_cnn_dataset_tiny_splits(tiny_root, monkeypatch):
    monkeypatch.setattr(utils, "SPECTRA_DATASETS", str(tiny_root))
    monkeypatch.setenv("SPECTRA_BATCH_SIZE", "4")
    train, val, test = utils.load_cnn_dataset("tiny-imagenet", 0.7, 0.2)       # flags off: the legacy carve
    assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (9, 3, 6)
    assert _kinds(test.dataset.transform) == ["ToTensor", "Normalize"]
    x, y = next(iter(test))
    assert tuple(x.shape) == (4, 3, 64, 64) and y.tolist() == [0, 1, 2, 0]
    monkeypatch.setenv("SPECTRA_VAL_FROM_TEST", "1")
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    train, val, test = utils.load_cnn_dataset("tiny-imagenet", 0.7, 0.2)
    assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (12, 3, 3)
    assert isinstance(train.dataset, utils.TinyImageNet)                      # the whole train split, augmented
    assert _kinds(train.dataset.transform) == ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    val_idx, test_idx = _reference_split(6)
    assert list(val.dataset.indices) == val_idx and list(test.dataset.indices) == test_idx
    assert _kinds(val.dataset.dataset.transform) == ["ToTensor", "Normalize"]
    assert any(line.startswith("FT aug on tiny-imagenet: RandomCrop(64, pad 8)+Flip on train only (n_train=12, n_val=3)")
               for line in lines)
    assert any(line.startswith("Val from test on tiny-imagenet: n_train=12 (whole train split), n_val=3, n_test=3")
               for line in lines)
    loader, aug = utils.final_ft_train_loader(train, 4)
    assert aug == "loader" and loader.batch_size == 4
    monkeypatch.setenv("SPECTRA_FT_AUG_GPU", "1")
    train, val, test = utils.load_cnn_dataset("tiny-imagenet", 0.7, 0.2)
    assert isinstance(train, utils.GpuCropFlipLoader) and train.pad == 8 and len(train) == 3
    assert any("RandomCrop(64, pad 8)+Flip on the GPU" in line for line in lines)
    batches = list(train)
    assert len(batches) == 3 and all(tuple(x.shape) == (4, 3, 64, 64) for x, _y in batches)
    assert sorted(int(v) for x, y in batches for v in y) == [0] * 4 + [1] * 4 + [2] * 4
    assert train._x.shape == (12, 3, 80, 80)
    loader, aug = utils.final_ft_train_loader(train, 6)
    assert aug == "loader" and isinstance(loader, utils.GpuCropFlipLoader) and loader.pad == 8 and loader.batch_size == 6
    with pytest.raises(ValueError):                                             # one val source only
        monkeypatch.setenv("SPECTRA_VAL_FROM_TRAIN", "1")
        utils.load_cnn_dataset("tiny-imagenet", 0.7, 0.2)


def test_gpu_loader_pad8_matches_torchvision_crop_and_flip(tiny_root):
    base = utils.TinyImageNet(tiny_root, "train")
    loader = utils.GpuCropFlipLoader(base, 4, TINY_MEAN, TINY_STD, device=torch.device("cpu"), pad=8)
    loader._materialise()
    assert loader._x.shape == (12, 3, 80, 80) and loader._x.dtype == torch.uint8
    oy = torch.tensor([0, 16, 8, 3, 15, 11, 1, 7, 12, 4, 9, 16])
    ox = torch.tensor([16, 0, 8, 13, 2, 5, 10, 6, 14, 1, 15, 0])
    flip = torch.tensor([False, True, False, True, True, False, True, False, False, True, True, False])
    x = utils.crop_flip_batch(loader._x, oy, ox, flip, 64).float().div_(255.0)
    x = (x - loader._m) / loader._s
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(TINY_MEAN, TINY_STD)])
    for i in range(12):
        img = Image.fromarray(base.data[i])
        img = TF.crop(TF.pad(img, 8, fill=0), int(oy[i]), int(ox[i]), 64, 64)
        if flip[i]:
            img = TF.hflip(img)
        assert torch.allclose(x[i], norm(img), atol=1e-6), i
    torch.manual_seed(0)
    seen = []
    for _ in range(40):
        for xb, _yb in loader:
            seen.append(xb)
    assert all(tuple(b.shape) == (4, 3, 64, 64) for b in seen)


def test_final_ft_loader_adds_crop64_when_the_loader_is_plain(tiny_root):
    plain = utils.TinyImageNet(tiny_root, "train", transform=utils.build_transform("tiny-imagenet", {}))
    loader = torch.utils.data.DataLoader(plain, batch_size=4, shuffle=True)
    out, aug = utils.final_ft_train_loader(loader, 4)
    assert aug == "crop+flip" and out is not loader
    steps = out.dataset.transform.transforms
    assert [type(s).__name__ for s in steps] == ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert steps[0].size == (64, 64) and steps[0].padding == 8
    assert plain.transform.transforms[0].__class__.__name__ == "ToTensor"         # the original is untouched


@pytest.mark.skipif(not os.path.isfile(REAL_TINY), reason="Tiny-ImageNet is not on this machine")
def test_real_tiny_imagenet_reads_as_expected(monkeypatch):
    monkeypatch.setattr(utils, "SPECTRA_DATASETS", REAL_ROOT)
    eval_tf = utils.build_transform("tiny-imagenet", {})
    train = utils.TinyImageNet(REAL_ROOT, "train", transform=eval_tf)
    val = utils.TinyImageNet(REAL_ROOT, "val", transform=eval_tf)
    assert len(train) == 100000 and len(val) == 10000 and len(train.classes) == 200
    assert train.classes == sorted(train.classes) and utils.infer_num_classes(train) == 200
    counts = np.bincount(np.array(train.targets), minlength=200)
    assert counts.min() == counts.max() == 500
    assert np.bincount(np.array(val.targets), minlength=200).min() == 50
    for i in (0, 12345, 99999):
        x, y = train[i]
        path = train.samples[i][0]
        assert tuple(x.shape) == (3, 64, 64) and train.classes[y] == os.path.basename(path).split("_")[0]
    x, y = val[9999]
    assert tuple(x.shape) == (3, 64, 64) and 0 <= y < 200
    monkeypatch.setenv("SPECTRA_VAL_FROM_TEST", "1")
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")
    monkeypatch.setenv("SPECTRA_BATCH_SIZE", "256")
    tr, va, te = utils.load_cnn_dataset("tiny-imagenet", 0.7, 0.2)
    assert (len(tr.dataset), len(va.dataset), len(te.dataset)) == (100000, 5000, 5000)
    val_idx, test_idx = _reference_split(10000)
    assert list(va.dataset.indices) == val_idx and list(te.dataset.indices) == test_idx
    assert not set(va.dataset.indices) & set(te.dataset.indices)
    assert utils.get_input_shape(va) == (3, 64, 64)
    monkeypatch.setenv("SPECTRA_FT_AUG_GPU", "1")
    tr, _va, _te = utils.load_cnn_dataset("tiny-imagenet", 0.7, 0.2)
    assert isinstance(tr, utils.GpuCropFlipLoader) and tr.pad == 8 and len(tr) == 391   # 100000 / 256, not decoded


# ------------------------------------------------------------------ the wave-3 factories


def _count(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("name", w3.FACTORIES)
def test_factory_forwards_64px_and_resolves_by_alias(name):
    torch.manual_seed(0)
    fn, resolved = utils.resolve_instantiation_func(w3, name)
    assert resolved == name
    model = fn(num_classes=200, large_input=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 200)
    flops = utils.calc_flops(model, (3, 64, 64))
    params = _count(model)
    print(f"[w3] {name}: params {params / 1e6:.3f}M MACs@64 {flops / 1e6:.1f}M")
    lo, hi = {"resnet18": (11.0e6, 11.5e6), "vgg16_bn": (15.0e6, 16.5e6), "mobilenet_v2": (2.3e6, 2.6e6)}[name]
    assert lo < params < hi
    # the spellings a checkpoint filename carries (``arch.replace("_", "-")``) and the pool's ``mobilenet_v2x1``
    for alias in {"resnet18": ("resnet18",), "vgg16_bn": ("vgg16-bn",),
                  "mobilenet_v2": ("mobilenet-v2", "mobilenet_v2x1", "mobilenet-v2x1")}[name]:
        fn2, _r = utils.resolve_instantiation_func(w3, alias)
        assert _count(fn2(num_classes=200, large_input=False)) == params
    model10 = fn(num_classes=10, large_input=False)
    assert model10(torch.randn(1, 3, 64, 64)).shape == (1, 10)


def test_factory_geometry():
    r18 = w3.resnet18(200, False)
    assert isinstance(r18.maxpool, torch.nn.Identity) and r18.conv1.kernel_size == (3, 3) and r18.conv1.stride == (1, 1)
    feats = torch.nn.Sequential(r18.conv1, r18.bn1, r18.relu, r18.maxpool, r18.layer1, r18.layer2, r18.layer3, r18.layer4)
    assert tuple(feats(torch.zeros(1, 3, 64, 64)).shape) == (1, 512, 8, 8)
    vgg = w3.vgg16_bn(200, False)
    assert tuple(vgg.features(torch.zeros(1, 3, 64, 64)).shape) == (1, 512, 2, 2)
    assert vgg.classifier[0].in_features == 2048 and vgg.classifier[-1].out_features == 200
    mb = w3.mobilenet_v2(200, False)
    assert tuple(mb.features(torch.zeros(1, 3, 64, 64)).shape) == (1, 1280, 4, 4)
    assert mb.features[0][0].stride == (1, 1) and mb.classifier[-1].in_features == 1280


@pytest.mark.parametrize("name", w3.FACTORIES)
def test_factory_prunes_structurally_at_uniform_half(name):
    torch.manual_seed(0)
    model = getattr(w3, name)(num_classes=200, large_input=False).eval()
    groups = channel_groups.build_channel_groups(model)
    assert groups is not None
    mwr = ModelWithRows(model)
    plan = group_sensitivity.group_plan(mwr, groups)
    assert len(plan) >= 10 and all(g.prunable for g, _r in plan)
    before = _count(model)
    shape = (3, 64, 64)
    for _group, row in plan:
        mwr = prune_current_model(mwr, 0.5, row, quiet=True, record=False, input_shape=shape, importance="l1")
        assert mwr.last_prune_outcome["mode"] == "structural", (name, row, mwr.last_prune_outcome)
        mwr = ModelWithRows(mwr.model)
    after = _count(mwr.model)
    assert 0.15 < after / before < 0.45, (name, after / before)
    with torch.no_grad():
        out = mwr.model.eval()(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 200) and torch.isfinite(out).all()


# ------------------------------------------------------------------ the pretraining script


def test_scheduler_and_loss_knobs():
    import train_pretrained_checkpoint as tpc
    from types import SimpleNamespace
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = tpc.make_scheduler(opt, SimpleNamespace(schedule="multistep", milestones=[100, 150], epochs=200,
                                                    warmup_epochs=0))
    assert isinstance(sched, torch.optim.lr_scheduler.MultiStepLR)              # the old default, untouched
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = tpc.make_scheduler(opt, SimpleNamespace(schedule="cosine", milestones=[], epochs=10, warmup_epochs=2))
    lrs = []
    for _epoch in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.05) and lrs[1] == pytest.approx(0.1) and lrs[2] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:])) and lrs[-1] < 0.01


def test_pretrain_smoke_runs_two_steps_on_cpu_and_writes_nothing(tiny_root, tmp_path, monkeypatch, capsys):
    import train_pretrained_checkpoint as tpc
    monkeypatch.setattr(utils, "SPECTRA_DATASETS", str(tiny_root))
    ckpt_root, manifest = tmp_path / "ckpts", tmp_path / "manifest.json"
    argv = ["train_pretrained_checkpoint.py", "--arch", "mobilenet_v2",
            "--script", str(ROOT / "spectra_models_instantiation" / "tiny_imagenet_w3.py"),
            "--dataset", "tiny-imagenet", "--source", "unit", "--select", "last",
            "--epochs", "100", "--batch-size", "4", "--lr", "0.1", "--schedule", "cosine", "--warmup-epochs", "2",
            "--weight-decay", "5e-4", "--workers", "0", "--ckpt-root", str(ckpt_root), "--manifest", str(manifest)]
    monkeypatch.setattr(sys, "argv", argv + ["--smoke-steps", "2"])
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit):                                         # the GPU guard still holds
            tpc.main()
    monkeypatch.setattr(sys, "argv", argv + ["--smoke-steps", "2", "--allow-cpu"])
    tpc.main()
    out = capsys.readouterr().out
    assert "smoke: 2 steps on " in out and "mobilenet_v2 on tiny-imagenet (n_train=12, n_test=6, classes=3)" in out
    assert "batch (4, 3, 64, 64) -> logits (4, 3)" in out and "nothing saved" in out
    assert not ckpt_root.exists() and not manifest.exists()


# ------------------------------------------------------------------ CIFAR / SVHN / Fashion-MNIST unchanged


class _FakeCifar(datasets.CIFAR10):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        n = 60 if train else 40
        rng = np.random.default_rng(0 if train else 1)
        self.data = rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
        self.targets = [i % 10 for i in range(n)]
        self.transform, self.target_transform, self.train = transform, target_transform, train
        self.classes = [str(i) for i in range(10)]


class _FakeCifar100(_FakeCifar):
    pass


def test_cifar_contract_unchanged_with_the_new_flags_off(monkeypatch):
    monkeypatch.setattr(utils.datasets, "CIFAR10", _FakeCifar)          # the aug branch names the class ...
    monkeypatch.setattr(utils.datasets, "CIFAR100", _FakeCifar100)
    for name, fake in (("cifar-10", _FakeCifar), ("cifar-100", _FakeCifar100)):   # ... the builders bound it at import
        monkeypatch.setitem(utils.DATASET_BUILDERS, name, utils._torchvision_split(
            fake, {"train": True, "download": True}, {"train": False, "download": True}))
    monkeypatch.setenv("SPECTRA_BATCH_SIZE", "16")
    assert utils.val_from_train_per_class() == 0 and utils.GpuCropFlipLoader.pad == 4
    for name in ("cifar-10", "cifar-100"):
        assert _kinds(utils.build_transform(name, {}, train=True)) == ["ToTensor", "Normalize"]
        assert utils.DatasetRegistry.key_for(name) == name and not utils.gpu_ft_aug(name)
        train, val, test = utils.load_cnn_dataset(name, 0.7, 0.2)            # legacy carve from train
        assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (46, 14, 40)
        assert isinstance(train, torch.utils.data.DataLoader) and train.batch_size == 16
    # the eval line of record: VAL_FROM_TEST + FT_AUG (+ the GPU loader)
    monkeypatch.setenv("SPECTRA_VAL_FROM_TEST", "1")
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")
    train_t = utils.build_transform("cifar-10", {}, train=True)
    assert _kinds(train_t) == ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert (train_t.transforms[0].size, train_t.transforms[0].padding) == ((32, 32), 4)
    assert tuple(train_t.transforms[-1].mean) == CIFAR_MEAN and tuple(train_t.transforms[-1].std) == CIFAR_STD
    assert utils.DatasetRegistry.key_for("cifar-10") == "cifar-10|aug=1"
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    train, val, test = utils.load_cnn_dataset("cifar-10", 0.7, 0.2)
    assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (60, 20, 20)
    val_idx, test_idx = _reference_split(40)
    assert list(val.dataset.indices) == val_idx and list(test.dataset.indices) == test_idx
    assert _kinds(train.dataset.transform) == ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert _kinds(val.dataset.dataset.transform) == ["ToTensor", "Normalize"]
    assert any(line.startswith("FT aug on cifar-10: RandomCrop+Flip on train only (n_train=60, n_val=20)") for line in lines)
    assert utils.final_ft_train_loader(train, 16)[1] == "loader"
    monkeypatch.setenv("SPECTRA_FT_AUG_GPU", "1")
    assert utils.DatasetRegistry.key_for("cifar-10") == "cifar-10|aug=1|gpuaug=1"
    train, val, test = utils.load_cnn_dataset("cifar-10", 0.7, 0.2)
    assert isinstance(train, utils.GpuCropFlipLoader) and train.pad == 4 and len(train) == 4
    assert train.with_batch_size(32).pad == 4 and len(train.with_batch_size(32)) == 2
    assert any("RandomCrop+Flip on the GPU, train split device-resident (n_train=60, batch=16)" in line for line in lines)
    train._device = torch.device("cpu")
    train._materialise()
    assert train._x.shape == (60, 3, 40, 40)
    oy, ox = torch.tensor([0, 8, 4, 1, 7, 3]), torch.tensor([8, 0, 4, 6, 2, 5])
    flip = torch.tensor([False, True, False, True, True, False])
    x = utils.crop_flip_batch(train._x[:6], oy, ox, flip).float().div_(255.0)
    x = (x - train._m) / train._s
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    base = train.dataset
    for i in range(6):
        img = TF.crop(TF.pad(Image.fromarray(base.data[i]), 4, fill=0), int(oy[i]), int(ox[i]), 32, 32)
        if flip[i]:
            img = TF.hflip(img)
        assert torch.allclose(x[i], norm(img), atol=1e-6), i
    assert utils.final_ft_train_loader(train, 16)[1] == "loader"
    # the hold-out datasets keep their recipe
    monkeypatch.delenv("SPECTRA_FT_AUG_GPU")
    for spec in ("svhn", FMNIST):
        name, opts = utils.parse_dataset_spec(spec)
        assert _kinds(utils.build_transform(name, opts, train=True)) == _kinds(utils.build_transform(name, opts))
        assert not utils.gpu_ft_aug(utils.canonical_dataset_name(name), opts)
        assert "aug" not in utils.DatasetRegistry.key_for(spec)
    monkeypatch.setenv("SPECTRA_FT_AUG_HOLDOUT", "1")
    assert _kinds(utils.build_transform("svhn", {}, train=True)) == ["RandomCrop", "ToTensor", "Normalize"]
    assert _kinds(utils.build_transform("fashion-mnist", {"image_size": 32, "to_rgb": True}, train=True)) == [
        "Grayscale", "Resize", "RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert utils.DatasetRegistry.key_for("svhn").endswith("|haug=crop")
    assert "valtrain" not in utils.DatasetRegistry.key_for("cifar-10")
