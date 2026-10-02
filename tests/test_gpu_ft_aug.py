"""CPU tests for ``SPECTRA_FT_AUG_GPU`` (default off): device-resident CIFAR train split with crop+flip.

    python -m pytest tests/test_gpu_ft_aug.py -v

The GPU loader must draw the same augmentation as torchvision RandomCrop(32, pad 4)+Flip, keep
labels with their images, cover the split once per epoch, leave val/test alone, and not get a
second crop in the final fine-tune. Nothing changes with the flag off.
"""

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

import src.utils as utils  # noqa: E402

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CPU = torch.device("cpu")


def _loader(data, targets, batch=64, mean=MEAN, std=STD):
    base = _FakeCifar(None, data=data, targets=targets)
    return utils.GpuCropFlipLoader(base, batch, mean, std, device=CPU)


class _FakeCifar(datasets.CIFAR10):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None,
                 data=None, targets=None):
        n = 60 if train else 40
        rng = np.random.default_rng(0 if train else 1)
        self.data = data if data is not None else rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
        self.targets = list(targets) if targets is not None else [i % 10 for i in range(len(self.data))]
        self.transform, self.target_transform, self.train = transform, target_transform, train
        self.classes = [str(i) for i in range(10)]


def test_crop_flip_matches_torchvision():
    rng = np.random.default_rng(3)
    imgs = rng.integers(0, 256, (6, 32, 32, 3), dtype=np.uint8)
    oy = torch.tensor([0, 8, 4, 1, 7, 3])
    ox = torch.tensor([8, 0, 4, 6, 2, 5])
    flip = torch.tensor([False, True, False, True, True, False])
    loader = _loader(imgs, range(6))
    loader._materialise()
    x = utils.crop_flip_batch(loader._x, oy, ox, flip).float().div_(255.0)
    x = (x - loader._m) / loader._s
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    for i in range(6):
        img = TF.crop(TF.pad(Image.fromarray(imgs[i]), 4, fill=0), int(oy[i]), int(ox[i]), 32, 32)
        if flip[i]:
            img = TF.hflip(img)
        assert torch.allclose(x[i], norm(img), atol=1e-6), i


def _decode(x):
    """Offsets and flip from images whose channel 0 holds row+1 and channel 1 holds col+1."""
    v = (x * 255).round().long()
    flipped = (v[:, 1, 16, 17] - v[:, 1, 16, 16]) < 0
    ox = torch.where(flipped, v[:, 1, 16, 16] - 12, v[:, 1, 16, 16] - 13)
    oy = v[:, 0, 16, 16] - 13
    return oy, ox, flipped


def test_offsets_uniform_on_0_to_8_and_flip_half():
    torch.manual_seed(0)
    n = 9000
    rows, cols = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    img = np.stack([rows + 1, cols + 1, np.zeros_like(rows)], axis=-1).astype(np.uint8)
    loader = _loader(np.repeat(img[None], n, 0), [0] * n, batch=1000, mean=(0, 0, 0), std=(1, 1, 1))
    oy, ox, flipped = (torch.cat(t) for t in zip(*(_decode(x) for x, _ in loader)))
    for off in (oy, ox):
        counts = torch.bincount(off, minlength=9)
        assert counts.numel() == 9 and off.min() == 0 and off.max() == 8
        assert ((counts - n / 9).abs() < 0.2 * n / 9).all(), counts.tolist()
    assert 0.47 < flipped.float().mean().item() < 0.53


def test_epoch_covers_each_example_once_and_labels_follow():
    torch.manual_seed(1)
    n = 200
    data = np.zeros((n, 32, 32, 3), dtype=np.uint8)
    data[..., 2] = (np.arange(n) + 1)[:, None, None]
    targets = [i % 7 for i in range(n)]
    loader = _loader(data, targets, batch=64, mean=(0, 0, 0), std=(1, 1, 1))
    assert len(loader) == 4
    seen, sizes = [], []
    for x, y in loader:
        assert x.dtype == torch.float32 and y.dtype == torch.long and x.shape[1:] == (3, 32, 32)
        idx = (x[:, 2, 16, 16] * 255).round().long() - 1
        assert torch.equal(y, idx % 7)
        seen += idx.tolist()
        sizes.append(len(y))
    assert sorted(seen) == list(range(n)) and sizes == [64, 64, 64, 8]


def test_subset_indices_are_honoured():
    n = 50
    data = np.zeros((n, 32, 32, 3), dtype=np.uint8)
    data[..., 2] = (np.arange(n) + 1)[:, None, None]
    base = _FakeCifar(None, data=data, targets=range(n))
    keep = [3, 9, 27, 41]
    loader = utils.GpuCropFlipLoader(torch.utils.data.Subset(torch.utils.data.Subset(base, list(range(1, 50))),
                                                             [k - 1 for k in keep]),
                                     8, (0, 0, 0), (1, 1, 1), device=CPU)
    x, y = next(iter(loader))
    assert sorted(y.tolist()) == keep
    assert sorted(((x[:, 2, 16, 16] * 255).round().long() - 1).tolist()) == keep


def test_default_off_and_guards(monkeypatch):
    monkeypatch.delenv("SPECTRA_FT_AUG_GPU", raising=False)
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")
    monkeypatch.delenv("SPECTRA_FT_AUTOAUG", raising=False)
    assert not utils.gpu_ft_aug("cifar-10")
    assert "gpuaug" not in utils.DatasetRegistry.key_for("cifar-10")
    monkeypatch.setenv("SPECTRA_FT_AUG_GPU", "1")
    assert utils.gpu_ft_aug("cifar-10") and utils.gpu_ft_aug("cifar-100")
    assert utils.DatasetRegistry.key_for("cifar-10").endswith("|aug=1|gpuaug=1")
    assert not utils.gpu_ft_aug("svhn")
    assert not utils.gpu_ft_aug("cifar-10", {"image_size": 64})
    monkeypatch.setenv("SPECTRA_FT_AUTOAUG", "1")
    assert not utils.gpu_ft_aug("cifar-10")
    monkeypatch.delenv("SPECTRA_FT_AUTOAUG")
    monkeypatch.delenv("SPECTRA_FT_AUG")
    assert not utils.gpu_ft_aug("cifar-10")


def _root(ds):
    while isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    return ds


@pytest.mark.parametrize("val_from_test", ["1", "0"])
@pytest.mark.parametrize("gpu", ["1", "0"])
def test_load_cnn_dataset_wiring(monkeypatch, val_from_test, gpu):
    monkeypatch.setattr(utils.datasets, "CIFAR10", _FakeCifar)
    monkeypatch.setattr(utils, "DATALOADER_WORKERS", 0)
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")
    monkeypatch.delenv("SPECTRA_FT_AUTOAUG", raising=False)
    monkeypatch.setenv("SPECTRA_FT_AUG_GPU", gpu)
    monkeypatch.setenv("SPECTRA_VAL_FROM_TEST", val_from_test)
    monkeypatch.setenv("SPECTRA_BATCH_SIZE", "16")
    train, val, test = utils.load_cnn_dataset("cifar-10", 0.7, 0.2)
    for loader in (val, test):
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert "RandomCrop" not in [type(s).__name__ for s in _root(loader.dataset).transform.transforms]
    assert utils.infer_num_classes(train.dataset) == 10
    if gpu == "0":
        assert isinstance(train, torch.utils.data.DataLoader)
        assert utils.final_ft_train_loader(train)[1] == "loader"
        return
    assert isinstance(train, utils.GpuCropFlipLoader)
    n_train = 60 if val_from_test == "1" else len(train.dataset)
    assert len(train.dataset) == n_train
    x, y = next(iter(train))
    assert x.shape == (16, 3, 32, 32) and y.shape == (16,)
    loader, aug = utils.final_ft_train_loader(train)
    assert loader is train and aug == "loader"
    loader, aug = utils.final_ft_train_loader(train, batch_size=8)
    assert isinstance(loader, utils.GpuCropFlipLoader) and loader.batch_size == 8 and aug == "loader"
    assert next(iter(loader))[0].shape[0] == 8
