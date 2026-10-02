"""CPU tests for ``SPECTRA_FT_AUG_HOLDOUT`` (default off): the G2 hold-out checkpoints' fine-tune aug.

    python -m pytest tests/test_holdout_ft_aug.py -v

SVHN gets RandomCrop(32, pad 4); Fashion-MNIST (grayscale -> 3 channels, resized to 32) gets
RandomCrop + horizontal flip; val and test stay unaugmented; nothing changes with the flag off.
"""

import sys
from pathlib import Path

import pytest
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import src.utils as utils  # noqa: E402

FMNIST = {"name": "fashion-mnist", "image_size": 32, "to_rgb": True}


def _kinds(t):
    return [type(s).__name__ for s in t.transforms]


def _train_eval(spec):
    name, opts = utils.parse_dataset_spec(spec)
    return utils.build_transform(name, opts, train=True), utils.build_transform(name, opts)


def test_default_off_changes_nothing(monkeypatch):
    monkeypatch.delenv("SPECTRA_FT_AUG_HOLDOUT", raising=False)
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")  # the CIFAR flag alone never reached these datasets
    for spec in ("svhn", FMNIST):
        train_t, eval_t = _train_eval(spec)
        assert _kinds(train_t) == _kinds(eval_t)
        assert "haug" not in utils.DatasetRegistry.key_for(spec)
        assert utils.holdout_ft_aug(utils.canonical_dataset_name(utils.parse_dataset_spec(spec)[0])) is None


def test_recipes_match_the_checkpoints_training(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_AUG_HOLDOUT", "1")
    monkeypatch.delenv("SPECTRA_FT_AUG", raising=False)
    train_t, eval_t = _train_eval("svhn")
    assert _kinds(train_t) == ["RandomCrop", "ToTensor", "Normalize"]
    assert _kinds(eval_t) == ["ToTensor", "Normalize"]
    crop = train_t.transforms[0]
    assert crop.size == (32, 32) and crop.padding == 4
    train_t, eval_t = _train_eval(FMNIST)
    assert _kinds(train_t) == ["Grayscale", "Resize", "RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert _kinds(eval_t) == ["Grayscale", "Resize", "ToTensor", "Normalize"]
    assert utils.DatasetRegistry.key_for("svhn").endswith("|haug=crop")
    assert utils.DatasetRegistry.key_for(FMNIST).endswith("|haug=crop+flip")
    cifar_train, cifar_eval = _train_eval("cifar-10")
    assert _kinds(cifar_train) == _kinds(cifar_eval)  # CIFAR stays on SPECTRA_FT_AUG


def test_same_ops_as_train_pretrained_checkpoint(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_AUG_HOLDOUT", "1")
    tpc = pytest.importorskip("train_pretrained_checkpoint")
    for name, spec in (("svhn", "svhn"), ("fashion-mnist", FMNIST)):
        theirs, _ = tpc.train_transform(name)
        mine, _ = _train_eval(spec)
        assert _kinds(mine) == _kinds(theirs)
        for a, b in zip(mine.transforms, theirs.transforms):
            if isinstance(a, transforms.RandomCrop):
                assert (a.size, a.padding) == (b.size, b.padding)
            if isinstance(a, transforms.Normalize):
                assert list(a.mean) == list(b.mean) and list(a.std) == list(b.std)


class _Fake(torch.utils.data.Dataset):
    def __init__(self, transform, n):
        self.transform, self.n = transform, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.zeros(3, 32, 32), 0


@pytest.mark.parametrize("val_from_test", ["1", "0"])
def test_loader_augments_train_only(monkeypatch, val_from_test):
    monkeypatch.setenv("SPECTRA_FT_AUG_HOLDOUT", "1")
    monkeypatch.setenv("SPECTRA_VAL_FROM_TEST", val_from_test)
    monkeypatch.setenv("SPECTRA_BATCH_SIZE", "4")
    monkeypatch.setattr(utils, "DATALOADER_WORKERS", 0)
    monkeypatch.setitem(utils.DATASET_BUILDERS, "svhn", lambda t: (_Fake(t, 40), _Fake(t, 20)))
    train, val, test = utils.load_cnn_dataset("svhn", 0.7, 0.2)

    def root(ds):
        while isinstance(ds, torch.utils.data.Subset):
            ds = ds.dataset
        return ds

    assert "RandomCrop" in _kinds(root(train.dataset).transform)
    for loader in (val, test):
        assert "RandomCrop" not in _kinds(root(loader.dataset).transform)
    if val_from_test == "1":
        assert len(train.dataset) == 40 and len(val.dataset) == 10 and len(test.dataset) == 10
    else:
        assert len(train.dataset) + len(val.dataset) == 40 and len(test.dataset) == 20
    loader, aug = utils.final_ft_train_loader(train)
    assert aug == "loader" and loader is train  # no second crop in the final fine-tune


def test_final_ft_label_unchanged_with_flag_off(monkeypatch):
    monkeypatch.delenv("SPECTRA_FT_AUG_HOLDOUT", raising=False)
    monkeypatch.setenv("SPECTRA_BATCH_SIZE", "4")
    t = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    loader = torch.utils.data.DataLoader(_Fake(t, 8), batch_size=4)
    assert utils.final_ft_train_loader(loader) == (loader, "none")
