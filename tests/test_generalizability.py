"""
SPECTRA is meant to be an off-the-shelf agent: any CNN, any classification dataset, no
per-architecture or per-dataset code. These tests pin the properties that claim depends on.

    python -m pytest tests/test_generalizability.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import src.channel_groups as channel_groups  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.utils as utils  # noqa: E402
from NetworkFeatureExtraction.src.FeatureExtractors.TopologyFE import TopologyFE  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402


# --------------------------------------------------------------------------- datasets

def test_every_dataset_named_in_the_thesis_is_supported():
    """The proposal's abstract names these; a missing one cannot be experimented on."""
    for name in ("cifar-10", "cifar-100", "fashion-mnist", "svhn", "places365", "imagenet1k"):
        assert utils.canonical_dataset_name(name) in utils.DATASET_BUILDERS, name


def test_dataset_aliases_resolve_to_one_canonical_entry():
    assert utils.canonical_dataset_name("CIFAR10") == "cifar-10"
    assert utils.canonical_dataset_name("fashionMNIST") == "fashion-mnist"
    assert utils.canonical_dataset_name("imagenet") == "imagenet1k"
    assert utils.canonical_dataset_name("/data/custom") == "/data/custom"


def test_env_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("SPECTRA_AMP", raising=False)
    assert utils.env_flag("SPECTRA_AMP") is False
    monkeypatch.setenv("SPECTRA_AMP", "1")
    assert utils.env_flag("SPECTRA_AMP") is True


def test_cifar_ft_aug_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SPECTRA_FT_AUG", raising=False)
    tf = utils.build_transform("cifar-10", {}, train=True)
    kinds = [type(t).__name__ for t in tf.transforms]
    assert "RandomCrop" not in kinds
    assert "RandomHorizontalFlip" not in kinds


def test_cifar_ft_aug_only_on_train_split(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")
    train_tf = utils.build_transform("cifar-10", {}, train=True)
    eval_tf = utils.build_transform("cifar-10", {}, train=False)
    train_kinds = [type(t).__name__ for t in train_tf.transforms]
    eval_kinds = [type(t).__name__ for t in eval_tf.transforms]
    assert "RandomCrop" in train_kinds
    assert "RandomHorizontalFlip" in train_kinds
    assert "RandomCrop" not in eval_kinds
    assert "RandomHorizontalFlip" not in eval_kinds


def test_dataset_cache_key_includes_ft_aug(monkeypatch):
    monkeypatch.delenv("SPECTRA_FT_AUG", raising=False)
    plain = utils.DatasetRegistry.key_for("cifar-10")
    monkeypatch.setenv("SPECTRA_FT_AUG", "1")
    aug = utils.DatasetRegistry.key_for("cifar-10")
    assert plain == "cifar-10"
    assert aug.endswith("|aug=1")
    assert plain != aug


def test_normalisation_is_per_dataset():
    """A single 0.5/0.5 normalisation shifts the activation statistics the agent reads."""
    cifar = utils.build_transform("cifar-10", {})
    mnist = utils.build_transform("mnist", {})
    cifar_norm = [t for t in cifar.transforms if hasattr(t, "mean")][0]
    mnist_norm = [t for t in mnist.transforms if hasattr(t, "mean")][0]

    assert len(cifar_norm.mean) == 3
    assert len(mnist_norm.mean) == 1
    assert tuple(cifar_norm.mean) != tuple(mnist_norm.mean)


def test_dataset_spec_accepts_preprocessing_options():
    """A grayscale dataset can feed a 3-channel network trained at another resolution."""
    from PIL import Image

    name, options = utils.parse_dataset_spec({"name": "mnist", "image_size": 32, "to_rgb": True})
    assert name == "mnist" and options == {"image_size": 32, "to_rgb": True}

    transform = utils.build_transform(name, options)
    converted = transform(Image.new("L", (28, 28)))  # a single-channel 28x28 MNIST digit
    assert converted.shape == (3, 32, 32)


def test_registry_keys_separate_different_preprocessing():
    registry = utils.DatasetRegistry(0.7, 0.2)
    plain = registry.key_for("mnist")
    resized = registry.key_for({"name": "mnist", "image_size": 32, "to_rgb": True})
    assert plain != resized


def test_preloaded_datasets_do_not_lazy_load_the_rest_of_the_json(monkeypatch, tmp_path):
    """--datasets cifar-10 used to still instantiate CIFAR-100 JSON nets via lazy load."""
    c10 = tmp_path / "c10.pt"
    c100 = tmp_path / "c100.pt"
    c10.write_bytes(b"x")
    c100.write_bytes(b"x")
    script = tmp_path / "dummy.py"
    script.write_text("def resnet20(**kw):\n    return None\n")

    registry = utils.DatasetRegistry(0.7, 0.2)
    registry.restrict_to_preloaded = True
    registry._entries["cifar-10"] = {
        "loaders": ("train", "val", "test"),
        "num_classes": 10,
        "input_shape": (3, 32, 32),
    }

    loaded = []

    def fake_load(arch, dataset_path, script_path, checkpoint_path, optional_kwargs,
                  num_classes, input_shape):
        loaded.append((checkpoint_path, dataset_path))
        return nn.Identity()

    monkeypatch.setattr(utils, "load_model_from_script", fake_load)

    out = utils.instantiate_networks_and_load_datasets({
        str(c10): ["resnet20", str(script), "cifar-10"],
        str(c100): ["resnet20", str(script), "cifar-100"],
    }, registry)

    assert str(c10) in out
    assert str(c100) not in out
    assert loaded == [(str(c10), "cifar-10")]
    with pytest.raises(KeyError, match="cifar-100"):
        registry.get("cifar-100")


def test_preload_allows_the_requested_datasets_then_locks(monkeypatch):
    seen_restrict = []

    def fake_get(self, spec):
        seen_restrict.append(self.restrict_to_preloaded)
        key = self.key_for(spec)
        self._entries[key] = {
            "loaders": ("train", "val", "test"),
            "num_classes": 10,
            "input_shape": (3, 32, 32),
        }
        return self._entries[key]

    monkeypatch.setattr(utils.DatasetRegistry, "get", fake_get)
    registry = utils.preload_datasets(["cifar-10"], 0.7, 0.2)
    assert seen_restrict == [False]
    assert registry.restrict_to_preloaded is True
    assert "cifar-10" in registry


def test_unknown_dataset_reports_the_supported_ones():
    with pytest.raises(ValueError, match="cifar-10"):
        utils.load_cnn_dataset("not-a-dataset", 0.7, 0.2)


def test_paths_are_configurable_rather_than_hardcoded():
    assert "SPECTRA_DATASETS" in utils.__dict__ or os.environ.get("SPECTRA_DATASETS") is None
    import src.A2C_Agent_Reinforce as a2c
    assert not a2c.TRAINED_AGENTS_DIR.startswith("/sise/")


# --------------------------------------------------------------------------- architectures

def test_topology_covers_common_layer_types():
    """An unmapped module contributes an all-zero token, hiding it from the agent."""
    topology = TopologyFE()
    required = [nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm,
                nn.ReLU6, nn.GELU, nn.Hardswish, nn.LeakyReLU, nn.SiLU,
                nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.Dropout, nn.Flatten, nn.Identity]
    missing = [t.__name__ for t in required if t not in topology.layer_type_to_function]
    assert not missing, f"unmapped layer types: {missing}"


def test_topology_handles_adaptive_pooling_without_kernel_size():
    topology = TopologyFE()
    encoded = topology.layer_type_to_function[nn.AdaptiveAvgPool2d](nn.AdaptiveAvgPool2d(1))
    assert len(encoded) == 7 and encoded[0] == 7


def test_every_layer_token_has_the_same_width():
    """Fixed-width tokens are what let one agent read architectures it has never seen."""
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.norm = nn.GroupNorm(2, 4)
            self.act = nn.Hardswish()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(self.flatten(self.pool(self.act(self.norm(self.conv(x))))))

    tokens = TopologyFE().extract_feature_map(ModelWithRows(Mixed()))
    assert {len(t) for t in tokens} == {7}


def test_activations_are_captured_for_functional_activation_networks():
    """
    Hooking only activation *modules* leaves networks that call F.relu with no activation
    statistics at all -- an all-zero third of the state.
    """
    from NetworkFeatureExtraction.src.FeatureExtractors.ActivationsStatisticsFE import (
        ActivationsStatisticsFE)

    class Functional(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.bn = nn.BatchNorm2d(4)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            out = torch.relu(self.bn(self.conv(x)))  # functional, not a module
            return self.fc(self.flatten(self.pool(out)))

    loader = [(torch.randn(2, 3, 16, 16), torch.zeros(2, dtype=torch.long))]
    extractor = ActivationsStatisticsFE(loader, torch.device("cpu"))
    stats = extractor.extract_feature_map(ModelWithRows(Functional().eval()))

    assert any(any(abs(v) > 0 for v in row) for row in stats), "no activation statistics captured"


@pytest.mark.parametrize("model_factory", [
    lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)),
    lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GroupNorm(2, 8), nn.Hardswish(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)),
    lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, groups=1), nn.BatchNorm2d(8), nn.ReLU6(),
                          nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.BatchNorm2d(8),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)),
])
def test_unseen_architectures_are_analysable_and_prunable(model_factory):
    from src.NetworkEnv import prune_current_model

    model = model_factory().eval()
    sample = torch.randn(2, 3, 16, 16)
    reference = model(sample)

    groups = channel_groups.build_channel_groups(model)
    assert groups is not None

    model_with_rows = ModelWithRows(model)
    prune_current_model(model_with_rows, 0.5, 0)
    pruned = model_with_rows.model.eval()

    assert pruned(sample).shape == reference.shape


def test_pruning_never_removes_the_classifier_outputs():
    """Whatever the architecture, the label space must survive."""
    for out_features in (2, 10, 1000):
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(), nn.Linear(8, out_features)).eval()
        groups = channel_groups.build_channel_groups(model)
        final = channel_groups.group_of(groups, model[3])
        assert final is not None and not final.prunable


def test_unsupported_constructs_degrade_instead_of_raising():
    """An architecture the analysis cannot model must still be prunable by masking."""
    class Exotic(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
            self.fc = nn.Linear(8, 2)

        def forward(self, x):
            flat = self.conv(x).flatten(2).transpose(1, 2)
            attended, _ = self.attn(flat, flat, flat)
            return self.fc(attended.mean(dim=1))

    from src.NetworkEnv import prune_current_model

    model = Exotic().eval()
    sample = torch.randn(2, 3, 8, 8)
    model_with_rows = ModelWithRows(model)
    prune_current_model(model_with_rows, 0.5, 0)

    assert model_with_rows.model.eval()(sample).shape == (2, 2)
    assert pruning.alive_filters(model.conv).numel() == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
