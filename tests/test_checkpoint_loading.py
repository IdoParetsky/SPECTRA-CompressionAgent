"""Checkpoint unwrap / alias / RepVGG-deploy / akamaster option-A loading."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import src.utils as utils  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src.NetworkEnv import prune_current_model  # noqa: E402
from spectra_models_instantiation.resnet_akamaster import resnet20 as akamaster_resnet20  # noqa: E402
from spectra_models_instantiation.repvgg_chenyaofo import repvgga0, repvgg_a0  # noqa: E402


def test_unwrap_akamaster_wrapper_drops_best_prec1():
    weights = {"conv1.weight": torch.ones(1), "bn1.weight": torch.ones(1)}
    wrapped = {"best_prec1": 91.73, "state_dict": weights}
    assert utils.unwrap_checkpoint_state_dict(wrapped) == weights


def test_unwrap_model_state_dict_and_module():
    weights = {"conv.weight": torch.ones(1, 1, 1, 1)}
    wrapped = {"model_config": {"arch": "resnet50"}, "model_state_dict": weights}
    assert utils.unwrap_checkpoint_state_dict(wrapped) == weights

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1, bias=False)
            nn.init.ones_(self.conv.weight)

    nested = {"state_dict": Tiny(), "epoch": 3}
    got = utils.unwrap_checkpoint_state_dict(nested)
    assert list(got) == ["conv.weight"]
    inner = {"fc.weight": torch.zeros(2, 2)}
    assert utils.unwrap_checkpoint_state_dict({"model": inner, "epoch": 12}) == inner


def test_align_strips_module_prefix():
    sd = {"module.conv.weight": torch.ones(1)}
    aligned = utils.align_module_prefix(sd, ["conv.weight"])
    assert list(aligned) == ["conv.weight"]


def test_alias_shortcut_and_linear_to_torchvision_names():
    sd = {
        "linear.weight": torch.ones(2, 4),
        "linear.bias": torch.zeros(2),
        "layer2.0.shortcut.0.weight": torch.ones(1),
    }
    aligned = utils.apply_key_aliases(sd, ["fc.weight", "fc.bias", "layer2.0.downsample.0.weight"])
    assert "fc.weight" in aligned and "linear.weight" not in aligned
    assert "layer2.0.downsample.0.weight" in aligned


def test_ignorable_bn_buffers_do_not_fail_load():
    model = nn.Sequential(nn.Conv2d(3, 4, 1, bias=False), nn.BatchNorm2d(4))
    sd = {k: v.clone() for k, v in model.state_dict().items()
          if not k.endswith("num_batches_tracked")}
    missing, unexpected = utils.load_state_dict_compatible(model, sd)
    assert missing == []
    assert unexpected == []


def test_real_missing_weights_still_fail():
    model = nn.Linear(4, 2)
    missing, unexpected = utils.load_state_dict_compatible(model, {})
    assert "weight" in missing
    assert "bias" in missing
    assert unexpected == []


def test_thop_profiler_keys_are_ignorable():
    model = nn.Conv2d(3, 4, 1, bias=False)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["total_ops"] = torch.zeros(1)
    sd["total_params"] = torch.zeros(1)
    sd["features.0.total_ops"] = torch.zeros(1)
    missing, unexpected = utils.load_state_dict_compatible(model, sd)
    assert missing == []
    assert unexpected == []


def test_repvgg_filename_alias_and_deploy_inference():
    module = SimpleNamespace(repvgg_a0=repvgg_a0, repvgga0=repvgga0)
    fn, name = utils.resolve_instantiation_func(module, "repvgga0")
    assert name in ("repvgga0", "repvgg_a0")
    assert fn is not None

    train_keys = {"stage0.rbr_dense.conv.weight": torch.zeros(1)}
    deploy_keys = {"stage0.rbr_reparam.weight": torch.zeros(1)}
    assert utils.infer_repvgg_deploy(train_keys) is False
    assert utils.infer_repvgg_deploy(deploy_keys) is True


def test_akamaster_option_a_shortcut_matches_paper_pad():
    import torch.nn.functional as F
    from spectra_models_instantiation.resnet_akamaster import PadShortcut

    x = torch.randn(2, 16, 32, 32)
    paper = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 8, 8), "constant", 0)
    ours = PadShortcut(16, 32)(x)
    assert torch.allclose(ours, paper)


def test_akamaster_option_a_shortcut_is_frozen_identity():
    model = akamaster_resnet20(num_classes=10)
    trainable = [n for n, p in model.named_parameters() if "shortcut" in n and p.requires_grad]
    assert trainable == []
    assert any(n.endswith("shortcut.channel.weight") for n, _ in model.named_parameters())


def test_akamaster_resnet20_prunes_and_runs():
    model = akamaster_resnet20(num_classes=10).eval()
    sample = torch.randn(1, 3, 32, 32)
    before = sum(p.numel() for p in model.parameters())
    for row_idx in range(len(ModelWithRows(model).row_to_main_layer)):
        rows = ModelWithRows(model)
        prune_current_model(rows, 0.8, row_idx)
        model = rows.model
    out = model.eval()(sample)
    assert out.shape == (1, 10)
    assert sum(p.numel() for p in model.parameters()) < before


def test_repvgg_train_mode_forwards():
    model = repvgg_a0(num_classes=10, large_input=False, deploy=False).eval()
    out = model(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 10)
    assert any("rbr_dense.conv.weight" in k for k in model.state_dict())
    assert not any("rbr_reparam" in k for k in model.state_dict())


def test_optioned_fashionmnist_is_allowed_when_canonical_was_preloaded():
    registry = utils.DatasetRegistry(0.7, 0.2)
    registry.restrict_to_preloaded = True
    registry._allowed_canonical = {"fashion-mnist"}
    registry._entries["fashion-mnist"] = {
        "loaders": ("train", "val", "test"),
        "num_classes": 10,
        "input_shape": (1, 28, 28),
    }
    spec = {"name": "fashion-mnist", "image_size": 32, "to_rgb": True}
    assert spec in registry
    assert "cifar-100" not in registry


def test_dfpc_mobilenet_named_blocks_and_shortcuts():
    from spectra_models_instantiation.mobilenetv2_dfpc import mobilenet_v2_dfpc
    model = mobilenet_v2_dfpc(num_classes=100)
    keys = list(model.state_dict())
    assert "conv1.weight" in keys and "layers.0.conv1.weight" in keys
    assert "linear.weight" in keys and model.linear.out_features == 100
    assert "layers.0.shortcut.0.weight" in keys
    assert "layers.2.shortcut.0.weight" not in keys
    out = model.eval()(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 100)


def test_vgg19_bn_linear_head():
    from spectra_models_instantiation.vgg_chenyaofo import vgg19_bn_linear
    model = vgg19_bn_linear(num_classes=100, large_input=False)
    keys = list(model.state_dict())
    assert "features.0.weight" in keys
    assert "classifier.weight" in keys
    assert "classifier.0.weight" not in keys
    assert model.classifier.weight.shape == (100, 512)
    assert model.eval()(torch.randn(1, 3, 32, 32)).shape == (1, 100)


def test_vgg19_bn_blocks_match_depgraph_layout():
    from spectra_models_instantiation.vgg_repdistiller import vgg19_bn
    model = vgg19_bn(num_classes=100, large_input=False)
    keys = list(model.state_dict())
    assert "block0.0.weight" in keys and "block4.9.weight" in keys
    assert "classifier.weight" in keys
    assert model.eval()(torch.randn(1, 3, 32, 32)).shape == (1, 100)


def test_sublinear_lenet_named_3x3():
    from spectra_models_instantiation.sublinear_instantiation import lenet_mnist_sublinear
    model = lenet_mnist_sublinear(num_classes=10)
    keys = list(model.state_dict())
    assert keys[:2] == ["conv1.weight", "conv1.bias"]
    assert model.conv1.kernel_size == (3, 3)
    assert model.fc1.in_features == 400
    assert model.eval()(torch.randn(1, 1, 28, 28)).shape == (1, 10)
