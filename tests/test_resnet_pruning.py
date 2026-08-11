"""
Structured pruning against a real torchvision ResNet-18.

ResNets are the architecture family the thesis proposal targets, and they are precisely
where a flat view of the layer list is misleading: inside a BasicBlock, conv1 feeds conv2
directly (resizable), while conv2's output is added to the identity shortcut (not
resizable without also editing the shortcut).

    python -m pytest tests/test_resnet_pruning.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402  (installs a CPU StaticConf)

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
import src.pruning as pruning  # noqa: E402

torchvision = pytest.importorskip("torchvision")
from torchvision.models import resnet18  # noqa: E402


def _row_of(model_with_rows, module):
    layer_idx = next(idx for idx, layer in enumerate(model_with_rows.all_layers) if layer is module)
    return next(row for row, main in model_with_rows.row_to_main_layer.items() if main == layer_idx)


def test_intra_block_conv_is_structurally_pruned():
    from src.NetworkEnv import prune_current_model

    model = resnet18(weights=None).eval()
    sample = torch.randn(1, 3, 64, 64)
    params_before = sum(p.numel() for p in model.parameters())

    model_with_rows = ModelWithRows(model)
    target = model.layer1[0].conv1  # feeds conv1 -> bn1 -> relu -> conv2, no shortcut in between
    prune_current_model(model_with_rows, 0.5, _row_of(model_with_rows, target))

    pruned = model_with_rows.model.eval()

    assert pruned.layer1[0].conv1.out_channels == 32
    assert pruned.layer1[0].bn1.num_features == 32
    assert pruned.layer1[0].conv2.in_channels == 32
    assert sum(p.numel() for p in pruned.parameters()) < params_before
    assert pruned(sample).shape == (1, 1000)


def test_residual_stage_is_pruned_as_one_coupled_group():
    """
    Every tensor summed into layer1's residual path shares one channel dimension: the stem
    convolution, both blocks' conv2, and the corresponding BatchNorms. They must shrink
    together, and every layer reading that dimension must be resized with them.
    """
    from src.NetworkEnv import prune_current_model

    model = resnet18(weights=None).eval()
    sample = torch.randn(1, 3, 64, 64)
    params_before = sum(p.numel() for p in model.parameters())

    model_with_rows = ModelWithRows(model)
    target = model.layer1[0].conv2  # its output is summed with the block's identity shortcut
    prune_current_model(model_with_rows, 0.5, _row_of(model_with_rows, target))

    pruned = model_with_rows.model.eval()

    # Producers of the shared dimension
    assert pruned.conv1.out_channels == 32
    assert pruned.layer1[0].conv2.out_channels == 32
    assert pruned.layer1[1].conv2.out_channels == 32
    assert pruned.bn1.num_features == 32
    # Consumers of the shared dimension
    assert pruned.layer1[0].conv1.in_channels == 32
    assert pruned.layer1[1].conv1.in_channels == 32
    assert pruned.layer2[0].conv1.in_channels == 32
    assert pruned.layer2[0].downsample[0].in_channels == 32

    assert pruned(sample).shape == (1, 1000)
    assert sum(p.numel() for p in pruned.parameters()) < params_before


def test_classifier_output_dimension_is_never_pruned():
    """The final Linear defines the label space and must keep all of its outputs."""
    import src.channel_groups as channel_groups

    model = resnet18(weights=None).eval()
    groups = channel_groups.build_channel_groups(model)
    fc_group = channel_groups.group_of(groups, model.fc)

    assert fc_group is not None
    assert not fc_group.prunable
    assert fc_group.reason == "model output"


def test_every_prunable_layer_leaves_the_network_runnable():
    """Compress each Conv/Linear in turn and confirm the network still executes."""
    from src.NetworkEnv import prune_current_model

    model = resnet18(weights=None).eval()
    sample = torch.randn(1, 3, 64, 64)

    model_with_rows = ModelWithRows(model)
    num_rows = len(model_with_rows.row_to_main_layer)

    for row_idx in range(num_rows):
        model_with_rows = ModelWithRows(model)
        prune_current_model(model_with_rows, 0.8, row_idx)
        model = model_with_rows.model
        assert model.eval()(sample).shape == (1, 1000), f"row {row_idx} broke the forward pass"

    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in resnet18(weights=None).parameters())


@pytest.mark.parametrize("builder,expect_shrink", [
    ("vgg16", True),
    ("resnet50", True),
    ("densenet121", True),
    ("mobilenet_v2", True),
    ("googlenet", True),
])
def test_architecture_family_compresses_and_still_runs(builder, expect_shrink):
    """
    One uniform pass over every prunable layer of each architecture in the thesis scope.

    DenseNet and GoogLeNet exercise concatenation (channels must be removed at the right
    offset inside the merged tensor) and MobileNet exercises depthwise convolutions.
    """
    from torchvision import models
    from src.NetworkEnv import prune_current_model

    kwargs = {"init_weights": False, "aux_logits": False} if builder == "googlenet" else {}
    model = getattr(models, builder)(weights=None, **kwargs).eval()
    sample = torch.randn(1, 3, 64, 64)
    params_before = sum(p.numel() for p in model.parameters())

    num_rows = len(ModelWithRows(model).row_to_main_layer)
    for row_idx in range(num_rows):
        model_with_rows = ModelWithRows(model)
        prune_current_model(model_with_rows, 0.8, row_idx)
        model = model_with_rows.model

    model = model.eval()
    assert model(sample).shape[0] == 1
    if expect_shrink:
        assert sum(p.numel() for p in model.parameters()) < params_before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
