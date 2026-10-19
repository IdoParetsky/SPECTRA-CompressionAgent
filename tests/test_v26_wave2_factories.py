"""
Wave-2 pretraining-catalog factories: VGG width variants and the CIFAR GoogLeNet.

CPU only, no datasets. The existing VGG factories must keep their exact state_dict
layout (the classifier-width argument defaults to 512), the new factories must build,
resolve by name (and hyphen alias), expose prunable channel groups to SPECTRA's
discovery, and survive a uniform 50 % structural keep through the real prune path.

    python -m pytest tests/test_v26_wave2_factories.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src.NetworkEnv import prune_current_model  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
from src.utils import resolve_instantiation_func  # noqa: E402
from spectra_models_instantiation import vgg_chenyaofo, googlenet_cifar  # noqa: E402

VGG_NEW = [
    ("vgg11_bn_x050", 11, 0.5), ("vgg11_bn_x075", 11, 0.75),
    ("vgg13_bn_x050", 13, 0.5), ("vgg13_bn_x075", 13, 0.75),
    ("vgg16_bn_x050", 16, 0.5), ("vgg16_bn_x075", 16, 0.75),
]
VGG_OLD = [("vgg11_bn", 11), ("vgg13_bn", 13), ("vgg16_bn", 16), ("vgg19_bn", 19)]
ALL_NEW = [(name, vgg_chenyaofo) for name, _, _ in VGG_NEW] + [("googlenet", googlenet_cifar)]


def _count(model):
    return sum(p.numel() for p in model.parameters())


def _build(module, name, num_classes):
    return getattr(module, name)(num_classes=num_classes, large_input=False)


def _expected_vgg_shapes(cfg, num_classes, cls_in=512):
    """state_dict key -> shape for make_layers(cfg, batch_norm=True) + the 3-Linear classifier."""
    shapes = {}
    idx, in_ch = 0, 3
    for v in cfg:
        if v == 'M':
            idx += 1
            continue
        shapes[f"features.{idx}.weight"] = (v, in_ch, 3, 3)
        shapes[f"features.{idx}.bias"] = (v,)
        for key in ("weight", "bias", "running_mean", "running_var"):
            shapes[f"features.{idx + 1}.{key}"] = (v,)
        shapes[f"features.{idx + 1}.num_batches_tracked"] = ()
        idx += 3
        in_ch = v
    for i, (o, n) in zip((0, 3, 6), ((512, cls_in), (512, 512), (num_classes, 512))):
        shapes[f"classifier.{i}.weight"] = (o, n)
        shapes[f"classifier.{i}.bias"] = (o,)
    return shapes


def _shapes(model):
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


# (a) every new factory builds, maps (2, 3, 32, 32) -> (2, num_classes), sane parameter counts

@pytest.mark.parametrize("num_classes", [10, 100])
@pytest.mark.parametrize("name,module", ALL_NEW)
def test_new_factory_builds_and_runs(name, module, num_classes):
    model = _build(module, name, num_classes).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, num_classes)


@pytest.mark.parametrize("num_classes", [10, 100])
@pytest.mark.parametrize("name,depth,mult", VGG_NEW)
def test_vgg_width_variant_parameter_count(name, depth, mult, num_classes):
    base = _count(_build(vgg_chenyaofo, f"vgg{depth}_bn", num_classes))
    ratio = _count(_build(vgg_chenyaofo, name, num_classes)) / base
    # Convs scale ~mult^2; the two 512x512 hidden Linears stay at 512, which lifts the ratio.
    lo, hi = (0.22, 0.32) if mult == 0.5 else (0.52, 0.64)
    assert lo < ratio < hi, f"{name}: {ratio:.3f} of the x1.0 parameter count"


@pytest.mark.parametrize("num_classes,lo,hi", [(10, 6.0e6, 6.3e6), (100, 6.0e6, 6.35e6)])
def test_googlenet_parameter_count(num_classes, lo, hi):
    assert lo < _count(_build(googlenet_cifar, "googlenet", num_classes)) < hi


def test_scale_cfg_rounds_floors_and_keeps_pools():
    assert vgg_chenyaofo.scale_cfg([64, 'M', 128, 'M', 512], 0.5) == [32, 'M', 64, 'M', 256]
    assert vgg_chenyaofo.scale_cfg([64, 512], 0.75) == [48, 384]
    assert vgg_chenyaofo.scale_cfg([10, 'M'], 0.5) == [8, 'M']  # floor at 8


@pytest.mark.parametrize("name,depth,mult", VGG_NEW)
def test_vgg_width_variant_layout(name, depth, mult):
    model = _build(vgg_chenyaofo, name, 10)
    expected = _expected_vgg_shapes(vgg_chenyaofo.scale_cfg(vgg_chenyaofo.cfgs[depth], mult), 10,
                                    cls_in=int(round(512 * mult)))
    assert _shapes(model) == expected
    assert model.classifier[0].in_features == int(round(512 * mult))


# (b) the existing VGG factories keep exactly their state_dict keys and shapes

@pytest.mark.parametrize("num_classes", [10, 100])
@pytest.mark.parametrize("name,depth", VGG_OLD)
def test_existing_vgg_state_dict_unchanged(name, depth, num_classes):
    model = _build(vgg_chenyaofo, name, num_classes)
    expected = _expected_vgg_shapes(vgg_chenyaofo.cfgs[depth], num_classes)
    assert _shapes(model) == expected
    assert list(model.state_dict().keys()) == list(expected.keys())  # same order too


def test_vgg_default_classifier_width_is_512():
    model = vgg_chenyaofo.VGG(vgg_chenyaofo.make_layers([64, 'M']), num_classes=10)
    assert model.classifier[0].in_features == 512


# (c) the instantiation resolver finds each new name (and the hyphen alias)

@pytest.mark.parametrize("name,module", ALL_NEW)
def test_resolver_finds_new_factories(name, module):
    func, resolved = resolve_instantiation_func(module, name)
    assert resolved == name and func is getattr(module, name)


@pytest.mark.parametrize("name,depth,mult", VGG_NEW)
def test_resolver_hyphen_alias(name, depth, mult):
    func, resolved = resolve_instantiation_func(vgg_chenyaofo, name.replace("_", "-"))
    assert resolved == name and func is getattr(vgg_chenyaofo, name)


# (d) channel-group discovery and a uniform 50 % structural keep

@pytest.mark.parametrize("name,module", ALL_NEW)
def test_channel_groups_discovered(name, module, capsys):
    model = _build(module, name, 10).eval()
    groups = channel_groups.build_channel_groups(model)
    assert groups is not None, f"{name}: symbolic trace failed, no structural pruning possible"
    prunable = [g for g in groups if g.prunable]
    assert prunable, f"{name}: no prunable channel group"
    # Report the groups discovery refuses (visible with ``pytest -s``). For GoogLeNet these
    # would be the dimensions tied to the model output or to an unmatched concat/operand
    # width; they are reported, not forced.
    blocked = sorted({(g.width, g.reason) for g in groups if not g.prunable})
    print(f"{name}: {len(prunable)} prunable / {len(groups) - len(prunable)} unprunable groups; "
          f"unprunable (width, reason): {blocked}")


def _prune_every_row(model, keep, step=1):
    num_rows = len(ModelWithRows(model).row_to_main_layer)
    for row_idx in range(0, num_rows, step):
        mwr = ModelWithRows(model)
        prune_current_model(mwr, keep, row_idx, quiet=True, record=False, input_shape=(3, 32, 32))
        model = mwr.model
    return model


@pytest.mark.parametrize("num_classes", [10, 100])
@pytest.mark.parametrize("name,module", [("vgg11_bn_x050", vgg_chenyaofo), ("vgg16_bn_x075", vgg_chenyaofo),
                                         ("googlenet", googlenet_cifar)])
def test_uniform_half_keep_still_runs(name, module, num_classes):
    model = _build(module, name, num_classes).eval()
    before = _count(model)
    pruned = _prune_every_row(model, 0.5).eval()
    with torch.no_grad():
        out = pruned(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, num_classes)
    assert _count(pruned) < before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
