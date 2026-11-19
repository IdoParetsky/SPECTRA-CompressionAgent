"""
Sealed test set (Ido's GO "S", 18 Nov 2026): the CIFAR-size factories of never-seen CNN families, and the logic of
``scripts/sealed_structure_check.py``.

The factories (``spectra_models_instantiation/*_cifar.py``): every one builds through the loader's contract, maps
(2, 3, 32, 32) to (2, num_classes), has the parameter count it was written to have (the reference implementations'
counts where one exists), keeps the row order ``ModelWithRows`` needs (stem first, classifier last) and traces to the
same channel groups in eval and in train mode. Whether SPECTRA's cuts are REAL on each network is the structure
check's job, not this file's: only the plain depthwise chain (MobileNet v1) is cut end to end here.

The checker: REAL / PARTLY-MASKED / MASKED / FAILS-TO-TRACE on small hand-made nets whose answer does not depend on
which architectures ``src/channel_groups.py`` supports today.

CPU only, no datasets (several minutes on the login node; run detached, one file per pytest process):

    python -m pytest tests/test_sealed_families.py -v
"""

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
import src.NetworkEnv as network_env  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.utils as utils  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load(REPO / "scripts" / "sealed_structure_check.py", "sealed_structure_check")
MODELS = REPO / "spectra_models_instantiation"
CANDIDATES = check.CANDIDATES
ARCHS = [c[2] for c in CANDIDATES]
BY_ARCH = {c[2]: c for c in CANDIDATES}
SEALED_FAMILIES = ("GoogLeNet", "MobileNet v1", "SqueezeNet", "DLA", "EfficientNet", "MobileNetV3", "ConvNeXt",
                   "HarDNet", "GhostNet", "SE-ResNet")
CONTROL_FAMILIES = ("ResNeXt-29", "RegNet")

# Parameters for 10 classes (BGU login node, torch 2.4.1, 18 Nov 2026). Reference counts they reproduce:
# kuangliu/pytorch-cifar's GoogLeNet 6.17 M, MobileNet 3,217,226, SimpleDLA 15.14 M, DLA 16.29 M, EfficientNetB0
# 3.60 M, ResNeXt29 32x4d 4.77 M and 2x64d 9.13 M, RegNetX_200MF 2.32 M, RegNetY_400MF 5.71 M.
PARAMS_10 = {
    "googlenet_x025": 395254, "googlenet_x050": 1551354, "googlenet_x075": 3475858, "googlenet": 6166250,
    "googlenet_x100": 6166250,
    "mobilenetv1_x025": 215642, "mobilenetv1_x050": 823722, "mobilenetv1_x075": 1824250, "mobilenetv1_x100": 3217226,
    "squeezenet1_1_x050": 185834, "squeezenet1_1": 730570, "squeezenet1_0": 732010, "squeezenet1_0_x200": 2900682,
    "simpledla_x025": 950534, "simpledla_x050": 3791202, "dla_x025": 1022598, "dla_x050": 4078690,
    "simpledla_x100": 15142970, "dla_x100": 16291386,
    "efficientnet_b0_x050": 969452, "efficientnet_b0_x075": 2141530, "efficientnet_b0": 3598598,
    "efficientnet_b1": 6104234,
    "mobilenetv3_small_x075": 892610, "mobilenetv3_small_x100": 1528106, "mobilenetv3_large_x075": 2426186,
    "mobilenetv3_large_x100": 4214842,
    "convnext_cifar_w24": 1242418, "convnext_cifar_atto": 3376290, "convnext_cifar_femto": 4834906,
    "convnext_cifar_pico": 8535498,
    "hardnet39ds": 2473478, "hardnet68ds": 3165852, "hardnet68_x050": 4161110, "hardnet68_x075": 9325562,
    "ghostnet_x050": 1318838, "ghostnet_x075": 2462320, "ghostnet_x100": 3913790, "ghostnet_x130": 6089926,
    "se_resnet20": 274490, "se_resnet32": 470266, "se_resnet56": 861818, "se_resnet110": 1742810,
    "resnext29_8x4d": 360138, "resnext29_16x4d": 1281354, "resnext29_32x4d": 4774218, "resnext29_2x64d": 9128778,
    "regnetx_200mf": 2321946, "regnetx_400mf": 4779338, "regnety_400mf": 5714362,
}
# The brief's aim for the set: roughly 0.1 M to 10 M parameters. The "extra" rows are the reference sizes above it.
AIM = (0.1e6, 10.0e6)

_BUILT = {}


def _model(arch, num_classes=10):
    """The network of a registry row in eval mode. The 10-class one is built once and shared by the read-only tests
    (about 0.7 GB for the whole registry); any other class count is built on demand."""
    _family, script, _arch, _role = BY_ARCH[arch]
    if num_classes != 10:
        return check.build(script, arch, num_classes).eval()
    if arch not in _BUILT:
        torch.manual_seed(0)
        _BUILT[arch] = check.build(script, arch, 10).eval()
    return _BUILT[arch]


def _count(model):
    return sum(p.numel() for p in model.parameters())


def _module(stem):
    return _load(MODELS / f"{stem}.py", stem)


# ----------------------------------------------------------------------------------------------- the registry


def test_registry_names_four_sealed_variants_per_family_and_the_two_controls():
    assert len(set(ARCHS)) == len(ARCHS)
    assert set(PARAMS_10) == set(ARCHS)
    for family, script, _arch, role in CANDIDATES:
        assert (MODELS / script).is_file(), script
        assert role in ("sealed", "extra", "control", "alias")
        assert (role == "control") == (family in CONTROL_FAMILIES)
    for family in SEALED_FAMILIES:
        assert sum(c[0] == family and c[3] == "sealed" for c in CANDIDATES) == 4, family
    assert {c[0] for c in CANDIDATES} == set(SEALED_FAMILIES) | set(CONTROL_FAMILIES)
    assert [c[2] for c in CANDIDATES if c[3] == "alias"] == ["googlenet_x100"]
    assert sum(c[3] == "sealed" for c in CANDIDATES) == 40


# ----------------------------------------------------------------------------------------------- every factory


@pytest.mark.parametrize("num_classes", [10, 100])
@pytest.mark.parametrize("arch", ARCHS)
def test_factory_builds_and_maps_a_cifar_batch_to_logits(arch, num_classes):
    model = _model(arch, num_classes)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(1)))
    assert out.shape == (2, num_classes)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ARCHS)
def test_parameter_count_is_the_expected_one_and_in_the_sets_range(arch):
    role = BY_ARCH[arch][3]
    n10, n100 = _count(_model(arch, 10)), _count(_model(arch, 100))
    assert n10 == PARAMS_10[arch]
    if role == "extra":
        assert n10 > AIM[1]
    else:
        assert AIM[0] < n10 < AIM[1]
    # only the classifier depends on the class count: (fan-in + 1) more parameters per extra class
    last = [m for m in _model(arch, 10).modules() if isinstance(m, (nn.Conv2d, nn.Linear))][-1]
    fan_in = last.in_features if isinstance(last, nn.Linear) else last.in_channels
    assert n100 - n10 == 90 * (fan_in + 1)


@pytest.mark.parametrize("arch", ARCHS)
def test_factory_meets_the_loaders_contract(arch):
    _family, script, _arch, _role = BY_ARCH[arch]
    module = _module(Path(script).stem)
    factory, resolved = utils.resolve_instantiation_func(module, arch)
    assert resolved == arch and factory is getattr(module, arch)
    alias, alias_name = utils.resolve_instantiation_func(module, arch.replace("_", "-"))
    assert alias is factory and alias_name == arch
    # load_model_from_script fills num_classes and large_input by name; a "width" argument would be scraped from
    # the checkpoint's file name
    assert list(inspect.signature(factory).parameters) == ["num_classes", "large_input"]
    shapes = {k: tuple(v.shape) for k, v in _model(arch, 10).state_dict().items()}
    large = factory(num_classes=10, large_input=True)
    assert {k: tuple(v.shape) for k, v in large.state_dict().items()} == shapes   # accepted and ignored


@pytest.mark.parametrize("arch", ARCHS)
def test_rows_start_at_the_stem_and_end_at_the_classifier(arch):
    """``ModelWithRows`` reads layers in registration order: its first layer must be the stem conv, and
    ``group_sensitivity.group_plan`` leaves out the last row as the classifier."""
    model = _model(arch, 10)
    mwr = ModelWithRows(model)
    first = mwr.all_layers[0]
    assert isinstance(first, nn.Conv2d) and first.in_channels == 3
    last = mwr.all_layers[mwr.row_to_main_layer[max(mwr.row_to_main_layer)]]
    width = last.out_features if isinstance(last, nn.Linear) else last.out_channels
    assert width == 10
    layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm))]
    assert len({id(m) for m in layers}) == len(layers)
    listed = {id(m) for m in mwr.all_layers}
    assert all(id(m) in listed for m in layers)                       # every weight layer is a leaf of the walk
    names = [name for name, _m in model.named_modules()]
    assert len(names) == len(set(names))
    assert not any(isinstance(m, nn.Sequential) and len(m) == 0 for m in model.modules())


@pytest.mark.parametrize("arch", ARCHS)
def test_channel_groups_trace_the_same_in_eval_and_train_mode(arch):
    model = _model(arch, 10)
    names = {id(m): n for n, m in model.named_modules()}

    def signature(groups):
        return sorted((check.group_label(g, names), g.prunable, g.reason) for g in groups if g.producers or g.depthwise)

    eval_groups = channel_groups.build_channel_groups(model)
    model.train()
    try:
        train_groups = channel_groups.build_channel_groups(model)
    finally:
        model.eval()
    assert eval_groups is not None and train_groups is not None, f"{arch}: torch.fx cannot trace the network"
    assert signature(eval_groups) == signature(train_groups)
    assert any(g.prunable and g.width >= 2 for g in eval_groups if g.producers or g.depthwise)
    # every Conv2d / Linear the forward calls owns exactly one group dimension, unless the tracer lost its input
    owners = [id(m) for g in eval_groups for m in list(g.producers) + list(g.depthwise)]
    assert len(owners) == len(set(owners))


@pytest.mark.parametrize("arch", ARCHS)
def test_one_backward_pass_reaches_every_parameter(arch):
    """A train-mode forward and backward on a random batch (no data, no optimizer step): an in-place activation
    or add that autograd cannot differentiate would only show when the origin is trained on a GPU, and a layer
    the forward never calls would sit in SPECTRA's rows and parameter count with no gradient."""
    torch.manual_seed(0)
    _family, script, _arch, _role = BY_ARCH[arch]
    model = check.build(script, arch, 10).train()
    out = model(torch.randn(4, 3, 32, 32))
    nn.functional.cross_entropy(out, torch.tensor([0, 1, 2, 3])).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"{arch}: no gradient reaches {missing[:5]}"
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())


@pytest.mark.parametrize("arch", ARCHS)
def test_every_layer_is_a_type_the_agents_topology_tokens_know(arch):
    """``TopologyFE`` gives an unrecognised layer an all-zero token, which would hide part of a never-seen network
    from the agent's state: every leaf layer of the sealed nets must be a type it maps."""
    from NetworkFeatureExtraction.src.FeatureExtractors.TopologyFE import TopologyFE
    extractor = TopologyFE()
    mwr = ModelWithRows(_model(arch, 10))
    unknown = sorted({type(layer).__name__ for layer in mwr.all_layers
                      if type(layer) not in extractor.layer_type_to_function})
    assert not unknown, f"{arch}: layer types without a topology token: {unknown}"
    tokens = extractor.extract_feature_map(mwr)
    assert len(tokens) == len(mwr.all_layers) and all(len(t) == 7 and any(t) for t in tokens)


# ----------------------------------------------------------------------------------------------- reference counts


def test_googlenet_width_one_is_the_existing_googlenet():
    wide, base = _module("googlenet_width_cifar"), _module("googlenet_cifar")
    for num_classes in (10, 100):
        a = wide.GoogLeNet(num_classes=num_classes, width_mult=1.0).state_dict()
        b = base.GoogLeNet(num_classes=num_classes).state_dict()
        assert list(a) == list(b)
        assert all(a[k].shape == b[k].shape for k in a)
    assert wide.scale_width(16, 0.25) == 8 and wide.scale_width(208, 0.75) == 156   # floor of 8, plain rounding


def test_mobilenetv3_has_torchvisions_parameter_counts_with_the_imagenet_head():
    m = _module("mobilenetv3_cifar")
    assert _count(m.MobileNetV3("large", num_classes=1000)) == 5483032
    assert _count(m.MobileNetV3("small", num_classes=1000)) == 2542856
    # the CIFAR change is strides only: total stride 8, so a 32-px input leaves a 4x4 map before pooling
    for arch in ("small", "large"):
        net = m.MobileNetV3(arch, num_classes=10).eval()
        with torch.no_grad():
            assert net.features(torch.zeros(1, 3, 32, 32)).shape[-2:] == (4, 4)
        assert _count(m.MobileNetV3(arch, num_classes=10, cifar_strides=False)) == _count(net)


def test_squeezenet_without_batch_norm_has_torchvisions_parameter_counts():
    s = _module("squeezenet_cifar")
    assert _count(s.SqueezeNet("1_1", num_classes=1000, batch_norm=False)) == 1235496
    # 1.0: torchvision's 1,248,424 less the weights a 7x7 stem has over the 3x3 CIFAR stem (3 * 96 * (49 - 9))
    assert _count(s.SqueezeNet("1_0", num_classes=1000, batch_norm=False)) == 1248424 - 3 * 96 * 40
    assert not any(isinstance(m, nn.BatchNorm2d) for m in s.SqueezeNet("1_1", batch_norm=False).modules())
    with pytest.raises(ValueError):
        s.SqueezeNet("2_0")


def test_hardnet_has_the_published_parameter_counts_with_the_imagenet_head():
    h = _module("hardnet_cifar")
    assert _count(h.HarDNet(68, False, 1000)) == 17565348        # 17.6 M
    assert _count(h.HarDNet(68, True, 1000)) == 4180602          # 4.2 M
    assert _count(h.HarDNet(39, True, 1000)) == 3488228          # 3.5 M
    assert _count(h.HarDNet(85, False, 1000)) == 36670212        # 36.7 M
    block = h.HarDBlock(64, 14, 1.7, 8)
    assert block.links == [[0], [1, 0], [2], [3, 2, 0], [4], [5, 4], [6], [7, 6, 4, 0]]   # layer k reads k - 2^i
    assert all(layer.conv.out_channels % 2 == 0 for layer in block.layers)


def test_ghostnet_has_the_published_parameter_counts_with_the_imagenet_head():
    g = _module("ghostnet_cifar")
    assert _count(g.GhostNet(num_classes=1000, width_mult=1.0)) == 5181980      # 5.2 M
    assert _count(g.GhostNet(num_classes=1000, width_mult=0.5)) == 2587028      # 2.6 M
    assert _count(g.GhostNet(num_classes=1000, width_mult=1.3)) == 7358116      # 7.3 M
    # with ratio 2 and even widths the two halves are exactly ``oup`` channels: no slice is traced
    assert not any(m.trim for m in g.GhostNet(num_classes=10, width_mult=1.0).modules()
                   if isinstance(m, g.GhostModule))
    odd = g.GhostModule(8, 9).eval()
    assert odd.trim and odd(torch.zeros(1, 8, 4, 4)).shape[1] == 9


def test_kuangliu_reference_networks_have_their_parameter_counts():
    assert PARAMS_10["mobilenetv1_x100"] == 3217226
    assert PARAMS_10["efficientnet_b0"] == 5288548 - (320 * 1280 + 2 * 1280) - (1280 * 1000 + 1000) + (320 * 10 + 10)
    assert round(PARAMS_10["simpledla_x100"] / 1e4) == 1514 and round(PARAMS_10["dla_x100"] / 1e4) == 1629
    assert round(PARAMS_10["resnext29_32x4d"] / 1e4) == 477 and round(PARAMS_10["resnext29_2x64d"] / 1e4) == 913
    assert round(PARAMS_10["regnetx_200mf"] / 1e4) == 232 and round(PARAMS_10["regnety_400mf"] / 1e4) == 571
    assert round(PARAMS_10["googlenet"] / 1e4) == 617


def test_efficientnet_scaling_follows_the_reference_rounding():
    e = _module("efficientnet_cifar")
    assert [e.round_filters(c, 1.0) for c in e.B0["out_channels"]] == list(e.B0["out_channels"])
    assert [e.round_filters(c, 0.5) for c in (32, 16, 24, 40, 80, 112, 192, 320)] == [16, 8, 16, 24, 40, 56, 96, 160]
    assert [e.round_repeats(r, 1.1) for r in e.B0["num_blocks"]] == [2, 3, 3, 4, 4, 5, 2]
    b0 = e.EfficientNet(10)
    assert not hasattr(b0.layers[0], "conv1")                  # ratio 1: no expansion conv is built
    assert all(hasattr(block, "conv1") for block in list(b0.layers)[1:])
    assert len(b0.layers) == 16 and len(e.EfficientNet(10, depth_mult=1.1).layers) == 23


def test_convnext_and_seresnet_and_dla_shapes():
    c = _module("convnext_cifar")
    net = c.ConvNeXt(10, dims=(24, 48, 96, 192)).eval()
    with torch.no_grad():
        assert net.features(torch.zeros(1, 3, 32, 32)).shape == (1, 192, 2, 2)     # 2x2 stem, three downsamplings
    x = torch.randn(2, 5, 3, 3)
    norm = c.LayerNorm2d(5).eval()
    with torch.no_grad():                                                            # the channels-first LayerNorm
        want = (x - x.mean(1, keepdim=True)) / torch.sqrt((x - x.mean(1, keepdim=True)).pow(2).mean(1, keepdim=True)
                                                          + 1e-6)
        torch.testing.assert_close(norm(x), want, atol=1e-5, rtol=1e-5)
    s = _module("seresnet_cifar")
    assert [b.se.fc[0].out_features for b in s.se_resnet20(10, False).layer1] == [1, 1, 1]   # 16 // 16
    assert len(s.se_resnet110(10, False).layer3) == 18
    d = _module("dla_cifar")
    assert hasattr(d.dla_x025(10, False).layer4, "prev_root") and not hasattr(d.simpledla_x025(10, False).layer4,
                                                                             "prev_root")


# ----------------------------------------------------------------------------------------------- the checker


class _Plain(nn.Module):
    """Three conv-BN-ReLU layers and a classifier: every group is cut for real."""

    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8)
        self.conv2, self.bn2 = nn.Conv2d(8, 12, 3, padding=1, bias=False), nn.BatchNorm2d(12)
        self.conv3, self.bn3 = nn.Conv2d(12, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16)
        self.pool, self.fc = nn.AdaptiveAvgPool2d(1), nn.Linear(16, 10)

    def features(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return torch.relu(self.bn3(self.conv3(x)))

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.features(x)), 1))


class _FlipAfterConv3(_Plain):
    """``conv3``'s channels pass through a channel flip, an operation the tracer does not know: the group is refused."""

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(torch.flip(self.features(x), dims=[1])), 1))


class _OneConvFlipped(nn.Module):
    """The only group that is not the classifier's passes through the flip: nothing can be cut."""

    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8)
        self.pool, self.fc = nn.AdaptiveAvgPool2d(1), nn.Linear(8, 10)

    def forward(self, x):
        x = torch.flip(torch.relu(self.bn1(self.conv1(x))), dims=[1])
        return self.fc(torch.flatten(self.pool(x), 1))


class _HiddenReader(_Plain):
    """``conv2`` reads ``conv1``'s channels through a slice, which hides it from the tracer: cutting ``conv1``'s group
    cannot be a real cut (today the edit breaks the forward and the walk masks it)."""

    def features(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x[:, :8])))
        return torch.relu(self.bn3(self.conv3(x)))


class _Untraceable(_Plain):
    def forward(self, x):
        if x.sum() > 0:           # data-dependent control flow: torch.fx cannot trace it
            x = x * 2
        return super().forward(x)


class _TrainOnlyBranch(_Plain):
    def forward(self, x):
        if self.training and x.sum() > 0:    # traces in eval mode only
            x = x * 2
        return super().forward(x)


def _checked(cls):
    torch.manual_seed(0)
    return check.check_model(cls())


def test_checker_calls_a_plain_chain_real_and_its_numbers_add_up():
    res = _checked(_Plain)
    assert res["verdict"] == "REAL" and res["error"] is None and res["masked_groups"] == []
    assert res["groups"] == 4 and res["planned"] == 3 and res["real_groups"] == 3 and res["realized"] == 3
    assert res["blocked"] == [] and res["untracked_layers"] == [] and res["trace_train"] == "same"
    assert res["structure_ok"] and res["cost_models_ok"]
    kept = [res["cuts"][k]["params_kept"] for k in ("0.75", "0.5", "0.25")]
    assert kept == sorted(kept, reverse=True) and kept[0] < 1.0
    for key, keep in (("0.75", 0.75), ("0.5", 0.5), ("0.25", 0.25)):
        cut = res["cuts"][key]
        assert cut["cuts"] == cut["structural"] == 3 and cut["masked"] == [] and cut["off_target"] == []
        assert cut["forward_ok"] and cut["shrinks"] and cut["eq_ok"] and cut["pm_exact"] and cut["fm_exact"]
        # the comparison with the zeroed copy is not vacuous: the output is not all zeros and the cut moved it
        assert cut["eq_informative"] and cut["out_scale"] > 0 and cut["moved"] > 1e-3 and cut["eq_gap"] < 1e-12
        assert cut["widths"] == cut["targets"] == {"0": int(round(8 * keep)), "1": int(round(12 * keep)),
                                                   "2": int(round(16 * keep))}
    assert res["floor_cut"]["widths"] == {"0": 2, "1": 2, "2": 2}       # keep 0.1, at least 2 channels
    assert res["floor_cut"]["params_kept"] < kept[-1]
    shares = res["shares"]
    assert sum(v["params"] for v in shares.values()) == pytest.approx(1.0)
    assert sum(v["macs"] for v in shares.values()) == pytest.approx(1.0)
    assert shares["blocked"]["params"] == 0 and shares["output"]["params"] == pytest.approx(170 / res["params"])
    json.dumps(res)                                                     # the result is what the JSON file holds


def test_checker_names_the_group_the_tracer_refuses():
    res = _checked(_FlipAfterConv3)
    assert res["verdict"] == "PARTLY-MASKED"
    assert [b["group"] for b in res["blocked"]] == ["conv3 w=16"] and "flip" in res["blocked"][0]["reason"]
    assert res["planned"] == 2 and res["real_groups"] == 2 and res["masked_planned"] == []
    assert len(res["masked_groups"]) == 1 and res["masked_groups"][0].startswith("conv3 w=16 [")
    assert res["shares"]["blocked"]["params"] == pytest.approx((12 * 16 * 9 + 2 * 16) / res["params"])
    assert all(cut["masked"] == [] and cut["eq_ok"] for cut in res["cuts"].values())


def test_checker_calls_a_net_with_no_cuttable_group_masked():
    res = _checked(_OneConvFlipped)
    assert res["verdict"] == "MASKED" and res["planned"] == 0 and res["real_groups"] == 0
    assert res["cuts"] == {} and "floor_cut" not in res
    assert [b["group"] for b in res["blocked"]] == ["conv1 w=8"]


def test_checker_catches_a_reader_the_tracer_does_not_see():
    res = _checked(_HiddenReader)
    assert res["verdict"] == "PARTLY-MASKED"
    named = [g for g in res["masked_groups"] if g.startswith("conv1 w=8 [")]
    assert len(named) == 1, res["masked_groups"]
    assert res["real_groups"] == 2                                      # conv2's and conv3's groups are still real
    cut = res["floor_cut"]
    assert cut["forward_ok"] and cut["eq_ok"] and cut["masked"] == []


def test_checker_catches_a_cut_that_runs_but_misroutes_channels(monkeypatch):
    """A structural edit that keeps the wrong input channels of every reader has the right shapes, runs and shrinks.
    Only the comparison with the zeroed copy can tell: the checker must not call any such group real."""
    import src.group_sensitivity as group_sensitivity
    import src.pruning as pruning
    real = pruning.surviving_input_channels

    def first_channels(ref, group_width, keep_idx):
        kept = real(ref, group_width, keep_idx)
        return torch.arange(kept.numel(), dtype=kept.dtype, device=kept.device)

    monkeypatch.setattr(pruning, "surviving_input_channels", first_channels)
    monkeypatch.setattr(group_sensitivity, "_REALIZED", {})
    res = _checked(_Plain)
    assert res["verdict"] == "MASKED" and res["real_groups"] == 0 and res["realized"] == 0
    assert len(res["masked_planned"]) == 3 and "floor_cut" not in res
    for cut in res["cuts"].values():
        assert cut["forward_ok"] and cut["shrinks"] and cut["masked"] == [] and cut["off_target"] == []
        assert not cut["eq_ok"] and cut["eq_gap"] > 1e-3
    monkeypatch.setattr(group_sensitivity, "_REALIZED", {})


@pytest.mark.parametrize("cls,first", [(_Untraceable, "every group"), (_TrainOnlyBranch, "train-mode trace fails")])
def test_checker_reports_a_net_torch_fx_cannot_trace(cls, first):
    res = _checked(cls)
    assert res["verdict"] == "FAILS-TO-TRACE" and res["masked_groups"][0].startswith(first)
    assert res["trace_eval"] is (cls is _TrainOnlyBranch)


def test_checker_keeps_an_exception_as_an_error_row():
    class Broken(nn.Module):
        def forward(self, x):
            raise RuntimeError("no forward")

    res = check.check_model(Broken())
    assert res["verdict"] == "ERROR" and "no forward" in res["error"] and "Traceback" in res["traceback"]
    missing = check.check_candidate("X", "mobilenetv1_cifar.py", "no_such_factory", "sealed")
    assert missing["verdict"] == "ERROR" and "no_such_factory" in missing["error"]


def test_recording_cuts_restores_the_walks_prune_function():
    real = network_env.prune_current_model
    with pytest.raises(KeyError):
        with check.recording_cuts([]):
            assert network_env.prune_current_model is not real
            raise KeyError("x")
    assert network_env.prune_current_model is real


def test_condition_is_deterministic_and_leaves_no_dead_filter():
    a, b = check.condition(_Plain(), seed=3), check.condition(_Plain(), seed=3)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    assert all(float(m.weight.reshape(m.weight.size(0), -1).abs().sum(dim=1).min()) > 0
               for m in a.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    scale = check.condition(_module("convnext_cifar").ConvNeXt(10, dims=(8, 8, 8, 8)))
    assert all(float(block.gamma.min()) >= 0.5 for stage in scale.features for block in stage
               if hasattr(block, "gamma"))                               # the 1e-6 layer scale would hide a branch


def test_report_has_one_full_row_per_network_and_names_what_is_not_real():
    results = []
    for cls, verdict in ((_Plain, "REAL"), (_FlipAfterConv3, "PARTLY-MASKED"), (_HiddenReader, "PARTLY-MASKED"),
                         (_OneConvFlipped, "MASKED"), (_Untraceable, "FAILS-TO-TRACE")):
        res = _checked(cls)
        assert res["verdict"] == verdict
        res.update(family="Toy", script="toy.py", arch=cls.__name__, role="sealed")
        results.append(res)
    text = check.render_report(results)
    rows = [line for line in text.splitlines() if line.startswith("| Toy ")]
    assert len(rows) == 5 and all(line.count("|") == 18 for line in rows)       # 17 columns
    # the verdict cell says which kind of group is not cut: refused by the tracer (never planned) or masked in the plan
    assert rows[0].endswith("| **REAL** |") and rows[1].endswith("| **PARTLY-MASKED** (1 unprunable) |")
    assert rows[2].endswith("| **PARTLY-MASKED** (1 masked in plan) |")
    assert rows[3].endswith("| **MASKED** (1 unprunable) |") and rows[4].endswith("| **FAILS-TO-TRACE** |")
    assert "### _Plain" not in text and "### _FlipAfterConv3 (Toy): PARTLY-MASKED" in text
    assert "conv3 w=16" in text and "- **Toy** (5 networks" in text
    broken = dict(results[0], verdict="ERROR", error="RuntimeError: a | b", traceback="Traceback ...", arch="_Broken")
    row = [line for line in check.render_report([broken]).splitlines() if line.startswith("| Toy ")][0]
    assert row.count("|") == 18 and "RuntimeError: a / b" in row and row.endswith("| **ERROR** |")


def test_command_line_lists_the_candidates(capsys):
    assert check.main(["--list"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(CANDIDATES) and "GoogLeNet\tgooglenet|googlenet_cifar.py\tsealed" in lines


def test_command_line_checks_mobilenet_v1_end_to_end(tmp_path, capsys):
    """The one family cut end to end here: a depthwise-separable chain is the primitive SPECTRA already cuts on
    MobileNetV2, so the smallest MobileNet v1 must come out REAL, with its JSON and the report written."""
    assert check.main(["--out", str(tmp_path), "--only", "mobilenetv1_x025"]) == 0
    out = capsys.readouterr().out
    res = json.loads((tmp_path / "mobilenetv1_cifar__mobilenetv1_x025.json").read_text(encoding="utf-8"))
    assert res["verdict"] == "REAL" and res["arch"] == "mobilenetv1_x025" and res["params"] == 215642
    assert res["groups"] == 15 and res["planned"] == res["real_groups"] == 14 and res["blocked"] == []
    assert all(cut["masked"] == [] and cut["eq_ok"] and cut["pm_exact"] and cut["fm_exact"]
               for cut in res["cuts"].values())
    assert "mobilenetv1_x025: REAL" in out and "REAL 1" in out
    assert "| MobileNet v1 | mobilenetv1_x025 | sealed |" in (tmp_path / "sealed_structure_report.md").read_text(
        encoding="utf-8")
    assert check.main(["--out", str(tmp_path), "--only", "mobilenetv1_x025"]) == 0     # a second run reuses the JSON
    assert "skip mobilenetv1_x025 (exists)" in capsys.readouterr().out


def test_command_line_names_the_networks_for_a_shell_loop(capsys):
    """``--names`` feeds the one-process-per-network loop the login node needs (it kills long-lived python
    processes): every distinct network once, no alias, the filters honoured, no ``--out`` needed."""
    assert check.main(["--names"]) == 0
    names = capsys.readouterr().out.split()
    assert names == [c[2] for c in CANDIDATES if c[3] != "alias"] and len(names) == 49
    assert check.main(["--names", "--role", "control"]) == 0
    assert capsys.readouterr().out.split() == [c[2] for c in CANDIDATES if c[3] == "control"]
    assert check.main(["--names", "--family", "dla,ghost"]) == 0
    assert len(capsys.readouterr().out.split()) == 10
    with pytest.raises(SystemExit):
        check.main(["--names", "--only", "no_such_network"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
