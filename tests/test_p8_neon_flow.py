"""
P8 (V5, 18 Sep): NEON "layer replacement" on CNN groups — recipes C-G / C-G+.

Hirsch & Katz 2022 Sec. 3: after the action the analysed layer is *replaced* by a new layer of
width a_t * W, randomly initialised; every other layer is frozen and the new one is trained
until convergence; the feature maps are refreshed before the next state. Upstream
NEON_NetworkEnv.py (--prune False) rebuilt the producer Linear, the consumer Linear and the
BatchNorm, and is_learn_new_layers_only kept exactly those trainable.

CPU only, no datasets, default-off flags. Covers the units the brief asks for: dummy-forward
after replacement, identity skip, group consumers trainable, moments refresh, policy_config pin.

    python -m pytest tests/test_p8_neon_flow.py -v
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf, ConcatNet, _row_index_of_layer  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from NetworkFeatureExtraction.src.FeatureExtractors.ActivationsStatisticsFE import ActivationsStatisticsFE  # noqa: E402
from src import fortify  # noqa: E402
from src.Configuration.StaticConf import StaticConf  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
from src.NetworkEnv import NetworkEnv, prune_current_model  # noqa: E402
import src.pruning as pruning  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402

P8_FLAGS = ("SPECTRA_FT_REINIT_EDITED", "SPECTRA_FT_REINIT_THEN_POLISH", "SPECTRA_REFRESH_ALL_FEATURES",
            "SPECTRA_FT_REINIT_SELECT", "SPECTRA_FT_REINIT_SCOPE", "SPECTRA_FT_REINIT_EPOCHS",
            "SPECTRA_FT_REINIT_PATIENCE", "SPECTRA_FT_POLISH_EPOCHS", "SPECTRA_FT_POLISH_PATIENCE",
            "SPECTRA_FT_POLISH_LR_MULT")


@pytest.fixture(autouse=True)
def _clean_p8_env(monkeypatch):
    for key in P8_FLAGS:
        monkeypatch.delenv(key, raising=False)
    yield


def _layer_index(mwr, module):
    return next(i for i, layer in enumerate(mwr.all_layers) if layer is module)


def _prune_stream(model, rate=0.8):
    """Structurally cut the stage-1 residual stream of a thin r20 through the real prune path."""
    mwr = ModelWithRows(model)
    conv2 = model.layer1[0].conv2
    row = _row_index_of_layer(mwr, _layer_index(mwr, conv2))
    prune_current_model(mwr, rate, row, quiet=True, record=False, input_shape=(3, 32, 32))
    assert mwr.last_prune_outcome["mode"] == "structural"
    return mwr


def _tiny_loader(n=16, classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)


# ------------------------------------------------------------------ flags / recipe names

def test_p8_flags_default_off_and_recipe_names(monkeypatch):
    assert fortify.ft_reinit_edited() is False
    assert fortify.ft_reinit_then_polish() is False
    assert fortify.refresh_all_features() is False
    assert fortify.ft_recipe(False) == "A"
    assert fortify.ft_recipe(True) == "B"
    monkeypatch.setenv("SPECTRA_FT_REINIT_EDITED", "1")
    assert fortify.ft_recipe(False) == "C-G"
    monkeypatch.setenv("SPECTRA_FT_REINIT_THEN_POLISH", "1")
    assert fortify.ft_recipe(True) == "C-G+"          # polish implies reinit, beats B
    assert fortify.ft_reinit_epochs() == 60 and fortify.ft_reinit_patience() == 6
    assert fortify.ft_reinit_select() == "val"
    monkeypatch.setenv("SPECTRA_FT_REINIT_SELECT", "train")
    assert fortify.ft_reinit_select() == "train"
    monkeypatch.setenv("SPECTRA_FT_REINIT_EPOCHS", "24")
    monkeypatch.setenv("SPECTRA_FT_POLISH_LR_MULT", "0.05")
    assert fortify.ft_reinit_epochs() == 24 and fortify.ft_polish_lr_mult() == pytest.approx(0.05)


# ------------------------------------------------------------------ bookkeeping of the group edit

def test_structural_prune_records_group_edit_with_full_consumer_slices():
    """Residual stream: producers = stem + every conv2; consumers = every conv1 (+ downsample)."""
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    conv1s = [blk.conv1 for blk in model.layer1]
    mwr_before = ModelWithRows(model)
    conv1_idx = {_layer_index(mwr_before, m) for m in conv1s}
    mwr = _prune_stream(model)
    edit = mwr.last_group_edit
    assert edit["new_width"] == 3
    assert edit["producers"] == mwr.last_prune_outcome["group_layer_indices"]
    assert conv1_idx <= set(edit["consumers"])
    for idx in conv1_idx:
        module = mwr.all_layers[idx]
        # The block conv1 reads exactly this stream: the whole new input is the group's slice.
        assert sorted(edit["consumers"][idx]) == list(range(module.in_channels))
    assert edit["norms"] and all(len(pos) == 3 for pos in edit["norms"].values())
    # The set of trainable ids NEON would keep = producers + norms + consumers of the edit.
    edited_modules = [mwr.all_layers[i] for i in edit["producers"]] + \
                     [mwr.all_layers[int(i)] for i in edit["norms"]] + \
                     [mwr.all_layers[int(i)] for i in edit["consumers"]]
    edited_ids = {id(p) for m in edited_modules for p in m.parameters()}
    assert edited_ids == set(mwr.last_edited_param_ids)


def test_masked_fallback_leaves_nothing_to_replace(monkeypatch):
    """A masked edit rewrites no module: no group edit, reinit is a no-op."""
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = ModelWithRows(model)
    mwr.last_group_edit = {"producers": [0]}
    layer = mwr.all_layers[0]
    pruning.mask_layer_filters(layer, torch.arange(1))
    mwr.last_group_edit = None
    assert pruning.reinit_group_edit(mwr, None)["reinit"] is False
    assert pruning.reinit_group_edit(mwr, {})["reinit"] is False


# ------------------------------------------------------------------ layer replacement itself

def test_reinit_replaces_pretrained_group_and_keeps_forward_valid():
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = _prune_stream(model)
    edit = mwr.last_group_edit
    survivors = {i: mwr.all_layers[i].weight.detach().clone() for i in edit["producers"]}
    consumer_before = {int(i): mwr.all_layers[int(i)].weight.detach().clone() for i in edit["consumers"]}

    summary = pruning.reinit_group_edit(mwr, edit)
    assert summary["reinit"] is True
    assert summary["producers"] == len(edit["producers"])
    assert summary["norms"] == len(edit["norms"])
    assert summary["consumers_full"] == len(edit["consumers"]) and summary["consumers_slice"] == 0
    assert summary["params_reinit"] > 0
    for i, old in survivors.items():
        new = mwr.all_layers[i].weight.detach()
        assert new.shape == old.shape and not torch.allclose(new, old)   # thrown away, not kept
        assert new.std().item() > 0
    for i, old in consumer_before.items():
        assert not torch.allclose(mwr.all_layers[i].weight.detach(), old)
    for i in edit["norms"]:
        norm = mwr.all_layers[int(i)]
        assert torch.all(norm.weight == 1) and torch.all(norm.bias == 0)
        assert torch.all(norm.running_mean == 0) and torch.all(norm.running_var == 1)
    # Dummy forward after replacement: the new group is installed at the new width.
    out = mwr.model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10) and torch.isfinite(out).all()
    assert mwr.model.layer1[1].conv2.out_channels == 3


def test_reinit_scope_producers_keeps_consumer_weights(monkeypatch):
    """SPECTRA_FT_REINIT_SCOPE=producers: Gilad's oral wording — only the pruned layer(s) + norms."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = _prune_stream(model)
    edit = mwr.last_group_edit
    consumers_before = {int(i): mwr.all_layers[int(i)].weight.detach().clone() for i in edit["consumers"]}
    producers_before = {i: mwr.all_layers[i].weight.detach().clone() for i in edit["producers"]}
    summary = pruning.reinit_group_edit(mwr, edit, scope="producers")
    assert summary["scope"] == "producers" and summary["consumers_full"] == 0 and summary["consumers_slice"] == 0
    assert summary["producers"] == len(edit["producers"]) and summary["norms"] == len(edit["norms"])
    for i, old in consumers_before.items():
        assert torch.equal(mwr.all_layers[i].weight.detach(), old)
    for i, old in producers_before.items():
        assert not torch.allclose(mwr.all_layers[i].weight.detach(), old)
    assert fortify.ft_reinit_scope() == "group"
    monkeypatch.setenv("SPECTRA_FT_REINIT_SCOPE", "producers")
    assert fortify.ft_reinit_scope() == "producers"


def test_reinit_on_concat_touches_only_the_pruned_branch_slice():
    """
    ConcatNet: `a` (8 ch) and `b` (6 ch) are concatenated; `norm` and `head` read all 14.
    Halving `a` must re-draw only a's slice of head / norm; b's pretrained slice is kept.
    """
    torch.manual_seed(1)
    model = ConcatNet().eval()
    mwr = ModelWithRows(model)
    row = _row_index_of_layer(mwr, _layer_index(mwr, model.a))
    prune_current_model(mwr, 0.5, row, quiet=True, record=False, input_shape=(3, 16, 16))
    assert mwr.last_prune_outcome["mode"] == "structural"
    edit = mwr.last_group_edit
    head_idx = _layer_index(mwr, mwr.model.head)
    norm_idx = _layer_index(mwr, mwr.model.norm)
    assert edit["new_width"] == 4
    assert sorted(edit["consumers"][head_idx]) == [0, 1, 2, 3]       # a's kept channels come first
    assert sorted(edit["norms"][norm_idx]) == [0, 1, 2, 3]
    head_b_before = mwr.model.head.weight.detach()[:, 4:].clone()
    norm_b_mean_before = mwr.model.norm.running_mean.detach()[4:].clone()
    norm_b_w_before = mwr.model.norm.weight.detach()[4:].clone()
    head_a_before = mwr.model.head.weight.detach()[:, :4].clone()

    summary = pruning.reinit_group_edit(mwr, edit)
    assert summary["consumers_slice"] == 1 and summary["consumers_full"] == 0
    assert torch.allclose(mwr.model.head.weight.detach()[:, 4:], head_b_before)      # b kept
    assert not torch.allclose(mwr.model.head.weight.detach()[:, :4], head_a_before)  # a fresh
    assert torch.allclose(mwr.model.norm.running_mean.detach()[4:], norm_b_mean_before)
    assert torch.allclose(mwr.model.norm.weight.detach()[4:], norm_b_w_before)
    assert torch.all(mwr.model.norm.weight.detach()[:4] == 1) and torch.all(mwr.model.norm.running_var.detach()[:4] == 1)
    assert mwr.model(torch.randn(2, 3, 16, 16)).shape == (2, 2)


# ------------------------------------------------------------------ recovery under C-G / C-G+

def _bare_env(model, mode="agent_train", train_only=False):
    env = NetworkEnv.__new__(NetworkEnv)
    conf = StaticConf.get_instance().conf_values
    env.conf = SimpleNamespace(
        train_compressed_layer_only=train_only, device=conf.device, learning_rate=1e-3,
        num_epochs=2, compression_rates_dict={0: 1.0, 1: 0.9, 2: 0.8})
    env.current_model = model
    env.row_idx = 1
    env.mode = mode
    env.train_loader = _tiny_loader(seed=1)
    env.val_loader = _tiny_loader(n=8, seed=2)
    return env


def _handler(model):
    return ClassificationHandler(model, nn.CrossEntropyLoss())


def test_cg_freezes_everything_but_the_new_group_and_selects_on_val(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_REINIT_EDITED", "1")
    monkeypatch.setenv("SPECTRA_FT_REINIT_EPOCHS", "2")
    monkeypatch.setenv("SPECTRA_FT_REINIT_PATIENCE", "2")
    monkeypatch.setenv("SPECTRA_TRAIN_FT_EPOCHS", "1")           # must NOT cap the group training
    import src.run_recorder as recorder
    seen = []
    monkeypatch.setattr(recorder, "record", lambda kind, **kw: seen.append((kind, kw)))

    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = _prune_stream(model)
    edited = set(mwr.last_edited_param_ids)
    outside_before = {id(p): p.detach().clone() for p in mwr.model.parameters() if id(p) not in edited}
    env = _bare_env(mwr.model)
    handler = _handler(mwr.model)
    outcome = dict(mwr.last_prune_outcome)

    recipe = env._recover_after_prune(handler, mwr, outcome, is_to_train=True)
    assert recipe == "C-G" and outcome["ft_recipe"] == "C-G" and outcome["reinit"]["reinit"] is True
    trainable = {id(p) for p in mwr.model.parameters() if p.requires_grad}
    assert trainable == edited                                     # producers + norms + consumers
    for p in mwr.model.parameters():                               # frozen rest untouched
        if id(p) in outside_before:
            assert torch.equal(p.detach(), outside_before[id(p)])
    fts = [kw for kind, kw in seen if kind == "finetune"]
    assert len(fts) == 1 and fts[0]["select"] == "val" and fts[0]["phase"] == "C-G group"
    assert fts[0]["epochs_budget"] == 2 and fts[0]["trainable_params"] == sum(
        p.numel() for p in mwr.model.parameters() if p.requires_grad)


def test_cgp_adds_a_full_net_low_lr_polish(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_REINIT_THEN_POLISH", "1")
    monkeypatch.setenv("SPECTRA_FT_REINIT_EPOCHS", "1")
    monkeypatch.setenv("SPECTRA_FT_POLISH_EPOCHS", "1")
    import src.run_recorder as recorder
    seen = []
    monkeypatch.setattr(recorder, "record", lambda kind, **kw: seen.append((kind, kw)))
    printed = []
    import src.utils as utils
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))

    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = _prune_stream(model)
    env = _bare_env(mwr.model)
    outcome = dict(mwr.last_prune_outcome)
    recipe = env._recover_after_prune(_handler(mwr.model), mwr, outcome, is_to_train=True)
    assert recipe == "C-G+" and outcome["ft_recipe"] == "C-G+"
    phases = [kw["phase"] for kind, kw in seen if kind == "finetune"]
    assert phases == ["C-G+ group", "C-G+ polish"]
    assert all(p.requires_grad for p in mwr.model.parameters())     # polish unfroze the net
    recipes = [line for line in printed if "Fine-tune recipe" in line]
    assert "lr=0.001" in recipes[0] and "lr=0.0001" in recipes[1]  # polish at 0.1x


def test_recipe_a_is_byte_identical_when_flags_are_off(monkeypatch):
    """Default path: full net trainable, train-loss selection, SPECTRA_TRAIN_FT_EPOCHS honoured."""
    monkeypatch.setenv("SPECTRA_TRAIN_FT_EPOCHS", "1")
    monkeypatch.setenv("SPECTRA_TRAIN_FT_PATIENCE", "1")
    import src.run_recorder as recorder
    seen = []
    monkeypatch.setattr(recorder, "record", lambda kind, **kw: seen.append((kind, kw)))
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = _prune_stream(model)
    env = _bare_env(mwr.model)
    outcome = dict(mwr.last_prune_outcome)
    recipe = env._recover_after_prune(_handler(mwr.model), mwr, outcome, is_to_train=True)
    assert recipe == "A" and "ft_recipe" not in outcome and "reinit" not in outcome
    assert all(p.requires_grad for p in mwr.model.parameters())
    fts = [kw for kind, kw in seen if kind == "finetune"]
    assert fts[0]["select"] == "train_loss" and fts[0]["epochs_budget"] == 1 and fts[0]["phase"] is None


def test_cg_on_a_masked_step_falls_back_to_recipe_a(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_REINIT_EDITED", "1")
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = ModelWithRows(model)
    mwr.last_group_edit = None
    mwr.last_edited_param_ids = None
    env = _bare_env(mwr.model)
    outcome = {"mode": "masked", "reason": "test"}
    recipe = env._recover_after_prune(_handler(mwr.model), mwr, outcome, is_to_train=False)
    assert recipe == "A" and outcome["ft_recipe"] == "A" and outcome["reinit"]["reinit"] is False
    assert all(p.requires_grad for p in mwr.model.parameters())


def test_identity_step_never_reaches_recovery():
    """Rate 1.0 skips prune and FT under every recipe (NetworkEnv.step: identity branch)."""
    import inspect
    from src import NetworkEnv as env_module
    src = inspect.getsource(env_module.NetworkEnv.step)
    head, tail = src.split("if compression_rate == 1:", 1)
    identity_branch = tail.split("else:", 1)[0]
    assert "_recover_after_prune" not in identity_branch
    assert "_recover_after_prune" in tail.split("else:", 1)[1]


# ------------------------------------------------------------------ feature-maps update

def test_full_refresh_updates_downstream_moments_but_row_refresh_does_not(monkeypatch):
    """
    NEON's last step: after an edit the *downstream* activations change too. The live
    row-span refresh keeps stale downstream moments; SPECTRA_REFRESH_ALL_FEATURES asks for
    update_indices=None, which recomputes every observable layer.
    """
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = ModelWithRows(model)
    fe = ActivationsStatisticsFE(_tiny_loader(n=8, seed=3), torch.device("cpu"))
    monkeypatch.setenv("SPECTRA_PROBE_BATCHES", "1")
    fe.num_probe_batches = 1
    before = [list(row) for row in fe.extract_feature_map(mwr, None)]
    with torch.no_grad():
        model.embed[0].weight.mul_(3.0)                    # edit the stem
    last_conv = _layer_index(mwr, model.layer3[2].conv2)
    stale = fe.extract_feature_map(mwr, [0])                # live behaviour: only the row's span
    assert stale[last_conv] == before[last_conv]           # downstream moments still cached
    fresh = fe.extract_feature_map(mwr, None)               # P8 flag → full refresh
    assert fresh[last_conv] != before[last_conv]
    assert fresh[0] != before[0]


# ------------------------------------------------------------------ policy contract

def test_policy_config_pins_the_recipe(tmp_path, monkeypatch):
    import a2c_agent_reinforce_runner as runner
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    for key in ("SPECTRA_FT_REINIT_EDITED", "SPECTRA_FT_REINIT_THEN_POLISH",
                "SPECTRA_REFRESH_ALL_FEATURES", "SPECTRA_FT_REINIT_SELECT"):
        assert key in A2CAgentReinforce.POLICY_CONTRACT_KEYS
    ckpt = tmp_path / "agent_checkpoints"
    ckpt.mkdir()
    (ckpt / "latest_best_actor.pt").write_bytes(b"x")
    (ckpt / "policy_config.json").write_text(json.dumps({
        "compression_rates": [1.0, 0.9, 0.8], "action_rankings": [None, None, None],
        "ft_recipe": "C-G+",
        "env": {"SPECTRA_FT_REINIT_EDITED": "1", "SPECTRA_FT_REINIT_THEN_POLISH": "1",
                "SPECTRA_REFRESH_ALL_FEATURES": "1"}, "passes": 2}), encoding="utf-8")
    monkeypatch.delenv("SPECTRA_POLICY_CONFIG", raising=False)
    monkeypatch.delenv("SPECTRA_EVAL_PASSES", raising=False)
    monkeypatch.setenv("SPECTRA_FT_REINIT_EDITED", "0")
    monkeypatch.setenv("SPECTRA_FT_REINIT_THEN_POLISH", "0")
    monkeypatch.setenv("SPECTRA_REFRESH_ALL_FEATURES", "0")
    args = SimpleNamespace(actor_checkpoint_path=str(ckpt / "latest_best_actor.pt"),
                           compression_rates=[1.0, 0.9, 0.8], action_rankings=None,
                           ranking_menu=None, passes=1)
    runner.apply_policy_config(args)
    assert fortify.ft_recipe(False) == "C-G+" and fortify.refresh_all_features() is True
    assert args.passes == 2


def test_write_policy_config_records_recipe(tmp_path, monkeypatch):
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    import src.logging_utils as logging_utils
    monkeypatch.setenv("SPECTRA_FT_REINIT_EDITED", "1")
    monkeypatch.setattr(logging_utils, "run_dir", lambda: str(tmp_path))
    agent = A2CAgentReinforce.__new__(A2CAgentReinforce)
    agent.conf = SimpleNamespace(
        compression_rates_dict={0: 1.0, 1: 0.9, 2: 0.8}, action_rankings_dict={0: None, 1: None, 2: None},
        ranking_menu=None, num_actions=3, passes=2, allowed_acc_reduction=10.0,
        train_compressed_layer_only=False)
    path = agent.write_policy_config()
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    assert cfg["ft_recipe"] == "C-G" and cfg["env"]["SPECTRA_FT_REINIT_EDITED"] == "1"
