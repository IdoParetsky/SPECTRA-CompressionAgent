"""CPU tests for A-LSQ, C-PCA, BN recalibration, and budget/STOP. All flags default off.

    python -m pytest tests/test_recovery_edits.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
import src.recovery_edits as recovery_edits  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("SPECTRA_FT_REINIT_EDITED", "SPECTRA_FT_REINIT_THEN_POLISH",
                "SPECTRA_FT_LSQ_CONSUMERS", "SPECTRA_FT_BN_RECAL", "SPECTRA_ACTION_MENU"):
        monkeypatch.delenv(key, raising=False)
    yield


def test_flags_default_off_and_do_not_rename_recipe_a():
    assert fortify.ft_pca_reinit() is False
    assert fortify.ft_lsq_consumers() is False
    assert fortify.ft_bn_recal() is False
    assert fortify.action_menu() == "rates"
    assert fortify.is_stop_rate(-1.0) is False
    assert fortify.ft_recipe(False) == "A"
    assert fortify.ft_reinit_edited() is False


def test_recipe_names_are_gated(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_REINIT_EDITED", "1")
    assert fortify.ft_recipe(False) == "C-G"
    assert fortify.ft_pca_reinit() is False
    monkeypatch.setenv("SPECTRA_FT_REINIT_EDITED", "pca")
    assert fortify.ft_pca_reinit() is True
    assert fortify.ft_reinit_edited() is False
    assert fortify.ft_recipe(False) == "C-PCA"
    monkeypatch.delenv("SPECTRA_FT_REINIT_EDITED")
    monkeypatch.setenv("SPECTRA_FT_LSQ_CONSUMERS", "1")
    assert fortify.ft_recipe(False) == "A-LSQ"


def test_lsq_reconstructs_a_conv_whose_dropped_channels_were_unused():
    torch.manual_seed(0)
    teacher = nn.Conv2d(4, 3, 3, padding=1, bias=False)
    with torch.no_grad():
        teacher.weight[:, 2:] = 0
    x = torch.randn(2, 4, 8, 8)
    y = teacher(x)
    student = nn.Conv2d(2, 3, 3, padding=1, bias=False)
    fitted = recovery_edits.lsq_conv2d_weight(x, y, [0, 1], student)
    assert fitted is not None
    weight, bias = fitted
    assert bias is None
    assert tuple(weight.shape) == tuple(student.weight.shape)
    with torch.no_grad():
        student.weight.copy_(weight)
    err = (student(x[:, [0, 1]]) - y).abs().mean().item()
    assert err < 1e-3            # ridge 1e-5 of the Gram diagonal: exact to ~1e-4, not to 1e-8


def test_channel_basis_keeps_a_rank_k_stream():
    torch.manual_seed(1)
    channels, k = 6, 2
    true = torch.linalg.qr(torch.randn(channels, k)).Q
    latent = torch.randn(4, k, 5, 5)
    y = torch.einsum("ck,nkhw->nchw", true, latent)
    basis = recovery_edits.channel_basis(y, k)
    assert basis is not None and tuple(basis.shape) == (channels, k)
    centered = y - y.mean(dim=(0, 2, 3), keepdim=True)
    latent_hat = torch.einsum("nchw,ck->nkhw", centered, basis)
    recon = torch.einsum("nkhw,ck->nchw", latent_hat, basis)
    assert (recon - centered).abs().mean().item() < 1e-4


def test_consumer_mix_is_the_projected_old_map():
    torch.manual_seed(2)
    weight = torch.randn(3, 5, 1, 1)
    basis = torch.linalg.qr(torch.randn(5, 2)).Q
    mixed = recovery_edits.mix_in_channels(weight, basis)
    assert tuple(mixed.shape) == (3, 2, 1, 1)
    restored = torch.einsum("okhw,ik->oihw", mixed, basis)
    projected = torch.einsum("oihw,ij->ojhw", weight, basis @ basis.T)
    assert torch.allclose(restored, projected, atol=1e-5)


def test_budget_keep_rate_is_a_network_fraction():
    assert fortify.budget_keep_rate(0.10, 0.0) == 1.0
    assert fortify.budget_keep_rate(0.0, 0.02) == 1.0
    assert fortify.budget_keep_rate(0.10, 0.02) == pytest.approx(0.8)
    assert fortify.budget_keep_rate(0.10, 0.10) == pytest.approx(0.0)
    assert fortify.budget_keep_rate(0.10, 0.50) == pytest.approx(0.0)


def test_stop_is_legal_on_a_stem_only_under_the_budget_menu(monkeypatch):
    rates = {0: 1.0, 1: -1.0}
    plain = fortify.legal_action_mask(rates, row_index=0, alive_count=8, device="cpu",
                                      force_identity=True)
    assert bool(plain[0]) and not bool(plain[1])
    monkeypatch.setenv("SPECTRA_ACTION_MENU", "budget")
    masked = fortify.legal_action_mask(rates, row_index=0, alive_count=8, device="cpu",
                                       force_identity=True)
    assert bool(masked[0]) and bool(masked[1])
    assert fortify.is_stop_rate(-1.0) is True
    assert fortify.is_stop_rate(0.02) is False


def test_apply_lsq_writes_the_fitted_kernel():
    torch.manual_seed(3)
    conv = nn.Conv2d(4, 2, 1, bias=False)
    with torch.no_grad():
        conv.weight[:, 2:] = 0
    x = torch.randn(2, 4, 4, 4)
    y = conv(x).detach()
    # The pruned module keeps the two live input channels and starts from a wrong slice.
    pruned = nn.Conv2d(2, 2, 1, bias=False)
    with torch.no_grad():
        pruned.weight.zero_()
    holder = SimpleNamespace(
        all_layers=[pruned],
        last_group_edit={"consumer_in_idx": {0: [0, 1]}, "new_width": 2},
    )
    captured = {"io": {0: {"x": x, "y": y}}}
    summary = recovery_edits.apply_lsq(holder, captured)
    assert summary["lsq"] == 1
    err = (pruned(x[:, [0, 1]]) - y).abs().mean().item()
    assert err < 1e-3


# ------------------------------------------------------------------ Fable 27 Sep review additions

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src.Configuration.StaticConf import StaticConf  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
from src.NetworkEnv import NetworkEnv, prune_current_model  # noqa: E402
import src.action_costs as action_costs  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402

BUDGET_RATES = {0: 1.0, 1: 0.01, 2: 0.02, 3: 0.04, 4: -1.0}


def _tiny_loader(n=64, seed=0, classes=10):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=32)


def _layer_index(mwr, module):
    return next(i for i, layer in enumerate(mwr.all_layers) if layer is module)


def _row_of_layer(mwr, layer_idx):
    return next(row for row, main in mwr.row_to_main_layer.items() if main == layer_idx)


def _bare_env(model):
    env = NetworkEnv.__new__(NetworkEnv)
    conf = StaticConf.get_instance().conf_values
    env.conf = SimpleNamespace(train_compressed_layer_only=False, device=conf.device, learning_rate=1e-3,
                               num_epochs=1, compression_rates_dict=dict(BUDGET_RATES))
    env.current_model = model
    env.row_idx = 1
    env.mode = "agent_train"
    env.train_loader = _tiny_loader(seed=1)
    env.val_loader = _tiny_loader(n=32, seed=2)
    env._pass_locked_layers = set()
    env._reset_episode_reward_stats()
    return env


def test_effective_rates_map_budget_onto_keep_rates_and_flag_infeasible(monkeypatch):
    monkeypatch.setenv("SPECTRA_ACTION_MENU", "budget")
    mapped = fortify.effective_rates(BUDGET_RATES, group_param_fraction=0.10)
    assert mapped[0] == (1.0, False, True)
    assert mapped[4] == (1.0, True, True)                      # STOP: identity step that ends the episode
    assert mapped[1][0] == pytest.approx(0.9) and mapped[1][2]
    assert mapped[2][0] == pytest.approx(0.8) and mapped[2][2]
    assert mapped[3][0] == pytest.approx(0.6) and mapped[3][2]
    poor = fortify.effective_rates(BUDGET_RATES, group_param_fraction=0.02)
    assert poor[3][0] == 0.0 and poor[3][2] is False             # 4% through a 2% group: infeasible
    # one channel of a 4-wide group owning 50% of the net removes 12.5% — far past a 1% ask
    coarse = fortify.effective_rates(BUDGET_RATES, group_param_fraction=0.50, group_width=4)
    assert coarse[1][2] is False and coarse[2][2] is False and coarse[3][2] is False
    fine = fortify.effective_rates(BUDGET_RATES, group_param_fraction=0.50, group_width=64)
    assert fine[1][2] and fine[3][2]                              # 0.78% per channel fits a 1% ask
    monkeypatch.delenv("SPECTRA_ACTION_MENU")
    plain = fortify.effective_rates({0: 1.0, 1: 0.9}, group_param_fraction=0.10)
    assert plain == {0: (1.0, False, True), 1: (0.9, False, True)}


def test_budget_mask_judges_the_mapped_keep_rate(monkeypatch):
    monkeypatch.setenv("SPECTRA_ACTION_MENU", "budget")
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    # group owns 10 % of the net, 16 channels alive: 1/2/4 % → keep 0.9/0.8/0.6, all realisable
    mask = fortify.legal_action_mask(BUDGET_RATES, row_index=5, alive_count=16, device="cpu",
                                     group_param_fraction=0.10)
    assert mask.tolist() == [True, True, True, True, True]
    # group owns 2 %: the 4 % request is infeasible; STOP stays legal
    mask = fortify.legal_action_mask(BUDGET_RATES, row_index=5, alive_count=16, device="cpu",
                                     group_param_fraction=0.02)
    assert mask.tolist() == [True, True, False, False, True]   # 2 % = the whole group: infeasible too
    # locked group: identity + STOP only, budget cuts never leak through
    mask = fortify.legal_action_mask(BUDGET_RATES, row_index=5, alive_count=16, device="cpu",
                                     force_identity=True, group_param_fraction=0.10)
    assert mask.tolist() == [True, False, False, False, True]
    # 1 % through a 50 % group on a 4-channel row: keep 0.98 → width 4 → no-op → illegal
    mask = fortify.legal_action_mask(BUDGET_RATES, row_index=5, alive_count=4, device="cpu",
                                     group_param_fraction=0.50)
    assert mask[1].item() is False


def test_action_costs_price_the_mapped_budget_action(monkeypatch):
    monkeypatch.setenv("SPECTRA_ACTION_MENU", "budget")
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    target = model.layer3[0].conv1                         # a wide, singleton group
    costs = action_costs.estimate_action_costs(model, target, list(BUDGET_RATES.values()), (3, 32, 32),
                                               groups=groups, device=torch.device("cpu"))
    assert costs.shape == (5, action_costs.ACTION_FEATURE_DIM)
    assert costs[0, 1].item() == 0.0 and costs[4, 1].item() == 0.0     # identity and STOP cost nothing
    assert costs[4, 0].item() == pytest.approx(-1.0)                     # raw action value kept in slot 0
    total = sum(p.numel() for p in model.parameters())
    group = channel_groups.group_of(groups, target)
    owned = action_costs.group_removal_cost(group, group.width, {})[0] / total
    priced = 0
    for i, req in ((1, 0.01), (2, 0.02), (3, 0.04)):
        keep, _, feasible = fortify.effective_rates({0: req}, owned, group_width=group.width)[0]
        if feasible:
            # priced within one channel of the request (rounding of the realised width)
            assert abs(costs[i, 1].item() - req) <= owned / max(group.width, 1) + 1e-6
            priced += 1
        else:
            assert costs[i, 1].item() == 0.0
    assert priced >= 1
    del mwr


def test_lsq_is_exact_when_well_determined_and_handles_linear():
    torch.manual_seed(4)
    teacher = nn.Conv2d(6, 5, 3, padding=1, bias=True)
    with torch.no_grad():
        teacher.weight[:, 4:] = 0                          # two dead input channels
    x = torch.randn(16, 6, 8, 8)
    y = teacher(x)
    student = nn.Conv2d(4, 5, 3, padding=1, bias=True)
    weight, bias = recovery_edits.lsq_conv2d_weight(x, y, [0, 1, 2, 3], student, ridge=0.0)
    with torch.no_grad():
        student.weight.copy_(weight)
        student.bias.copy_(bias)
    assert (student(x[:, :4]) - y).abs().mean().item() < 1e-3
    lin = nn.Linear(8, 3)
    with torch.no_grad():
        lin.weight[:, 6:] = 0
    xl = torch.randn(64, 8)
    yl = lin(xl)
    stud = nn.Linear(6, 3)
    w, b = recovery_edits.lsq_linear_weight(xl, yl, list(range(6)), stud, ridge=0.0)
    with torch.no_grad():
        stud.weight.copy_(w)
        stud.bias.copy_(b)
    assert (stud(xl[:, :6]) - yl).abs().mean().item() < 1e-4
    # ridge keeps an underdetermined fit finite
    few = torch.randn(1, 6, 2, 2)
    fitted = recovery_edits.lsq_conv2d_weight(few, teacher(few), [0, 1, 2, 3], student, ridge=1e-3)
    assert fitted is not None and torch.isfinite(fitted[0]).all()


def test_capture_accumulates_calibration_batches(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_CALIB_IMAGES", "8")
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    mwr = ModelWithRows(model)
    row = _row_of_layer(mwr, _layer_index(mwr, model.layer1[0].conv2))
    cap = recovery_edits.capture_pre_prune(mwr, row, _tiny_loader(n=64), torch.device("cpu"), n_batches=2)
    assert cap is not None and cap["producer_indices"]
    some = next(iter(cap["io"].values()))
    assert some["x"].shape[0] == 16 and some["y"].shape[0] == 16       # 2 batches × 8 images
    conv1_idx = _layer_index(mwr, model.layer1[1].conv1)
    assert conv1_idx in cap["io"]                                       # consumers are captured too


def _prune_stream_with_capture(model, flags_env):
    mwr = ModelWithRows(model)
    row = _row_of_layer(mwr, _layer_index(mwr, model.layer1[0].conv2))
    env = _bare_env(model)
    env.row_idx = row + 1
    env._pre_prune_io = recovery_edits.capture_pre_prune(mwr, row, env.train_loader, torch.device("cpu"),
                                                         n_batches=2)
    assert env._pre_prune_io is not None
    prune_current_model(mwr, 0.8, row, quiet=True, record=False, input_shape=(3, 32, 32))
    assert mwr.last_prune_outcome["mode"] == "structural"
    return env, mwr, dict(mwr.last_prune_outcome)


def test_a_lsq_refits_every_stream_consumer_on_a_residual_net(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_LSQ_CONSUMERS", "1")
    monkeypatch.setenv("SPECTRA_FT_CALIB_IMAGES", "16")
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    env, mwr, outcome = _prune_stream_with_capture(model, {})
    consumers = dict(mwr.last_group_edit["consumer_in_idx"])
    # reconstruction error of the sliced (recipe A) consumers before the refit
    cap = env._pre_prune_io
    def _err(idx):
        m = mwr.all_layers[idx]
        s = cap["io"][idx]
        with torch.no_grad():
            return (m(s["x"][:, consumers[idx]]) - s["y"]).abs().mean().item()
    before = {idx: _err(idx) for idx in consumers if idx in cap["io"] and isinstance(mwr.all_layers[idx], nn.Conv2d)}
    handler = ClassificationHandler(mwr.model, nn.CrossEntropyLoss())
    recipe = env._recover_after_prune(handler, mwr, outcome, is_to_train=False)
    assert recipe == "A-LSQ" and outcome["ft_recipe"] == "A-LSQ"
    assert outcome["lsq"]["lsq"] >= len(before) and outcome["lsq"]["skipped"] == 0
    after = {idx: _err(idx) for idx in before}
    assert all(after[i] <= before[i] + 1e-6 for i in before)
    assert sum(after.values()) < sum(before.values())
    assert mwr.model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_c_pca_rotates_every_producer_and_resets_group_norms(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_REINIT_EDITED", "pca")
    monkeypatch.setenv("SPECTRA_FT_BN_RECAL", "1")
    monkeypatch.setenv("SPECTRA_FT_CALIB_IMAGES", "16")
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    env, mwr, outcome = _prune_stream_with_capture(model, {})
    edit = mwr.last_group_edit
    sliced = {i: mwr.all_layers[i].weight.detach().clone() for i in edit["producers"]}
    handler = ClassificationHandler(mwr.model, nn.CrossEntropyLoss())
    recipe = env._recover_after_prune(handler, mwr, outcome, is_to_train=False)
    assert recipe == "C-PCA" and outcome["ft_recipe"] == "C-PCA"
    pca = outcome["pca"]
    assert pca["pca_producers"] == len(edit["producers"]) and pca["skipped"] == 0
    assert pca["pca_consumers"] == len(edit["consumers"]) and pca["width"] == edit["new_width"]
    for i, old in sliced.items():
        assert not torch.allclose(mwr.all_layers[i].weight.detach(), old)   # generated, not sliced
    for i in edit["norms"]:
        norm = mwr.all_layers[int(i)]
        assert torch.all(norm.weight == 1) and torch.all(norm.bias == 0)
    assert outcome["bn_recal"] > 0
    assert torch.isfinite(mwr.model(torch.randn(2, 3, 32, 32))).all()


def test_c_pca_skips_a_group_with_a_depthwise_owner():
    holder = SimpleNamespace(all_layers=[], last_group_edit={"depthwise": [3], "producers": [2], "new_width": 2})
    summary = recovery_edits.apply_pca(holder, {"io": {}, "weights": {}, "bias": {}, "producer_indices": [2, 3]})
    assert summary["skipped"] == 1 and summary["reason"] == "depthwise owner"


def test_stop_reward_is_in_per_step_units(monkeypatch):
    assert fortify.stop_reward_scale() == 100.0
    monkeypatch.setenv("SPECTRA_STOP_REWARD_SCALE", "0")
    assert fortify.stop_reward_scale() == 0.0


def test_warmcos_schedule_warms_up_then_decays(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_OPTIM", "adamw")
    monkeypatch.setenv("SPECTRA_FT_SCHEDULE", "warmcos")
    monkeypatch.setenv("SPECTRA_FT_LR_MIN", "1e-5")
    captured = {}
    real = torch.optim.lr_scheduler.LambdaLR

    class _Spy(real):
        def __init__(self, optimizer, lr_lambda, **kw):
            captured["fn"] = lr_lambda
            super().__init__(optimizer, lr_lambda, **kw)

    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", _Spy)
    printed = []
    import src.utils as utils
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))
    handler = ClassificationHandler(model, nn.CrossEntropyLoss())
    loader = _tiny_loader(n=64)                            # 2 batches per epoch
    handler.train_model(loader, max_epochs=3, patience=10)
    fn = captured["fn"]
    assert fn(0) < fn(1) <= 1.0                            # warmup over the first epoch (2 steps)
    assert fn(1) == pytest.approx(1.0)
    assert fn(6) == pytest.approx(1e-5 / 1e-3, abs=1e-6)   # cosine floor after the last step
    assert fn(2) >= fn(3) > fn(4) > fn(5) > fn(6)
    recipe = [line for line in printed if "Fine-tune recipe" in line][0]
    assert "optim=adamw" in recipe and "schedule=warmcos" in recipe and "wd=0.0005" in recipe


def test_radam_is_an_accepted_optimizer(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_OPTIM", "radam")
    monkeypatch.setenv("SPECTRA_FT_SCHEDULE", "warmcos")
    monkeypatch.setenv("SPECTRA_FT_WARMUP_EPOCHS", "0")
    printed = []
    import src.utils as utils
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))
    ClassificationHandler(model, nn.CrossEntropyLoss()).train_model(_tiny_loader(n=64), max_epochs=1, patience=10)
    assert any("optim=radam" in line for line in printed)


def test_bn_recalibration_moves_stale_running_stats_and_restores_momentum():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1, bias=False)
            self.bn = nn.BatchNorm2d(4)

        def forward(self, x):
            return self.bn(self.conv(x))

    net = Net()
    with torch.no_grad():
        net.bn.running_mean.fill_(5.0)
    original_momentum = net.bn.momentum
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(16, 3, 8, 8)), batch_size=8)
    moved = recovery_edits.recalibrate_batchnorm(net, loader, torch.device("cpu"), n_batches=2)
    assert moved == 1
    assert net.bn.momentum == original_momentum
    assert not torch.allclose(net.bn.running_mean, torch.full((4,), 5.0))
