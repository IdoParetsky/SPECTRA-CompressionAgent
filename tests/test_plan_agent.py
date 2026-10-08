"""Plan-as-action agent: param model, decode, masked cut, policy, trainer and the alloc ``agent`` kind."""

import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, plan_agent, plan_trainer, recovery_edits, state_dump  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402
from spectra_models_instantiation import (densenet_cifar, mobilenetv2_chenyaofo, resnet_chenyaofo,  # noqa: E402
                                          thin_res_net, vgg_chenyaofo)

SHAPE = (3, 32, 32)
RATES = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
V10_FLAGS = {"SPECTRA_FORTIFY": "1", "SPECTRA_BUDGET_IN_STATE": "1", "SPECTRA_STATE_SLACK": "1",
             "SPECTRA_STATE_GROUPCOST": "1", "SPECTRA_FIXED_TARGET": "1", "SPECTRA_STATE_SENS": "1"}
LAYOUT = [("base", 0, 38), ("fortify", 38, 42), ("budget", 42, 43), ("slack", 43, 45),
          ("groupcost", 45, 49), ("target", 49, 51), ("sens", 51, 53), ("action", 53, 63)]
ZOO = {
    "thin_r20_w4": lambda: thin_res_net.resnet20(num_classes=10, large_input=False, width=4),
    "r20_chenyaofo": lambda: resnet_chenyaofo.resnet20(10, False),
    "mbv2_x05": lambda: mobilenetv2_chenyaofo.mobilenet_v2x05(10, False),
    "vgg11_bn": lambda: vgg_chenyaofo.vgg11_bn(10, False),
    "densenet40": lambda: densenet_cifar.densenet40(10, False),
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SPECTRA_PLAN_") or key.startswith("SPECTRA_ALLOC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


@pytest.fixture
def v10_env(monkeypatch):
    for key, value in V10_FLAGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SPECTRA_STATE_TOKENS", raising=False)


def toy_state(num_layers, width=63, seed=0):
    from src.action_costs import ACTION_FEATURE_DIM
    g = torch.Generator().manual_seed(seed)
    cids = torch.arange(num_layers) // 2
    return {"layer_features": torch.randn(num_layers, width, generator=g),
            "layer_types": torch.full((num_layers,), 2, dtype=torch.long),
            "coupling_ids": cids, "block_ids": cids, "target_index": 0,
            "action_costs": torch.rand(5, ACTION_FEATURE_DIM, generator=g)}


def _randomize_norms(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            n = module.num_features
            module.weight.data = torch.rand(n, generator=g) + 0.5
            module.bias.data = torch.randn(n, generator=g) * 0.1
            module.running_mean = torch.randn(n, generator=g) * 0.1
            module.running_var = torch.rand(n, generator=g) + 0.5


def _plan(model):
    return group_sensitivity.group_plan(ModelWithRows(model))


@pytest.mark.parametrize("name", list(ZOO))
def test_param_model_and_masked_cut_match_the_real_cut(name):
    torch.manual_seed(0)
    model = ZOO[name]().eval()
    _randomize_norms(model)
    plan = _plan(model)
    pm = plan_agent.ParamModel(model, plan)
    cutter = plan_agent.MaskedCut(model)
    assert cutter.rows == pm.rows and pm.total0 == utils.calc_num_parameters(model)
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(4, *SHAPE, generator=gen)
    for _trial in range(1 if name == "densenet40" else 2):
        keeps = {row: (1.0 if float(torch.rand(1, generator=gen)) < 0.2
                       else float(torch.empty(1).uniform_(0.15, 0.95, generator=gen))) for row in pm.rows}
        widths = plan_agent.widths_of(keeps, pm.widths0)
        rates = plan_agent.rates_of(widths, pm.widths0)
        real = alloc_walk.cut_to(model, plan, rates, SHAPE).eval()
        assert pm.params(widths) == utils.calc_num_parameters(real)
        assert alloc_walk.group_widths(real, pm.rows) == widths
        masked = cutter.cut(rates).eval()
        torch.testing.assert_close(masked(x), real(x), atol=1e-4, rtol=1e-4)
    assert pm.params(pm.widths0) == pm.total0


def test_widths_and_rates_round_trip_through_target_width():
    widths0 = {0: 16, 5: 32, 9: 7}
    assert plan_agent.widths_of({0: 1.0, 5: 0.34, 9: 0.0}, widths0) == {0: 16, 5: 11, 9: 1}
    widths = {0: 16, 5: 11, 9: 1}
    rates = plan_agent.rates_of(widths, widths0)
    assert rates[0] == 1.0
    for row in (5, 9):
        assert pruning.target_width(widths0[row], rates[row]) == widths[row]


def test_decode_hits_the_target_follows_the_scores_and_ignores_a_shift():
    model = ZOO["thin_r20_w4"]().eval()
    pm = plan_agent.ParamModel(model, _plan(model))
    for kappa in (0.4, 0.6, 0.8):
        _widths, info = plan_agent.decode([0.0] * len(pm.rows), pm, kappa)
        assert abs(info["kept"] - kappa) < 0.02
        keeps = list(info["keeps"].values())
        assert max(keeps) - min(keeps) < 1e-12
    z = [0.3 * k for k in range(len(pm.rows))]
    widths, info = plan_agent.decode(z, pm, 0.6)
    keeps = [info["keeps"][row] for row in pm.rows]
    assert all(a <= b + 1e-12 for a, b in zip(keeps, keeps[1:])) and keeps[0] < keeps[-1]
    assert abs(info["kept"] - 0.6) < 0.02
    assert plan_agent.decode([v + 5.0 for v in z], pm, 0.6)[0] == widths
    assert min(info["keeps"].values()) >= 0.1 - 1e-12


def test_scale_decode_is_the_alloc_family_and_holds_groups():
    model = ZOO["thin_r20_w4"]().eval()
    plan = _plan(model)
    pm = plan_agent.ParamModel(model, plan)
    held = {row for group, row in plan if len(group.producers) > 1}
    assert held
    widths, info = plan_agent.scale_decode({row: 1.0 for row in pm.rows}, pm, 0.6, held=held)
    assert abs(info["kept"] - 0.6) < 0.02
    assert all(info["keeps"][row] == 1.0 and widths[row] == pm.widths0[row] for row in held)
    cut = {info["keeps"][row] for row in pm.rows if row not in held}
    assert max(cut) - min(cut) < 1e-12


def test_polish_closes_the_rounding_gap_without_crossing_floors():
    model = ZOO["thin_r20_w4"]().eval()
    pm = plan_agent.ParamModel(model, _plan(model))
    keeps = {row: 0.62 for row in pm.rows}
    rounded = plan_agent.widths_of(keeps, pm.widths0)
    target = 0.6 * pm.total0
    widths, p = plan_agent.polish(keeps, rounded, pm, target, 0.1)
    assert abs(p - target) <= abs(pm.params(rounded) - target)
    assert p == pm.params(widths) and abs(p / pm.total0 - 0.6) < 0.02
    assert all(max(1, round(0.1 * pm.widths0[r])) <= widths[r] <= pm.widths0[r] for r in pm.rows)
    frozen = set(pm.rows[:3])
    widths, _p = plan_agent.polish({r: 0.1 for r in pm.rows}, plan_agent.widths_of({r: 0.1 for r in pm.rows},
                                   pm.widths0), pm, 0.9 * pm.total0, 0.1, fixed=frozen)
    assert all(widths[r] == max(1, round(0.1 * pm.widths0[r])) for r in frozen)


def _loader(n=64, batch=16, seed=0, shuffle=True):
    g = torch.Generator().manual_seed(seed)
    data = torch.utils.data.TensorDataset(torch.randn(n, *SHAPE, generator=g), torch.randint(0, 10, (n,), generator=g))
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=shuffle)


def _instance(zero=()):
    torch.manual_seed(0)
    model = ZOO["thin_r20_w4"]().eval()
    _randomize_norms(model)
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    plan = group_sensitivity.group_plan(mwr, groups)
    rows = state_dump.token_rows(mwr, groups, plan)
    sens = {row: 0.01 * (k + 1) for k, (_g, row) in enumerate(plan)}
    return plan_agent.NetInstance("toy.pt", model, plan, toy_state(len(rows)), rows, LAYOUT, _loader(),
                                  _loader(seed=1, shuffle=False), "cpu", zero=zero, sens=sens, input_shape=SHAPE)


def test_instance_check_agrees_with_the_real_cut(v10_env):
    inst = _instance()
    chk = inst.check(0.6)
    assert chk["analytic"] == pytest.approx(chk["real"], abs=1e-9)
    assert chk["masked_r"] == pytest.approx(chk["real_r"]) and chk["dead"] == 0
    group = next(g for g, _row in inst.cutter.plan if len(g.producers) == 1 and not g.depthwise)
    with torch.no_grad():
        group.producers[0].weight[0] = 0
    assert inst.check(0.6)["dead"] == 0
    with torch.no_grad():
        group.producers[0].weight[0] = 0
    inst.cutter.origin = {k: v.clone() for k, v in inst.cutter.work.state_dict().items()}
    assert inst.check(0.6)["dead"] == 1


def test_instance_reward_is_the_real_cut_read_on_the_val_batches(v10_env):
    inst = _instance()
    assert inst.n_groups == len(inst.pm.rows) and int(inst.token_mask.sum()) == inst.token_k.numel()
    batches = inst.batches("bn4", seed=3)
    assert len(batches) == 4 and torch.equal(inst.batches("bn4", seed=3)[0], batches[0])
    assert inst.batches("cut", seed=3) == []
    widths, _info = plan_agent.decode([0.1 * k for k in range(inst.n_groups)], inst.pm, 0.6)
    rates = plan_agent.rates_of(widths, inst.pm.widths0)
    real = alloc_walk.cut_to(inst.model, inst.plan, rates, SHAPE).eval()
    r_cut = inst.reward(widths, "cut", [])
    assert r_cut == pytest.approx(100.0 * (plan_agent.accuracy(real, inst.val) - inst.origin_val))
    r_bn = inst.reward(widths, "bn4", batches)
    recovery_edits.recalibrate_batchnorm(real, batches, "cpu", 4)
    assert r_bn == pytest.approx(100.0 * (plan_agent.accuracy(real, inst.val) - inst.origin_val))
    assert inst.reward(widths, "cut", []) == pytest.approx(r_cut)
    for kind in ("uniform", "sens", "inner"):
        _w, info = inst.reference(kind, 0.6)
        assert abs(info["kept"] - 0.6) < 0.03
    _w, inner = inst.reference("inner", 0.6)
    assert inst.held and all(inner["keeps"][row] == 1.0 for row in inst.held)
    with pytest.raises(ValueError):
        inst.reward(widths, "ft1", [])


def test_plan_state_sets_the_target_and_clears_the_action_slots(v10_env):
    spans = plan_agent.spans_of(LAYOUT)
    state = toy_state(6)
    out = plan_agent.plan_state(state, spans, 0.45, zero=("sens",))
    feats = out["layer_features"]
    assert torch.all(feats[:, 49] == 0.45) and torch.all(feats[:, 50] == 1.0)
    assert torch.count_nonzero(feats[:, 51:63]) == 0
    assert torch.equal(feats[:, :49], state["layer_features"][:, :49])
    assert torch.count_nonzero(state["layer_features"][:, 53:63]) > 0


def test_token_index_maps_tokens_to_their_group():
    mask, k = plan_agent.token_index([5, -1, 5, 9, 12], [5, 9, 12], "cpu")
    assert mask.tolist() == [True, False, True, True, True] and k.tolist() == [0, 0, 1, 2]


def test_policy_starts_uniform_centres_its_scores_and_learns_through_the_head(v10_env):
    state = toy_state(8)
    policy = plan_agent.PlanPolicy(63)
    assert all(m.p == 0.0 for m in policy.modules() if isinstance(m, nn.Dropout))
    mask = torch.tensor([True] * 7 + [False])
    k = torch.tensor([0, 0, 1, 1, 2, 2, 3])
    mu = policy(state, mask, k, 4)
    assert torch.equal(mu, torch.zeros(4))
    z = plan_agent.sample_scores(mu, 0.5, 6, torch.Generator().manual_seed(0))
    assert z.shape == (6, 4) and not z.requires_grad
    loss = -(torch.randn(6) * plan_agent.log_prob(z, mu, 0.5)).mean()
    loss.backward()
    assert float(policy.head.weight.grad.abs().sum()) > 0
    with torch.no_grad():
        policy.head.weight.normal_()
        policy.head.bias.fill_(3.0)
    mu = policy(state, mask, k, 4)
    assert abs(float(mu.sum())) < 1e-5 and float(mu.std()) > 0


def test_policy_checkpoint_round_trip(tmp_path, v10_env):
    policy = plan_agent.PlanPolicy(63)
    with torch.no_grad():
        policy.head.weight.normal_()
    path = tmp_path / "plan_agent" / "policy.pt"
    plan_agent.save_policy(policy, str(path), tokens="layer", zero=["sens"], k_min=0.1)
    loaded, blob = plan_agent.load_policy(str(path))
    assert blob["zero"] == ["sens"] and blob["feature_dim"] == 63 and not loaded.training
    state = toy_state(8)
    mask, k = torch.ones(8, dtype=torch.bool), torch.arange(8) // 2
    torch.testing.assert_close(loaded(state, mask, k, 4), policy(state, mask, k, 4))


def test_trainer_config_defaults_schedule_and_errors(monkeypatch):
    assert not plan_trainer.enabled()
    cfg = plan_trainer.config()
    assert (cfg.instances, cfg.k, cfg.proxy, cfg.tokens, cfg.kappa_lo, cfg.kappa_hi, cfg.batch) == \
        (300, 8, "bn32", "layer", 0.35, 0.85, 4)
    assert cfg.summary_kappas == (0.4, 0.6, 0.8) and cfg.max_minutes == 0.0
    assert plan_trainer.sigma_at(cfg, 0) == 0.5 and plan_trainer.sigma_at(cfg, 300) == pytest.approx(0.2)
    assert plan_trainer.sigma_at(cfg, 90) == pytest.approx(0.35)
    for key, value in (("SPECTRA_PLAN_PROXY", "ft1"), ("SPECTRA_PLAN_K", "1"), ("SPECTRA_PLAN_KAPPA", "0.9,0.5"),
                       ("SPECTRA_PLAN_TOKENS", "bert")):
        monkeypatch.setenv(key, value)
        with pytest.raises(ValueError):
            plan_trainer.config()
        monkeypatch.delenv(key)
    monkeypatch.setenv("SPECTRA_PLAN_TRAIN", "1")
    assert plan_trainer.enabled()


def test_trainer_runs_end_to_end_on_a_toy_instance(monkeypatch, tmp_path, v10_env):
    inst = _instance()
    monkeypatch.setattr(plan_agent.NetInstance, "from_env", classmethod(lambda cls, env, p, m, l, **kw: inst))
    monkeypatch.setattr(plan_trainer, "out_dir", lambda: str(tmp_path))
    for key, value in {"SPECTRA_PLAN_INSTANCES": "5", "SPECTRA_PLAN_K": "3", "SPECTRA_PLAN_BATCH": "2",
                       "SPECTRA_PLAN_PROXY": "bn2", "SPECTRA_PLAN_REF_EVERY": "2", "SPECTRA_PLAN_SAVE_EVERY": "4",
                       "SPECTRA_PLAN_NETS": "toy"}.items():
        monkeypatch.setenv(key, value)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    plan_trainer.run(None, [("toy.pt", (None, None)), ("other.pt", (None, None))])
    with open(tmp_path / "plan_train.jsonl", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    steps = [r for r in records if "it" in r]
    assert len(steps) == 5 and all(len(r["rewards"]) == 3 for r in steps)
    assert "mean_plan" in steps[0] and "refs" in steps[1] and "refs" not in steps[2]
    assert set(steps[1]["refs"]) == {"uniform", "sens", "inner"}
    assert {"policy_it00004.pt", "policy_it00005.pt", "policy_latest.pt"} <= set(os.listdir(tmp_path))
    assert len([r for r in records if "summary" in r]) == 3
    _policy, blob = plan_agent.load_policy(str(tmp_path / "policy_latest.pt"))
    assert blob["tokens"] == "layer" and blob["proxy"] == "bn2" and blob["nets"] == ["toy.pt"]
    assert any(line.startswith("[plan] DONE 5 instances") for line in lines)
    assert sum(line.startswith("[plan] net ") for line in lines) == 1


def test_trainer_refuses_when_no_network_matches(monkeypatch, v10_env):
    monkeypatch.setenv("SPECTRA_PLAN_NETS", "nothing")
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        plan_trainer.run(None, [("toy.pt", (None, None))])


def test_alloc_agent_kind_needs_a_checkpoint_and_plans_through_the_policy(monkeypatch):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "agent")
    assert alloc_walk.kind() == "agent"
    with pytest.raises(ValueError):
        alloc_walk.agent_path()
    monkeypatch.setenv("SPECTRA_PLAN_AGENT", "/x/runs/job1/plan_agent/policy_latest.pt")
    calls = []

    def fake(env, target, path, k_min):
        calls.append((target, path, k_min))
        return {5: 3}, {"kind": "agent", "alpha": 0.0, "target": target, "kept": 0.57, "keeps": {5: 0.5},
                        "origin_widths": {5: 6}, "sens": {5: 0.1}, "held": 0, "policy": "plan_agent/policy_latest.pt"}

    monkeypatch.setattr(plan_agent, "plan_for_env", fake)
    env = types.SimpleNamespace(selected_net_path="net.pt", current_model=ZOO["thin_r20_w4"]().eval(),
                                conf=types.SimpleNamespace(device="cpu"), target_keep=0.6, train_loader=None)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    cache = alloc_walk._state(env)
    assert cache["widths"] == {5: 3}
    assert calls == [(pytest.approx(0.58), "/x/runs/job1/plan_agent/policy_latest.pt", 0.1)]
    assert "agent plan_agent/policy_latest.pt" in lines[0]


def test_plan_for_env_decodes_the_saved_policy_on_the_reset_state(monkeypatch, tmp_path, v10_env):
    model = ZOO["thin_r20_w4"]().eval()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    plan = group_sensitivity.group_plan(mwr, groups)
    rows = state_dump.token_rows(mwr, groups, plan)
    state = toy_state(len(rows))
    monkeypatch.setattr(state_dump, "encode_origin", lambda env, m, g: dict(state))
    policy = plan_agent.PlanPolicy(63)
    with torch.no_grad():
        policy.head.weight.normal_()
    path = tmp_path / "plan_agent" / "policy_it00300.pt"
    plan_agent.save_policy(policy, str(path), tokens="layer", zero=["sens"], k_min=0.1)
    env = types.SimpleNamespace(conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES),
                                current_model=model, target_keep=0.6,
                                _dependency_groups=lambda m: channel_groups.build_channel_groups(m.model))
    widths, info = plan_agent.plan_for_env(env, 0.58, str(path))
    pm = plan_agent.ParamModel(model, plan)
    assert abs(info["kept"] - 0.58) < 0.02 and set(widths) == set(pm.rows)
    assert info["origin_widths"] == pm.widths0 and info["policy"] == "plan_agent/policy_it00300.pt"
    prepared = plan_agent.plan_state(state, plan_agent.spans_of(LAYOUT), 0.6, ("sens",))
    mask, k = plan_agent.token_index(rows, pm.rows, "cpu")
    with torch.no_grad():
        mu = policy.eval()(prepared, mask, k, len(pm.rows)).tolist()
    assert widths == plan_agent.decode(mu, pm, 0.58)[0]
