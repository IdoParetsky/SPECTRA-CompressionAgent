"""
v18 ``SPECTRA_PLAN_STRUCTURAL_ONLY``: the plans leave out every group whose cut the walk does not realize as a
structural edit (``group_sensitivity.realizes_structurally``), so those groups stay whole and every plan's predicted
size is the size its cut reaches. On ShuffleNetV2 the stage streams pass through chunk / cat / channel shuffle: their
cuts are masked (or, for the last stream of stage 4, run but misroute channels behind the shuffle), which shrinks
nothing, so ``ParamModel`` / ``FlopModel`` counted cuts that never happen. With the flag, random-init ShuffleNetV2 x1.0
plans 18 of its 37 groups (the stem, the 16 branch-internal groups and the head), and every plan, under every
allocation kind and both budgets, lands on its predicted size with no masked edit. Default off: every plan is as in
tree_v17 and no check runs.

CPU only, no datasets.  python -m pytest tests/test_v18_structural_only.py -v
"""

import copy
import os
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import MiniShuffle, _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import RATES, SHAPE, V10_FLAGS, ZOO, _randomize_norms, toy_state  # noqa: E402
from tests.test_v16_sens_cost import _batches  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from spectra_models_instantiation.shufflenetv2_chenyaofo import shufflenetv2x1, shufflenetv2x15  # noqa: E402
from src import alloc_walk, fortify, plan_agent, state_dump  # noqa: E402
import src.NetworkEnv as network_env  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG = "SPECTRA_PLAN_STRUCTURAL_ONLY"
FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_MIN_WIDTH")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


@pytest.fixture
def v10_env(monkeypatch):
    for key, value in V10_FLAGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SPECTRA_STATE_TOKENS", raising=False)


def _model(name):
    torch.manual_seed(0)
    model = (shufflenetv2x1(10, False) if name == "shufflenetv2x1" else ZOO[name]()).eval()
    _randomize_norms(model)
    return model


@pytest.fixture(scope="module")
def shuffle():
    """Random-init ShuffleNetV2 x1.0 (CIFAR-10): its groups, the plans with the flag off and on, and the cost models of
    the structural-only plan."""
    model = _model("shufflenetv2x1")
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    off = group_sensitivity.group_plan(mwr, groups, structural_only=False)
    on = group_sensitivity.group_plan(mwr, groups, structural_only=True)
    return types.SimpleNamespace(model=model, mwr=mwr, groups=groups, off=off, on=on,
                                 pm=plan_agent.ParamModel(model, on), fm=plan_agent.FlopModel(model, on, SHAPE),
                                 names={id(m): n for n, m in model.named_modules()})


def _tree_v17_plan(mwr, groups):
    """``group_plan`` as tree_v17 wrote it."""
    plan, seen = [], set()
    for row in sorted(mwr.row_to_main_layer)[:-1]:
        group = channel_groups.group_of(groups, mwr.all_layers[mwr.row_to_main_layer[row]])
        if group is None or not group.prunable or group.width < 2 or id(group) in seen:
            continue
        seen.add(id(group))
        plan.append((group, row))
    return plan


def _refuse(*_args, **_kwargs):
    raise AssertionError("the structural check ran")


def _no_mask(*_args, **_kwargs):
    raise AssertionError("a prune fell back to masking")


def _cut(monkeypatch, model, plan, rates):
    """``alloc_walk.cut_to`` with every prune's outcome recorded and the masked fallback instrumented to fail."""
    modes, real_prune = [], network_env.prune_current_model

    def recording(model_with_rows, *args, **kwargs):
        out = real_prune(model_with_rows, *args, **kwargs)
        modes.append(out.last_prune_outcome["mode"])
        return out

    with monkeypatch.context() as patch:
        patch.setattr(network_env, "prune_current_model", recording)
        patch.setattr(pruning, "mask_layer_filters", _no_mask)
        real = alloc_walk.cut_to(model, plan, rates, SHAPE).eval()
    return modes, real


def _kept(model, budget, total0):
    return (utils.calc_num_parameters(model) if budget == "params" else utils.calc_flops(model, SHAPE)) / total0


# ------------------------------------------------------------------ the flag


def test_flag_is_off_by_default_and_parses_like_the_other_flags(monkeypatch):
    assert fortify.plan_structural_only() is False
    for raw, want in (("1", True), ("true", True), (" Yes ", True), ("on", True), ("0", False), ("", False),
                      ("off", False)):
        monkeypatch.setenv(FLAG, raw)
        assert fortify.plan_structural_only() is want


@pytest.mark.parametrize("name", ["thin_r20_w4", "shufflenetv2x1"])
def test_flag_off_plans_are_tree_v17s_and_run_no_check(monkeypatch, name):
    model = _model(name)
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(model)
    ref = _tree_v17_plan(mwr, groups)
    rows = [row for _g, row in ref]
    if name == "shufflenetv2x1":
        assert len(ref) == 37                                         # the stage streams are in the default plan
    with monkeypatch.context() as patch:
        patch.setattr(group_sensitivity, "structural_plan", _refuse)
        patch.setattr(group_sensitivity, "_realizes", _refuse)
        for raw in (None, "0", "off", ""):
            if raw is None:
                patch.delenv(FLAG, raising=False)
            else:
                patch.setenv(FLAG, raw)
            plan = group_sensitivity.group_plan(mwr, groups)
            assert [(id(g), r) for g, r in plan] == [(id(g), r) for g, r in ref]
            assert [r for _g, r in group_sensitivity.group_plan(ModelWithRows(model))] == rows
            assert plan_agent.ParamModel(model, plan).rows == rows and plan_agent.MaskedCut(model).rows == rows
    if name == "thin_r20_w4":
        # every cut of a ResNet is structural: the flag on plans the same groups, and the rule plans are unchanged
        off_widths, off_info = alloc_walk.plan_targets(model, [], SHAPE, "uniform", 0.5)
        monkeypatch.setenv(FLAG, "1")
        assert [(id(g), r) for g, r in group_sensitivity.group_plan(mwr, groups)] == [(id(g), r) for g, r in ref]
        on_widths, on_info = alloc_walk.plan_targets(model, [], SHAPE, "uniform", 0.5)
        assert on_widths == off_widths and on_info["kept"] == off_info["kept"] and on_info["keeps"] == off_info["keeps"]


def test_the_sensitivity_state_channels_ignore_the_flag(monkeypatch):
    model = _model("thin_r20_w4")
    batches = _batches()
    off, off_summary = group_sensitivity.layer_features(model, batches, SHAPE)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(group_sensitivity, "structural_plan", _refuse)
    on, on_summary = group_sensitivity.layer_features(model, batches, SHAPE)
    assert torch.equal(on, off) and on_summary["groups"] == off_summary["groups"]


# ------------------------------------------------------------------ which groups ShuffleNetV2 plans


def test_shufflenet_plans_the_stem_the_branch_groups_and_the_head_and_leaves_every_stream_whole(shuffle):
    names = shuffle.names
    planned = [names[id(g.producers[0])] for g, _r in shuffle.on]
    assert len(shuffle.off) == 37 and len(planned) == 18
    assert planned[0] == "conv1.0" and planned[-1] == "conv5.0"
    assert all(n.endswith("branch2.0") for n in planned[1:-1])      # pw1 + depthwise inside each block's branch 2
    kept = {row for _g, row in shuffle.on}
    left = [(g, row) for g, row in shuffle.off if row not in kept]
    assert len(left) == 19 and all(names[id(g.producers[0])].endswith(("branch1.2", "branch2.5")) for g, _r in left)
    # 16 streams are read through chunk halves by several later blocks; three are read whole
    whole = [names[id(g.producers[0])] for g, _r in left if group_sensitivity.reads_whole(g)]
    assert whole == ["stage2.3.branch2.5", "stage3.7.branch2.5", "stage4.3.branch2.5"]
    assert all(group_sensitivity.reads_whole(g) for g, _r in shuffle.on)


def test_why_the_three_streams_read_whole_are_left_whole(shuffle):
    """Stages 2 and 3's last streams also feed the next stage's branch-1 depthwise conv, which reads several groups and
    is no recorded consumer, so their cut breaks the forward and is masked. Stage 4's last stream feeds conv5 through a
    channel shuffle: an odd width breaks the shuffle (masked), an even one runs, but the shuffle re-interleaves the
    stream with the shortened branch while conv5's input is sliced on the old order, so the net computes something
    else than the masked cut."""
    by_name = {shuffle.names[id(g.producers[0])]: (g, row) for g, row in shuffle.off}

    def walk_cut(row, rate):
        return network_env.prune_current_model(ModelWithRows(copy.deepcopy(shuffle.model)), rate, row, quiet=True,
                                               record=False, input_shape=SHAPE, importance="l1")

    for name in ("stage2.3.branch2.5", "stage3.7.branch2.5"):
        outcome = walk_cut(by_name[name][1], 0.5).last_prune_outcome
        assert outcome["mode"] == "masked" and "dummy forward" in outcome["reason"]
    group, row = by_name["stage4.3.branch2.5"]
    assert walk_cut(row, 117 / 232.0).last_prune_outcome["mode"] == "masked"
    cut = walk_cut(row, 116 / 232.0)
    assert cut.last_prune_outcome["mode"] == "structural"
    masked, twin = copy.deepcopy((shuffle.model, group))
    plan_agent._zero_group(twin, torch.tensor(cut.last_group_edit["keep_idx"], dtype=torch.long))
    x = torch.randn(2, *SHAPE, generator=torch.Generator().manual_seed(1)).double()
    with torch.no_grad():
        want = masked.double().eval()(x)
        gap = float((cut.model.double().eval()(x) - want).abs().max())
    assert gap > 1e-6 * (1.0 + float(want.abs().max()))               # float64 noise is ~1e-15
    assert not group_sensitivity.realizes_structurally(shuffle.model, group, row)


def test_the_check_is_cached_per_architecture_and_leaves_model_rng_and_bn_scores_alone(monkeypatch):
    monkeypatch.setattr(group_sensitivity, "_REALIZED", {})
    model = MiniShuffle(8).eval()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    pruning.bind_bn_scales(model)
    scores = dict(pruning._BN_ABS_GAMMA)
    rng = torch.random.get_rng_state()
    calls, real = [], group_sensitivity._realizes
    monkeypatch.setattr(group_sensitivity, "_realizes", lambda *args: calls.append(args[2]) or real(*args))
    names = {id(m): n for n, m in model.named_modules()}
    off = group_sensitivity.group_plan(ModelWithRows(model), structural_only=False)
    on = group_sensitivity.group_plan(ModelWithRows(model), structural_only=True)
    # the stem is read through both chunk halves; the pw branch's cut runs but the shuffle re-interleaves it
    assert [names[id(g.producers[0])] for g, _r in off] == ["stem", "pw", "head"]
    assert [names[id(g.producers[0])] for g, _r in on] == ["head"] and len(calls) == 3
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert set(pruning._BN_ABS_GAMMA) == set(scores) and all(pruning._BN_ABS_GAMMA[k] is v for k, v in scores.items())
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items()) and not model.training
    torch.manual_seed(3)
    other = MiniShuffle(8).eval()                                     # another checkpoint of the architecture
    assert [r for _g, r in group_sensitivity.group_plan(ModelWithRows(other), structural_only=True)] == \
        [r for _g, r in on] and len(calls) == 3


def test_shufflenet_x15_shares_the_x1_answers(shuffle):
    model = shufflenetv2x15(10, False).eval()
    mwr = ModelWithRows(model)
    on = group_sensitivity.group_plan(mwr, structural_only=True)
    assert [r for _g, r in on] == [r for _g, r in shuffle.on]
    assert [g.width for g, _r in on][1:5] == [88] * 4                 # stage 2's branch groups at x1.5


# ------------------------------------------------------------------ plans land on their predicted size


def test_the_narrowest_plan_is_exact_and_puts_both_06_targets_in_reach(shuffle, monkeypatch):
    floor = plan_agent.widths_of({row: 0.1 for row in shuffle.pm.rows}, shuffle.pm.widths0, 2)
    assert floor == {row: max(min(w0, 2), int(round(0.1 * w0))) for row, w0 in shuffle.pm.widths0.items()}
    modes, real = _cut(monkeypatch, shuffle.model, shuffle.on, plan_agent.rates_of(floor, shuffle.pm.widths0))
    assert len(modes) == len(shuffle.on) and set(modes) == {"structural"}
    assert utils.calc_num_parameters(real) == shuffle.pm.params(floor)
    assert _kept(real, "flops", shuffle.fm.total0) == pytest.approx(shuffle.fm.kept(floor), rel=1e-9)
    assert 0.1 < shuffle.pm.kept(floor) < 0.2 and 0.1 < shuffle.fm.kept(floor) < 0.2     # 0.154 params, 0.141 FLOPs
    assert alloc_walk.group_widths(real, shuffle.pm.rows) == floor


@pytest.mark.parametrize("budget", ["params", "flops"])
def test_random_plans_land_where_the_cost_model_says_with_no_masked_edit(shuffle, monkeypatch, budget):
    cm = shuffle.pm if budget == "params" else shuffle.fm
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(2, *SHAPE, generator=gen)
    for kappa in (0.4, 0.6, 0.75, 0.9):
        z = (1.5 * torch.randn(len(cm.rows), generator=gen)).tolist()
        widths, info = plan_agent.decode(z, cm, kappa, 0.1, min_width=2)
        assert abs(info["kept"] - kappa) < 0.01
        modes, real = _cut(monkeypatch, shuffle.model, shuffle.on, plan_agent.rates_of(widths, cm.widths0))
        assert modes and set(modes) == {"structural"}
        kept = _kept(real, budget, cm.total0)
        assert abs(kept - info["kept"]) < 0.01
        if budget == "params":
            assert utils.calc_num_parameters(real) == shuffle.pm.params(widths)
        else:
            assert kept == pytest.approx(info["kept"], rel=1e-9)
        assert alloc_walk.group_widths(real, cm.rows) == widths
        with torch.no_grad():
            assert real(x).shape == (2, 10)


def test_masked_cut_computes_the_real_cut_under_the_flag(shuffle, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    cutter = plan_agent.MaskedCut(shuffle.model)
    assert cutter.rows == shuffle.pm.rows
    widths, _info = plan_agent.decode([0.0] * len(shuffle.pm.rows), shuffle.pm, 0.6, 0.1, min_width=2)
    rates = plan_agent.rates_of(widths, shuffle.pm.widths0)
    _modes, real = _cut(monkeypatch, shuffle.model, shuffle.on, rates)
    x = torch.randn(4, *SHAPE, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(cutter.cut(rates).eval()(x), real(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("budget", ["params", "flops"])
@pytest.mark.parametrize("kind_name", ["uniform", "sens", "sens_cost", "inner"])
def test_every_rule_plan_lands_within_001_of_its_target_on_structural_cuts(shuffle, monkeypatch, kind_name, budget):
    monkeypatch.setenv(FLAG, "1")
    batches = _batches() if kind_name in ("sens", "sens_cost") else []
    # the check's own probe cuts may mask (that is what it detects): answer it before masking is made to fail
    assert len(group_sensitivity.group_plan(ModelWithRows(shuffle.model))) == len(shuffle.on)
    with monkeypatch.context() as patch:
        patch.setattr(pruning, "mask_layer_filters", _no_mask)          # the bisection's cuts and the sens probes too
        widths, info = alloc_walk.plan_targets(shuffle.model, batches, SHAPE, kind_name, 0.6, budget=budget,
                                               min_width=2)
    rows = [row for _g, row in shuffle.on]
    assert set(widths) == set(rows) == set(info["keeps"]) and info["held"] == 0   # no group has two producers here
    assert abs(info["kept"] - 0.6) < 0.01
    assert all(widths[row] >= min(2, w0) for row, w0 in info["origin_widths"].items())


@pytest.mark.parametrize("budget", ["params", "flops"])
def test_agent_plan_lands_within_001_and_is_cut_structurally(shuffle, monkeypatch, tmp_path, v10_env, budget):
    monkeypatch.setenv(FLAG, "1")
    rows = state_dump.token_rows(shuffle.mwr, shuffle.groups, shuffle.on)
    assert sorted({r for r in rows if r >= 0}) == shuffle.pm.rows      # the streams' tokens map to no planned group
    state = toy_state(len(rows))
    monkeypatch.setattr(state_dump, "encode_origin", lambda env, m, g: dict(state))
    torch.manual_seed(5)
    policy = plan_agent.PlanPolicy(63)
    with torch.no_grad():
        policy.head.weight.normal_(std=2.0)
    path = tmp_path / "plan_agent" / "policy_latest.pt"
    plan_agent.save_policy(policy, str(path), tokens="layer", zero=["sens"], k_min=0.1, min_width=2, budget=budget)
    env = types.SimpleNamespace(conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES),
                                current_model=shuffle.model, target_keep=0.6, _input_shape=lambda: SHAPE,
                                _dependency_groups=lambda m: channel_groups.build_channel_groups(m.model))
    kw = {"budget": "flops", "kappa": 0.6} if budget == "flops" else {}
    widths, info = plan_agent.plan_for_env(env, 0.58, str(path), **kw)
    cm = shuffle.pm if budget == "params" else shuffle.fm
    assert set(widths) == set(cm.rows) and info["min_width"] == 2 and abs(info["kept"] - 0.58) < 0.01
    assert len(set(info["keeps"].values())) > 1                       # a non-uniform plan
    modes, real = _cut(monkeypatch, shuffle.model, shuffle.on, plan_agent.rates_of(widths, cm.widths0))
    assert set(modes) == {"structural"}
    assert _kept(real, budget, cm.total0) == pytest.approx(info["kept"], rel=1e-9)
