"""CPU tests for the V9 fine-menu cell (28 Sep): width ladder, action dedupe, ``mildest``,
stream protection, eval rollback, size-matched TRAJ label, group-first FT flag, the
``relation_bias`` load backfill, and the DepGraph VGG-19 factory. Everything defaults off.

    python -m pytest tests/test_v9_fine_menu.py -v
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.pruning as pruning  # noqa: E402
from src.NetworkEnv import NetworkEnv, prune_current_model  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20, resnet56  # noqa: E402

V9_KEYS = ("SPECTRA_WIDTH_LADDER", "SPECTRA_ACTION_DEDUPE", "SPECTRA_PROTECT_STREAMS",
           "SPECTRA_EVAL_ROLLBACK", "SPECTRA_EVAL_SIZE_MATCH", "SPECTRA_FT_GROUP_FIRST_EPOCHS",
           "SPECTRA_FT_GROUP_FIRST_PATIENCE", "SPECTRA_ACTION_MENU", "SPECTRA_MIN_WIDTH_FOR_PRUNE",
           "SPECTRA_FORTIFY", "SPECTRA_STEM_ROWS")
MENU = {0: 1.0, 1: 0.9, 2: 0.8}
MENU95 = {0: 1.0, 1: 0.95, 2: 0.9, 3: 0.8}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in V9_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


def _mask(rates, width, **kw):
    return fortify.legal_action_mask(rates, row_index=5, alive_count=width, device="cpu", **kw)


def _realised(rates, width, policy):
    """Width left after the heuristic's pick on a ``width``-wide group, through the live mapping."""
    legal = _mask(rates, width)
    idx = int(fortify.heuristic_eval_action(legal, rates, policy=policy, device="cpu").item())
    keep, _stop, _feasible = fortify.effective_rates(rates, 0.0, group_width=width)[idx]
    return width if keep >= 1.0 else pruning.target_width(width, keep)


# ---------------------------------------------------------------- defaults and contract

def test_flags_default_off_and_rates_menu_unchanged():
    assert fortify.width_ladder_max() == 0
    assert not fortify.action_dedupe()
    assert not fortify.protect_streams()
    assert not fortify.eval_rollback()
    assert fortify.eval_size_match() is None
    assert fortify.ft_group_first_epochs() == 0
    for w in range(1, 65):
        assert fortify.effective_rates(MENU95, 0.0, group_width=w) == {
            i: (r, False, True) for i, r in MENU95.items()}


def test_v9_keys_in_the_policy_contract():
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    for key in ("SPECTRA_WIDTH_LADDER", "SPECTRA_ACTION_DEDUPE", "SPECTRA_PROTECT_STREAMS",
                "SPECTRA_FT_GROUP_FIRST_EPOCHS"):
        assert key in A2CAgentReinforce.POLICY_CONTRACT_KEYS
    assert set(A2CAgentReinforce.ACTION_GEOMETRY_KEYS) <= set(A2CAgentReinforce.POLICY_CONTRACT_KEYS)


# ---------------------------------------------------------------- rate geometry (A5 duplicates)

def test_09_and_08_are_one_action_on_widths_3_to_7():
    for w in range(3, 8):
        assert pruning.target_width(w, 0.9) == pruning.target_width(w, 0.8) == w - 1
    assert pruning.target_width(8, 0.9) == 7 and pruning.target_width(8, 0.8) == 6


def test_095_duplicates_09_below_16_and_is_finer_above():
    for w in range(2, 16):
        assert pruning.target_width(w, 0.95) == pruning.target_width(w, 0.9)
    assert (pruning.target_width(16, 0.95), pruning.target_width(16, 0.9)) == (15, 14)
    assert (pruning.target_width(32, 0.95), pruning.target_width(32, 0.9)) == (30, 29)
    assert (pruning.target_width(64, 0.95), pruning.target_width(64, 0.9)) == (61, 58)


# ---------------------------------------------------------------- width ladder

def test_ladder_turns_rates_into_channel_counts(monkeypatch):
    monkeypatch.setenv("SPECTRA_WIDTH_LADDER", "8")
    rates = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7}
    m4 = fortify.effective_rates(rates, 0.0, group_width=4)
    assert [pruning.target_width(4, m4[i][0]) for i in (1, 2, 3)] == [3, 2, 1]
    assert all(m4[i][2] for i in rates)
    m3 = fortify.effective_rates(rates, 0.0, group_width=3)
    assert not m3[3][2]                                  # remove 3 of 3 is infeasible
    assert [pruning.target_width(3, m3[i][0]) for i in (1, 2)] == [2, 1]
    m2 = fortify.effective_rates(rates, 0.0, group_width=2)
    assert m2[1][2] and not m2[2][2] and not m2[3][2]
    m16 = fortify.effective_rates(rates, 0.0, group_width=16)  # above the ladder: fractions
    assert [m16[i][0] for i in rates] == [1.0, 0.9, 0.8, 0.7]


def test_ladder_mask_keeps_narrow_rates_distinct(monkeypatch):
    assert _mask(MENU, 4).tolist() == [True, True, True]      # two legal entries, one action
    monkeypatch.setenv("SPECTRA_ACTION_DEDUPE", "1")
    assert _mask(MENU, 4).tolist() == [True, False, True]     # the duplicate 0.9 is masked
    monkeypatch.delenv("SPECTRA_ACTION_DEDUPE")
    monkeypatch.setenv("SPECTRA_WIDTH_LADDER", "8")
    assert _mask(MENU, 4).tolist() == [True, True, True]      # now two different cuts
    assert _mask(MENU, 2).tolist() == [True, False, False]    # w=2 is under the fortify min width
    # locked rows stay identity-only; an infeasible ladder entry never looks like identity
    assert _mask({0: 1.0, 1: 0.9, 2: 0.7}, 3, force_identity=True).tolist() == [True, False, False]


@pytest.mark.parametrize("ladder", ["4", "8", "15"])
def test_mild_walk_unchanged_by_ladder_up_to_15(monkeypatch, ladder):
    base = {w: _realised(MENU, w, "mild") for w in range(2, 65)}
    monkeypatch.setenv("SPECTRA_WIDTH_LADDER", ladder)
    assert {w: _realised(MENU, w, "mild") for w in range(2, 65)} == base


def test_ladder_16_would_change_the_mild_walk(monkeypatch):
    base = _realised(MENU, 16, "mild")
    monkeypatch.setenv("SPECTRA_WIDTH_LADDER", "16")
    assert (base, _realised(MENU, 16, "mild")) == (14, 15)


def test_ladder_changes_the_greedy_walk_on_narrow_groups(monkeypatch):
    assert _realised(MENU, 8, "l1") == 6
    monkeypatch.setenv("SPECTRA_WIDTH_LADDER", "8")
    assert _realised(MENU, 8, "l1") == 6                       # 0.8 -> remove 2 on w=8 either way
    assert _realised(MENU, 12, "l1") == 10                     # above the ladder: unchanged
    assert _realised({0: 1.0, 1: 0.9, 2: 0.7}, 6, "l1") == 3   # 0.7 -> remove 3 (was 4)


# ---------------------------------------------------------------- 0.95 + dedupe + mildest

def test_mildest_with_dedupe_equals_mild_below_16_and_is_finer_above(monkeypatch):
    monkeypatch.setenv("SPECTRA_ACTION_DEDUPE", "1")
    for w in range(3, 16):
        assert _realised(MENU95, w, "mildest") == _realised(MENU, w, "mild") == w - 1
    assert _realised(MENU95, 16, "mildest") == 15
    assert _realised(MENU95, 64, "mildest") == 61
    assert _realised(MENU95, 64, "mild") == 58                 # mild still takes its 0.9


def test_dedupe_masks_095_where_it_duplicates_09(monkeypatch):
    monkeypatch.setenv("SPECTRA_ACTION_DEDUPE", "1")
    assert _mask(MENU95, 12).tolist() == [True, False, True, True]
    assert _mask(MENU95, 4).tolist() == [True, False, False, True]   # all three cuts leave 3
    assert _mask(MENU95, 32).tolist() == [True, True, True, True]


def test_mildest_picks_the_weakest_legal_cut():
    legal = torch.tensor([True, True, True, True])
    assert int(fortify.heuristic_eval_action(legal, MENU95, policy="mildest", device="cpu")) == 1
    legal = torch.tensor([True, False, True, True])
    assert int(fortify.heuristic_eval_action(legal, MENU95, policy="mildest", device="cpu")) == 2
    legal = torch.tensor([True, False, False, False])
    assert int(fortify.heuristic_eval_action(legal, MENU95, policy="mildest", device="cpu")) == 0


# ---------------------------------------------------------------- env wiring on a real net

def test_protect_streams_marks_residual_rows_only():
    model = resnet20(num_classes=10, large_input=False, width=4)
    mwr = ModelWithRows(model)
    assert NetworkEnv._is_stream_row(mwr, model.embed[0])
    assert NetworkEnv._is_stream_row(mwr, model.layer1[0].conv2)
    assert NetworkEnv._is_stream_row(mwr, model.layer2[0].downsample[0])
    assert not NetworkEnv._is_stream_row(mwr, model.layer1[0].conv1)
    assert not NetworkEnv._is_stream_row(mwr, model.layer3[2].conv1)


def test_ladder_keep_rate_drives_the_structural_cut(monkeypatch):
    model = resnet20(num_classes=10, large_input=False, width=4)
    mwr = ModelWithRows(model)
    target = model.layer1[0].conv1
    row = next(r for r, li in mwr.row_to_main_layer.items() if mwr.all_layers[li] is target)
    assert NetworkEnv._ladder_keep_rate(mwr, row, 0.8) == 0.8          # ladder off: untouched
    monkeypatch.setenv("SPECTRA_WIDTH_LADDER", "8")
    keep = NetworkEnv._ladder_keep_rate(mwr, row, 0.8)
    assert keep == pytest.approx(0.5)
    mwr = prune_current_model(mwr, keep, row, quiet=True, record=False, input_shape=(3, 32, 32))
    assert pruning.layer_width(mwr.model.layer1[0].conv1) == 2
    assert NetworkEnv._ladder_keep_rate(ModelWithRows(mwr.model), row, 0.7) == 1.0  # 3 of 2: identity


def test_action_costs_price_the_ladder(monkeypatch):
    from src.action_costs import estimate_action_costs
    model = resnet20(num_classes=10, large_input=False, width=4)
    target = model.layer1[0].conv1
    plain = estimate_action_costs(model, target, [1.0, 0.9, 0.8], (3, 32, 32), device="cpu")
    assert torch.isclose(plain[1, 1], plain[2, 1])                     # one action, two slots
    monkeypatch.setenv("SPECTRA_WIDTH_LADDER", "8")
    laddered = estimate_action_costs(model, target, [1.0, 0.9, 0.8], (3, 32, 32), device="cpu")
    assert torch.isclose(laddered[2, 1], 2 * laddered[1, 1])           # 2 channels vs 1


def _bare_env(model):
    env = NetworkEnv.__new__(NetworkEnv)
    env.current_model = model
    env.original_acc = 0.9
    env.last_val_acc = 0.85
    env._ratio_cache = {"k": 1}
    env._pass_locked_layers = set()
    env._rollback_locked_layers = set()
    env.last_step_outcome = {"mode": "identity"}
    env.last_step_layer_idx = None
    return env


def test_rollback_restores_the_model_and_locks_the_group():
    env = _bare_env(nn.Linear(4, 4))
    snap = env.rollback_snapshot()
    before = env.current_model.weight.detach().clone()
    with torch.no_grad():
        env.current_model.weight.zero_()
    env.last_val_acc = 0.5
    env.last_step_outcome = {"mode": "structural", "group_layer_indices": [3, 5, 7]}
    env.last_step_layer_idx = 3
    assert env.rollback_to(snap) == [3, 5, 7]
    assert torch.equal(env.current_model.weight, before)
    assert env.last_val_acc == 0.85 and env._ratio_cache == {}
    assert env.group_locked(5) and not env.group_locked(4)             # locked without group-once
    env.last_step_outcome = {"mode": "masked"}
    env.last_step_layer_idx = 9
    assert env.rollback_to(env.rollback_snapshot()) == [9]
    assert env.group_locked(9)


# ---------------------------------------------------------------- size-matched TRAJ label

def _pt(param, flop, val):
    return {"step": 0, "param": param, "flop": flop, "val_dacc_pp": val, "test_dacc_pp": val}


def test_size_match_parse(monkeypatch):
    for raw, want in (("flop:0.39", ("flop", 0.39)), ("FLOPS:0.39", ("flop", 0.39)),
                      ("param:0.42", ("param", 0.42)), ("params:0.42", ("param", 0.42)),
                      ("bogus:0.4", None), ("flop:1.5", None), ("flop", None), ("", None)):
        monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", raw)
        assert fortify.eval_size_match() == want


def test_size_match_label_is_the_first_point_at_or_below_target():
    pts = [_pt(1.0, 1.0, 0.0), _pt(0.8, 0.7, -2.0), _pt(0.6, 0.45, -8.0),
           _pt(0.5, 0.38, -12.0), _pt(0.4, 0.30, -20.0)]
    picked = fortify.select_trajectory_points(pts, min_param=0.7, tau_pp=10.0,
                                              size_match=("flop", 0.39))
    assert picked["size_match"]["flop"] == 0.38                      # reported even out of band
    assert picked["val_best"]["param"] == 0.6
    assert fortify.select_trajectory_points(pts, min_param=0.7, tau_pp=10.0,
                                            size_match=("param", 0.1))["size_match"] is None
    assert "size_match" not in fortify.select_trajectory_points(pts, min_param=0.7, tau_pp=10.0)


def test_group_first_flags(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_GROUP_FIRST_EPOCHS", "3")
    assert fortify.ft_group_first_epochs() == 3 and fortify.ft_group_first_patience() == 2
    monkeypatch.setenv("SPECTRA_FT_GROUP_FIRST_PATIENCE", "5")
    assert fortify.ft_group_first_patience() == 5


# ---------------------------------------------------------------- checkpoint compatibility

class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3)
        self.relation_bias = nn.Parameter(torch.zeros(3))


def test_pre_v8_actor_loads_with_relation_bias_zero_filled(tmp_path):
    from src.A2C_Agent_Reinforce import load_agent_checkpoint
    old = _TinyEncoder()
    state = {k: v for k, v in old.state_dict().items() if not k.endswith("relation_bias")}
    path = tmp_path / "actor.pt"
    torch.save({"state_dict": state}, path)
    new = _TinyEncoder()
    with torch.no_grad():
        new.relation_bias.fill_(3.0)
    load_agent_checkpoint(new, str(path), "cpu")
    assert torch.equal(new.relation_bias, torch.zeros(3))
    assert torch.equal(new.proj.weight, old.proj.weight)
    torch.save({"state_dict": {"relation_bias": torch.zeros(3)}}, path)
    with pytest.raises(RuntimeError):                                  # other keys stay strict
        load_agent_checkpoint(_TinyEncoder(), str(path), "cpu")


def test_old_actor_replay_turns_action_geometry_off(tmp_path, monkeypatch):
    import a2c_agent_reinforce_runner as runner
    ckpt = tmp_path / "agent_checkpoints"
    ckpt.mkdir()
    (ckpt / "latest_best_actor.pt").write_bytes(b"x")
    (ckpt / "policy_config.json").write_text(json.dumps({
        "compression_rates": [1.0, 0.9, 0.8], "action_rankings": [None, None, None],
        "env": {"SPECTRA_STATE_ALIGN": "next"}, "passes": 2}), encoding="utf-8")
    monkeypatch.delenv("SPECTRA_POLICY_CONFIG", raising=False)
    monkeypatch.setenv("SPECTRA_EVAL_PASSES", "2")
    monkeypatch.setenv("SPECTRA_STATE_ALIGN", "next")
    monkeypatch.setenv("SPECTRA_WIDTH_LADDER", "8")
    monkeypatch.setenv("SPECTRA_PROTECT_STREAMS", "1")
    monkeypatch.setenv("SPECTRA_FT_GROUP_FIRST_EPOCHS", "2")           # recovery: submitter's choice
    args = SimpleNamespace(actor_checkpoint_path=str(ckpt / "latest_best_actor.pt"),
                           compression_rates=[1.0, 0.9, 0.8], action_rankings=None, passes=2)
    runner.apply_policy_config(args)
    assert "SPECTRA_WIDTH_LADDER" not in os.environ
    assert "SPECTRA_PROTECT_STREAMS" not in os.environ
    assert os.environ["SPECTRA_FT_GROUP_FIRST_EPOCHS"] == "2"


# ---------------------------------------------------------------- dry walks (kill-table predictions)

def _dry(net, rule, passes):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import dry_walk_geometry as dw
    return [round(s["param"], 6) for s in dw.dry_walk(net, rule, passes, flops=False)]


def test_095_and_ladder_walks_equal_mild_on_r20_w2():
    net = lambda: resnet20(num_classes=10, large_input=False, width=2)  # noqa: E731
    mild = _dry(net(), "mild", 2)
    assert _dry(net(), "mildest95", 2) == mild
    assert _dry(net(), "mild_ladder8", 2) == mild


def test_ladder_walk_equals_mild_on_r56_w4_first_pass():
    net = lambda: resnet56(num_classes=10, large_input=False, width=4)  # noqa: E731
    assert _dry(net(), "mild_ladder8", 1) == _dry(net(), "mild", 1)


# ---------------------------------------------------------------- DepGraph VGG-19 factory

def test_depgraph_vgg19_layout_forward_and_groups():
    from spectra_models_instantiation.vgg_depgraph import vgg19_bn
    model = vgg19_bn(num_classes=100, large_input=False).eval()
    keys = model.state_dict().keys()
    assert {"block0.0.weight", "block0.0.bias", "block0.1.running_mean", "block4.10.weight",
            "classifier.weight", "classifier.bias"} <= set(keys)
    assert len(keys) == 16 * 7 + 2
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 16
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 100)
    groups = channel_groups.build_channel_groups(model) or []
    assert sum(1 for g in groups if g.prunable) == 16
    assert len(ModelWithRows(model).row_to_main_layer) == 17
