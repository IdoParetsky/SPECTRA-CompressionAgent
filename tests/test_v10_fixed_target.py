"""
v10 (4 Oct, Ido GO 19:23 "fixed_target"): fixed-target episodes and the group-sensitivity state.

Each episode draws a target keep; a cut that would pass it is narrowed to land on it and the
episode ends there; every step pays its change in val accuracy, so the return is the val Δacc
at the target (minus a penalty for ending above it). Two state channels carry the target, two
carry the sensitivity of the group each layer produces (A0's measure).

CPU only, no datasets.  python -m pytest tests/test_v10_fixed_target.py -v
"""

import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf, _row_index_of_layer  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import fortify  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402
from src.BERTInputModeler import (BERTInputModeler, TOKEN_BASE_DIM, token_feature_dim,  # noqa: E402
                                  action_cost_slot_dim)
from src.NetworkEnv import NetworkEnv, AGENT_TRAIN, EVAL_TEST  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402

V10_ENV = ("SPECTRA_FIXED_TARGET", "SPECTRA_STATE_SENS", "SPECTRA_TARGET_KEEP_RANGE",
           "SPECTRA_TARGET_MISS_PENALTY", "SPECTRA_TARGET_PROBE_KEEPS", "SPECTRA_PROBE_SCORE",
           "SPECTRA_EVAL_SIZE_MATCH", "SPECTRA_EVAL_SIZE_POINTS", "SPECTRA_WIDTH_LADDER",
           "SPECTRA_ACTION_MENU", "SPECTRA_STATE_GROUPCOST", "SPECTRA_STATE_SLACK",
           "SPECTRA_BUDGET_IN_STATE", "SPECTRA_GROUP_ONCE_PER_PASS", "SPECTRA_STATE_ALIGN")


@pytest.fixture(autouse=True)
def _clean_v10_env(monkeypatch):
    for key in V10_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


def _tiny_loader(n=16, classes=10, seed=0, batch=8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch)


def _layer_index(mwr, module):
    return next(i for i, layer in enumerate(mwr.all_layers) if layer is module)


# ------------------------------------------------------------------ flags

def test_flags_default_off_and_parse(monkeypatch):
    assert not fortify.fixed_target() and not fortify.state_sens()
    assert fortify.target_keep_range() == (0.35, 0.85)
    assert fortify.target_miss_penalty() == 2.0
    assert fortify.target_probe_keeps() == (0.8, 0.6, 0.4)
    assert fortify.probe_score_kind() == "cut"
    monkeypatch.setenv("SPECTRA_TARGET_KEEP_RANGE", "0.9:0.5")
    assert fortify.target_keep_range() == (0.5, 0.9)
    monkeypatch.setenv("SPECTRA_TARGET_PROBE_KEEPS", "0.7,0.5")
    assert fortify.target_probe_keeps() == (0.7, 0.5)
    monkeypatch.setenv("SPECTRA_TARGET_MISS_PENALTY", "0")
    assert fortify.target_miss_penalty() == 0.0
    monkeypatch.setenv("SPECTRA_PROBE_SCORE", "target")
    assert fortify.probe_score_kind() == "target"


def test_target_channels_and_score():
    assert fortify.target_channels(1.0, 0.6) == pytest.approx([0.6, 1.0])
    assert fortify.target_channels(0.8, 0.6) == pytest.approx([0.6, 0.5])
    assert fortify.target_channels(0.6, 0.6) == pytest.approx([0.6, 0.0])
    assert fortify.target_channels(0.3, 0.6) == pytest.approx([0.6, -0.75])
    assert fortify.target_channels(0.0, 0.6)[1] == -1.0
    # Any miss is penalised at 2 pp per point of the network: 0.25 % costs 0.5 pp, 5 % costs 10 pp.
    assert fortify.target_score(-3.0, 0.6025, 0.6) == pytest.approx(-3.5)
    assert fortify.target_score(-3.0, 0.65, 0.6) == pytest.approx(-13.0)
    assert fortify.target_score(-3.0, 0.55, 0.6) == pytest.approx(-3.0)


def test_policy_config_pins_the_v10_keys():
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    assert {"SPECTRA_FIXED_TARGET", "SPECTRA_STATE_SENS"} <= set(A2CAgentReinforce.POLICY_CONTRACT_KEYS)
    assert {"SPECTRA_TARGET_KEEP_RANGE", "SPECTRA_TARGET_MISS_PENALTY",
            "SPECTRA_TARGET_PROBE_KEEPS"} <= set(A2CAgentReinforce.POLICY_INFO_KEYS)


# ------------------------------------------------------------------ tokens

def test_token_width_and_channels(monkeypatch):
    monkeypatch.setenv("SPECTRA_FORTIFY", "1")
    base = token_feature_dim(5)
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    assert token_feature_dim(5) == base + fortify.STATE_TARGET_DIM
    monkeypatch.setenv("SPECTRA_STATE_SENS", "1")
    assert token_feature_dim(5) == base + fortify.STATE_TARGET_DIM + fortify.STATE_SENS_DIM

    builder = BERTInputModeler()
    L = 3
    maps = {"Topology": [[2, 3, 8, 3, 1, 1, 1]] * L, "Activations": [[0.1] * 12] * L,
            "Weights": [[0.2] * 19] * L}
    sens = torch.tensor([[0.5, 1.0], [-0.2, 0.0], [0.0, 0.5]])
    tokens = builder._build_layer_tokens(maps, 0, action_costs=None, coupling_ids=torch.arange(L),
                                         target_extras=[0.6, 0.5], layer_sens=sens)
    slot = action_cost_slot_dim()
    end = tokens.shape[1] - slot
    assert end == TOKEN_BASE_DIM + fortify.fortify_token_dim()
    assert torch.allclose(tokens[:, end - 2:end], sens)
    assert torch.allclose(tokens[:, end - 4:end - 2], torch.tensor([[0.6, 0.5]] * L))
    # Without a caller value: the reset target channels and zero sensitivity, same width.
    tokens0 = builder._build_layer_tokens(maps, 0, action_costs=None, coupling_ids=torch.arange(L))
    assert tokens0.shape == tokens.shape
    assert torch.allclose(tokens0[:, end - 4:end - 2], torch.tensor([[0.0, 1.0]] * L))
    assert torch.count_nonzero(tokens0[:, end - 2:end]) == 0


# ------------------------------------------------------------------ target choice

def test_episode_target_draw_explicit_and_eval(monkeypatch):
    env = NetworkEnv.__new__(NetworkEnv)
    env._target_rng = np.random.default_rng([0, 0, 10])
    env.mode = AGENT_TRAIN
    draws = [env._episode_target() for _ in range(400)]
    assert 0.35 <= min(draws) and max(draws) <= 0.85 and statistics.pstdev(draws) > 0.1
    again = np.random.default_rng([0, 0, 10])
    assert draws[0] == pytest.approx(again.uniform(0.35, 0.85))           # seeded, reproducible
    assert env._episode_target(0.7) == 0.7                                # the probe's explicit target
    env.mode = EVAL_TEST
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "param:0.6")
    assert env._episode_target() == pytest.approx(0.6)
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "flop:0.5")
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "param:0.8,0.55")
    assert env._episode_target() == pytest.approx(0.55)                    # deepest param point
    monkeypatch.delenv("SPECTRA_EVAL_SIZE_POINTS")
    assert env._episode_target() == pytest.approx(0.6)                     # fallback, with a warning


# ------------------------------------------------------------------ landing / reward / done

class _ScriptedHandler:
    """Learning handler whose val accuracy follows a script, one value per evaluation."""

    def __init__(self, model, accs):
        self.model = model
        self._accs = accs

    def evaluate_model(self, loader):
        return self._accs.pop(0)

    def unfreeze_all_layers(self):
        pass


def _target_env(model, target, accs, *, passes=1, origin_acc=0.90):
    """A NetworkEnv around ``model`` whose recovery is skipped and whose val accuracy is scripted."""
    env = NetworkEnv.__new__(NetworkEnv)
    env.conf = SimpleNamespace(prune=True, passes=passes, save_pruned_checkpoints=False,
                               device="cpu", seed=0, allowed_acc_reduction=10)
    env.mode = AGENT_TRAIN
    env.current_model = model
    env.original_params = utils.calc_num_parameters(model)
    env.original_flops = None
    env.train_loader = _tiny_loader()
    env.val_loader = _tiny_loader(n=8, seed=1)
    env.selected_net_path = "tiny_r20w4.pt"
    env.row_idx = 1
    env.actions_history = []
    env._ratio_cache = {}
    env._pass_locked_layers = set()
    env._rollback_locked_layers = set()
    env._episode_group_cuts = {}
    env.last_step_outcome = {"mode": "identity"}
    env.last_step_layer_idx = None
    env._reset_episode_reward_stats()
    env.original_acc = origin_acc
    env.last_val_acc = origin_acc
    env.target_keep = target
    env._target_prev_acc = origin_acc
    env._target_final = None
    env._layer_sens = None
    env.create_learning_handler = lambda m: _ScriptedHandler(m, accs)
    env._recover_after_prune = lambda *a, **k: "A"
    env._register_group_lock = lambda outcome: None
    env._dependency_groups = lambda mwr: None
    env.tau = lambda: 10.0
    seen = []
    env.feature_extractor = SimpleNamespace(
        encode_to_bert_input=lambda *a, **k: seen.append(k) or {"state": len(seen)})
    return env, seen


def test_a_cut_past_the_target_lands_on_it_and_ends_the_episode(monkeypatch):
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    env, seen = _target_env(model, target=1.0, accs=[0.80])
    mwr = ModelWithRows(model)
    row = _row_index_of_layer(mwr, _layer_index(mwr, model.layer3[0].conv1))
    env.row_idx = row + 1
    half = env.preview_param_ratio(0.5)
    target = 0.5 * (1.0 + half)                 # a 0.5 cut of this layer passes it
    env.target_keep = target
    rate, landed = env._land_on_target(0.5)
    assert 0.5 < rate < 1.0 and landed <= target + 1e-9
    assert env.preview_param_ratio(min(1.0 - 1e-6, rate + 1.0 / 1024)) > target   # the mildest such cut
    assert env._land_on_target(1.0 - 1e-6) == (1.0 - 1e-6, None)                # no change: no landing

    _, reward, done = env.step(0.5)
    kept = env.param_ratio()
    assert done and kept <= target + 1e-9 and kept > half
    assert env._target_final["kept"] == pytest.approx(kept)
    assert reward == pytest.approx(-10.0)        # 0.90 -> 0.80, landed: no penalty
    assert env.episode_target_score() == pytest.approx(-10.0)
    assert seen[-1]["target_extras"] == pytest.approx(fortify.target_channels(kept, target))


def test_a_cut_just_above_the_target_does_not_end_the_walk(monkeypatch):
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    env, _ = _target_env(model, target=1.0, accs=[0.85])
    mwr = ModelWithRows(model)
    env.row_idx = _row_index_of_layer(mwr, _layer_index(mwr, model.layer3[0].conv1)) + 1
    after = env.preview_param_ratio(0.5)
    env.target_keep = after - 0.0025
    assert env._land_on_target(0.5) == (0.5, None)
    _, reward, done = env.step(0.5)
    assert env.param_ratio() == pytest.approx(after)
    assert not done and env._target_final is None
    assert reward == pytest.approx(-5.0)


def test_rewards_telescope_to_the_val_delta_and_a_miss_is_penalised(monkeypatch):
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    num_rows = len(ModelWithRows(model).all_rows) - 1
    accs = [0.90 - 0.01 * (i % 4) - 0.002 * i for i in range(1, num_rows + 1)]
    env, _ = _target_env(model, target=0.2, accs=list(accs))
    rewards, done, steps = [], False, 0
    while not done:
        _, reward, done = env.step(0.9 if steps % 3 == 0 else 1.0)
        rewards.append(reward)
        steps += 1
    assert steps == num_rows                      # one pass, never reached 0.2
    kept = env.param_ratio()
    delta = (accs[-1] - 0.90) * 100.0
    assert kept > 0.2
    expected = delta - 2.0 * (kept - 0.2) * 100.0
    assert fortify.target_score(delta, kept, 0.2) == pytest.approx(expected)
    assert sum(rewards) == pytest.approx(expected)
    assert sum(rewards[:-1]) == pytest.approx((accs[-2] - 0.90) * 100.0)   # telescoping, step by step
    assert env.episode_target_score() == pytest.approx(expected)


def test_flag_off_keeps_the_legacy_reward(monkeypatch):
    monkeypatch.setattr(utils, "compute_reward", lambda *a, **k: 123.0)
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    env, seen = _target_env(model, target=None, accs=[0.88, 0.87])
    _, reward, done = env.step(1.0)
    assert reward == 123.0 and not done and env._target_final is None
    assert seen[-1]["target_extras"] is None
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    env.target_keep, env._target_prev_acc = 0.5, 0.88
    _, reward, _ = env.step(1.0)
    assert reward == pytest.approx(-1.0)                  # 0.88 -> 0.87 replaces the legacy reward


# ------------------------------------------------------------------ agent: episode score and probe

class _StubEnv:
    """Every walk ends after two steps; the fixed-target score is −10 × (1 − target)."""

    selected_net_path = "/x/stub.pt"

    def __init__(self, target=None):
        self.target_keep = target
        self.calls = []
        self.n = 0

    def reset(self, test_net_path=None, target_keep=None):
        self.calls.append((test_net_path, target_keep))
        if target_keep is not None:
            self.target_keep = target_keep
        self.n = 0
        return {"s": torch.zeros(1)}

    def legal_action_mask(self, device=None):
        return torch.tensor([True, True])

    def step(self, rate, ranking=None):
        self.n += 1
        return {"s": torch.zeros(1)}, -1.0, self.n >= 2

    def episode_target_score(self):
        return -10.0 * (1.0 - self.target_keep)


def _stub_agent(env):
    from src.A2C_Agent_Reinforce import A2CAgentReinforce
    agent = A2CAgentReinforce.__new__(A2CAgentReinforce)
    agent.env = env
    agent.conf = SimpleNamespace(rollout_limit=None, device="cpu", compression_rates_dict={0: 1.0, 1: 0.9},
                                 action_rankings_dict={})
    agent.actor_model = lambda state: torch.distributions.Categorical(probs=torch.tensor([[0.3, 0.7]]))
    agent.critic_model = lambda state: torch.zeros(1, 1)
    agent.episode_idx = 7
    return agent


def test_collect_episode_scores_the_fixed_target_return():
    agent = _stub_agent(_StubEnv(target=0.5))
    ep = agent._collect_episode(uniform=False)
    assert ep["done"] and len(ep["steps"]) == 2
    assert ep["val_best_score"] == ep["inbudget_score"] == pytest.approx(-5.0)
    assert ep["target_keep"] == 0.5
    legacy = _stub_agent(_StubEnv())
    legacy.env.episode_checkpoint_score = lambda: 0.25
    legacy.env.episode_val_best_compression = lambda: 0.3
    ep = legacy._collect_episode(uniform=False)
    assert ep["val_best_score"] == 0.3 and "target_keep" not in ep


def test_probe_target_walks_each_net_at_each_target_and_mild_once(monkeypatch):
    monkeypatch.setenv("SPECTRA_PROBE_SCORE", "target")
    monkeypatch.setenv("SPECTRA_TARGET_PROBE_KEEPS", "0.8:0.6")
    env = _StubEnv()
    agent = _stub_agent(env)
    rates_played = []
    step = env.step
    env.step = lambda rate, ranking=None: rates_played.append(rate) or step(rate, ranking)
    nets = ["/x/resnet56-width6_c10.pt", "/x/resnet20-width10_c10.pt"]
    cells = [(nets[0], 0.8), (nets[0], 0.6), (nets[1], 0.8), (nets[1], 0.6)]
    assert agent.probe_score(nets) == pytest.approx(-3.0)
    assert env.calls == cells + cells                       # the mild reference first, then the actor
    assert rates_played[:8] == [0.9] * 8                    # mild: keep 0.9 at every legal step
    assert agent._probe_mild_ref == pytest.approx(-3.0)
    agent.probe_score(nets)
    assert env.calls == cells * 3                           # the reference is walked once


# ------------------------------------------------------------------ group sensitivity

def test_normalise_log_ratio_and_percentile():
    out = group_sensitivity.normalise({0: 0.1, 3: 0.2, 5: 0.4, 7: -0.05})
    # Raw rises clip at 0, the floor is 5 % of the median (0.15), the median of the floored is 0.15.
    span = group_sensitivity.LOG_SPAN
    assert out[0][0] == pytest.approx(np.log(0.1 / 0.15) / span)
    assert out[5][0] == pytest.approx(np.log(0.4 / 0.15) / span)
    assert out[7][0] == pytest.approx(np.log(0.0075 / 0.15) / span)
    assert [out[r][1] for r in (7, 0, 3, 5)] == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert group_sensitivity.normalise({}) == {}


def test_sensitivity_is_measured_once_per_network():
    env = NetworkEnv.__new__(NetworkEnv)
    env.selected_net_path = "/x/a.pt"
    env._sens_cache = {"/x/a.pt": "cached"}
    assert env._group_sensitivity_features() == "cached"


def test_group_sensitivity_features_on_a_thin_resnet():
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    params = utils.calc_num_parameters(model)
    batches = group_sensitivity.calibration_batches(_tiny_loader(n=32), 2, "cpu")
    assert len(batches) == 2
    features, summary = group_sensitivity.layer_features(model, batches, (3, 32, 32))
    mwr = ModelWithRows(model)
    assert features.shape == (len(mwr.all_layers), fortify.STATE_SENS_DIM)
    assert summary["groups"] == len(group_sensitivity.group_plan(mwr)) >= 4
    assert summary["layers"] >= summary["groups"]
    assert features[:, 0].abs().max() <= 1.0
    assert 0.0 <= features[:, 1].min() and features[:, 1].max() <= 1.0
    # Stage 1's residual stream is one group: the stem and every stage-1 conv2 carry its channels.
    stem = features[_layer_index(mwr, model.embed[0])]
    assert torch.count_nonzero(stem) > 0
    for block in model.layer1:
        assert torch.equal(features[_layer_index(mwr, block.conv2)], stem)
    assert utils.calc_num_parameters(model) == params                     # the origin is untouched
