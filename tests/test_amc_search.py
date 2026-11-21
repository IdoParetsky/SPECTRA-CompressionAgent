"""
``SPECTRA_ALLOC_KIND=amc`` (default off; queue row AMC, part C): AMC's per-network DDPG channel-allocation search (He et
al., ECCV 2018) as an allocation kind, ``src/amc_search.py``, ported from the released code (mit-han-lab/amc). One
decision per planned group in plan order; AMC's budget rule on ``plan_agent.FlopModel`` keeps the FLOPs target reachable
after every action; an episode's reward is its plan's val-half accuracy (``plan_proxies.PlanScorer``); the best episode
(ties to the earlier) is landed by ``alloc_walk._land`` and walked like any plan. The TEST half is never read and every
global RNG state is put back. With another kind nothing changes and the module is not imported.

CPU only, no datasets.  python -m pytest tests/test_amc_search.py -v
"""

import copy
import inspect
import math
import os
import random
import re
import statistics
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import RATES, SHAPE, ZOO, _loader, _randomize_norms  # noqa: E402
from tests.test_v10_fixed_target import _target_env  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, amc_search, plan_agent, plan_proxies  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_EVAL_PROXY",
                 "SPECTRA_MIN_WIDTH", "SPECTRA_GROUP_ONCE", "SPECTRA_SEED", "SPECTRA_FT_")
NET = "toy.pt"
# AMC's released defaults, each with its source line in the module docstring (mit-han-lab/amc at amc_search.AMC_COMMIT)
AMC_DEFAULTS = {"episodes": 800, "warmup": 100, "lbound": 0.2, "rbound": 1.0, "rmsize": 100, "bsize": 64, "tau": 0.01,
                "discount": 1.0, "init_delta": 0.5, "delta_decay": 0.95, "lr_a": 1e-4, "lr_c": 1e-3, "hidden1": 300,
                "hidden2": 300, "moving_alpha": 0.5, "window_length": 1}
EPISODE_LINE = re.compile(r"^\[alloc\] amc (\S+) ep=(\d+) (random|ddpg) reward=(\d\.\d{4}) FLOPs x(\d\.\d{3}) "
                          r"params x(\d\.\d{3})( best)?$")
TIMES = ("seconds", "score_seconds")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


class _NeverRead:
    """The TEST loader of the fake env: reading it during the search fails the test."""

    def __iter__(self):
        pytest.fail("the TEST loader was read during the amc search")

    def __len__(self):
        return 0


class _Counted:
    """A loader that counts how many times it is iterated."""

    def __init__(self, loader, reads):
        self.loader, self.reads = loader, reads

    def __iter__(self):
        self.reads.append(1)
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class _Env(types.SimpleNamespace):
    """A fake env whose TEST loader fails the test when it is so much as looked up."""

    @property
    def test_loader(self):
        pytest.fail("the amc search asked the env for its TEST loader")


def _handler(model):
    return ClassificationHandler(model, nn.CrossEntropyLoss())


def _model(name="thin_r20_w4"):
    torch.manual_seed(0)
    model = ZOO[name]().eval()
    _randomize_norms(model)
    return model


def _env(model=None, target_keep=0.6, seed=0, cls=types.SimpleNamespace):
    """A fake env for ``alloc_walk._state``: a tiny origin with toy train / val loaders, the run seed ``conf.seed`` and
    a TEST loader that must never be read."""
    extra = {} if cls is _Env else {"test_loader": _NeverRead()}
    return cls(
        selected_net_path=f"/nets/{NET}", current_model=_model() if model is None else model, target_keep=target_keep,
        conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES, passes=6, seed=seed),
        train_loader=_loader(), val_loader=_loader(seed=1, shuffle=False), create_learning_handler=_handler,
        _input_shape=lambda: SHAPE, _dependency_groups=lambda mwr: channel_groups.build_channel_groups(mwr.model),
        **extra)


def _amc(monkeypatch, episodes=6, warmup=3, proxy="cut", flop=0.6, **flags):
    """The kind under a FLOPs budget whose walk target is ``flop`` (the plan aims 0.02 below it)."""
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "amc")
    monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "flops")
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", f"flop:{flop}")
    monkeypatch.setenv("SPECTRA_ALLOC_AMC_EPISODES", str(episodes))
    monkeypatch.setenv("SPECTRA_ALLOC_AMC_WARMUP", str(warmup))
    if proxy is not None:
        monkeypatch.setenv("SPECTRA_ALLOC_AMC_PROXY", proxy)
    for key, value in flags.items():
        monkeypatch.setenv(key, str(value))


def _run(monkeypatch, env=None, lines=None, records=None):
    """``alloc_walk._state`` on ``env`` with its lines and records captured: ``(entry, lines, records)``."""
    lines, records = [] if lines is None else lines, [] if records is None else records
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
    entry = alloc_walk._state(_env() if env is None else env)
    return entry, lines, records


def _record(records, event):
    found = [k for e, k in records if e == event]
    assert len(found) == 1, [e for e, _k in records]
    return found[0]


def _timeless(record):
    return {key: value for key, value in record.items() if key not in TIMES}


def _ints(table):
    return {int(row): int(width) for row, width in table.items()}


def _plan_of(model):
    return group_sensitivity.group_plan(ModelWithRows(model), channel_groups.build_channel_groups(model))


def _rng_state():
    """The global torch / random / numpy state as plain Python values (comparable with ``==``)."""
    npy = np.random.get_state()
    return torch.get_rng_state().tolist(), random.getstate(), (npy[0], npy[1].tolist(), tuple(npy[2:]))


def _space(name="thin_r20_w4", target=0.58, lbound=0.2, k_min=0.1, min_width=1):
    """A tiny origin's ``GroupEnv`` at a kept-FLOPs ``target`` with what it was built from: ``space``, ``fm`` (the
    ``FlopModel``), ``model``, ``plan``, ``features``, ``flops`` and ``floors``."""
    model = _model(name)
    plan = _plan_of(model)
    fm = plan_agent.FlopModel(model, plan, SHAPE)
    features, flops = amc_search.group_features(model, ModelWithRows(model), plan, SHAPE)
    floors = amc_search.keep_floors(fm.widths0, lbound, k_min, min_width)
    return types.SimpleNamespace(space=amc_search.GroupEnv(fm, floors, target, features, flops), fm=fm, model=model,
                                 plan=plan, features=features, flops=flops, floors=floors)


class _Chain:
    """A toy cost model with AMC's own structure: a chain of layers whose FLOPs are f_i * inputs_i * outputs_i, the
    image (3 channels) and the logits (10) fixed, one group between each pair of layers. ``loop`` > 0 adds a layer
    that reads and writes the middle group (loop * width^2), which no rule linear in the width covers."""

    def __init__(self, widths0=(16, 32, 64), flops=(9.0, 4.0, 2.0, 1.0), loop=0.0):
        self.rows = list(range(len(widths0)))
        self.widths0 = dict(enumerate(widths0))
        self.f, self.loop = list(flops), float(loop)
        self.total0 = self.cost(self.widths0)

    def cost(self, widths):
        w = [3] + [int(widths.get(row, self.widths0[row])) for row in self.rows] + [10]
        return float(sum(f * w[i] * w[i + 1] for i, f in enumerate(self.f)) + self.loop * w[2] * w[2])


def _chain_env(target=0.4, lbound=0.2, loop=0.0):
    chain = _Chain(loop=loop)
    features = [[float(t), float(w0), 3.0, 8.0, 8.0, 1.0, 3.0, 100.0 * (t + 1)] for t, w0 in chain.widths0.items()]
    floors = amc_search.keep_floors(chain.widths0, lbound, 0.1)
    return amc_search.GroupEnv(chain, floors, target, features, [100.0, 200.0, 300.0]), chain


# ------------------------------------------------------------------ the kind, the knobs and AMC's defaults


def test_the_kind_is_registered_beside_the_old_kinds_and_the_defaults_are_amcs(monkeypatch):
    assert alloc_walk.kind() == "sens"                                    # the default is still sens
    assert alloc_walk.AMC_KINDS == ("amc",) and "amc" not in alloc_walk.KINDS
    assert alloc_walk.KINDS[-2:] == alloc_walk.MAGNITUDE_KINDS            # the pinned tuple is as it was
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", " AMC ")
    assert alloc_walk.kind() == "amc"
    for bad in ("amc2", "ddpg", "amc,sens"):
        monkeypatch.setenv("SPECTRA_ALLOC_KIND", bad)
        with pytest.raises(ValueError, match="'amc'"):                    # the message lists the new kind too
            alloc_walk.kind()
    assert amc_search.DEFAULTS == AMC_DEFAULTS and amc_search.hyper_parameters() == AMC_DEFAULTS
    assert amc_search.episodes() == 800 and amc_search.warmup() == 100 and amc_search.proxy() == "bn8"
    assert (amc_search.AGENT_LOW, amc_search.AGENT_HIGH) == (0.0, 1.0) and amc_search.STATE_DIM == 11
    assert len(amc_search.AMC_COMMIT) == 40
    monkeypatch.setenv("SPECTRA_ALLOC_AMC_EPISODES", "6")
    monkeypatch.setenv("SPECTRA_ALLOC_AMC_WARMUP", "3")
    assert amc_search.hyper_parameters() == {**AMC_DEFAULTS, "episodes": 6, "warmup": 3}
    monkeypatch.setenv("SPECTRA_ALLOC_AMC_WARMUP", "0")
    assert amc_search.warmup() == 0
    for name, bad in (("SPECTRA_ALLOC_AMC_EPISODES", "0"), ("SPECTRA_ALLOC_AMC_EPISODES", "many"),
                      ("SPECTRA_ALLOC_AMC_WARMUP", "-1"), ("SPECTRA_ALLOC_AMC_WARMUP", "1.5")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError, match=name):
            amc_search.hyper_parameters()
        monkeypatch.delenv(name)
    for raw, want in ((" BN8 ", "bn8"), ("cut", "cut"), ("fs32", "fs32"), ("ft1", "ft1"), ("", "bn8")):
        monkeypatch.setenv("SPECTRA_ALLOC_AMC_PROXY", raw)
        assert amc_search.proxy() == want
    for bad in ("ft12_4", "rc1", "bn0", "acc"):                           # the plan trainer's names only
        monkeypatch.setenv("SPECTRA_ALLOC_AMC_PROXY", bad)
        with pytest.raises(ValueError):
            amc_search.proxy()
    landing = inspect.signature(alloc_walk.plan_targets).parameters       # the landing every rule runs
    assert (amc_search.LAND_ITERS, amc_search.LAND_TOL) == (landing["iters"].default, landing["tol"].default)


# ------------------------------------------------------------------ the agent's parts


def _truncated_cdf(x, mu, sigma):
    normal = statistics.NormalDist()
    low, high = normal.cdf((0.0 - mu) / sigma), normal.cdf((1.0 - mu) / sigma)
    return (normal.cdf((x - mu) / sigma) - low) / (high - low)


@pytest.mark.parametrize("mu,sigma", [(0.5, 0.5), (0.05, 0.3), (0.97, 0.1), (0.3, 0.02), (0.0, 0.5), (1.0, 0.25)])
def test_the_noise_is_a_truncated_normal_drawn_from_a_private_generator(mu, sigma):
    before = _rng_state()
    rng = random.Random(3)
    draws = sorted(amc_search.truncated_normal(rng, mu, sigma) for _ in range(4000))
    assert _rng_state() == before                                          # the global generators stand
    assert 0.0 <= draws[0] and draws[-1] <= 1.0
    gap = max(abs(_truncated_cdf(x, mu, sigma) - (i + 0.5) / len(draws)) for i, x in enumerate(draws))
    assert gap < 0.03                                                      # Kolmogorov-Smirnov, 0.1 % level at n = 4000
    again = random.Random(3)
    assert sorted(amc_search.truncated_normal(again, mu, sigma) for _ in range(4000)) == draws


def test_the_noise_matches_scipys_truncnorm_which_amc_calls():
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(11)
    for mu, sigma in ((0.5, 0.5), (0.1, 0.3), (0.9, 0.05), (0.6, 0.5 * 0.95 ** 30)):
        draws = [amc_search.truncated_normal(rng, mu, sigma) for _ in range(4000)]
        theirs = stats.truncnorm((0.0 - mu) / sigma, (1.0 - mu) / sigma, loc=mu, scale=sigma)   # lib/agent.py L231
        assert stats.kstest(draws, theirs.cdf).pvalue > 0.001
        assert sum(draws) / len(draws) == pytest.approx(float(theirs.mean()), abs=4.0 * float(theirs.std()) / 63.0)


def test_the_noise_dies_out_with_its_std_and_keeps_the_stream_aligned():
    rng = random.Random(0)
    assert amc_search.truncated_normal(rng, 0.37, 0.0) == 0.37             # an underflowed std: the actor's action
    assert amc_search.truncated_normal(rng, 0.37, 0.5 * 0.95 ** 699) == pytest.approx(0.37, abs=1e-12)
    assert amc_search.truncated_normal(rng, 1.7, 0.1) <= 1.0 and amc_search.truncated_normal(rng, -0.4, 0.1) >= 0.0
    twin = random.Random(0)
    for _ in range(4):                                                     # one uniform per draw, whatever the std
        twin.random()
    assert rng.random() == twin.random()


def test_the_replay_memory_is_amcs_ring_buffer_and_never_draws_the_newest_entry():
    memory = amc_search.ReplayMemory(5, random.Random(0))
    with pytest.raises(RuntimeError):
        memory.sample(2)
    for i in range(3):
        memory.append([float(i)], 0.1 * i, 1.0 + i, i == 2)
    assert len(memory) == 3 and memory[0] == ((0.0,), 0.0, 1.0, False) and memory[2] == ((2.0,), 0.2, 3.0, True)
    with pytest.raises(KeyError):
        memory[3]
    state0, action, reward, state1, alive = memory.sample(64)             # two transitions: drawn with replacement
    assert len(state0) == len(action) == len(reward) == len(state1) == len(alive) == 64
    assert {s[0] for s in state0} == {0.0, 1.0}                           # never entry 2, the newest
    assert all(n[0] == s[0] + 1 and a == pytest.approx(0.1 * s[0]) and r == 1.0 + s[0]
               for s, a, r, n in zip(state0, action, reward, state1))     # the next state is the next entry's
    assert alive == [1.0] * 64
    for i in range(3, 9):
        memory.append([float(i)], 0.1 * i, 1.0 + i, False)
    assert len(memory) == 5 and [memory[i][0][0] for i in range(5)] == [4.0, 5.0, 6.0, 7.0, 8.0]   # the oldest went
    big = amc_search.ReplayMemory(200, random.Random(1))
    for i in range(101):                                                  # ten-step episodes
        big.append([float(i)], 0.0, 0.5, i % 10 == 9)
    first_steps = set()
    for _ in range(40):
        state0, _a, _r, state1, alive = big.sample(64)
        index = [int(s[0]) for s in state0]
        assert len(set(index)) == 64 and max(index) <= 99                 # without replacement, never the newest
        assert all(int(n[0]) == i + 1 for i, n in zip(index, state1))
        assert all(m == (0.0 if i % 10 == 9 else 1.0) for i, m in zip(index, alive))   # 0 beyond a terminal step
        first_steps |= {i for i in index if i % 10 == 0 and i > 0}
    assert first_steps == set(range(10, 100, 10))                         # an episode's first step is drawn like any


def test_the_actor_and_critic_are_amcs_networks_from_private_generators():
    before = _rng_state()
    hp = amc_search.hyper_parameters()
    agent = amc_search.DDPG(amc_search.STATE_DIM, 7, hp, seed=5)
    twin = amc_search.DDPG(amc_search.STATE_DIM, 7, hp, seed=5)
    other = amc_search.DDPG(amc_search.STATE_DIM, 7, hp, seed=6)
    assert _rng_state() == before                                          # building them moved no global generator
    shapes = {name: tuple(p.shape) for name, p in agent.actor.named_parameters()}
    assert shapes == {"fc1.weight": (300, 11), "fc1.bias": (300,), "fc2.weight": (300, 300), "fc2.bias": (300,),
                      "fc3.weight": (1, 300), "fc3.bias": (1,)}
    shapes = {name: tuple(p.shape) for name, p in agent.critic.named_parameters()}
    assert shapes == {"fc11.weight": (300, 11), "fc11.bias": (300,), "fc12.weight": (300, 1), "fc12.bias": (300,),
                      "fc2.weight": (300, 300), "fc2.bias": (300,), "fc3.weight": (1, 300), "fc3.bias": (1,)}
    for net, target in ((agent.actor, agent.actor_target), (agent.critic, agent.critic_target)):
        assert all(torch.equal(a, b) for a, b in zip(net.parameters(), target.parameters()))   # equal at the start
        for layer in net.modules():
            if isinstance(layer, nn.Linear):                              # PyTorch's default range for a Linear
                bound = 1.0 / math.sqrt(layer.in_features)
                assert float(layer.weight.abs().max()) <= bound * (1.0 + 1e-6)
                assert float(layer.bias.abs().max()) <= bound * (1.0 + 1e-6)
                assert float(layer.weight.abs().max()) > 0.9 * bound
    assert all(torch.equal(a, b) for a, b in zip(agent.actor.parameters(), twin.actor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(agent.critic.parameters(), twin.critic.parameters()))
    assert not torch.equal(agent.actor.fc1.weight, other.actor.fc1.weight)
    assert isinstance(agent.actor_optim, torch.optim.Adam) and isinstance(agent.critic_optim, torch.optim.Adam)
    assert agent.actor_optim.param_groups[0]["lr"] == 1e-4 and agent.critic_optim.param_groups[0]["lr"] == 1e-3
    assert agent.actor_optim.param_groups[0]["betas"] == (0.9, 0.999)
    assert agent.memory.limit == 100 * 7 and agent.moving_average is None
    state = [0.1 * i for i in range(amc_search.STATE_DIM)]
    assert agent.delta(100 + 1) == pytest.approx(0.5 * 0.95)              # the std of the noise, from the warm-up on
    assert agent.delta(100 + 10) == pytest.approx(0.5 * 0.95 ** 10)
    actions = [agent.select_action(state, 101) for _ in range(200)]
    assert all(0.0 <= a <= 1.0 for a in actions) and len(set(actions)) == 200
    assert actions == [twin.select_action(state, 101) for _ in range(200)]   # the same seed, the same noise
    mu = float(agent.actor(torch.tensor([state], dtype=torch.float32)).item())
    assert 0.0 < mu < 1.0 and agent.select_action(state, 100 + 700) == pytest.approx(mu, abs=1e-9)   # the noise is gone
    draws = [agent.random_action() for _ in range(2000)]
    assert all(0.0 <= a < 1.0 for a in draws) and 0.45 < sum(draws) / len(draws) < 0.55
    assert min(draws) < 0.05 and max(draws) > 0.95                         # uniform over [0, 1), not [lbound, 1]
    assert _rng_state() == before


def _amc_update(actor, actor_target, critic, critic_target, actor_optim, critic_optim, batch, moving_average,
                tau=0.01, discount=1.0, alpha=0.5):
    """``lib/agent.py`` L110-158 transcribed as it stands (NumPy batches, ``to_tensor`` as float32; only the critic's
    two inputs are passed as two arguments, a list there), for the check below. Returns the new moving average."""
    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = batch
    to_tensor = lambda array: torch.from_numpy(array).float()  # noqa: E731
    batch_mean_reward = np.mean(reward_batch)
    if moving_average is None:
        moving_average = batch_mean_reward
    else:
        moving_average += alpha * (batch_mean_reward - moving_average)
    reward_batch = reward_batch - moving_average
    with torch.no_grad():
        next_q_values = critic_target(to_tensor(next_state_batch), actor_target(to_tensor(next_state_batch)))
    target_q_batch = to_tensor(reward_batch) + discount * to_tensor(terminal_batch.astype(float)) * next_q_values
    critic.zero_grad()
    q_batch = critic(to_tensor(state_batch), to_tensor(action_batch))
    value_loss = nn.MSELoss()(q_batch, target_q_batch)
    value_loss.backward()
    critic_optim.step()
    actor.zero_grad()
    policy_loss = -critic(to_tensor(state_batch), actor(to_tensor(state_batch)))
    policy_loss = policy_loss.mean()
    policy_loss.backward()
    actor_optim.step()
    for target, source in ((actor_target, actor), (critic_target, critic)):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    return moving_average


def test_an_update_is_amcs_critic_step_actor_step_baseline_and_soft_target_update():
    hp = {**amc_search.DEFAULTS, "bsize": 16}
    agent = amc_search.DDPG(4, 3, hp, seed=2)
    filler = random.Random(9)
    for i in range(40):                                                   # three-step episodes, one reward each
        if i % 3 == 0:
            reward = filler.random()
        agent.observe(reward, [filler.random() for _ in range(4)], filler.random(), i % 3 == 2)
    ref = {name: copy.deepcopy(getattr(agent, name)) for name in ("actor", "actor_target", "critic", "critic_target")}
    ref_actor_optim = torch.optim.Adam(ref["actor"].parameters(), lr=1e-4)
    ref_critic_optim = torch.optim.Adam(ref["critic"].parameters(), lr=1e-3)
    moving_average = None
    for step in range(3):
        twin = random.Random()
        twin.setstate(agent.rng.getstate())                               # the batch the agent is about to draw
        s0, a, r, s1, alive = amc_search.ReplayMemory.sample(_view(agent.memory, twin), 16)
        batch = (np.array(s0, "double"), np.array(a, "double").reshape(16, 1), np.array(r, "double").reshape(16, 1),
                 np.array(s1, "double"), np.array(alive, "double").reshape(16, 1))
        moving_average = _amc_update(ref["actor"], ref["actor_target"], ref["critic"], ref["critic_target"],
                                     ref_actor_optim, ref_critic_optim, batch, moving_average)
        with torch.no_grad():                                             # the caller's mode does not switch it off
            agent.update_policy()
        assert agent.updates == step + 1 and agent.moving_average == pytest.approx(float(moving_average), rel=1e-12)
        for name, net in ref.items():
            for ours, theirs in zip(getattr(agent, name).parameters(), net.parameters()):
                torch.testing.assert_close(ours.detach(), theirs.detach(), rtol=1e-5, atol=1e-7)
    start = amc_search.DDPG(4, 3, hp, seed=2)
    assert not torch.equal(agent.actor.fc3.weight, start.actor.fc3.weight)          # both networks moved
    assert not torch.equal(agent.critic.fc3.weight, start.critic.fc3.weight)
    assert not torch.equal(agent.actor_target.fc3.weight, start.actor_target.fc3.weight)
    drift = (agent.actor_target.fc3.weight - start.actor_target.fc3.weight).abs().max()
    assert drift < (agent.actor.fc3.weight - start.actor.fc3.weight).abs().max()    # the targets trail (tau 0.01)


def _view(memory, rng):
    """``memory``'s entries behind another generator, so a test can draw the batch the agent will draw."""
    view = amc_search.ReplayMemory(memory.limit, rng)
    view.data, view.start, view.length = memory.data, memory.start, memory.length
    return view


# ------------------------------------------------------------------ the state


def test_minmax_scales_each_column_over_the_network_and_leaves_a_constant_one():
    rows = [[0.0, 3.0, 16.0], [1.0, 3.0, 64.0], [2.0, 3.0, 32.0]]
    assert amc_search.minmax(rows) == [[0.0, 3.0, 0.0], [0.5, 3.0, 1.0], [1.0, 3.0, pytest.approx(1 / 3)]]
    assert rows == [[0.0, 3.0, 16.0], [1.0, 3.0, 64.0], [2.0, 3.0, 32.0]] and amc_search.minmax([]) == []


def test_a_group_is_described_by_its_first_producer_and_the_flops_of_all_its_producers():
    model = _model()
    model.layer1.train()                                                   # mixed flags must come back as they were
    modes = {name: module.training for name, module in model.named_modules()}
    mwr, plan = ModelWithRows(model), _plan_of(model)
    features, flops = amc_search.group_features(model, mwr, plan, SHAPE)
    assert {name: module.training for name, module in model.named_modules()} == modes
    model.eval()
    macs = utils.per_module_macs(model, SHAPE)
    assert len(features) == len(flops) == len(plan) == 12
    assert all(len(row) == len(amc_search.STATIC_FEATURES) == 8 for row in features)
    streams = 0
    for t, ((group, row), feat) in enumerate(zip(plan, features)):
        layer = mwr.all_layers[mwr.row_to_main_layer[row]]                 # the layer at the group's first walk row:
        assert any(layer is producer for producer in group.producers)      # its first producer in depth order
        assert feat[0] == t and feat[1] == group.width == layer.out_channels
        assert feat[2] == layer.in_channels and feat[5] == layer.stride[0] and feat[6] == layer.kernel_size[0]
        assert feat[7] == flops[t] == sum(macs[id(m)] for m in list(group.producers) + list(group.depthwise)) > 0
        streams += len(group.producers) > 1
    assert streams == 3                                                    # one residual stream per stage, decided once
    assert features[0][2:5] == [3.0, 32.0, 32.0]                           # the stem reads the 32 x 32 image
    assert {feat[3] for feat in features} == {feat[4] for feat in features} == {32.0, 16.0, 8.0}
    assert {feat[5] for feat in features} == {1.0, 2.0} and {feat[6] for feat in features} == {3.0}
    assert sum(flops) < utils.calc_flops(model, SHAPE)                     # the classifier and the element-wise ops
    mobile = _model("mbv2_x05")                                            # a depthwise conv writes its group too
    mwr, plan = ModelWithRows(mobile), _plan_of(mobile)
    features, flops = amc_search.group_features(mobile, mwr, plan, SHAPE)
    macs = utils.per_module_macs(mobile, SHAPE)
    owned = [(group, f) for (group, _row), f in zip(plan, flops) if group.depthwise]
    assert owned and all(f == sum(macs[id(m)] for m in list(group.producers) + list(group.depthwise))
                         > sum(macs[id(m)] for m in group.producers) for group, f in owned)


def test_the_state_is_the_eleven_features_with_reduced_rest_and_the_previous_keep():
    built = _space(target=0.58)
    space, fm = built.space, built.fm
    groups = len(space.rows)
    state = space.reset()
    assert len(state) == amc_search.STATE_DIM == 11 and len(space.static) == groups == 12
    assert [row[0] for row in space.static] == pytest.approx([t / (groups - 1) for t in range(groups)])
    assert sorted({row[1] for row in space.static}) == pytest.approx([0.0, 1 / 3, 1.0])   # widths 4, 8 and 16
    assert all(0.0 <= row[j] <= 1.0 for row in space.static for j in (0, 1, 2, 3, 4, 5, 7))
    assert [row[6] for row in space.static] == [3.0] * groups              # every kernel is 3: a constant column stays
    assert state[:8] == space.static[0] and state[8:] == (0.0, pytest.approx(space.rest[0]), 1.0)
    assert space.rest == pytest.approx([sum(built.flops[t + 1:]) / fm.total0 for t in range(groups)])
    assert space.rest[-1] == 0.0 and all(a > b for a, b in zip(space.rest, space.rest[1:]))
    decided = {}
    for t in range(groups):
        row = space.rows[t]
        state, done = space.step(0.5)
        decided[row] = space.widths[row]
        assert decided[row] == round(0.5 * fm.widths0[row])                # half of 4, 8 or 16: the wall is not near
        if t + 1 < groups:
            reduced = 1.0 - fm.cost({**fm.widths0, **decided}) / fm.total0  # the FLOPs the decisions so far removed
            assert state[:8] == space.static[t + 1] and not done
            assert state[8:] == (pytest.approx(reduced), pytest.approx(space.rest[t + 1]), 0.5)
            assert 0.0 < state[8] < 1.0
        else:
            assert done and state[:8] == space.static[t]                   # the last step returns its own state
    first = space.reset()
    assert first[8:] == (0.0, pytest.approx(space.rest[0]), 1.0) and space.widths == {} and space.t == 0


# ------------------------------------------------------------------ the budget rule


def test_the_wall_is_amcs_linear_rule_on_a_chain_of_layers():
    space, chain = _chain_env(target=0.4)
    budget = 0.4 * chain.total0
    rng = random.Random(4)
    capped = 0
    for _ in range(300):
        space.reset()
        for t, row in enumerate(space.rows):
            # AMC's _action_wall (env L273-296): FLOPs = other + this * ratio with the groups before as decided and the
            # groups after at their lowest; the largest ratio is (target - other) / this
            widths = {**space.lo, **space.widths}
            zero, one = dict(widths), dict(widths)
            zero[row], one[row] = 0, chain.widths0[row]
            other, this = chain.cost(zero), chain.cost(one) - chain.cost(zero)
            max_preserve_ratio = (budget - other) / this
            widest = min(chain.widths0[row], int(math.floor(max_preserve_ratio * chain.widths0[row] + 1e-9)))
            assert space.wall() == max(space.lo[row], widest)
            action = rng.random()
            space.step(action)
            want = amc_search.realised_width(chain.widths0[row], max(action, 0.2))
            assert space.widths[row] == max(space.lo[row], min(want, widest))
            capped += want > widest
        assert chain.cost(space.widths) <= budget
    assert capped > 50                                                     # the rule did bind


@pytest.mark.parametrize("loop", [0.0, 1.5])
def test_the_wall_is_the_widest_feasible_width_also_where_the_cost_is_not_linear(loop):
    space, chain = _chain_env(target=0.35, loop=loop)                      # loop: a layer reading and writing group 1
    budget = 0.35 * chain.total0
    rng = random.Random(8)
    capped = 0
    for _ in range(200):
        space.reset()
        for row in space.rows:
            rest = {**space.lo, **space.widths}
            fits = [w for w in range(space.lo[row], chain.widths0[row] + 1) if chain.cost({**rest, row: w}) <= budget]
            assert fits and space.wall() == max(fits)                      # by brute force over every width
            space.step(rng.random())
            capped += space.widths[row] == max(fits) < chain.widths0[row]
            assert chain.cost({**space.lo, **space.widths}) <= budget      # the rest can still meet the target
        assert chain.cost(space.widths) <= budget
    assert capped > 20


@pytest.mark.parametrize("name,target", [("thin_r20_w4", 0.58), ("thin_r20_w4", 0.25), ("r20_chenyaofo", 0.43),
                                         ("r20_chenyaofo", 0.18), ("mbv2_x05", 0.43), ("vgg11_bn", 0.18)])
def test_every_action_stays_feasible_under_the_wall(name, target):
    built = _space(name, target)
    space, fm, model, plan = built.space, built.fm, built.model, built.plan
    budget = target * fm.total0
    slack = 1e-9 * fm.total0
    rng = random.Random(5)
    policies = [("ones", lambda: 1.0), ("zeros", lambda: 0.0), ("out of range", lambda: rng.uniform(-1.0, 2.0)),
                ("greedy then random", None)] + [("random", rng.random)] * 4
    assert fm.cost(space.lo) <= budget                                    # the target is reachable to begin with
    bound = 0
    for label, policy in policies:
        state, done, t = space.reset(), False, 0
        while not done:
            row, w0 = space.rows[space.t], fm.widths0[space.rows[space.t]]
            before = {**space.lo, **space.widths}
            cap = space.wall()
            action = (1.0 if t < 2 else rng.random()) if policy is None else policy()
            state, done = space.step(action)
            width = space.widths[row]
            keep = max(min(max(action, 0.0), 1.0), space.floor[row])
            assert width == max(space.lo[row], min(cap, amc_search.realised_width(w0, keep)))
            assert space.lo[row] <= width <= w0 and space.keeps[row] == width / w0
            # the claim: with this group decided, the groups not yet decided can still meet the target at their lowest
            assert fm.cost({**space.lo, **space.widths}) <= budget + slack
            if cap < w0:                                                   # and the wall is tight: one channel more
                over = dict(before)                                        # would put the target out of reach
                over[row] = cap + 1
                assert fm.cost(over) > budget + slack
                bound += width == cap
            assert len(state) == 11 and all(math.isfinite(v) for v in state)
            t += 1
        assert set(space.widths) == set(fm.rows)
        assert fm.cost(space.widths) <= budget + slack                    # a finished plan never keeps more
        if label in ("ones", "zeros"):
            cut = alloc_walk.cut_to(model, plan, plan_agent.rates_of(space.widths, fm.widths0), SHAPE)
            assert utils.calc_flops(cut, SHAPE) / fm.total0 == pytest.approx(fm.kept(space.widths), abs=1e-9)
            assert utils.calc_flops(cut, SHAPE) <= budget + slack          # the real cut agrees with the cost model
            assert alloc_walk.group_widths(cut, fm.rows) == space.widths
        if label in ("ones", "zeros", "out of range", "greedy then random"):
            # the landing's own cut at scale 1 (keeps clipped to the floors, as _land clips them): the same widths
            landed = {row: min(1.0, max(space.floor[row], space.keeps[row])) for row in fm.rows}
            again = alloc_walk.cut_to(model, plan, landed, SHAPE)
            assert alloc_walk.group_widths(again, fm.rows) == space.widths
        if label == "zeros":
            assert space.widths == space.lo and fm.kept(space.widths) == pytest.approx(space.lowest)
        if label == "ones":
            assert fm.kept(space.widths) > space.lowest                    # the budget is spent where it is asked for
    assert bound > 0


def test_the_floors_are_amcs_lbound_and_never_below_the_rules_floors():
    widths0 = {1: 4, 2: 16, 3: 64}
    assert amc_search.keep_floors(widths0, 0.2, 0.1) == {1: 0.2, 2: 0.2, 3: 0.2}       # AMC's lbound
    assert amc_search.keep_floors(widths0, 0.2, 0.3) == {1: 0.3, 2: 0.3, 3: 0.3}       # SPECTRA_ALLOC_MIN_KEEP above it
    assert amc_search.keep_floors(widths0, 0.2, 0.1, min_width=8) == {1: 1.0, 2: 0.5, 3: 0.2}   # SPECTRA_PLAN_MIN_WIDTH
    assert [amc_search.realised_width(16, k) for k in (1.0, 0.999, 0.5, 0.2, 0.0)] == [16, 15, 8, 3, 1]
    built = _space(min_width=3)
    space, fm = built.space, built.fm
    assert all(space.lo[row] >= min(3, fm.widths0[row]) for row in fm.rows)
    done = False
    space.reset()
    while not done:
        _state, done = space.step(0.0)
    assert space.widths == space.lo and all(space.widths[row] >= min(3, fm.widths0[row]) for row in fm.rows)


def test_a_target_below_what_the_floors_leave_is_refused(monkeypatch):
    built = _space(target=0.58)
    space, fm, features, flops, floors = built.space, built.fm, built.features, built.flops, built.floors
    assert 0.0 < space.lowest < 0.2 and space.lowest == pytest.approx(fm.kept(space.lo))
    with pytest.raises(ValueError, match="lowest keep"):
        amc_search.GroupEnv(fm, floors, space.lowest - 0.01, features, flops)
    amc_search.GroupEnv(fm, floors, space.lowest + 0.001, features, flops)   # just reachable: accepted
    _amc(monkeypatch, flop=round(space.lowest, 3))                         # the plan aims 0.02 below it
    lines, records = [], []
    with pytest.raises(ValueError, match=f"SPECTRA_ALLOC_KIND=amc on {NET}: the FLOPs target"):
        _run(monkeypatch, lines=lines, records=records)
    assert records == [] and not any("ep=" in line for line in lines)      # refused before any episode


# ------------------------------------------------------------------ the loop


class _SpyAgent:
    """Stands in for the DDPG agent: fixed actions, every call noted."""

    def __init__(self):
        self.calls = []

    def random_action(self):
        self.calls.append(("random",))
        return 0.9

    def select_action(self, state, episode):
        self.calls.append(("select", episode, tuple(state)))
        return 0.5

    def observe(self, reward, state, action, terminal):
        self.calls.append(("observe", reward, tuple(state), action, terminal))

    def update_policy(self):
        self.calls.append(("update",))


def test_the_loop_is_amcs_warm_up_then_one_update_per_stored_step_and_the_first_best_wins():
    space, chain = _chain_env(target=0.4)
    agent, scored, logged = _SpyAgent(), [], []
    script = iter([0.3, 0.7, 0.5, 0.7, 0.2])
    runs, best = amc_search.search(space, agent, lambda widths: scored.append(dict(widths)) or next(script), 5, 1,
                                   lambda episode, run, is_best: logged.append((episode, run["reward"], is_best)))
    assert len(runs) == 5 and best == 1                                    # 0.7 first at episode 1: the tie stays there
    assert logged == [(0, 0.3, True), (1, 0.7, True), (2, 0.5, False), (3, 0.7, False), (4, 0.2, False)]
    assert [run["random"] for run in runs] == [True, True, False, False, False]   # episode <= warm-up acts at random
    assert [run["reward"] for run in runs] == [0.3, 0.7, 0.5, 0.7, 0.2] and scored == [run["widths"] for run in runs]
    calls, steps = iter(agent.calls), len(space.rows)
    for episode, run in enumerate(runs):
        acted = [next(calls) for _ in range(steps)]
        if episode <= 1:
            assert acted == [("random",)] * steps
        else:
            assert [a[:2] for a in acted] == [("select", episode)] * steps
            assert [a[2][0] for a in acted] == [row[0] for row in space.static]    # each group's own state, in order
        action = 0.9 if episode <= 1 else 0.5
        for t in range(steps):                                             # stored after the episode, with its reward
            kind, reward, state, stored, terminal = next(calls)
            assert (kind, reward, stored, terminal) == ("observe", run["reward"], action, t == steps - 1)
            assert state[0] == space.static[t][0]
            if episode > 1:
                assert next(calls) == ("update",)                          # one update per stored step, after warm-up
        assert all(run["keeps"][row] == run["widths"][row] / chain.widths0[row] for row in space.rows)
        assert chain.cost(run["widths"]) <= 0.4 * chain.total0             # the wall stood between action and plan
    assert next(calls, None) is None
    assert all(run["widths"] == {0: 14, 1: 29, 2: 13} for run in runs[:2])  # 0.9 asked of each: the wall cut the last
    assert all(run["widths"] == {0: 8, 1: 16, 2: 32} for run in runs[2:])   # 0.5 asked of each: within the budget
    runs, best = amc_search.search(space, _SpyAgent(), lambda widths: 0.5, 3, 7)   # warm-up beyond the search
    assert best == 0 and all(run["random"] for run in runs)


def test_a_real_agent_learns_on_a_toy_reward_it_can_see():
    space, chain = _chain_env(target=0.4)
    hp = {**amc_search.DEFAULTS, "episodes": 60, "warmup": 15}
    agent = amc_search.DDPG(amc_search.STATE_DIM, len(space.rows), hp, seed=3)
    reward = lambda widths: widths[0] / chain.widths0[0]                   # keep the first group wide  # noqa: E731
    runs, best = amc_search.search(space, agent, reward, 60, 15)
    assert agent.updates == (60 - 16) * len(space.rows) and len(agent.memory) == 60 * len(space.rows)
    early = sum(run["reward"] for run in runs[:16]) / 16.0                 # the random episodes
    late = sum(run["reward"] for run in runs[-15:]) / 15.0
    assert late > early + 0.2 and late > 0.85 and runs[best]["reward"] >= late   # the policy moved toward the reward


def test_a_plan_of_one_group_is_searched_and_an_empty_plan_is_refused(monkeypatch):
    chain = _Chain(widths0=(16,), flops=(9.0, 1.0))                        # cost 37 per channel
    floors = amc_search.keep_floors(chain.widths0, 0.2, 0.1)
    space = amc_search.GroupEnv(chain, floors, 0.5, [[0.0, 16.0, 3.0, 8.0, 8.0, 1.0, 3.0, 100.0]], [100.0])
    assert space.reset() == (0.0, 16.0, 3.0, 8.0, 8.0, 1.0, 3.0, 100.0, 0.0, 0.0, 1.0)   # one row: nothing to scale by
    hp = {**amc_search.DEFAULTS, "episodes": 8, "warmup": 2}
    agent = amc_search.DDPG(amc_search.STATE_DIM, 1, hp, seed=1)
    runs, best = amc_search.search(space, agent, lambda widths: widths[0] / 16.0, 8, 2)
    assert len(runs) == 8 and agent.updates == 5 and len(agent.memory) == 8      # every step is its episode's last
    assert all(3 <= run["widths"][0] <= 8 for run in runs)                 # lbound below, the wall above (296 / 37)
    assert runs[best]["reward"] == max(run["reward"] for run in runs)
    with pytest.raises(ValueError, match="no group"):
        amc_search.GroupEnv(_NoRows(), {}, 0.5, [], [])
    _amc(monkeypatch)
    monkeypatch.setattr(group_sensitivity, "group_plan", lambda *a, **k: [])
    lines, records = [], []
    with pytest.raises(ValueError, match=f"SPECTRA_ALLOC_KIND=amc on {NET}: the plan has no group"):
        _run(monkeypatch, lines=lines, records=records)
    assert lines == [] and records == []


class _NoRows:
    rows, widths0, total0 = [], {}, 1.0

    def cost(self, widths):
        return 1.0


# ------------------------------------------------------------------ through alloc_walk._state: the required properties


def test_two_runs_with_the_same_seeds_give_the_same_plan_and_another_seed_another_search(monkeypatch):
    _amc(monkeypatch, episodes=6, warmup=3, proxy="bn2", SPECTRA_ALLOC_SAMPLE_SEED="5")
    a, lines_a, ra = _run(monkeypatch, _env(seed=2))
    b, lines_b, rb = _run(monkeypatch, _env(seed=2))
    ra, rb = _record(ra, "alloc_amc"), _record(rb, "alloc_amc")
    assert a["widths"] == b["widths"] and _timeless(ra) == _timeless(rb)
    episodes_a, episodes_b = ([line for line in lines if EPISODE_LINE.match(line)] for lines in (lines_a, lines_b))
    assert len(episodes_a) == 6 and episodes_a == episodes_b
    assert ra["seed"] == alloc_walk.randbest_seed(5, 2) == 5 * 100003 + 2
    assert (ra["sample_seed"], ra["run_seed"]) == (5, 2)
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SEED", "6")
    _c, _lines, rc = _run(monkeypatch, _env(seed=2))
    rc = _record(rc, "alloc_amc")
    assert rc["seed"] != ra["seed"] and rc["flops"] != ra["flops"]         # other actions, other plans
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SEED", "5")
    _d, _lines, rd = _run(monkeypatch, _env(seed=3))                       # another run seed (SPECTRA_SEED)
    rd = _record(rd, "alloc_amc")
    assert rd["seed"] != ra["seed"] and rd["flops"] != ra["flops"]


@pytest.mark.parametrize("proxy", ["cut", "bn3", "fs2"])
def test_the_scorer_is_never_asked_for_the_test_half(monkeypatch, proxy):
    _amc(monkeypatch, episodes=5, warmup=2, proxy=proxy, SPECTRA_EVAL_PROXY_SEED="4")
    env = _env(cls=_Env)                                                   # env.test_loader fails the test if touched
    reads = []
    env.val_loader = _Counted(env.val_loader, reads)
    built, asked = [], []
    real = plan_proxies.PlanScorer

    class Spy(real):
        def __init__(self, model, plan, val_loader, device, **kwargs):
            built.append({"val": val_loader, **kwargs})
            super().__init__(model, plan, val_loader, device, **kwargs)

        def score(self, widths, proxy_name, batches):
            asked.append((dict(widths), proxy_name))
            return super().score(widths, proxy_name, batches)

    monkeypatch.setattr(plan_proxies, "PlanScorer", Spy)
    _entry, _lines, records = _run(monkeypatch, env)
    assert len(built) == 1 and built[0]["val"] is env.val_loader           # one scorer, built on the val half
    assert built[0]["train_loader"] is env.train_loader                   # bn / fs batches come from the train split
    loaders = [value for value in built[0].values()
               if isinstance(value, (torch.utils.data.DataLoader, _Counted, _NeverRead))]
    assert len(loaders) == 2 and all(loader is env.val_loader or loader is env.train_loader for loader in loaders)
    assert len(reads) == 1                                                 # and read once, for all episodes
    rec = _record(records, "alloc_amc")
    assert rec["proxy"] == proxy and rec["proxy_seed"] == 4 and all(name == proxy for _w, name in asked)
    assert len(asked) == rec["scored"] <= rec["episodes"] == 5
    assert all(0.0 <= r <= 1.0 for r in rec["rewards"])


def test_the_reward_is_the_val_score_a_seen_plan_is_not_scored_again_and_the_first_best_is_landed(monkeypatch):
    _amc(monkeypatch, episodes=6, warmup=2, proxy="fs2")
    table, order = {}, []
    script = iter([0.30, 0.90, 0.50, 0.90, 0.10, 0.20])

    def fake(self, widths, proxy, batches):
        key = tuple(sorted(widths.items()))
        assert key not in table, "a plan seen before was scored again"
        table[key] = next(script)
        order.append((dict(widths), proxy, len(batches)))
        return table[key]

    monkeypatch.setattr(plan_proxies.PlanScorer, "score", fake)
    landed = []
    real_land = alloc_walk._land
    monkeypatch.setattr(alloc_walk, "_land", lambda *a: landed.append(a) or real_land(*a))
    entry, lines, records = _run(monkeypatch)
    rec, planned = _record(records, "alloc_amc"), _record(records, "alloc_plan")
    assert all(proxy == "fs2" and n_batches == 2 for _w, proxy, n_batches in order) and rec["fs_lr"] == 0.01
    assert rec["scored"] == len(order) == len(table) == 6                  # six different plans, each scored once
    assert rec["rewards"] == [0.30, 0.90, 0.50, 0.90, 0.10, 0.20] and rec["score"] == 0.90
    pick = rec["pick"]
    assert pick == 1                                                       # the highest reward, the earlier on a tie
    assert table[tuple(sorted(_ints(rec["widths"]).items()))] == rec["score"]
    episodes = [EPISODE_LINE.match(line) for line in lines if line.startswith(f"[alloc] amc {NET} ep=")]
    assert len(episodes) == 6 and all(episodes) and [int(m.group(2)) for m in episodes] == list(range(6))
    assert [m.group(3) for m in episodes] == ["random"] * 3 + ["ddpg"] * 3   # warm-up 2: episodes 0, 1 and 2
    assert [float(m.group(4)) for m in episodes] == pytest.approx(rec["rewards"], abs=5e-5)
    assert [float(m.group(5)) for m in episodes] == pytest.approx(rec["flops"], abs=5e-4)
    best_so_far = [max(rec["rewards"][:i + 1]) for i in range(6)]
    assert [bool(m.group(7)) for m in episodes] == [i == 0 or best_so_far[i] > best_so_far[i - 1] for i in range(6)]
    # the landing: the rules' bisection, on the best episode's keeps as weights, at the plan's target, on FLOPs
    assert len(landed) == 1
    _model_, _plan, rows, weights, floors, target, _size, _size0, shape, iters, tol = landed[0]
    assert weights == {int(row): keep for row, keep in rec["keeps"].items()} and target == pytest.approx(0.58)
    assert (shape, iters, tol) == (SHAPE, 16, 0.003) and set(floors.values()) == {0.2} and list(weights) == rows
    assert entry["widths"] == _ints(rec["landed"]["widths"]) and planned["kept"] == pytest.approx(rec["landed"]["kept"])
    assert rec["landed"]["moved"] == sum(entry["widths"][row] != _ints(rec["widths"])[row] for row in rows)
    # the gates' line, last, naming the kind; the summary before it; the plan record
    assert lines[-1].startswith(f"[alloc] {NET}: amc best episode {pick} of 6 (val {rec['score']:.4f}, proxy fs2, "
                                f"warm-up 2, seed 0) plan keeps x{planned['kept']:.3f} of the FLOPs (target x0.580 = "
                                "walk target")
    assert lines[-2].startswith(f"[alloc] amc {NET}: best episode {pick} of 6 (warm-up 2), val {rec['score']:.4f} at "
                                f"FLOPs x{rec['kept']:.3f} ")
    assert f"proxy fs2, {rec['scored']} plans scored, {rec['updates']} updates; landed x" in lines[-2]
    assert lines[0].startswith(f"[alloc] amc {NET}: 6 episodes (the first 3 random), {len(rows)} groups, FLOPs target "
                               "x0.580 ")
    assert planned["kind"] == "amc" and planned["budget"] == "flops" and planned["target"] == pytest.approx(0.58)
    assert planned["amc"] == {"pick": pick, "episodes": 6, "warmup": 2, "score": rec["score"], "proxy": "fs2",
                              "seed": 0}
    assert set(int(r) for r in planned["rows"]) == set(rows)
    assert [e for e, _k in records] == ["alloc_amc", "alloc_plan"]
    assert rec["updates"] == 3 * len(rows)                                 # episodes 3, 4 and 5, one per stored step


@pytest.mark.parametrize("flop", [0.45, 0.20])
def test_a_short_search_returns_a_plan_that_lands_within_0p011_of_the_flops_target(monkeypatch, flop):
    _amc(monkeypatch, episodes=6, warmup=3, proxy="bn2", flop=flop)
    model = _model("r20_chenyaofo")                                        # 16 / 32 / 64 channels: fine enough steps
    origin = {key: value.clone() for key, value in model.state_dict().items()}
    entry, lines, records = _run(monkeypatch, _env(model=model))
    rec, planned = _record(records, "alloc_amc"), _record(records, "alloc_plan")
    target = flop - 0.02                                                   # the walk target less the undershoot
    assert (rec["episodes"], rec["warmup"], rec["proxy"], rec["budget"]) == (6, 3, "bn2", "flops")
    assert rec["target"] == pytest.approx(target) and planned["target"] == pytest.approx(target)
    assert abs(planned["kept"] - target) <= 0.011                          # the usual tolerance
    plan = _plan_of(model)
    flops0 = utils.calc_flops(model, SHAPE)
    keeps = {int(row): keep for row, keep in rec["landed"]["keeps"].items()}
    cut = alloc_walk.cut_to(model, plan, keeps, SHAPE)                     # the landed plan, cut for real
    kept = utils.calc_flops(cut, SHAPE) / flops0
    assert kept == pytest.approx(planned["kept"], abs=1e-12) and abs(kept - target) <= 0.011
    assert alloc_walk.group_widths(cut, list(entry["widths"])) == entry["widths"]
    assert all(keep >= 0.2 - 1e-12 for keep in keeps.values())             # AMC's lbound holds through the landing
    # every episode's plan met the budget on its own; the best one is the plan that was landed
    assert all(f <= target + 1e-9 for f in rec["flops"]) and rec["kept"] == rec["flops"][rec["pick"]]
    fm = plan_agent.FlopModel(model, plan, SHAPE)
    assert fm.kept(_ints(rec["widths"])) == pytest.approx(rec["kept"])
    assert all(rec["keeps"][str(row)] == _ints(rec["widths"])[row] / fm.widths0[row] for row in fm.rows)
    assert rec["hyper"] == {**AMC_DEFAULTS, "episodes": 6, "warmup": 3} and rec["amc_commit"] == amc_search.AMC_COMMIT
    assert rec["groups"] == len(fm.rows) and rec["state_dim"] == 11 and rec["seconds"] >= rec["score_seconds"] >= 0
    assert rec["updates"] == 2 * len(fm.rows) and rec["device"] == "cpu"
    for key, value in model.state_dict().items():
        assert torch.equal(value, origin[key]), key                        # the origin is only read


def test_a_search_at_amcs_own_length_runs_end_to_end_and_improves_a_smooth_reward(monkeypatch):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "amc")                        # no episode knob: AMC's 800 and 100
    monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "flops")
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "flop:0.45")
    monkeypatch.setenv("SPECTRA_ALLOC_AMC_PROXY", "cut")
    model = _model("r20_chenyaofo")
    plan = _plan_of(model)
    fm = plan_agent.FlopModel(model, plan, SHAPE)
    _features, flops = amc_search.group_features(model, ModelWithRows(model), plan, SHAPE)
    share = {row: f / sum(flops) for row, f in zip(fm.rows, flops)}

    def smooth(self, widths, proxy, batches):                             # more of a costly group helps, less and less
        return max(0.0, 0.9 + 0.2 * sum(share[row] * math.log(widths[row] / fm.widths0[row]) for row in fm.rows))

    monkeypatch.setattr(plan_proxies.PlanScorer, "score", smooth)
    entry, lines, records = _run(monkeypatch, _env(model=model))
    rec, planned = _record(records, "alloc_amc"), _record(records, "alloc_plan")
    groups = len(fm.rows)
    assert rec["episodes"] == 800 and rec["warmup"] == 100 and rec["hyper"] == AMC_DEFAULTS
    assert rec["updates"] == 699 * groups                                  # episodes 101 to 799, one per stored step
    phases = [EPISODE_LINE.match(line).group(3) for line in lines if line.startswith(f"[alloc] amc {NET} ep=")]
    assert phases == ["random"] * 101 + ["ddpg"] * 699                    # episode <= warm-up acts at random
    assert len(rec["rewards"]) == len(rec["flops"]) == 800 and all(f <= 0.43 + 1e-9 for f in rec["flops"])
    assert rec["score"] == max(rec["rewards"]) and rec["pick"] == rec["rewards"].index(rec["score"])
    early, late = sum(rec["rewards"][:101]) / 101.0, sum(rec["rewards"][-100:]) / 100.0
    assert late > early + 0.02                                             # the policy beats its own random warm-up
    assert rec["score"] >= late and abs(planned["kept"] - 0.43) <= 0.011
    assert lines[0].startswith(f"[alloc] amc {NET}: 800 episodes (the first 101 random), {groups} groups, ")


def test_the_plan_comes_back_in_the_format_every_kind_hands_the_walk(monkeypatch):
    _amc(monkeypatch, episodes=4, warmup=1, proxy="cut")
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    env = _env()
    widths, info = amc_search.amc_plan(env, 0.58, budget="flops")
    fm = plan_agent.FlopModel(env.current_model, _plan_of(env.current_model), SHAPE)
    assert set(info) == {"kind", "alpha", "target", "kept", "budget", "keeps", "origin_widths", "sens", "held",
                         "weights", "amc"}
    assert (info["kind"], info["alpha"], info["target"], info["budget"], info["held"]) == ("amc", 0.0, 0.58, "flops", 0)
    assert list(widths) == fm.rows == list(info["keeps"]) == list(info["weights"])
    assert info["origin_widths"] == fm.widths0
    assert info["sens"] == info["weights"] and all(0.0 < keep <= 1.0 for keep in info["weights"].values())
    assert all(0.2 <= info["keeps"][row] <= 1.0 and 1 <= widths[row] <= fm.widths0[row] for row in fm.rows)
    assert info["kept"] == pytest.approx(fm.kept(widths))                  # the landed cut's own FLOPs
    summary = info["amc"]
    assert set(summary) == {"pick", "episodes", "warmup", "score", "proxy", "seed", "est_kept", "scored", "moved",
                            "seconds", "rewards"}
    assert (summary["episodes"], summary["warmup"], summary["proxy"], summary["seed"]) == (4, 1, "cut", 0)
    assert summary["score"] == summary["rewards"][summary["pick"]] == max(summary["rewards"])
    best = {row: round(info["weights"][row] * fm.widths0[row]) for row in fm.rows}   # the best episode's own widths
    assert summary["est_kept"] == pytest.approx(fm.kept(best)) and summary["est_kept"] <= 0.58 + 1e-9
    assert summary["moved"] == sum(widths[row] != best[row] for row in fm.rows)


def test_the_search_puts_every_global_rng_state_and_the_origins_flags_back(monkeypatch):
    _amc(monkeypatch, episodes=5, warmup=2, proxy="fs2")
    env = _env()
    env.current_model.layer2.train()                                       # mixed flags must come back exactly
    modes = {name: module.training for name, module in env.current_model.named_modules()}
    assert len(set(modes.values())) == 2
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    before = _rng_state()
    a, _lines, ra = _run(monkeypatch, env)
    assert _rng_state() == before
    assert {name: module.training for name, module in env.current_model.named_modules()} == modes
    after = (torch.rand(3), random.random(), np.random.rand())
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    assert torch.equal(torch.rand(3), after[0]) and random.random() == after[1] and np.random.rand() == after[2]
    env = _env()
    env.current_model.layer2.train()
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    b, _lines, rb = _run(monkeypatch, env)                                 # nor does it depend on where they stood
    assert a["widths"] == b["widths"]
    assert _timeless(_record(ra, "alloc_amc")) == _timeless(_record(rb, "alloc_amc"))


def test_only_the_flops_budget_is_accepted(monkeypatch):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "amc")                        # the default budget: params
    monkeypatch.setattr(plan_proxies, "PlanScorer", lambda *a, **k: pytest.fail("refused runs build no scorer"))
    lines, records = [], []
    with pytest.raises(ValueError, match="SPECTRA_ALLOC_BUDGET=flops"):
        _run(monkeypatch, lines=lines, records=records)
    assert lines == [] and records == []                                   # refused before any work or line
    with pytest.raises(ValueError, match="SPECTRA_ALLOC_BUDGET=flops"):
        amc_search.amc_plan(_env(), 0.58, budget="params")
    monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "flops")
    monkeypatch.setenv("SPECTRA_ALLOC_AMC_PROXY", "ft12_4")                # a bad proxy too
    with pytest.raises(ValueError):
        _run(monkeypatch, lines=lines, records=records)
    assert lines == [] and records == []


def test_the_width_floor_holds_in_every_episode_and_through_the_landing(monkeypatch):
    _amc(monkeypatch, episodes=5, warmup=2, proxy="cut", SPECTRA_PLAN_MIN_WIDTH="3", SPECTRA_ALLOC_MIN_KEEP="0.3")
    seen = []
    monkeypatch.setattr(plan_proxies.PlanScorer, "score",
                        lambda self, widths, proxy, batches: seen.append(dict(widths)) or 0.5)
    entry, lines, records = _run(monkeypatch)
    rec, planned = _record(records, "alloc_amc"), _record(records, "alloc_plan")
    origin = _plan_of(_model())
    widths0 = {row: group.width for group, row in origin}
    for widths in seen + [entry["widths"], _ints(rec["widths"])]:
        assert all(widths[row] >= min(3, widths0[row]) for row in widths0)             # SPECTRA_PLAN_MIN_WIDTH
        assert all(widths[row] >= round(0.3 * widths0[row]) for row in widths0)        # SPECTRA_ALLOC_MIN_KEEP
    assert rec["min_width"] == 3 and rec["k_min"] == 0.3 and planned["min_width"] == 3
    assert rec["pick"] == 0 and lines[-1].endswith("; group width floor 3")           # all tied: the first episode


def test_under_the_grid_round_the_plan_is_rounded_like_any_plan_by_its_keeps(monkeypatch):
    _amc(monkeypatch, episodes=5, warmup=2, proxy="cut", SPECTRA_ALLOC_GRID_ROUND="1", SPECTRA_GROUP_ONCE_PER_PASS="1")
    env = _env()
    entry, lines, records = _run(monkeypatch, env)
    fm = plan_agent.FlopModel(env.current_model, _plan_of(env.current_model), SHAPE)
    assert "grid" in entry and entry["grid"]["targets"] == entry["widths"]
    assert 0.6 - 0.05 <= fm.kept(entry["widths"]) <= 0.6 + 1e-9            # at or below the walk target
    grid = _record(records, "alloc_grid_round")
    assert grid["kind"] == "amc" and grid["walk_target"] == 0.6 and grid["budget"] == "flops"
    assert [e for e, _k in records] == ["alloc_amc", "alloc_plan", "alloc_grid_round"]
    assert lines[-1].startswith("[alloc] grid round: ") and lines[-2].startswith(f"[alloc] {NET}: amc best episode ")
    rows = [1, 2, 3]
    weights = {1: 0.4, 2: 0.2, 3: 1.0}
    assert alloc_walk.grid_priority({"kind": "amc", "weights": weights, "sens": weights}, rows) == weights
    assert alloc_walk.grid_priority({"kind": "uniform", "sens": {r: 1.0 for r in rows}, "alpha": 0.5}, rows) is None
    z = {1: 0.4, 2: -1.0, 3: 2.0}
    assert alloc_walk.grid_priority({"kind": "randbest", "sens": z}, rows) == z   # the other kinds rank as before


def test_the_walk_follows_the_amc_plan_to_the_flops_target(monkeypatch):
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    monkeypatch.setenv("SPECTRA_FIXED_TARGET_METRIC", "flop")
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
    _amc(monkeypatch, episodes=5, warmup=2, proxy="cut", SPECTRA_ALLOC_GRID_ROUND="1")
    monkeypatch.delenv("SPECTRA_EVAL_SIZE_MATCH")                          # the env's fixed FLOPs target is the walk's
    model = _model()
    env, _seen = _target_env(model, target=0.6, accs=[0.9] * 400, passes=6)
    del env._register_group_lock                                          # the env's own group-once lock
    env.conf.compression_rates_dict = RATES
    env.original_flops = utils.calc_flops(model, SHAPE)
    env.test_loader = _NeverRead()
    lines, records = [], []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
    done, steps = False, 0
    while not done and steps < 300:
        legal = env.legal_action_mask(device="cpu")
        pick = int(alloc_walk.action(env, legal, RATES, "cpu").item())
        assert bool(legal[pick])
        _, _, done = env.step(RATES[pick])
        steps += 1
    kept = env.flops_ratio()
    assert done and 0.6 - 0.03 <= kept <= 0.6 + 1e-9
    state = env._alloc_walk[env.selected_net_path]
    assert not state["fallback"]
    rec = _record(records, "alloc_amc")                                    # the search ran once for the net
    assert rec["episodes"] == 5 and rec["run_seed"] == 0 and rec["network"] == env.selected_net_path
    assert rec["target"] == pytest.approx(0.58)
    targets = state["grid"]["targets"]
    now = alloc_walk.group_widths(env.current_model, list(targets))
    assert sum(now[row] < targets[row] for row in targets) <= 1            # only the landing cut may leave the plan
    assert sum(line.startswith("[alloc] tiny_r20w4.pt: amc best episode ") for line in lines) == 1


# ------------------------------------------------------------------ default off


def test_another_kind_never_enters_the_search(monkeypatch):
    monkeypatch.setattr(amc_search, "amc_plan", lambda *a, **k: pytest.fail("another kind entered the amc search"))
    monkeypatch.setattr(amc_search, "DDPG", lambda *a, **k: pytest.fail("another kind built the amc agent"))
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "uniform")
    monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: [])
    entry, lines, records = _run(monkeypatch)
    assert entry["widths"] and lines == [lines[0]]
    assert lines[0].startswith(f"[alloc] {NET}: uniform alpha=0.5 plan keeps x")
    assert [e for e, _k in records] == ["alloc_plan"] and "amc" not in _record(records, "alloc_plan")
    for budget in ("params", "flops"):                                    # a rule under the FLOPs budget, as before
        monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", budget)
        monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "flop:0.6")
        for name in ("global_mag", "lamp"):
            monkeypatch.setenv("SPECTRA_ALLOC_KIND", name)
            entry, lines, records = _run(monkeypatch)
            assert [e for e, _k in records] == ["alloc_plan"] and _record(records, "alloc_plan")["kind"] == name
            assert "amc" not in _record(records, "alloc_plan") and not any("amc" in line for line in lines)
    monkeypatch.delenv("SPECTRA_ALLOC_BUDGET")
    monkeypatch.delenv("SPECTRA_EVAL_SIZE_MATCH")
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "randbest")                   # the other per-network search
    monkeypatch.setenv("SPECTRA_ALLOC_RANDBEST_N", "3")
    monkeypatch.setenv("SPECTRA_ALLOC_RANDBEST_PROXY", "cut")
    entry, lines, records = _run(monkeypatch)
    assert [e for e, _k in records] == ["alloc_randbest", "alloc_plan"]
    assert "amc" not in _record(records, "alloc_plan") and lines[-1].startswith(f"[alloc] {NET}: randbest best draw ")


def test_the_module_is_imported_only_under_the_kind():
    code = ("import sys; sys.path.insert(0, {root!r}); from src import alloc_walk; "
            "assert alloc_walk.kind() == 'sens'; assert 'src.amc_search' not in sys.modules, 'imported eagerly'; "
            "print('lazy')").format(root=str(ROOT))
    env = {key: value for key, value in os.environ.items() if not key.startswith("SPECTRA_ALLOC_")}
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, universal_newlines=True, timeout=600)
    assert out.returncode == 0 and "lazy" in out.stdout, out.stderr[-2000:]
