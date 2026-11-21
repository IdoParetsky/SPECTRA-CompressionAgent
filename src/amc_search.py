"""
AMC's per-network channel-allocation search as an allocation kind of the eval walk (``SPECTRA_ALLOC_KIND=amc``,
default off; queue row AMC, part C, registered 21 Nov 2026 08:48 IST, amended 09:34 IST). A comparator, not SPECTRA's
method: this module is imported only under that kind.

AMC (He, Lin, Liu, Wang, Li and Han, "AMC: AutoML for Model Compression and Acceleration on Mobile Devices", ECCV
2018) searches one network at one resource target. A DDPG agent visits the layers in order and gives each one its
preserve ratio; a budget rule (the "action wall") bounds every action so that the target can still be met; the accuracy
of the pruned network is the reward of the whole episode; the best strategy over all episodes is kept and fine-tuned.
This module ports that search from AMC's released code, https://github.com/mit-han-lab/amc at commit ``AMC_COMMIT``
(``amc_search.py``, ``env/channel_pruning_env.py``, ``env/rewards.py``, ``lib/agent.py``, ``lib/memory.py``), onto
SPECTRA's coupled channel groups and FLOPs cost model. ``alloc_walk._state`` calls ``amc_plan`` once per network, on the
origin, where it calls ``randbest_plan`` for that kind; the walk, the cut and the recovery then run as for every kind.

What one search does (``amc_plan``):

1. The decisions are the rows the rules plan over (``group_sensitivity.group_plan``), in that order: one keep per
   coupled group. The budget is priced on ``plan_agent.FlopModel``, so only ``SPECTRA_ALLOC_BUDGET=flops`` is accepted.
2. ``GroupEnv`` is AMC's channel-pruning env on those groups: the state of a decision, the action wall, the step.
3. ``DDPG`` is AMC's agent: actor, critic, their target copies, the replay memory, random warm-up actions, then the
   actor's action under truncated-normal noise whose std decays, and the moving-average reward baseline.
4. ``search`` is AMC's loop. An episode's plan is scored by ``plan_proxies.PlanScorer`` on the val half with the proxy
   ``SPECTRA_ALLOC_AMC_PROXY`` (``bn8``: accuracy after batch-norm recalibration, the queue row's amendment 1), and that
   accuracy is the reward of every step of the episode. The TEST half is never read.
5. The best episode (the first, on a tie) hands its keeps to ``alloc_walk._land``, the bisection every rule lands with,
   so the plan the walk follows keeps the target FLOPs to that bisection's tolerance.

Hyper-parameters. Each is the released code's default, with its file and line at ``AMC_COMMIT`` (``DEFAULTS``):

* training episodes 800: ``amc_search.py`` L65 (``--train_episode``). The loop (L108) counts every episode, the
  warm-up ones included.
* warm-up episodes 100: ``amc_search.py`` L48 (``--warmup``). Episodes 0 to 100 act at random (L115 tests
  ``episode <= warmup``, so 101 episodes); the networks are updated from episode 101 on (L154, ``episode > warmup``).
* lbound 0.2 and rbound 1.0: ``amc_search.py`` L29 and L30. In ``env/channel_pruning_env.py`` every ratio starts at
  lbound (L353), the wall clips an action to [0, 1] (L271) and never returns less than lbound (L296); rbound is stored
  (L37) and not read again.
* replay size 100 per decision: ``amc_search.py`` L52 (``--rmsize``), multiplied by the number of layers at L238.
* batch size 64: ``amc_search.py`` L51 (``--bsize``).
* tau 0.01, the soft update of the target networks: ``amc_search.py`` L54; ``lib/agent.py`` L219-223.
* discount 1.0: ``amc_search.py`` L50; ``lib/agent.py`` L132-133.
* initial delta 0.5 and its decay 0.95 per episode: ``amc_search.py`` L56 and L58. ``lib/agent.py`` L185 sets
  delta = 0.5 * 0.95 ** (episode - warmup), the std of the truncated normal drawn at L187 (L229-231).
* actor learning rate 1e-4 and critic learning rate 1e-3, Adam: ``amc_search.py`` L47 (``--lr_a``) and L46
  (``--lr_c``); ``lib/agent.py`` L70 and L74.
* hidden sizes 300 and 300: ``amc_search.py`` L44 and L45, passed at ``lib/agent.py`` L63-65 (the 400 in the class
  signatures at L19 and L38 is never used).
* reward baseline: a moving average of the batch's mean reward with alpha 0.5, started at the first batch's mean and
  subtracted from the batch: ``lib/agent.py`` L107-108 and L116-121.
* the agent's own action range [0, 1], for the random warm-up actions and the truncated normal: ``lib/agent.py``
  L89-90 (lbound belongs to the env).
* window length 1: ``amc_search.py`` L53. Critic loss: mean squared error, ``lib/agent.py`` L14 and L140. Weight
  initialisation: PyTorch's default for ``nn.Linear`` (``--init_w``, ``amc_search.py`` L64, is not passed on:
  ``lib/agent.py`` L66 is commented out).
* reward: accuracy / 100, ``env/rewards.py`` L9-10 (``acc_reward``, the default at ``amc_search.py`` L31).

AMC decides layer by layer; SPECTRA cuts coupled groups. The mapping:

* Decision. AMC's action t is the preserve ratio of the input channels of prunable layer t
  (``env/channel_pruning_env.py`` L93-97); the first layer's input, the image, is forced to 1 (L355) and in
  MobileNetV2 the layers of one residual stream copy the first one's action (L89-106). Here an action is the keep of
  one coupled group (the channels its producers write and its consumers read, cut together), so every step is a real
  decision and a residual stream is decided once.
* State. The eleven features of the paper's equation 1, per group: its index t in plan order, n (the group's width,
  the output channels of its producers), c, h, w, stride and k (input channels, input height and width, stride and
  kernel of the group's first producer in depth order: the layer at the group's first walk row; a Linear has stride 0
  and kernel 1 as in AMC, L524-525, and h = w = 1), FLOPs[t] (the MACs of every layer that writes the group's
  channels: its producers and depthwise convolutions, ``utils.per_module_macs``), reduced (the FLOPs the decisions so
  far removed, over the origin's), rest (FLOPs[j] summed over the groups after t, over the origin's) and the previous
  group's realised keep (1 at the first step, L166). The first eight are scaled to [0, 1] by their minimum and maximum
  over the network's groups (L538-542; a constant column is left as it is, as there). The released code's own
  embedding (L504-532) has ten features: the layer type and weight size stand where h, w and FLOPs[t] stand here, its
  ``reduced`` counts the undecided layers at lbound (L305-317) and its first ``rest`` is read from weight sizes
  (L170). The state here follows the paper's definitions.
* Budget rule. AMC's wall (L267-298) assumes FLOPs linear in the current ratio, takes the layers before as decided
  and the layers after at lbound, and returns min(action, (target - other) / this), then at least lbound. ``GroupEnv``
  applies the same rule on integer widths through ``FlopModel.cost``: the widest width of the current group at which
  the cost, with the groups before as decided and the groups after at their lowest width, is at most target * total.
  Every decided width therefore leaves the target reachable, and a finished plan never keeps more than the target.
* Rounding and floors. AMC rounds a layer's channels up to a multiple of 8 after the wall (L198-201). Here a keep
  becomes the width the landing's own cut leaves (``pruning.target_width``), before the wall. No keep goes below
  max(lbound, ``SPECTRA_ALLOC_MIN_KEEP``, ``SPECTRA_PLAN_MIN_WIDTH`` / the group's width), the floors every rule
  plans with, and the same floors go to the landing.
* Feasibility. AMC asserts preserve_ratio > lbound (L51), which would refuse a FLOPs target of 0.18 that every group
  at keep 0.2 meets several times over. Here the check is the true one: a target below what every group at its lowest
  width keeps is refused with a ValueError.
* Replay. ``ReplayMemory`` is ``lib/memory.py``'s ``SequentialMemory`` at window length 1, with one difference: AMC's
  sampler draws again when a transition's predecessor ended an episode (L146-152), which in AMC's env is the forced
  first step. Here the first step is a real decision, so no transition is skipped.

What is SPECTRA's, as for every allocation kind, and not AMC's:

* The reward's network. AMC rewards the network after its least-squares reconstruction of each pruned layer's weights
  (L206-247; ``--n_calibration_batches``, ``--n_points_per_layer``), by top-5 accuracy by default (``amc_search.py``
  L32), on images held out of the train split (L319-328). Here the reward is ``PlanScorer``'s read: the origin cut by
  SPECTRA's L1 ranking to the plan's widths, ``SPECTRA_ALLOC_AMC_PROXY`` applied, top-1 accuracy on the val half as a
  fraction (AMC's scale, accuracy / 100). A plan seen before is not scored again: ``PlanScorer`` runs under
  ``SPECTRA_EVAL_PROXY_SEED`` every time, so the score is a function of the widths alone, as AMC's validation is.
* The landing. AMC exports its best strategy as it is. Here the best episode's keeps are the weights of
  ``alloc_walk._land``, so the plan keeps target FLOPs within that bisection's tolerance and the walk, the cut and the
  recovery are those of every kind. ``SPECTRA_ALLOC_GRID_ROUND`` then rounds the plan in ``alloc_walk._state`` as it
  rounds any plan, ranking the groups by their keeps (``alloc_walk.grid_priority``); episodes are scored before it, as
  ``randbest`` and ``fallback`` score theirs.
* Random numbers. AMC draws from the global NumPy, Python and torch generators. Here the agent has private ones
  (``random.Random`` for the actions, the noise and the replay batches, a ``torch.Generator`` for the initial
  weights), seeded by ``alloc_walk.randbest_seed`` of ``SPECTRA_ALLOC_SAMPLE_SEED`` and the run's seed; the search and
  the landing run inside ``plan_proxies.seeded``, so the Python, NumPy, torch CPU and CUDA states are put back and the
  walk and the final fine-tune draw from the stream the run had before the search. AMC also sets
  ``cudnn.deterministic`` (``amc_search.py`` L10); here the process's own setting stands
  (``SPECTRA_EVAL_FT_DETERMINISTIC=1`` is the run-wide switch).

Knobs: ``SPECTRA_ALLOC_AMC_EPISODES`` and ``SPECTRA_ALLOC_AMC_WARMUP`` (unset: AMC's 800 and 100; for smokes),
``SPECTRA_ALLOC_AMC_PROXY`` (``bn8``; one of the plan trainer's: cut, bn<N>, fs<S>, ft<E>). One line per episode, a
summary line and one ``alloc_amc`` record carry the search.
"""

from __future__ import annotations

import math
import os
import random
import statistics
import time

import torch
from torch import nn

import src.pruning as pruning
import src.utils as utils

AMC_COMMIT = "65363c84649fba9623d94dda8ed8baff4dc01175"  # github.com/mit-han-lab/amc, the commit the lines above cite
DEFAULTS = {"episodes": 800, "warmup": 100, "lbound": 0.2, "rbound": 1.0, "rmsize": 100, "bsize": 64, "tau": 0.01,
            "discount": 1.0, "init_delta": 0.5, "delta_decay": 0.95, "lr_a": 1e-4, "lr_c": 1e-3, "hidden1": 300,
            "hidden2": 300, "moving_alpha": 0.5, "window_length": 1}
AGENT_LOW, AGENT_HIGH = 0.0, 1.0  # lib/agent.py L89-90: the agent acts in [0, 1]; lbound and the wall are the env's
DEFAULT_PROXY = "bn8"  # queue row AMC, amendment 1 (21 Nov 2026 09:34 IST)
STATIC_FEATURES = ("t", "n", "c", "h", "w", "stride", "k", "flops")  # scaled over the network's groups
DYNAMIC_FEATURES = ("reduced", "rest", "prev_keep")  # fractions already, written at every step
STATE_DIM = len(STATIC_FEATURES) + len(DYNAMIC_FEATURES)
LAND_ITERS, LAND_TOL = 16, 0.003  # alloc_walk.plan_targets' own defaults: the landing every rule runs
_NORMAL = statistics.NormalDist()


# ------------------------------------------------------------------ knobs


def _int_knob(name: str, default: int, low: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: expected an integer of at least {low}") from None
    if value < low:
        raise ValueError(f"{name}={raw!r}: expected an integer of at least {low}")
    return value


def episodes() -> int:
    """``SPECTRA_ALLOC_AMC_EPISODES`` (unset: AMC's 800): how many episodes one search runs, the warm-up ones
    included, as AMC counts them."""
    return _int_knob("SPECTRA_ALLOC_AMC_EPISODES", DEFAULTS["episodes"], 1)


def warmup() -> int:
    """``SPECTRA_ALLOC_AMC_WARMUP`` (unset: AMC's 100): episodes 0 to this one act at random and only fill the replay
    memory (AMC tests ``episode <= warmup``, so W means W + 1 random episodes); the noise's decay counts from it."""
    return _int_knob("SPECTRA_ALLOC_AMC_WARMUP", DEFAULTS["warmup"], 0)


def proxy() -> str:
    """``SPECTRA_ALLOC_AMC_PROXY`` (``bn8``): the proxy an episode's plan is scored with, one of the plan trainer's
    (``plan_agent.parse_proxy``: cut, bn<N>, fs<S>, ft<E>)."""
    name = os.environ.get("SPECTRA_ALLOC_AMC_PROXY", "").strip().lower() or DEFAULT_PROXY
    from src import plan_agent
    plan_agent.parse_proxy(name)
    return name


def hyper_parameters() -> dict:
    """AMC's defaults (``DEFAULTS``) with the two smoke knobs applied."""
    return {**DEFAULTS, "episodes": episodes(), "warmup": warmup()}


# ------------------------------------------------------------------ the agent (lib/agent.py, lib/memory.py)


def truncated_normal(rng, mu: float, sigma: float, lower: float = AGENT_LOW, upper: float = AGENT_HIGH) -> float:
    """One draw of N(mu, sigma^2) truncated to [lower, upper], from the private generator ``rng``
    (``random.Random``). AMC calls ``scipy.stats.truncnorm.rvs`` and clips the result (``lib/agent.py`` L187-188,
    L229-231); the SciPy 1.1 its README lists draws it by inverting the truncated CDF at one uniform. This is the same
    inversion, so the distribution is AMC's (``tests/test_amc_search.py`` checks it against SciPy's) and neither SciPy
    nor a global generator is read. ``mu`` is the actor's sigmoid output, so it lies in [lower, upper]. A sigma that
    has underflowed to 0 returns ``mu`` (with AMC's 800 episodes the last std is 0.5 * 0.95 ** 699, about 1e-16, and
    no draw reaches this)."""
    mu = min(max(float(mu), lower), upper)
    u = rng.random()
    if not sigma > 0.0:
        return mu
    low, high = _NORMAL.cdf((lower - mu) / sigma), _NORMAL.cdf((upper - mu) / sigma)
    p = min(max(low + u * (high - low), 1e-300), 1.0 - 1e-16)
    return min(max(mu + sigma * _NORMAL.inv_cdf(p), lower), upper)


class ReplayMemory:
    """``lib/memory.py``'s ``SequentialMemory`` at window length 1 (L120-219) on its ring buffer (L39-64): entry i holds
    (observation, action, reward, terminal), and a sampled transition j reads its next observation from entry j + 1,
    so the newest entry is never drawn. A batch is drawn without replacement when the memory holds enough transitions
    and with replacement otherwise (L19-36), from the private generator ``rng``. A terminal transition's next
    observation belongs to the following episode; its mask is 0, so nothing reads it (L199). Unlike AMC's sampler, no
    transition is skipped for following a terminal one (L146-152; the module docstring says why)."""

    def __init__(self, limit: int, rng):
        self.limit, self.rng = int(limit), rng
        self.data, self.start, self.length = [None] * self.limit, 0, 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if index < 0 or index >= self.length:
            raise KeyError(index)
        return self.data[(self.start + index) % self.limit]

    def append(self, observation, action: float, reward: float, terminal: bool) -> None:
        if self.length < self.limit:
            self.length += 1
        else:  # full: the oldest entry goes
            self.start = (self.start + 1) % self.limit
        self.data[(self.start + self.length - 1) % self.limit] = (tuple(observation), float(action), float(reward),
                                                                  bool(terminal))

    def sample(self, batch_size: int):
        """``(state0, action, reward, state1, alive)``, five lists of ``batch_size``; ``alive`` is 0 for a terminal
        transition and 1 otherwise."""
        last = self.length - 1  # transitions 0 .. last - 1 have a successor entry
        if last < 1:
            raise RuntimeError("the replay memory needs two entries before a batch is drawn")
        if last >= batch_size:
            picks = self.rng.sample(range(last), int(batch_size))
        else:
            picks = [self.rng.randrange(last) for _ in range(int(batch_size))]
        state0, action, reward, state1, alive = [], [], [], [], []
        for j in picks:
            s, a, r, terminal = self[j]
            state0.append(s)
            action.append(a)
            reward.append(r)
            state1.append(self[j + 1][0])
            alive.append(0.0 if terminal else 1.0)
        return state0, action, reward, state1, alive


class Actor(nn.Module):
    """``lib/agent.py`` L18-34: state -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> sigmoid, one action in (0, 1)."""

    def __init__(self, nb_states: int, nb_actions: int, hidden1: int, hidden2: int):
        super().__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)

    def forward(self, x):
        return torch.sigmoid(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))))


class Critic(nn.Module):
    """``lib/agent.py`` L37-53: Q(s, a) = fc3(ReLU(fc2(ReLU(fc11(s) + fc12(a)))))."""

    def __init__(self, nb_states: int, nb_actions: int, hidden1: int, hidden2: int):
        super().__init__()
        self.fc11 = nn.Linear(nb_states, hidden1)
        self.fc12 = nn.Linear(nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x, a):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc11(x) + self.fc12(a)))))


def _init_linears(module: nn.Module, generator) -> None:
    """PyTorch's default ``nn.Linear`` initialisation (weight and bias uniform within 1 / sqrt(fan_in), what
    ``kaiming_uniform_(a=sqrt(5))`` and the bias rule come to), drawn from the private ``generator`` in module order."""
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                bound = 1.0 / math.sqrt(layer.in_features)
                for param in (layer.weight, layer.bias):
                    draw = torch.rand(param.shape, generator=generator, dtype=torch.float64)
                    param.copy_((2.0 * draw - 1.0) * bound)


class DDPG:
    """``lib/agent.py``'s ``DDPG`` (L56-231) with ``hp`` (``hyper_parameters``): the actor and the critic, their target
    copies (equal at the start, L76-77), Adam on each, the replay memory of ``rmsize`` entries per decision, and the
    moving-average reward baseline. Its random draws come from generators of its own, seeded by ``seed``: a
    ``random.Random`` for the actions, the noise and the replay batches, a ``torch.Generator`` for the initial weights.
    Building the layers draws their default initialisation from the global torch generator before it is overwritten;
    that state is put back, so the global generators are left as they were."""

    def __init__(self, nb_states: int, n_decisions: int, hp: dict, seed: int, device="cpu"):
        self.hp, self.device = dict(hp), torch.device(device)
        self.rng = random.Random(int(seed))
        generator = torch.Generator().manual_seed(int(seed))
        hidden = (int(hp["hidden1"]), int(hp["hidden2"]))
        global_state = torch.get_rng_state()
        self.actor, self.actor_target = Actor(nb_states, 1, *hidden), Actor(nb_states, 1, *hidden)
        self.critic, self.critic_target = Critic(nb_states, 1, *hidden), Critic(nb_states, 1, *hidden)
        torch.set_rng_state(global_state)
        _init_linears(self.actor, generator)
        _init_linears(self.critic, generator)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        for net in (self.actor, self.actor_target, self.critic, self.critic_target):
            net.to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(hp["lr_a"]))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(hp["lr_c"]))
        self.memory = ReplayMemory(int(hp["rmsize"]) * int(n_decisions), self.rng)
        self.moving_average = None
        self.updates = 0

    def _tensor(self, values):
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def delta(self, episode: int) -> float:
        """The std of the exploration noise at ``episode`` (L185): init_delta * delta_decay ** (episode - warmup)."""
        return float(self.hp["init_delta"]) * float(self.hp["delta_decay"]) ** (int(episode) - int(self.hp["warmup"]))

    def random_action(self) -> float:
        """A warm-up action, uniform in [0, 1) (L177-180)."""
        return self.rng.uniform(AGENT_LOW, AGENT_HIGH)

    def select_action(self, state, episode: int) -> float:
        """The actor's action for ``state`` under truncated-normal noise of std ``delta(episode)`` (L182-191)."""
        with torch.no_grad():
            mu = float(self.actor(self._tensor([list(state)])).item())
        return truncated_normal(self.rng, mu, self.delta(episode), AGENT_LOW, AGENT_HIGH)

    def observe(self, reward: float, state, action: float, terminal: bool) -> None:
        """Store one step (L172-175): the state, the agent's own action, the episode's reward, the terminal flag."""
        self.memory.append(state, action, reward, terminal)

    def update_policy(self) -> None:
        """One step of the critic and of the actor on one replay batch, then the soft update of both target networks
        (L110-158), in AMC's order. Gradients are enabled here whatever the caller's mode is."""
        state0, action, reward, state1, alive = self.memory.sample(int(self.hp["bsize"]))
        mean = sum(reward) / len(reward)  # L116-121: the moving-average baseline, subtracted from the batch
        if self.moving_average is None:
            self.moving_average = mean
        else:
            self.moving_average += float(self.hp["moving_alpha"]) * (mean - self.moving_average)
        state, nxt = self._tensor(state0), self._tensor(state1)
        act, mask = self._tensor(action).unsqueeze(1), self._tensor(alive).unsqueeze(1)
        rew = self._tensor([r - self.moving_average for r in reward]).unsqueeze(1)
        with torch.enable_grad():
            with torch.no_grad():  # L126-133: the target Q, zero beyond a terminal step
                target_q = rew + float(self.hp["discount"]) * mask * self.critic_target(nxt, self.actor_target(nxt))
            self.critic.zero_grad()  # L136-142
            value_loss = nn.functional.mse_loss(self.critic(state, act), target_q)
            value_loss.backward()
            self.critic_optim.step()
            self.actor.zero_grad()  # L145-154
            policy_loss = -self.critic(state, self.actor(state)).mean()
            policy_loss.backward()
            self.actor_optim.step()
        tau = float(self.hp["tau"])
        with torch.no_grad():  # L157-158, L219-223
            for target, source in ((self.actor_target, self.actor), (self.critic_target, self.critic)):
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.mul_(1.0 - tau).add_(param, alpha=tau)
        self.updates += 1


# ------------------------------------------------------------------ the env (env/channel_pruning_env.py)


def realised_width(w0: int, keep: float) -> int:
    """The width the landing's cut leaves a ``w0``-wide group at ``keep``: ``alloc_walk.cut_to`` leaves a group whole
    at keep 1 and otherwise cuts it to ``pruning.target_width`` (the nearest width, at least one channel removed)."""
    if float(keep) >= 1.0 - 1e-9:
        return int(w0)
    return int(pruning.target_width(int(w0), float(keep)))


def keep_floors(widths0: dict, lbound: float, k_min: float, min_width: int = 1) -> dict:
    """``{row: lowest keep}``: AMC's lbound, and no lower than the floors every rule plans with
    (``alloc_walk.plan_targets``): ``k_min`` (``SPECTRA_ALLOC_MIN_KEEP``) and, with ``min_width`` above 1
    (``SPECTRA_PLAN_MIN_WIDTH``), ``min_width`` over the group's origin width."""
    out = {}
    for row, w0 in widths0.items():
        floor = max(float(lbound), float(k_min))
        if int(min_width) > 1:
            floor = max(floor, min(1.0, int(min_width) / float(w0)))
        out[row] = min(1.0, floor)
    return out


def _modes(model: nn.Module) -> dict:
    return {name: bool(module.training) for name, module in model.named_modules()}


def _set_modes(model: nn.Module, modes: dict) -> None:
    for name, module in model.named_modules():
        if name in modes:
            module.training = modes[name]


def layer_inputs(model: nn.Module, input_shape, device=None) -> dict:
    """``{id(module): (h, w)}``: the height and width of the input every Conv2d of ``model`` reads on one sample of
    ``input_shape`` ((1, 1) for a Linear), from one hooked forward in eval mode (the modules' train / eval flags are
    put back). A module called more than once reports its first call."""
    seen, handles = {}, []

    def hook(module, inputs, _output):
        x = inputs[0]
        seen.setdefault(id(module), (int(x.shape[2]), int(x.shape[3])) if x.dim() == 4 else (1, 1))

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    modes = _modes(model)
    model.eval()
    try:
        with torch.no_grad():
            where = device if device is not None else next(model.parameters()).device
            model(torch.zeros(1, *input_shape, device=where))
    finally:
        for handle in handles:
            handle.remove()
        _set_modes(model, modes)
    return seen


def group_features(model: nn.Module, mwr, plan, input_shape, device=None):
    """``(features, flops)`` of the planned groups in plan order. ``features[t]`` is the static part of group t's state
    before scaling, in ``STATIC_FEATURES``' order: t, n (the group's width), then c, h, w, stride and k of the layer at
    the group's first walk row (its first producer in depth order; a Linear has stride 0, kernel 1 and h = w = 1), and
    the group's FLOPs. ``flops[t]`` is that last entry: the MACs (``utils.per_module_macs`` on the origin) of every
    layer that writes the group's channels, its producers and depthwise convolutions. The model's train / eval flags
    are put back."""
    modes = _modes(model)
    try:
        shapes = layer_inputs(model, input_shape, device)
        macs = utils.per_module_macs(model, input_shape, device)
    finally:
        _set_modes(model, modes)
    features, flops = [], []
    for t, (group, row) in enumerate(plan):
        layer = mwr.all_layers[mwr.row_to_main_layer[row]]
        h, w = shapes.get(id(layer), (1, 1))
        if isinstance(layer, nn.Conv2d):
            c, stride, k = layer.in_channels, layer.stride[0], layer.kernel_size[0]
        else:
            c, stride, k = layer.in_features, 0, 1
        f = float(sum(macs.get(id(m), 0) for m in list(group.producers) + list(group.depthwise)))
        features.append([float(t), float(group.width), float(c), float(h), float(w), float(stride), float(k), f])
        flops.append(f)
    return features, flops


def minmax(rows):
    """AMC's state scaling (``env/channel_pruning_env.py`` L538-542): every column of ``rows`` to [0, 1] by its
    minimum and maximum over the rows; a constant column is left as it is, as there."""
    out = [list(row) for row in rows]
    for j in range(len(out[0]) if out else 0):
        low, high = min(row[j] for row in out), max(row[j] for row in out)
        if high - low > 0:
            for row in out:
                row[j] = (row[j] - low) / (high - low)
    return out


class GroupEnv:
    """
    AMC's ``ChannelPruningEnv`` (``env/channel_pruning_env.py``) on SPECTRA's coupled groups: one step per planned
    group, in plan order; an action is the group's keep.

    ``cost`` is the budget's cost model (``plan_agent.FlopModel``: ``rows``, ``widths0``, ``total0`` and ``cost``, the
    kept MACs at given widths), ``floors`` each group's lowest keep (``keep_floors``), ``target`` the kept fraction to
    meet, ``features`` and ``flops`` come from ``group_features``. A target below what every group at its lowest width
    keeps is refused (ValueError).

    ``step(action)`` is AMC's ``step`` (L87-156) with its ``_action_wall`` (L267-298). The action is clipped to
    [0, rbound] (L271) and raised to the group's floor (L296), the keep becomes the width the landing's cut leaves
    (``realised_width``), and that width is capped by ``wall()``; the floor wins over the cap, as in AMC, though here
    the cap is never below it. ``wall()`` is the widest width at which the cost, with the groups before as decided and
    the groups after at their lowest width, is at most the budget; it is found by bisection on the integer width, so it
    holds for a cost that is not linear in the width (a layer that reads and writes one group). After every step the
    undecided groups can therefore still meet the target, and a finished plan costs at most the budget.

    The state (``observe``) is the group's scaled static features, then reduced, rest and the previous group's
    realised keep, as the module docstring defines them. ``widths`` and ``keeps`` hold the decided groups'.
    """

    def __init__(self, cost, floors, target: float, features, flops, rbound: float = 1.0):
        self.cost, self.rows = cost, [int(row) for row in cost.rows]
        if not self.rows:
            raise ValueError("the plan has no group to decide")
        self.w0 = {int(row): int(width) for row, width in cost.widths0.items()}
        self.total0, self.target, self.rbound = float(cost.total0), float(target), float(rbound)
        self.budget = self.target * self.total0
        self.limit = self.budget + 1e-9 * self.total0  # the budget, with room for float rounding only
        self.floor = {row: min(1.0, float(floors[row])) for row in self.rows}
        self.lo = {row: realised_width(self.w0[row], self.floor[row]) for row in self.rows}
        lowest = float(cost.cost(self.lo))
        self.lowest = lowest / self.total0
        if lowest > self.limit:
            raise ValueError(f"the FLOPs target x{self.target:.3f} is below the x{self.lowest:.3f} that every group at "
                             f"its lowest keep leaves; no plan meets it")
        self.static = [tuple(row) for row in minmax(features)]
        self.rest = [sum(flops[t + 1:]) / self.total0 for t in range(len(self.rows))]
        self.reset()

    def observe(self, t: int, reduced: float, prev_keep: float):
        return self.static[t] + (float(reduced), float(self.rest[t]), float(prev_keep))

    def reset(self):
        """Start an episode and return its first state: nothing reduced, the previous action 1 (L158-177)."""
        self.t, self.widths, self.keeps = 0, {}, {}
        self.trial = dict(self.lo)  # the decided groups at their widths, the others at their lowest
        self.full = dict(self.w0)  # the decided groups at their widths, the others whole
        self.state = self.observe(0, 0.0, 1.0)
        return self.state

    def wall(self) -> int:
        """The widest width the current group may keep (see the class docstring)."""
        row = self.rows[self.t]
        low, high, trial = self.lo[row], self.w0[row], self.trial
        trial[row] = high
        if float(self.cost.cost(trial)) > self.limit:
            while high - low > 1:  # low always fits: the groups before left the target reachable
                mid = (low + high) // 2
                trial[row] = mid
                if float(self.cost.cost(trial)) <= self.limit:
                    low = mid
                else:
                    high = mid
            high = low
        trial[row] = self.lo[row]
        return high

    def step(self, action: float):
        """Decide the current group from the agent's ``action``: ``(next state, done)``. The last step returns the
        state it was given, as AMC does (L138)."""
        row = self.rows[self.t]
        keep = max(min(max(float(action), 0.0), self.rbound), self.floor[row])
        width = max(min(realised_width(self.w0[row], keep), self.wall()), self.lo[row])
        self.widths[row] = self.trial[row] = self.full[row] = width
        self.keeps[row] = width / float(self.w0[row])
        self.t += 1
        done = self.t >= len(self.rows)
        if not done:  # L151-153: the next state's reduced, rest and last action
            reduced = 1.0 - float(self.cost.cost(self.full)) / self.total0
            self.state = self.observe(self.t, reduced, self.keeps[row])
        return self.state, done


# ------------------------------------------------------------------ the loop (amc_search.py, train)


def search(space, agent, score, n_episodes: int, n_warmup: int, log=None):
    """
    AMC's search loop (``amc_search.py`` L102-181): ``(runs, best)``.

    Episode e of ``n_episodes`` takes one action per step of ``space`` (``GroupEnv``): the agent's random action while
    ``e <= n_warmup`` (L115), else its noisy policy action (L119). ``score(widths)`` then gives the episode its one
    reward (in AMC the env's, computed at the last step: ``env/channel_pruning_env.py`` L119-128). Every step of the
    episode is stored with that reward and with the agent's own action, the one from before the wall (L125,
    L149-153), and from episode ``n_warmup + 1`` on the agent is updated once after each step stored (L154-155).
    ``runs[e]`` holds the episode's reward, widths, realised keeps and whether it was random; ``best`` is the episode
    of the highest reward, the earlier one on a tie (AMC replaces its best strategy only on a strictly higher reward,
    ``env/channel_pruning_env.py`` L130). ``log(e, run, is_best)`` is called after each episode.
    """
    runs, best = [], None
    for episode in range(int(n_episodes)):
        acts_randomly = episode <= int(n_warmup)
        state, done, steps = space.reset(), False, []
        while not done:
            action = agent.random_action() if acts_randomly else agent.select_action(state, episode)
            following, done = space.step(action)
            steps.append((state, action, done))
            state = following
        widths = dict(space.widths)
        run = {"reward": float(score(widths)), "widths": widths, "keeps": dict(space.keeps), "random": acts_randomly}
        for s_t, a_t, terminal in steps:
            agent.observe(run["reward"], s_t, a_t, terminal)
            if not acts_randomly:
                agent.update_policy()
        runs.append(run)
        is_best = best is None or run["reward"] > runs[best]["reward"]
        if is_best:
            best = episode
        if log is not None:
            log(episode, run, is_best)
    return runs, best


# ------------------------------------------------------------------ the allocation kind


def amc_plan(env, target: float, budget: str = "flops"):
    """
    ``(widths, info)`` of AMC's search on the network ``env`` was reset on, in ``alloc_walk.plan_targets``' format
    with ``kind`` amc, so ``alloc_walk._state`` logs, records, grid-rounds and walks it as any plan. ``target`` is the
    kept FLOPs the plan aims at (the walk target less ``SPECTRA_ALLOC_UNDERSHOOT``, as for every kind).

    The search is the module docstring's: ``GroupEnv`` on the origin's ``plan_agent.FlopModel`` with the floors of
    ``keep_floors`` (AMC's lbound, ``alloc_walk.min_keep``, ``fortify.plan_min_width``), ``DDPG`` on the env's device,
    ``search`` for ``episodes()`` episodes with ``warmup()`` of warm-up, each plan scored by ``plan_proxies.PlanScorer``
    with ``proxy()`` on ``env.val_loader`` only, on the batches ``SPECTRA_EVAL_PROXY_SEED`` draws once for all episodes
    and under that seed at every score (so a plan seen before is read from a table). The best episode's realised keeps
    are then the weights of ``alloc_walk._land`` at ``target`` with the same floors: at scale 1 that bisection cuts
    exactly the best episode's widths, and it moves the scale only as far as the target asks. ``info["weights"]`` and
    ``info["sens"]`` are the best episode's keeps, ``info["keeps"]`` the landed ones, ``info["kept"]`` the FLOPs the
    landed cut really keeps, and ``info["amc"]`` the summary the plan line and the ``alloc_plan`` record read.

    Everything runs inside ``plan_proxies.seeded`` on the search's own seed (``alloc_walk.randbest_seed`` of
    ``SPECTRA_ALLOC_SAMPLE_SEED`` and the run's seed), so every global RNG state is put back; the origin's train / eval
    flags are put back too, and its weights are only read. A budget other than ``flops`` is refused before any work.
    One line per episode, a summary line and an ``alloc_amc`` record carry the search.
    """
    from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
    from src import alloc_walk, fortify, group_sensitivity, plan_agent, plan_proxies
    if budget != "flops":
        raise ValueError(f"SPECTRA_ALLOC_KIND=amc searches under AMC's FLOPs constraint and prices its budget rule on "
                         f"plan_agent.FlopModel: set SPECTRA_ALLOC_BUDGET=flops (the budget is {budget!r})")
    hp, proxy_name = hyper_parameters(), proxy()
    seeds = (alloc_walk.sample_seed(), alloc_walk.run_seed(env))
    seed, proxy_seed = alloc_walk.randbest_seed(*seeds), plan_proxies.proxy_seed()
    net = os.path.basename(str(env.selected_net_path))
    device = env.conf.device
    k_min, min_width = alloc_walk.min_keep(), fortify.plan_min_width()
    started = time.perf_counter()
    clock, table = {"score": 0.0}, {}
    with plan_proxies.seeded(seed):  # the search's own stream; every global RNG state is put back after it
        model = env.current_model.to(device)
        modes = _modes(model)
        try:
            mwr = ModelWithRows(model)
            plan = group_sensitivity.group_plan(mwr, env._dependency_groups(mwr))
            if not plan:
                raise ValueError(f"SPECTRA_ALLOC_KIND=amc on {net}: the plan has no group to decide")
            shape = env._input_shape()
            fm = plan_agent.FlopModel(model, plan, shape, device)
            rows, params0 = list(fm.rows), float(fm.pm.total0)
            floors = keep_floors(fm.widths0, hp["lbound"], k_min, min_width)
            features, flops = group_features(model, mwr, plan, shape, device)
            try:
                space = GroupEnv(fm, floors, target, features, flops, hp["rbound"])
            except ValueError as error:
                raise ValueError(f"SPECTRA_ALLOC_KIND=amc on {net}: {error}") from None
            scorer = plan_proxies.PlanScorer(model, plan, env.val_loader, device, train_loader=env.train_loader,
                                             input_shape=shape,
                                             create_handler=getattr(env, "create_learning_handler", None), name=net)
            batches = scorer.batches(proxy_name, proxy_seed)
            agent = DDPG(STATE_DIM, len(rows), hp, seed, device)
            utils.print_flush(
                f"[alloc] amc {net}: {hp['episodes']} episodes (the first {min(hp['episodes'], hp['warmup'] + 1)} "
                f"random), {len(rows)} groups, FLOPs target x{float(target):.3f} (every group at its lowest keep "
                f"x{space.lowest:.3f}), proxy {proxy_name}, lbound {hp['lbound']:g}, seed {seed}, agent on "
                f"{agent.device}; cost model {fm.probes} probes in {fm.seconds:.1f} s")

            def score(widths):
                key = tuple(int(widths[row]) for row in rows)
                if key not in table:
                    tick = time.perf_counter()
                    with plan_proxies.seeded(proxy_seed):  # the score is a function of the widths alone
                        table[key] = float(scorer.score(widths, proxy_name, batches))
                    clock["score"] += time.perf_counter() - tick
                return table[key]

            def log(episode, run, is_best):
                run["flops"] = float(fm.kept(run["widths"]))
                run["params"] = fm.params(run["widths"]) / params0
                utils.print_flush(
                    f"[alloc] amc {net} ep={episode} {'random' if run['random'] else 'ddpg'} "
                    f"reward={run['reward']:.4f} FLOPs x{run['flops']:.3f} params x{run['params']:.3f}"
                    + (" best" if is_best else ""))

            runs, best = search(space, agent, score, hp["episodes"], hp["warmup"], log)
            del scorer, batches
            picked = runs[best]
            weights = {row: float(picked["keeps"][row]) for row in rows}
            size0 = float(utils.calc_flops(model, shape))
            keeps, frac, widths = alloc_walk._land(model, plan, rows, weights, floors, target,
                                                   lambda m: utils.calc_flops(m, shape), size0, shape, LAND_ITERS,
                                                   LAND_TOL)
        finally:
            _set_modes(model, modes)
    seconds = time.perf_counter() - started
    moved = sum(int(widths.get(row, -1)) != int(picked["widths"][row]) for row in rows)
    rewards = [run["reward"] for run in runs]
    utils.print_flush(
        f"[alloc] amc {net}: best episode {best} of {len(runs)} (warm-up {hp['warmup']}), val {picked['reward']:.4f} "
        f"at FLOPs x{picked['flops']:.3f} (spread {min(rewards):.4f}-{max(rewards):.4f}, median "
        f"{statistics.median(rewards):.4f}), proxy {proxy_name}, {len(table)} plans scored, {agent.updates} updates; "
        f"landed x{frac:.3f} (target x{float(target):.3f}, {moved} groups moved), {seconds / 60.0:.1f} min")
    try:
        import src.run_recorder as run_recorder
        run_recorder.record(
            "alloc_amc", network=str(env.selected_net_path), proxy=proxy_name, episodes=len(runs),
            warmup=hp["warmup"], seed=seed, sample_seed=seeds[0], run_seed=seeds[1], proxy_seed=proxy_seed,
            budget=budget, target=float(target), k_min=k_min, **({"min_width": min_width} if min_width > 1 else {}),
            **({"fs_lr": plan_agent.fs_lr()} if proxy_name.startswith("fs") else {}),
            pick=best, score=picked["reward"], kept=picked["flops"], params=picked["params"],
            keeps={str(row): weights[row] for row in rows},
            widths={str(row): int(picked["widths"][row]) for row in rows},
            landed={"kept": float(frac), "moved": int(moved), "keeps": {str(row): float(keeps[row]) for row in rows},
                    "widths": {str(row): int(width) for row, width in widths.items()}},
            hyper=dict(hp), amc_commit=AMC_COMMIT, groups=len(rows), state_dim=STATE_DIM, lowest=space.lowest,
            device=str(agent.device), scored=len(table), updates=agent.updates,
            seconds=round(seconds, 1), score_seconds=round(clock["score"], 1),
            rewards=rewards, flops=[run["flops"] for run in runs])
    except Exception:  # noqa: BLE001 - the record is a convenience
        pass
    summary = {"pick": best, "episodes": len(runs), "warmup": hp["warmup"], "score": picked["reward"],
               "proxy": proxy_name, "seed": seed, "est_kept": picked["flops"], "scored": len(table), "moved": moved,
               "seconds": seconds, "rewards": rewards}
    return widths, {"kind": "amc", "alpha": 0.0, "target": float(target), "kept": float(frac), "budget": budget,
                    "keeps": keeps, "origin_widths": dict(fm.widths0), "sens": dict(weights), "held": 0,
                    "weights": weights, "amc": summary, **({"min_width": int(min_width)} if min_width > 1 else {})}
