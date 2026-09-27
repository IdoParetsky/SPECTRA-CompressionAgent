"""
SPECTRA fortifications. On by default (SPECTRA_FORTIFY=1); set SPECTRA_FORTIFY=0 to disable.

When enabled:
  * Stem / narrow-width action masking (identity-only on fragile layers)
  * Post-warmup entropy coefficient anneal (AMC-style exploration decay)
  * Extra per-layer representation channels (depth, stem, coupling, width)

Always available (not gated): mid-run ``train_resume.pt`` save/load helpers used by the
agent so USR1 / preempt can warm-continue without a full cold start.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.distributions import Categorical

import src.pruning as pruning

FORTIFY_TOKEN_DIM = 4  # relative_depth, is_stem, is_coupled, width_norm


def fortify_enabled() -> bool:
    raw = os.environ.get("SPECTRA_FORTIFY", "1").strip().lower()
    return raw in ("1", "true", "yes")


def budget_in_state() -> bool:
    """Broadcast remaining-param ratio as an extra token channel (default off)."""
    raw = os.environ.get("SPECTRA_BUDGET_IN_STATE", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def state_slack() -> bool:
    """
    Two extra token channels (default off): **accuracy slack** and **pass progress**.

    The NEON trichotomy is scored on the *cumulative* val Δacc against the origin
    (``NetworkEnv.step`` → ``compute_reward(new_acc, original_acc, …)``), so the optimal
    policy is "cut while the τ band has room, stop when it is spent". Until 13 Sep the
    state carried no trace of how much of the band was used, so no policy could implement
    that rule — the only state-free safe schedule is a mild one. Slack is
    ``clip((τ + Δacc_pp) / τ, −1, 1)`` (1 = untouched band, 0 = at τ, <0 = over); progress
    is ``actions_taken / (rows · passes)``. Pin ``SPECTRA_STATE_SLACK=1`` (new actors
    only: token width changes).
    """
    return _flag("SPECTRA_STATE_SLACK")


def accuracy_slack(delta_acc_pp: float, tau_pp: float) -> float:
    tau = max(float(tau_pp), 1e-6)
    return max(-1.0, min(1.0, (tau + float(delta_acc_pp)) / tau))


def algo() -> str:
    """``SPECTRA_ALGO``: ``a2c`` (historical one-episode-per-update) or ``ppo``."""
    raw = os.environ.get("SPECTRA_ALGO", "a2c").strip().lower()
    return "ppo" if raw == "ppo" else "a2c"


def ppo_episodes_per_update() -> int:
    return max(1, int(os.environ.get("SPECTRA_PPO_EPISODES", "4")))


def ppo_epochs() -> int:
    return max(1, int(os.environ.get("SPECTRA_PPO_EPOCHS", "4")))


def ppo_clip() -> float:
    return float(os.environ.get("SPECTRA_PPO_CLIP", "0.2"))


def ppo_gae_lambda() -> float:
    return float(os.environ.get("SPECTRA_PPO_GAE_LAMBDA", "0.95"))


def ppo_target_kl() -> float:
    """Stop the PPO epoch loop early once approx-KL exceeds this (0 disables)."""
    return float(os.environ.get("SPECTRA_PPO_TARGET_KL", "0.03"))


def ppo_value_coef() -> float:
    return float(os.environ.get("SPECTRA_PPO_VALUE_COEF", "0.5"))


def ppo_warmup_episodes() -> int:
    """Uniform-action episodes before the first PPO update (critic / return-scale warm-up)."""
    return max(0, int(os.environ.get("SPECTRA_PPO_WARMUP_EPISODES", "0")))


def agent_lr(default: float) -> float:
    """
    Optimiser lr for the actor/critic. ``--learning_rate`` also sets the fine-tune Adam lr
    (ClassificationHandler), so the agent gets its own knob: ``SPECTRA_AGENT_LR``.
    """
    raw = os.environ.get("SPECTRA_AGENT_LR", "").strip()
    return float(raw) if raw else float(default)


def encoder_dropout() -> float:
    """Dropout inside the trainable state encoder (``SPECTRA_ENCODER_DROPOUT``, default 0.1)."""
    raw = os.environ.get("SPECTRA_ENCODER_DROPOUT", "").strip()
    return float(raw) if raw else 0.1


def policy_head_zero_init() -> bool:
    """Zero the actor's last Linear so the initial policy is exactly uniform (default off)."""
    return _flag("SPECTRA_POLICY_HEAD_ZERO_INIT")


def train_ft_epochs():
    """
    Fine-tune epoch cap used **only** in ``AGENT_TRAIN`` mode (``SPECTRA_TRAIN_FT_EPOCHS``).

    TEST keeps ``--num_epochs`` (40) for every method. A shorter recovery during policy
    training buys 3–4× more episodes per GPU-day; it is pessimistic w.r.t. TEST (the policy
    sees less recovery than it will get), which errs on the conservative side.
    """
    raw = os.environ.get("SPECTRA_TRAIN_FT_EPOCHS", "").strip()
    return int(raw) if raw else None


def train_ft_patience():
    raw = os.environ.get("SPECTRA_TRAIN_FT_PATIENCE", "").strip()
    return int(raw) if raw else None


# ---------------------------------------------------------------- v3 (16 Sep): state group-cost

STATE_GROUPCOST_DIM = 4  # group param share, group MAC share, owner-count share, cuts this episode


def state_groupcost() -> bool:
    """
    Four extra token channels per layer (default off): the **group cost** of cutting it.

    A residual/concat/depthwise tie makes several layers share one channel dimension; on a
    CIFAR ResNet one stream is owned by 9–10 rows. Two layers with identical local statistics
    can differ by an order of magnitude in what pruning them actually removes from the whole
    net. v2 exposed that only for the *target* row (action-cost slots). Per layer this adds:
    param share of the layer's group (all producers + consumers' slices), MAC share of the
    group, owner count / max owner count, and cuts already applied to that group this
    episode (``min(1, n/2)``). Pin ``SPECTRA_STATE_GROUPCOST=1`` (new actors only).
    """
    return _flag("SPECTRA_STATE_GROUPCOST")


def train_tau(default: float) -> float:
    """
    τ used for the reward and the slack channel **in AGENT_TRAIN mode only**
    (``SPECTRA_TRAIN_TAU``). Both are τ-relative, so a stricter training band is a curriculum
    that makes the band edge reachable on robust train nets (v2: 2.5 % of steps over budget).
    Default off = ``--allowed_acc_reduction``.
    """
    raw = os.environ.get("SPECTRA_TRAIN_TAU", "").strip()
    return float(raw) if raw else float(default)


# ---------------------------------------------------------------- v4-1 (16 Sep): factored rate x ranking head

def factored_head() -> bool:
    """
    ``SPECTRA_FACTORED_HEAD=1`` (V4-1): the actor carries two heads — rate over
    ``--compression_rates`` and ranking over ``--ranking_menu`` — and an action is the pair.
    Grows the criterion menu (L1, FPGM, BN-scale, SVD, Taylor) without a 13-way softmax: each
    head sees every sample and correlated criteria stop diluting credit. Off = v2/v3 actors
    unchanged (single Categorical over ``(rate, ranking)`` pairs or rates).
    """
    return _flag("SPECTRA_FACTORED_HEAD")


def is_factored_dist(dist) -> bool:
    return hasattr(dist, "rate") and hasattr(dist, "rank") and hasattr(dist, "log_prob")


# ---------------------------------------------------------------- v5 P8 (18 Sep): NEON layer replacement
#
# Hirsch & Katz 2022, Sec. 3 "Layer replacement": rather than removing neurons, NEON generated
# a *new* layer of the desired width (l'_i = a_t * W_{l_i}), initialised randomly, installed
# it, froze every other layer and trained the new module until convergence, then refreshed
# the feature maps before the next state. Upstream NEON_NetworkEnv.py: --prune False built a
# fresh nn.Linear(in, new_size) *and* a fresh nn.Linear(new_size, out) for the consumer, plus a
# fresh BatchNorm1d; is_learn_new_layers_only kept exactly those modules trainable
# (build_parameters_to_freeze returns the *trainable* ids — the name is inverted) and
# train_model ran up to num_epoch with patience 10 on the train loss, restoring the best
# state. SPECTRA's live --prune keeps the surviving filters (the method NEON rejected) and
# fine-tunes the full net (recipe A). The flags below put NEON-C back for CNN *groups*:
#
#   recipe A    keep remaining filters, full-net FT                      (live default)
#   recipe B    keep remaining filters, edited group only                (--train_compressed_layer_only=True; 0/32 OK, §12)
#   recipe C-G  SPECTRA_FT_REINIT_EDITED=1: reinit the edited group (producers at the new width,
#               group BN reset, consumers' input slices for the group's channels), freeze the
#               rest, train the edited set until the *val* accuracy plateaus  (Gilad-literal)
#   recipe C-G+ SPECTRA_FT_REINIT_THEN_POLISH=1: C-G, then a short full-net low-LR polish
#               (assigned SPECTRA CNN method; residual adds and BN make the frozen rest not
#               independent of the fresh group the way a dense stem is)
#
# All default off; v2/v3/V4 replays are byte-identical. Identity (rate 1.0) skips prune and FT
# under every recipe. Masked fallbacks (no structural edit) have no new module to reinit and
# fall back to recipe A for that step (recorded as reinit=False).

FT_RECIPES = ("A", "B", "C-G", "C-G+")


def ft_reinit_edited() -> bool:
    """``SPECTRA_FT_REINIT_EDITED=1`` — NEON-C on CNN groups (recipe C-G), default off.

    The value ``pca`` is not this flag. It selects C-PCA (``ft_pca_reinit``), which
    keeps recipe A's full-net fine-tune and only changes how the new weights are built.
    """
    return _flag("SPECTRA_FT_REINIT_EDITED")


def ft_pca_reinit() -> bool:
    """``SPECTRA_FT_REINIT_EDITED=pca`` — principal-direction replacement (C-PCA), default off."""
    return os.environ.get("SPECTRA_FT_REINIT_EDITED", "").strip().lower() == "pca"


def ft_lsq_consumers() -> bool:
    """``SPECTRA_FT_LSQ_CONSUMERS=1`` — least-squares refit of consumer kernels (A-LSQ), default off."""
    return _flag("SPECTRA_FT_LSQ_CONSUMERS")


def ft_bn_recal() -> bool:
    """``SPECTRA_FT_BN_RECAL=1`` — reset and re-estimate BatchNorm running stats after a cut, default off."""
    return _flag("SPECTRA_FT_BN_RECAL")


def ft_calib_batches(default: int = 2) -> int:
    """Train batches used to fit A-LSQ / C-PCA (``SPECTRA_FT_CALIB_BATCHES``, default 2)."""
    return max(1, _env_int_or("SPECTRA_FT_CALIB_BATCHES", default))


def action_menu() -> str:
    """``rates`` (default) or ``budget`` (``SPECTRA_ACTION_MENU``).

    Under ``budget``, a rate in ``(0, 1)`` is the fraction of the *whole network's*
    parameters to remove through the current group, and a negative rate is STOP.
    """
    raw = os.environ.get("SPECTRA_ACTION_MENU", "rates").strip().lower()
    return "budget" if raw == "budget" else "rates"


def is_stop_rate(rate) -> bool:
    """True when this action ends the episode. Only under the budget menu, and only if rate < 0."""
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return False
    return action_menu() == "budget" and value < 0.0


def budget_keep_rate(group_param_fraction: float, remove_network_fraction: float) -> float:
    """Map "remove this fraction of the network via this group" onto a layer keep-rate.

    A group that owns 10% of the parameters, asked to remove 2% of the network, keeps
    80% of its channels. Asking for more than the group owns keeps nothing (rate 0).
    A group that owns nothing, or a zero request, stays at rate 1 (identity).
    """
    owned = float(group_param_fraction)
    remove = float(remove_network_fraction)
    if remove <= 0.0 or owned <= 1e-12:
        return 1.0
    cut = min(1.0, remove / owned)
    return max(0.0, 1.0 - cut)


# Under the budget menu a request that would remove the whole group (or more) is not a
# legal cut: it is masked, never rounded down to one channel.
BUDGET_MIN_KEEP = 0.05


def stop_reward_scale(default: float = 100.0) -> float:
    """
    Multiplier on the slack-weighted in-band area paid to STOP (``SPECTRA_STOP_REWARD_SCALE``).
    Area is (fraction of the network removed) × (slack / τ); the per-step in-band reward is
    ρ in percentage points, so ×100 puts STOP in the same units. 0 makes STOP a free exit.
    """
    return _env_float_or("SPECTRA_STOP_REWARD_SCALE", default)


# A budget action whose smallest realisable cut (one channel of the group) removes more than
# this multiple of the request is infeasible: the env must not spend 12 % of the net on a 1 % ask.
BUDGET_OVERSHOOT_TOLERANCE = 1.5


def effective_rates(compression_rates: Dict[int, float], group_param_fraction: float,
                    group_width: Optional[int] = None) -> Dict[int, Tuple[float, bool, bool]]:
    """
    ``{action index: (keep rate the env will apply, is_stop, feasible)}`` for the current group.

    ``rates`` menu: identity mapping, every action feasible, nothing is STOP.
    ``budget`` menu (``SPECTRA_ACTION_MENU=budget``): a value ≥ 1 is identity; a negative
    value is STOP (identity step that ends the episode); a value in (0, 1) is "remove this
    fraction of the *network's* parameters through this group" and maps onto a keep rate
    via :func:`budget_keep_rate`. Infeasible = the request exceeds what the group owns
    (keep rate below ``BUDGET_MIN_KEEP``), or — when ``group_width`` is known — one channel of
    the group already removes more than ``BUDGET_OVERSHOOT_TOLERANCE`` × the request (the
    realised cut would not be the action the agent asked for). One mapping feeds the env
    step, the legal mask and the action-cost slots, so the three never disagree.
    """
    out: Dict[int, Tuple[float, bool, bool]] = {}
    budget = action_menu() == "budget"
    per_channel = None
    if group_width is not None and int(group_width) > 0:
        per_channel = float(group_param_fraction) / float(group_width)
    for idx, raw in compression_rates.items():
        value = float(raw)
        if not budget:
            out[idx] = (value, False, True)
        elif value < 0.0:
            out[idx] = (1.0, True, True)
        elif value >= 1.0:
            out[idx] = (1.0, False, True)
        else:
            keep = budget_keep_rate(group_param_fraction, value)
            feasible = keep >= BUDGET_MIN_KEEP
            if feasible and per_channel is not None and per_channel > BUDGET_OVERSHOOT_TOLERANCE * value:
                feasible = False
            out[idx] = (keep, False, feasible)
    return out


def ft_reinit_then_polish() -> bool:
    """``SPECTRA_FT_REINIT_THEN_POLISH=1`` — C-G followed by a short full-net polish (C-G+)."""
    return _flag("SPECTRA_FT_REINIT_THEN_POLISH")


def ft_recipe(train_compressed_layer_only: bool = False) -> str:
    """Name of the fine-tune recipe in force; polish implies reinit.

    C-PCA and A-LSQ are closed-form weight edits followed by recipe A's full-net
    fine-tune. They do not take the C-G "train only the new group" path.
    """
    if ft_reinit_then_polish():
        return "C-G+"
    if ft_pca_reinit():
        return "C-PCA"
    if ft_reinit_edited():
        return "C-G"
    if ft_lsq_consumers():
        return "A-LSQ"
    return "B" if train_compressed_layer_only else "A"


def _env_int_or(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def _env_float_or(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def ft_reinit_epochs(default: int = 60) -> int:
    """
    Epoch cap for training the reinitialised group (``SPECTRA_FT_REINIT_EPOCHS``, default 60).
    NEON's "until convergence" was num_epoch=100 with patience 10 on the train loss; a CNN
    group trained from scratch needs more than the 12-epoch policy-training budget, so this
    cap is separate from ``SPECTRA_TRAIN_FT_EPOCHS`` and ``--num_epochs``. Stops early on
    ``ft_reinit_patience`` epochs without a val improvement.
    """
    return _env_int_or("SPECTRA_FT_REINIT_EPOCHS", default)


def ft_reinit_patience(default: int = 6) -> int:
    """Val-plateau patience for the reinitialised group (``SPECTRA_FT_REINIT_PATIENCE``, default 6)."""
    return _env_int_or("SPECTRA_FT_REINIT_PATIENCE", default)


def ft_reinit_scope() -> str:
    """
    Which tensors of the edited group are re-drawn (``SPECTRA_FT_REINIT_SCOPE``):

    * ``group`` (default, NEON-source-literal): producers **and** consumer input slices and the
      group norms — upstream rebuilt ``Linear(in, new)``, ``Linear(new, out)`` and the BN.
    * ``producers`` (Gilad's oral wording, "the remaining filters of the newly-pruned layer"):
      producers and the group norms only; consumers keep their surviving input slices and
      adapt by training. On a residual stream this re-draws one stage's conv2s, not its conv1s.

    Both keep the same trainable set (the whole edited group). Part of the policy contract.
    """
    raw = os.environ.get("SPECTRA_FT_REINIT_SCOPE", "group").strip().lower()
    return "producers" if raw in ("producers", "producer", "owners") else "group"


def ft_reinit_select() -> str:
    """
    Model selection inside the group training: ``val`` (default; the reward is post-FT val
    Δacc, and the paper flow says train the new layer *until convergence*) or ``train``
    (NEON upstream: best train loss, patience on the train loss).
    """
    raw = os.environ.get("SPECTRA_FT_REINIT_SELECT", "val").strip().lower()
    return "train" if raw == "train" else "val"


def ft_polish_epochs(default: int = 8) -> int:
    """Epoch cap for the C-G+ full-net polish (``SPECTRA_FT_POLISH_EPOCHS``, default 8)."""
    return _env_int_or("SPECTRA_FT_POLISH_EPOCHS", default)


def ft_polish_patience(default: int = 3) -> int:
    """Val patience for the polish (``SPECTRA_FT_POLISH_PATIENCE``, default 3)."""
    return _env_int_or("SPECTRA_FT_POLISH_PATIENCE", default)


def ft_polish_lr_mult(default: float = 0.1) -> float:
    """Polish learning-rate multiplier on the FT lr (``SPECTRA_FT_POLISH_LR_MULT``, default 0.1)."""
    return _env_float_or("SPECTRA_FT_POLISH_LR_MULT", default)


def eval_counterfactual() -> bool:
    """
    ``SPECTRA_EVAL_COUNTERFACTUAL=1`` (V6, default off) — at every actor step of an eval walk,
    also ask the frozen actor what it would do on three *counterfactual* states built from the
    real one (layer features zeroed; layer features shuffled across positions; everything but
    the positional/type/marker channels zeroed) and log whether the argmax changes. One extra
    forward per variant, no extra FT. Fraction of steps whose action depends on the content is
    the cheapest identification of "does the policy read the state at all" — the question that
    must be answered before any encoder GPU (ledger §16 was measured under a uniform policy).
    """
    return _flag("SPECTRA_EVAL_COUNTERFACTUAL")


def refresh_all_features() -> bool:
    """
    ``SPECTRA_REFRESH_ALL_FEATURES=1`` — NEON "feature-maps update": after a structural edit,
    re-extract the activation moments of *every* layer before the next state (upstream
    rebuilt the FeatureExtractor each step). Default off keeps the live behaviour (only the
    edited row's span is refreshed; downstream moments stay cached from before the edit).
    The P8 profiles turn it on; it is part of the policy contract.
    """
    return _flag("SPECTRA_REFRESH_ALL_FEATURES")


def mask_policy(dist, legal: torch.Tensor):
    """Apply the legal-rate mask to a plain Categorical or to the rate head of a factored one."""
    if is_factored_dist(dist):
        return dist.with_rate(apply_action_mask(dist.rate, legal))
    return apply_action_mask(dist, legal)


def pick_action(dist, legal: torch.Tensor, *, deterministic: bool, device):
    """
    ``(rate_idx: int, rank_idx: int | None, logp: float)`` from a masked policy.
    Ranking index is None for a plain Categorical (the caller maps rate index → ranking via
    ``action_rankings_dict``) and for identity under a factored head.
    """
    masked = mask_policy(dist, legal)
    if is_factored_dist(masked):
        if deterministic:
            r, k = masked.argmax()
        else:
            r, k = masked.sample()
        r_i, k_i = int(r.item()), int(k.item())
        logp = float(masked.log_prob(r_i, k_i).item())
        if r_i == masked.identity_index:
            return r_i, None, logp
        return r_i, k_i, logp
    if deterministic:
        probs = masked.probs
        while probs.dim() > 1:
            probs = probs[0]
        r = probs.argmax().reshape(1)
    else:
        r = masked.sample().reshape(1)
    return int(r.item()), None, float(masked.log_prob(r.to(masked.probs.device)).reshape(-1)[0].item())


# ---------------------------------------------------------------- v3 (16 Sep): keep-learning governor

def probe_every() -> int:
    """Deterministic fixed-probe evaluation every N on-policy episodes (0 = off)."""
    return max(0, int(os.environ.get("SPECTRA_PROBE_EVERY", "0") or 0))


def probe_net_patterns():
    """Substrings of catalog paths that form the probe set (``SPECTRA_PROBE_NETS``, comma list)."""
    raw = os.environ.get("SPECTRA_PROBE_NETS", "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()]


def probe_score_kind() -> str:
    """
    What the fixed probe measures (``SPECTRA_PROBE_SCORE``):

    * ``cut`` (default, v3–V6): mean ``1 − kept`` at the deepest in-band point of the argmax
      walk. Saturates at the deepest *legal* walk — on thin probe nets that is the mild clone —
      and is blind to Δacc, so a kinder policy at equal depth cannot score higher. Every 12/4
      arm froze at the same 0.262 for this reason (V7 diagnosis, 21 Sep).
    * ``area``: mean slack-weighted in-band cut area (``NetworkEnv.episode_inband_area``):
      Σ_in-band steps (size removed) × (remaining slack / τ). Deeper-in-band and
      kinder-at-equal-depth both raise it; an over-budget walk earns nothing past the band.
    """
    raw = os.environ.get("SPECTRA_PROBE_SCORE", "cut").strip().lower()
    return "area" if raw == "area" else "cut"


def state_tokens() -> str:
    """
    What a state token is (``SPECTRA_STATE_TOKENS``, V8 representation cell):

    * ``layers`` (default, every actor so far): one token per layer; channel coupling enters
      only as a learned same-group attention scalar.
    * ``groups``: one token per prune unit (coupling id) = mean of its member layer tokens +
      four structure columns, with a learned feeds / fed-by relation bias
      (``src/group_tokens.py``). Changes the token width → pinned in ``policy_config``.
    """
    raw = os.environ.get("SPECTRA_STATE_TOKENS", "layers").strip().lower()
    return "groups" if raw == "groups" else "layers"


def min_episodes(default: int) -> int:
    """Never stop on patience before this many episodes (``SPECTRA_MIN_EPISODES``)."""
    raw = os.environ.get("SPECTRA_MIN_EPISODES", "").strip()
    return int(raw) if raw else int(default)


def patience_episodes(default: int) -> int:
    """Episodes without a selection-score improvement before stopping (``SPECTRA_PATIENCE_EPISODES``)."""
    raw = os.environ.get("SPECTRA_PATIENCE_EPISODES", "").strip()
    return int(raw) if raw else int(default)


def rewind_best() -> bool:
    """
    ``SPECTRA_REWIND_BEST=1`` (Ido 15 Sep): when the **probe** score has not improved for
    ``SPECTRA_REWIND_PATIENCE`` episodes, reload the best snapshot's actor+critic, reset Adam,
    bump the entropy coefficient for ``SPECTRA_REWIND_ENTROPY_EPISODES`` episodes and continue
    (PBT "exploit" / Go-Explore "return, then explore"). Never on the raw 4-net batch max.
    """
    return _flag("SPECTRA_REWIND_BEST")


def rewind_patience() -> int:
    return max(1, int(os.environ.get("SPECTRA_REWIND_PATIENCE", "50")))


def rewind_max() -> int:
    return max(0, int(os.environ.get("SPECTRA_REWIND_MAX", "3")))


def rewind_entropy() -> float:
    return float(os.environ.get("SPECTRA_REWIND_ENTROPY", "0.02"))


def rewind_entropy_episodes() -> int:
    return max(1, int(os.environ.get("SPECTRA_REWIND_ENTROPY_EPISODES", "30")))


class LearningGovernor:
    """
    Decides *when a train stops*, *what counts as improvement* and *when to rewind*.

    v2 stopped when the 4-episode ``batch_score`` had not beaten its historical max for
    ``max(n_nets, 100)`` episodes. That max is a biased order statistic of a noisy,
    composition-dependent score (audit H3): B died at episode 116 with a still-mixed
    policy, A at 256 while its train-best never moved TEST. Here the score that is
    patience'd is the caller's *selection score* (a deterministic fixed probe when
    ``SPECTRA_PROBE_EVERY`` is on), there is a minimum on-policy lifetime, and an optional
    probe-gated rewind to the elite weights. Pure bookkeeping — no torch — so it is unit
    testable; the trainer performs the actual reload.
    """

    def __init__(self, *, min_episodes: int, patience: int, rewind: bool,
                 rewind_patience: int, rewind_max: int, best_score: float = float("-inf"),
                 since_improvement: int = 0):
        self.min_episodes = int(min_episodes)
        self.patience = int(patience)
        self.rewind = bool(rewind)
        self.rewind_patience = int(rewind_patience)
        self.rewind_max = int(rewind_max)
        self.best_score = float(best_score)
        self.since_improvement = int(since_improvement)
        self.since_rewind_or_improvement = 0
        self.rewinds = 0
        self.has_elite = best_score > float("-inf")

    def observe(self, score, episodes_added: int) -> dict:
        """
        Register a selection score after ``episodes_added`` new on-policy episodes.
        ``score`` may be None (no new probe this round): only the counters advance.
        Returns ``{"new_best", "rewind", "since"}``.
        """
        new_best = False
        if score is not None and float(score) > self.best_score:
            self.best_score = float(score)
            self.since_improvement = 0
            self.since_rewind_or_improvement = 0
            self.has_elite = True
            new_best = True
        else:
            self.since_improvement += int(episodes_added)
            self.since_rewind_or_improvement += int(episodes_added)
        do_rewind = (self.rewind and self.has_elite and not new_best
                     and self.rewinds < self.rewind_max
                     and self.since_rewind_or_improvement >= self.rewind_patience)
        if do_rewind:
            self.rewinds += 1
            self.since_rewind_or_improvement = 0
        return {"new_best": new_best, "rewind": do_rewind, "since": self.since_improvement}

    def should_stop(self, episode_idx: int) -> bool:
        return int(episode_idx) >= self.min_episodes and self.since_improvement >= self.patience


INBUDGET_OVER_PENALTY = 1_000_000.0


def inbudget_checkpointing() -> bool:
    """
    Save ``latest_best_*`` on in-budget compression, not max discounted return.

    Default on for ``structural_unified`` / ``structural_prefer``. Cube-root / NEON training left
    ``latest_best`` on the identity policy because a 0-return never-prune beat
    every legal cut's large negative return (ledger Chain B 20945576).
    Override with ``SPECTRA_CHECKPOINT=return`` to restore discounted-return
    selection, or ``=inbudget_compression`` to force it on any mode.
    """
    raw = os.environ.get("SPECTRA_CHECKPOINT", "").strip().lower()
    if raw in ("return", "discounted", "neon", "0", "false", "off"):
        return False
    if raw in ("inbudget", "inbudget_compression", "1", "true", "yes", "on"):
        return True
    mode = os.environ.get("SPECTRA_REWARD_MODE", "neon").strip().lower()
    if mode in ("structural_unified", "structural_prefer"):
        return True
    # Cubes otherwise select identity as latest_best (20945576): 0-return never-prune
    # beats every legal cut's large negative return.
    if actor_skip_overbudget() and mode in (
            "neon", "structural", "structural_guard", "structural_band"):
        return True
    return False


def checkpoint_criterion() -> str:
    """
    ``return`` (discounted return), ``inbudget`` (F1 ρ-sum with a −1e6 over-budget penalty)
    or ``val_best`` (``1 − kept`` at the deepest in-band point of the episode — the
    training-side twin of the TRAJ ``val_best`` TEST point, bounded in [0, 1) and comparable
    across nets; ``SPECTRA_CHECKPOINT=val_best``). PPO batches average this per-episode
    score; the F1 penalty would let one over-budget episode sink a whole batch.
    """
    raw = os.environ.get("SPECTRA_CHECKPOINT", "").strip().lower()
    if raw in ("val_best", "valbest", "inband_kept"):
        return "val_best"
    return "inbudget" if inbudget_checkpointing() else "return"


def inbudget_checkpoint_score(rho_sum: float, overshoot_sum: float, any_over: bool) -> float:
    """Higher is better. Any over-budget episode loses to every in-budget one."""
    if any_over:
        return -INBUDGET_OVER_PENALTY - float(overshoot_sum)
    return float(rho_sum)


def stem_rows() -> int:
    """How many leading prunable rows are treated as stem (identity-only under fortify)."""
    return max(0, int(os.environ.get("SPECTRA_STEM_ROWS", "1")))


def min_width_for_prune() -> int:
    """Layers at or below this alive width may only take the identity action under fortify."""
    return max(1, int(os.environ.get("SPECTRA_MIN_WIDTH_FOR_PRUNE", "2")))


def entropy_anneal_horizon() -> int:
    return max(1, int(os.environ.get("SPECTRA_ENTROPY_ANNEAL_HORIZON", "100")))


def entropy_min_coef(base: float) -> float:
    raw = os.environ.get("SPECTRA_ENTROPY_MIN", "").strip()
    if raw:
        return float(raw)
    return 0.2 * base


def entropy_coef(episode_idx: int, warmup_len: int, base: float) -> float:
    """Constant base during warmup; linear decay toward entropy_min after warmup when fortify on."""
    if not fortify_enabled() or episode_idx < warmup_len:
        return base
    t = min((episode_idx - warmup_len) / float(entropy_anneal_horizon()), 1.0)
    lo = entropy_min_coef(base)
    return base + (lo - base) * t


STATE_SLACK_DIM = 2  # accuracy slack, pass progress


def fortify_token_dim() -> int:
    n = FORTIFY_TOKEN_DIM if fortify_enabled() else 0
    if budget_in_state():
        n += 1
    if state_slack():
        n += STATE_SLACK_DIM
    if state_groupcost():
        n += STATE_GROUPCOST_DIM
    return n


def build_fortify_features(
    num_layers: int,
    coupling_ids: torch.Tensor,
    topology: Sequence,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Representation fortification channels (already ~[0,1], not z-scored with raw moments):

      0 relative_depth in [0,1]
      1 is_stem (first SPECTRA_STEM_ROWS layers)
      2 is_coupled (shares coupling id with another layer — skip / group signal)
      3 width_norm = out_channels_or_features / max_width (from topology cols)
    """
    if num_layers == 0:
        return torch.zeros(0, FORTIFY_TOKEN_DIM, device=device, dtype=dtype)

    depths = torch.arange(num_layers, device=device, dtype=dtype) / max(num_layers - 1, 1)
    stem = torch.zeros(num_layers, device=device, dtype=dtype)
    stem[: min(stem_rows(), num_layers)] = 1.0

    coupled = torch.zeros(num_layers, device=device, dtype=dtype)
    if coupling_ids is not None and coupling_ids.numel() == num_layers:
        for cid in torch.unique(coupling_ids):
            idx = (coupling_ids == cid).nonzero(as_tuple=False).flatten()
            if idx.numel() > 1:
                coupled[idx] = 1.0

    widths = []
    for i in range(num_layers):
        row = topology[i] if i < len(topology) and topology[i] else [0.0] * 7
        kind = int(row[0]) if row else 0
        if kind == 2:  # Conv
            w = float(row[2]) if len(row) > 2 else 0.0
        elif kind == 1:  # Linear
            w = float(row[6]) if len(row) > 6 else 0.0
        else:
            w = float(row[2]) if len(row) > 2 else 0.0
        widths.append(max(w, 0.0))
    width_t = torch.tensor(widths, device=device, dtype=dtype)
    width_norm = width_t / width_t.max().clamp_min(1.0)

    return torch.stack([depths, stem, coupled, width_norm], dim=1)


def group_once_per_pass() -> bool:
    """
    A coupled channel group may be structurally cut at most once per pass (default off).

    The row walk visits every Conv/Linear once per pass, but a residual stream is *owned*
    by many rows: in a CIFAR ResNet stage every block's ``conv2`` (plus the shortcut conv)
    produces the same channel dimension, so a stage of 9 blocks exposes its stream to 9–10
    rate decisions per pass while each block-internal ``conv1`` is exposed once. Because
    ``target_width`` is applied to the *alive* width, repeated visits compound:
    Path 3 argmax on r56-w4 (job 20945568) took 0.9 on six consecutive owning rows and
    walked the stage-2 stream 8→7→6→5→4→3→2 (75 % of that stream) while the conv1s lost one
    channel each; that is the −25 pp cliff at 0.667 params.

    When this is on, ``NetworkEnv`` remembers the layer indices of every group it has
    structurally pruned in the current pass and ``legal_action_mask`` forces identity on
    later rows whose main layer belongs to such a group. Rate semantics are unchanged: a
    0.8 on a stream is one 20 % cut of that stream per pass. Frozen Path 3 replays (flag
    unset) are untouched. Pin with ``SPECTRA_GROUP_ONCE_PER_PASS=1``.
    """
    raw = os.environ.get("SPECTRA_GROUP_ONCE_PER_PASS", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def snapshot_baseline():
    """
    ``SPECTRA_SNAPSHOT_BASELINE=<float>``: when a new ``latest_best`` beats this score,
    the trainer freezes a copy under ``runs/<job>/snapshots/`` and drops a
    ``SNAPSHOT_READY.json`` marker so an ops watcher can fork ``eval_c10_thin_traj``
    without stopping the train. Unset (default) disables the hook. ``latest_best`` alone
    is not a scientific checkpoint: it is the episode with the best *train* score, which
    for the frozen s42 actor was warm-up episode 34 (a uniform-random walk).
    """
    raw = os.environ.get("SPECTRA_SNAPSHOT_BASELINE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def legal_action_mask(
    compression_rates: Dict[int, float],
    *,
    row_index: int,
    alive_count: int,
    device,
    force_identity: bool = False,
    group_param_fraction: Optional[float] = None,
) -> torch.Tensor:
    """
    Bool mask over discrete actions.

    Always on (library hygiene):
      * rates that cannot change width (target_width == alive) → illegal if rate < 1
      * layers already at 1 alive channel → identity only

    When fortify is enabled (default) additionally:
      * stem rows → only rate == 1.0
      * narrow layers (alive <= min_width) → only rate == 1.0

    ``force_identity`` is the caller's own reason to allow identity only (e.g. the row's
    coupled group was already cut this pass under ``SPECTRA_GROUP_ONCE_PER_PASS``).

    Budget menu (``SPECTRA_ACTION_MENU=budget``): legality is decided on the **mapped**
    keep rate (``effective_rates`` with ``group_param_fraction``), STOP is legal on every
    row including stems and locked groups, and a request the group cannot pay for is
    illegal (never rounded to a one-channel cut).
    """
    n = len(compression_rates)
    mask = torch.ones(n, dtype=torch.bool, device=device)

    force_identity = bool(force_identity) or alive_count <= 1
    if fortify_enabled():
        force_identity = force_identity or (row_index < stem_rows()) or (
            alive_count <= min_width_for_prune())

    budget = action_menu() == "budget"
    unknown_fraction = budget and group_param_fraction is None
    mapped = effective_rates(compression_rates,
                             0.0 if group_param_fraction is None else float(group_param_fraction),
                             group_width=None if group_param_fraction is None else int(alive_count))
    for idx, raw in compression_rates.items():
        keep, is_stop, feasible = mapped[idx]
        # STOP (budget menu, rate < 0) is legal on every row, including stems.
        if is_stop:
            mask[idx] = True
            continue
        if force_identity:
            mask[idx] = abs(float(keep) - 1.0) < 1e-9 and not (budget and 0.0 < float(raw) < 1.0)
            continue
        if unknown_fraction and 0.0 < float(raw) < 1.0:
            # Caller could not price the group: a budget cut stays legal; the env prices it.
            continue
        if not feasible:
            mask[idx] = False
            continue
        if float(keep) >= 1.0:
            continue
        # No-op compressions are pure credit-assignment noise (always illegal).
        if pruning.target_width(alive_count, float(keep)) >= alive_count:
            mask[idx] = False

    if not mask.any():
        for idx, rate in compression_rates.items():
            if abs(float(rate) - 1.0) < 1e-9:
                mask[idx] = True
                break
        if not mask.any():
            mask[0] = True
    return mask


def actor_skip_overbudget() -> bool:
    """
    Keep NEON cubes off the generic encoder when the walk left the τ-band.

    Empty-band nets (C100 residuals, skinny r56-w4) produce only −reduction³
    steps; those gradients teach "never prune" and fold latest_best to identity.
    Critic still sees the violation. Pin with ``SPECTRA_ACTOR_SKIP_OVERBUDGET=1``.
    """
    raw = os.environ.get("SPECTRA_ACTOR_SKIP_OVERBUDGET", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def policy_gradient_advantages(adv: torch.Tensor, step_over, skip_overbudget: bool):
    """
    Advantage tensor for the actor plus a kind flag: ``full``, ``masked``, ``skip``.

    Whole-episode skip is only for walks that were over-budget on *every* step
    (empty band). Mixed walks zero the violating steps so the in-budget prefix
    still teaches the intended arm, and the stop boundary is not a −reduction³
    blast through the shared encoder. Standardise on kept steps only.
    """
    adv = adv.detach()
    n = int(adv.shape[0])
    over = torch.zeros(n, dtype=torch.bool, device=adv.device)
    if skip_overbudget and step_over:
        flags = [bool(x) for x in list(step_over)[:n]]
        if flags:
            over[:len(flags)] = torch.tensor(flags, dtype=torch.bool, device=adv.device)
    over_b = over.view([-1] + [1] * (adv.ndim - 1))
    if skip_overbudget and n > 0 and bool(over.all().item()):
        return adv, "skip"
    if skip_overbudget and n > 0 and bool(over.any().item()):
        kept = adv[~over]
        if kept.numel() > 1:
            mu = kept.mean()
            sd = kept.std(unbiased=False)
            adv = (adv - mu) / (sd + 1e-8)
        adv = torch.where(over_b, torch.zeros_like(adv), adv)
        return adv, "masked"
    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return adv, "full"


def train_respects_size_floor() -> bool:
    """
    Apply the eval param/FLOP floor during *training* (identity-pad + look-ahead).

    Default off so already-running trains keep the old MDP. Paper TEST identity-pads
    at 0.70 params; without this the actor is trained on post-floor states eval never
    visits. Pin with ``SPECTRA_TRAIN_RESPECT_FLOOR=1``.
    """
    raw = os.environ.get("SPECTRA_TRAIN_RESPECT_FLOOR", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def eval_min_param_ratio() -> float:
    """Stop applying non-identity prune actions in eval once params fall below this fraction."""
    return float(os.environ.get("SPECTRA_EVAL_MIN_PARAM_RATIO", "0.70"))


def eval_trajectory_enabled() -> bool:
    """Unconstrained TEST walk: no identity-pad; quote a curve, not one stop.

    Phase A still uses look-ahead so the walk *labels* a ~0.70 hold. Phase B
    then applies the actor's blocked cut and continues to the end. After every
    real prune+FT the test loader is scored. Selection is on **val** Δacc
    (never on test). Default off so Path 3 identity-pad TESTs stay reproducible.
    Pin ``SPECTRA_EVAL_TRAJECTORY=1``.
    """
    raw = os.environ.get("SPECTRA_EVAL_TRAJECTORY", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def trajectory_release_floor(at_budget: bool, actor_idx: int, guarded_idx: int) -> bool:
    """True when trajectory Phase A should record the hold and enter Phase B."""
    if at_budget:
        return True
    return int(actor_idx) != int(guarded_idx)


def select_trajectory_points(points, *, min_param: float, tau_pp: float) -> dict:
    """Pick labeled TEST points from a recorded prune curve.

    ``points`` are dicts with ``param``, ``flop``, ``val_dacc_pp``, ``test_dacc_pp``.
    ``val_best`` is the *most compressed* in-budget val point (not the kindest
    Δacc). Test Δacc is reported at that step but never used to pick it.
    """
    pts = list(points or [])
    origin = pts[0] if pts else None
    terminal = pts[-1] if pts else None
    floor_cross = None
    for p in pts:
        if float(p["param"]) <= float(min_param) + 1e-12:
            floor_cross = p
            break
    above = [p for p in pts if float(p["param"]) + 1e-12 >= float(min_param)]
    floor_hold = None
    if above:
        floor_hold = min(
            above,
            key=lambda p: (float(p["param"]), float(p["flop"]), -float(p["val_dacc_pp"])),
        )
    in_tau = [p for p in pts if float(p["val_dacc_pp"]) + 1e-12 >= -float(tau_pp)]
    val_best = None
    if in_tau:
        val_best = min(
            in_tau,
            key=lambda p: (float(p["param"]), float(p["flop"]), -float(p["val_dacc_pp"])),
        )
    return {
        "origin": origin,
        "floor_hold": floor_hold,
        "floor_cross": floor_cross,
        "val_best": val_best,
        "terminal": terminal,
    }


def eval_min_flop_ratio() -> float:
    """
    Eval FLOP floor (fraction kept). ``0`` disables (default) so running jobs
    are unchanged. Typical A/B: ``0.70``. Residual 3x3 groups can sit at 70%
    params and ~55% FLOPs; this stops the walk on compute, not just size.
    """
    raw = os.environ.get("SPECTRA_EVAL_MIN_FLOP_RATIO", "0").strip()
    if not raw:
        return 0.0
    return max(0.0, float(raw))


def eval_lookahead_enabled() -> bool:
    """Refuse a prune whose previewed param/FLOP ratio would land below a floor.

    Explicit ``SPECTRA_EVAL_LOOKAHEAD=0`` reproduces the leaky Path 3 TESTs
    (C10-thin r20 printed ``params x0.600`` despite ``MIN_PARAM=0.70``: one
    0.8 group-cut overshoots, then identity-pad). Unset now means *on*
    whenever a param or FLOP floor is live — the FLOP path already did this.

    Trajectory TESTs own Phase-A look-ahead in the runner; this flag stays
    off there so identity-pad cannot also fire.
    """
    if eval_trajectory_enabled():
        return False
    raw = os.environ.get("SPECTRA_EVAL_LOOKAHEAD", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return eval_min_flop_ratio() > 0 or eval_min_param_ratio() > 0


def eval_prefer_param_per_flop() -> bool:
    """
    Eval overlay: when a FLOP floor is on, prefer cuts with high Δparams/ΔFLOPs
    so parameter count falls before the FLOP budget binds. Default off.
    """
    raw = os.environ.get("SPECTRA_EVAL_PREFER_PARAM_PER_FLOP", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def eval_at_size_floor(env) -> tuple:
    """
    ``(stop, reason)`` for eval identity-pad.

    reason is ``param``, ``flop``, or ``""``. FLOPs are not probed unless the
    FLOP floor is on.
    """
    min_param = eval_min_param_ratio()
    if float(env.param_ratio()) <= min_param:
        return True, "param"
    min_flop = eval_min_flop_ratio()
    if min_flop > 0 and float(env.flops_ratio()) <= min_flop:
        return True, "flop"
    return False, ""


def identity_action_index(compression_rates: Dict[int, float]) -> int:
    return next(
        (i for i, r in compression_rates.items() if abs(float(r) - 1.0) < 1e-9),
        0,
    )


def _rate_respects_eval_floors(env, rate: float, min_param: float, min_flop: float) -> bool:
    if abs(float(rate) - 1.0) < 1e-9:
        return True
    preview_both = getattr(env, "preview_ratios", None)
    if min_flop > 0 and callable(preview_both):
        param_r, flop_r = preview_both(rate)
        if float(param_r) + 1e-9 < min_param:
            return False
        return float(flop_r) + 1e-9 >= min_flop
    if float(env.preview_param_ratio(rate)) + 1e-9 < min_param:
        return False
    if min_flop > 0:
        preview_f = getattr(env, "preview_flops_ratio", None)
        if callable(preview_f) and float(preview_f(rate)) + 1e-9 < min_flop:
            return False
    return True


def action_respecting_param_floor(
    env,
    action: torch.Tensor,
    legal: torch.Tensor,
    compression_rates: Dict[int, float],
    min_ratio: float,
    device,
) -> torch.Tensor:
    """
    Keep the chosen action unless applying it would jump below the param floor
    (and the FLOP floor when ``SPECTRA_EVAL_MIN_FLOP_RATIO`` is on).

    Then pick the strongest legal prune whose dry-run still stays on every
    active floor; otherwise identity. Look-ahead is on whenever a param or
    FLOP floor is live (``SPECTRA_EVAL_LOOKAHEAD=0`` restores the leaky
    Path 3 walk). Without it, one residual 0.8 jumps past 0.70 params to
    ~0.60, or past 0.72 FLOPs to ~0.55.
    """
    identity = identity_action_index(compression_rates)
    min_flop = eval_min_flop_ratio()
    idx = int(action.item())
    rate = float(compression_rates[idx])
    if abs(rate - 1.0) < 1e-9:
        return action
    if _rate_respects_eval_floors(env, rate, min_ratio, min_flop):
        return action
    legal_idx = legal.nonzero(as_tuple=False).flatten().tolist()
    prune_opts = sorted(
        (
            (int(i), float(compression_rates[int(i)]))
            for i in legal_idx
            if abs(float(compression_rates[int(i)]) - 1.0) >= 1e-9
        ),
        key=lambda item: item[1],
    )
    for cand_i, cand_rate in prune_opts:
        if _rate_respects_eval_floors(env, cand_rate, min_ratio, min_flop):
            return torch.tensor([cand_i], device=device)
    return torch.tensor([identity], device=device)


def action_preferring_param_per_flop(
    env,
    action: torch.Tensor,
    legal: torch.Tensor,
    compression_rates: Dict[int, float],
    min_ratio: float,
    device,
) -> torch.Tensor:
    """
    Among floor-legal prune rates, pick the highest Δparams/ΔFLOPs.

    Skip (identity) if that ratio is below 1: the cut would spend more of the
    FLOP budget than of the parameter budget, which is how skinny ResNet-56
    stops at ~91% params when the FLOP floor is 0.70.
    No-op when the FLOP floor is off.
    """
    identity = identity_action_index(compression_rates)
    min_flop = eval_min_flop_ratio()
    if min_flop <= 0:
        return action
    cur_p = float(env.param_ratio())
    cur_f = float(env.flops_ratio())
    best_i = identity
    best_score = -1.0
    for i in legal.nonzero(as_tuple=False).flatten().tolist():
        rate = float(compression_rates[int(i)])
        if abs(rate - 1.0) < 1e-9:
            continue
        if not _rate_respects_eval_floors(env, rate, min_ratio, min_flop):
            continue
        preview = getattr(env, "preview_ratios", None)
        if callable(preview):
            prev_p, prev_f = preview(rate)
        else:
            prev_p = env.preview_param_ratio(rate)
            prev_f = env.preview_flops_ratio(rate)
        dp = cur_p - float(prev_p)
        df = cur_f - float(prev_f)
        if dp <= 1e-12:
            continue
        score = 1e9 if df <= 1e-12 else dp / df
        if score > best_score + 1e-12:
            best_score = score
            best_i = int(i)
    if best_i == identity or best_score < 1.0 - 1e-12:
        return torch.tensor([identity], device=device)
    return torch.tensor([best_i], device=device)


def eval_policy_name() -> str:
    """
    Eval action source: ``actor`` (learned DRL) or a rate heuristic (``l1`` / ``mild`` /
    ``random``). All four share the same prune + fine-tune + floor loop; only the *rate*
    picker changes. ``l1`` here means greedy strongest cut, not a different ranking.
    """
    return os.environ.get("SPECTRA_EVAL_POLICY", "actor").strip().lower() or "actor"


def eval_deterministic() -> bool:
    """
    Evaluate the frozen policy deterministically: ``argmax`` instead of ``sample``,
    and the actor/critic in ``eval()`` so encoder dropout is off.

    Default off so already-quoted TEST rows stay reproducible. Sampling a Categorical
    at test time reports a *mixture*, not the learned policy: at the entropy these
    agents converge to (~0.88 of ln 3) roughly a third of steps take a non-argmax rate,
    and one aggressive rate on a narrow layer is unrecoverable. See ledger §54.
    """
    raw = os.environ.get("SPECTRA_EVAL_DETERMINISTIC", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def state_align_next() -> bool:
    """
    Mark the *next* prunable row in the post-step state (default off).

    ``NetworkEnv.step`` historically encoded the layer that was just pruned, then
    incremented ``row_idx``. The actor therefore chose the next rate from the
    previous layer's marker and action-cost slots. Frozen actors were trained
    that way — leave this off when replaying them. New trains pin
    ``SPECTRA_STATE_ALIGN=next``.
    """
    raw = os.environ.get("SPECTRA_STATE_ALIGN", "prev").strip().lower()
    return raw in ("next", "1", "true", "yes", "on")


def reward_needs_flops() -> bool:
    """FLOP probes are only required for the ½ρ_w+½ρ_f unified / prefer mix."""
    mode = os.environ.get("SPECTRA_REWARD_MODE", "neon").strip().lower()
    return mode in ("structural_unified", "structural_prefer")


def skip_eval() -> bool:
    """Skip both in-job eval walks. Use when a chained child quotes ``eval_test``."""
    raw = os.environ.get("SPECTRA_SKIP_EVAL", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def skip_eval_train() -> bool:
    """Skip the duplicated train-loader prune+FT walk at eval.

    ``eval_train`` then ``eval_test`` are two full prune+FT walks from the
    pristine checkpoint. Both fine-tune on the train loader; they differ
    mainly in the final accuracy loader. Paper quotes ``eval_test`` only.
    Default off so running jobs and already-quoted walks stay unchanged.
    Skip-train sbatch profiles turn this on (``SPECTRA_SKIP_EVAL_TRAIN=1``).
    """
    raw = os.environ.get("SPECTRA_SKIP_EVAL_TRAIN", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def set_policy_eval_mode(*models) -> None:
    """Put actor/critic in inference mode (no-op unless deterministic eval is on)."""
    if not eval_deterministic():
        return
    for model in models:
        if model is not None:
            model.eval()


def policy_action(dist: Categorical, legal: torch.Tensor, *, device) -> torch.Tensor:
    """Action from the (masked) frozen policy: argmax under deterministic eval, else sample."""
    masked = apply_action_mask(dist, legal)
    if not eval_deterministic():
        return masked.sample()
    probs = masked.probs
    while probs.dim() > 1:
        probs = probs[0]
    return probs.argmax().reshape(1).to(device)


def heuristic_eval_action(
    legal: torch.Tensor,
    compression_rates: Dict[int, float],
    *,
    policy: str,
    device,
) -> torch.Tensor:
    """
    Pick a discrete compression rate without the actor.

    The environment already ranks filters by magnitude inside each chosen layer
    (Li et al. 2017 L1 by default; see ``src/pruning.py``). These baselines only choose
    *how hard* to prune the current layer — the same action menu, fine-tune budget,
    illegal-action mask, and parameter floor as DRL eval. That is the fair comparison
    for "did the agent learn a better *schedule* than a heuristic?"

    l1 / aggressive / uniform: strongest legal prune (lowest rate < 1).
        The name ``l1`` is historical: it is *greedy rate selection*, not a different
        ranking. Filters are still ranked by ``filter_importance``.
    mild: prefer 0.9 if legal, else strongest prune.
    random: uniform among legal non-identity actions.
    """
    identity = next(
        (i for i, r in compression_rates.items() if abs(float(r) - 1.0) < 1e-9),
        0,
    )
    legal_idx = legal.nonzero(as_tuple=False).flatten().tolist()
    prune_idx = [
        int(i) for i in legal_idx
        if abs(float(compression_rates[int(i)]) - 1.0) >= 1e-9
    ]
    if not prune_idx:
        return torch.tensor([identity], device=device)

    name = (policy or "l1").strip().lower()
    if name in ("random", "uniform_random"):
        pick = prune_idx[int(torch.randint(0, len(prune_idx), (1,)).item())]
        return torch.tensor([pick], device=device)
    if name in ("mild", "l1_mild"):
        nines = [i for i in prune_idx if abs(float(compression_rates[i]) - 0.9) < 1e-9]
        if nines:
            return torch.tensor([nines[0]], device=device)
    best = min(prune_idx, key=lambda i: float(compression_rates[i]))
    return torch.tensor([best], device=device)


def critic_huber_delta() -> float:
    """Smooth-L1 beta for critic; override with SPECTRA_CRITIC_HUBER_DELTA (0 disables)."""
    return float(os.environ.get("SPECTRA_CRITIC_HUBER_DELTA", "100.0"))


def apply_action_mask(dist: Categorical, legal: torch.Tensor) -> Categorical:
    """Zero illegal probs and renorm; used for both sample and log_prob."""
    if legal is None or bool(legal.all()):
        return dist
    probs = dist.probs.clone()
    legal_b = legal
    while legal_b.dim() < probs.dim():
        legal_b = legal_b.unsqueeze(0)
    probs = probs * legal_b.to(dtype=probs.dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return Categorical(probs=probs)


def sample_masked_action(
    dist: Categorical,
    legal: torch.Tensor,
    *,
    uniform: bool,
    device,
) -> torch.Tensor:
    """Sample from masked policy, or uniform over legal actions during warmup."""
    masked = apply_action_mask(dist, legal)
    if not uniform:
        return masked.sample()
    legal_idx = legal.nonzero(as_tuple=False).flatten()
    choice = legal_idx[torch.randint(0, legal_idx.numel(), (1,), device=device)]
    return choice
