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
from typing import Dict, Sequence

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


def fortify_token_dim() -> int:
    n = FORTIFY_TOKEN_DIM if fortify_enabled() else 0
    if budget_in_state():
        n += 1
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


def legal_action_mask(
    compression_rates: Dict[int, float],
    *,
    row_index: int,
    alive_count: int,
    device,
) -> torch.Tensor:
    """
    Bool mask over discrete actions.

    Always on (library hygiene):
      * rates that cannot change width (target_width == alive) → illegal if rate < 1
      * layers already at 1 alive channel → identity only

    When fortify is enabled (default) additionally:
      * stem rows → only rate == 1.0
      * narrow layers (alive <= min_width) → only rate == 1.0
    """
    n = len(compression_rates)
    mask = torch.ones(n, dtype=torch.bool, device=device)

    force_identity = alive_count <= 1
    if fortify_enabled():
        force_identity = force_identity or (row_index < stem_rows()) or (
            alive_count <= min_width_for_prune())

    for idx, rate in compression_rates.items():
        if force_identity:
            mask[idx] = abs(float(rate) - 1.0) < 1e-9
            continue
        if float(rate) >= 1.0:
            continue
        # No-op compressions are pure credit-assignment noise (always illegal).
        if pruning.target_width(alive_count, float(rate)) >= alive_count:
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
