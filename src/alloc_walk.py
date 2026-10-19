"""
Allocation-following eval walk (``SPECTRA_EVAL_POLICY=alloc``; no actor, no training).

A0 (ledger §201 / §204 / §205) cut every coupled group once to a sensitivity-weighted keep and
recovered for 40 epochs: at equal kept parameters that beat a uniform allocation on both ResNet-56
cells. This policy asks whether the lever survives the walk. Once per network, on the origin, it
fixes a target width for every group — ``SPECTRA_ALLOC_KIND=uniform`` (one fraction for all),
``sens`` (keep ∝ (s / median s)^α, s = the calibration-loss rise when the group alone is cut to
half, the measurement behind the v10 sensitivity channels) or ``inner`` (every coupled group with
more than one producer, i.e. a residual stream, held at full width and one fraction for the rest:
the structure the sens plans take on the thin ResNets, ledger §229) — with the scale bisected so that
cutting every group once to its target keeps ``target − SPECTRA_ALLOC_UNDERSHOOT`` of the
parameters. ``widths`` bisects nothing: every group takes the width ``SPECTRA_ALLOC_WIDTHS`` (JSON
``{module name: out channels}``, e.g. another method's pruned net) names for its producers, so a run
isolates that allocation under this walk's ranking and recovery; the fixed target then only says
where the walk stops. ``sample`` draws one plan the way the plan-as-action agent will: the weights
of ``SPECTRA_ALLOC_SAMPLE_AROUND`` (sens or uniform) times exp(σ ε) per group, ε ~ N(0, 1) seeded by
``SPECTRA_ALLOC_SAMPLE_SEED``, σ = ``SPECTRA_ALLOC_SAMPLE_SIGMA``, then bisected like ``sens``.
``agent`` takes the mean plan of the frozen plan agent at ``SPECTRA_PLAN_AGENT`` (``src/plan_agent.py``).
``agent_sample`` draws one plan around that mean as the agent samples in training, z = μ + σ ε
with ε ~ N(0, I) seeded by ``SPECTRA_ALLOC_SAMPLE_SEED`` and σ = ``SPECTRA_ALLOC_SAMPLE_SIGMA``,
decoded like the mean (D-PROXY-2).
``randbest`` (v23) is the per-network random-search baseline of Li et al. (CVPR 2022, "Revisiting Random Channel
Pruning") in the agent's own action space, set beside the frozen agent: N = ``SPECTRA_ALLOC_RANDBEST_N`` (100) plans
z_g ~ N(0, σ²) per group, σ = ``SPECTRA_ALLOC_RANDBEST_SIGMA`` (1.0), from a CPU generator seeded by
``SPECTRA_ALLOC_SAMPLE_SEED`` and the run's seed (``randbest_seed``), each decoded as ``agent`` decodes the mean plan
(``plan_agent.decode`` at the same target, k_min, width floor and budget) and scored on the val half with the proxy
``SPECTRA_ALLOC_RANDBEST_PROXY`` (no default: cut, bn<N>, fs<S> or ft<E>, the trainer's reward through
``plan_proxies.PlanScorer``, on the batches ``SPECTRA_EVAL_PROXY_SEED`` draws). The best val (ties to the lower draw)
is the plan the walk then follows exactly as it follows the agent's. The TEST half is never read, and the global RNG is
put back after the search, so the walk and the final fine-tune draw as they do under ``agent``.
``fallback`` (v27) plans twice on the origin, with the frozen agent exactly as ``agent`` does and with a hand rule,
``SPECTRA_ALLOC_FALLBACK_PRIOR`` (``sens`` or ``sens_cost``, no default; "the prior") exactly as that rule's own row does. It
scores both plans on the val half with ``SPECTRA_ALLOC_FALLBACK_PROXY`` (a comma-separated list of cut, bn<N>, fs<S> or
ft<E>, no default; ``plan_proxies.PlanScorer`` on the batches ``SPECTRA_EVAL_PROXY_SEED`` draws) and follows the agent's plan
unless the prior's scores higher on the FIRST proxy by more than ``SPECTRA_ALLOC_FALLBACK_MARGIN`` (val accuracy as a
fraction, default 0.0; a tie goes to the agent). Every proxy is scored and reported, the first one decides. The contract is
exactness: each plan is made from the same global RNG state and the same origin weights as its own row makes it (the
origin's weights are checked after each plan and after the scoring), the RNG is then put back to the state the picked arm
alone leaves, and the picked plan is returned unchanged (``info["kind"]`` stays the arm's own, one key ``fallback`` is
added), so the walk and the final fine-tune run as under that arm alone and, under ``SPECTRA_EVAL_FT_DETERMINISTIC=1``, a
fallback row equals the picked arm's row seed for seed. The TEST half is never read. Any effect ``plan_for_env`` has on
the env itself stays even when the prior is picked, so the exactness against the prior's own row is a check registered on
the GPU, not a guarantee of this code.
``SPECTRA_ALLOC_BUDGET=flops`` (default params) measures the plan's kept fraction, its target and the stall
check in MACs (``utils.calc_flops``) instead of parameters, the T0-F variant.
Each decision then plays the legal cut whose resulting width is closest to its group's
target (ties to the milder cut, identity once it is there). Group-once, the recovery and the
fixed-target landing are the walk's own, so a row differs from greedy / mild only in which groups
keep being cut. The undershoot makes the walk cross the target before every group has arrived;
should it still stall above the target for a whole pass, it falls back to the strongest legal cut.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import statistics
import time

import numpy as np
import torch

import src.channel_groups as channel_groups
import src.fortify as fortify
import src.group_sensitivity as group_sensitivity
import src.pruning as pruning
import src.utils as utils
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows

KINDS = ("uniform", "sens", "inner", "widths", "sample", "agent", "agent_sample", "sens_cost", "randbest", "fallback")
FALLBACK_PRIORS = ("sens", "sens_cost")
SAMPLE_AROUND = ("sens", "uniform")
BUDGETS = ("params", "flops")


def kind() -> str:
    """``SPECTRA_ALLOC_KIND`` (sens): the allocation the walk follows."""
    name = os.environ.get("SPECTRA_ALLOC_KIND", "sens").strip().lower() or "sens"
    if name not in KINDS:
        raise ValueError(f"SPECTRA_ALLOC_KIND={name!r}: expected one of {KINDS}")
    return name


def budget() -> str:
    """``SPECTRA_ALLOC_BUDGET`` (params): what the plan's kept fraction and the walk's target are measured in."""
    name = os.environ.get("SPECTRA_ALLOC_BUDGET", "params").strip().lower() or "params"
    if name not in BUDGETS:
        raise ValueError(f"SPECTRA_ALLOC_BUDGET={name!r}: expected one of {BUDGETS}")
    return name


def cut_ranking(ranking):
    """
    The ranking ``env.step`` takes for a cut the walk plays: ``SPECTRA_CUT_RANK`` (``fortify.cut_rank``; ``random`` is
    the control that keeps the walk's widths and picks the surviving channels at random) when it is set, else
    ``ranking`` unchanged. The eval loop applies it to its own ``env.step`` calls only, under
    ``SPECTRA_EVAL_POLICY=alloc``: ``plan_targets`` / ``cut_to`` / ``group_sensitivity`` and the plan agent cut with an
    explicit ``l1``, so the plan (and the agent's tokens, which carry sensitivities measured with the ranking in force)
    never see it.
    """
    mode = fortify.cut_rank()
    return mode if mode else ranking


def alpha() -> float:
    """``SPECTRA_ALLOC_ALPHA`` (0.5): sensitivity power of ``sens``, A0's default."""
    return float(os.environ.get("SPECTRA_ALLOC_ALPHA", "0.5"))


def undershoot() -> float:
    """``SPECTRA_ALLOC_UNDERSHOOT`` (0.02): the plan keeps this much less than the walk's target."""
    return max(0.0, float(os.environ.get("SPECTRA_ALLOC_UNDERSHOOT", "0.02")))


def min_keep() -> float:
    """``SPECTRA_ALLOC_MIN_KEEP`` (0.1): no group's planned keep goes below this, A0's floor."""
    return min(1.0, max(0.01, float(os.environ.get("SPECTRA_ALLOC_MIN_KEEP", "0.1"))))


def sample_around() -> str:
    """``SPECTRA_ALLOC_SAMPLE_AROUND`` (sens): the plan whose weights ``sample`` perturbs."""
    name = os.environ.get("SPECTRA_ALLOC_SAMPLE_AROUND", "sens").strip().lower() or "sens"
    if name not in SAMPLE_AROUND:
        raise ValueError(f"SPECTRA_ALLOC_SAMPLE_AROUND={name!r}: expected one of {SAMPLE_AROUND}")
    return name


def sample_sigma() -> float:
    """``SPECTRA_ALLOC_SAMPLE_SIGMA`` (0.5): std of the per-group noise, on the log-weights of ``sample`` and on
    the scores of ``agent_sample``."""
    return max(0.0, float(os.environ.get("SPECTRA_ALLOC_SAMPLE_SIGMA", "0.5")))


def sample_seed() -> int:
    """``SPECTRA_ALLOC_SAMPLE_SEED`` (0): seed of the noise of ``sample`` and ``agent_sample``."""
    return int(os.environ.get("SPECTRA_ALLOC_SAMPLE_SEED", "0"))


def sample_noise(rows, sigma, seed):
    """``{row: exp(sigma * eps)}``, eps ~ N(0, 1) drawn in ``rows`` order from ``random.Random(seed)``."""
    rng = random.Random(int(seed))
    return {row: math.exp(float(sigma) * rng.gauss(0.0, 1.0)) for row in rows}


def agent_path() -> str:
    """``SPECTRA_PLAN_AGENT``: the plan-agent checkpoint ``agent`` / ``agent_sample`` follows
    (``src/plan_agent.py``)."""
    path = os.environ.get("SPECTRA_PLAN_AGENT", "").strip()
    if not path:
        raise ValueError("SPECTRA_ALLOC_KIND=agent or agent_sample needs SPECTRA_PLAN_AGENT=<plan-agent checkpoint>")
    return path


def randbest_n() -> int:
    """``SPECTRA_ALLOC_RANDBEST_N`` (100): how many random plans ``randbest`` draws and scores."""
    n = int(os.environ.get("SPECTRA_ALLOC_RANDBEST_N", "100"))
    if n < 1:
        raise ValueError(f"SPECTRA_ALLOC_RANDBEST_N={n}: need at least one draw")
    return n


def randbest_sigma() -> float:
    """``SPECTRA_ALLOC_RANDBEST_SIGMA`` (1.0): std of each draw's per-group scores, z_g ~ N(0, σ²)."""
    return max(0.0, float(os.environ.get("SPECTRA_ALLOC_RANDBEST_SIGMA", "1.0")))


def randbest_proxy() -> str:
    """``SPECTRA_ALLOC_RANDBEST_PROXY`` (no default): the proxy every draw is scored with, one of the plan trainer's
    (``plan_agent.parse_proxy``: cut, bn<N>, fs<S>, ft<E>), the one the agent of record trained on."""
    name = os.environ.get("SPECTRA_ALLOC_RANDBEST_PROXY", "").strip().lower()
    if not name:
        raise ValueError("SPECTRA_ALLOC_KIND=randbest needs SPECTRA_ALLOC_RANDBEST_PROXY=<cut | bn<N> | fs<S> | ft<E>>, "
                         "the proxy the plan agent trained on (SPECTRA_PLAN_PROXY)")
    from src import plan_agent
    plan_agent.parse_proxy(name)
    return name


def run_seed(env) -> int:
    """The run's seed (``--seed``, ``SPECTRA_SEED`` in the submit scripts): ``env.conf.seed`` when the env carries one,
    else ``SPECTRA_SEED``, else 0."""
    seed = getattr(getattr(env, "conf", None), "seed", None)
    if seed is None:
        seed = os.environ.get("SPECTRA_SEED", "").strip() or 0
    return int(seed)


def randbest_seed(sample: int, run: int) -> int:
    """One generator seed from ``SPECTRA_ALLOC_SAMPLE_SEED`` and the run's seed: the same pair, the same N plans."""
    return int(sample) * 100003 + int(run)


def randbest_draws(n, n_groups, sigma, seed):
    """``[[z_g]]``: the scores of ``n`` draws, z_g ~ N(0, σ²) independently per group in plan order, from a CPU
    generator seeded by ``seed`` (float64, as ``plan_agent.plan_for_env`` draws ``agent_sample``'s ε); the global RNG
    is untouched. One ``randn`` call per draw, so draw i is the same under any N (a longer search extends a shorter
    one; one call for all N would reorder the stream past torch's 16-element vectorised path)."""
    generator = torch.Generator().manual_seed(int(seed))
    return [(float(sigma) * torch.randn(int(n_groups), generator=generator, dtype=torch.float64)).tolist()
            for _ in range(int(n))]


def fallback_prior() -> str:
    """``SPECTRA_ALLOC_FALLBACK_PRIOR`` (no default): the hand rule ``fallback`` plans with beside the agent, one of
    ``FALLBACK_PRIORS`` (``sens``, ``sens_cost``)."""
    name = os.environ.get("SPECTRA_ALLOC_FALLBACK_PRIOR", "").strip().lower()
    if name not in FALLBACK_PRIORS:
        raise ValueError(f"SPECTRA_ALLOC_KIND=fallback needs SPECTRA_ALLOC_FALLBACK_PRIOR set to one of "
                         f"{FALLBACK_PRIORS}, got {name!r}")
    return name


def fallback_proxies() -> list:
    """``SPECTRA_ALLOC_FALLBACK_PROXY`` (no default): the proxies ``fallback`` scores both plans with, a comma-separated
    list in order (blanks stripped, lower-cased, empty items dropped, no duplicates), each one of the plan trainer's
    (``plan_agent.parse_proxy``: cut, bn<N>, fs<S>, ft<E>). The first decides the pick; every one is scored and reported."""
    raw = os.environ.get("SPECTRA_ALLOC_FALLBACK_PROXY", "")
    names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    if not names:
        raise ValueError("SPECTRA_ALLOC_KIND=fallback needs SPECTRA_ALLOC_FALLBACK_PROXY=<cut | bn<N> | fs<S> | ft<E>"
                         "[,...]>, the first one decides the pick")
    from src import plan_agent
    for name in names:
        plan_agent.parse_proxy(name)
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"SPECTRA_ALLOC_FALLBACK_PROXY={raw!r}: duplicate proxies {repeated}")
    return names


def fallback_margin() -> float:
    """``SPECTRA_ALLOC_FALLBACK_MARGIN`` (0.0): how far above the agent's val accuracy (a fraction, the scores
    ``PlanScorer`` returns) the prior's plan must score for ``fallback`` to walk the prior's plan."""
    margin = float(os.environ.get("SPECTRA_ALLOC_FALLBACK_MARGIN", "0.0"))
    if not margin >= 0.0:
        raise ValueError(f"SPECTRA_ALLOC_FALLBACK_MARGIN={margin}: need a margin of at least 0")
    return margin


def widths_table() -> dict:
    """``SPECTRA_ALLOC_WIDTHS``: the JSON ``{module name: out channels}`` that ``widths`` copies."""
    path = os.environ.get("SPECTRA_ALLOC_WIDTHS", "").strip()
    if not path:
        raise ValueError("SPECTRA_ALLOC_KIND=widths needs SPECTRA_ALLOC_WIDTHS=<json {module name: out channels}>")
    with open(path, encoding="utf-8") as fh:
        return {str(name): int(width) for name, width in json.load(fh).items()}


def copied_keeps(model, plan, table):
    """
    ``({row: keep}, unnamed)``: a group keeps the narrowest width ``table`` names for its producers, over
    its own width. A group none of whose producers is named stays whole and is counted in ``unnamed``.
    """
    names = {id(module): name for name, module in model.named_modules()}
    keeps, unnamed = {}, 0
    for group, row in plan:
        named = [table[names[id(p)]] for p in group.producers if names.get(id(p)) in table]
        unnamed += not named
        keeps[row] = min(1.0, min(named) / float(group.width)) if named else 1.0
    return keeps, unnamed


def weights(kind_name, sens, a=0.5, cost=None):
    """Relative keep per row before scaling: 1 (uniform, inner) or (s / median)^α, s floored at 5 % of the median.

    ``sens_cost`` (v16) uses (v / median v)^α with v = s / c, s and the cost c (saved by the same half cut) each
    floored at 5 % of its median; ``cost`` is its ``{row: saved}`` and must name the rows of ``sens``.
    """
    keys = list(sens)
    if kind_name in ("uniform", "inner"):
        return {key: 1.0 for key in keys}
    if kind_name == "sens_cost":
        if cost is None or set(cost) != set(keys):
            raise ValueError("weights('sens_cost') needs cost={row: saved} naming the same rows as sens")

        def floored(raw):
            positive = [max(0.0, float(raw[key])) for key in keys]
            floor = max(1e-6, 0.05 * statistics.median(positive))
            return {key: max(float(raw[key]), floor) for key in keys}

        s, c = floored(sens), floored(cost)
        v = {key: s[key] / c[key] for key in keys}
        mid = statistics.median(v.values())
        return {key: (v[key] / mid) ** a for key in keys}
    positive = [max(0.0, float(sens[key])) for key in keys]
    floor = max(1e-6, 0.05 * statistics.median(positive))
    s = {key: max(float(sens[key]), floor) for key in keys}
    mid = statistics.median(s.values())
    return {key: (s[key] / mid) ** a for key in keys}


def cut_to(model, plan, keeps, input_shape):
    """A copy of ``model`` with every planned group cut once (L1) to its keep; 1 leaves it alone."""
    from src.NetworkEnv import prune_current_model
    mwr = ModelWithRows(copy.deepcopy(model))
    for _group, row in plan:
        rate = float(keeps.get(row, 1.0))
        if rate >= 1.0 - 1e-9:
            continue
        mwr = prune_current_model(mwr, rate, row, quiet=True, record=False, input_shape=input_shape,
                                  importance="l1")
        mwr = ModelWithRows(mwr.model)
    return mwr.model


def group_widths(model, rows):
    """``{row: width}`` of the coupled group each row's layer produces."""
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    out = {}
    for row in rows:
        group = channel_groups.group_of(groups, mwr.all_layers[mwr.row_to_main_layer[row]])
        if group is not None:
            out[row] = int(group.width)
    return out


def plan_targets(model, batches, input_shape, kind_name, target, a=0.5, keep_floor=0.1, iters=16, tol=0.003,
                 budget="params", min_width=1):
    """
    ``(widths, info)``: ``widths[row]`` is the target width of the group whose first walk row is
    ``row``. Keeps are clip(c · weight, keep_floor, 1), or 1 for a group ``inner`` holds; c is
    bisected until the one-shot cut keeps ``target`` of the parameters, or of the MACs under
    ``budget="flops"`` (widths are integers, so the closest c found). ``widths`` copies its table
    instead and ignores ``target``. ``min_width`` above 1 (``fortify.plan_min_width``) lifts every
    group's keep floor to ``min_width`` / its origin width, the walk's legal floor.
    """
    size = (lambda m: utils.calc_flops(m, input_shape)) if budget == "flops" else utils.calc_num_parameters
    plan = group_sensitivity.group_plan(ModelWithRows(model))
    rows = [row for _group, row in plan]
    if kind_name == "widths":
        keeps, unnamed = copied_keeps(model, plan, widths_table())
        cut = cut_to(model, plan, keeps, input_shape)
        frac = size(cut) / size(model)
        widths = group_widths(cut, rows)
        del cut
        return widths, {"kind": kind_name, "alpha": float(a), "target": float(target), "kept": float(frac),
                        "keeps": keeps, "origin_widths": group_widths(model, rows),
                        "sens": {row: 1.0 for row in rows}, "held": 0, "unnamed": unnamed}
    base_kind = sample_around() if kind_name == "sample" else kind_name
    cost = None
    if base_kind == "sens_cost":
        raw = {}
        sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, input_shape, costs=raw)
        cost = {row: raw[row][1] if budget == "flops" else raw[row][0] for row in rows}
        w = weights(base_kind, sens, a, cost=cost)
    else:
        if base_kind == "sens":
            sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, input_shape)
        else:
            sens = {row: 1.0 for row in rows}
        w = weights(base_kind, sens, a)
    sample = None
    if kind_name == "sample":
        sample = {"around": base_kind, "sigma": sample_sigma(), "seed": sample_seed()}
        noise = sample_noise(rows, sample["sigma"], sample["seed"])
        w = {row: w[row] * noise[row] for row in rows}
    held = {row for group, row in plan if kind_name == "inner" and len(group.producers) > 1}
    floor = {row: keep_floor for row in rows}
    if int(min_width) > 1:
        for (group, row) in plan:
            floor[row] = max(keep_floor, min(1.0, int(min_width) / float(group.width)))
    size0 = size(model)
    lo, hi = 0.0, 1.0 / min(w.values())
    best = None
    for _ in range(iters):
        c = 0.5 * (lo + hi)
        keeps = {row: 1.0 if row in held else min(1.0, max(floor[row], c * w[row])) for row in rows}
        cut = cut_to(model, plan, keeps, input_shape)
        frac = size(cut) / size0
        if best is None or abs(frac - target) < abs(best[1] - target):
            best = (keeps, frac, group_widths(cut, rows))
        del cut
        if abs(frac - target) <= tol:
            break
        if frac > target:
            hi = c
        else:
            lo = c
    keeps, frac, widths = best
    info = {"kind": kind_name, "alpha": float(a), "target": float(target), "kept": float(frac),
            "keeps": keeps, "origin_widths": group_widths(model, rows), "sens": sens, "held": len(held)}
    if cost is not None:
        info["cost"] = cost
        info["cost_budget"] = budget
    if sample is not None:
        info["sample"] = sample
    return widths, info


def choose(width, target, rates, legal_idx, identity):
    """The legal action whose resulting group width is closest to ``target``; ties go to the milder cut."""
    best, best_key = identity, (abs(width - target), -1.0)
    for idx in legal_idx:
        rate = float(rates[int(idx)])
        if rate >= 1.0 - 1e-9:
            continue
        kept = pruning.target_width(width, rate)
        key = (abs(kept - target), -rate)
        if key < best_key:
            best, best_key = int(idx), key
    return best


def flops_walk_target(env) -> float:
    """The kept MACs a ``SPECTRA_ALLOC_BUDGET=flops`` walk heads for: the episode's fixed target when that is measured
    in FLOPs (``fortify.fixed_target_metric``), else the flop ``SPECTRA_EVAL_SIZE_MATCH``, else the deepest flop
    ``SPECTRA_EVAL_SIZE_POINTS`` entry, else 0.6."""
    if env.target_keep is not None and fortify.fixed_target_metric() == "flop":
        return float(env.target_keep)
    match = fortify.eval_size_match()
    if match is not None and match[0] == "flop":
        return float(match[1])
    points = [t for kind_name, t in fortify.eval_size_points() if kind_name == "flop"]
    return float(min(points)) if points else 0.6


# ------------------------------------------------------------------ v18 grid rounding (SPECTRA_ALLOC_GRID_ROUND)

GRID_BAND = 0.01  # a rounded plan keeps at most the walk target and, where the grid allows, no more than this less


def walk_cut(width, rate):
    """The width ``NetworkEnv.step`` leaves when a ``width``-wide group is cut at ``rate``: the width ladder's keep
    (``fortify.effective_rates``, as ``NetworkEnv._ladder_keep_rate``) when the ladder is on, then
    ``pruning.target_width``. A rate the ladder cannot pay for is an identity step."""
    keep = float(rate)
    if fortify.width_ladder_max() > 0 and 0.0 < keep < 1.0:
        keep, _stop, feasible = fortify.effective_rates({0: keep}, 0.0, group_width=int(width))[0]
        if not feasible:
            return int(width)
    return pruning.target_width(int(width), float(keep))


def walk_to(w0, target, rates, legal_at, max_cuts):
    """``(width, cuts)`` where the walk's ``choose`` chain from ``w0`` aimed at ``target`` stops: ``legal_at(width)``
    gives the legal action indices at that width, ``walk_cut`` the cut, and at most ``max_cuts`` cuts are made."""
    identity = next((i for i, r in rates.items() if abs(float(r) - 1.0) < 1e-9), 0)
    width, cuts = int(w0), 0
    while cuts < int(max_cuts):
        pick = choose(width, int(target), rates, legal_at(width), identity)
        if pick == identity:
            break
        new = walk_cut(width, rates[pick])
        if new >= width:
            break
        width, cuts = new, cuts + 1
    return width, cuts


def grid_widths(w0, rates, legal_at, max_cuts):
    """``{width: cuts}``: every width the walk lands on exactly from ``w0`` (``walk_to`` aimed at it stops there) and
    the cuts it takes; ``w0`` itself at none."""
    out = {int(w0): 0}
    for target in range(int(w0) - 1, 0, -1):
        width, cuts = walk_to(w0, target, rates, legal_at, max_cuts)
        if width == target:
            out[target] = cuts
    return out


def walk_grids(model, plan, rates, passes, groups=None, aims=None):
    """
    ``{row: {"grid": {width: cuts}, "cap": max cuts, "walk": width or None}}`` for each planned group of ``model``
    (the origin): the widths the walk reaches exactly (``grid_widths``) under the legal mask the env builds
    (``fortify.legal_action_mask``: identity only at alive <= 1, on stem rows and at or below ``min_width_for_prune``
    under fortify, no-op rates illegal; every row identity under ``SPECTRA_PROTECT_STREAMS`` for a residual stream), the
    env's cut (``walk_cut``) and the cuts ``passes`` passes allow (one per pass under ``SPECTRA_GROUP_ONCE_PER_PASS``,
    else one per non-stem row producing the group). ``walk`` is where the walk aimed at ``aims[row]`` stops. Eval
    rollback locks, masked cuts and the eval size floors act at walk time only (``_grid_watch`` reports them).
    """
    mwr = ModelWithRows(model)
    if groups is None:
        groups = channel_groups.build_channel_groups(mwr.model) or []
    first = {id(group): row for group, row in plan}
    visits = {row: [] for _group, row in plan}
    for row in sorted(mwr.row_to_main_layer)[:-1]:
        group = channel_groups.group_of(groups, mwr.all_layers[mwr.row_to_main_layer[row]])
        if group is not None and id(group) in first:
            visits[first[id(group)]].append(row)
    stem = fortify.stem_rows() if fortify.fortify_enabled() else 0
    masks = {}

    def legal_for(row_index):
        def at(width):
            key = (row_index, int(width))
            if key not in masks:
                mask = fortify.legal_action_mask(rates, row_index=row_index, alive_count=int(width), device="cpu")
                masks[key] = [int(i) for i in mask.nonzero(as_tuple=False).flatten().tolist()]
            return masks[key]
        return at

    out = {}
    for group, row in plan:
        protected = fortify.protect_streams() and fortify.is_residual_stream(group)
        cuttable = [] if protected else [r for r in visits[row] if r >= stem]
        cap =int(passes) * (1 if fortify.group_once_per_pass() else len(cuttable)) if cuttable else 0
        legal_at = legal_for(cuttable[0] if cuttable else row)
        w0 = int(group.width)
        aim = None if aims is None else aims.get(row)
        out[row] = {"grid": grid_widths(w0, rates, legal_at, cap), "cap": cap,
                    "walk": None if aim is None else walk_to(w0, aim, rates, legal_at, cap)[0]}
    return out


def grid_priority(info, rows):
    """``{row: score}`` the grid rounding steps groups by (lowest first down, highest first up): the decoder's weight
    for the rules (``weights`` of ``info``'s sensitivities, with its cost under ``sens_cost`` and its noise under
    ``sample``), the score z for the agent (log prior weight + z for a T2 residual plan); None for one weight on every group (uniform, inner, widths), which steps
    by closeness to the plan instead."""
    kind_name = info.get("kind")
    sens = info.get("sens") or {}
    if kind_name in ("agent", "agent_sample", "randbest"):
        z = {row: float(sens.get(row, 0.0)) for row in rows}
        prior = (info.get("prior") or {}).get("weights") if info.get("residual") else None
        if prior:  # T2: a residual plan ranks groups by log(prior weight) + z (plan_agent.residual_decode)
            zmax = float((info.get("prior") or {}).get("zmax", 0.0))
            if zmax > 0:  # v20 trust region: the decoder exponentiated z_eff = zmax · tanh(z / zmax), so rank by that
                from src import plan_agent
                z = {row: plan_agent.trust_score(z[row], zmax) for row in rows}
            return {row: math.log(max(1e-12, float(prior.get(row, 1.0)))) + z[row] for row in rows}
        return z
    sample = info.get("sample") if kind_name == "sample" else None
    base = sample["around"] if sample else kind_name
    if base not in ("sens", "sens_cost", "sample") and sample is None:
        return None
    keyed = {row: float(sens.get(row, 1.0)) for row in rows}
    cost = {row: float(info["cost"][row]) for row in rows} if base == "sens_cost" else None
    w = weights(base, keyed, float(info.get("alpha", alpha())), cost=cost)
    if sample:
        noise = sample_noise(rows, sample["sigma"], sample["seed"])
        w = {row: w[row] * noise[row] for row in rows}
    return w


def grid_round_widths(planned, grids, kept_of, target, priority=None, fixed=(), floors=None, band=GRID_BAND):
    """
    ``(widths, report)``: the plan ``planned`` (``{row: width}``) on the walk's grid (``grids[row]``: the widths it
    lands on exactly). Each group starts at the grid width nearest its plan (ties to the wider); then, while the kept
    size ``kept_of(widths)`` is above ``target``, one group steps down to its next grid width, and once it is at or
    below, single groups step back up while it is more than ``band`` below, never above ``target``. A step goes by
    ``priority`` (lowest score first down, highest first up, each group once per round, ties by row order) or, with
    ``priority`` None, to the group the step leaves closest to its plan relative to it. ``fixed`` rows keep their
    planned width; no group steps below ``floors[row]`` (or its start, when that is lower). ``report``: kept, the
    starting widths and the steps down / up.
    """
    rows = list(planned)
    order = {row: k for k, row in enumerate(rows)}
    fixed, floors = set(fixed), floors or {}
    ladder = {row: sorted(grids.get(row) or [planned[row]]) for row in rows}
    widths = {row: int(planned[row]) if row in fixed
              else min(ladder[row], key=lambda w, p=planned[row]: (abs(w - p), -w)) for row in rows}
    snapped = dict(widths)
    lo = {row: min(widths[row], int(floors.get(row, 1))) for row in rows}

    def neighbour(row, sign):
        if sign < 0:
            below = [w for w in ladder[row] if lo[row] <= w < widths[row]]
            return max(below) if below else None
        above = [w for w in ladder[row] if w > widths[row]]
        return min(above) if above else None

    def key(row, width, steps, sign):
        if priority is None:
            return abs(width - planned[row]) / float(max(1, planned[row])), order[row]
        return steps[row], sign * float(priority[row]), order[row]

    kept, down, up = float(kept_of(widths)), {row: 0 for row in rows}, {row: 0 for row in rows}
    while kept > target + 1e-9:
        moves = [(key(row, w, down, 1), row, w) for row in rows if row not in fixed
                 for w in [neighbour(row, -1)] if w is not None]
        if not moves:
            break
        _key, row, w = min(moves)
        widths[row], down[row] = w, down[row] + 1
        kept = float(kept_of(widths))
    while kept < target - band - 1e-9:
        moves = sorted((key(row, w, up, -1), row, w) for row in rows if row not in fixed
                       for w in [neighbour(row, 1)] if w is not None)
        for _key, row, w in moves:
            trial = dict(widths)
            trial[row] = w
            k = float(kept_of(trial))
            if k <= target + 1e-9:
                widths, kept, up[row] = trial, k, up[row] + 1
                break
        else:
            break
    return widths, {"kept": kept, "snapped": snapped, "down": sum(down.values()), "up": sum(up.values())}


def _grid_round_state(env, net, entry, model, info, walk_target, flops):
    """``SPECTRA_ALLOC_GRID_ROUND``: ``entry``'s plan on the walk's grid (``walk_grids``, ``grid_round_widths``) on the
    budget's cost model (``plan_agent.ParamModel`` / ``FlopModel``), measured on the origin ``model``; logged, recorded,
    and kept in ``entry["grid"]`` for ``_grid_watch``. Inner's held streams stay whole, as does every group a widths
    table leaves at its origin width; no group steps below the plan floors (``min_keep``,
    ``fortify.plan_min_width``)."""
    from src import plan_agent
    if fortify.action_menu() == "budget":
        utils.print_flush("[alloc] WARNING grid round skipped: SPECTRA_ACTION_MENU=budget prices each cut on the net "
                          "as the walk finds it, which no offline grid follows")
        return
    rates = env.conf.compression_rates_dict
    passes = max(1, int(getattr(env.conf, "passes", 1) or 1))
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    plan = group_sensitivity.group_plan(mwr, groups)
    widths = entry["widths"]
    planned = {row: int(widths[row]) for _group, row in plan if row in widths}
    cost = (plan_agent.FlopModel(model, plan, env._input_shape(), env.conf.device) if flops
            else plan_agent.ParamModel(model, plan))
    walks = walk_grids(model, plan, rates, passes, groups, aims=planned)
    held = {row for group, row in plan if info["kind"] == "inner" and len(group.producers) > 1 and row in planned}
    if info["kind"] == "widths":
        held |= {row for row, w in planned.items() if w >= cost.widths0[row]}
    floor = fortify.plan_min_width()
    floors = {row: max(1, int(round(min_keep() * w0)), min(w0, floor)) for row, w0 in cost.widths0.items()}
    rounded, report = grid_round_widths(planned, {row: list(walks[row]["grid"]) for row in planned}, cost.kept,
                                        walk_target, grid_priority(info, list(cost.rows)), fixed=held, floors=floors)
    entry["widths"] = {**widths, **rounded}
    entry["grid"] = {"targets": dict(rounded), "checked": 1.0, "below": set(), "fallback": False}
    moved = sum(rounded[row] != planned[row] for row in planned)
    grid = [float(rates[i]) for i in sorted(rates)]
    unit = "FLOPs" if flops else "params"
    utils.print_flush(
        f"[alloc] grid round: {moved} groups moved; predicted x{report['kept']:.3f} of the {unit} (walk target "
        f"x{walk_target:.3f}); grid {grid}"
        + ("" if report["kept"] <= walk_target + 1e-9 else
           "; WARNING: no grid plan at or below the walk target, the stall fallback stays the safety net"))
    try:
        import src.run_recorder as run_recorder
        run_recorder.record(
            "alloc_grid_round", network=str(net), kind=info["kind"], budget="flops" if flops else "params",
            walk_target=walk_target, kept=report["kept"], plan_kept=float(cost.kept(planned)),
            walk_kept=float(cost.kept({row: walks[row]["walk"] for row in planned})), moved=moved,
            down=report["down"], up=report["up"], rates=grid, passes=passes,
            rows={str(r): {"origin": cost.widths0[r], "plan": planned[r], "walk": walks[r]["walk"],
                           "grid": rounded[r], "cuts": walks[r]["grid"].get(rounded[r])} for r in planned})
    except Exception:  # noqa: BLE001 - the record is a convenience
        pass


def _grid_watch(env, entry):
    """Walk-time check of a grid-rounded plan: a WARNING once per group the walk takes below its grid width, and once
    if the stall fallback fires anyway. Either means a legality the offline grid did not see (an eval rollback lock, a
    masked cut, an eval size floor)."""
    grid = entry["grid"]
    try:
        if entry["fallback"] and not grid["fallback"]:
            grid["fallback"] = True
            now = group_widths(env.current_model, list(grid["targets"]))
            off = [f"row {r}: {now.get(r)} vs {t}" for r, t in grid["targets"].items() if now.get(r) != t]
            utils.print_flush(f"[alloc] WARNING grid round: the stall fallback fired with {len(off)} groups off their "
                              f"grid widths ({'; '.join(off[:8])}{'; ...' if len(off) > 8 else ''})")
        kept = float(env.flops_ratio() if budget() == "flops" else env.param_ratio())
        if kept >= grid["checked"] - 1e-12:
            return
        grid["checked"] = kept
        now = group_widths(env.current_model, list(grid["targets"]))
        for row, target in grid["targets"].items():
            width = now.get(row)
            if width is not None and width < target and row not in grid["below"]:
                grid["below"].add(row)
                utils.print_flush(f"[alloc] WARNING grid round: the walk took the group at row {row} to width "
                                  f"{width}, below its grid width {target}")
    except Exception as error:  # noqa: BLE001 - a diagnostic must not end the walk
        if not grid.get("error"):
            grid["error"] = True
            utils.print_flush(f"[alloc] grid watch unavailable ({type(error).__name__}: {error})")


# ------------------------------------------------------------------ v23 randbest (SPECTRA_ALLOC_KIND=randbest)


def randbest_plan(env, target, budget="params"):
    """
    ``(widths, info)`` of the best of N random plans in the plan agent's action space, in ``plan_agent.plan_for_env``'s
    format with ``kind`` randbest, so ``_state`` logs, records, grid-rounds and walks it exactly as the agent's mean
    plan: the per-network random-search baseline of Li et al. (CVPR 2022) scored with the agent's own training proxy.

    Draw i's scores are z_g ~ N(0, σ²) per planned group (``randbest_draws``; σ ``randbest_sigma``, N ``randbest_n``,
    the generator seeded by ``randbest_seed`` of ``SPECTRA_ALLOC_SAMPLE_SEED`` and the run's seed), decoded by
    ``plan_agent.decode`` at ``target`` on the ``budget``'s cost model (``ParamModel`` / ``FlopModel``) with the k_min
    and width floor ``plan_for_env`` would decode the agent with: the checkpoint's ``k_min`` and, when
    ``fortify.plan_min_width`` is off, its ``min_width`` if ``SPECTRA_PLAN_AGENT`` names the agent of record (read
    without building it, so the RNG stands), else ``min_keep`` and the flag's floor. Each plan is scored by
    ``plan_proxies.PlanScorer`` with ``randbest_proxy`` on ``env.val_loader`` only, on the batches
    ``SPECTRA_EVAL_PROXY_SEED`` draws once for all draws (common random numbers; ``SPECTRA_PLAN_FS_LR`` for fs<S>); the
    highest val accuracy wins, ties to the lower draw index (indices are 0-based). The search runs inside
    ``plan_proxies.seeded`` on its own seed, so the Python, NumPy, torch CPU and CUDA RNG states are put back after it
    and the walk and final fine-tune draw as they do under ``agent``. Only the per-draw scores, decoded widths and
    landed params / FLOPs ratios are kept (the FLOPs through a ``FlopModel``, built for the report under a params
    budget), never N models. One line per draw, a summary line and an ``alloc_randbest`` record carry every draw.
    ``SPECTRA_PLAN_RESIDUAL`` is refused: the draws live in the plain decoder's space, not a residual on a prior.
    """
    from src import plan_agent, plan_proxies
    if budget not in BUDGETS:
        raise ValueError(f"budget={budget!r}; expected one of {BUDGETS}")
    n, sigma, proxy = randbest_n(), randbest_sigma(), randbest_proxy()
    residual = fortify.plan_residual(default=None)
    if residual not in (None, "off"):
        raise ValueError(f"SPECTRA_ALLOC_KIND=randbest draws in the plain decoder's space (plan_agent.decode); "
                         f"SPECTRA_PLAN_RESIDUAL={residual} would decode the agent on a prior instead")
    seeds = (sample_seed(), run_seed(env))
    seed, proxy_seed = randbest_seed(*seeds), plan_proxies.proxy_seed()
    net = os.path.basename(str(env.selected_net_path))
    device = env.conf.device
    k_min, min_width = min_keep(), fortify.plan_min_width()
    agent = os.environ.get("SPECTRA_PLAN_AGENT", "").strip()
    if agent:  # the agent of record's checkpoint: its k_min, and its floor when no flag sets one, as plan_for_env reads
        blob = plan_agent.read_blob(plan_agent.ensemble_paths(agent)[0])
        k_min = float(blob.get("k_min", k_min))
        if min_width <= 1:
            min_width = max(1, int(blob.get("min_width", 1)))
    flops = budget == "flops"
    started = time.perf_counter()
    draws, best = [], None
    with plan_proxies.seeded(seed):  # the search's own stream; every global RNG state is put back after it
        model = env.current_model.to(device)
        mwr = ModelWithRows(model)
        plan = group_sensitivity.group_plan(mwr, env._dependency_groups(mwr))
        shape = env._input_shape()
        pm = plan_agent.FlopModel(model, plan, shape, device) if flops else plan_agent.ParamModel(model, plan)
        fm = pm if flops else plan_agent.FlopModel(model, plan, shape, device)
        params0 = float(pm.pm.total0 if flops else pm.total0)
        z = randbest_draws(n, len(pm.rows), sigma, seed)
        scorer = plan_proxies.PlanScorer(model, plan, env.val_loader, device, train_loader=env.train_loader,
                                         input_shape=shape, create_handler=getattr(env, "create_learning_handler", None),
                                         name=net)
        batches = scorer.batches(proxy, proxy_seed)
        for i in range(n):
            widths, decoded = plan_agent.decode(z[i], pm, target, k_min, min_width=min_width)
            score = float(scorer.score(widths, proxy, batches))
            p, f = pm.params(widths) / params0, float(fm.kept(widths))
            draws.append({"score": score, "kept": float(decoded["kept"]), "params": float(p), "flops": f,
                          "widths": widths})
            utils.print_flush(f"[alloc] randbest {net} draw={i} score={score:.4f} params x{p:.3f} FLOPs x{f:.3f}")
            if best is None or score > draws[best]["score"]:
                best = i
        del scorer, batches
    minutes = (time.perf_counter() - started) / 60.0
    scores = [d["score"] for d in draws]
    utils.print_flush(
        f"[alloc] randbest {net}: best draw {best} of {n}, val {scores[best]:.4f} (spread {min(scores):.4f}-"
        f"{max(scores):.4f}, median {statistics.median(scores):.4f}), proxy {proxy}, sigma {sigma:g}, {minutes:.1f} min")
    try:
        import src.run_recorder as run_recorder
        run_recorder.record(
            "alloc_randbest", network=str(env.selected_net_path), proxy=proxy, sigma=sigma, n=n, seed=seed,
            sample_seed=seeds[0], run_seed=seeds[1], proxy_seed=proxy_seed, budget=budget, target=float(target),
            k_min=k_min, **({"min_width": min_width} if min_width > 1 else {}), **({"agent": agent} if agent else {}),
            **({"fs_lr": plan_agent.fs_lr()} if proxy.startswith("fs") else {}),
            pick=best, score=scores[best], minutes=round(minutes, 3),
            draws=[{"score": d["score"], "kept": d["kept"], "params": d["params"], "flops": d["flops"],
                    "widths": {str(r): int(w) for r, w in d["widths"].items()}} for d in draws])
    except Exception:  # noqa: BLE001 - the record is a convenience
        pass
    widths = draws[best]["widths"]
    summary = {"pick": best, "n": n, "score": scores[best], "proxy": proxy, "sigma": sigma, "seed": seed,
               "minutes": minutes, "scores": scores}
    return widths, {"kind": "randbest", "alpha": 0.0, "target": float(target), "kept": draws[best]["kept"],
                    "budget": budget, "keeps": {row: widths[row] / float(pm.widths0[row]) for row in pm.rows},
                    "origin_widths": dict(pm.widths0), "sens": {row: float(v) for row, v in zip(pm.rows, z[best])},
                    "held": 0, "randbest": summary, **({"min_width": int(min_width)} if min_width > 1 else {})}


# ------------------------------------------------------------------ v27 fallback (SPECTRA_ALLOC_KIND=fallback)


def fallback_pick(agent_score: float, prior_score: float, margin: float) -> str:
    """``"prior"`` iff ``prior_score > agent_score + margin``, else ``"agent"``: a tie goes to the agent."""
    return "prior" if float(prior_score) > float(agent_score) + float(margin) else "agent"


def rng_snapshot() -> dict:
    """The Python, NumPy, torch CPU and (when CUDA is available) all CUDA RNG states, for ``rng_restore``."""
    state = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def rng_restore(state: dict) -> None:
    """Put back the generators ``rng_snapshot`` read."""
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["np"])
    random.setstate(state["py"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _state_dict_equal(model, snapshot) -> bool:
    """True iff ``model.state_dict()`` has ``snapshot``'s keys and every tensor is ``torch.equal`` to its copy;
    ``snapshot`` is ``{k: v.detach().clone() for k, v in model.state_dict().items()}``."""
    now = model.state_dict()
    if set(now) != set(snapshot):
        return False
    return all(torch.equal(v.detach(), snapshot[k]) for k, v in now.items())


def _modes(model) -> dict:
    """``{module name: training flag}`` of every module of ``model``, for ``_set_modes``."""
    return {name: bool(module.training) for name, module in model.named_modules()}


def _set_modes(model, modes) -> None:
    """Put back the train / eval flags ``_modes`` read, module by module (``Module.train(mode)`` would recurse)."""
    for name, module in model.named_modules():
        if name in modes:
            module.training = modes[name]


def fallback_plan(env, target, flops, agent_kw, plan_kw):
    """
    ``(widths, info)`` of the plan ``_state`` walks under ``SPECTRA_ALLOC_KIND=fallback``: the frozen agent's plan
    unless the hand rule ``fallback_prior`` plans better on the val half by more than ``fallback_margin`` on the first of
    ``fallback_proxies``, in which case the rule's plan. ``target``, ``flops``, ``agent_kw`` and ``plan_kw`` are the
    values ``_state`` computed for its other kinds.

    The prior is planned by exactly the rule arm's calls (``group_sensitivity.calibration_batches`` and ``plan_targets``
    with ``plan_kw``), then the RNG is put back to its starting state and the agent is planned by exactly the agent arm's
    call (``plan_agent.plan_for_env`` with ``agent_kw``), each from the same starting RNG state and origin as its own row
    makes it; the state after each plan is kept. Both plans are then scored by ``plan_proxies.PlanScorer`` (the agent
    first) with every proxy, on the val half only, on the batches ``SPECTRA_EVAL_PROXY_SEED`` draws once per proxy and
    shares between the two plans (common random numbers), inside ``plan_proxies.seeded``, which puts the RNG back to the
    agent plan's state on exit. The scored widths are the decoders' own, before any grid rounding, as randbest scores
    its draws. The first proxy decides (``fallback_pick``); the others are reported. The origin's weights are checked
    after each plan and after the scoring (a RuntimeError names the step that changed them), and its modules' train /
    eval flags are put back to the start before the agent plans, and to the state the picked arm's plan left after the
    scoring. The RNG is then set to the
    state the picked arm alone leaves, so the grid round, the walk and the final fine-tune draw exactly as under that
    arm, and the picked arm's ``(widths, info)`` is returned unchanged except that a shallow copy of ``info`` gains one
    key, ``fallback`` (pick, prior, proxies, margin, scores, minutes): ``info["kind"]`` stays the arm's own, so
    ``grid_priority``, ``_grid_round_state`` and ``_state``'s plan line treat the plan as that arm's. Nothing is refused
    about ``SPECTRA_PLAN_RESIDUAL``: the agent of record is a residual agent and runs as ``agent`` runs it. One line per
    proxy, a decision line and an ``alloc_fallback`` record carry every score and both plans' widths. An effect
    ``plan_for_env`` has on the env itself stays when the prior is picked. So does a DataLoader iterator that a plan
    first creates: with persistent workers (``utils.DATALOADER_WORKERS`` > 0, every loader but the device-resident
    ``GpuCropFlipLoader`` of CIFAR under GPU augmentation), the second plan reuses an iterator its own row would create, and
    its RNG draws can differ. The equality with the picked arm's row is therefore a check run on the GPU before any read.
    """
    from src import plan_agent, plan_proxies
    prior, proxies, margin = fallback_prior(), fallback_proxies(), fallback_margin()
    net = os.path.basename(str(env.selected_net_path))
    started = time.perf_counter()
    model = env.current_model.to(env.conf.device)
    m0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    modes0 = _modes(model)
    s0 = rng_snapshot()

    def intact(step):
        if not _state_dict_equal(model, m0):
            raise RuntimeError(f"{net}: the {step} changed the origin's weights")

    batches = group_sensitivity.calibration_batches(env.train_loader, group_sensitivity.CALIB_BATCHES, env.conf.device)
    widths_p, info_p = plan_targets(model, batches, env._input_shape(), prior, target, alpha(), min_keep(), **plan_kw)
    s_p, modes_p = rng_snapshot(), _modes(model)
    intact(f"{prior} plan")
    _set_modes(model, modes0)
    rng_restore(s0)
    widths_a, info_a = plan_agent.plan_for_env(env, target, agent_path(), min_keep(), **agent_kw)
    s_a, modes_a = rng_snapshot(), _modes(model)
    intact("agent plan")
    scores = {}
    with plan_proxies.seeded(plan_proxies.proxy_seed()):
        mwr = ModelWithRows(model)
        plan = group_sensitivity.group_plan(mwr, env._dependency_groups(mwr))
        scorer = plan_proxies.PlanScorer(model, plan, env.val_loader, env.conf.device, train_loader=env.train_loader,
                                         input_shape=env._input_shape(),
                                         create_handler=getattr(env, "create_learning_handler", None), name=net)
        for proxy in proxies:
            b = scorer.batches(proxy, plan_proxies.proxy_seed())
            scores[proxy] = {"agent": float(scorer.score(widths_a, proxy, b)),
                             "prior": float(scorer.score(widths_p, proxy, b))}
            del b
        del scorer
    intact("proxy scoring")
    pick = fallback_pick(scores[proxies[0]]["agent"], scores[proxies[0]]["prior"], margin)
    _set_modes(model, modes_p if pick == "prior" else modes_a)
    rng_restore(s_p if pick == "prior" else s_a)
    minutes = (time.perf_counter() - started) / 60.0
    for proxy in proxies:
        a, p = scores[proxy]["agent"], scores[proxy]["prior"]
        utils.print_flush(f"[alloc] fallback {net} {proxy}: agent {a:.4f} prior {prior} {p:.4f} "
                          f"(prior − agent {p - a:+.4f})")
    utils.print_flush(f"[alloc] fallback {net}: pick {pick} on {proxies[0]} (margin {margin:g}; {minutes:.1f} min)")
    try:
        import src.run_recorder as run_recorder
        run_recorder.record(
            "alloc_fallback", network=str(env.selected_net_path), prior=prior, proxies=list(proxies), margin=margin,
            pick=pick, budget="flops" if flops else "params", target=float(target),
            proxy_seed=plan_proxies.proxy_seed(), scores=scores,
            agent_widths={str(r): int(w) for r, w in widths_a.items()},
            prior_widths={str(r): int(w) for r, w in widths_p.items()}, minutes=round(minutes, 3),
            **({"fs_lr": plan_agent.fs_lr()} if any(p.startswith("fs") for p in proxies) else {}))
    except Exception:  # noqa: BLE001 - the record is a convenience
        pass
    widths, info = (widths_p, info_p) if pick == "prior" else (widths_a, info_a)
    info = dict(info)
    info["fallback"] = {"pick": pick, "prior": prior, "proxies": list(proxies), "margin": margin, "scores": scores,
                        "minutes": minutes}
    return widths, info


def _state(env):
    """Per-network plan, measured on the origin the first time the network comes round; under
    ``SPECTRA_ALLOC_GRID_ROUND`` put on the walk's grid (``_grid_round_state``) and checked at every later call
    (``_grid_watch``)."""
    cache = getattr(env, "_alloc_walk", None)
    if cache is None:
        cache = env._alloc_walk = {}
    net = env.selected_net_path
    if net not in cache:
        model = env.current_model.to(env.conf.device)
        flops = budget() == "flops"
        walk_target = float(env.target_keep if env.target_keep is not None else 0.6)
        if flops:
            walk_target = flops_walk_target(env)
        target = walk_target - undershoot()
        # Under the params budget every call below is the one ``tests/test_plan_agent.py`` fakes (no extra arguments).
        agent_kw = {"budget": "flops", "kappa": walk_target} if flops else {}
        plan_kw = {"budget": "flops"} if flops else {}
        floor = fortify.plan_min_width()
        if floor > 1:
            # plan_for_env reads SPECTRA_PLAN_MIN_WIDTH (or the policy blob's floor) itself; only the rule plans take it.
            plan_kw["min_width"] = floor
        if kind() in ("agent", "agent_sample"):
            from src import plan_agent
            if kind() == "agent_sample":
                widths, info = plan_agent.plan_for_env(env, target, agent_path(), min_keep(),
                                                       sample=(sample_sigma(), sample_seed()), **agent_kw)
            else:
                widths, info = plan_agent.plan_for_env(env, target, agent_path(), min_keep(), **agent_kw)
        elif kind() == "randbest":
            widths, info = randbest_plan(env, target, budget="flops" if flops else "params")
        elif kind() == "fallback":
            widths, info = fallback_plan(env, target, flops, agent_kw, plan_kw)
        else:
            batches = group_sensitivity.calibration_batches(
                env.train_loader, group_sensitivity.CALIB_BATCHES, env.conf.device)
            widths, info = plan_targets(model, batches, env._input_shape(), kind(), target, alpha(), min_keep(),
                                        **plan_kw)
        mwr = ModelWithRows(model)
        cache[net] = {"widths": widths, "n_rows": max(1, len(mwr.row_to_main_layer) - 1),
                      "idle": 0, "fallback": False, "last_kept": 1.0}
        keeps = sorted(info["keeps"].values())
        source = (f"widths of {os.path.basename(os.environ.get('SPECTRA_ALLOC_WIDTHS', ''))}"
                  if info["kind"] == "widths" else f"{info['kind']} alpha={info['alpha']:g}")
        if info["kind"] == "agent":
            source = f"agent {info['policy']}"
        if info["kind"] == "agent_sample":
            s = info["sample"]
            source = f"agent_sample sigma={s['sigma']:g} seed={s['seed']} around the mean of {info['policy']}"
        if info["kind"] == "randbest":
            r = info["randbest"]
            source = (f"randbest best draw {r['pick']} of {r['n']} (val {r['score']:.4f}, proxy {r['proxy']}, "
                      f"sigma {r['sigma']:g}, seed {r['seed']})")
        if info["kind"] == "sample" and info.get("sample"):
            sample = info["sample"]
            source = (f"sample around {sample['around']} sigma={sample['sigma']:g} seed={sample['seed']} "
                      f"alpha={info['alpha']:g}")
        utils.print_flush(
            f"[alloc] {os.path.basename(str(net))}: {source} plan keeps "
            f"x{info['kept']:.3f} of the {'FLOPs' if flops else 'params'} (target x{target:.3f} = walk target − "
            f"{undershoot():g}) over "
            f"{len(widths)} groups; group keep min {keeps[0]:.2f} median {statistics.median(keeps):.2f} "
            f"max {keeps[-1]:.2f}" + (f"; {info['held']} coupled groups held at full width" if info["held"] else "")
            + (f"; {info['unnamed']} groups not named in the table, kept whole" if info.get("unnamed") else "")
            + (f"; group width floor {floor}" if floor > 1 else ""))
        if fortify.cut_rank():
            utils.print_flush(
                f"[alloc] cut ranking: {fortify.cut_rank()}"
                + (f" (seed {pruning.random_rank_seed()})" if fortify.cut_rank() == "random" else "")
                + f" for {os.path.basename(str(net))}")
        try:
            import src.run_recorder as run_recorder
            run_recorder.record(
                "alloc_plan", network=str(net), kind=info["kind"], alpha=info["alpha"], target=target,
                kept=info["kept"], **({"sample": info["sample"]} if info.get("sample") else {}),
                **({"randbest": {key: info["randbest"][key] for key in ("pick", "n", "score", "proxy", "sigma", "seed")}}
                   if info.get("randbest") else {}),
                **({"budget": "flops"} if flops else {}), **({"min_width": floor} if floor > 1 else {}),
                rows={str(r): {"origin": info["origin_widths"].get(r), "target": widths[r],
                                                  "keep": round(info["keeps"][r], 4),
                                                  "sens": float(info["sens"][r])} for r in widths})
        except Exception:  # noqa: BLE001 - the record is a convenience
            pass
        if fortify.alloc_grid_round():
            _grid_round_state(env, net, cache[net], model, info, walk_target, flops)
    elif "grid" in cache[net]:
        _grid_watch(env, cache[net])
    return cache[net]


def action(env, legal, rates, device):
    """``SPECTRA_EVAL_POLICY=alloc``: one decision of the allocation-following walk."""
    identity = next((i for i, r in rates.items() if abs(float(r) - 1.0) < 1e-9), 0)
    legal_idx = [int(i) for i in legal.nonzero(as_tuple=False).flatten().tolist()]
    state = _state(env)
    flops = budget() == "flops"
    kept_now = float(env.flops_ratio() if flops else env.param_ratio())
    if kept_now > state["last_kept"] + 1e-9:
        state.update(idle=0, fallback=False)
    state["last_kept"] = kept_now
    if state["fallback"]:
        return torch.tensor([pick_strongest(rates, legal_idx, identity)], device=device)
    mwr = ModelWithRows(env.current_model)
    row = max(0, (env.row_idx or 1) - 1)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    group = channel_groups.group_of(groups, mwr.all_layers[mwr.row_to_main_layer[row]])
    pick = identity
    if group is not None:
        first = {id(g): r for g, r in group_sensitivity.group_plan(mwr, groups)}
        target = state["widths"].get(first.get(id(group)))
        if target is not None:
            pick = choose(int(group.width), int(target), rates, legal_idx, identity)
    state["idle"] = 0 if pick != identity else state["idle"] + 1
    if state["idle"] >= state["n_rows"] and env.target_keep is not None and kept_now > float(env.target_keep):
        state["fallback"] = True
        utils.print_flush(f"[alloc] every group at its target with {'FLOPs' if flops else 'params'} x{kept_now:.3f} "
                          f"above the target x{float(env.target_keep):.3f}; strongest legal cut from here")
        pick = pick_strongest(rates, legal_idx, identity)
    return torch.tensor([pick], device=device)


def pick_strongest(rates, legal_idx, identity):
    cuts = [i for i in legal_idx if float(rates[i]) < 1.0 - 1e-9]
    return min(cuts, key=lambda i: float(rates[i])) if cuts else identity
