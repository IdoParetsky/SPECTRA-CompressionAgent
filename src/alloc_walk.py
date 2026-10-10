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

import torch

import src.channel_groups as channel_groups
import src.fortify as fortify
import src.group_sensitivity as group_sensitivity
import src.pruning as pruning
import src.utils as utils
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows

KINDS = ("uniform", "sens", "inner", "widths", "sample", "agent", "agent_sample", "sens_cost")
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
    if kind_name in ("agent", "agent_sample"):
        z = {row: float(sens.get(row, 0.0)) for row in rows}
        prior = (info.get("prior") or {}).get("weights") if info.get("residual") else None
        if prior:  # T2: a residual plan ranks groups by log(prior weight) + z (plan_agent.residual_decode)
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
        try:
            import src.run_recorder as run_recorder
            run_recorder.record(
                "alloc_plan", network=str(net), kind=info["kind"], alpha=info["alpha"], target=target,
                kept=info["kept"], **({"sample": info["sample"]} if info.get("sample") else {}),
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
