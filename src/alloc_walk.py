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
import src.group_sensitivity as group_sensitivity
import src.pruning as pruning
import src.utils as utils
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows

KINDS = ("uniform", "sens", "inner", "widths", "sample", "agent", "agent_sample")
SAMPLE_AROUND = ("sens", "uniform")


def kind() -> str:
    """``SPECTRA_ALLOC_KIND`` (sens): the allocation the walk follows."""
    name = os.environ.get("SPECTRA_ALLOC_KIND", "sens").strip().lower() or "sens"
    if name not in KINDS:
        raise ValueError(f"SPECTRA_ALLOC_KIND={name!r}: expected one of {KINDS}")
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


def weights(kind_name, sens, a=0.5):
    """Relative keep per row before scaling: 1 (uniform, inner) or (s / median)^α, s floored at 5 % of the median."""
    keys = list(sens)
    if kind_name in ("uniform", "inner"):
        return {key: 1.0 for key in keys}
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


def plan_targets(model, batches, input_shape, kind_name, target, a=0.5, keep_floor=0.1, iters=16, tol=0.003):
    """
    ``(widths, info)``: ``widths[row]`` is the target width of the group whose first walk row is
    ``row``. Keeps are clip(c · weight, keep_floor, 1), or 1 for a group ``inner`` holds; c is
    bisected until the one-shot cut keeps ``target`` of the parameters (widths are integers, so the
    closest c found). ``widths`` copies its table instead and ignores ``target``.
    """
    plan = group_sensitivity.group_plan(ModelWithRows(model))
    rows = [row for _group, row in plan]
    if kind_name == "widths":
        keeps, unnamed = copied_keeps(model, plan, widths_table())
        cut = cut_to(model, plan, keeps, input_shape)
        frac = utils.calc_num_parameters(cut) / utils.calc_num_parameters(model)
        widths = group_widths(cut, rows)
        del cut
        return widths, {"kind": kind_name, "alpha": float(a), "target": float(target), "kept": float(frac),
                        "keeps": keeps, "origin_widths": group_widths(model, rows),
                        "sens": {row: 1.0 for row in rows}, "held": 0, "unnamed": unnamed}
    base_kind = sample_around() if kind_name == "sample" else kind_name
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
    params0 = utils.calc_num_parameters(model)
    lo, hi = 0.0, 1.0 / min(w.values())
    best = None
    for _ in range(iters):
        c = 0.5 * (lo + hi)
        keeps = {row: 1.0 if row in held else min(1.0, max(keep_floor, c * w[row])) for row in rows}
        cut = cut_to(model, plan, keeps, input_shape)
        frac = utils.calc_num_parameters(cut) / params0
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


def _state(env):
    """Per-network plan, measured on the origin the first time the network comes round."""
    cache = getattr(env, "_alloc_walk", None)
    if cache is None:
        cache = env._alloc_walk = {}
    net = env.selected_net_path
    if net not in cache:
        model = env.current_model.to(env.conf.device)
        target = float(env.target_keep if env.target_keep is not None else 0.6) - undershoot()
        if kind() in ("agent", "agent_sample"):
            from src import plan_agent
            if kind() == "agent_sample":
                widths, info = plan_agent.plan_for_env(env, target, agent_path(), min_keep(),
                                                       sample=(sample_sigma(), sample_seed()))
            else:
                widths, info = plan_agent.plan_for_env(env, target, agent_path(), min_keep())
        else:
            batches = group_sensitivity.calibration_batches(
                env.train_loader, group_sensitivity.CALIB_BATCHES, env.conf.device)
            widths, info = plan_targets(model, batches, env._input_shape(), kind(), target, alpha(), min_keep())
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
            f"x{info['kept']:.3f} of the params (target x{target:.3f} = walk target − {undershoot():g}) over "
            f"{len(widths)} groups; group keep min {keeps[0]:.2f} median {statistics.median(keeps):.2f} "
            f"max {keeps[-1]:.2f}" + (f"; {info['held']} coupled groups held at full width" if info["held"] else "")
            + (f"; {info['unnamed']} groups not named in the table, kept whole" if info.get("unnamed") else ""))
        try:
            import src.run_recorder as run_recorder
            run_recorder.record(
                "alloc_plan", network=str(net), kind=info["kind"], alpha=info["alpha"], target=target,
                kept=info["kept"], **({"sample": info["sample"]} if info.get("sample") else {}),
                rows={str(r): {"origin": info["origin_widths"].get(r), "target": widths[r],
                                                  "keep": round(info["keeps"][r], 4),
                                                  "sens": float(info["sens"][r])} for r in widths})
        except Exception:  # noqa: BLE001 - the record is a convenience
            pass
    return cache[net]


def action(env, legal, rates, device):
    """``SPECTRA_EVAL_POLICY=alloc``: one decision of the allocation-following walk."""
    identity = next((i for i, r in rates.items() if abs(float(r) - 1.0) < 1e-9), 0)
    legal_idx = [int(i) for i in legal.nonzero(as_tuple=False).flatten().tolist()]
    state = _state(env)
    kept_now = float(env.param_ratio())
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
        utils.print_flush(f"[alloc] every group at its target with params x{kept_now:.3f} above the target "
                          f"x{float(env.target_keep):.3f}; strongest legal cut from here")
        pick = pick_strongest(rates, legal_idx, identity)
    return torch.tensor([pick], device=device)


def pick_strongest(rates, legal_idx, identity):
    cuts = [i for i in legal_idx if float(rates[i]) < 1.0 - 1e-9]
    return min(cuts, key=lambda i: float(rates[i])) if cuts else identity
