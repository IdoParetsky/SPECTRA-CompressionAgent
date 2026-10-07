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
parameters. Each decision then plays the legal cut whose resulting width is closest to its group's
target (ties to the milder cut, identity once it is there). Group-once, the recovery and the
fixed-target landing are the walk's own, so a row differs from greedy / mild only in which groups
keep being cut. The undershoot makes the walk cross the target before every group has arrived;
should it still stall above the target for a whole pass, it falls back to the strongest legal cut.
"""

from __future__ import annotations

import copy
import os
import statistics

import torch

import src.channel_groups as channel_groups
import src.group_sensitivity as group_sensitivity
import src.pruning as pruning
import src.utils as utils
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows

KINDS = ("uniform", "sens", "inner")


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
    closest c found).
    """
    plan = group_sensitivity.group_plan(ModelWithRows(model))
    rows = [row for _group, row in plan]
    if kind_name == "sens":
        sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, input_shape)
    else:
        sens = {row: 1.0 for row in rows}
    w = weights(kind_name, sens, a)
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
        batches = group_sensitivity.calibration_batches(
            env.train_loader, group_sensitivity.CALIB_BATCHES, env.conf.device)
        widths, info = plan_targets(model, batches, env._input_shape(), kind(), target, alpha(), min_keep())
        mwr = ModelWithRows(model)
        cache[net] = {"widths": widths, "n_rows": max(1, len(mwr.row_to_main_layer) - 1),
                      "idle": 0, "fallback": False, "last_kept": 1.0}
        keeps = sorted(info["keeps"].values())
        utils.print_flush(
            f"[alloc] {os.path.basename(str(net))}: {info['kind']} alpha={info['alpha']:g} plan keeps "
            f"x{info['kept']:.3f} of the params (target x{target:.3f} = walk target − {undershoot():g}) over "
            f"{len(widths)} groups; group keep min {keeps[0]:.2f} median {statistics.median(keeps):.2f} "
            f"max {keeps[-1]:.2f}" + (f"; {info['held']} coupled groups held at full width" if info["held"] else ""))
        try:
            import src.run_recorder as run_recorder
            run_recorder.record(
                "alloc_plan", network=str(net), kind=info["kind"], alpha=info["alpha"], target=target,
                kept=info["kept"], rows={str(r): {"origin": info["origin_widths"].get(r), "target": widths[r],
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
