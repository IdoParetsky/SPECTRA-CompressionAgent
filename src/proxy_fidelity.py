"""
Proxy-fidelity battery (``SPECTRA_EVAL_PROXY_FIDELITY``; default off).

Does the recovery the agent trains with rank candidate cuts the way the final fine-tune does? The
agent never sees the final fine-tune: its reward is the val accuracy after a short recipe-A
fine-tune (12 epochs, patience 4, in policy training). Le & Hua (ICLR 2021) show the retraining
schedule alone can reorder pruning methods; EagleEye (Li et al., ECCV 2020) measures the same
correlation for candidate evaluators and proposes re-estimated BN statistics as the cheap one.

The first time a TRAJ TEST walk's kept params reach a target, the current model is a *state*.
From copies of it:

* ``identity`` — no cut;
* ``menu``     — the Stage-4 cut actions on one row: keep 0.9 / 0.8 x L1 / FPGM;
* ``crit``     — that row at keep 0.8 under BN-scale, SVD and Taylor (five criteria with the menu's
  0.8 entries);
* ``where``    — L1 cuts of other groups that remove the same share of the network as the menu's
  keep-0.8 L1 cut (:func:`src.fortify.budget_keep_rate`), spread over depth.

The row is the one the walk decides next, or the first one after it whose group is at least
``MIN_MENU_WIDTH`` channels wide (on narrower groups keep 0.9 and 0.8 cut the same channel).
Group-once locks are ignored: the battery ranks cuts of a state, it does not replay a decision.

Every candidate is scored by each proxy — ``none`` (the raw cut), ``bn`` (BN statistics
re-estimated), ``<E>x<P>`` (recipe A at that epoch cap and patience) — on val and TEST, and by the
final fine-tune (SGD, momentum 0.9, wd 5e-4, cosine, the TRAJ final-FT loader and batch) from the raw
cut, once per seed. A cut that repeats an earlier candidate's network exactly is recorded as a
duplicate and not scored. The walk's model, row and RNG state are left as they were found.
"""

from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import torch

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
import src.channel_groups as channel_groups
import src.fortify as fortify
import src.logging_utils as logging_utils
import src.pruning as pruning
import src.recovery_edits as recovery_edits
import src.run_recorder as run_recorder
import src.utils as utils
from src.NetworkEnv import prune_current_model

MENU = ((0.9, "l1"), (0.8, "l1"), (0.9, "fpgm"), (0.8, "fpgm"))
CRIT_RATE = 0.8
CRITERIA = ("bn_scale", "svd", "taylor")
MIN_MENU_WIDTH = 10
# Same keys and order as the runner's _FINAL_FT_ENV_KEYS (tests/test_proxy_fidelity.py checks).
FINAL_ENV_KEYS = ("SPECTRA_FT_OPTIM", "SPECTRA_FT_SGD_LR", "SPECTRA_FT_MOMENTUM", "SPECTRA_FT_WD",
                  "SPECTRA_FT_COSINE", "SPECTRA_FT_SCHEDULE", "SPECTRA_FT_MIXUP",
                  "SPECTRA_FT_LABEL_SMOOTH", "SPECTRA_FT_KD")


def maybe_run(env, net_path, point, targets, done):
    """Run the battery for every target the walk's last point reached for the first time."""
    for target in targets:
        if target not in done and float(point["param"]) <= float(target) + 1e-12:
            done.add(target)
            run_battery(env, net_path, target, point)


def group_table(model):
    """``([(row, group key, width, prunable)], ModelWithRows)``; the key is shared by a group's rows."""
    mwr = ModelWithRows(model)
    try:
        groups = channel_groups.build_channel_groups(mwr.model)
    except Exception:  # noqa: BLE001 - an untraceable net has no groups
        groups = None
    table = []
    for row in range(len(mwr.row_to_main_layer)):
        layer = mwr.all_layers[mwr.row_to_main_layer[row]]
        group = channel_groups.group_of(groups, layer) if groups else None
        if group is None:
            table.append((row, None, 0, False))
        else:
            table.append((row, id(group), int(group.width), bool(group.prunable)))
    return table, mwr


def menu_row(table, start):
    """``start``, or the first row after it (wrapping) whose group is prunable and wide enough."""
    n = len(table)
    for min_width in (MIN_MENU_WIDTH, 2):
        for k in range(n):
            row, key, width, prunable = table[(start + k) % n]
            if key is not None and prunable and width >= min_width:
                return row
    return None


def where_rows(table, mwr, row, share, k):
    """Up to ``k`` other groups whose cut removes ``share`` of the network: ``[(row, keep rate)]``."""
    seen = {table[row][1]}
    feasible = []
    for other, key, width, prunable in table:
        if key is None or not prunable or key in seen or width < 2:
            continue
        owned = recovery_edits.group_param_fraction(mwr, other)
        keep = fortify.budget_keep_rate(owned, share)
        if keep < fortify.BUDGET_MIN_KEEP or owned / width > fortify.BUDGET_OVERSHOOT_TOLERANCE * share:
            continue
        if pruning.target_width(width, keep) >= width:
            continue
        seen.add(key)
        feasible.append((other, keep))
    if len(feasible) <= k:
        return feasible
    picks = sorted({int(round(i * (len(feasible) - 1) / max(1, k - 1))) for i in range(k)})
    return [feasible[i] for i in picks]


def cut(env, base, row, rate, ranking):
    """A pruned copy of ``base`` and its prune outcome; ``base`` is never modified."""
    model = copy.deepcopy(base)
    if rate >= 1.0:
        return model, {"mode": "identity"}
    mwr = ModelWithRows(model)
    if pruning.normalize_importance_mode(ranking) == "taylor":
        pruning.bind_taylor_scores(model, env.train_loader, env.conf.device)
    mwr = prune_current_model(mwr, rate, row, quiet=True, record=False,
                              input_shape=env._input_shape(), importance=ranking)
    return mwr.model, dict(getattr(mwr, "last_prune_outcome", {}) or {})


def fingerprint(model):
    with torch.no_grad():
        params = list(model.parameters())
        return (sum(int(p.numel()) for p in params), round(sum(float(p.double().sum()) for p in params), 6))


def score(env, model):
    handler = env.create_learning_handler(model)
    return {"val": float(handler.evaluate_model(env.val_loader)),
            "test": float(handler.evaluate_model(env.test_loader))}


def proxy(env, model, name, epochs=None, patience=None):
    """One proxy on a copy: ``none``, ``bn`` or recipe A at ``epochs`` / ``patience``."""
    model = copy.deepcopy(model)
    if name == "bn":
        recovery_edits.recalibrate_batchnorm(model, env.train_loader, env.conf.device,
                                             n_batches=max(8, fortify.ft_calib_batches()))
    elif epochs:
        handler = env.create_learning_handler(model)
        handler.unfreeze_all_layers()
        handler.train_model(env.train_loader, max_epochs=epochs, patience=patience, tag=f"proxy {name}")
        model = handler.model
    return score(env, model)


def final(env, model, epochs, seed):
    """The TRAJ final recipe (no KD, inherit) on a copy, seeded; val/TEST and minutes."""
    model = copy.deepcopy(model)
    for param in model.parameters():
        param.requires_grad = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader, aug = utils.final_ft_train_loader(env.train_loader, fortify.eval_final_ft_batch())
    lr = fortify.eval_final_ft_lr()
    recipe = dict(zip(FINAL_ENV_KEYS, ("sgd", f"{lr:g}", "0.9", "5e-4", "1", "", "0", "0", "0")))
    saved = {k: os.environ.get(k) for k in FINAL_ENV_KEYS}
    t0 = time.perf_counter()
    try:
        os.environ.update(recipe)
        handler = env.create_learning_handler(model)
        handler.train_model(loader, allow_reinit_retry=False, max_epochs=epochs, patience=epochs + 1,
                            tag=f"proxy final s{seed}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    out = score(env, handler.model)
    out.update(minutes=round((time.perf_counter() - t0) / 60.0, 2), aug=aug, lr=lr, epochs=int(epochs))
    return out


def _rng_state():
    state = {"torch": torch.get_rng_state(), "np": np.random.get_state(), "py": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["np"])
    random.setstate(state["py"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_battery(env, net_path, target, point):
    """Score the candidate set of one state. Returns the ``proxy_fidelity`` records."""
    name = os.path.basename(net_path)
    recipe = fortify.ft_recipe(bool(getattr(env.conf, "train_compressed_layer_only", False)))
    blocked = [label for label, on in ((f"recipe {recipe}", recipe != "A"),
                                       ("A-LSQ", fortify.ft_lsq_consumers()),
                                       ("C-PCA", fortify.ft_pca_reinit()),
                                       ("BN recal", fortify.ft_bn_recal()),
                                       ("group-first", fortify.ft_group_first_epochs() > 0)) if on]
    if blocked:
        utils.print_flush(f"[proxy] {name} t={target:g}: skipped; the proxies mirror recipe A only "
                          f"({', '.join(blocked)})")
        return []
    rng = _rng_state()
    try:
        return _battery(env, net_path, name, target, point)
    finally:
        _set_rng_state(rng)


def _battery(env, net_path, name, target, point):
    base = env.current_model
    table, mwr = group_table(copy.deepcopy(base))
    start = max(0, int(env.row_idx) - 1) % max(1, len(table))
    row = menu_row(table, start)
    if row is None:
        utils.print_flush(f"[proxy] {name} t={target:g}: no prunable group; skipped")
        return []
    budgets = fortify.eval_proxy_budgets()
    seeds = fortify.eval_proxy_final_seeds()
    epochs = fortify.eval_proxy_final_epochs()
    shape = env._input_shape()
    params_base = float(utils.calc_num_parameters(base))
    origin_params = float(env.original_params)
    origin_flops = float(getattr(env, "original_flops", 0) or 0)
    val_origin, test_origin = float(env.original_acc), float(env._origin_test_acc)
    parent = score(env, base)
    utils.print_flush(
        f"[proxy] {name} t={target:g}: state step={point['step']} params x{float(point['param']):.3f} "
        f"FLOPs x{float(point['flop']):.3f} | val {parent['val']:.4f} TEST {parent['test']:.4f} | next row "
        f"{start} -> menu row {row} (width {table[row][2]}) | proxies none,bn,"
        f"{','.join(f'{e}x{p}' for e, p in budgets)} | final e{epochs} seeds {','.join(map(str, seeds))}")
    queue = ([("identity", row, 1.0, None)] + [("menu", row, r, k) for r, k in MENU]
             + [("crit", row, CRIT_RATE, c) for c in CRITERIA])
    seen, records, i = {}, [], 0
    while i < len(queue):
        kind, r, rate, ranking = queue[i]
        i += 1
        label = f"{kind} row={r} {rate:.3f}/{ranking or 'none'}"
        try:
            model, outcome = cut(env, base, r, rate, ranking)
            params = float(utils.calc_num_parameters(model))
            share = (params_base - params) / max(1.0, params_base)
            if kind == "menu" and rate == CRIT_RATE and ranking == "l1":
                for other, keep in where_rows(table, mwr, row, share, fortify.eval_proxy_where_rows()):
                    queue.append(("where", other, keep, "l1"))
            if rate < 1.0 and outcome.get("mode") != "structural":
                utils.print_flush(f"[proxy] {name} t={target:g} {label}: {outcome.get('mode')} cut; not scored")
                continue
            common = dict(network=net_path, target=float(target), state_step=int(point["step"]),
                          state_param=float(point["param"]), next_row=int(start), row=int(r), kind=kind,
                          rate=float(rate), ranking=ranking or "none", mode=outcome.get("mode"),
                          old_width=outcome.get("old_width"), new_width=outcome.get("new_width"),
                          param=params / origin_params, share=share)
            fp = fingerprint(model)
            if fp in seen:
                utils.print_flush(f"[proxy] {name} t={target:g} {label}: same network as {seen[fp]}; not scored")
                run_recorder.record("proxy_fidelity", duplicate_of=seen[fp], **common)
                continue
            seen[fp] = label
            flop = utils.calc_flops(model, shape) / origin_flops if origin_flops else None
            if rate < 1.0:
                scores = {"none": score(env, model), "bn": proxy(env, model, "bn")}
                for e, p in budgets:
                    scores[f"{e}x{p}"] = proxy(env, model, f"{e}x{p}", e, p)
            else:
                scores = {key: dict(parent) for key in ["none", "bn"] + [f"{e}x{p}" for e, p in budgets]}
            finals = {str(s): final(env, model, epochs, s) for s in seeds}
            record = dict(common, flop=flop, val_origin=val_origin, test_origin=test_origin, parent=parent,
                          scores=scores, finals=finals)
            run_recorder.record("proxy_fidelity", **record)
            records.append(record)
            proxies = " ".join(f"{k} {(v['val'] - val_origin) * 100:+.2f}" for k, v in scores.items())
            finals_pp = " ".join(f"s{k} {(v['test'] - test_origin) * 100:+.2f}" for k, v in finals.items())
            utils.print_flush(
                f"[proxy] {name} t={target:g} {label} | params x{params / origin_params:.3f} share {share:.4f} | "
                f"val Δ {proxies} | final TEST Δ {finals_pp} | "
                f"{sum(v['minutes'] for v in finals.values()):.1f} min final")
        except Exception as error:  # noqa: BLE001 - one candidate must not cost the others or the walk
            logging_utils.exception(f"[proxy] {name} t={target:g} {label} failed; continuing")
            run_recorder.issue("proxy_fidelity_failed", f"{type(error).__name__}: {error}",
                               network=net_path, label=label)
    utils.print_flush(f"[proxy] {name} t={target:g}: {len(records)} candidates scored")
    return records
