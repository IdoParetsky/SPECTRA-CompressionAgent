"""
State dump for the D-IMIT imitation probe (``SPECTRA_DUMP_STATES=1``, default off).

An eval job with the flag set does not walk. For every TEST network it resets the environment at
each target keep in ``SPECTRA_DUMP_KEEPS`` (default 0.4, 0.6, 0.8) and saves one record per
(net, κ) under ``SPECTRA_DUMP_DIR`` (default ``<run dir>/state_dump``):

* ``layer``: the state ``NetworkEnv.reset`` returns, i.e. what the actor reads;
* ``group``: the same state pooled into one token per coupling group (``group_tokens``), as
  ``SPECTRA_STATE_TOKENS=groups`` would build it;
* ``bert``: the frozen-BERT input of the same origin (``SPECTRA_STATE_ENCODER=bert``), or None;
* ``layout``: the token columns by channel, so a probe can zero a channel;
* ``token_rows`` / ``group_token_rows``: each token's planned group (its first walk row), -1 if none;
* ``plan``: the sens plan at κ − the alloc walk's undershoot (``alloc_walk.plan_targets``), whose
  per-group keeps the probe regresses on.
"""

import contextlib
import os
import time

import torch

DEFAULT_KEEPS = (0.4, 0.6, 0.8)


def enabled() -> bool:
    return os.environ.get("SPECTRA_DUMP_STATES", "").strip().lower() in ("1", "true", "yes", "on")


def keeps():
    raw = os.environ.get("SPECTRA_DUMP_KEEPS", "").strip()
    if not raw:
        return DEFAULT_KEEPS
    return tuple(float(v) for v in raw.replace(";", ",").split(",") if v.strip())


def out_dir() -> str:
    path = os.environ.get("SPECTRA_DUMP_DIR", "").strip()
    if not path:
        import src.logging_utils as logging_utils
        path = os.path.join(logging_utils.run_dir() or ".", "state_dump")
    os.makedirs(path, exist_ok=True)
    return path


def layout(width: int, num_actions: int, group_extra: int = 0):
    """``[(channel, start, end)]`` of a token, in ``BERTInputModeler._build_layer_tokens``' order."""
    from src import fortify
    from src.BERTInputModeler import TOKEN_BASE_DIM
    parts = [("base", TOKEN_BASE_DIM)]
    if fortify.fortify_enabled():
        parts.append(("fortify", fortify.FORTIFY_TOKEN_DIM))
    if fortify.budget_in_state():
        parts.append(("budget", 1))
    if fortify.state_slack():
        parts.append(("slack", fortify.STATE_SLACK_DIM))
    if fortify.state_groupcost():
        parts.append(("groupcost", fortify.STATE_GROUPCOST_DIM))
    if fortify.fixed_target():
        parts.append(("target", fortify.STATE_TARGET_DIM))
    if fortify.state_sens():
        parts.append(("sens", fortify.STATE_SENS_DIM))
    parts.append(("action", 2 * int(num_actions)))
    if group_extra:
        parts.append(("group_extra", int(group_extra)))
    out, start = [], 0
    for name, n in parts:
        out.append((name, start, start + n))
        start += n
    if start != int(width):
        raise ValueError(f"token layout {out} sums to {start}; the state is {width} wide")
    return out


def token_rows(model_with_rows, groups, plan):
    """Per layer token, the first walk row of the planned group it belongs to, or -1.

    Same mapping as ``group_sensitivity.layer_features``, so a token's target is the group whose
    sensitivity its ``STATE_SENS`` channels carry.
    """
    import src.channel_groups as channel_groups
    by_group = {id(group): row for group, row in plan}
    rows = []
    for layer in model_with_rows.all_layers:
        group = channel_groups.group_of(groups, layer) if groups else None
        rows.append(by_group.get(id(group), -1) if group is not None else -1)
    return rows


def group_token_rows(members, rows):
    """``(rows, conflicts)``: per group token, the planned row its member layers map to, or -1."""
    out, conflicts = [], 0
    for idx in members:
        found = sorted({rows[i] for i in idx if 0 <= i < len(rows) and rows[i] >= 0})
        conflicts += len(found) > 1
        out.append(found[0] if found else -1)
    return out, conflicts


@contextlib.contextmanager
def bert_input():
    """Build states as ``SPECTRA_STATE_ENCODER=bert`` does (the kind is read once at import)."""
    import src.BERTInputModeler as bim
    old = bim.STATE_ENCODER_KIND
    bim.STATE_ENCODER_KIND = "bert"
    try:
        yield
    finally:
        bim.STATE_ENCODER_KIND = old


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(value) for value in obj)
    return obj


def encode_origin(env, model_with_rows, groups):
    """The origin's state through the call ``NetworkEnv.reset`` makes, with the same arguments."""
    from src import fortify
    target_extras = (fortify.target_channels(1.0, env.target_keep)
                     if env.target_keep is not None else None)
    return env.feature_extractor.encode_to_bert_input(
        model_with_rows, model_with_rows.row_to_main_layer[env.row_idx - 1],
        dependency_groups=groups, param_ratio=1.0, extras=[1.0, 0.0], episode_cuts={},
        target_extras=target_extras, layer_sens=env._layer_sens)


def sens_plan(env, kappa):
    """``(target, widths, info)`` of the sens plan the alloc walk would follow at ``kappa``."""
    from src import alloc_walk, group_sensitivity
    model = env.current_model.to(env.conf.device)
    batches = group_sensitivity.calibration_batches(
        env.train_loader, group_sensitivity.CALIB_BATCHES, env.conf.device)
    target = float(kappa) - alloc_walk.undershoot()
    widths, info = alloc_walk.plan_targets(model, batches, env._input_shape(), "sens", target,
                                           alloc_walk.alpha(), alloc_walk.min_keep())
    return target, widths, info


def dump(env, net_path, net_model, net_loaders):
    """Save one record per κ for this network (see the module docstring)."""
    from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
    from src import fortify
    from src.BERTInputModeler import action_cost_slot_dim
    from src.group_sensitivity import group_plan
    from src.group_tokens import GROUP_TOKEN_EXTRA_DIM, group_token_state
    import src.utils as utils
    name = os.path.basename(str(net_path))
    stem = name[:-3] if name.endswith(".pt") else name
    num_actions = len(env.conf.compression_rates_dict)
    for kappa in keeps():
        started = time.perf_counter()
        state = env.reset(test_net_path=net_path, test_model=net_model, test_loaders=net_loaders,
                          target_keep=kappa)
        mwr = ModelWithRows(env.current_model)
        groups = env._dependency_groups(mwr)
        plan = group_plan(mwr, groups)
        rows = token_rows(mwr, groups, plan)
        width = int(state["layer_features"].size(1))
        group_state = group_token_state(state, mwr.all_layers, groups,
                                        slot_dim=action_cost_slot_dim(num_actions))
        if "token_members" not in group_state:
            group_state = None
        g_rows, conflicts = group_token_rows(group_state["token_members"] if group_state else [], rows)
        bert, bert_error = None, None
        try:
            with bert_input():
                bert = encode_origin(env, mwr, groups).get("bert")
        except Exception as error:  # noqa: BLE001 - arm (g) is reported as not run
            bert_error = f"{type(error).__name__}: {error}"
        target, widths, info = sens_plan(env, kappa)
        record = {
            "net": name, "kappa": float(kappa), "plan_target": target,
            "origin_val": float(env.original_acc), "origin_params": int(env.original_params),
            "layer": state, "group": group_state, "bert": bert, "bert_error": bert_error,
            "layout": layout(width, num_actions),
            "group_layout": (layout(int(group_state["layer_features"].size(1)), num_actions,
                                    GROUP_TOKEN_EXTRA_DIM) if group_state else None),
            "token_rows": rows, "group_token_rows": g_rows, "group_row_conflicts": conflicts,
            "num_layers": len(mwr.all_layers),
            "plan": {"rows": [int(r) for _g, r in plan],
                     "keeps": {int(r): float(k) for r, k in info["keeps"].items()},
                     "sens": {int(r): float(s) for r, s in info["sens"].items()},
                     "origin_widths": {int(r): int(w) for r, w in info["origin_widths"].items()},
                     "widths": {int(r): int(w) for r, w in widths.items()},
                     "kept": float(info["kept"])},
            "flags": {"state_sens": fortify.state_sens(), "groupcost": fortify.state_groupcost(),
                      "fixed_target": fortify.fixed_target(), "state_tokens": fortify.state_tokens(),
                      "num_actions": num_actions},
        }
        path = os.path.join(out_dir(), f"{stem}_k{kappa:.2f}.pt")
        torch.save(to_cpu(record), path)
        mapped = sum(r >= 0 for r in rows)
        utils.print_flush(
            f"[dump] {name} k={kappa:.2f}: {len(rows)} layer tokens ({mapped} on {len(plan)} planned groups), "
            f"{len(g_rows)} group tokens ({conflicts} conflicts), width {width}, "
            f"bert={'yes' if bert is not None else 'no: ' + str(bert_error)}; sens plan keeps "
            f"x{info['kept']:.3f} of the params (target x{target:.3f}); {time.perf_counter() - started:.1f}s -> {path}")
