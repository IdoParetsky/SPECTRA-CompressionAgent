import contextlib
import copy
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
import time
import warnings


from NetworkFeatureExtraction.src.ModelClasses.NetX.netX import NetX  # required for compatibility with `torch.load`
from src.A2C_Agent_Reinforce import A2CAgentReinforce
from src.NetworkEnv import *
import src.utils as utils
import src.distributed as ddp
import src.logging_utils as logging_utils
import src.run_recorder as run_recorder
import src.traj_models as traj_models
from src.Configuration.StaticConf import StaticConf

# Distributed is opt-in: a plain `python a2c_agent_reinforce_runner.py ...` on a single
# rtx_6000 never initialises NCCL. Launch with torchrun/srun to enable multi-GPU.
ddp.maybe_init_distributed()
rank = ddp.get_rank()
# Optional PyCharm remote debug (set SPECTRA_PYDEVD=1). Off by default for Cursor/SLURM runs.
if os.environ.get("SPECTRA_PYDEVD", "").strip() in ("1", "true", "True"):
    port = 12345 + rank  # each rank gets its own port, initial value chosen arbitrarily
    utils.print_flush(f"[Rank {rank}] Connecting debugger on port {port}")
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=port, stdoutToServer=True, stderrToServer=True, suspend=False)

import logging
logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)


def _traj_capture(env, step_index, rate, test_new, test_orig):
    val_new = float(getattr(env, "last_val_acc", env.original_acc))
    val_orig = float(env.original_acc)
    return {
        "step": int(step_index),
        "rate": float(rate),
        "param": float(env.param_ratio()),
        "flop": float(env.flops_ratio()),
        "val_acc": val_new,
        "val_origin": val_orig,
        "val_dacc_pp": (val_new - val_orig) * 100.0,
        "test_acc": float(test_new),
        "test_origin": float(test_orig),
        "test_dacc_pp": (float(test_new) - float(test_orig)) * 100.0,
    }


def _print_traj_point(key, name, point):
    if not point:
        utils.print_flush(f"[eval] TRAJ {key} {name} NONE")
        return
    utils.print_flush(
        f"[eval] TRAJ {key} {name} step={point['step']} | "
        f"acc {point['test_origin']:.3f} -> {point['test_acc']:.3f} "
        f"({point['test_dacc_pp'] / 100.0:+.3f}) | params x{point['param']:.3f} | "
        f"FLOPs x{point['flop']:.3f} | val Δacc {point['val_dacc_pp']:+.2f} pp")


def _print_traj_summary(net_path, picked):
    name = os.path.basename(net_path)
    keys = ("floor_hold", "floor_cross", "val_best", "terminal")
    if "size_match" in picked:
        keys = keys + ("size_match",)
    for key in keys:
        _print_traj_point(key, name, picked.get(key))


def _keep_final_ft_candidates(env, point, candidates, size_points, tau_pp):
    """
    Copy the pruned net when ``point`` becomes the running ``val_best`` (same key and tie order as
    ``fortify.select_trajectory_points``) or first reaches a size point. Returns the new labels.
    """
    from src import fortify as fortify_mod
    labels = []
    key = (float(point["param"]), float(point["flop"]), -float(point["val_dacc_pp"]))
    held = candidates.get("val_best")
    if (int(point["step"]) >= 0 and float(point["val_dacc_pp"]) + 1e-12 >= -float(tau_pp)
            and (held is None or key < held["key"])):
        labels.append("val_best")
    for kind, target in size_points:
        label = fortify_mod.size_point_label(kind, target)
        if label not in candidates and float(point[kind]) <= float(target) + 1e-12:
            labels.append(label)
    if labels:
        model = copy.deepcopy(env.current_model).cpu()
        for label in labels:
            candidates[label] = {"point": dict(point), "model": model, "key": key}
    return labels


_FINAL_FT_ENV_KEYS = ("SPECTRA_FT_OPTIM", "SPECTRA_FT_SGD_LR", "SPECTRA_FT_MOMENTUM", "SPECTRA_FT_WD",
                      "SPECTRA_FT_COSINE", "SPECTRA_FT_SCHEDULE", "SPECTRA_FT_MIXUP",
                      "SPECTRA_FT_LABEL_SMOOTH", "SPECTRA_FT_KD")


def _final_ft(env, net_path, label, candidate, epochs, save_dir=None, scratch=False):
    """
    Fine-tune a copy of one candidate with the fixed final recipe; print and record TEST/val.
    ``scratch=True`` re-initialises the copy first and trains it with the scratch budget and lr.
    """
    from src import fortify as fortify_mod
    name = os.path.basename(net_path)
    point = candidate["point"]
    model = copy.deepcopy(candidate["model"])
    if scratch:
        traj_models.reinit_parameters(model)
        epochs = fortify_mod.eval_final_ft_scratch_epochs()
        lr = fortify_mod.eval_final_ft_scratch_lr()
    else:
        lr = fortify_mod.eval_final_ft_lr()
    for param in model.parameters():
        param.requires_grad = True
    kd = fortify_mod.eval_final_ft_kd()
    if kd and getattr(env, "kd_teacher", None) is None:
        # env.reset() builds a teacher only under the walk's SPECTRA_FT_KD; without one the handler skips KD
        teacher = copy.deepcopy(env.data_dict[env.selected_net_path][0]).eval()
        for param in teacher.parameters():
            param.requires_grad = False
        env.kd_teacher = teacher
    loader, aug = utils.final_ft_train_loader(env.train_loader, fortify_mod.eval_final_ft_batch())
    schedule, warmup = fortify_mod.eval_final_ft_schedule(), fortify_mod.eval_final_ft_warmup()
    select = fortify_mod.eval_final_ft_select()
    graph = fortify_mod.eval_final_ft_cuda_graph()
    recipe = dict(zip(_FINAL_FT_ENV_KEYS, ("sgd", f"{lr:g}", "0.9", "5e-4", "1", schedule, "0", "0",
                                           "1" if kd else "0")))
    if schedule:
        recipe["SPECTRA_FT_WARMUP_EPOCHS"] = f"{warmup:g}"
    shape = f"warmcos w{warmup:g}" if schedule else "cos"
    saved = {k: os.environ.get(k) for k in recipe}
    t0 = time.perf_counter()
    try:
        os.environ.update(recipe)
        env.create_learning_handler(model).train_model(
            loader, allow_reinit_retry=False, max_epochs=epochs, patience=epochs + 1,
            tag=f"final FT {label}", **({"keep_last": True} if select == "last" else {}),
            **({"cuda_graph": True} if graph else {}))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    handler = env.create_learning_handler(model)
    test_acc = float(handler.evaluate_model(env.test_loader))
    val_acc = float(handler.evaluate_model(env.val_loader))
    minutes = (time.perf_counter() - t0) / 60.0
    test_origin, val_origin = float(point["test_origin"]), float(point["val_origin"])
    utils.print_flush(
        f"[eval] TRAJ final_ft {label} {name} step={point['step']} | "
        f"acc {test_origin:.3f} -> {test_acc:.3f} ({test_acc - test_origin:+.3f}) | "
        f"params x{point['param']:.3f} | FLOPs x{point['flop']:.3f} | "
        f"val Δacc {(val_acc - val_origin) * 100.0:+.2f} pp | walk acc {float(point['test_acc']):.3f} | "
        f"sgd lr={lr:g} m=0.9 wd=5e-4 {shape} e{epochs}{' keep=last' if select else ''} "
        f"bs={loader.batch_size} aug={aug} kd={int(kd)}{' graph=1' if graph else ''} "
        f"init={'scratch' if scratch else 'inherit'} | {minutes:.1f} min")
    run_recorder.record(
        "eval_traj_final_ft", network=net_path, label=label, step=int(point["step"]),
        param=float(point["param"]), flop=float(point["flop"]),
        test_origin=test_origin, test_walk=float(point["test_acc"]), test_final=test_acc,
        val_origin=val_origin, val_walk=float(point["val_acc"]), val_final=val_acc,
        epochs=int(epochs), lr=float(lr), batch=int(loader.batch_size), aug=aug, kd=bool(kd),
        scratch=bool(scratch), schedule=schedule or "cos", warmup=float(warmup) if schedule else 0.0,
        select=select or "train_loss", minutes=round(minutes, 2), **({"graph": True} if graph else {}))
    if save_dir:
        traj_models.save_candidate(
            model, traj_models.candidate_stem(save_dir, name, label, point["step"], f"__ft{epochs}"), point,
            {"network": net_path, "label": label, "final_ft": {
                "epochs": int(epochs), "lr": float(lr), "scratch": bool(scratch), "kd": bool(kd),
                "schedule": schedule or "cos", "warmup": float(warmup) if schedule else 0.0,
                "select": select or "train_loss", "test_final": test_acc, "val_final": val_acc}})
    return test_acc


def _run_final_ft(env, net_path, picked, candidates, epochs):
    """``SPECTRA_EVAL_FINAL_FT_EPOCHS`` / ``SPECTRA_EVAL_SAVE_TRAJ_MODELS`` after one TRAJ walk."""
    from src import fortify as fortify_mod
    name = os.path.basename(net_path)
    best, held = picked.get("val_best"), candidates.get("val_best")
    if held is not None and (best is None or int(held["point"]["step"]) != int(best["step"])):
        utils.print_flush(
            f"[eval] TRAJ final_ft {name}: kept val_best copy is step {held['point']['step']}, "
            f"selection says {best['step'] if best else None}; dropping the copy")
        candidates.pop("val_best")
    if fortify_mod.eval_final_ft_origin() and picked.get("origin") is not None:
        origin_model = env.data_dict[env.selected_net_path][0]
        candidates["origin"] = {"point": dict(picked["origin"]),
                                "model": copy.deepcopy(origin_model).cpu(), "key": None}
    save_dir = None
    if fortify_mod.eval_save_traj_models():
        save_dir = os.path.join(run_recorder.recorder().run_dir, "traj_models")
        os.makedirs(save_dir, exist_ok=True)
        for label, cand in candidates.items():
            traj_models.save_candidate(
                cand["model"], traj_models.candidate_stem(save_dir, name, label, cand["point"]["step"]),
                cand["point"], {"network": net_path, "label": label})
    import src.plan_proxies as plan_proxies
    names = plan_proxies.proxies()
    if names:
        measured = set()
        for label, cand in candidates.items():
            step = int(cand["point"]["step"])
            if label == "origin" or step in measured:
                continue
            measured.add(step)
            plan_proxies.measure(env, net_path, label, cand, names)
    if epochs <= 0:
        return
    scratch = fortify_mod.eval_final_ft_scratch()
    inits = {"": (False,), "both": (False, True), "only": (True,)}[scratch]
    done_steps = {}
    for label, cand in candidates.items():
        step = int(cand["point"]["step"])
        if step in done_steps:
            utils.print_flush(f"[eval] TRAJ final_ft {label} {name} step={step} | same point as "
                              f"{done_steps[step]}; not fine-tuned twice")
            continue
        for from_scratch in inits:
            tag = f"{label}+scratch" if from_scratch else label
            try:
                _final_ft(env, net_path, tag, cand, epochs, save_dir=save_dir, scratch=from_scratch)
            except Exception as error:  # noqa: BLE001 - one candidate must not cost the others
                logging_utils.exception(f"[eval] TRAJ final_ft {tag} {name} failed; continuing")
                run_recorder.issue("final_ft_failed", f"{type(error).__name__}: {error}",
                                   network=net_path, label=tag)
        done_steps[step] = label


def _final_ft_from_saved(env, net_path, save_dir, epochs):
    """``SPECTRA_EVAL_FINAL_FT_FROM``: final fine-tune of a saved walk's candidates, no new walk."""
    name = os.path.basename(net_path)
    loaded = traj_models.load_candidates(save_dir, name, env.data_dict[env.selected_net_path][0])
    loaded.pop("origin", None)
    test_new, test_orig, _ = env.score_test_loader()
    origin = _traj_capture(env, -1, 1.0, test_new, test_orig)
    utils.print_flush(f"[eval] TRAJ final_ft from {save_dir}: {sorted(loaded) or 'no candidates'} for {name}")
    if not loaded:
        return
    for label, cand in loaded.items():
        drift = float(cand["point"]["test_origin"]) - float(origin["test_origin"])
        if abs(drift) > 1e-9:
            utils.print_flush(f"[eval] TRAJ final_ft {label} {name}: saved test_origin differs by "
                              f"{drift:+.4f} from this job's (split or checkpoint changed?)")
    best = loaded.get("val_best")
    picked = {"origin": origin, "val_best": best["point"] if best else None}
    _run_final_ft(env, net_path, picked, loaded, epochs)


def apply_policy_config(args):
    """
    Replay a trained actor under the state/action contract it was trained with.

    Trainers write ``agent_checkpoints/policy_config.json`` (A2CAgentReinforce.write_policy_config).
    When ``--actor_checkpoint_path`` points into such a directory (or a snapshot copy of it),
    the contract keys (state alignment, group-once, slack/budget channels, encoder kind and
    dropout, ranking default, rate menu, per-action rankings) are pinned here — *before*
    ``StaticConf`` exists — and every difference is printed. ``SPECTRA_POLICY_CONFIG=0``
    disables the pinning (explicit A/B against the training contract). Frozen legacy actors
    have no such file and are untouched.
    """
    import json
    if os.environ.get("SPECTRA_POLICY_CONFIG", "auto").strip().lower() in ("0", "off", "false", "no"):
        return None
    actor = getattr(args, "actor_checkpoint_path", None)
    if not actor:
        return None
    path = os.path.join(os.path.dirname(os.path.abspath(actor)), "policy_config.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    changed = []
    env_cfg = cfg.get("env") or {}
    for key, value in env_cfg.items():
        if value is None:
            continue
        current = os.environ.get(key)
        if current != str(value):
            os.environ[key] = str(value)
            changed.append(f"{key}: {current!r} -> {value!r}")
    for key in A2CAgentReinforce.ACTION_GEOMETRY_KEYS:
        if key not in env_cfg and os.environ.get(key, "").strip() not in ("", "0"):
            changed.append(f"{key}: {os.environ[key]!r} -> unset (actor trained without it)")
            os.environ.pop(key)
    rates = cfg.get("compression_rates")
    if rates and [float(x) for x in rates] != [float(x) for x in args.compression_rates]:
        changed.append(f"compression_rates: {args.compression_rates} -> {rates}")
        args.compression_rates = [float(x) for x in rates]
    ranks = cfg.get("action_rankings")
    if ranks is not None:
        wanted = [str(r) if r else "none" for r in ranks]
        current = [str(r) for r in (args.action_rankings or [])]
        if not args.action_rankings and all(r == "none" for r in wanted):
            wanted = None
        if wanted is not None and current != wanted:
            changed.append(f"action_rankings: {args.action_rankings} -> {wanted}")
            args.action_rankings = wanted
    # V4-1 factored head: ranking menu of the second policy head.
    menu = cfg.get("ranking_menu")
    if cfg.get("factored_head") and menu:
        wanted_menu = [str(m) for m in menu]
        if [str(m) for m in (args.ranking_menu or [])] != wanted_menu:
            changed.append(f"ranking_menu: {args.ranking_menu} -> {wanted_menu}")
            args.ranking_menu = wanted_menu
    # Passes: v3 actors train with --passes 2 (band-edge exposure). Replay with the same
    # number of passes unless the submitter pinned SPECTRA_EVAL_PASSES explicitly.
    passes = cfg.get("passes")
    if (passes is not None and not os.environ.get("SPECTRA_EVAL_PASSES", "").strip()
            and int(passes) != int(getattr(args, "passes", passes))):
        changed.append(f"passes: {args.passes} -> {int(passes)}")
        args.passes = int(passes)
    # P8: a NEON-C actor (trained with layer replacement) must be replayed with it. The env
    # pins above already copied SPECTRA_FT_REINIT_* when the trainer exported them; an actor
    # whose config predates P8 has no such key and stays on the recipe the submitter chose.
    recipe = cfg.get("ft_recipe")
    if recipe:
        changed.append(f"ft_recipe (trained): {recipe}")
    utils.print_flush(f"[policy_config] {path}: "
                      + ("; ".join(changed) if changed else "matches the current flags"))
    return path


def _counterfactual_states(state):
    """
    Three content-perturbed copies of an encoder state dict (V6 representation probe).

    ``zero_layers``: layer feature rows zeroed (types, positions, target marker, action tokens
    kept) — does the policy read layer *content*? ``shuffle_layers``: feature rows and types
    permuted across positions with a fixed seed — does *which layer has which statistics*
    matter, or only the sequence as a bag? ``blind``: layer features and action-cost tokens
    zeroed — the policy sees only depth, layer types and the marker.
    """
    feats = state["layer_features"]
    n = int(feats.size(0))
    zero = dict(state)
    zero["layer_features"] = torch.zeros_like(feats)
    gen = torch.Generator(device="cpu").manual_seed(7919 * n + 17)
    perm = torch.randperm(n, generator=gen).to(feats.device)
    shuffled = dict(state)
    shuffled["layer_features"] = feats[perm]
    if "layer_types" in state and state["layer_types"] is not None and state["layer_types"].numel() == n:
        shuffled["layer_types"] = state["layer_types"][perm]
    blind = dict(zero)
    costs = state.get("action_costs")
    if costs is not None and torch.is_tensor(costs) and costs.numel():
        blind["action_costs"] = torch.zeros_like(costs)
    return {"zero_layers": zero, "shuffle_layers": shuffled, "blind": blind}


def counterfactual_probe(agent, state, legal, real_rate_idx, conf, fortify_mod):
    """
    ``SPECTRA_EVAL_COUNTERFACTUAL=1``: argmax of the frozen actor on the counterfactual states
    under the same legal mask, plus the real policy's max probability. Returns None when the
    flag is off or the state is not an encoder dict (legacy BERT tensor states).
    """
    if not fortify_mod.eval_counterfactual():
        return None
    if not isinstance(state, dict) or "layer_features" not in state:
        return None
    out = {"real": int(real_rate_idx)}
    with torch.no_grad():
        dist = agent.actor_model(state)
        masked = fortify_mod.mask_policy(dist, legal)
        probs = masked.rate.probs if fortify_mod.is_factored_dist(masked) else masked.probs
        out["pmax"] = round(float(probs.max().item()), 4)
        for name, variant in _counterfactual_states(state).items():
            rate_idx, _, _ = fortify_mod.pick_action(
                agent.actor_model(variant), legal, deterministic=True, device=conf.device)
            out[name] = int(rate_idx)
    out["content_used"] = int(out["zero_layers"] != out["real"] or out["shuffle_layers"] != out["real"])
    out["state_used"] = int(out["blind"] != out["real"])
    utils.print_flush(
        f"[cf] act={out['real']} zero={out['zero_layers']} shuf={out['shuffle_layers']} "
        f"blind={out['blind']} pmax={out['pmax']} content_used={out['content_used']} "
        f"state_used={out['state_used']}")
    try:
        import src.run_recorder as _recorder
        _recorder.record("counterfactual", **out)
    except Exception:
        pass
    return out


def _decide_timer(fortify_mod):
    """A ``step.decide`` stage around one eval-walk decision when ``SPECTRA_TIME_DECIDE=1``."""
    if not fortify_mod.time_decide():
        return contextlib.nullcontext()
    return logging_utils.stage("step.decide", level=logging.DEBUG)


def _heuristic_action(legal, conf, fortify_mod, eval_policy, env=None):
    with _decide_timer(fortify_mod):
        if eval_policy == "alloc":
            from src import alloc_walk
            return alloc_walk.action(env, legal, conf.compression_rates_dict, conf.device)
        return fortify_mod.heuristic_eval_action(
            legal, conf.compression_rates_dict, policy=eval_policy, device=conf.device)


def _actor_action(agent, state, legal, conf, fortify_mod):
    """(rate action tensor, ranking index or None) from the frozen policy, plain or factored."""
    with _decide_timer(fortify_mod):
        with torch.no_grad():
            dist = agent.actor_model(state)
        rate_idx, rank_idx, _ = fortify_mod.pick_action(
            dist, legal, deterministic=fortify_mod.eval_deterministic(), device=conf.device)
    counterfactual_probe(agent, state, legal, rate_idx, conf, fortify_mod)
    return torch.tensor([rate_idx], device=conf.device), rank_idx


def _ranking_for(conf, fortify_mod, rate_idx, rank_idx):
    """Ranking to apply: factored head -> menu entry (None on identity), else per-action table."""
    menu = list(getattr(conf, "ranking_menu", None) or [])
    if fortify_mod.factored_head() and menu:
        return None if rank_idx is None else menu[int(rank_idx) % len(menu)]
    return conf.action_rankings_dict.get(int(rate_idx))


def evaluate_model(mode, agent, train_dict=None, test_dict=None, fold_idx="N/A"):
    """
    Evaluate models using intra-model (train/test) and inter-model (cross-validation).

    Args:
        mode (str):                 'train' or 'test' (used for intra-model evaluation).
        agent (A2CAgentReinforce):  Trained RL agent.
        train_dict (dict):          {network_path: (model, dataset_name)} for training
                                    (Used for inter-model evaluation via cross-validation).
        test_dict (dict):           {network_path: (model, dataset_name)} for testing
                                    (Used for inter-model evaluation via cross-validation).
        fold_idx (int, optional):   The index of the cross-validation fold.

    Returns:
        DataFrame: Evaluation results.
    """
    conf = StaticConf.get_instance().conf_values

    # Use intra-model evaluation if no cross-validation dicts are provided
    if not all([train_dict, test_dict]):
        train_dict = conf.input_dict
        test_dict = conf.input_dict

    env = NetworkEnv(train_dict, mode, fold_idx)

    # Deterministic eval also takes the policy out of train mode: the state encoder carries
    # dropout, which otherwise perturbs the frozen agent's logits at TEST time.
    from src import fortify as _fortify_mod
    _fortify_mod.set_policy_eval_mode(getattr(agent, "actor_model", None),
                                      getattr(agent, "critic_model", None))

    # Under a multi-GPU launch each rank evaluates a disjoint slice of the networks and
    # writes its own results file, so the work is split rather than duplicated
    world_size, rank = ddp.get_world_size(), ddp.get_rank()
    shard = list(test_dict.items())[rank::world_size]
    if world_size > 1:
        utils.print_flush(f"Rank {rank} evaluating {len(shard)}/{len(test_dict)} networks")

    for model_idx, (net_path, (net_model, net_loaders)) in enumerate(shard):
        utils.print_flush(f"Evaluating model {model_idx + 1}/{len(shard)}: {net_path}")

        # One failing network must not abort the whole evaluation sweep: the remaining
        # networks still produce results, and the failure is recorded for triage.
        try:
            with logging_utils.context(net=os.path.basename(net_path), phase=mode):
                # Reset environment with test model instead of selecting from train_dict
                env.t_start = time.perf_counter()
                state = env.reset(test_net_path=net_path, test_model=net_model, test_loaders=net_loaders)
                done = False

                env._budget_logged = False
                from src import fortify as fortify_mod
                eval_policy = fortify_mod.eval_policy_name()
                traj = fortify_mod.eval_trajectory_enabled()
                if model_idx == 0:
                    utils.print_flush(
                        f"[eval] policy={eval_policy} det={int(fortify_mod.eval_deterministic())} "
                        f"lookahead={int(fortify_mod.eval_lookahead_enabled())} "
                        f"traj={int(traj)} "
                        f"min_param={fortify_mod.eval_min_param_ratio():.2f} "
                        f"min_flop={fortify_mod.eval_min_flop_ratio():.2f} "
                        f"align={'next' if fortify_mod.state_align_next() else 'prev'} "
                        f"group_once={int(fortify_mod.group_once_per_pass())} "
                        f"slack={int(fortify_mod.state_slack())} "
                        f"groupcost={int(fortify_mod.state_groupcost())} passes={conf.passes} "
                        f"factored={int(bool(fortify_mod.factored_head() and getattr(conf, 'ranking_menu', None)))} "
                        f"ranking_menu={list(getattr(conf, 'ranking_menu', None) or [])} "
                        f"rankings={[conf.action_rankings_dict.get(i) for i in sorted(conf.action_rankings_dict)]} "
                        f"ft_recipe={fortify_mod.ft_recipe(bool(conf.train_compressed_layer_only))} "
                        f"refresh_all={int(fortify_mod.refresh_all_features())} "
                        f"ladder={fortify_mod.width_ladder_max()} "
                        f"dedupe={int(fortify_mod.action_dedupe())} "
                        f"protect_streams={int(fortify_mod.protect_streams())} "
                        f"rollback={int(fortify_mod.eval_rollback())} "
                        f"size_match={fortify_mod.eval_size_match() or 'off'} "
                        f"group_first={fortify_mod.ft_group_first_epochs()} "
                        f"val_from_test={utils.val_from_test_fraction():g} "
                        f"batch={env.train_loader.batch_size} "
                        f"size_points={','.join(f'{k}:{t:g}' for k, t in fortify_mod.eval_size_points()) or 'off'} "
                        f"final_ft={fortify_mod.eval_final_ft_epochs()}"
                        f"{'+origin' if fortify_mod.eval_final_ft_origin() else ''}"
                        f"{'+kd' if fortify_mod.eval_final_ft_kd() else ''}"
                        f"{'+scratch:' + fortify_mod.eval_final_ft_scratch() if fortify_mod.eval_final_ft_scratch() else ''}"
                        f"{' from=' + fortify_mod.eval_final_ft_from() if fortify_mod.eval_final_ft_from() else ''}"
                        f"{' proxy=' + ','.join(f'{t:g}' for t in fortify_mod.eval_proxy_fidelity()) if fortify_mod.eval_proxy_fidelity() else ''}"
                        f" fixed_target={int(fortify_mod.fixed_target())} state_sens={int(fortify_mod.state_sens())}")
                    match = fortify_mod.eval_size_match()
                    if fortify_mod.fixed_target() and (match is None or match[0] != "param"):
                        utils.print_flush(
                            "[eval] WARNING: SPECTRA_FIXED_TARGET without SPECTRA_EVAL_SIZE_MATCH=param:<keep>; "
                            "the actor is told the deepest param size point (else 0.6) and the walk is "
                            "not ended at it")
                import src.state_dump as state_dump
                if mode == EVAL_TEST and state_dump.enabled():
                    state_dump.dump(env, net_path, net_model, net_loaders)
                    continue
                if traj and mode == EVAL_TEST and fortify_mod.eval_final_ft_from():
                    _final_ft_from_saved(env, net_path, fortify_mod.eval_final_ft_from(),
                                         fortify_mod.eval_final_ft_epochs())
                    continue
                size_match = fortify_mod.eval_size_match() if traj else None
                rollback = traj and fortify_mod.eval_rollback() and eval_policy not in ("actor",)
                size_points = fortify_mod.eval_size_points() if traj else ()
                final_ft_epochs = fortify_mod.eval_final_ft_epochs() if traj and mode == EVAL_TEST else 0
                keep_models = traj and mode == EVAL_TEST and (
                    final_ft_epochs > 0 or fortify_mod.eval_save_traj_models())
                candidates = {}
                # Paper TEST walks every remaining prunable row. The size floor
                # identity-pads unless SPECTRA_EVAL_TRAJECTORY=1, which labels a
                # ~0.70 hold then continues. Train rollout_limit must not apply
                # here (5-step trains vs full eval was the 20945744 defect).
                traj_points = []
                traj_phase_b = False
                last_test = None
                step_i = 0
                proxy_targets = fortify_mod.eval_proxy_fidelity() if traj and mode == EVAL_TEST else ()
                proxy_done = set()
                if traj and mode == EVAL_TEST:
                    test_new, test_orig, _ = env.score_test_loader()
                    last_test = (test_new, test_orig)
                    traj_points.append(
                        _traj_capture(env, -1, 1.0, test_new, test_orig))
                while not done:
                    chosen_rank = None  # factored head: ranking index chosen with the rate
                    legal = env.legal_action_mask(device=conf.device)
                    min_ratio = fortify_mod.eval_min_param_ratio()
                    at_budget, floor_kind = fortify_mod.eval_at_size_floor(env)
                    if traj:
                        if eval_policy not in ("actor",):
                            action = _heuristic_action(legal, conf, fortify_mod, eval_policy, env)
                        else:
                            action, chosen_rank = _actor_action(agent, state, legal, conf, fortify_mod)
                        if not traj_phase_b:
                            before = int(action.item())
                            guarded = fortify_mod.action_respecting_param_floor(
                                env, action, legal, conf.compression_rates_dict,
                                min_ratio, conf.device)
                            if fortify_mod.trajectory_release_floor(
                                    at_budget, before, int(guarded.item())):
                                test_new, test_orig, _ = env.score_test_loader()
                                last_test = (test_new, test_orig)
                                traj_points.append(
                                    _traj_capture(env, step_i, 1.0, test_new, test_orig))
                                traj_phase_b = True
                                if not env._budget_logged:
                                    utils.print_flush(
                                        f"[eval] TRAJ floor-hold x{env.param_ratio():.3f}; "
                                        f"continuing without identity-pad")
                                    env._budget_logged = True
                            else:
                                action = guarded
                    elif at_budget:
                        if not env._budget_logged:
                            min_flop = fortify_mod.eval_min_flop_ratio()
                            if floor_kind == "flop":
                                utils.print_flush(
                                    f"[eval] flop budget x{env.flops_ratio():.3f} "
                                    f"<= {min_flop}; identity-pad remaining steps")
                            else:
                                utils.print_flush(
                                    f"[eval] param budget x{env.param_ratio():.3f} "
                                    f"<= {min_ratio}; identity-pad remaining steps")
                            env._budget_logged = True
                        identity = next(
                            (i for i, r in conf.compression_rates_dict.items()
                             if abs(float(r) - 1.0) < 1e-9),
                            0)
                        action = torch.tensor([identity], device=conf.device)
                    elif eval_policy not in ("actor",):
                        action = _heuristic_action(legal, conf, fortify_mod, eval_policy, env)
                    else:
                        action, chosen_rank = _actor_action(agent, state, legal, conf, fortify_mod)

                    if (not traj
                            and fortify_mod.eval_lookahead_enabled()
                            and not at_budget):
                        before = int(action.item())
                        action = fortify_mod.action_respecting_param_floor(
                            env, action, legal, conf.compression_rates_dict,
                            min_ratio, conf.device)
                        if int(action.item()) != before:
                            min_flop = fortify_mod.eval_min_flop_ratio()
                            utils.print_flush(
                                f"[eval] lookahead: rate "
                                f"{conf.compression_rates_dict[before]} would "
                                f"drop below param {min_ratio:.3f}"
                                f"{f' / flop {min_flop:.3f}' if min_flop > 0 else ''}"
                                f"; using "
                                f"{conf.compression_rates_dict[int(action.item())]}")
                    if (not traj
                            and fortify_mod.eval_prefer_param_per_flop()
                            and fortify_mod.eval_min_flop_ratio() > 0
                            and not at_budget):
                        before = int(action.item())
                        action = fortify_mod.action_preferring_param_per_flop(
                            env, action, legal, conf.compression_rates_dict,
                            min_ratio, conf.device)
                        if int(action.item()) != before:
                            utils.print_flush(
                                f"[eval] prefer Δparams/ΔFLOPs: "
                                f"{conf.compression_rates_dict[before]} -> "
                                f"{conf.compression_rates_dict[int(action.item())]}")
                    compression_rate = conf.compression_rates_dict[int(action.item())]
                    ranking = _ranking_for(conf, fortify_mod, int(action.item()), chosen_rank)
                    is_cut = abs(float(compression_rate) - 1.0) >= 1e-9
                    snapshot = env.rollback_snapshot() if (rollback and is_cut) else None
                    next_state, reward, done = env.step(compression_rate, ranking=ranking)
                    rolled_back = False
                    if snapshot is not None:
                        val_pp = (float(env.last_val_acc) - float(env.original_acc)) * 100.0
                        if val_pp < -float(conf.allowed_acc_reduction):
                            locked = env.rollback_to(snapshot)
                            rolled_back = True
                            utils.print_flush(
                                f"[eval] rollback step={step_i} rate={compression_rate} "
                                f"val Δacc {val_pp:+.2f} pp < -τ; cut undone, "
                                f"{len(locked)} layer(s) locked for the walk")
                            run_recorder.record("eval_rollback", network=net_path, step=step_i,
                                                rate=float(compression_rate), val_dacc_pp=val_pp,
                                                locked=locked)
                        snapshot = None
                    if traj and mode == EVAL_TEST:
                        if is_cut and not rolled_back:
                            test_new, test_orig, _ = env.score_test_loader()
                            last_test = (test_new, test_orig)
                            traj_points.append(
                                _traj_capture(env, step_i, compression_rate,
                                              test_new, test_orig))
                            if keep_models:
                                _keep_final_ft_candidates(env, traj_points[-1], candidates, size_points,
                                                          float(conf.allowed_acc_reduction))
                            if proxy_targets:
                                from src import proxy_fidelity
                                proxy_fidelity.maybe_run(env, net_path, traj_points[-1], proxy_targets,
                                                         proxy_done)
                        elif done and last_test is not None:
                            traj_points.append(
                                _traj_capture(env, step_i, compression_rate,
                                              last_test[0], last_test[1]))
                        if (size_match is not None and not done and traj_points
                                and float(traj_points[-1][size_match[0]]) <= size_match[1] + 1e-12):
                            utils.print_flush(
                                f"[eval] TRAJ size_match {size_match[0]} "
                                f"x{traj_points[-1][size_match[0]]:.3f} <= {size_match[1]}; walk ends")
                            done = True
                    step_i += 1
                    state = next_state
                if traj and mode == EVAL_TEST:
                    picked = fortify_mod.select_trajectory_points(
                        traj_points,
                        min_param=fortify_mod.eval_min_param_ratio(),
                        tau_pp=float(conf.allowed_acc_reduction),
                        size_match=size_match)
                    _print_traj_summary(net_path, picked)
                    sized = fortify_mod.select_size_points(traj_points, size_points)
                    for label, point in sized.items():
                        _print_traj_point(label, os.path.basename(net_path), point)
                    run_recorder.record(
                        "eval_traj_summary",
                        network=net_path,
                        floor_hold=picked.get("floor_hold"),
                        floor_cross=picked.get("floor_cross"),
                        val_best=picked.get("val_best"),
                        terminal=picked.get("terminal"),
                        size_match=picked.get("size_match"),
                        size_points=sized or None,
                        points=traj_points)
                    if keep_models:
                        _run_final_ft(env, net_path, picked, candidates, final_ft_epochs)
                        candidates.clear()
        except Exception as error:
            logging_utils.exception(f"Evaluation of {net_path} failed; continuing with the rest")
            run_recorder.issue("eval_network_failed", f"{type(error).__name__}: {error}",
                               network=net_path, eval_mode=mode)


def main():
    """ Main function for training and evaluating the A2C agent. """
    conf = StaticConf.get_instance().conf_values

    with logging_utils.stage("agent.construct"):
        agent = A2CAgentReinforce()

    utils.print_flush(f"Starting test: {conf.test_name}")

    # Both actor+critic paths historically meant "eval only". For warm-start continued
    # training set SPECTRA_CONTINUE_TRAIN=1 (loads weights, still runs agent.train()).
    # Heuristic eval policies (SPECTRA_EVAL_POLICY=l1|mild|random) skip training and
    # do not need checkpoints — they only pick *how much* to cut this layer. The
    # environment still ranks which channels die (default L1, Li et al. 2017).
    continue_train = os.environ.get("SPECTRA_CONTINUE_TRAIN", "").strip().lower() in (
        "1", "true", "yes")
    from src import fortify as fortify_mod
    heuristic_eval = fortify_mod.eval_policy_name() not in ("actor",)
    skip_train = heuristic_eval or os.environ.get("SPECTRA_SKIP_TRAIN", "").strip().lower() in (
        "1", "true", "yes")
    pretrained = bool(conf.actor_checkpoint_path and conf.critic_checkpoint_path)
    if skip_train and not continue_train:
        utils.print_flush(
            f"Skipping training "
            f"(eval_policy={fortify_mod.eval_policy_name()}, SPECTRA_SKIP_TRAIN set).")
    elif pretrained and not continue_train:
        utils.print_flush(
            f"Agent is pre-trained, training is skipped "
            f"(actor_checkpoint={conf.actor_checkpoint_path}, "
            f"critic_checkpoint={conf.critic_checkpoint_path}). "
            f"Set SPECTRA_CONTINUE_TRAIN=1 to warm-start training from these weights.")
    else:
        if pretrained and continue_train:
            utils.print_flush(
                f"Warm-start training from actor={conf.actor_checkpoint_path} "
                f"critic={conf.critic_checkpoint_path}")
        with logging_utils.stage("phase.train"):
            with logging_utils.context(phase="train"):
                agent.train()

    # Perform standard intra-model evaluation.
    # eval_train then eval_test is two full prune+FT walks; both fine-tune on the
    # train loader and differ mainly in the final accuracy loader. Quote eval_test
    # only. SPECTRA_SKIP_EVAL_TRAIN=1 drops the first walk (~2× remaining eval wall).
    eval_modes = []
    if fortify_mod.skip_eval():
        utils.print_flush(
            "SPECTRA_SKIP_EVAL=1: skipping in-job eval (chained child quotes eval_test).")
    else:
        eval_modes = [EVAL_TEST] if fortify_mod.skip_eval_train() else [EVAL_TRAIN, EVAL_TEST]
        if fortify_mod.skip_eval_train():
            utils.print_flush(
                "SPECTRA_SKIP_EVAL_TRAIN=1: skipping eval_train walk (quote eval_test only).")
    for mode in eval_modes:
        with logging_utils.stage(f"phase.{mode}"):
            with logging_utils.context(phase=mode):
                evaluate_model(mode, agent)

    # Optionally, perform inter-model evaluation via cross-validation
    if conf.n_splits:  # Default is 0 (no CV), recommended value is 10
        utils.print_flush(f"Starting {conf.n_splits}-Fold Cross-Validation")
        folds = utils.get_cross_validation_splits(conf.input_dict)

        for fold_idx, (train_dict, test_dict) in enumerate(folds):
            with logging_utils.stage(f"phase.cv_fold_{fold_idx + 1}"):
                with logging_utils.context(phase="cv", fold=fold_idx + 1):
                    evaluate_model(EVAL_TEST, agent, train_dict, test_dict, fold_idx + 1)
        utils.print_flush("DONE Cross-Validation")


if __name__ == "__main__":
    # Logging is configured before anything else so that even an argument-parsing failure or
    # an import-time crash lands in the run directory with a full traceback.
    RUN_DIR = logging_utils.setup()
    logging_utils.start_heartbeat()
    utils.print_flush(f"SPECTRA run {logging_utils.run_id()} -> {RUN_DIR}")
    logging_utils.log_environment()

    # Anomaly detection roughly triples backward-pass cost, so it is opt-in via
    # SPECTRA_DETECT_ANOMALY=1 rather than always-on.
    if os.environ.get("SPECTRA_DETECT_ANOMALY", "").strip() in ("1", "true", "True"):
        torch.autograd.set_detect_anomaly(True)

    # GPUs have Tensor Cores capable of speeding up float32 matmul ops (used in conv/linear layers),
    # but PyTorch doesn't enable them by default. This increases performance without needing AMP
    torch.set_float32_matmul_precision('high')

    # Stabilizing torch.compile() in practice
    torch._dynamo.config.suppress_errors = True

    # Helps Conv2D tune performance across batch shapes
    torch.backends.cudnn.benchmark = True

    #Filtering out a harmless warning
    warnings.filterwarnings("ignore", message="xindex is not in var_ranges")

    args = utils.extract_args_from_cmd()
    apply_policy_config(args)
    utils.print_flush(args)

    assert args.train_split + args.val_split < 1, f"{args.train_split=} + {args.val_split=} >= 1"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    passes = f'_passes_{args.passes}' if args.passes else ""
    n_splits = f'n_splits_{args.n_splits}_' if args.n_splits else ""
    train_compressed_layer_only = "_train_compressed-layer-only" if args.train_compressed_layer_only else ""
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace("/", "-").replace(":", "-")

    # Every dataset is materialised once here and shared by all networks referencing it
    with logging_utils.stage("startup.preload_datasets", datasets=",".join(args.datasets or [])):
        preloaded_dataloaders_dict = utils.preload_datasets(args.datasets, args.train_split, args.val_split)

    utils.init_conf_values(
        # args.input_dict and args.database are left out due to file name's length limitation
        test_name=f'SPECTRA{train_compressed_layer_only}_acc-red_{args.allowed_acc_reduction}_'
                  f'gamma_{args.discount_factor}_lr_{args.learning_rate}_rollout-lim_{args.rollout_limit}_'
                  f'num-epochs_{args.num_epochs}{passes}_comp-rates_{args.compression_rates}_{n_splits}'
                  f'train_{args.train_split}_val_{args.val_split}_seed_{args.seed}_{dt_string}',
        dataloaders_dict=preloaded_dataloaders_dict,
        input_dict=utils.parse_input_argument(args.input, preloaded_dataloaders_dict),
        database_dict=utils.parse_input_argument(args.database, preloaded_dataloaders_dict),
        actor_checkpoint_path=args.actor_checkpoint_path,
        critic_checkpoint_path=args.critic_checkpoint_path,
        compression_rates_dict=utils.parse_compression_rates(args.compression_rates),
        action_rankings_dict=utils.parse_action_rankings(
            args.action_rankings, utils.parse_compression_rates(args.compression_rates)),
        ranking_menu=utils.parse_ranking_menu(args.ranking_menu),
        train_compressed_layer_only=args.train_compressed_layer_only,
        allowed_acc_reduction=args.allowed_acc_reduction,
        discount_factor=args.discount_factor,
        learning_rate=args.learning_rate,
        rollout_limit=args.rollout_limit,
        passes=args.passes,
        prune=args.prune,
        num_epochs=args.num_epochs,
        runtime_limit=args.runtime_limit,
        seed=args.seed,
        n_splits=args.n_splits,
        train_split=args.train_split,
        val_split=args.val_split,
        save_pruned_checkpoints=args.save_pruned_checkpoints,
        test_ts=dt_string
    )

    # The manifest pins down what was run (config, git commit, GPUs, SPECTRA_* switches), so a
    # results directory can be interpreted months later without the launch command.
    recorder_instance = run_recorder.RunRecorder.instance()
    recorder_instance.write_manifest({
        "argv": sys.argv,
        "args": vars(args),
        "config": run_recorder.config_snapshot(StaticConf.get_instance().conf_values),
    })
    run_recorder.record("run_start", test_name=StaticConf.get_instance().conf_values.test_name)

    RUN_STARTED = time.perf_counter()
    try:
        main()
    except Exception as fatal:
        # The excepthook logs the traceback; this records the terminal status so a summary of
        # a failed run still reports how far it got and why it stopped.
        run_recorder.record("run_end", status="failed",
                            error=f"{type(fatal).__name__}: {fatal}",
                            seconds=round(time.perf_counter() - RUN_STARTED, 2))
        recorder_instance.close()
        raise
    else:
        run_recorder.record("run_end", status="ok",
                            seconds=round(time.perf_counter() - RUN_STARTED, 2),
                            counters=recorder_instance.counters())
        utils.print_flush(f"Run finished in {(time.perf_counter() - RUN_STARTED) / 60:.1f} min; "
                          f"artefacts in {RUN_DIR}")
    finally:
        logging_utils.stop_heartbeat()
        recorder_instance.close()
