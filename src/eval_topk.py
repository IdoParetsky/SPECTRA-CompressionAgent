"""
Top-5 beside the TRAJ TEST lines (``SPECTRA_EVAL_TOP5=1``, v29, default off): the ImageNet evaluation line reports
top-1 and top-5 on TEST and val. Nothing here changes an existing field or line: with the flag off every function
is a no-op, and with it on each captured TRAJ point gains ``test_top5`` / ``test_origin_top5`` / ``val_top5`` /
``val_origin_top5`` (plus the pass's own top-1, ``test_top1_pass`` / ``val_top1_pass``, a check against the TRAJ
top-1 the walk logged) and one extra line is printed under the TRAJ line it belongs to:

    [eval] TRAJ top5 <key> <net> step=<s> | top5 <origin> -> <acc> (<delta>) | val top5 dacc <pp> pp | top1 pass <a> vs TRAJ <b>
    [eval] TRAJ final_ft_top5 <label> <net> step=<s> | top5 <origin> -> <acc> (<delta>) | val top5 dacc <pp> pp | top1 pass <a>

The origin's top-k is read once per network (TEST and val) and cached on the env; a captured point costs one pass
over TEST and one over val (``utils.evaluate_topk``). The TEST split is never read to make a choice: these are
readouts of points the walk already selected on val.
"""

import src.utils as utils

KS = (1, 5)


def enabled() -> bool:
    """``SPECTRA_EVAL_TOP5=1`` (default off)."""
    return utils.env_flag("SPECTRA_EVAL_TOP5")


def _origin(env):
    """``{"test": {k: acc}, "val": {k: acc}}`` of the unpruned network, read once per network and cached on ``env``."""
    cache = getattr(env, "_eval_topk_origin", None)
    if cache is None:
        cache = env._eval_topk_origin = {}
    key = env.selected_net_path
    if key not in cache:
        origin_model = env.data_dict[env.selected_net_path][0]
        device = env.conf.device
        cache[key] = {"test": utils.evaluate_topk(origin_model, env.test_loader, device, KS),
                      "val": utils.evaluate_topk(origin_model, env.val_loader, device, KS)}
    return cache[key]


def capture(env, point, model=None):
    """Add the top-5 fields to ``point`` (in place) for ``model`` (default the env's current model); the origin
    point (``step`` < 0) reuses the cached origin read. Returns ``point``; untouched when the flag is off."""
    if not enabled() or not point:
        return point
    origin = _origin(env)
    if int(point.get("step", 0)) < 0:
        test, val = origin["test"], origin["val"]
    else:
        model = env.current_model if model is None else model
        device = env.conf.device
        test = utils.evaluate_topk(model, env.test_loader, device, KS)
        val = utils.evaluate_topk(model, env.val_loader, device, KS)
    point.update({
        "test_top5": float(test[5]), "test_origin_top5": float(origin["test"][5]),
        "val_top5": float(val[5]), "val_origin_top5": float(origin["val"][5]),
        "test_top1_pass": float(test[1]), "val_top1_pass": float(val[1]),
    })
    return point


def line(key, name, point):
    """The extra TRAJ line for a captured point, or None when the point carries no top-5 (flag off)."""
    if not point or "test_top5" not in point:
        return None
    return (f"[eval] TRAJ top5 {key} {name} step={point['step']} | "
            f"top5 {point['test_origin_top5']:.3f} -> {point['test_top5']:.3f} "
            f"({point['test_top5'] - point['test_origin_top5']:+.3f}) | "
            f"val top5 Δacc {(point['val_top5'] - point['val_origin_top5']) * 100.0:+.2f} pp | "
            f"top1 pass {point['test_top1_pass']:.3f} vs TRAJ {float(point['test_acc']):.3f}")


def final(env, model, label, name, point):
    """Top-5 of a recovered final-FT candidate: prints the ``final_ft_top5`` line and returns the extra fields for
    the ``eval_traj_final_ft`` record (``test_final_top5`` ...); ``{}`` when the flag is off."""
    if not enabled():
        return {}
    origin = _origin(env)
    device = env.conf.device
    test = utils.evaluate_topk(model, env.test_loader, device, KS)
    val = utils.evaluate_topk(model, env.val_loader, device, KS)
    utils.print_flush(
        f"[eval] TRAJ final_ft_top5 {label} {name} step={point['step']} | "
        f"top5 {origin['test'][5]:.3f} -> {test[5]:.3f} ({test[5] - origin['test'][5]:+.3f}) | "
        f"val top5 Δacc {(val[5] - origin['val'][5]) * 100.0:+.2f} pp | top1 pass {test[1]:.3f}")
    return {"test_final_top5": float(test[5]), "val_final_top5": float(val[5]),
            "test_origin_top5": float(origin["test"][5]), "val_origin_top5": float(origin["val"][5]),
            "test_final_top1_pass": float(test[1])}
