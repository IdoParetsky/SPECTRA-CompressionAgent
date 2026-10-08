"""
D-PROXY (``docs/LEARNING_PROGRAM_OCT8.md`` §4): cheap stand-ins for a candidate's final accuracy, read on
the fixed val half before its final fine-tune. ``SPECTRA_EVAL_PROXIES`` (default off) names them, comma
separated, from:

* ``cut``     — the candidate as saved (no measurement; the TRAJ point already holds it);
* ``bn<N>``   — BatchNorm running statistics re-estimated on N train batches (EagleEye's adaptive BN);
* ``ft<E>``   — E epochs of the per-step fine-tune (patience E + 1);
* ``ft<E>_<P>`` — the per-step fine-tune at E epochs, patience P: ``ft12_4`` is v10's train recipe,
  ``ft40_10`` ft40's (ledger §112).

Every proxy works on a copy and runs under ``SPECTRA_EVAL_PROXY_SEED`` (default 0) with the global RNG
state saved and restored, so candidates in different jobs see the same batches (common random numbers)
and the final fine-tune that follows is not moved by the proxies. The per-step fine-tune runs with the
handler's own recipe (``SPECTRA_FT_*`` recipe keys are cleared for its duration), the recipe v10 trained
with; its "Fine-tune recipe" log line is the check.
"""
from __future__ import annotations

import contextlib
import copy
import os
import random
import re
import time

import torch

import src.utils as utils

_RECIPE_KEYS = ("SPECTRA_FT_OPTIM", "SPECTRA_FT_LR", "SPECTRA_FT_SGD_LR", "SPECTRA_FT_MOMENTUM", "SPECTRA_FT_WD",
                "SPECTRA_FT_COSINE", "SPECTRA_FT_SCHEDULE", "SPECTRA_FT_WARMUP_EPOCHS", "SPECTRA_FT_LR_MIN",
                "SPECTRA_FT_MIXUP", "SPECTRA_FT_LABEL_SMOOTH", "SPECTRA_FT_KD")
_NAME = re.compile(r"^(cut|bn(\d+)|ft(\d+)(?:_(\d+))?)$")


def proxies() -> list:
    """``SPECTRA_EVAL_PROXIES``: the proxy names, in the order given; empty when unset."""
    raw = os.environ.get("SPECTRA_EVAL_PROXIES", "").strip().lower()
    names = [n.strip() for n in raw.split(",") if n.strip()]
    for name in names:
        if not _NAME.match(name):
            raise ValueError(f"SPECTRA_EVAL_PROXIES: unknown proxy {name!r} (cut, bn<N>, ft<E>, ft<E>_<P>)")
    return names


def proxy_seed() -> int:
    """``SPECTRA_EVAL_PROXY_SEED`` (0): the seed every proxy runs under."""
    return int(os.environ.get("SPECTRA_EVAL_PROXY_SEED", "0"))


@contextlib.contextmanager
def seeded(seed):
    """Run the body under ``seed`` and put the global torch / CUDA / random state back afterwards."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    py = random.getstate()
    try:
        import numpy as np
        npy = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy is a dependency of the env
        np, npy = None, None
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed) % (2 ** 32))
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(py)
        if np is not None:
            np.random.set_state(npy)


@contextlib.contextmanager
def handler_recipe():
    """Clear the ``SPECTRA_FT_*`` recipe keys for the body (the handler's own recipe), then restore them."""
    saved = {k: os.environ.pop(k) for k in _RECIPE_KEYS if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def _score(env, model):
    handler = env.create_learning_handler(model)
    return float(handler.evaluate_model(env.val_loader)), float(handler.evaluate_model(env.test_loader))


def measure_one(env, model, name, label=""):
    """``{"val", "test", "minutes"}`` of proxy ``name`` on a copy of ``model`` (accuracies as fractions)."""
    from src import recovery_edits
    m = _NAME.match(name)
    t0 = time.perf_counter()
    work = copy.deepcopy(model).to(env.conf.device)
    with seeded(proxy_seed()):
        if m.group(2):
            recovery_edits.recalibrate_batchnorm(work, env.train_loader, env.conf.device, int(m.group(2)))
        elif m.group(3):
            epochs = int(m.group(3))
            patience = int(m.group(4)) if m.group(4) else epochs + 1
            for param in work.parameters():
                param.requires_grad = True
            with handler_recipe():
                env.create_learning_handler(work).train_model(
                    env.train_loader, allow_reinit_retry=False, max_epochs=epochs, patience=patience,
                    tag=f"proxy {name} {label}".strip())
        val, test = _score(env, work)
    del work
    return {"val": val, "test": test, "minutes": round((time.perf_counter() - t0) / 60.0, 3)}


def measure(env, net_path, label, candidate, names):
    """Every proxy in ``names`` for one candidate; prints one line and records ``eval_plan_proxy``."""
    import src.run_recorder as run_recorder
    point = candidate["point"]
    name = os.path.basename(net_path)
    out = {}
    for proxy in names:
        if proxy == "cut":
            out[proxy] = {"val": float(point["val_acc"]), "test": float(point["test_acc"]), "minutes": 0.0}
            continue
        try:
            out[proxy] = measure_one(env, candidate["model"], proxy, label)
        except Exception as error:  # noqa: BLE001 - one proxy must not cost the final fine-tune
            utils.print_flush(f"[proxy] {proxy} {label} {name} failed: {type(error).__name__}: {error}")
            out[proxy] = None
    val_origin, test_origin = float(point["val_origin"]), float(point["test_origin"])
    utils.print_flush(
        f"[proxy] {label} {name} step={point['step']} | params x{float(point['param']):.3f} | "
        + " | ".join(f"{p} val Δ {100.0 * (r['val'] - val_origin):+.2f} ({r['minutes']:.1f} min)" if r else f"{p} failed"
                     for p, r in out.items()))
    run_recorder.record(
        "eval_plan_proxy", network=net_path, label=label, step=int(point["step"]),
        param=float(point["param"]), flop=float(point["flop"]), val_origin=val_origin, test_origin=test_origin,
        seed=proxy_seed(), proxies=out)
    return out
