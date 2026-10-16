"""
D-PROXY (``docs/LEARNING_PROGRAM_OCT8.md`` §4): cheap stand-ins for a candidate's final accuracy, read on
the fixed val half before its final fine-tune. ``SPECTRA_EVAL_PROXIES`` (default off) names them, comma
separated, from:

* ``cut``     — the candidate as saved (no measurement; the TRAJ point already holds it);
* ``bn<N>``   — BatchNorm running statistics re-estimated on N train batches (EagleEye's adaptive BN);
* ``fs<S>``   — S SGD steps on S fixed train batches (``plan_agent.short_finetune``, lr ``SPECTRA_PLAN_FS_LR``), as
  the plan trainer's ``fs<S>`` reward runs them on its masked cut;
* ``ft<E>``   — E epochs of the per-step fine-tune (patience E + 1);
* ``ft<E>_<P>`` — the per-step fine-tune at E epochs, patience P: ``ft12_4`` is v10's train recipe,
  ``ft40_10`` ft40's (ledger §112).

Every proxy works on a copy and runs under ``SPECTRA_EVAL_PROXY_SEED`` (default 0) with the global RNG
state saved and restored, so candidates in different jobs see the same batches (common random numbers)
and the final fine-tune that follows is not moved by the proxies. The per-step fine-tune runs with the
handler's own recipe (``SPECTRA_FT_*`` recipe keys are cleared for its duration), the recipe v10 trained
with; its "Fine-tune recipe" log line is the check.

``PlanScorer`` (v23) is the plan trainer's reward read without its policy state, for a search over plans at eval
(``alloc_walk`` ``randbest``): the origin cut once to a plan's widths and read on the val half after ``cut``, ``bn<N>``,
``fs<S>`` or ``ft<E>`` (the trainer's names, ``plan_agent.parse_proxy``), with the trainer's own implementation and
batches. It never reads the TEST half.
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
_NAME = re.compile(r"^(cut|bn(\d+)|ft(\d+)(?:_(\d+))?|fs(0*[1-9]\d*))$")


def proxies() -> list:
    """``SPECTRA_EVAL_PROXIES``: the proxy names, in the order given; empty when unset."""
    raw = os.environ.get("SPECTRA_EVAL_PROXIES", "").strip().lower()
    names = [n.strip() for n in raw.split(",") if n.strip()]
    for name in names:
        if not _NAME.match(name):
            raise ValueError(f"SPECTRA_EVAL_PROXIES: unknown proxy {name!r} (cut, bn<N>, fs<S>, ft<E>, ft<E>_<P>)")
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
        elif m.group(5):
            from src import plan_agent
            plan_agent.short_finetune(
                work, plan_agent.ft_batches(env.train_loader, int(m.group(5)), proxy_seed(), env.conf.device),
                plan_agent.fs_lr())
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
    fs = {}
    if any(_NAME.match(p) and _NAME.match(p).group(5) for p in names):
        from src import plan_agent
        fs = {"fs_lr": plan_agent.fs_lr()}
    utils.print_flush(
        f"[proxy] {label} {name} step={point['step']} | params x{float(point['param']):.3f} | "
        + " | ".join(f"{p} val Δ {100.0 * (r['val'] - val_origin):+.2f} ({r['minutes']:.1f} min)" if r else f"{p} failed"
                     for p, r in out.items())
        + (f" | fs lr {fs['fs_lr']:g}" if fs else ""))
    run_recorder.record(
        "eval_plan_proxy", network=net_path, label=label, step=int(point["step"]),
        param=float(point["param"]), flop=float(point["flop"]), val_origin=val_origin, test_origin=test_origin,
        seed=proxy_seed(), proxies=out, **fs)
    return out


# ------------------------------------------------------------------ v23: the trainer's reward read, shared with a search


class PlanScorer:
    """
    The plan trainer's reward path without its policy state (``plan_agent.NetInstance`` less the encoder state, the
    tokens and the sensitivities): the origin ``model`` cut once to a plan's widths and read on ``val_loader`` after a
    proxy, as ``NetInstance.reward`` scores a training plan. ``cut`` / ``bn<N>`` / ``fs<S>`` score the masked working
    copy (``plan_agent.MaskedCut``: ``alloc_walk.cut_to``'s L1 cut written as zeros, re-masked after every SGD step of
    ``plan_agent.short_finetune`` at ``plan_agent.fs_lr``); ``ft<E>`` cuts for real (``alloc_walk.cut_to``) and
    fine-tunes that copy E epochs by the learning handler's own recipe under ``seeded(seed)``, as
    ``NetInstance.finetune_reward``. ``batches`` draws what a proxy needs once, so every plan scored on them shares the
    same batches (common random numbers), as ``NetInstance.batches``. A score is the val accuracy (a fraction) the
    trainer subtracts the origin's from. Only ``val_loader`` is read, once, at construction: the TEST half is never
    touched, so a search over plans (``alloc_walk`` ``randbest``) selects on val alone.
    """

    def __init__(self, model, plan, val_loader, device, train_loader=None, input_shape=None, create_handler=None,
                 name=""):
        from src import plan_agent
        self.model, self.plan, self.device, self.name = model, plan, device, str(name)
        self.train_loader, self.input_shape = train_loader, input_shape
        self.create_handler = create_handler if create_handler is not None else plan_agent.default_handler
        self.recipe_line = None  # the last ft<E> fine-tune's recipe, as NetInstance keeps it
        self.widths0 = {int(row): int(group.width) for group, row in plan}
        self.cutter = plan_agent.MaskedCut(model)
        if self.cutter.rows != list(self.widths0):
            raise RuntimeError(f"{self.name}: the working copy plans rows {self.cutter.rows}, "
                               f"the origin {list(self.widths0)}")
        self.val = plan_agent.eval_batches(val_loader, device)

    def batches(self, proxy: str, seed: int):
        """What ``score`` needs for ``proxy`` under ``seed``, drawn once and shared by every plan scored on them, as
        ``NetInstance.batches``: the fixed ``(x, y)`` batches of ``fs<S>`` (``plan_agent.ft_batches``), the
        recalibration batches of ``bn<N>`` (``plan_agent.recal_batches``), nothing for ``cut``, and for ``ft<E>`` the
        seed itself (its fine-tune draws its epochs from the loader under it)."""
        from src import plan_agent
        kind, n = plan_agent.parse_proxy(proxy)
        if kind == "ft":
            return int(seed)
        if kind == "fs":
            return plan_agent.ft_batches(self.train_loader, n, seed, self.device)
        return plan_agent.recal_batches(self.train_loader, n, seed, self.device) if kind == "bn" else []

    def score(self, widths, proxy: str, batches) -> float:
        """Val accuracy (a fraction) of the net cut once to ``widths`` after ``proxy`` on ``batches``:
        ``NetInstance.reward``'s read before its origin subtraction, by the same calls."""
        from src import plan_agent, recovery_edits
        kind, n = plan_agent.parse_proxy(proxy)
        if kind == "ft":
            return self._finetune_score(widths, n, batches)
        work = self.cutter.cut(plan_agent.rates_of(widths, self.widths0))
        if kind == "bn":
            recovery_edits.recalibrate_batchnorm(work, batches, self.device, len(batches))
        elif kind == "fs":
            plan_agent.short_finetune(work, batches, plan_agent.fs_lr(), after_step=self.cutter.remask)
        return float(plan_agent.accuracy(work, self.val))

    def _finetune_score(self, widths, epochs: int, seed) -> float:
        """``NetInstance.finetune_reward``'s read: the real cut (``alloc_walk.cut_to``) fine-tuned ``epochs`` epochs by
        the handler's own recipe (``handler_recipe``) under ``seeded(seed)``, entered after the cut as there; the
        handler's lines are captured, the recipe kept in ``recipe_line``, and a fine-tune that did not run raises."""
        from src import alloc_walk, plan_agent
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"the ft{epochs} score takes the seed batches() returns for it, not {type(seed).__name__}")
        epochs, lines = int(epochs), []
        with seeded(seed):
            real = alloc_walk.cut_to(self.model, self.plan, plan_agent.rates_of(widths, self.widths0), self.input_shape)
            real = real.to(self.device)
            for param in real.parameters():
                param.requires_grad = True
            with seeded(seed), handler_recipe(), plan_agent._captured(lines):
                self.create_handler(real).train_model(self.train_loader, allow_reinit_retry=False, max_epochs=epochs,
                                                      patience=epochs + 1, tag=f"plan ft{epochs} {self.name}")
            acc = float(plan_agent.accuracy(real, self.val))
        del real
        recipe = [line.split("Fine-tune recipe:", 1)[1].strip() for line in lines if "Fine-tune recipe:" in line]
        if not recipe or any("produced no loss" in line for line in lines):
            raise RuntimeError(f"{self.name}: the ft{epochs} fine-tune did not run: " + " | ".join(lines[-3:]))
        if any(" = nan" in line.lower() or " = inf" in line.lower() for line in lines if "loss" in line.lower()):
            utils.print_flush(f"[alloc] WARNING: {self.name}: the ft{epochs} fine-tune's loss went NaN or inf for this "
                              "plan; its score reads the diverged net")
        self.recipe_line = recipe[-1].split(" trainable=", 1)[0]
        return acc
