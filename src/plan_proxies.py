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


#: The env keys of the final recovery's recipe, in the order ``a2c_agent_reinforce_runner._FINAL_FT_ENV_KEYS`` has them
#: (a copy, so this module never imports the runner; ``tests/test_v32_fallback_judge.py`` checks the two are equal).
_FINAL_FT_ENV_KEYS = ("SPECTRA_FT_OPTIM", "SPECTRA_FT_SGD_LR", "SPECTRA_FT_MOMENTUM", "SPECTRA_FT_WD",
                      "SPECTRA_FT_COSINE", "SPECTRA_FT_SCHEDULE", "SPECTRA_FT_MIXUP",
                      "SPECTRA_FT_LABEL_SMOOTH", "SPECTRA_FT_KD")


@contextlib.contextmanager
def final_recipe():
    """The final recovery's recipe (v32, the ``rc<E>`` judge), as the runner's ``_final_ft`` sets it for the body and puts
    it back after: SGD at ``fortify.eval_final_ft_lr()``, momentum 0.9, weight decay 5e-4, the cosine schedule (the
    ``warmcos`` one with ``fortify.eval_final_ft_warmup()`` epochs under ``eval_final_ft_schedule()``), no mixup, no label
    smoothing and KD off (a judge has no teacher). Like the runner, it sets these keys over the environment and clears no
    other ``SPECTRA_FT_*`` key."""
    from src import fortify
    lr, schedule, warmup = fortify.eval_final_ft_lr(), fortify.eval_final_ft_schedule(), fortify.eval_final_ft_warmup()
    recipe = dict(zip(_FINAL_FT_ENV_KEYS, ("sgd", f"{lr:g}", "0.9", "5e-4", "1", schedule, "0", "0", "0")))
    if schedule:
        recipe["SPECTRA_FT_WARMUP_EPOCHS"] = f"{warmup:g}"
    saved = {k: os.environ.get(k) for k in recipe}
    try:
        os.environ.update(recipe)
        yield recipe
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
        self._cache = {}  # (kind, E, seed, widths, detail) -> a ft<E> / rc<E> draw's read (v32), per scorer
        self._final = None  # the final loader of rc<E> (v32), built on first use
        self.widths0 = {int(row): int(group.width) for group, row in plan}
        self.cutter = plan_agent.MaskedCut(model)
        if self.cutter.rows != list(self.widths0):
            raise RuntimeError(f"{self.name}: the working copy plans rows {self.cutter.rows}, "
                               f"the origin {list(self.widths0)}")
        self.val = plan_agent.eval_batches(val_loader, device)

    def batches(self, proxy: str, seed: int):
        """What ``score`` needs for ``proxy`` under ``seed``, drawn once and shared by every plan scored on them, as
        ``NetInstance.batches``: the fixed ``(x, y)`` batches of ``fs<S>`` (``plan_agent.ft_batches``), the
        recalibration batches of ``bn<N>`` (``plan_agent.recal_batches``), nothing for ``cut``, and for ``ft<E>`` (and
        v32's ``rc<E>``, with or without its ``x<k>`` suffix) the seed itself (its fine-tune draws its epochs from the
        loader under it)."""
        from src import plan_agent
        kind, n, _k = plan_agent.parse_proxy(proxy, judge=True)
        if kind in ("ft", "rc"):
            return int(seed)
        if kind == "fs":
            return plan_agent.ft_batches(self.train_loader, n, seed, self.device)
        return plan_agent.recal_batches(self.train_loader, n, seed, self.device) if kind == "bn" else []

    def score(self, widths, proxy: str, batches) -> float:
        """Val accuracy (a fraction) of the net cut once to ``widths`` after ``proxy`` on ``batches``:
        ``NetInstance.reward``'s read before its origin subtraction, by the same calls. For ``ft<E>x<k>`` and
        ``rc<E>x<k>`` (v32) the mean of the k draws."""
        return self._scored(widths, proxy, batches, False)["acc"]

    def score_detail(self, widths, proxy: str, batches) -> dict:
        """``score``'s read with the detail ``SPECTRA_ALLOC_FALLBACK_DETAIL`` logs, from one forward pass per scored
        model: ``{"acc", "loss", "bits", "n", "draws"}``, the val accuracy (the ``score``), the val mean cross-entropy,
        the per-image correctness bits over the val half in loader order (``numpy.packbits``, base64), the number of
        val images and each draw's accuracy (one for a single draw). For ``x<k>``: ``acc`` and ``loss`` are the means
        of the k draws, ``bits`` is the first draw's (the draw ``ft<E>`` / ``rc<E>`` alone reads) and ``bits_draws``
        holds every draw's."""
        return self._scored(widths, proxy, batches, True)

    def _read(self, model, detail: bool) -> dict:
        """``{"acc"}`` of ``model`` on the val half (``plan_agent.accuracy``); with ``detail`` one pass that also reads
        the mean cross-entropy and the packed per-image correctness bits."""
        from src import plan_agent
        if not detail:
            return {"acc": float(plan_agent.accuracy(model, self.val))}
        import base64
        import numpy as np
        was_training = model.training
        model.eval()
        correct, total, loss, bits = 0, 0, 0.0, []
        with torch.no_grad():
            for x, y in self.val:
                logits = model(x)
                hit = logits.argmax(dim=1) == y
                correct += int(hit.sum())
                total += int(y.numel())
                loss += float(torch.nn.functional.cross_entropy(logits.float(), y, reduction="sum"))
                bits.append(hit.cpu().numpy().astype(bool))
        model.train(was_training)
        flat = np.concatenate(bits) if bits else np.zeros(0, dtype=bool)
        return {"acc": correct / max(1, total), "loss": loss / max(1, total), "n": int(total),
                "bits": base64.b64encode(np.packbits(flat).tobytes()).decode("ascii")}

    def _scored(self, widths, proxy: str, batches, detail: bool) -> dict:
        from src import plan_agent, recovery_edits
        kind, n, k = plan_agent.parse_proxy(proxy, judge=True)
        if kind in ("ft", "rc"):
            draws = [self._train_read(widths, kind, n, batches if j == 0 else self._seed_plus(kind, n, batches, j),
                                      detail) for j in range(k)]
        else:
            work = self.cutter.cut(plan_agent.rates_of(widths, self.widths0))
            if kind == "bn":
                recovery_edits.recalibrate_batchnorm(work, batches, self.device, len(batches))
            elif kind == "fs":
                plan_agent.short_finetune(work, batches, plan_agent.fs_lr(), after_step=self.cutter.remask)
            draws = [self._read(work, detail)]
        accs = [d["acc"] for d in draws]
        out = {"acc": accs[0] if len(accs) == 1 else sum(accs) / len(accs)}
        if detail:
            losses = [d["loss"] for d in draws]
            out.update(loss=losses[0] if len(losses) == 1 else sum(losses) / len(losses), bits=draws[0]["bits"],
                       n=draws[0]["n"], draws=accs)
            if len(draws) > 1:
                out["bits_draws"] = [d["bits"] for d in draws]
        return out

    @staticmethod
    def _seed_plus(kind, epochs, seed, j):
        """The proxy seed of draw ``j`` of ``ft<E>x<k>`` / ``rc<E>x<k>``: ``seed + j``, ``seed`` being the one
        ``batches`` returned (checked as a seed)."""
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"the {kind}{epochs} score takes the seed batches() returns for it, "
                             f"not {type(seed).__name__}")
        return seed + j

    def _finetune_score(self, widths, epochs: int, seed) -> float:
        """``NetInstance.finetune_reward``'s read: the real cut (``alloc_walk.cut_to``) fine-tuned ``epochs`` epochs by
        the handler's own recipe (``handler_recipe``) under ``seeded(seed)``, entered after the cut as there; the
        handler's lines are captured, the recipe kept in ``recipe_line``, and a fine-tune that did not run raises."""
        return self._train_read(widths, "ft", epochs, seed, False)["acc"]

    def _final_loader(self):
        """``(loader, aug, train_model kwargs)`` of the final recovery (``rc<E>``), built once per scorer as the
        runner's ``_final_ft`` builds them: ``utils.final_ft_train_loader`` at ``fortify.eval_final_ft_batch()``,
        ``keep_last`` when ``eval_final_ft_select() == "last"``, ``cuda_graph`` under ``eval_final_ft_cuda_graph()``."""
        if self._final is None:
            from src import fortify
            loader, aug = utils.final_ft_train_loader(self.train_loader, fortify.eval_final_ft_batch())
            kw = {}
            if fortify.eval_final_ft_select() == "last":
                kw["keep_last"] = True
            if fortify.eval_final_ft_cuda_graph():
                kw["cuda_graph"] = True
            self._final = (loader, aug, kw)
        return self._final

    def _train_read(self, widths, kind: str, epochs: int, seed, detail: bool) -> dict:
        """One draw of ``ft<E>`` (``kind="ft"``: the handler's own recipe, ``NetInstance.finetune_reward``'s read) or
        ``rc<E>`` (``kind="rc"``: the final recovery's recipe, ``final_recipe``, on the final loader, KD off, the
        cosine over ``E`` epochs) on the real cut (``alloc_walk.cut_to``) to ``widths``, under ``seeded(seed)`` entered
        after the cut; the handler's lines are captured, the recipe kept in ``recipe_line``, a fine-tune that did not
        run raises, and the val half is read by ``_read``. Cached by ``(kind, E, seed, widths, detail)`` for the life
        of the scorer, so a draw two proxies share (``ft1`` and ``ft1x3``'s first) is trained once."""
        from src import alloc_walk, plan_agent
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"the {kind}{epochs} score takes the seed batches() returns for it, "
                             f"not {type(seed).__name__}")
        key = (kind, int(epochs), seed, tuple(sorted((int(r), int(w)) for r, w in widths.items())), bool(detail))
        if key in self._cache:
            return self._cache[key]
        epochs, lines = int(epochs), []
        loader, aug, kw = self._final_loader() if kind == "rc" else (self.train_loader, None, {})
        recipe_scope = handler_recipe() if kind == "ft" else final_recipe()
        with seeded(seed):
            real = alloc_walk.cut_to(self.model, self.plan, plan_agent.rates_of(widths, self.widths0), self.input_shape)
            real = real.to(self.device)
            for param in real.parameters():
                param.requires_grad = True
            with seeded(seed), recipe_scope, plan_agent._captured(lines):
                self.create_handler(real).train_model(loader, allow_reinit_retry=False, max_epochs=epochs,
                                                      patience=epochs + 1, tag=f"plan {kind}{epochs} {self.name}", **kw)
            got = self._read(real, detail)
        del real
        recipe = [line.split("Fine-tune recipe:", 1)[1].strip() for line in lines if "Fine-tune recipe:" in line]
        if not recipe or any("produced no loss" in line for line in lines):
            raise RuntimeError(f"{self.name}: the {kind}{epochs} fine-tune did not run: " + " | ".join(lines[-3:]))
        if any(" = nan" in line.lower() or " = inf" in line.lower() for line in lines if "loss" in line.lower()):
            utils.print_flush(f"[alloc] WARNING: {self.name}: the {kind}{epochs} fine-tune's loss went NaN or inf for "
                              "this plan; its score reads the diverged net")
        self.recipe_line = (recipe[-1].split(" trainable=", 1)[0]
                            + (f" bs={loader.batch_size} aug={aug}" if kind == "rc" else ""))
        self._cache[key] = got
        return got
