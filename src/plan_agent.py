"""
Plan-as-action agent (``docs/LEARNING_PROGRAM_OCT8.md`` §5): one score per coupling group, decoded in
one pass into the widths of a single cut that keeps a fraction κ of the parameters.

* ``ParamModel``: the parameter count after each planned group is cut to an integer width, computed
  from the channel groups without cutting. The real cut is ``alloc_walk.cut_to``.
* ``FlopModel``: the MACs (``utils.calc_flops``) after the same cuts, the T0-F FLOPs budget; both models
  expose ``cost``, the kept params or MACs the decoders bisect on.
* ``decode``: scores z → widths. keep_g = clip(σ(z_g + b), k_min, 1), rounded to the nearest width,
  with the scalar b bisected on the cost model so the kept cost comes closest to κ.
* ``scale_decode``: the alloc walk's family, keep_g = clip(c · w_g, k_min, 1) with c bisected the same
  way, for the uniform / sens / inner reference plans.
* ``residual_decode`` (T2, ``SPECTRA_PLAN_RESIDUAL``, v18): the agent as a residual on a no-agent rule,
  keep_g = clip(c · w_g · exp(z_g), k_min, 1) with w the sens / sens_cost weights of ``prior_weights``
  (``prior_kind`` picks the rule per budget under ``auto``), so the zero-init agent's mean plan is the rule's
  plan exactly and it learns only a change to it. ``SPECTRA_PLAN_RESIDUAL_ZMAX`` > 0 (v20) is a trust region on
  that change: exp(z_g) becomes exp(zmax · tanh(z_g / zmax)) (``trust_score``), within e^±zmax of the rule's weight.
* ``menu_decode`` (v36, ``SPECTRA_PLAN_MENU``, default off): the residual on a per-network blend of several no-agent
  rules. Each rule k of the menu (``MENU_KINDS``; the first is the anchor, the residual's own prior) gives weights w_k
  on the net (``menu_rules``: the weights the alloc walk's row of that kind hands its landing), normalised by their
  geometric mean; log w_g = Σ_k π_k log ŵ_k,g + the residual's score, with π = softmax(u) (``menu_shares``) and u the
  K logits of the policy's network-level head (``PlanPolicy.menu_logits``, on the mean token output and the rules'
  plan costs, ``menu_features``). A pure share (π = e_k) is that rule's own ``scale_decode`` plan exactly, and the
  zero-init head's mean plan is the anchor's (shares under ``MENU_SHARE_FLOOR`` are dropped at the mean).
* ``MaskedCut``: ``cut_to``'s L1 cut written as zeros on a working copy, for the reward; ``remask`` zeroes
  the same entries again, so the ``fs<S>`` proxy (``short_finetune``) trains a copy that stays the cut.
* ``NetInstance.finetune_reward`` (v23, ``SPECTRA_PLAN_PROXY=ft<E>``, default off): the eval-side ``ft<E>`` proxy
  (``plan_proxies.measure_one``) as a training reward: the net cut for real (``alloc_walk.cut_to``), fine-tuned E
  epochs by the learning handler's own recipe under the instance's seed, then read on the val batches.
* ``PlanPolicy``: the agent's state encoder with a per-token linear head. A group's score is the mean
  over its tokens, centred over groups (b absorbs any common shift), and plans are z = μ + σ ε.
* ``NetInstance``: one catalog net's origin read once (state, plan, token map, param model, val batches,
  sensitivities), so an instance (net, κ) costs only its plans.
* ``plan_for_env``: a frozen agent's mean plan in ``alloc_walk.plan_targets``' format
  (``SPECTRA_ALLOC_KIND=agent``) or, with ``sample``, one Gaussian draw around it
  (``SPECTRA_ALLOC_KIND=agent_sample``), so the existing walk, final fine-tune and TEST read it.
"""

import contextlib
import copy
import math
import os
import time

import torch
from torch import nn

TOKEN_KINDS = ("layer", "group")
PROXIES = ("cut", "bn", "fs", "ft")
BUDGETS = ("params", "flops")
PRIOR_KINDS = ("sens", "sens_cost")
MENU_KINDS = ("sens", "uniform", "sens_cost", "sens_thresh")  # v36: the rules a menu may list (see menu_rules)
MENU_INIT = 6.0  # SPECTRA_PLAN_MENU_INIT: the anchor's logit at the zero-init menu head (the other rules start at 0)
MENU_SIGMA = 0.5  # SPECTRA_PLAN_MENU_SIGMA: the std of the menu logits the trainer samples
MENU_SHARE_FLOOR = 0.005  # the mean plan drops a rule under this share (at the default init: every rule but the anchor)
Z_CLIP = 30.0  # |z| beyond this is clipped in residual_decode: exp stays finite, the plan is already saturated
FS_MOMENTUM = 0.9  # the fs<S> proxy's short fine-tune: SGD momentum and weight decay, shared by the trainer and
FS_WD = 5e-4  # the eval-side proxy (plan_proxies); the lr is fs_lr()


def fs_lr() -> float:
    """``SPECTRA_PLAN_FS_LR`` (0.01): the lr of the ``fs<S>`` proxy's SGD steps, on both sides."""
    return float(os.environ.get("SPECTRA_PLAN_FS_LR", "0.01"))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class ParamModel:
    """Parameters of ``model`` after the groups of ``plan`` (``[(group, row)]``) are cut to given widths.

    Producers and depthwise owners lose output channels, normalisations lose features, and consumers
    lose the input channels they read from the group (times the spatial size behind a flatten). A
    consumer that reads only part of a group loses that share of the removed channels.
    """

    def __init__(self, model: nn.Module, plan):
        self.rows = [int(row) for _group, row in plan]
        self.widths0 = {int(row): int(group.width) for group, row in plan}
        self.total0 = int(sum(p.numel() for p in model.parameters()))
        self._shape = {}
        for module in model.modules():
            own = int(sum(p.numel() for p in module.parameters(recurse=False)))
            if not own:
                continue
            if isinstance(module, nn.Conv2d):
                self._shape[id(module)] = ("conv", module.out_channels, module.in_channels, module.groups,
                                           module.kernel_size[0] * module.kernel_size[1],
                                           module.bias is not None, own)
            elif isinstance(module, nn.Linear):
                self._shape[id(module)] = ("linear", module.out_features, module.in_features, 1, 1,
                                           module.bias is not None, own)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self._shape[id(module)] = ("norm", module.num_features, 0, 1, 1, False, own)
        self._edits = []
        for group, row in plan:
            w0 = int(group.width)
            consumers = []
            for ref in group.consumers:
                module = ref.module
                share = len(ref.positions) / w0 if ref.positions else 1.0
                factor = 1
                if isinstance(module, nn.Linear) and ref.total and module.in_features != ref.total:
                    factor = module.in_features // ref.total
                consumers.append((id(module), factor, share))
            norms = [(id(ref.module), len(ref.positions) / w0 if ref.positions else 1.0) for ref in group.norms]
            self._edits.append((int(row), w0, [id(m) for m in group.producers],
                                [id(m) for m in group.depthwise], consumers, norms))

    def _cuts(self, widths):
        """``(out_cut, in_cut, depthwise)``: channels removed per module id from its outputs / its inputs, and the
        depthwise owners among the cut modules."""
        out_cut, in_cut, depthwise = {}, {}, set()
        for row, w0, producers, owners, consumers, norms in self._edits:
            removed = w0 - int(widths.get(row, w0))
            if removed <= 0:
                continue
            for mid in producers:
                out_cut[mid] = out_cut.get(mid, 0) + removed
            for mid in owners:
                out_cut[mid] = out_cut.get(mid, 0) + removed
                depthwise.add(mid)
            for mid, factor, share in consumers:
                in_cut[mid] = in_cut.get(mid, 0) + int(round(removed * share)) * factor
            for mid, share in norms:
                out_cut[mid] = out_cut.get(mid, 0) + int(round(removed * share))
        return out_cut, in_cut, depthwise

    def params(self, widths) -> int:
        out_cut, in_cut, depthwise = self._cuts(widths)
        total = self.total0
        for mid in set(out_cut) | set(in_cut):
            if mid not in self._shape:
                continue
            kind, out0, in0, groups0, kk, bias, own = self._shape[mid]
            out = out0 - out_cut.get(mid, 0)
            if kind == "norm":
                new = own // out0 * out
            elif mid in depthwise:
                new = out * (in0 // groups0) * kk + (out if bias else 0)
            else:
                new = out * ((in0 - in_cut.get(mid, 0)) // groups0) * kk + (out if bias else 0)
            total += new - own
        return int(total)

    def cost(self, widths) -> int:
        """The decoders' interface: the kept cost, here params."""
        return self.params(widths)

    def kept(self, widths) -> float:
        return self.params(widths) / float(self.total0)


def widths_of(keeps, widths0, min_width: int = 1):
    """Nearest integer width per group, at least 1 (``min_width`` where the origin is that wide: the eval walk's
    legal floor, ``fortify.plan_min_width``) and at most the origin's."""
    return {row: max(1, min(w0, int(min_width)), min(w0, int(round(float(keeps[row]) * w0))))
            for row, w0 in widths0.items()}


def rates_of(widths, widths0):
    """``alloc_walk.cut_to``'s keeps for ``widths``: w / w0, and exactly 1 for an uncut group
    (``pruning.target_width`` removes a channel at any rate below 1 and maps w / w0 back to w)."""
    return {row: (1.0 if int(widths[row]) >= w0 else int(widths[row]) / float(w0)) for row, w0 in widths0.items()}


class FlopModel:
    """MACs (``utils.calc_flops``) of ``model`` after the groups of ``plan`` are cut to given widths, without cutting
    at evaluation time; ``rows`` / ``widths0`` / ``total0`` / ``cost`` / ``kept`` as ``ParamModel``, whose ``params``
    it also exposes.

    Conv2d and Linear MACs are structural: ``ParamModel``'s cuts times each module's output multiplier (its spatial
    size, summed over its calls), read from one hooked forward at batch 1. The rest (BatchNorm, activations) is
    assumed linear in every planned group's width, with the per-group coefficient measured once by a real
    single-group cut to half width (``alloc_walk.cut_to``); ``tests/test_v15_flops_budget.py`` checks exact
    equality with ``calc_flops`` of the real cut. ``probes`` cuts in ``seconds``.
    """

    def __init__(self, model: nn.Module, plan, input_shape, device=None):
        from src import alloc_walk
        import src.utils as utils
        started = time.perf_counter()
        self.pm = ParamModel(model, plan)
        self.rows, self.widths0 = self.pm.rows, self.pm.widths0
        self.mult, handles = {}, []

        def hook(module, _inputs, output):
            per = module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
            self.mult[id(module)] = self.mult.get(id(module), 0) + int(output.numel()) // per

        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(hook))
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                dev = device if device is not None else next(model.parameters()).device
                model(torch.zeros(1, *input_shape, device=dev))
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
        self.struct0 = sum(self._macs(mid, shape[1]) for mid, shape in self.pm._shape.items() if shape[0] != "norm")
        self.total0 = float(utils.calc_flops(model, input_shape, device))
        self.elem0 = self.total0 - self.struct0
        self.e, self.probes = {}, 0
        for row, w0 in self.widths0.items():
            self.e[row] = 0.0
            if w0 <= 1:
                continue
            widths = dict(self.widths0)
            widths[row] = w = max(1, w0 // 2)
            cut = alloc_walk.cut_to(model, plan, rates_of(widths, self.widths0), input_shape)
            real = float(utils.calc_flops(cut, input_shape, device))
            del cut
            self.e[row] = (self.elem0 - (real - self.struct(widths))) / float(w0 - w)
            self.probes += 1
        self.seconds = time.perf_counter() - started

    def _macs(self, mid, out, in_cut=0, depthwise=False):
        """MACs of module ``mid`` left with ``out`` outputs and ``in_cut`` fewer inputs (none for a depthwise owner)."""
        _kind, _out0, in0, groups0, kk, _bias, _own = self.pm._shape[mid]
        return out * ((in0 if depthwise else in0 - in_cut) // groups0) * kk * self.mult.get(mid, 0)

    def struct(self, widths) -> int:
        """Conv2d and Linear MACs at ``widths``."""
        out_cut, in_cut, depthwise = self.pm._cuts(widths)
        total = self.struct0
        for mid in set(out_cut) | set(in_cut):
            shape = self.pm._shape.get(mid)
            if shape is None or shape[0] == "norm":
                continue
            total += (self._macs(mid, shape[1] - out_cut.get(mid, 0), in_cut.get(mid, 0), mid in depthwise)
                      - self._macs(mid, shape[1]))
        return int(total)

    def cost(self, widths) -> float:
        """Kept MACs at ``widths``: the structural part plus the linear elementwise part."""
        elem = self.elem0 - sum(self.e[row] * (w0 - int(widths.get(row, w0))) for row, w0 in self.widths0.items())
        return self.struct(widths) + elem

    def params(self, widths) -> int:
        return self.pm.params(widths)

    def kept(self, widths) -> float:
        return self.cost(widths) / self.total0


def _bisect(keeps_at, pm, target, lo, hi, iters, min_width: int = 1):
    """``(x, widths, cost)`` closest to ``target`` kept cost (params or MACs, ``pm.cost``), for a family whose kept
    cost rises with x."""
    def at(x):
        widths = widths_of(keeps_at(x), pm.widths0, min_width)
        return x, widths, pm.cost(widths)

    best = min((at(lo), at(hi)), key=lambda r: abs(r[2] - target))
    for _ in range(iters):
        mid = at(0.5 * (lo + hi))
        if abs(mid[2] - target) < abs(best[2] - target):
            best = mid
        if mid[2] > target:
            hi = mid[0]
        else:
            lo = mid[0]
    return best


def polish(keeps, widths, pm, target, k_min: float = 0.1, fixed=(), min_width: int = 1):
    """``(widths, cost)``: one channel at a time, the group rounded furthest the wrong way moves, while that
    brings the kept cost (params or MACs, ``pm.cost``) closer to ``target`` (same-width groups cross a rounding
    threshold together, so a shared scale alone can miss the target by several percent on a thin net). No group
    goes below ``min_width`` (or its origin width when that is narrower)."""
    widths, fixed = dict(widths), set(fixed)
    floor = {row: max(1, int(round(k_min * w0)), min(w0, int(min_width))) for row, w0 in pm.widths0.items()}
    residual = lambda r: float(keeps[r]) * pm.widths0[r] - widths[r]  # noqa: E731
    p = pm.cost(widths)
    for _ in range(4 * len(widths) + 4):
        if p > target:
            movable = [r for r in pm.rows if r not in fixed and widths[r] > floor[r]]
            row, step = (min(movable, key=residual), -1) if movable else (None, 0)
        elif p < target:
            movable = [r for r in pm.rows if r not in fixed and widths[r] < pm.widths0[r]]
            row, step = (max(movable, key=residual), 1) if movable else (None, 0)
        else:
            break
        if row is None:
            break
        trial = dict(widths)
        trial[row] += step
        q = pm.cost(trial)
        if abs(q - target) >= abs(p - target):
            break
        widths, p = trial, q
    return widths, p


def decode(z, pm, kappa: float, k_min: float = 0.1, iters: int = 48, min_width: int = 1):
    """``(widths, info)`` for scores ``z`` (one per ``pm.rows``) at kept-cost target ``kappa`` (``ParamModel`` or
    ``FlopModel``); no group below ``min_width`` (``fortify.plan_min_width``; 1 = the k_min floor alone)."""
    scores = {row: float(v) for row, v in zip(pm.rows, z)}
    keeps_at = lambda b: {r: min(1.0, max(k_min, sigmoid(scores[r] + b))) for r in pm.rows}  # noqa: E731
    b, widths, _p = _bisect(keeps_at, pm, kappa * pm.total0, -40.0, 40.0, iters, min_width)
    keeps = keeps_at(b)
    widths, p = polish(keeps, widths, pm, kappa * pm.total0, k_min, min_width=min_width)
    return widths, {"b": b, "kept": p / pm.total0, "keeps": keeps}


def scale_decode(weights, pm, kappa: float, k_min: float = 0.1, held=(), iters: int = 48, min_width: int = 1):
    """``(widths, info)`` with keep_g = clip(c · w_g, k_min, 1), or 1 for a held group (the alloc walk's family);
    ``min_width`` as ``decode``."""
    w = {row: max(1e-9, float(weights.get(row, 1.0))) for row in pm.rows}
    held = set(held)
    keeps_at = lambda c: {r: 1.0 if r in held else min(1.0, max(k_min, c * w[r])) for r in pm.rows}  # noqa: E731
    c, widths, _p = _bisect(keeps_at, pm, kappa * pm.total0, 0.0, 1.0 / min(w.values()), iters, min_width)
    keeps = keeps_at(c)
    widths, p = polish(keeps, widths, pm, kappa * pm.total0, k_min, fixed=held, min_width=min_width)
    return widths, {"c": c, "kept": p / pm.total0, "keeps": keeps}


def prior_kind(mode: str, budget: str = "params") -> str:
    """The rule a ``SPECTRA_PLAN_RESIDUAL`` mode starts from on a ``budget`` instance: ``sens`` / ``sens_cost`` as
    named; ``auto`` = sens_cost on params and sens on FLOPs, the best rule at each (ledger §357–§359)."""
    if mode == "auto":
        return "sens" if budget == "flops" else "sens_cost"
    if mode not in PRIOR_KINDS:
        raise ValueError(f"residual mode {mode!r}: expected one of {PRIOR_KINDS + ('auto',)}")
    return mode


def prior_weights(kind: str, rows, sens, costs=None, budget: str = "params", a: float = 0.5):
    """``{row: weight}`` of the ``sens`` / ``sens_cost`` rule over ``rows`` (``alloc_walk.weights``, the alloc walk's
    own floors and median) from the loss rises ``sens`` and, for sens_cost, ``costs`` = ``{row: (params saved, MACs
    saved)}`` of ``group_sensitivity`` priced in the ``budget`` (params saved, or MACs saved under flops)."""
    from src import alloc_walk
    if kind not in PRIOR_KINDS:
        raise ValueError(f"prior kind {kind!r}: expected one of {PRIOR_KINDS}")
    if budget not in BUDGETS:
        raise ValueError(f"budget={budget!r}; expected one of {BUDGETS}")
    cost = None
    if kind == "sens_cost":
        if costs is None:
            raise ValueError("the sens_cost prior needs the costs of the sensitivity cuts (group_sensitivity costs=)")
        cost = {row: float(costs[row][1 if budget == "flops" else 0]) for row in rows}
    return alloc_walk.weights(kind, {row: float(sens[row]) for row in rows}, a, cost=cost)


def trust_score(v: float, zmax: float) -> float:
    """The score ``residual_decode`` exponentiates for a raw score ``v`` under the trust region (zmax > 0,
    ``SPECTRA_PLAN_RESIDUAL_ZMAX``): zmax · tanh(v / zmax), odd, within ±zmax and ≈ v for |v| ≪ zmax (it reaches ±zmax
    exactly once tanh rounds to 1, |v| ≳ 19 · zmax); ``v`` itself at zmax 0."""
    return float(zmax) * math.tanh(float(v) / float(zmax)) if zmax > 0 else float(v)


def residual_decode(z, prior, pm, kappa: float, k_min: float = 0.1, held=(), iters: int = 48, min_width: int = 1,
                    zmax: float = 0.0):
    """``(widths, info)`` for scores ``z`` (one per ``pm.rows``) as a residual on ``prior`` (``{row: weight}``,
    ``prior_weights``): keep_g = clip(c · w_g · exp(z_g), k_min, 1), c bisected, floored and polished exactly as
    ``scale_decode`` does, whose plan for ``prior`` this is at z = 0 (exp(0) · w = w). A unit of z moves a group's keep
    by about 100 % relative before c re-lands the budget; a group clipped at 1 (c · w_g ≥ 1) only moves once its z goes
    negative enough, and the floors (``k_min``, ``min_width``) bind as in every decoder. ``info`` adds the weights
    decoded (``w``) and the scores.

    ``zmax`` > 0 is the trust region (v20): z_g is replaced by z_eff = ``trust_score(z_g, zmax)`` before exp, so each
    weight stays within e^±zmax of the prior's and, as c then moves by at most as much, each unclipped keep within about
    e^±2·zmax of the prior plan's before rounding; z = 0 is still the prior's plan exactly, and ``info`` adds ``z_eff``
    and ``zmax``. A common shift of z is no longer absorbed by c (tanh is not linear); the policy's mean is centred
    anyway. The squash sits inside this deterministic map from the sampled action z to a plan: the trainer still draws
    z ~ N(μ, σ²) and scores ``log_prob(z)``, so its REINFORCE gradient E[A ∇ log π(z)] stays unbiased for the bounded
    plan's reward, with no Jacobian term (that is owed only by a density over the squashed value, which nothing takes).
    0 = as before: z clipped at ±``Z_CLIP``, which only keeps exp finite (it binds under the trust region only for
    zmax > ``Z_CLIP``)."""
    scores = {row: float(v) for row, v in zip(pm.rows, z)}
    eff = {row: trust_score(v, zmax) for row, v in scores.items()} if zmax > 0 else scores
    w = {row: float(prior[row]) * math.exp(max(-Z_CLIP, min(Z_CLIP, eff[row]))) for row in pm.rows}
    widths, info = scale_decode(w, pm, kappa, k_min, held=held, iters=iters, min_width=min_width)
    info["w"], info["z"] = w, scores
    if zmax > 0:
        info["z_eff"], info["zmax"] = eff, float(zmax)
    return widths, info


# ------------------------------------------------------------------ v36 menu (SPECTRA_PLAN_MENU, default off)
# The agent chooses and blends several no-agent rules per network, then corrects the blend with the residual's scores.
# ``MENU_KINDS`` are the alloc kinds whose plan is a weight per group landed by the scale family clip(c · w, floor, 1),
# which ``scale_decode`` decodes: uniform, sens and sens_cost (``alloc_walk.weights``) and sens_thresh, whose landing
# (``alloc_walk._land``) takes each group's pick at the threshold as its weight. Not rules of a menu: ``inner`` (it
# holds groups, which no weight says), ``widths`` (a table, never landed), ``sens_cost_at`` (it measures again at every
# target), the sampled and agent kinds; ``sens_curve``, ``global_mag`` and ``lamp`` land as sens_thresh does and are
# left out only because nothing asks for them yet.


def check_menu_kinds(kinds, where="the menu"):
    """``kinds`` as a tuple, or a ValueError: a menu lists at least two rules of ``MENU_KINDS``, none twice (the anchor
    alone would be the residual agent itself)."""
    kinds = tuple(str(kind) for kind in kinds)
    bad = [kind for kind in kinds if kind not in MENU_KINDS]
    if bad:
        raise ValueError(f"{where}: {bad[0]!r} is not a menu rule; expected rules of {MENU_KINDS}")
    if len(set(kinds)) != len(kinds):
        raise ValueError(f"{where}: a rule is listed twice")
    if len(kinds) < 2:
        raise ValueError(f"{where}: a menu needs at least two rules, the anchor first")
    return kinds


def menu_kinds(default=None):
    """``SPECTRA_PLAN_MENU`` (v36; unset or empty = off, ``default``): the rules the menu agent blends, a comma list of
    ``MENU_KINDS`` with the anchor first, e.g. ``sens,uniform,sens_cost,sens_thresh``."""
    raw = os.environ.get("SPECTRA_PLAN_MENU", "").strip().lower()
    if not raw:
        return default
    return check_menu_kinds([part.strip() for part in raw.split(",") if part.strip()], f"SPECTRA_PLAN_MENU={raw!r}")


def _menu_number(name, default, positive=False):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: expected a number") from None
    if not -float("inf") < value < float("inf") or (positive and value <= 0.0):  # refuses nan too
        raise ValueError(f"{name}={raw!r}: expected a finite number" + (" above 0" if positive else ""))
    return value


def menu_init() -> float:
    """``SPECTRA_PLAN_MENU_INIT`` (6.0): B, the anchor's logit in the zero-init menu head's bias (B, 0, …, 0)."""
    return _menu_number("SPECTRA_PLAN_MENU_INIT", MENU_INIT)


def menu_sigma() -> float:
    """``SPECTRA_PLAN_MENU_SIGMA`` (0.5): σ_u, the std of the menu logits the trainer samples (u = ψ + σ_u · ε). It is
    constant over a run: the scores' σ schedule does not apply to it."""
    return _menu_number("SPECTRA_PLAN_MENU_SIGMA", MENU_SIGMA, positive=True)


def check_menu_anchor(kinds, mode, budget, where="the menu"):
    """Refuse (ValueError) a menu whose anchor, its first rule, is not the rule the residual already starts from on a
    ``budget`` instance (``prior_kind`` of the ``SPECTRA_PLAN_RESIDUAL`` mode ``mode``), or a menu without the
    residual: the menu blends rules under the residual's scores, and at the anchor alone it is the residual agent."""
    if mode is None or mode == "off":
        raise ValueError(f"{where}: the menu needs the residual (SPECTRA_PLAN_RESIDUAL is off); its anchor "
                         f"{kinds[0]!r} is the residual's prior")
    prior = prior_kind(mode, budget)
    if prior != kinds[0]:
        raise ValueError(f"{where}: the menu's anchor (its first rule) is {kinds[0]!r}, but the residual's prior on a "
                         f"{budget} instance is {prior!r} (SPECTRA_PLAN_RESIDUAL={mode}); list {prior!r} first")


def menu_shares(u, floor: float = 0.0):
    """π = softmax(``u``): the menu's shares, one per rule, as floats. ``floor`` > 0 is for the mean plan only (the
    eval's and the trainer's, u = ψ; ``MENU_SHARE_FLOOR``): a rule holding less than ``floor`` is dropped and the rest
    renormalised. The default init, ψ = (6, 0, …, 0), leaves 0.25 % on every other rule, so the zero-init agent's mean
    plan is the anchor's plan exactly; without the floor that 0.7 % of a menu of four changed widths in 31 of 40 (net,
    budget, target) cells on the tests' five nets, in up to eleven groups by a channel or two. (A one-hot only once the
    anchor holds 1 − 1e-6 would never fire at B = 6, where it holds 0.9926; a temperature sharp enough to reach it
    would put every sampled plan on the anchor too.) The init is exact for B above 5.3 (``plan_trainer.run`` warns
    below). A sampled plan takes the plain softmax."""
    values = [float(v) for v in u]
    top = max(values)
    e = [math.exp(v - top) for v in values]
    total = sum(e)
    pi = [v / total for v in e]
    if floor > 0 and any(p < floor for p in pi):  # nothing under the floor: the softmax as it is
        kept = [p if p >= floor else 0.0 for p in pi]
        total = sum(kept)
        if total > 0:
            pi = [p / total for p in kept]
    return pi


def menu_mix(pi, rules, rows):
    """``{row: weight}``: the blend of ``rules`` (``[{row: weight}]``, ``menu_rules``) at shares ``pi``, the weights the
    residual's scores then multiply: exp(Σ_k π_k log ŵ_k,g), with ŵ_k = w_k over its geometric mean over ``rows`` (a
    rule's scale does not reach the blend, and the decoder's c absorbs the blend's). A pure share (π = e_k) returns rule
    k's weights as given, so its plan is ``scale_decode``'s for that rule bit for bit: c absorbs a scale only up to
    rounding ties, and on the tests' nets the normalised weights of one rule alone changed widths in 7 of 40 cells
    (weights floored to one value put equal-width groups on a rounding boundary together)."""
    pi = [float(p) for p in pi]
    if len(pi) != len(rules):
        raise ValueError(f"menu_mix: {len(pi)} shares for {len(rules)} rules")
    live = [k for k, p in enumerate(pi) if p != 0.0]
    if len(live) == 1 and pi[live[0]] == 1.0:
        return {row: float(rules[live[0]][row]) for row in rows}
    logs = []
    for k in live:
        raw = {row: math.log(max(1e-9, float(rules[k][row]))) for row in rows}
        mean = sum(raw.values()) / len(raw)
        logs.append((pi[k], {row: raw[row] - mean for row in rows}))
    return {row: math.exp(sum(p * log_w[row] for p, log_w in logs)) for row in rows}


def menu_decode(z, pi, rules, pm, kappa: float, k_min: float = 0.1, held=(), iters: int = 48, min_width: int = 1,
                zmax: float = 0.0):
    """``(widths, info)`` of the menu agent's action: shares ``pi`` over ``rules`` and scores ``z`` (one per
    ``pm.rows``). log w_g = Σ_k π_k log ŵ_k,g + z_g (zmax · tanh(z_g / zmax) under the trust region), w_g = exp(·),
    decoded by ``scale_decode``. It is ``residual_decode`` on the blend ``menu_mix`` as its prior, so the Z_CLIP guard,
    the floors, the polish and ``min_width`` are the residual's own; at π = e_anchor it is the residual agent's decode
    bit for bit, and at π = e_k, z = 0 it is ``scale_decode`` of rule k. ``info`` adds ``pi`` and ``base`` (the
    blend, what the scores multiply)."""
    base = menu_mix(pi, rules, pm.rows)
    widths, info = residual_decode(z, base, pm, kappa, k_min, held=held, iters=iters, min_width=min_width, zmax=zmax)
    info["pi"], info["base"] = [float(p) for p in pi], base
    return widths, info


def menu_rules(kinds, pm, kappa: float, sens, costs=None, curves=None, budget: str = "params", a: float = 0.5,
               k_min: float = 0.1, min_width: int = 1, name: str = "the network"):
    """``[{row: weight}]``, one per rule of ``kinds`` over ``pm.rows``: the weights the alloc walk's own row of that
    kind (``SPECTRA_ALLOC_KIND``) hands its landing bisection on this network at the kept-cost target ``kappa``, priced
    in ``budget``. ``uniform``, ``sens`` and ``sens_cost`` are ``alloc_walk.weights`` (the last two through
    ``prior_weights``, from the loss rises ``sens`` and the ``costs`` of the same half cuts) and do not depend on
    ``kappa``. ``sens_thresh`` is ``alloc_walk.curve_weights`` on ``curves`` (``{"qs", "loss", "raw"}``, one
    ``alloc_walk.curve_probes`` measurement per network): each group's pick at the one threshold whose estimated saving
    meets ``kappa``, at least the group's keep floor (``k_min``, lifted to ``min_width`` over its width as
    ``plan_targets`` lifts it), with ``pm.total0`` as the size; a few dozen milliseconds per target. A RuntimeError
    names the measurement a rule lacks."""
    from src import alloc_walk
    rows, out = list(pm.rows), []
    for kind in check_menu_kinds(kinds):
        if kind == "uniform":
            out.append(alloc_walk.weights("uniform", {row: 1.0 for row in rows}, a))
        elif kind == "sens_thresh":
            if curves is None:
                raise RuntimeError(f"{name}: the menu rule sens_thresh needs the origin's damage curves "
                                   "(alloc_walk.curve_probes; NetInstance.from_env with_curves=True)")
            floor = {row: float(k_min) for row in rows}
            if int(min_width) > 1:
                for row in rows:
                    floor[row] = max(float(k_min), min(1.0, int(min_width) / float(pm.widths0[row])))
            curve = alloc_walk.probe_curves(rows, curves["qs"], curves["loss"], curves["raw"], budget)
            out.append(alloc_walk.curve_weights("sens_thresh", curve, floor, pm.total0, kappa, rows)[3])
        else:
            if sens is None or (kind == "sens_cost" and costs is None):
                raise RuntimeError(f"{name}: the menu rule {kind} needs the sensitivities"
                                   + (" and costs" if kind == "sens_cost" else "") + " of the origin")
            out.append(prior_weights(kind, rows, sens, costs, budget, a))
    return out


def menu_features(rules, pm, target: float, k_min: float = 0.1, min_width: int = 1, kappa=None):
    """``(x_net, plans)``: the menu head's network-level inputs and each rule's own plan at ``target``
    (``scale_decode`` with the agent's floors, ``[(widths, info)]`` in the rules' order). x_net is
    log(params kept by rule k's plan / params kept by the anchor's plan) for each rule after the anchor, then κ
    (``kappa``, the target the state carries; ``target`` when None, as in the trainer): what each rule's plan costs in
    parameters at an equal budget, known before any fine-tune. On a params budget the ratios sit near 0."""
    plans = [scale_decode(w, pm, target, k_min, min_width=min_width) for w in rules]
    params = [float(pm.params(widths)) for widths, _info in plans]
    x_net = [math.log(p / params[0]) for p in params[1:]] + [float(target if kappa is None else kappa)]
    return x_net, plans


def _complement(kept, total):
    mask = torch.ones(int(total), dtype=torch.bool)
    mask[kept.detach().cpu().long()] = False
    return torch.nonzero(mask, as_tuple=False).flatten()


@torch.no_grad()
def _zero_group(group, keep_idx, record=None):
    """Zero what ``pruning.prune_group_structurally`` would remove for survivors ``keep_idx``. A list ``record`` also
    gets one ``(parameter, idx, dim)`` per zeroing (the parameter itself, ``idx`` on its device, ``dim`` the axis
    indexed), which ``MaskedCut.remask`` replays."""
    from src import pruning
    width = int(group.width)
    keep = keep_idx.detach().cpu().long()
    removed = _complement(keep, width)

    def zero(param, idx, dim):
        idx = idx.to(param.device)
        if dim == 0:
            param[idx] = 0
        else:
            param[:, idx] = 0
        if record is not None:
            record.append((param, idx, dim))

    for module in list(group.producers) + list(group.depthwise):
        zero(module.weight, removed, 0)
        if module.bias is not None:
            zero(module.bias, removed, 0)
    for ref in group.consumers:
        module = ref.module
        kept = pruning.surviving_input_channels(ref, width, keep)
        if isinstance(module, nn.Conv2d):
            total = module.in_channels
        else:
            total = module.in_features
            if module.in_features != ref.total:
                kept = pruning._expand_indices_for_flatten(kept, ref.total, module.in_features)
        zero(module.weight, _complement(kept, total), 1)
    for ref in group.norms:
        norm = ref.module
        idx = _complement(pruning.surviving_input_channels(ref, width, keep), norm.num_features)
        if norm.weight is not None:
            zero(norm.weight, idx, 0)
            zero(norm.bias, idx, 0)


class MaskedCut:
    """``alloc_walk.cut_to``'s L1 cut written as zeros on a working copy: same survivors, same function.

    Groups are cut in plan order and each group's survivors are ranked on the copy as the earlier groups
    left it, as in the real sequential cut (a zeroed input slice adds nothing to an L1 filter sum, like a
    removed one). The copy keeps its shapes, so its parameter count is ``ParamModel``'s, not its own.
    """

    def __init__(self, model: nn.Module):
        from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
        from src import channel_groups
        from src.group_sensitivity import group_plan
        self.work = copy.deepcopy(model)
        mwr = ModelWithRows(self.work)
        self.plan = group_plan(mwr, channel_groups.build_channel_groups(self.work) or [])
        self.origin = {key: value.detach().clone() for key, value in self.work.state_dict().items()}
        self._masks = []

    @property
    def rows(self):
        return [int(row) for _group, row in self.plan]

    def cut(self, rates):
        from src import pruning
        self.work.load_state_dict(self.origin)
        self._masks = []
        with torch.no_grad():
            for group, row in self.plan:
                rate = float(rates.get(row, 1.0))
                if rate >= 1.0 - 1e-9:
                    continue
                keep = pruning.select_group_survivors(group, rate, mode="l1")
                if keep is None:
                    raise RuntimeError(f"group at row {row}: no L1 survivors")
                _zero_group(group, keep, record=self._masks)
        return self.work

    @torch.no_grad()
    def remask(self):
        """Zero again every entry the last ``cut`` zeroed (what a fine-tune step may have moved)."""
        for param, idx, dim in self._masks:
            if dim == 0:
                param[idx] = 0
            else:
                param[:, idx] = 0


def _no_dropout(module: nn.Module):
    for sub in module.modules():
        if isinstance(sub, nn.Dropout):
            sub.p = 0.0
        elif isinstance(sub, nn.MultiheadAttention):
            sub.dropout = 0.0


class PlanPolicy(nn.Module):
    """State encoder + per-token linear head → one centred score per planned group (all 0 at init).

    ``menu`` = K > 0 (v36, ``SPECTRA_PLAN_MENU``; 0 = no menu, the module as before) adds ``menu_head``, a Linear map
    from the mean token output and the K network-level inputs of ``menu_features`` to the menu's K logits. Its
    weights start at zero and its bias at (``menu_init``, 0, …, 0), so the agent starts on the anchor whatever the
    encoder reads. Its constructor's random draw is put back, so the encoder and the score head get the weights, and
    the global RNG the state, of a policy without a menu.
    """

    def __init__(self, feature_dim: int, encoder: str = "transformer", menu: int = 0, menu_init: float = MENU_INIT):
        super().__init__()
        from src.Model.StateEncoder import build_state_encoder
        self.feature_dim, self.encoder_kind = int(feature_dim), encoder
        self.encoder = build_state_encoder(encoder, self.feature_dim)
        self.encoder.pool = lambda encoded, target_index, n: encoded[0, :n]
        _no_dropout(self.encoder)
        self.head = nn.Linear(self.encoder.output_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.menu = int(menu)
        if self.menu:
            rng = torch.get_rng_state()
            self.menu_head = nn.Linear(self.encoder.output_dim + self.menu, self.menu)
            torch.set_rng_state(rng)
            with torch.no_grad():
                self.menu_head.weight.zero_()
                self.menu_head.bias.zero_()
                self.menu_head.bias[0] = float(menu_init)

    def forward(self, state, token_mask, token_k, n_groups):
        scores = self.head(self.encoder(state)).squeeze(-1)[token_mask]
        sums = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype).index_add(0, token_k, scores)
        count = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype).index_add(
            0, token_k, torch.ones_like(scores))
        mu = sums / count.clamp(min=1.0)
        return mu - mu.mean()

    def menu_logits(self, state, x_net):
        """ψ, the menu's K logits (v36): ``menu_head`` on [the mean of the encoder's outputs over all of the net's
        tokens ; ``x_net``] (``menu_features``: K − 1 plan-cost ratios and κ). A second pass through the encoder,
        beside ``forward``'s, which stays as it was; the gradient reaches the encoder through both."""
        encoded = self.encoder(state)
        x_net = torch.as_tensor(x_net, dtype=encoded.dtype, device=encoded.device)
        if x_net.numel() != self.menu:
            raise ValueError(f"menu_logits: {x_net.numel()} network-level inputs for a menu of {self.menu} rules")
        return self.menu_head(torch.cat([encoded.mean(dim=0), x_net]))


def sample_scores(mu, sigma: float, k: int, generator=None):
    eps = torch.randn((k, mu.numel()), generator=generator, device=mu.device, dtype=mu.dtype)
    return mu.detach().unsqueeze(0) + float(sigma) * eps


def log_prob(z, mu, sigma: float):
    """Gaussian log-density of each row of ``z`` around ``mu`` (constants dropped)."""
    return (-(z - mu.unsqueeze(0)) ** 2 / (2.0 * float(sigma) ** 2)).sum(dim=-1)


def menu_log_prob(z, mu, sigma: float, u, psi, sigma_u: float):
    """log π(action) of the menu agent (v36), one per sampled plan: its action is the scores and the menu logits, drawn
    independently, so this is ``log_prob`` of each row of ``z`` around ``mu`` plus the Gaussian log-density of the same
    row of ``u`` around ``psi`` at std ``sigma_u`` (constants dropped). As with the trust region, the softmax and the
    decode are a fixed map from the action to a plan, so no Jacobian term is owed."""
    return log_prob(z, mu, sigma) + log_prob(u, psi, sigma_u)


def save_policy(policy: PlanPolicy, path: str, **meta):
    if len(meta.get("menu") or ()) != int(getattr(policy, "menu", 0)):  # v36; 0 against 0 without a menu
        raise ValueError(f"save_policy: the policy has a menu head for {getattr(policy, 'menu', 0)} rules, the "
                         f"checkpoint meta lists menu={meta.get('menu')!r}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "feature_dim": policy.feature_dim,
                "encoder": policy.encoder_kind, **meta}, path)


def read_blob(path: str):
    """The checkpoint's dict as saved (no module is built, so the global RNG is untouched)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def check_menu(blob, where="the plan-agent checkpoint"):
    """The menu contract of a checkpoint against this job's ``SPECTRA_PLAN_MENU`` (v36): the checkpoint's rules as a
    tuple, or None when neither has a menu (every checkpoint and job before v36). Refused (ValueError): a menu policy
    without the flag, a menu policy under another menu (other rules, or another order: the head's outputs follow it),
    and a policy without a menu under the flag."""
    saved = tuple(blob.get("menu") or ()) or None
    flag = menu_kinds()
    if saved is None and flag is None:
        return None
    if saved is None:
        raise ValueError(f"{where} has no menu, but SPECTRA_PLAN_MENU={','.join(flag)} is set; unset it for this agent")
    if flag is None:
        raise ValueError(f"{where} is a menu policy; set SPECTRA_PLAN_MENU={','.join(saved)} to load it")
    if flag != saved:
        raise ValueError(f"{where} trained with the menu {','.join(saved)}; SPECTRA_PLAN_MENU={','.join(flag)} is "
                         "another menu")
    return saved


def policy_of(blob, device="cpu"):
    kinds = check_menu(blob)
    extra = {} if kinds is None else {"menu": len(kinds), "menu_init": float(blob.get("menu_init", MENU_INIT))}
    policy = PlanPolicy(blob["feature_dim"], blob.get("encoder", "transformer"), **extra)
    policy.load_state_dict(blob["state_dict"])
    return policy.to(device).eval()


def load_policy(path: str, device="cpu"):
    blob = read_blob(path)
    return policy_of(blob, device), blob


def _labels(y):
    return y.argmax(dim=1) if y.dim() > 1 and y.shape[1] > 1 else y.long()


def eval_batches(loader, device):
    """The loader's batches on ``device`` (labels as class indices), read once."""
    return [(x.to(device), _labels(y).to(device)) for x, y in loader]


def recal_batches(loader, n, seed, device):
    """The first ``n`` batches the loader yields under ``seed`` (``plan_proxies.seeded``), as D-PROXY's bn<N> sees them."""
    from src.plan_proxies import seeded
    out = []
    with seeded(seed):
        for batch in loader:
            out.append((batch[0] if isinstance(batch, (tuple, list)) else batch).to(device))
            if len(out) >= int(n):
                break
    return out


def ft_batches(loader, n, seed, device):
    """The first ``n`` ``(x, y)`` batches the loader yields under ``seed`` (``plan_proxies.seeded``), labels as class
    indices: the fixed batches of the ``fs<S>`` proxy."""
    from src.plan_proxies import seeded
    out = []
    with seeded(seed):
        for x, y in loader:
            out.append((x.to(device), _labels(y).to(device)))
            if len(out) >= int(n):
                break
    return out


def short_finetune(model, batches, lr, after_step=None):
    """SGD (momentum ``FS_MOMENTUM``, weight decay ``FS_WD``) at ``lr``, one step per ``(x, y)`` in ``batches``, in
    train mode with every parameter trainable (the flags are put back); ``after_step`` runs without grad after each
    step (``MaskedCut.remask``). The model is left in eval mode."""
    with torch.enable_grad():
        flags = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad = True
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=FS_MOMENTUM, weight_decay=FS_WD)
        for x, y in batches:
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
            if after_step is not None:
                with torch.no_grad():
                    after_step()
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad = flag
        model.eval()


@torch.no_grad()
def accuracy(model, batches) -> float:
    was_training = model.training
    model.eval()
    correct = total = 0
    for x, y in batches:
        correct += int((model(x).argmax(dim=1) == y).sum())
        total += int(y.numel())
    model.train(was_training)
    return correct / max(1, total)


def default_handler(model):
    """The learning handler ``NetworkEnv.create_learning_handler`` builds, a ``ClassificationHandler`` with
    cross-entropy (its KD teacher plays no part under ``plan_proxies.handler_recipe``): what the ``ft<E>`` reward
    fine-tunes with when an instance was built without an env (``NetInstance.from_env`` passes the env's own)."""
    from src.ModelHandlers.ClassificationHandler import ClassificationHandler
    return ClassificationHandler(model, nn.CrossEntropyLoss())


@contextlib.contextmanager
def _captured(into):
    """Route ``utils.print_flush`` (the handler's recipe and epoch lines go through it, bound at call time) into the
    list ``into`` for the body, then put it back."""
    import src.utils as utils
    before = utils.print_flush
    utils.print_flush = lambda msg, *args, **kwargs: into.append(str(msg))
    try:
        yield
    finally:
        utils.print_flush = before


def _digits(text: str) -> bool:
    return text.isascii() and text.isdigit()


def parse_proxy(name: str, judge: bool = False):
    """``("cut", 0)``, ``("bn", N)``, ``("fs", S)`` or ``("ft", E)`` (v23: E ≥ 1 epochs of the handler's fine-tune).

    ``judge=True`` (v32, default off; the safety net's judge grammar, read by ``plan_proxies.PlanScorer`` and
    ``alloc_walk.fallback_proxies``) returns a triple ``(kind, n, k)`` instead: the same names, plus ``rc<E>`` (kind
    ``"rc"``, E ≥ 1 epochs of the final recovery recipe, cosine compressed to E epochs) and the repeat suffix ``x<k>``
    (k ≥ 2, on ``ft<E>`` and ``rc<E>`` only: the mean of k draws), ``k`` being 1 without a suffix. The plan trainer's
    grammar (``judge=False``, its two-tuple) is untouched: ``rc<E>`` and ``x<k>`` are refused there."""
    name = (name or "").strip().lower()
    if not judge:
        if name == "cut":
            return "cut", 0
        for kind in ("bn", "fs", "ft"):
            if name.startswith(kind) and name[2:].isdigit() and int(name[2:]) > 0:
                return kind, int(name[2:])
        raise ValueError(f"plan reward proxy {name!r}: expected cut, bn<N>, fs<S> or ft<E>")
    if name == "cut":
        return "cut", 0, 1
    for kind in ("bn", "fs", "ft", "rc"):
        if not name.startswith(kind):
            continue
        head, sep, tail = name[2:].partition("x")
        if not (_digits(head) and int(head) > 0 and (not sep or _digits(tail))):
            break
        if not sep:
            return kind, int(head), 1
        if kind not in ("ft", "rc"):
            raise ValueError(f"plan reward proxy {name!r}: the repeat suffix x<k> is allowed on ft<E> and rc<E> only")
        if int(tail) < 2:
            raise ValueError(f"plan reward proxy {name!r}: the repeat suffix needs k >= 2")
        return kind, int(head), int(tail)
    raise ValueError(f"plan reward proxy {name!r}: expected cut, bn<N>, fs<S>, ft<E>, rc<E>, ft<E>x<k> or rc<E>x<k>")


def spans_of(layout):
    return {name: (start, end) for name, start, end in layout}


def plan_state(state, spans, kappa=None, zero=(), budget="params"):
    """The policy's input: the target channels at (κ, 1), the per-step action slots and ``zero``'s channels at 0; the
    first action slot then carries the budget the plan is decoded on (0 params, 1 FLOPs)."""
    if budget not in BUDGETS:
        raise ValueError(f"budget={budget!r}; expected one of {BUDGETS}")
    feats = state["layer_features"].clone()
    if kappa is not None and "target" in spans:
        start = spans["target"][0]
        feats[:, start] = float(kappa)
        feats[:, start + 1] = 1.0
    for name in ("action",) + tuple(zero):
        if name in spans:
            start, end = spans[name]
            feats[:, start:end] = 0.0
    if budget == "flops" and "action" in spans:
        feats[:, spans["action"][0]] = 1.0
    out = dict(state)
    out["layer_features"] = feats
    return out


def token_index(token_rows, rows, device):
    """``(mask over tokens, group index per mapped token)`` for a token → first-walk-row map."""
    index = {row: k for k, row in enumerate(rows)}
    mask = torch.tensor([int(r) in index for r in token_rows], dtype=torch.bool, device=device)
    k = torch.tensor([index[int(r)] for r in token_rows if int(r) in index], dtype=torch.long, device=device)
    return mask, k


class NetInstance:
    """A catalog net's origin, read once; ``reward`` scores one plan on it."""

    def __init__(self, name, model, plan, state, token_rows, layout, train_loader, val_loader, device,
                 zero=(), sens=None, input_shape=None, cost=None, create_handler=None, curves=None):
        started = time.perf_counter()
        self.name, self.device, self.zero, self.input_shape = str(name), device, tuple(zero), input_shape
        # model -> learning handler, for the ft<E> reward (v23): the env's own ``create_learning_handler`` through
        # ``from_env``, else the handler the env would build; ``recipe_line`` is the last ft<E> fine-tune's recipe
        self.create_handler = create_handler if create_handler is not None else default_handler
        self.recipe_line = None
        self.model = model.to(device)
        self.plan = plan
        self.pm = ParamModel(self.model, plan)
        self.fm = None  # the FLOPs model, probed the first time a flops budget asks for it (cost_model)
        self.cutter = MaskedCut(self.model)
        if self.cutter.rows != self.pm.rows:
            raise RuntimeError(f"{self.name}: the working copy plans rows {self.cutter.rows}, the origin {self.pm.rows}")
        self.spans = spans_of(layout)
        self.state = {key: (value.detach().to(device) if torch.is_tensor(value) else value)
                      for key, value in state.items() if key != "token_members"}
        self.feature_dim = int(self.state["layer_features"].size(1))
        self.token_mask, self.token_k = token_index(token_rows, self.pm.rows, device)
        self.n_groups = len(self.pm.rows)
        if self.n_groups == 0 or int(self.token_k.numel()) == 0:
            raise RuntimeError(f"{self.name}: no planned group has a token")
        missing = set(range(self.n_groups)) - set(self.token_k.tolist())
        if missing:
            raise RuntimeError(f"{self.name}: {len(missing)} planned groups have no token")
        self.train_loader = train_loader
        self.val = eval_batches(val_loader, device)
        self.origin_val = accuracy(self.model, self.val)
        self.sens = sens
        self.cost = cost  # {row: (params saved, MACs saved)} by the sensitivity half cut: the sens_cost prior (T2)
        # v36: {"qs", "loss", "raw", "seconds"}, alloc_walk.curve_probes on the origin: the menu's sens_thresh rule
        self.curves = curves
        self._prior = {}
        self.held = {int(row) for group, row in plan if len(group.producers) > 1}
        self.seconds = time.perf_counter() - started

    @classmethod
    def from_env(cls, env, net_path, net_model, net_loaders, kappa0=0.6, tokens="layer", zero=(), with_sens=True,
                 with_cost=False, with_curves=False):
        """Reset ``env`` on the net and read its origin as the D-IMIT dump does; ``with_cost`` also prices the
        sensitivity cuts (``group_sensitivity`` ``costs=``) for the sens_cost prior. ``with_curves`` (v36, the menu's
        sens_thresh rule) also measures the damage curves (``alloc_walk.curve_probes``) on the sensitivities'
        calibration batches, after them, keeps the seconds that took and puts every RNG back
        (``alloc_walk.rng_restore``), so the job draws from then on as it does without them."""
        from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
        from src import group_sensitivity, state_dump
        from src.BERTInputModeler import action_cost_slot_dim
        from src.group_sensitivity import group_plan
        from src.group_tokens import GROUP_TOKEN_EXTRA_DIM, group_token_state
        if tokens not in TOKEN_KINDS:
            raise ValueError(f"tokens={tokens!r}; expected one of {TOKEN_KINDS}")
        state = env.reset(test_net_path=net_path, test_model=net_model, test_loaders=net_loaders,
                          target_keep=kappa0)
        device = env.conf.device
        model = env.current_model.to(device)
        mwr = ModelWithRows(model)
        groups = env._dependency_groups(mwr)
        plan = group_plan(mwr, groups)
        rows = state_dump.token_rows(mwr, groups, plan)
        num_actions = len(env.conf.compression_rates_dict)
        extra = 0
        if tokens == "group":
            state = group_token_state(state, mwr.all_layers, groups, slot_dim=action_cost_slot_dim(num_actions))
            if "token_members" not in state:
                raise RuntimeError(f"{os.path.basename(str(net_path))}: no group tokens")
            rows, _conflicts = state_dump.group_token_rows(state["token_members"], rows)
            extra = GROUP_TOKEN_EXTRA_DIM
        layout = state_dump.layout(int(state["layer_features"].size(1)), num_actions, extra)
        sens = cost = None
        if with_sens:
            batches = group_sensitivity.calibration_batches(env.train_loader, group_sensitivity.CALIB_BATCHES, device)
            cost = {} if with_cost else None
            sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, env._input_shape(), costs=cost)
        curves = None
        if with_curves:
            if not with_sens:
                raise ValueError("with_curves measures on the sensitivities' calibration batches: it needs with_sens")
            from src import alloc_walk
            tick, rng = time.perf_counter(), alloc_walk.rng_snapshot()
            qs, loss, raw = alloc_walk.curve_probes(model, plan, batches, env._input_shape())
            alloc_walk.rng_restore(rng)  # its cuts build modules, which draw: the job's RNG stays the one without them
            curves = {"qs": qs, "loss": loss, "raw": raw, "seconds": time.perf_counter() - tick}
        return cls(os.path.basename(str(net_path)), model, plan, state, rows, layout,
                   env.train_loader, env.val_loader, device, zero=zero, sens=sens, input_shape=env._input_shape(),
                   cost=cost, create_handler=env.create_learning_handler, curves=curves)

    def check(self, kappa: float = 0.6):
        """The uniform plan at ``kappa`` cut for real (``alloc_walk.cut_to``): analytic vs real kept params, masked vs
        real Δ val without recalibration, and the planned groups' dead channels (all-zero filters, which the real
        cut counts out of the survivors)."""
        from src import alloc_walk, pruning
        widths, info = self.reference("uniform", kappa)
        real = alloc_walk.cut_to(self.model, self.plan, rates_of(widths, self.pm.widths0), self.input_shape)
        real = real.to(self.device)
        out = {"analytic": float(info["kept"]),
               "real": sum(p.numel() for p in real.parameters()) / float(self.pm.total0),
               "real_r": 100.0 * (accuracy(real, self.val) - self.origin_val)}
        del real
        out["masked_r"] = self.reward(widths, "cut", [])
        self.cutter.work.load_state_dict(self.cutter.origin)
        dead = 0
        for group, _row in self.cutter.plan:
            importance = pruning.group_importance(group, "l1")
            dead += int((importance <= 0).sum()) if importance is not None else 0
        out["dead"] = dead
        return out

    def check_flops(self, kappa: float = 0.6):
        """The uniform plan at ``kappa`` on the FLOPs model: analytic vs real (``calc_flops`` of ``alloc_walk.cut_to``)
        kept FLOPs, and the model's probe count and seconds."""
        from src import alloc_walk
        import src.utils as utils
        fm = self.cost_model("flops")
        widths, info = self.reference("uniform", kappa, budget="flops")
        real = alloc_walk.cut_to(self.model, self.plan, rates_of(widths, self.pm.widths0), self.input_shape)
        out = {"analytic": float(info["kept"]),
               "real": float(utils.calc_flops(real, self.input_shape, self.device)) / fm.total0,
               "probes": fm.probes, "seconds": fm.seconds}
        del real
        return out

    def cost_model(self, budget: str = "params"):
        """``ParamModel`` for ``params``; for ``flops`` the net's ``FlopModel``, built on first use."""
        if budget not in BUDGETS:
            raise ValueError(f"budget={budget!r}; expected one of {BUDGETS}")
        if budget == "params":
            return self.pm
        if self.fm is None:
            self.fm = FlopModel(self.model, self.plan, self.input_shape, self.device)
        return self.fm

    def state_at(self, kappa: float, budget: str = "params"):
        return plan_state(self.state, self.spans, kappa, self.zero, budget)

    def batches(self, proxy: str, seed: int):
        """What ``reward`` needs for ``proxy`` under the instance's ``seed``, drawn once and shared by all its plans
        (common random numbers): the fixed ``(x, y)`` batches of ``fs<S>``, the recalibration batches of ``bn<N>``,
        nothing for ``cut``, and for ``ft<E>`` the seed itself (its fine-tune draws its epochs from the loader under
        it)."""
        kind, n = parse_proxy(proxy)
        if kind == "ft":
            return int(seed)
        if kind == "fs":
            return ft_batches(self.train_loader, n, seed, self.device)
        return recal_batches(self.train_loader, n, seed, self.device) if kind == "bn" else []

    def reward(self, widths, proxy: str, batches) -> float:
        """Δ val accuracy (pp) of one cut to ``widths``, after ``proxy`` (BatchNorm re-estimated on ``batches``, or the
        masked copy fine-tuned a step per ``(x, y)`` in ``batches``, re-masked after every step). ``ft<E>`` (v23)
        instead cuts the net for real and fine-tunes that copy E epochs with the handler's own recipe
        (``finetune_reward``); ``batches`` is then the seed ``batches`` returned for it."""
        from src import recovery_edits
        kind, n = parse_proxy(proxy)
        if kind == "ft":
            if isinstance(batches, bool) or not isinstance(batches, int):
                raise ValueError(f"the {proxy} reward takes the seed batches() returns for it, "
                                 f"not {type(batches).__name__}")
            return self.finetune_reward(widths, n, batches)
        work = self.cutter.cut(rates_of(widths, self.pm.widths0))
        if kind == "bn":
            recovery_edits.recalibrate_batchnorm(work, batches, self.device, len(batches))
        elif kind == "fs":
            short_finetune(work, batches, fs_lr(), after_step=self.cutter.remask)
        return 100.0 * (accuracy(work, self.val) - self.origin_val)

    def finetune_reward(self, widths, epochs: int, seed: int) -> float:
        """The ``ft<E>`` reward (v23): Δ val accuracy (pp) of the net cut for real to ``widths`` (``alloc_walk.cut_to``,
        the structural cut the eval walk makes; never the masked copy, which the handler could not re-mask) after
        ``epochs`` epochs of the learning handler's own fine-tune, the eval-side proxy's recipe
        (``plan_proxies.measure_one``): ``train_model`` on this net's train loader with ``max_epochs`` = E and patience
        E + 1, every parameter trainable, under ``handler_recipe`` (the ``SPECTRA_FT_*`` keys cleared and put back) and
        ``seeded(seed)``, entered after the cut: the cut rebuilds modules, whose constructors draw from the global RNG by
        an amount that depends on the plan, so the seed is set right before the fine-tune and every plan of an instance
        starts it from the same state (the same shuffle, and the same crops and flips when the loader draws them in
        this process; a persistent worker keeps its own stream), as the eval side seeds its candidate's. An outer
        ``seeded`` puts the global torch / CUDA / random / numpy state back after the whole reward. The handler's own
        lines (recipe, epochs) are captured rather than printed once per plan: the recipe, without its trailing
        ``trainable`` count (the plan's), is kept in ``recipe_line`` for the trainer's once-per-run line, and a
        fine-tune that did not run (no recipe line, or an empty loader) raises rather than return the bare cut's own
        accuracy."""
        from src import alloc_walk
        from src.plan_proxies import handler_recipe, seeded
        epochs, lines = int(epochs), []
        with seeded(seed):
            real = alloc_walk.cut_to(self.model, self.plan, rates_of(widths, self.pm.widths0), self.input_shape)
            real = real.to(self.device)
            for param in real.parameters():
                param.requires_grad = True
            with seeded(seed), handler_recipe(), _captured(lines):
                self.create_handler(real).train_model(self.train_loader, allow_reinit_retry=False, max_epochs=epochs,
                                                      patience=epochs + 1, tag=f"plan ft{epochs} {self.name}")
            acc = accuracy(real, self.val)
        del real
        recipe = [line.split("Fine-tune recipe:", 1)[1].strip() for line in lines if "Fine-tune recipe:" in line]
        if not recipe or any("produced no loss" in line for line in lines):
            raise RuntimeError(f"{self.name}: the ft{epochs} fine-tune did not run: " + " | ".join(lines[-3:]))
        if any(" = nan" in line.lower() or " = inf" in line.lower() for line in lines if "loss" in line.lower()):
            import src.utils as utils  # the captured lines never reach the job log, which smoke gates grep for NaN
            utils.print_flush(f"[plan] WARNING: {self.name}: the ft{epochs} fine-tune's loss went NaN or inf for this "
                              "plan; its reward reads the diverged net")
        self.recipe_line = recipe[-1].split(" trainable=", 1)[0]
        return 100.0 * (acc - self.origin_val)

    def prior(self, kind: str, budget: str = "params", a: float = 0.5):
        """``{row: weight}`` of the ``sens`` / ``sens_cost`` rule on this net priced in ``budget`` (T2's prior,
        ``prior_weights``), computed once per (kind, budget, a); None without the sensitivities (or, for sens_cost,
        the costs) it needs."""
        if self.sens is None or (kind == "sens_cost" and self.cost is None):
            return None
        key = (str(kind), str(budget), float(a))
        if key not in self._prior:
            self._prior[key] = prior_weights(kind, self.pm.rows, self.sens, self.cost, budget, a)
        return self._prior[key]

    def rules_at(self, kinds, kappa: float, budget: str = "params", a: float = 0.5, k_min: float = 0.1,
                 min_width: int = 1):
        """``[{row: weight}]``: the menu's rules ``kinds`` on this net at ``kappa`` priced in ``budget`` (v36,
        ``menu_rules`` on the origin's sensitivities, costs and curves, with the decoders' ``k_min`` and
        ``min_width``); the anchor's entry equals ``prior`` of that kind. A RuntimeError names what a rule lacks."""
        return menu_rules(kinds, self.cost_model(budget), kappa, self.sens, self.cost, self.curves, budget, a, k_min,
                          min_width, name=self.name)

    def reference(self, kind: str, kappa: float, k_min: float = 0.1, a: float = 0.5, budget: str = "params",
                  min_width: int = 1):
        """``(widths, info)`` of the uniform / sens / inner plan at ``kappa`` on the ``budget`` cost model; None for
        sens without sensitivities. ``min_width`` as ``decode``. ``sens_cost`` (the T2 prior on a params budget) is
        the same family on ``prior``'s weights; None without the costs."""
        from src import alloc_walk
        if kind == "sens_cost":
            weights = self.prior(kind, budget, a)
            if weights is None:
                return None
            return scale_decode(weights, self.cost_model(budget), kappa, k_min, min_width=min_width)
        if kind == "sens" and self.sens is None:
            return None
        sens = self.sens if kind == "sens" else {row: 1.0 for row in self.pm.rows}
        weights = alloc_walk.weights(kind, {row: float(sens[row]) for row in self.pm.rows}, a)
        return scale_decode(weights, self.cost_model(budget), kappa, k_min, held=self.held if kind == "inner" else (),
                            min_width=min_width)


def plan_for_env(env, target: float, policy_path: str, k_min: float = 0.1, sample=None, budget="params", kappa=None):
    """``(widths, info)`` of the frozen agent's mean plan for the net ``env`` was reset on, in
    ``alloc_walk.plan_targets``' format; the state is the one reset built (``state_dump.encode_origin``).
    With ``sample=(sigma, seed)`` it decodes one draw z = mu + sigma * eps (eps from a CPU generator seeded by
    ``seed``) instead of the mean, the plan distribution the trainer samples from. ``budget="flops"`` decodes
    ``target`` on a ``FlopModel`` (``kept`` is then the kept FLOPs) and flags the budget in the state; ``kappa``
    is the target the state carries (default ``env.target_keep``, else ``target``). The decoder's width floor is
    ``SPECTRA_PLAN_MIN_WIDTH`` when that is set above 1, else the floor the policy trained with (``min_width`` in
    the checkpoint, written by the trainer only when it was above 1), else none.

    A T2 checkpoint (``residual`` in the blob, or ``SPECTRA_PLAN_RESIDUAL`` set: ``eval_residual``) is decoded with
    ``residual_decode`` on the prior rule's weights for this net and ``budget``, measured through the calibration path
    the alloc walk's sens / sens_cost kinds use (``group_sensitivity.calibration_batches`` on ``env.train_loader`` at
    the point ``alloc_walk._state`` draws them, before the policy module is built, so the shuffled loader sees the
    same global RNG state and the prior is that comparator's rule); ``info`` then carries ``residual`` and a
    ``prior`` entry with the rule's own plan and the agent's distance from it. Under the trust region (``residual_zmax``
    in the blob, or ``SPECTRA_PLAN_RESIDUAL_ZMAX``, above 0) the decode is bounded by it, and ``prior`` adds ``zmax``
    so the grid rounding ranks the groups by the scores decoded (``alloc_walk.grid_priority``).

    v22 (ENS, default off): ``policy_path`` may list several checkpoints, comma-separated (``ensemble_paths``). The
    agent is then one frozen policy, the mean of their scores mu = (1/n) sum_i mu_i, decoded once. Every member must
    agree with the first on ``ENSEMBLE_KEYS``. The global RNG is put back after the extra modules are built, so the
    recovery downstream draws as it does with one checkpoint. ``info`` then carries ``ensemble`` (n, and the members'
    spread: the mean over groups of their scores' standard deviation). One path is decoded exactly as before.

    v36 (menu, default off): a menu checkpoint (``menu`` in the blob) is loaded only under the same
    ``SPECTRA_PLAN_MENU`` (``check_menu``; a plain checkpoint under the flag is refused too), and its anchor must be the
    rule the residual starts from on this ``budget`` (``check_menu_anchor``). The prior's sensitivities are measured
    where and as before (with the costs when the menu lists sens_cost), then the damage curves on the same batches when
    it lists sens_thresh (the RNG is put back after them, so the recovery downstream draws as after the prior's
    measurement alone); ``menu_rules`` gives every rule's weights at ``target``, ``menu_features`` their plans' costs
    (with ``kappa``, the target the state carries, as κ), the policy's menu head the logits ψ, and the frozen mean plan
    is ``menu_decode`` of the mean scores at the mean shares (``menu_shares`` with the checkpoint's
    ``menu_share_floor``). ``info`` gains ``menu`` (kinds, pi, psi, x, each rule's own plan's kept budget and params
    ratio, seconds) and the residual line ends with the shares. ``info["prior"]`` still describes the anchor's own
    plan, except that its ``weights`` are the blend's (``info["base"]`` of the decode: what the scores multiplied), so
    ``alloc_walk.grid_priority`` ranks the groups by what was decoded; at the anchor alone the two are the same. A
    sampled plan (``sample``) and an ensemble are not defined for a menu policy yet and are refused."""
    from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
    from src import fortify, group_sensitivity, state_dump
    from src.BERTInputModeler import action_cost_slot_dim
    from src.group_sensitivity import group_plan
    from src.group_tokens import GROUP_TOKEN_EXTRA_DIM, group_token_state
    import src.utils as utils
    from src.feature_standardizer import resolve_standardizer_path
    if budget not in BUDGETS:
        raise ValueError(f"budget={budget!r}; expected one of {BUDGETS}")
    device = env.conf.device
    paths = ensemble_paths(policy_path)
    blob = read_blob(paths[0])
    members = [read_blob(p) for p in paths[1:]]
    for p, other in zip(paths[1:], members):
        for key in ENSEMBLE_KEYS:
            if other.get(key) != blob.get(key):
                raise ValueError(f"plan-agent ensemble: {p} differs from {paths[0]} in {key} "
                                 f"({other.get(key)!r} against {blob.get(key)!r})")
    kinds = check_menu(blob, f"the plan agent at {paths[0]}")  # v36; None with no menu anywhere: all below as before
    if kinds is not None and (members or sample is not None):
        raise ValueError(f"the plan agent at {paths[0]} is a menu policy: an ensemble and a sampled plan "
                         "(agent_sample) are not defined for it yet; plan with one checkpoint's mean (agent)")
    residual, alpha, zmax = eval_residual(blob)
    if kinds is not None:
        check_menu_anchor(kinds, residual, budget, f"the plan agent at {paths[0]}")
    batches = None
    if residual != "off":
        batches = group_sensitivity.calibration_batches(env.train_loader, group_sensitivity.CALIB_BATCHES, device)
    policy = policy_of(blob, device)
    others = []
    if members:  # building more modules moves the global RNG that the recovery draws from later: put it back
        rng = torch.get_rng_state()
        others = [policy_of(b, device) for b in members]
        torch.set_rng_state(rng)
    trained, here = blob.get("standardizer") or "", resolve_standardizer_path() or ""
    if trained and os.path.abspath(trained) != os.path.abspath(here):
        utils.print_flush(f"[alloc] WARNING: the plan agent trained under standardizer {trained}; this job reads {here}")
    trained_budget = str(blob.get("budget", "params"))
    if trained_budget not in (budget, "mixed"):
        utils.print_flush(f"[alloc] WARNING: the plan agent trained under a {trained_budget} budget; "
                          f"this job decodes on {budget}")
    model = env.current_model.to(device)
    mwr = ModelWithRows(model)
    groups = env._dependency_groups(mwr)
    plan = group_plan(mwr, groups)
    pm = ParamModel(model, plan) if budget == "params" else FlopModel(model, plan, env._input_shape(), device)
    state = state_dump.encode_origin(env, mwr, groups)
    rows = state_dump.token_rows(mwr, groups, plan)
    num_actions = len(env.conf.compression_rates_dict)
    extra = 0
    if blob.get("tokens", "layer") == "group":
        state = group_token_state(state, mwr.all_layers, groups, slot_dim=action_cost_slot_dim(num_actions))
        rows, _conflicts = state_dump.group_token_rows(state["token_members"], rows)
        extra = GROUP_TOKEN_EXTRA_DIM
    spans = spans_of(state_dump.layout(int(state["layer_features"].size(1)), num_actions, extra))
    state = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in state.items() if k != "token_members"}
    if kappa is None:
        kappa = env.target_keep if env.target_keep is not None else target
    state = plan_state(state, spans, float(kappa), tuple(blob.get("zero", ())), budget)
    mask, token_k = token_index(rows, pm.rows, device)
    with torch.no_grad():
        mu = policy(state, mask, token_k, len(pm.rows)).tolist()
        ens = None
        if others:
            each = [mu] + [p(state, mask, token_k, len(pm.rows)).tolist() for p in others]
            mu = [sum(v) / len(each) for v in zip(*each)]
            spread = sum(math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) for v, m in zip(zip(*each), mu)) / max(1, len(mu))
            ens = {"n": len(each), "spread": float(spread)}
            utils.print_flush(f"[alloc] plan agent ensemble of {len(each)} policies (mean score), member spread "
                              f"{spread:.3f}: " + ", ".join(_run_label(p) for p in paths))
    z, drawn = mu, None
    if sample is not None:
        sigma, seed = float(sample[0]), int(sample[1])
        eps = torch.randn(len(mu), generator=torch.Generator().manual_seed(seed), dtype=torch.float64).tolist()
        z = [m + sigma * e for m, e in zip(mu, eps)]
        drawn = {"around": "agent", "sigma": sigma, "seed": seed,
                 "dist": math.sqrt(sum((a - m) ** 2 for a, m in zip(z, mu)))}
    min_width = fortify.plan_min_width()
    if min_width <= 1:
        min_width = max(1, int(blob.get("min_width", 1)))
    extra = {}
    if residual == "off":
        widths, info = decode(z, pm, target, float(blob.get("k_min", k_min)), min_width=min_width)
    else:
        kind = prior_kind(residual, budget)
        costs = {} if kind == "sens_cost" or (kinds is not None and "sens_cost" in kinds) else None
        sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, env._input_shape(), costs=costs)
        prior = prior_weights(kind, pm.rows, sens, costs, budget, alpha)
        k_min_eff = float(blob.get("k_min", k_min))
        menu = None
        if kinds is None:
            widths, info = residual_decode(z, prior, pm, target, k_min_eff, min_width=min_width, zmax=zmax)
        else:  # v36: the rules on this net, their plans' costs for the menu head, the mean shares, one decode
            started, curves = time.perf_counter(), None
            if "sens_thresh" in kinds:  # the RNG is put back: the recovery downstream draws as after the prior alone
                from src import alloc_walk
                rng = alloc_walk.rng_snapshot()
                curves = dict(zip(("qs", "loss", "raw"),
                                  alloc_walk.curve_probes(model, plan, batches, env._input_shape())))
                alloc_walk.rng_restore(rng)
            rules = menu_rules(kinds, pm, target, sens, costs, curves, budget, alpha, k_min_eff, min_width)
            x_net, plans = menu_features(rules, pm, target, k_min_eff, min_width, kappa=float(kappa))
            with torch.no_grad():
                psi = policy.menu_logits(state, x_net).tolist()
            pi = menu_shares(psi, float(blob.get("menu_share_floor", MENU_SHARE_FLOOR)))
            widths, info = menu_decode(z, pi, rules, pm, target, k_min_eff, min_width=min_width, zmax=zmax)
            params0 = float(pm.params(pm.widths0))
            menu = {"kinds": list(kinds), "pi": pi, "psi": [float(v) for v in psi], "x": x_net,
                    "kept": [float(rule_info["kept"]) for _w, rule_info in plans],
                    "params": [pm.params(rule_widths) / params0 for rule_widths, _i in plans],
                    "seconds": time.perf_counter() - started}
        base, base_info = scale_decode(prior, pm, target, k_min_eff, min_width=min_width)
        delta = {row: (widths[row] - base[row]) / float(pm.widths0[row]) for row in pm.rows}
        moved = sum(widths[row] != base[row] for row in pm.rows)
        extra = {"residual": residual,
                 "prior": {"kind": kind, "alpha": float(alpha), "kept": float(base_info["kept"]), "widths": base,
                           "weights": prior if menu is None else info["base"], "moved": int(moved),
                           "max_abs": max(abs(d) for d in delta.values()),
                           "dist": math.sqrt(sum(d * d for d in delta.values())),
                           **({"zmax": float(zmax)} if zmax > 0 else {})},
                 **({"menu": menu} if menu is not None else {})}
        utils.print_flush(
            f"[alloc] plan agent residual on {kind} alpha={alpha:g}" + (f" zmax {zmax:g}" if zmax > 0 else "")
            + f": prior keeps x{base_info['kept']:.3f}, agent "
            f"x{info['kept']:.3f}; {moved}/{len(pm.rows)} groups moved, max |Δkeep| {extra['prior']['max_abs']:.2f}, "
            f"dist {extra['prior']['dist']:.3f}"
            + ("" if menu is None else "; menu pi " + " ".join(f"{k} {p:.3f}" for k, p in zip(kinds, menu["pi"]))))
    return widths, {"kind": "agent" if drawn is None else "agent_sample", "alpha": 0.0, "target": float(target),
                    "kept": float(info["kept"]), "budget": budget,
                    "keeps": {row: widths[row] / float(pm.widths0[row]) for row in pm.rows},
                    "origin_widths": dict(pm.widths0), "sens": {row: float(v) for row, v in zip(pm.rows, z)},
                    "held": 0, "policy": _policy_label(paths[0]) if len(paths) == 1
                    else "+".join(_run_label(p) for p in paths),
                    **({"sample": drawn, "mu": {row: float(m) for row, m in zip(pm.rows, mu)}} if drawn else {}),
                    **({"min_width": int(min_width)} if min_width > 1 else {}),
                    **({"ensemble": ens} if ens else {}), **extra}


ENSEMBLE_KEYS = ("feature_dim", "encoder", "tokens", "zero", "k_min", "min_width", "budget", "standardizer",
                 "residual", "residual_alpha", "residual_zmax", "menu")  # menu (v36): absent = None on both sides


def ensemble_paths(policy_path: str):
    """The checkpoints ``policy_path`` names: one path, or several comma-separated (v22, ENS)."""
    paths = [p.strip() for p in str(policy_path).split(",") if p.strip()]
    if not paths:
        raise ValueError("plan_for_env: no plan-agent checkpoint given")
    return paths


def _policy_label(path: str) -> str:
    """``<dir>/<file>`` of a checkpoint, as ``info["policy"]`` has always named one."""
    return os.path.basename(os.path.dirname(os.path.abspath(path))) + "/" + os.path.basename(path)


def _run_label(path: str) -> str:
    """``<run>/<dir>/<file>``: tells an ensemble's members apart (each seed's train writes plan_agent/...)."""
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path)))) + "/" + _policy_label(path)


def eval_residual(blob):
    """``(mode, alpha, zmax)`` a frozen agent decodes with: the checkpoint's ``residual`` / ``residual_alpha`` /
    ``residual_zmax`` (off / 0.5 / 0 when absent: the pre-T2 and the unbounded checkpoints), each overridden by
    ``SPECTRA_PLAN_RESIDUAL`` / ``SPECTRA_PLAN_RESIDUAL_ALPHA`` / ``SPECTRA_PLAN_RESIDUAL_ZMAX`` when that is set, with
    a WARNING line when it differs from what the agent trained with. Off reads neither: (off, None, 0)."""
    from src import fortify
    import src.utils as utils
    mode = str(blob.get("residual", "off"))
    flag = fortify.plan_residual(default=None)
    if flag is not None and flag != mode:
        utils.print_flush(f"[alloc] WARNING: the plan agent trained with residual={mode}; "
                          f"SPECTRA_PLAN_RESIDUAL={flag} overrides it")
        mode = flag
    if mode == "off":
        return mode, None, 0.0
    if mode != "auto":
        prior_kind(mode)
    alpha = float(blob.get("residual_alpha", 0.5))
    flag_alpha = fortify.plan_residual_alpha(default=None)
    if flag_alpha is not None and flag_alpha != alpha:
        utils.print_flush(f"[alloc] WARNING: the plan agent trained with residual_alpha={alpha:g}; "
                          f"SPECTRA_PLAN_RESIDUAL_ALPHA={flag_alpha:g} overrides it")
        alpha = flag_alpha
    zmax = float(blob.get("residual_zmax", 0.0))
    flag_zmax = fortify.plan_residual_zmax(default=None)
    if flag_zmax is not None and flag_zmax != zmax:
        utils.print_flush(f"[alloc] WARNING: the plan agent trained with residual_zmax={zmax:g}; "
                          f"SPECTRA_PLAN_RESIDUAL_ZMAX={flag_zmax:g} overrides it")
        zmax = flag_zmax
    return mode, alpha, zmax
