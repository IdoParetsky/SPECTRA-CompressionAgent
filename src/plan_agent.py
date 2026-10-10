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
* ``MaskedCut``: ``cut_to``'s L1 cut written as zeros on a working copy, for the reward.
* ``PlanPolicy``: the agent's state encoder with a per-token linear head. A group's score is the mean
  over its tokens, centred over groups (b absorbs any common shift), and plans are z = μ + σ ε.
* ``NetInstance``: one catalog net's origin read once (state, plan, token map, param model, val batches,
  sensitivities), so an instance (net, κ) costs only its plans.
* ``plan_for_env``: a frozen agent's mean plan in ``alloc_walk.plan_targets``' format
  (``SPECTRA_ALLOC_KIND=agent``) or, with ``sample``, one Gaussian draw around it
  (``SPECTRA_ALLOC_KIND=agent_sample``), so the existing walk, final fine-tune and TEST read it.
"""

import copy
import math
import os
import time

import torch
from torch import nn

TOKEN_KINDS = ("layer", "group")
PROXIES = ("cut", "bn")
BUDGETS = ("params", "flops")


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


def _complement(kept, total):
    mask = torch.ones(int(total), dtype=torch.bool)
    mask[kept.detach().cpu().long()] = False
    return torch.nonzero(mask, as_tuple=False).flatten()


@torch.no_grad()
def _zero_group(group, keep_idx):
    """Zero what ``pruning.prune_group_structurally`` would remove for survivors ``keep_idx``."""
    from src import pruning
    width = int(group.width)
    keep = keep_idx.detach().cpu().long()
    removed = _complement(keep, width)
    for module in list(group.producers) + list(group.depthwise):
        idx = removed.to(module.weight.device)
        module.weight[idx] = 0
        if module.bias is not None:
            module.bias[idx] = 0
    for ref in group.consumers:
        module = ref.module
        kept = pruning.surviving_input_channels(ref, width, keep)
        if isinstance(module, nn.Conv2d):
            total = module.in_channels
        else:
            total = module.in_features
            if module.in_features != ref.total:
                kept = pruning._expand_indices_for_flatten(kept, ref.total, module.in_features)
        module.weight[:, _complement(kept, total).to(module.weight.device)] = 0
    for ref in group.norms:
        norm = ref.module
        idx = _complement(pruning.surviving_input_channels(ref, width, keep), norm.num_features)
        if norm.weight is not None:
            norm.weight[idx.to(norm.weight.device)] = 0
            norm.bias[idx.to(norm.bias.device)] = 0


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

    @property
    def rows(self):
        return [int(row) for _group, row in self.plan]

    def cut(self, rates):
        from src import pruning
        self.work.load_state_dict(self.origin)
        for group, row in self.plan:
            rate = float(rates.get(row, 1.0))
            if rate >= 1.0 - 1e-9:
                continue
            keep = pruning.select_group_survivors(group, rate, mode="l1")
            if keep is None:
                raise RuntimeError(f"group at row {row}: no L1 survivors")
            _zero_group(group, keep)
        return self.work


def _no_dropout(module: nn.Module):
    for sub in module.modules():
        if isinstance(sub, nn.Dropout):
            sub.p = 0.0
        elif isinstance(sub, nn.MultiheadAttention):
            sub.dropout = 0.0


class PlanPolicy(nn.Module):
    """State encoder + per-token linear head → one centred score per planned group (all 0 at init)."""

    def __init__(self, feature_dim: int, encoder: str = "transformer"):
        super().__init__()
        from src.Model.StateEncoder import build_state_encoder
        self.feature_dim, self.encoder_kind = int(feature_dim), encoder
        self.encoder = build_state_encoder(encoder, self.feature_dim)
        self.encoder.pool = lambda encoded, target_index, n: encoded[0, :n]
        _no_dropout(self.encoder)
        self.head = nn.Linear(self.encoder.output_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, state, token_mask, token_k, n_groups):
        scores = self.head(self.encoder(state)).squeeze(-1)[token_mask]
        sums = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype).index_add(0, token_k, scores)
        count = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype).index_add(
            0, token_k, torch.ones_like(scores))
        mu = sums / count.clamp(min=1.0)
        return mu - mu.mean()


def sample_scores(mu, sigma: float, k: int, generator=None):
    eps = torch.randn((k, mu.numel()), generator=generator, device=mu.device, dtype=mu.dtype)
    return mu.detach().unsqueeze(0) + float(sigma) * eps


def log_prob(z, mu, sigma: float):
    """Gaussian log-density of each row of ``z`` around ``mu`` (constants dropped)."""
    return (-(z - mu.unsqueeze(0)) ** 2 / (2.0 * float(sigma) ** 2)).sum(dim=-1)


def save_policy(policy: PlanPolicy, path: str, **meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "feature_dim": policy.feature_dim,
                "encoder": policy.encoder_kind, **meta}, path)


def load_policy(path: str, device="cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    policy = PlanPolicy(blob["feature_dim"], blob.get("encoder", "transformer"))
    policy.load_state_dict(blob["state_dict"])
    return policy.to(device).eval(), blob


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


def parse_proxy(name: str):
    """``("cut", 0)`` or ``("bn", N)``."""
    name = (name or "").strip().lower()
    if name == "cut":
        return "cut", 0
    if name.startswith("bn") and name[2:].isdigit() and int(name[2:]) > 0:
        return "bn", int(name[2:])
    raise ValueError(f"plan reward proxy {name!r}: expected cut or bn<N>")


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
                 zero=(), sens=None, input_shape=None):
        started = time.perf_counter()
        self.name, self.device, self.zero, self.input_shape = str(name), device, tuple(zero), input_shape
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
        self.held = {int(row) for group, row in plan if len(group.producers) > 1}
        self.seconds = time.perf_counter() - started

    @classmethod
    def from_env(cls, env, net_path, net_model, net_loaders, kappa0=0.6, tokens="layer", zero=(), with_sens=True):
        """Reset ``env`` on the net and read its origin as the D-IMIT dump does."""
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
        sens = None
        if with_sens:
            batches = group_sensitivity.calibration_batches(env.train_loader, group_sensitivity.CALIB_BATCHES, device)
            sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, env._input_shape())
        return cls(os.path.basename(str(net_path)), model, plan, state, rows, layout,
                   env.train_loader, env.val_loader, device, zero=zero, sens=sens, input_shape=env._input_shape())

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
        kind, n = parse_proxy(proxy)
        return recal_batches(self.train_loader, n, seed, self.device) if kind == "bn" else []

    def reward(self, widths, proxy: str, batches) -> float:
        """Δ val accuracy (pp) of one cut to ``widths``, after ``proxy`` (BatchNorm re-estimated on ``batches``)."""
        from src import recovery_edits
        kind, _n = parse_proxy(proxy)
        work = self.cutter.cut(rates_of(widths, self.pm.widths0))
        if kind == "bn":
            recovery_edits.recalibrate_batchnorm(work, batches, self.device, len(batches))
        return 100.0 * (accuracy(work, self.val) - self.origin_val)

    def reference(self, kind: str, kappa: float, k_min: float = 0.1, a: float = 0.5, budget: str = "params",
                  min_width: int = 1):
        """``(widths, info)`` of the uniform / sens / inner plan at ``kappa`` on the ``budget`` cost model; None for
        sens without sensitivities. ``min_width`` as ``decode``."""
        from src import alloc_walk
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
    the checkpoint, written by the trainer only when it was above 1), else none."""
    from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
    from src import fortify, state_dump
    from src.BERTInputModeler import action_cost_slot_dim
    from src.group_sensitivity import group_plan
    from src.group_tokens import GROUP_TOKEN_EXTRA_DIM, group_token_state
    import src.utils as utils
    from src.feature_standardizer import resolve_standardizer_path
    if budget not in BUDGETS:
        raise ValueError(f"budget={budget!r}; expected one of {BUDGETS}")
    device = env.conf.device
    policy, blob = load_policy(policy_path, device)
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
    widths, info = decode(z, pm, target, float(blob.get("k_min", k_min)), min_width=min_width)
    return widths, {"kind": "agent" if drawn is None else "agent_sample", "alpha": 0.0, "target": float(target),
                    "kept": float(info["kept"]), "budget": budget,
                    "keeps": {row: widths[row] / float(pm.widths0[row]) for row in pm.rows},
                    "origin_widths": dict(pm.widths0), "sens": {row: float(v) for row, v in zip(pm.rows, z)},
                    "held": 0, "policy": os.path.basename(os.path.dirname(os.path.abspath(policy_path)))
                    + "/" + os.path.basename(policy_path),
                    **({"sample": drawn, "mu": {row: float(m) for row, m in zip(pm.rows, mu)}} if drawn else {}),
                    **({"min_width": int(min_width)} if min_width > 1 else {})}
