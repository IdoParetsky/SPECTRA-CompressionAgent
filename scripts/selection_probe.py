"""
Selection headroom probe (S0 in docs/paper/FILTER_SELECTION_NAP_DESIGN.md): does *which*
filters a cut keeps matter once *how many* each group keeps is fixed?

Every coupled channel group of a pretrained network is cut once, to the same keep-rate, by the
walk's own structural edit (``prune_current_model``); only the ranking that picks the survivors
changes, so every criterion yields a network of identical shape. Each cut network is then
fine-tuned with the walk's recipe A at several epoch budgets.

Criteria, all scored once on the unpruned network:
  l1 l2 svd fpgm bn_scale  SPECTRA's weight rankers (Li et al. 2017; He et al. 2019; Liu et al. 2017)
  taylor                   first-order Taylor |sum w * dL/dw| (Molchanov et al. 2017 / 2019)
  act, apoz                mean post-ReLU activation; 1 - APoZ (Hu et al. 2016)
  hrank                    mean feature-map rank (HRank, Lin et al. CVPR 2020)
  ablation                 calibration-loss increase when the channel alone is removed
                           (the single-channel oracle of Molchanov et al. 2017)
  anti_l1                  keep the smallest-L1 channels: the worst named ranking
  random                   uniform masks, --random_masks of them

L1 is fine-tuned under --l1_seeds seeds, so the noise a criterion has to clear is measured.
Budget ``bn`` re-estimates BatchNorm statistics without training (EagleEye's adaptive-BN proxy).

    python scripts/selection_probe.py --checkpoint NET.pth --script resnet_chenyaofo.py \
        --arch resnet56 --dataset cifar-10 --keep 0.8 0.6 --budgets 0 bn 1 3 10 40

Writes to $SPECTRA_RUN_DIR/results/: selection_probe.jsonl (one row per mask and budget, then a
summary row), selection_masks.jsonl (surviving channels per group), selection_features.npz
(per-channel criterion scores plus NAPv2's statistics of each filter's weights and gradient, for
the zero-GPU analysis) and, with --nap_dir, NAPv2 feature-map sequences of the fine-tunes.
Accuracies are on protocol P's val/test halves. They measure a lever; they are never a method's
TEST row.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.channel_groups as channel_groups
import src.logging_utils as logging_utils
import src.pruning as pruning
import src.recovery_edits as recovery_edits
import src.utils as utils
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows
from src.Configuration.ConfigurationValues import ConfigurationValues
from src.Configuration.StaticConf import StaticConf
from src.ModelHandlers.ClassificationHandler import ClassificationHandler
from src.NetworkEnv import prune_current_model

WEIGHT_CRITERIA = ("l1", "l2", "svd", "fpgm", "bn_scale")
DATA_CRITERIA = ("taylor", "act", "apoz", "hrank", "ablation")
CRITERIA = WEIGHT_CRITERIA + DATA_CRITERIA + ("anti_l1", "random")
# nap2/stats.py's twelve statistics, global form; its percentile stat is five values (0/25/75/50/100).
NAP_STATS = ("mean", "variance", "median", "std", "max", "min", "covariance", "skewness",
             "kurtosis", "q0", "q25", "q75", "q50", "q100", "L1", "L2")


def _ensure_conf(device, num_epochs: int):
    if StaticConf.get_instance() is not None:
        return
    StaticConf(ConfigurationValues(
        device=device, test_name="selection-probe", input_dict={},
        compression_rates_dict={0: 1.0}, runtime_limit=1, num_epochs=num_epochs,
        train_compressed_layer_only=False,
        allowed_acc_reduction=5, discount_factor=0.99, learning_rate=1e-3,
        rollout_limit=None, passes=1, prune=True, seed=0, n_splits=0,
        train_split=0.7, val_split=0.2, database_dict={},
        actor_checkpoint_path=None, critic_checkpoint_path=None,
        save_pruned_checkpoints=False, test_ts="probe",
    ))


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ groups and the cut


def _module_names(root):
    return {id(m): name[len("module."):] if name.startswith("module.") else name
            for name, m in root.named_modules()}


def group_key(root, group, names=None):
    """A group's identity across structural edits: the sorted names of the modules that own it."""
    names = names if names is not None else _module_names(root)
    return tuple(sorted(names.get(id(m), "?") for m in list(group.producers) + list(group.depthwise)))


def cut_plan(model):
    """(key, group, row) for every group the walk can cut, at its first row, in walk order.

    The walk never acts on the last row (the classifier), so neither does the probe.
    """
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    names = _module_names(mwr.model)
    plan, seen = [], set()
    for row in sorted(mwr.row_to_main_layer)[:-1]:
        layer = mwr.all_layers[mwr.row_to_main_layer[row]]
        group = channel_groups.group_of(groups, layer)
        if group is None or not group.prunable or group.width < 2:
            continue
        key = group_key(mwr.model, group, names)
        if key not in seen:
            seen.add(key)
            plan.append((key, group, row))
    return plan


def rank_table(raw, alive, largest_first=True):
    """Positive ranks on alive channels (higher = keep first) and 0 on dead ones.

    ``select_group_survivors`` keeps the top ``target_width`` channels among those with positive
    importance, so ranks make it keep exactly the top of ``raw`` among the channels the walk
    counts as alive, whatever the sign or scale of the raw score.
    """
    order = raw.double() if largest_first else -raw.double()
    ranks = torch.empty_like(order)
    ranks[torch.argsort(order, stable=True)] = torch.arange(1, order.numel() + 1, dtype=torch.float64)
    ranks[~alive] = 0.0
    return ranks


class RankingOverride:
    """Points SPECTRA's group ranking at precomputed tables while a cut runs.

    Groups without a table (or of another width) fall back to the configured ranking.
    """

    def __init__(self):
        self.tables, self.mwr, self.kept = None, None, {}
        self._importance = pruning.group_importance
        self._survivors = pruning.select_group_survivors

    def __enter__(self):
        pruning.group_importance = self.importance
        pruning.select_group_survivors = self.survivors
        return self

    def __exit__(self, *exc):
        pruning.group_importance = self._importance
        pruning.select_group_survivors = self._survivors
        return False

    def _table(self, group):
        if self.tables is None or self.mwr is None:
            return None
        table = self.tables.get(group_key(self.mwr.model, group))
        return table if table is not None and table.numel() == group.width else None

    def importance(self, group, mode=None):
        table = self._table(group)
        if table is None:
            return self._importance(group, mode)
        owner = next(iter(list(group.producers) + list(group.depthwise)))
        return table.to(device=owner.weight.device, dtype=torch.float32)

    def survivors(self, group, compression_rate, mode=None):
        keep = self._survivors(group, compression_rate, mode)
        if keep is not None and self.tables is not None:
            self.kept[group_key(self.mwr.model, group)] = [int(i) for i in keep.tolist()]
        return keep


def cut(model, plan, keep, tables, override, input_shape):
    """One structural cut per group at ``keep``, survivors chosen by ``tables``."""
    mwr = ModelWithRows(copy.deepcopy(model))
    override.tables, override.kept = tables, {}
    modes = []
    try:
        for _, _, row in plan:
            override.mwr = mwr
            mwr = prune_current_model(mwr, keep, row, quiet=True, record=False, input_shape=input_shape)
            modes.append((getattr(mwr, "last_prune_outcome", None) or {}).get("mode"))
            mwr = ModelWithRows(mwr.model)
    finally:
        override.tables, override.mwr = None, None
    return mwr.model, modes, dict(override.kept)


def shape_signature(model):
    return [int(m.out_channels) if isinstance(m, nn.Conv2d) else int(m.out_features)
            for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


# ------------------------------------------------------------------ criteria


def calibration_batches(loader, n_batches: int, device, seed: int = 0):
    _seed_everything(seed)
    batches = []
    for x, y in loader:
        if y.dim() > 1 and y.shape[1] > 1:
            y = y.argmax(dim=1)
        batches.append((x.to(device), y.long().to(device)))
        if len(batches) >= n_batches:
            break
    return batches


def _votes(parts):
    """SPECTRA's group vote (``pruning.group_importance``): each part over its own max, summed."""
    return torch.stack([p / p.abs().max().clamp(min=1e-12) for p in parts]).sum(dim=0)


def read_points(group):
    """Where each group channel's output can be read and zeroed: (module, {channel: index}).

    Norm outputs when the group has norms (a channel is gone once its normalised output is
    zero), producer outputs otherwise. Points that do not cover every channel are skipped.
    """
    full = list(range(group.width))
    points = []
    for ref in group.norms:
        pidx = ref.producer_idx or tuple(range(len(ref.positions)))
        points.append((ref.module, {int(j): int(p) for j, p in zip(pidx, ref.positions)}))
    points = [(m, cols) for m, cols in points if sorted(cols) == full]
    if not points:
        points = [(m, {j: j for j in full}) for m in group.producers
                  if getattr(m, "weight", None) is not None and m.weight.shape[0] == group.width]
    return points


def weight_scores(model, plan, batches):
    """SPECTRA's weight rankers and first-order Taylor, through ``pruning.group_importance``."""
    pruning.bind_bn_scales(model)
    pruning.bind_taylor_scores(model, batches, n_batches=len(batches))
    scores = {}
    for mode in WEIGHT_CRITERIA + ("taylor",):
        table = {}
        for key, group, _ in plan:
            importance = pruning.group_importance(group, mode)
            if importance is not None:
                table[key] = importance.detach().double().cpu()
        scores[mode] = table
    return scores


@torch.no_grad()
def activation_scores(model, plan, batches, rank_batches: int = 1):
    """Mean post-ReLU activation, 1 - APoZ and mean feature-map rank (HRank) per channel."""
    model.eval()
    sums, handles, state = {}, [], {"batch": 0}

    def hook_for(slot, cols):
        index = torch.tensor([cols[j] for j in range(len(cols))])

        def hook(module, inputs, output):
            a = torch.relu(output.index_select(1, index.to(output.device))).float()
            dims = [d for d in range(a.dim()) if d != 1]
            entry = sums.setdefault(slot, {"act": 0.0, "nz": 0.0, "rank": 0.0, "n": 0, "nr": 0})
            entry["act"] = entry["act"] + a.mean(dim=dims).double().cpu()
            entry["nz"] = entry["nz"] + (a > 0).float().mean(dim=dims).double().cpu()
            entry["n"] += 1
            if a.dim() == 4 and state["batch"] < rank_batches:
                n, c, h, w = a.shape
                ranks = torch.linalg.matrix_rank(a.transpose(0, 1).reshape(c * n, h, w))
                entry["rank"] = entry["rank"] + ranks.reshape(c, n).float().mean(dim=1).double().cpu()
                entry["nr"] += 1
        return hook

    for key, group, _ in plan:
        for i, (module, cols) in enumerate(read_points(group)):
            handles.append(module.register_forward_hook(hook_for((key, i), cols)))
    try:
        for x, _ in batches:
            model(x)
            state["batch"] += 1
    finally:
        for handle in handles:
            handle.remove()

    act, apoz, hrank = {}, {}, {}
    for key, _, _ in plan:
        entries = [entry for (k, _), entry in sums.items() if k == key]
        if not entries:
            continue
        act[key] = _votes([e["act"] / e["n"] for e in entries])
        apoz[key] = _votes([e["nz"] / e["n"] for e in entries])
        ranked = [e["rank"] / e["nr"] for e in entries if e["nr"]]
        hrank[key] = _votes(ranked) if ranked else act[key].clone()  # 2-D outputs have no map rank
    return {"act": act, "apoz": apoz, "hrank": hrank}


@torch.no_grad()
def ablation_scores(model, plan, batches):
    """Calibration-loss increase when one channel is removed and nothing else changes."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    count = sum(int(x.shape[0]) for x, _ in batches)

    def mean_loss():
        return sum(float(loss_fn(model(x), y)) for x, y in batches) / count

    base = mean_loss()
    target = {"channel": None}

    def hook_for(cols):
        def hook(module, inputs, output):
            if target["channel"] is None:
                return None
            output = output.clone()
            output[:, cols[target["channel"]]] = 0
            return output
        return hook

    scores = {}
    for key, group, _ in plan:
        points = read_points(group)
        if not points:
            continue
        handles = [module.register_forward_hook(hook_for(cols)) for module, cols in points]
        damage = torch.zeros(group.width, dtype=torch.float64)
        try:
            for j in range(group.width):
                target["channel"] = j
                damage[j] = mean_loss() - base
        finally:
            target["channel"] = None
            for handle in handles:
                handle.remove()
        scores[key] = damage
    return scores, base


def nap_filter_stats(weight):
    """NAPv2's statistics (``nap2/stats.py``, global form) of every output filter: [filters, 16].

    Constant filters get zero skewness and kurtosis where scipy would return nan.
    """
    w = weight.detach().double().reshape(weight.shape[0], -1)
    n = w.shape[1]
    mean = w.mean(dim=1)
    centred = w - mean[:, None]
    m2 = centred.pow(2).mean(dim=1)
    flat = m2 <= 0
    safe = torch.where(flat, torch.ones_like(m2), m2)
    skew = torch.where(flat, torch.zeros_like(m2), centred.pow(3).mean(dim=1) / safe.pow(1.5))
    kurt = torch.where(flat, torch.zeros_like(m2), centred.pow(4).mean(dim=1) / safe.pow(2) - 3.0)
    levels = [0.0, 0.25, 0.75, 0.5, 1.0]
    if w.numel() <= 2 ** 24:
        q = torch.quantile(w, torch.tensor(levels, dtype=w.dtype, device=w.device), dim=1)
    else:  # torch.quantile's input cap
        q = torch.from_numpy(np.percentile(w.cpu().numpy(), [100 * v for v in levels], axis=1)).to(w)
    cov = centred.pow(2).sum(dim=1) / max(n - 1, 1)
    return torch.stack([mean, m2, q[3], m2.sqrt(), w.max(dim=1).values, w.min(dim=1).values, cov,
                        skew, kurt, q[0], q[1], q[2], q[3], q[4], w.abs().sum(dim=1),
                        w.norm(dim=1)], dim=1).cpu()


def mean_gradients(model, plan, batches):
    """Mean calibration-loss gradient of every producer weight (eval-mode BN)."""
    model.eval()
    producers = {id(p): p for _, group, _ in plan for p in group.producers
                 if getattr(p, "weight", None) is not None}
    for p in producers.values():
        p.weight.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    loss_fn = nn.CrossEntropyLoss()
    for x, y in batches:
        loss_fn(model(x), y).backward()
    grads = {i: p.weight.grad.detach().clone() / len(batches)
             for i, p in producers.items() if p.weight.grad is not None}
    model.zero_grad(set_to_none=True)
    return grads


def consumer_l1(group):
    """Per channel, the L1 of every consumer's input slice, each consumer over its own max, summed."""
    parts = []
    for ref in group.consumers:
        weight = getattr(ref.module, "weight", None)
        if weight is None or weight.dim() < 2 or getattr(ref.module, "groups", 1) != 1:
            continue
        per_input = weight.detach().abs().transpose(0, 1).reshape(weight.shape[1], -1).sum(dim=1)
        per_input = per_input.double().cpu()
        total = torch.zeros(group.width, dtype=torch.float64)
        pidx = ref.producer_idx or tuple(range(len(ref.positions)))
        for j, pos in zip(pidx, ref.positions):
            if 0 <= int(pos) < per_input.numel() and 0 <= int(j) < group.width:
                total[int(j)] += per_input[int(pos)]
        parts.append(total)
    return _votes(parts) if parts else torch.zeros(group.width, dtype=torch.float64)


def channel_features(plan, grads, scores):
    """One row per (group, channel): position, every criterion score, consumer L1 and NAPv2's
    statistics of the producers' filters and of their gradient (mean over producers)."""
    columns = (["group", "channel", "depth", "width", "n_producers"]
               + [c for c in WEIGHT_CRITERIA + DATA_CRITERIA] + ["out_l1"]
               + [f"w_{s}" for s in NAP_STATS] + [f"g_{s}" for s in NAP_STATS])
    blocks = []
    for gi, (key, group, _) in enumerate(plan):
        width = group.width
        owners = [p for p in group.producers
                  if getattr(p, "weight", None) is not None and p.weight.shape[0] == width]
        w_stats = (torch.stack([nap_filter_stats(p.weight) for p in owners]).mean(dim=0)
                   if owners else torch.full((width, len(NAP_STATS)), float("nan"), dtype=torch.float64))
        g_owners = [grads[id(p)] for p in owners if id(p) in grads]
        g_stats = (torch.stack([nap_filter_stats(g) for g in g_owners]).mean(dim=0)
                   if g_owners else torch.full((width, len(NAP_STATS)), float("nan"), dtype=torch.float64))
        crit = [scores.get(c, {}).get(key, torch.full((width,), float("nan"), dtype=torch.float64))
                for c in WEIGHT_CRITERIA + DATA_CRITERIA]
        head = torch.tensor([[gi, j, gi / max(len(plan) - 1, 1), width, len(group.producers)]
                             for j in range(width)], dtype=torch.float64)
        blocks.append(torch.cat([head, torch.stack(crit, dim=1).double(),
                                 consumer_l1(group)[:, None], w_stats, g_stats], dim=1))
    return columns, torch.cat(blocks).numpy()


def all_scores(model, plan, batches, rank_batches: int = 1):
    scores = weight_scores(model, plan, batches)
    scores.update(activation_scores(model, plan, batches, rank_batches))
    scores["ablation"], base_loss = ablation_scores(model, plan, batches)
    return scores, base_loss


def mask_specs(criteria, random_masks: int, l1_seeds: int):
    """(criterion, mask seed, fine-tune seed) for every mask the probe trains."""
    specs = []
    for crit in criteria:
        if crit == "random":
            specs += [("random", s, s) for s in range(random_masks)]
        elif crit == "l1":
            specs += [("l1", 0, s) for s in range(l1_seeds)]
        else:
            specs.append((crit, 0, 0))
    return specs


def score_tables(spec, scores, alive, plan):
    """Rank tables for one mask; groups a criterion could not score fall back to L1."""
    crit, mask_seed, _ = spec
    generator = torch.Generator().manual_seed(1000 + int(mask_seed))
    tables, fallbacks = {}, 0
    for key, group, _ in plan:
        live = alive[key]
        if crit == "random":
            tables[key] = rank_table(torch.rand(group.width, generator=generator, dtype=torch.float64), live)
        elif crit == "anti_l1":
            tables[key] = rank_table(scores["l1"][key], live, largest_first=False)
        elif key in scores.get(crit, {}):
            tables[key] = rank_table(scores[crit][key], live)
        else:
            tables[key] = rank_table(scores["l1"][key], live)
            fallbacks += 1
    return tables, fallbacks


# ------------------------------------------------------------------ recovery


class _Snapshotting:
    """Train loader that lets NAPv2's SnapshotCollector see every optimiser step."""

    def __init__(self, loader, collector):
        self.loader, self.collector = loader, collector

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield batch
            self.collector.step()


def recover(cut_model, budget, loaders, device, seed: int, patience=None, bn_batches: int = 50,
            collector_factory=None):
    """Recipe A at one budget (``bn``: BatchNorm re-estimation only). Returns val, test, collector."""
    train_loader, val_loader, test_loader = loaders
    model = copy.deepcopy(cut_model).to(device)
    _seed_everything(seed)
    handler = ClassificationHandler(model, nn.CrossEntropyLoss())
    handler.unfreeze_all_layers()
    collector = None
    if budget == "bn":
        recovery_edits.recalibrate_batchnorm(model, train_loader, device, n_batches=bn_batches)
    elif int(budget) > 0:
        loader = train_loader
        if collector_factory is not None:
            collector = collector_factory(model, int(budget))
            if collector is not None:
                loader = _Snapshotting(train_loader, collector)
        handler.train_model(loader, max_epochs=int(budget), patience=patience)
    return float(handler.evaluate_model(val_loader)), float(handler.evaluate_model(test_loader)), collector


def load_napv2(path):
    sys.path.insert(0, str(path))
    from nap2.feature_maps import create_feature_map_sequence
    from nap2.snapshot_collector import SnapshotCollector
    from nap2.stats import extract_all_stats
    return SimpleNamespace(SnapshotCollector=SnapshotCollector, extract_all_stats=extract_all_stats,
                           create_feature_map_sequence=create_feature_map_sequence)


def nap_feature_maps(nap, collector):
    """NAPv2 [snapshots, 65, 100, 12] maps of a fine-tune's weights and gradients (raw, not log-normalised)."""
    def maps(snapshots):
        kept = {step: {name: t for name, t in layers.items() if t is not None}
                for step, layers in snapshots.items()}
        if not kept:
            return np.zeros((0, 65, 100, 12), dtype=np.float32)
        return nap.create_feature_map_sequence(nap.extract_all_stats(kept)).astype(np.float32)
    return maps(collector.get_weight_snapshots()), maps(collector.get_gradient_snapshots())


# ------------------------------------------------------------------ readout


def kendall_tau(x, y):
    """Kendall's tau-b (ties allowed)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    iu = np.triu_indices(len(x), 1)
    dx = np.sign(x[:, None] - x[None, :])[iu]
    dy = np.sign(y[:, None] - y[None, :])[iu]
    denom = np.sqrt(float((dx != 0).sum()) * float((dy != 0).sum()))
    return float((dx * dy).sum() / denom) if denom > 0 else float("nan")


def agreement(scores, alive, plan, reference: str):
    """Width-weighted mean within-group Kendall tau of every criterion against ``reference``."""
    out = {}
    for crit, table in scores.items():
        if crit == reference:
            continue
        num = den = 0.0
        for key, _, _ in plan:
            if key not in table or key not in scores.get(reference, {}):
                continue
            live = alive[key].numpy()
            if live.sum() < 2:
                continue
            tau = kendall_tau(table[key].numpy()[live], scores[reference][key].numpy()[live])
            if np.isfinite(tau):
                num += tau * live.sum()
                den += live.sum()
        out[crit] = round(num / den, 4) if den else None
    return out


def jaccard(kept_a, kept_b):
    """Width-weighted mean Jaccard overlap of two masks' surviving channels."""
    num = den = 0.0
    for key, a in kept_a.items():
        if key not in kept_b:
            continue
        a, b = set(a), set(kept_b[key])
        union = len(a | b)
        if union:
            num += len(a & b)
            den += union
    return round(num / den, 4) if den else None


def _spread(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": round(float(arr.mean()), 3),
            "sd": round(float(arr.std(ddof=1)), 3) if len(values) > 1 else None,
            "min": round(float(arr.min()), 3), "max": round(float(arr.max()), 3)}


def summarize(rows, keeps, budgets):
    """Per keep and budget: every criterion's val/test change, and the lever numbers."""
    summary = {}
    for keep in keeps:
        for budget in budgets:
            cell = [r for r in rows if r["keep"] == keep and r["budget"] == str(budget)]
            if not cell:
                continue
            by_crit = {}
            for r in cell:
                by_crit.setdefault(r["criterion"], []).append(r)
            stats = {crit: {"val": _spread([r["d_val_pp"] for r in rs]),
                            "test": _spread([r["d_test_pp"] for r in rs])}
                     for crit, rs in by_crit.items()}
            l1 = stats.get("l1", {}).get("val")
            lever = {}
            if l1:
                named = {c: s["val"]["mean"] for c, s in stats.items() if c not in ("l1", "random", "anti_l1")}
                if named:
                    best = max(named, key=named.get)
                    lever["best_named"] = best
                    lever["best_minus_l1_pp"] = round(named[best] - l1["mean"], 3)
                if "ablation" in stats:
                    lever["ablation_minus_l1_pp"] = round(stats["ablation"]["val"]["mean"] - l1["mean"], 3)
                if "random" in stats:
                    lever["l1_minus_random_pp"] = round(l1["mean"] - stats["random"]["val"]["mean"], 3)
                    lever["random_sd_pp"] = stats["random"]["val"]["sd"]
                if "anti_l1" in stats:
                    lever["l1_minus_anti_l1_pp"] = round(l1["mean"] - stats["anti_l1"]["val"]["mean"], 3)
                lever["l1_ft_seed_sd_pp"] = l1["sd"]
            summary[f"keep={keep} budget={budget}"] = {"criteria": stats, "lever": lever}
    return summary


def _print_tables(rows, keeps, budgets, field):
    crits = []
    for r in rows:
        if r["criterion"] not in crits:
            crits.append(r["criterion"])
    for keep in keeps:
        utils.print_flush(f"-- keep={keep}: {field} change (pp), mean over masks/seeds --")
        utils.print_flush(f"{'criterion':>10} " + " ".join(f"{str(b):>8}" for b in budgets))
        for crit in crits:
            cells = []
            for budget in budgets:
                vals = [r[field] for r in rows if r["keep"] == keep and r["criterion"] == crit
                        and r["budget"] == str(budget)]
                cells.append(f"{np.mean(vals):>+8.2f}" if vals else f"{'-':>8}")
            utils.print_flush(f"{crit:>10} " + " ".join(cells))


# ------------------------------------------------------------------ main


def _budget(token: str):
    return "bn" if token == "bn" else int(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--script", required=True, help="Instantiation script path")
    parser.add_argument("--arch", required=True, help="Constructor name inside the script")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", default=None, help="Network label in the results (default: checkpoint stem)")
    parser.add_argument("--width", type=int, default=None, help="Width kwarg for thin-ResNet constructors")
    parser.add_argument("--keep", type=float, nargs="+", default=[0.8, 0.6],
                        help="Keep-rate applied to every group (the walk's compression rate)")
    parser.add_argument("--budgets", type=_budget, nargs="+", default=[0, "bn", 1, 3, 10, 40],
                        help="Fine-tune epochs per mask; 'bn' = BatchNorm re-estimation only")
    parser.add_argument("--criteria", nargs="+", default=list(CRITERIA), choices=CRITERIA)
    parser.add_argument("--random_masks", type=int, default=5)
    parser.add_argument("--l1_seeds", type=int, default=3, help="Fine-tune seeds for the L1 mask")
    parser.add_argument("--calib_batches", type=int, default=4,
                        help="Train batches scoring the data-driven criteria (protocol P: 256 each)")
    parser.add_argument("--rank_batches", type=int, default=1, help="Batches HRank's map ranks use")
    parser.add_argument("--bn_batches", type=int, default=50)
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stop patience (default SPECTRA_FINETUNE_PATIENCE, as the walk)")
    parser.add_argument("--nap_dir", default=None,
                        help="NAPv2 checkout: record its feature maps over the fine-tunes of --nap_budget")
    parser.add_argument("--nap_budget", type=int, default=40)
    parser.add_argument("--nap_interval", type=int, default=100, help="Mini-batches between snapshots")
    parser.add_argument("--nap_max", type=int, default=40)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    logging_utils.setup()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    numeric = [b for b in args.budgets if b != "bn"]
    _ensure_conf(device, max(numeric) if numeric else 0)
    StaticConf.get_instance().conf_values.device = device
    tag = args.tag or Path(args.checkpoint).stem

    registry = utils.preload_datasets([args.dataset], args.train_split, args.val_split)
    optional = {"width": args.width} if args.width is not None else {}
    model = utils.load_model_from_script(
        args.arch, args.dataset, args.script, args.checkpoint, optional,
        num_classes=registry.num_classes(args.dataset),
        input_shape=registry.input_shape(args.dataset),
    ).to(device).eval()
    loaders = registry.loaders(args.dataset)
    input_shape = tuple(registry.input_shape(args.dataset))

    base = ClassificationHandler(copy.deepcopy(model), nn.CrossEntropyLoss())
    base_val, base_test = float(base.evaluate_model(loaders[1])), float(base.evaluate_model(loaders[2]))
    params0 = utils.calc_num_parameters(model)
    flops0 = utils.calc_flops(model, input_shape, device)
    del base

    plan = cut_plan(model)
    utils.print_flush(
        f"Selection probe {tag} ({args.dataset}) on {device}: val {base_val:.4f} test {base_test:.4f}, "
        f"{params0 / 1e6:.3f} M params, {len(plan)} groups / {sum(g.width for _, g, _ in plan)} channels; "
        f"keep {args.keep} budgets {args.budgets} criteria {args.criteria}; "
        f"FT_AUG={os.environ.get('SPECTRA_FT_AUG', '0')} VAL_FROM_TEST={os.environ.get('SPECTRA_VAL_FROM_TEST', '0')}")

    started = time.perf_counter()
    batches = calibration_batches(loaders[0], args.calib_batches, device)
    scores, base_loss = all_scores(model, plan, batches, args.rank_batches)
    grads = mean_gradients(model, plan, batches)
    plan = [entry for entry in plan if entry[0] in scores["l1"]]  # alive = what SPECTRA's L1 can rank
    alive = {key: scores["l1"][key] > 0 for key, _, _ in plan}
    utils.print_flush(f"Scored {len(scores)} criteria in {time.perf_counter() - started:.0f}s "
                      f"(calibration loss {base_loss:.4f} on {sum(int(x.shape[0]) for x, _ in batches)} images)")

    out_dir = os.path.join(logging_utils.run_dir(), "results")
    os.makedirs(out_dir, exist_ok=True)
    columns, table = channel_features(plan, grads, scores)
    np.savez_compressed(os.path.join(out_dir, "selection_features.npz"), data=table,
                        columns=np.array(columns), groups=np.array(["|".join(k) for k, _, _ in plan]),
                        net=np.array(tag), dataset=np.array(args.dataset))
    vs_oracle = agreement(scores, alive, plan, "ablation")
    vs_l1 = agreement(scores, alive, plan, "l1")
    utils.print_flush(f"Within-group Kendall tau vs ablation: {vs_oracle}")
    utils.print_flush(f"Within-group Kendall tau vs l1:       {vs_l1}")

    nap = load_napv2(args.nap_dir) if args.nap_dir else None
    nap_dir_out = os.path.join(out_dir, "nap_maps")

    def collector_factory(net, budget):
        if nap is None or budget != args.nap_budget:
            return None
        return nap.SnapshotCollector(net, interval=args.nap_interval, max_snapshots=args.nap_max)

    results_path = os.path.join(out_dir, "selection_probe.jsonl")
    masks_path = os.path.join(out_dir, "selection_masks.jsonl")
    specs = mask_specs(args.criteria, args.random_masks, args.l1_seeds)
    rows, kept_by_mask = [], {}
    with RankingOverride() as override:
        for keep in args.keep:
            reference = None
            for spec in specs:
                crit, mask_seed, ft_seed = spec
                tables, fallbacks = score_tables(spec, scores, alive, plan)
                t0 = time.perf_counter()
                cut_model, modes, kept = cut(model, plan, keep, tables, override, input_shape)
                signature = shape_signature(cut_model)
                if reference is None:
                    reference = signature
                elif signature != reference:
                    raise RuntimeError(f"{crit} at keep {keep} cut a different shape: allocation is not matched")
                params_kept = utils.calc_num_parameters(cut_model) / params0
                flops_kept = utils.calc_flops(cut_model, input_shape, device) / flops0
                structural = sum(1 for m in modes if m == "structural")
                kept_by_mask[(keep, crit, mask_seed, ft_seed)] = kept
                with open(masks_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"net": tag, "keep": keep, "criterion": crit, "mask_seed": mask_seed,
                                             "kept": {"|".join(k): v for k, v in kept.items()}}) + "\n")
                for budget in args.budgets:
                    t1 = time.perf_counter()
                    val, test, collector = recover(cut_model, budget, loaders, device, ft_seed,
                                                   patience=args.patience, bn_batches=args.bn_batches,
                                                   collector_factory=collector_factory)
                    row = {"net": tag, "dataset": args.dataset, "keep": keep, "criterion": crit,
                           "mask_seed": mask_seed, "ft_seed": ft_seed, "budget": str(budget),
                           "val": round(val, 5), "test": round(test, 5),
                           "base_val": round(base_val, 5), "base_test": round(base_test, 5),
                           "d_val_pp": round((val - base_val) * 100, 3),
                           "d_test_pp": round((test - base_test) * 100, 3),
                           "params_kept": round(params_kept, 5), "flops_kept": round(flops_kept, 5),
                           "structural": f"{structural}/{len(modes)}", "fallback_groups": fallbacks,
                           "seconds": round(time.perf_counter() - t1, 1)}
                    if collector is not None:
                        os.makedirs(nap_dir_out, exist_ok=True)
                        w_maps, g_maps = nap_feature_maps(nap, collector)
                        name = f"{tag}_k{keep}_{crit}_m{mask_seed}_s{ft_seed}_e{budget}.npz"
                        np.savez_compressed(os.path.join(nap_dir_out, name), weights=w_maps, gradients=g_maps,
                                            val=val, test=test, base_val=base_val, interval=args.nap_interval)
                        row["nap_maps"] = name
                    rows.append(row)
                    with open(results_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    utils.print_flush(
                        f"[sel] {tag} keep={keep} {crit}/m{mask_seed}/s{ft_seed} budget={budget}: "
                        f"val {row['d_val_pp']:+.2f} pp test {row['d_test_pp']:+.2f} pp | params x{params_kept:.3f} "
                        f"FLOPs x{flops_kept:.3f} | structural {structural}/{len(modes)} | {row['seconds']:.0f}s")
                utils.print_flush(f"[sel] mask {crit}/m{mask_seed}/s{ft_seed} done in {time.perf_counter() - t0:.0f}s")
                del cut_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    overlap = {}
    for keep in args.keep:
        oracle = kept_by_mask.get((keep, "ablation", 0, 0))
        l1_mask = kept_by_mask.get((keep, "l1", 0, 0))
        for (k, crit, ms, fs), kept in kept_by_mask.items():
            if k != keep or fs != ms:
                continue
            overlap[f"keep={keep} {crit}/m{ms}"] = {
                "vs_ablation": jaccard(kept, oracle) if oracle else None,
                "vs_l1": jaccard(kept, l1_mask) if l1_mask else None}

    _print_tables(rows, args.keep, args.budgets, "d_val_pp")
    _print_tables(rows, args.keep, args.budgets, "d_test_pp")
    summary = summarize(rows, args.keep, args.budgets)
    for name, cell in summary.items():
        utils.print_flush(f"[lever] {tag} {name}: {cell['lever']}")
    utils.print_flush(f"[overlap] {tag} Jaccard of kept channels: {overlap}")
    with open(results_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "summary", "net": tag, "dataset": args.dataset,
                                 "base_val": base_val, "base_test": base_test,
                                 "tau_vs_ablation": vs_oracle, "tau_vs_l1": vs_l1,
                                 "jaccard": overlap, "cells": summary}) + "\n")
    utils.print_flush(f"Wrote {results_path}, {masks_path} and selection_features.npz "
                      f"in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
