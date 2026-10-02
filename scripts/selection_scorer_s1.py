#!/usr/bin/env python
"""
S1: a learned NAP-F filter scorer, leave one network out (zero GPU; design doc §6.3, gate G1 in §8).

    python scripts/selection_scorer_s1.py <run_dir> <run_dir> <run_dir> [--out s1.json]

Each ``run_dir`` is an S0 cell (``results/selection_features.npz`` + ``results/selection_probe.jsonl``).
The label is the within-group rank of the single-channel ablation oracle; the features are every other
NAP-F column, rank- and robust-z-normalised within each group so scales from different networks never
meet, plus the group's position. For each held-out net the learner and its hyper-parameters are chosen
by cross-fitting between the two training nets only; the held-out net is scored once, at the end.

The metric is S0's own: width-weighted mean within-group Kendall tau-b against the oracle over live
channels (L1 > 0), groups weighted by their live count. Every hand criterion is scored by the same code
and must reproduce the ``tau_vs_ablation`` S0 printed (checked; a mismatch aborts).

G1: held-out tau >= best hand criterion's tau + 0.05 on >= 2 of the 3 held-out nets.

``--m8`` adds a zero-GPU re-read of S0's lever: at keep 0.6, each budget's best named criterion minus
the L1 mean, against a null in which every named criterion equals L1 and differs by fine-tune noise only
(SD = the pooled L1 fine-tune-seed SD of that cell). Val only, as S0's calls.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection_probe import kendall_tau  # noqa: E402  (S0's tau-b, unchanged)

HAND = ("l1", "l2", "svd", "fpgm", "bn_scale", "taylor", "act", "apoz", "hrank")
ORACLE = "ablation"
EXTRA_HAND = ("out_l1",)        # consumer-side L1: printed, not part of G1's bar (S0 did not print it)
POSITION = ("depth", "width", "n_producers")
G1_MARGIN = 0.05
G1_NETS = 2


# ------------------------------------------------------------------ data

def load_cell(run_dir):
    """``{"net", "columns", "data", "summary", "rows"}`` of one S0 cell."""
    res = os.path.join(run_dir, "results") if os.path.isdir(os.path.join(run_dir, "results")) else run_dir
    npz = np.load(os.path.join(res, "selection_features.npz"), allow_pickle=False)
    rows, summary = [], None
    path = os.path.join(res, "selection_probe.jsonl")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("kind") == "summary":
                    summary = rec
                else:
                    rows.append(rec)
    return {"net": str(npz["net"]), "columns": [str(c) for c in npz["columns"]], "data": np.asarray(npz["data"]),
            "summary": summary, "rows": rows}


def column(cell, name):
    return cell["data"][:, cell["columns"].index(name)]


def channel_columns(cell):
    """Every per-channel feature column: the criteria except the oracle, consumer L1, NAP statistics."""
    skip = {"group", "channel", ORACLE} | set(POSITION)
    return [c for c in cell["columns"] if c not in skip]


def groups_of(cell):
    """``[(row indices of live channels)]`` per group with >= 2 live channels (S0's ``alive``)."""
    gid, l1 = column(cell, "group"), column(cell, "l1")
    out = []
    for g in np.unique(gid):
        idx = np.where((gid == g) & (l1 > 0))[0]
        if len(idx) >= 2:
            out.append(idx)
    return out


# ------------------------------------------------------------------ metric

def tau_groups(score, oracle, groups):
    """S0's agreement(): live-count-weighted mean within-group tau-b. Groups with no score are skipped."""
    num = den = 0.0
    for idx in groups:
        s = score[idx]
        if not np.all(np.isfinite(s)):
            continue
        tau = kendall_tau(s, oracle[idx])
        if np.isfinite(tau):
            num += tau * len(idx)
            den += len(idx)
    return num / den if den else float("nan")


# ------------------------------------------------------------------ features

def _rank01(x):
    """Average-tie percentile ranks in [0, 1]; NaN -> 0.5."""
    out = np.full(len(x), 0.5)
    ok = np.isfinite(x)
    if ok.sum() >= 2:
        vals = x[ok]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(len(vals))
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        out[ok] = ranks / max(1, len(vals) - 1)
    return out


def _robust_z(x):
    """Within-group robust z of asinh(x / median|x|); NaN -> 0; clipped to [-5, 5]."""
    out = np.zeros(len(x))
    ok = np.isfinite(x)
    if ok.sum() >= 2:
        v = x[ok]
        scale = np.median(np.abs(v)) or (np.mean(np.abs(v)) or 1.0)
        a = np.arcsinh(v / scale)
        med = np.median(a)
        iqr = np.subtract(*np.percentile(a, [75, 25])) or (np.std(a) or 1.0)
        out[ok] = np.clip((a - med) / iqr, -5, 5)
    return out


def design(cell, groups, feature_cols):
    """``(X, y, group_of_row, names)`` over live channels; y = within-group oracle percentile."""
    oracle = column(cell, ORACLE)
    depth, width, nprod = column(cell, "depth"), column(cell, "width"), column(cell, "n_producers")
    blocks, ys, gix = [], [], []
    for gi, idx in enumerate(groups):
        feats = []
        for c in feature_cols:
            x = column(cell, c)[idx]
            feats.append(_rank01(x))
            feats.append(_robust_z(x))
        live_frac = len(idx) / max(1.0, float(width[idx[0]]))
        pos = np.tile([depth[idx[0]], np.log2(max(1.0, width[idx[0]])), np.log2(1.0 + nprod[idx[0]]), live_frac],
                      (len(idx), 1))
        blocks.append(np.column_stack(feats + [pos]))
        ys.append(_rank01(oracle[idx]))
        gix.append(np.full(len(idx), gi))
    names = [f"{c}:{k}" for c in feature_cols for k in ("rank", "z")] + ["depth", "log2_width", "log2_producers",
                                                                          "live_frac"]
    return np.vstack(blocks), np.concatenate(ys), np.concatenate(gix), names


# ------------------------------------------------------------------ learners

def _pairs(X, y, gix, rng, per_group=400):
    """Within-group pair differences for a pairwise logistic ranker, randomly oriented (ties dropped)."""
    dx, dy = [], []
    for g in np.unique(gix):
        idx = np.where(gix == g)[0]
        pairs = list(itertools.combinations(idx, 2))
        if len(pairs) > per_group:
            pairs = [pairs[i] for i in rng.choice(len(pairs), per_group, replace=False)]
        for i, j in pairs:
            if y[i] == y[j]:
                continue
            if rng.random() < 0.5:
                i, j = j, i
            dx.append(X[i] - X[j])
            dy.append(int(y[i] > y[j]))
    return np.asarray(dx), np.asarray(dy)


def _net_weights(sizes):
    """Per-row weights that give every net the same total and average 1."""
    total = float(sum(sizes))
    return np.concatenate([np.full(n, total / (len(sizes) * n)) for n in sizes])


def learners():
    """``[(name, params, factory)]``: small grids, chosen by inner cross-fit only."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    grid = []
    for alpha in (1.0, 10.0, 100.0):
        grid.append(("ridge", {"alpha": alpha}, lambda a=alpha: Ridge(alpha=a)))
    for iters, leaves in ((150, 15), (300, 15), (300, 31)):
        grid.append(("gbm", {"max_iter": iters, "max_leaf_nodes": leaves},
                     lambda i=iters, l=leaves: HistGradientBoostingRegressor(
                         max_iter=i, learning_rate=0.05, max_leaf_nodes=l, min_samples_leaf=40,
                         l2_regularization=1.0, random_state=0)))
    for c in (0.01, 0.1, 1.0):
        grid.append(("pairwise", {"C": c}, lambda cc=c: LogisticRegression(C=cc, max_iter=2000)))
    return grid


def fit_model(name, factory, parts, seed=0):
    """Fit on a list of ``(X, y, gix)``; every net weighs the same. Returns the fitted estimator."""
    model = factory()
    if name == "pairwise":
        rng = np.random.default_rng(seed)
        dxs, dys = [], []
        for X, y, gix in parts:
            dx, dy = _pairs(X, y, gix, rng)
            dxs.append(dx)
            dys.append(dy)
        model.fit(np.vstack(dxs), np.concatenate(dys), sample_weight=_net_weights([len(d) for d in dys]))
        return model
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    model.fit(X, y, sample_weight=_net_weights([len(p[1]) for p in parts]))
    return model


def score_with(name, model, X):
    """Higher = keep. The pairwise ranker scores by its linear margin."""
    return X @ model.coef_.ravel() if name == "pairwise" else model.predict(X)


def fit(name, factory, parts, seed=0):
    model = fit_model(name, factory, parts, seed)
    return lambda X: score_with(name, model, X)


# ------------------------------------------------------------------ S1

FAMILIES = {
    "hand": lambda c: c in HAND + EXTRA_HAND,
    "nap_w": lambda c: c.startswith("w_"),
    "nap_g": lambda c: c.startswith("g_"),
    "no_gradient": lambda c: not c.startswith("g_") and c != "taylor",
    "all": lambda c: True,
}


def prepare(cells, keep_feature=None):
    feature_cols = [c for c in channel_columns(cells[0]) if all(c in k["columns"] for k in cells)]
    if keep_feature is not None:
        feature_cols = [c for c in feature_cols if keep_feature(c)]
    prepared = []
    for cell in cells:
        groups = groups_of(cell)
        X, y, gix, _ = design(cell, groups, feature_cols)
        prepared.append({"cell": cell, "groups": groups, "X": X, "y": y, "gix": gix,
                         "oracle": column(cell, ORACLE)})
    return feature_cols, prepared


def evaluate(cells, keep_feature=None):
    nets = [c["net"] for c in cells]
    feature_cols, prepared = prepare(cells, keep_feature)
    # Hand criteria by S0's metric, and the reproduction check against S0's printed tau.
    hand, checks = {}, []
    for p in prepared:
        cell = p["cell"]
        hand[cell["net"]] = {c: tau_groups(column(cell, c), p["oracle"], p["groups"])
                             for c in HAND + EXTRA_HAND if c in cell["columns"]}
        printed = (cell["summary"] or {}).get("tau_vs_ablation") or {}
        for c in HAND:
            if c in printed and printed[c] is not None:
                checks.append((cell["net"], c, round(hand[cell["net"]][c], 4), printed[c]))
    bad = [c for c in checks if abs(c[2] - c[3]) > 2e-3]
    if bad:
        raise SystemExit(f"hand-criterion tau does not reproduce S0's printed values: {bad}")
    folds = []
    for k, held in enumerate(prepared):
        train = [p for j, p in enumerate(prepared) if j != k]
        inner = []
        for name, params, factory in learners():
            taus = []
            for a, b in ((0, 1), (1, 0)):
                predict = fit(name, factory, [(train[a]["X"], train[a]["y"], train[a]["gix"])])
                taus.append(tau_groups_rows(predict(train[b]["X"]), train[b]))
            inner.append((float(np.mean(taus)), name, params, factory))
        inner.sort(key=lambda t: -t[0])
        best_inner, name, params, factory = inner[0]
        predict = fit(name, factory, [(p["X"], p["y"], p["gix"]) for p in train])
        tau_held = tau_groups_rows(predict(held["X"]), held)
        net = held["cell"]["net"]
        bar_crit = max((c for c in HAND if c in hand[net]), key=lambda c: hand[net][c])
        train_nets = [p["cell"]["net"] for p in train]
        fair = max(HAND, key=lambda c: np.mean([hand[t][c] for t in train_nets]))
        folds.append({"held_out": net, "train": train_nets, "learner": name, "params": params,
                      "inner_tau": round(best_inner, 4), "tau": round(tau_held, 4),
                      "best_hand": bar_crit, "best_hand_tau": round(hand[net][bar_crit], 4),
                      "margin": round(tau_held - hand[net][bar_crit], 4),
                      "train_chosen_hand": fair, "train_chosen_hand_tau": round(hand[net][fair], 4),
                      "pass": bool(tau_held >= hand[net][bar_crit] + G1_MARGIN),
                      "inner_grid": [(round(t, 4), n, p) for t, n, p, _ in inner]})
    passes = sum(f["pass"] for f in folds)
    return {"nets": nets, "features": len(feature_cols), "hand": {n: {c: round(v, 4) for c, v in h.items()}
                                                                  for n, h in hand.items()},
            "reproduced": len(checks), "folds": folds, "g1_passes": passes, "g1": passes >= G1_NETS}


def tau_groups_rows(pred, p):
    """tau of row-ordered predictions (design() order) against the oracle, S0's weighting."""
    num = den = 0.0
    for gi, idx in enumerate(p["groups"]):
        rows = np.where(p["gix"] == gi)[0]
        tau = kendall_tau(pred[rows], p["oracle"][idx])
        if np.isfinite(tau):
            num += tau * len(idx)
            den += len(idx)
    return num / den if den else float("nan")


# ------------------------------------------------------------------ M8 re-read

def m8_reread(cell, keep=0.6, draws=20000, seed=0):
    """Per trained budget: best named minus L1 mean, its null p-value, and L1 minus random / anti-L1 (val pp).

    Budgets 0 and ``bn`` are left out: no fine-tune runs there, so seeds cannot spread and a null is moot.
    """
    rows = [r for r in cell["rows"] if abs(float(r["keep"]) - keep) < 1e-9]
    budgets = []
    for r in rows:
        if r["budget"] not in budgets and r["budget"] not in ("0", "bn"):
            budgets.append(r["budget"])
    l1_sd = []
    for b in budgets:
        v = [r["d_val_pp"] for r in rows if r["criterion"] == "l1" and r["budget"] == b]
        if len(v) >= 2:
            l1_sd.append(np.var(v, ddof=1))
    sigma = float(np.sqrt(np.mean(l1_sd))) if l1_sd else float("nan")
    rng = np.random.default_rng(seed)
    out = []
    for b in budgets:
        at = [r for r in rows if r["budget"] == b]
        l1 = [r["d_val_pp"] for r in at if r["criterion"] == "l1"]
        named = {r["criterion"]: r["d_val_pp"] for r in at
                 if r["criterion"] not in ("l1", "random", "anti_l1", ORACLE)}
        rand = [r["d_val_pp"] for r in at if r["criterion"] == "random"]
        anti = [r["d_val_pp"] for r in at if r["criterion"] == "anti_l1"]
        if not l1 or not named or not np.isfinite(sigma):
            continue
        l1m = float(np.mean(l1))
        best = max(named, key=named.get)
        obs = named[best] - l1m
        null = (rng.normal(0, sigma, (draws, len(named))).max(axis=1)
                - rng.normal(0, sigma / np.sqrt(len(l1)), draws))
        p = float((null >= obs).mean())
        se_rand = np.sqrt(sigma ** 2 / len(l1) + (np.var(rand, ddof=1) if len(rand) > 1 else sigma ** 2) / max(1, len(rand)))
        out.append({"budget": b, "best_named": best, "best_minus_l1": round(obs, 3), "p_null_max": round(p, 3),
                    "sigma_ft": round(sigma, 3), "n_named": len(named),
                    "l1_sd": round(float(np.std(l1, ddof=1)), 3) if len(l1) > 1 else None,
                    "l1_minus_random": round(l1m - float(np.mean(rand)), 3) if rand else None,
                    "z_l1_vs_random": round((l1m - float(np.mean(rand))) / se_rand, 2) if rand else None,
                    "l1_minus_anti": round(l1m - float(np.mean(anti)), 3) if anti else None,
                    "ablation_minus_l1": round(next((r["d_val_pp"] for r in at if r["criterion"] == ORACLE),
                                                    float("nan")) - l1m, 3)})
    return {"net": cell["net"], "keep": keep, "sigma_ft_pooled": round(sigma, 3), "budgets": out}


def render(result, m8=None):
    lines = [f"S1 learned NAP-F scorer, leave one network out ({result['features']} feature columns x rank/z "
             f"+ 4 position; hand tau reproduced S0 on {result['reproduced']} values)", "",
             "| held-out net | trained on | learner (inner cross-fit tau) | held-out tau | best hand (tau) | "
             "margin | train-chosen hand (tau) | G1 |", "|---|---|---|---|---|---|---|---|"]
    for f in result["folds"]:
        params = ",".join(f"{k}={v}" for k, v in f["params"].items())
        lines.append(f"| {f['held_out']} | {' + '.join(f['train'])} | {f['learner']} {params} ({f['inner_tau']:+.3f}) | "
                     f"**{f['tau']:+.3f}** | {f['best_hand']} ({f['best_hand_tau']:+.3f}) | {f['margin']:+.3f} | "
                     f"{f['train_chosen_hand']} ({f['train_chosen_hand_tau']:+.3f}) | {'pass' if f['pass'] else 'fail'} |")
    lines += ["", f"G1 (held-out tau >= best hand + {G1_MARGIN} on >= {G1_NETS} of 3): "
              f"{'PASS' if result['g1'] else 'FAIL'} ({result['g1_passes']}/3)", "", "Hand criteria vs the oracle:"]
    for net, h in result["hand"].items():
        lines.append(f"- {net}: " + ", ".join(f"{c} {v:+.3f}" for c, v in h.items()))
    if m8:
        lines += ["", "M8 re-read (keep 0.6, val pp; null = every named criterion equals L1, noise = pooled L1 seed SD):",
                  "| net | budget | L1 seed SD | best named | best − L1 | p(max of named ≥ obs under null) | "
                  "L1 − random (z) | L1 − anti-L1 | oracle − L1 |", "|---|---|---|---|---|---|---|---|---|"]
        def fmt(v, spec="+.2f"):
            return "-" if v is None else format(v, spec)

        for m in m8:
            for b in m["budgets"]:
                lines.append(f"| {m['net']} | {b['budget']} | {fmt(b['l1_sd'], '.2f')} | {b['best_named']} | "
                             f"{b['best_minus_l1']:+.2f} | "
                             f"{b['p_null_max']:.2f} | {fmt(b['l1_minus_random'])} ({fmt(b['z_l1_vs_random'], '+.1f')}) | "
                             f"{fmt(b['l1_minus_anti'])} | {fmt(b['ablation_minus_l1'])} |")
    return "\n".join(lines)


def export_model(cells, path):
    """The S2 scorer: the grid point with the best mean leave-one-net-out tau over all cells, refit on all."""
    import pickle

    feature_cols, prepared = prepare(cells)
    scored = []
    for name, params, factory in learners():
        taus = []
        for k, held in enumerate(prepared):
            train = [(p["X"], p["y"], p["gix"]) for j, p in enumerate(prepared) if j != k]
            taus.append(tau_groups_rows(fit(name, factory, train)(held["X"]), held))
        scored.append((float(np.mean(taus)), name, params, factory, taus))
    scored.sort(key=lambda t: -t[0])
    lono, name, params, factory, taus = scored[0]
    model = fit_model(name, factory, [(p["X"], p["y"], p["gix"]) for p in prepared])
    bundle = {"learner": name, "params": params, "feature_cols": feature_cols, "model": model,
              "trained_on": [p["cell"]["net"] for p in prepared], "lono_tau": round(lono, 4),
              "lono_fold_tau": [round(t, 4) for t in taus]}
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    return {k: v for k, v in bundle.items() if k != "model"}


def score_net(bundle, columns, data):
    """``{group index: scores over that group's live channels}`` for one probe table (S2's ``nap_f``)."""
    cell = {"net": "", "columns": list(columns), "data": np.asarray(data)}
    groups = groups_of(cell)
    X, _, gix, _ = design(cell, groups, bundle["feature_cols"])
    pred = score_with(bundle["learner"], bundle["model"], X)
    gid = column(cell, "group")
    return {int(gid[idx[0]]): (idx, pred[gix == gi]) for gi, idx in enumerate(groups)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs=3, help="the three S0 run directories")
    parser.add_argument("--out", default="", help="write the full result as JSON")
    parser.add_argument("--m8", action="store_true", help="add the M8 null re-read")
    parser.add_argument("--families", action="store_true", help="also refit on each feature family alone")
    parser.add_argument("--export", default="", help="fit the S2 scorer on all three cells and pickle it here")
    args = parser.parse_args(argv)
    cells = [load_cell(r) for r in args.runs]
    result = evaluate(cells)
    m8 = [m8_reread(c) for c in cells] if args.m8 else None
    print(render(result, m8))
    families = {}
    if args.families:
        print("\nFeature families (same nested protocol; held-out tau per net):")
        for fam, keep in FAMILIES.items():
            r = evaluate(cells, keep)
            families[fam] = {f["held_out"]: f["tau"] for f in r["folds"]}
            print(f"- {fam} ({r['features']} columns): "
                  + ", ".join(f"{f['held_out']} {f['tau']:+.3f} ({f['learner']})" for f in r["folds"]))
    exported = None
    if args.export:
        exported = export_model(cells, args.export)
        print(f"\nExported S2 scorer to {args.export}: {exported}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"s1": result, "m8": m8, "families": families, "export": exported}, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
