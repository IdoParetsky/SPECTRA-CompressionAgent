#!/usr/bin/env python
"""
S2: S0's selection probe with S1's learned scorer as one more criterion (design doc §8, gate G2).

    python scripts/selection_probe_s2.py --nap_f_model <S1 export .pkl> [--oracle_seeds 3] \
        <scripts/selection_probe.py arguments>
    python scripts/selection_probe_s2.py --readout <S2 run dir> [<S2 run dir> ...]

``scripts/selection_probe.py`` runs unchanged apart from three hooks:
- ``nap_f`` joins the criteria. It is the S1 scorer (fit on the three S0 cells) applied to this net's NAP-F
  table, which the probe builds on the unpruned network before any mask is cut. Groups the scorer cannot
  rank (fewer than 2 live channels) keep L1's order.
- L1 and ``nap_f`` both get ``--l1_seeds`` fine-tune seeds and the oracle gets ``--oracle_seeds``. Seeds are
  shared, so every (criterion, seed) run is paired with L1's run at the same seed and data order.
- The scorer's provenance is printed before the probe's banner.

The readout reads validation only. Per cell and budget b, H_b is the mean over seeds of nap_f minus L1
(paired, keep 0.6, pp) and sigma_ft is the SD of L1 over its seeds. Calls, registered before any S2 run:
- PASS: H_40 >= max(0.3, 2 sigma_ft) on every cell, and H_40 >= -sigma_ft on every cell.
- CHEAP-FT: not PASS, H_40 >= -sigma_ft everywhere, and for one b in {bn, 1, 3} H_b >= max(0.5, 2 sigma_ft(b))
  on every cell: a scorer for cheap in-loop fine-tunes only, no second agent.
- HARM: H_40 < -sigma_ft on some cell.
- FAIL: otherwise. Keep L1 and do not start S3.
The oracle line O_b (ablation minus L1, its seeds paired) says whether the label itself is worth learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selection_probe as probe  # noqa: E402
import selection_scorer_s1 as s1  # noqa: E402

NAP_F = "nap_f"
CHEAP_BUDGETS = ("bn", "1", "3")


def nap_f_tables(bundle, plan, scores, columns, table):
    """Per group, the scorer's value on live channels (higher = keep) and L1's order where it cannot rank."""
    per_group = s1.score_net(bundle, columns, table)
    channel = np.asarray(table)[:, list(columns).index("channel")].astype(int)
    tables, fallback = {}, 0
    for gi, (key, group, _) in enumerate(plan):
        if gi not in per_group:
            tables[key] = scores["l1"][key].double().clone()
            fallback += 1
            continue
        rows, pred = per_group[gi]
        values = torch.full((group.width,), float(np.min(pred)) - 1.0, dtype=torch.float64)
        values[torch.as_tensor(channel[rows], dtype=torch.long)] = torch.as_tensor(pred, dtype=torch.float64)
        tables[key] = values
    return tables, fallback


def install(bundle, oracle_seeds: int = 3, module=probe):
    """Patch ``module`` (the probe) in place; returns a function that undoes it."""
    saved = {name: getattr(module, name) for name in ("CRITERIA", "channel_features", "mask_specs")}
    original_features = saved["channel_features"]

    def channel_features(plan, grads, scores):
        columns, table = original_features(plan, grads, scores)
        scores[NAP_F], fallback = nap_f_tables(bundle, plan, scores, columns, table)
        print(f"[s2] nap_f scored {len(plan) - fallback}/{len(plan)} groups; L1 order on {fallback}", flush=True)
        return columns, table

    def mask_specs(criteria, random_masks: int, l1_seeds: int):
        specs = []
        for crit in criteria:
            if crit == "random":
                specs += [("random", s, s) for s in range(random_masks)]
            elif crit in ("l1", NAP_F):
                specs += [(crit, 0, s) for s in range(l1_seeds)]
            elif crit == s1.ORACLE:
                specs += [(crit, 0, s) for s in range(min(oracle_seeds, l1_seeds))]
            else:
                specs.append((crit, 0, 0))
        return specs

    if NAP_F not in module.CRITERIA:
        module.CRITERIA = tuple(module.CRITERIA) + (NAP_F,)
    module.channel_features = channel_features
    module.mask_specs = mask_specs

    def restore():
        for name, value in saved.items():
            setattr(module, name, value)

    return restore


# ------------------------------------------------------------------ readout

def _rows(run_dir):
    res = os.path.join(run_dir, "results") if os.path.isdir(os.path.join(run_dir, "results")) else run_dir
    rows = []
    with open(os.path.join(res, "selection_probe.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("kind") != "summary":
                    rows.append(rec)
    return rows


def paired(rows, crit, budget, keep=0.6):
    """``(mean of crit - L1 over shared fine-tune seeds, its SE, n, L1 SD over all its seeds)``."""
    at = [r for r in rows if r["budget"] == budget and abs(float(r["keep"]) - keep) < 1e-9]
    l1 = {r["ft_seed"]: r["d_val_pp"] for r in at if r["criterion"] == "l1"}
    other = {r["ft_seed"]: r["d_val_pp"] for r in at if r["criterion"] == crit and (crit != "random"
                                                                                 or r["mask_seed"] == r["ft_seed"])}
    seeds = sorted(set(l1) & set(other))
    sd = float(np.std(list(l1.values()), ddof=1)) if len(l1) > 1 else float("nan")
    if not seeds:
        return float("nan"), float("nan"), 0, sd
    diff = np.array([other[s] - l1[s] for s in seeds])
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return float(diff.mean()), se, len(diff), sd


def readout(run_dirs, keep=0.6):
    cells = {}
    for run in run_dirs:
        rows = _rows(run)
        net = rows[0]["net"] if rows else os.path.basename(run.rstrip("/"))
        budgets = []
        for r in rows:
            if r["budget"] not in budgets:
                budgets.append(r["budget"])
        cells[net] = {b: {c: paired(rows, c, b, keep) for c in (NAP_F, s1.ORACLE, "random", "anti_l1")}
                      for b in budgets}
    calls = g2_call(cells)
    return cells, calls


def g2_call(cells):
    def h(net, b, crit=NAP_F):
        return cells[net].get(b, {}).get(crit, (float("nan"),) * 4)

    nets = list(cells)
    h40 = {n: h(n, "40") for n in nets}
    harm = [n for n in nets if h40[n][0] < -h40[n][3]]
    passed = all(h40[n][0] >= max(0.3, 2 * h40[n][3]) for n in nets) and not harm
    cheap = [b for b in CHEAP_BUDGETS
             if all(h(n, b)[0] >= max(0.5, 2 * h(n, b)[3]) for n in nets)]
    if passed:
        call = "PASS"
    elif harm:
        call = "HARM"
    elif cheap:
        call = "CHEAP-FT"
    else:
        call = "FAIL"
    return {"call": call, "harm": harm, "cheap_budgets": cheap, "nets": nets}


def render_readout(cells, calls):
    lines = ["| cell | budget | L1 seed SD | nap_f − L1 (SE, n) | oracle − L1 (SE, n) | random − L1 (n) | anti-L1 − L1 |",
             "|---|---|---|---|---|---|---|"]
    for net, by_b in cells.items():
        for b, d in by_b.items():
            nf, orc, rnd, anti = d[NAP_F], d[s1.ORACLE], d["random"], d["anti_l1"]
            lines.append(f"| {net} | {b} | {nf[3]:.2f} | {nf[0]:+.2f} ({nf[1]:.2f}, {nf[2]}) | "
                         f"{orc[0]:+.2f} ({orc[1]:.2f}, {orc[2]}) | {rnd[0]:+.2f} ({rnd[2]}) | {anti[0]:+.2f} |")
    lines.append(f"\nG2 call: **{calls['call']}** (harm on {calls['harm'] or 'none'}; "
                 f"cheap-FT budgets passing on every cell: {calls['cheap_budgets'] or 'none'})")
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--nap_f_model", default="")
    pre.add_argument("--oracle_seeds", type=int, default=3)
    pre.add_argument("--readout", nargs="+", default=None)
    known, rest = pre.parse_known_args(argv)
    if known.readout:
        cells, calls = readout(known.readout)
        print(render_readout(cells, calls))
        return 0
    if not known.nap_f_model:
        raise SystemExit("--nap_f_model is required (or --readout)")
    with open(known.nap_f_model, "rb") as fh:
        raw = fh.read()
    bundle = pickle.loads(raw)
    install(bundle, known.oracle_seeds)
    print(f"[s2] nap_f scorer {hashlib.md5(raw).hexdigest()[:12]}: {bundle['learner']} {bundle['params']} "
          f"trained on {bundle['trained_on']}, leave-one-net-out tau {bundle['lono_tau']} "
          f"{bundle['lono_fold_tau']}; oracle seeds {known.oracle_seeds}", flush=True)
    sys.argv = [sys.argv[0]] + rest
    return probe.main()


if __name__ == "__main__":
    raise SystemExit(main())
