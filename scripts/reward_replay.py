#!/usr/bin/env python
"""
O38 reward replay: score finished no-agent walks under several reward shapes (CPU, val only).

    python scripts/reward_replay.py <run_dir | events.jsonl> [...] [--tau 10] [--gamma 1.0]

A walk's ``step`` events (eval_test phase) carry what the environment scores: the cumulative val
Δacc against the origin (``new_acc − baseline_acc``, as ``NetworkEnv.step`` passes it) and the
per-step realised parameter cut ρ = 100·(1 − params_after / params_before). Every shape below is
the live ``src.utils.compute_reward`` under one (``SPECTRA_REWARD_MODE``, ``SPECTRA_REWARD_SCALE``)
pair, so the replay cannot drift from the env.

Per network and shape: the return, the cut where the cumulative return peaks (where a
return-maximiser would stop on this walk), params kept and val Δ there, and the share of the
positive return paid on the gain arm (Δ > 0). The branch census and the ρ distribution do not
depend on the shape. The walk's order is fixed, so this shows what each reward prefers along
that walk, not what a policy trained on it would do. Never reads TEST.
"""
import argparse
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SHAPES = (
    ("live Stage 4", "structural", "cbrt_cubes"),
    ("cubic gain", "structural", "cbrt_miss"),
    ("NEON raw, realised ρ", "structural", "raw"),
    ("full cbrt", "structural", "cbrt"),
    ("band miss", "structural_band", "cbrt_cubes"),
    ("NEON literal, nominal ρ", "neon", "raw"),
)


def load_walks(path):
    """``{network: [step rows in step order]}`` from ``step`` events; later rows win per step."""
    files = sorted(glob.glob(os.path.join(path, "events", "rank*.jsonl"))) if os.path.isdir(path) else [path]
    walks = {}
    for name in files:
        with open(name, encoding="utf-8") as fh:
            for line in fh:
                if '"event": "step"' not in line and '"event":"step"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                phase = ev.get("phase") or ev.get("mode")
                if ev.get("event") != "step" or (phase and phase != "eval_test") or ev.get("new_acc") is None:
                    continue
                net = os.path.basename(str(ev.get("network") or ev.get("net") or "?"))
                walks.setdefault(net, {})[int(ev["step_index"])] = ev
    return {net: [rows[k] for k in sorted(rows)] for net, rows in walks.items()}


def step_inputs(ev):
    """(new_acc, baseline_acc, rate, params_before, params_after, is_cut) as the env scored them."""
    before, after = float(ev["params_before_m"]), float(ev["params_after_m"])
    if ev.get("prune_mode") == "structural":
        return float(ev["new_acc"]), float(ev["baseline_acc"]), float(ev["compression_rate"]), before, after, True
    # identity / skipped / infeasible: no edit, so no size credit under any shape
    return float(ev["new_acc"]), float(ev["baseline_acc"]), 1.0, before, before, False


def replay(rows, mode, scale, tau, gamma):
    keys = ("SPECTRA_REWARD_MODE", "SPECTRA_REWARD_SCALE")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["SPECTRA_REWARD_MODE"], os.environ["SPECTRA_REWARD_SCALE"] = mode, scale
    try:
        return _replay(rows, tau, gamma)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _replay(rows, tau, gamma):
    import src.utils as utils
    origin = float(rows[0]["params_before_m"])
    ret, disc, best = 0.0, 1.0, (0.0, None)
    pos_total = pos_gain = 0.0
    cut_no = 0
    for ev in rows:
        new_acc, base, rate, before, after, is_cut = step_inputs(ev)
        r = float(utils.compute_reward(new_acc, base, rate, params_before=before, params_after=after, tau=tau))
        ret += disc * r
        disc *= gamma
        if r > 0:
            pos_total += r
            if (new_acc - base) * 100.0 > 0:
                pos_gain += r
        if is_cut:
            cut_no += 1
            if ret > best[0]:
                best = (ret, (cut_no, after / origin, (new_acc - base) * 100.0))
    return ret, best, (pos_gain / pos_total if pos_total > 0 else 0.0)


def census(rows, tau):
    cuts = [step_inputs(ev) for ev in rows]
    cuts = [c for c in cuts if c[5]]
    dacc = [(c[0] - c[1]) * 100.0 for c in cuts]
    rho = [max(0.0, (1.0 - c[4] / c[3]) * 100.0) for c in cuts if c[3] > 0]
    origin = float(rows[0]["params_before_m"])
    return {
        "cuts": len(cuts),
        "gain": sum(d > 0 for d in dacc),
        "band": sum(-tau <= d <= 0 for d in dacc),
        "miss": sum(d < -tau for d in dacc),
        "rho_med": statistics.median(rho) if rho else 0.0,
        "rho_p90": sorted(rho)[int(0.9 * (len(rho) - 1))] if rho else 0.0,
        "rho_max": max(rho) if rho else 0.0,
        "rho_gt1": sum(r > 1.0 for r in rho) / len(rho) if rho else 0.0,
        "keep_end": float(cuts[-1][4]) / origin if cuts else 1.0,
        "dacc_end": dacc[-1] if dacc else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("walks", nargs="+", help="run dirs or events.jsonl files")
    ap.add_argument("--tau", type=float, default=10.0, help="band in pp (Stage 4 and the walks: 10)")
    ap.add_argument("--gamma", type=float, default=1.0, help="discount along the walk (1 = undiscounted)")
    args = ap.parse_args(argv)
    for path in args.walks:
        label = os.path.basename(os.path.normpath(path))
        for net, rows in sorted(load_walks(path).items()):
            c = census(rows, args.tau)
            if not c["cuts"]:
                print(f"{label} {net}: no structural cuts")
                continue
            print(f"\n### {label} — {net} (τ = {args.tau:g} pp, γ = {args.gamma:g})")
            print(f"{c['cuts']} cuts: gain arm {c['gain']} ({c['gain'] / c['cuts']:.0%}), in band {c['band']}, "
                  f"miss {c['miss']}. Walk ends at params kept {c['keep_end']:.3f}, val Δ {c['dacc_end']:+.2f} pp.")
            print(f"Per-cut ρ (% of the current net): median {c['rho_med']:.2f}, p90 {c['rho_p90']:.2f}, "
                  f"max {c['rho_max']:.2f}; share of cuts with ρ > 1: {c['rho_gt1']:.0%}.")
            print("| Shape | mode + scale | Return | Peak at cut | Params kept at peak | Val Δ at peak (pp) "
                  "| Gain-arm share of + return |")
            print("|---|---|---|---|---|---|---|")
            for name, mode, scale in SHAPES:
                ret, (peak, where), share = replay(rows, mode, scale, args.tau, args.gamma)
                if where is None:
                    peak_s, keep_s, dacc_s = "none (never above 0)", "1.000", "—"
                else:
                    peak_s = f"{where[0]} of {c['cuts']}"
                    keep_s = f"{where[1]:.3f}"
                    dacc_s = f"{where[2]:+.2f}"
                print(f"| {name} | {mode} + {scale} | {ret:.4g} | {peak_s} | {keep_s} | {dacc_s} | {share:.0%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
