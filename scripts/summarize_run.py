#!/usr/bin/env python
"""
Turn a SPECTRA run directory into a readable status report.

    python scripts/summarize_run.py                 # newest run under runs/
    python scripts/summarize_run.py runs/job123_... # a specific run
    python scripts/summarize_run.py --json          # machine-readable digest

The report answers, in order: did the run finish, what went wrong, where did the time go,
how well is the *library* pruning, how is the *agent* behaving, and what did evaluation
produce. The final section turns those measurements into concrete suggestions, which is the
part meant to drive the next iteration rather than merely describe the last one.

Standard library only: this has to run on the login node without activating an environment.
"""

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

# Thresholds that turn measurements into recommendations. Deliberately conservative: a flag
# should mean "look at this", not "this is definitely broken".
MASKED_FALLBACK_WARN = 0.10      # fraction of prune actions that could not be structural
ENTROPY_COLLAPSE_WARN = 0.30     # fraction of the maximum entropy for the action space
ACTION_DOMINANCE_WARN = 0.70     # one action chosen this often => little exploration
SLOW_STEP_WARN = 60.0            # seconds per RL step
ACC_DROP_WARN = 5.0              # percentage points of accuracy lost on average
CRITIC_LOSS_WARN = 1e6           # value-head targets this large indicate a reward-scale problem


def find_latest_run(root: str = "runs") -> Optional[str]:
    candidates = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
                  and os.path.isdir(os.path.join(d, "events"))]
    return max(candidates, key=os.path.getmtime) if candidates else None


def load_events(run_dir: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(run_dir, "events", "rank*.jsonl"))):
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # A run killed mid-write leaves a partial final line; keep the rest.
                    print(f"warning: {path}:{line_no} is not valid JSON, skipped", file=sys.stderr)
    events.sort(key=lambda e: (e.get("ts", ""), e.get("rank", 0)))
    return events


def load_manifest(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def by_event(events, name) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("event") == name]


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    if not rows:
        return ["_(none)_", ""]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    out.append("")
    return out


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def build_report(run_dir: str) -> Dict[str, Any]:
    """Compute every section's data once, so markdown and JSON stay consistent."""
    events = load_events(run_dir)
    manifest = load_manifest(run_dir)

    steps = by_event(events, "step")
    episodes = by_event(events, "episode")
    evals = by_event(events, "eval")
    issues = by_event(events, "issue")
    prunes = by_event(events, "prune")
    stages = by_event(events, "stage")
    crashes = by_event(events, "crash")
    run_end = by_event(events, "run_end")

    # --- status -------------------------------------------------------------
    if crashes:
        status = "crashed"
    elif run_end:
        status = run_end[-1].get("status", "unknown")
    elif events:
        status = "incomplete (no run_end record)"
    else:
        status = "no events recorded"

    # --- issues -------------------------------------------------------------
    issue_counts = Counter((i.get("kind"), i.get("detail", "")) for i in issues)
    failed_stages = Counter(s.get("stage") for s in stages if not s.get("ok", True))

    # --- timing -------------------------------------------------------------
    stage_times = defaultdict(list)
    for entry in stages:
        if entry.get("seconds") is not None:
            stage_times[entry.get("stage")].append(float(entry["seconds"]))
    timing = {name: _stats(values) for name, values in stage_times.items()}

    # --- pruning coverage ---------------------------------------------------
    prune_modes = Counter(p.get("mode") for p in prunes)
    blocked_reasons = Counter(p.get("reason") for p in prunes if p.get("mode") == "masked")
    masked_by_type = Counter(p.get("layer_type") for p in prunes if p.get("mode") == "masked")
    total_prunes = sum(prune_modes.values())
    masked_fraction = prune_modes.get("masked", 0) / total_prunes if total_prunes else 0.0

    # --- agent behaviour ----------------------------------------------------
    action_counts = Counter()
    for episode in episodes:
        action_counts.update(episode.get("actions") or [])
    rate_counts = Counter(s.get("compression_rate") for s in steps)

    # Actions the environment could not act on: the agent asked for compression but the layer
    # kept every channel, so the reward it received carries no information about the choice.
    noop_steps = [s for s in steps
                  if s.get("compression_rate") not in (None, 1, 1.0)
                  and s.get("old_width") is not None
                  and s.get("old_width") == s.get("new_width")]
    noop_fraction = len(noop_steps) / len(steps) if steps else 0.0

    critic_losses = [abs(e["critic_loss"]) for e in episodes if e.get("critic_loss") is not None]
    rewards = [s["reward"] for s in steps if s.get("reward") is not None]

    returns = [e["discounted_return"] for e in episodes if e.get("discounted_return") is not None]
    entropies = [e["entropy"] for e in episodes if e.get("entropy") is not None]
    step_seconds = [s["seconds"] for s in steps if s.get("seconds") is not None]

    def trend(values):
        if len(values) < 6:
            return None
        third = max(len(values) // 3, 1)
        return {"first_third": statistics.fmean(values[:third]),
                "last_third": statistics.fmean(values[-third:])}

    # --- evaluation ---------------------------------------------------------
    eval_rows = []
    for entry in evals:
        eval_rows.append({
            "network": os.path.basename(str(entry.get("network", "?"))),
            "mode": entry.get("eval_mode"),
            "pass": entry.get("pass_index"),
            "origin_acc": entry.get("origin_acc"),
            "new_acc": entry.get("new_acc"),
            "delta_acc": entry.get("delta_acc"),
            "param_ratio": entry.get("param_ratio"),
            "flops_ratio": entry.get("flops_ratio"),
        })
    acc_deltas = [r["delta_acc"] for r in eval_rows if r["delta_acc"] is not None]
    param_ratios = [r["param_ratio"] for r in eval_rows if r["param_ratio"] is not None]
    flops_ratios = [r["flops_ratio"] for r in eval_rows if r["flops_ratio"] is not None]

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "status": status,
        "counts": {
            "events": len(events), "episodes": len(episodes), "steps": len(steps),
            "evaluations": len(evals), "issues": len(issues), "prune_actions": total_prunes,
        },
        "run_end": run_end[-1] if run_end else {},
        "crashes": crashes,
        "issue_counts": issue_counts,
        "failed_stages": failed_stages,
        "timing": timing,
        "prune_modes": prune_modes,
        "masked_fraction": masked_fraction,
        "blocked_reasons": blocked_reasons,
        "masked_by_type": masked_by_type,
        "action_counts": action_counts,
        "rate_counts": rate_counts,
        "noop_fraction": noop_fraction,
        "noop_steps": len(noop_steps),
        "critic_losses": critic_losses,
        "rewards": rewards,
        "returns": returns,
        "return_trend": trend(returns),
        "entropies": entropies,
        "entropy_trend": trend(entropies),
        "step_seconds": step_seconds,
        "eval_rows": eval_rows,
        "acc_deltas": acc_deltas,
        "param_ratios": param_ratios,
        "flops_ratios": flops_ratios,
    }


def recommendations(report: Dict[str, Any]) -> List[str]:
    """Rule-based reading of the measurements: what to fix next, and why."""
    notes = []

    if report["status"] not in ("ok",):
        notes.append(f"**Run did not complete cleanly** (status: `{report['status']}`). "
                     f"Start from `logs/errors.log`; every failure there carries a traceback.")

    if report["failed_stages"]:
        worst = ", ".join(f"`{k}` x{v}" for k, v in report["failed_stages"].most_common(5))
        notes.append(f"**Failing stages:** {worst}. These raised inside a timed block, so the "
                     f"traceback is in the rank log at the matching timestamp.")

    masked = report["masked_fraction"]
    if report["prune_modes"] and masked > MASKED_FALLBACK_WARN:
        top = ", ".join(f"`{r}` x{c}" for r, c in report["blocked_reasons"].most_common(3))
        notes.append(
            f"**Library gap: {masked:.0%} of compression actions fell back to masking** "
            f"(reasons: {top}). Masked layers shed no FLOPs, so the reward is measuring an "
            f"optimisation that did not physically happen. Extend `src/channel_groups.py` to "
            f"resize these patterns before drawing conclusions about the agent.")
    elif report["prune_modes"]:
        notes.append(f"Structured pruning succeeded on {1 - masked:.0%} of actions "
                     f"({report['prune_modes'].get('structural', 0)}/{sum(report['prune_modes'].values())}).")

    if report["noop_steps"]:
        notes.append(
            f"**{report['noop_steps']} steps ({report['noop_fraction']:.0%}) requested "
            f"compression but changed no channel.** The reward for those steps reflects damage "
            f"done by earlier actions, not the action taken, so they inject noise into credit "
            f"assignment. Check the width-rounding rule in `src/pruning.py::target_width`.")

    critic_losses = report["critic_losses"]
    if critic_losses and statistics.fmean(critic_losses) > CRITIC_LOSS_WARN:
        reward_span = (f"rewards span {min(report['rewards']):.0f}..{max(report['rewards']):.0f}"
                       if report["rewards"] else "")
        notes.append(
            f"**Critic loss averages {statistics.fmean(critic_losses):.3g}** ({reward_span}). "
            f"`compute_reward` cubes a percentage-scale term, so a single step can be worth "
            f"1e5 while the value head is initialised near 0. Gradient clipping bounds the "
            f"step size but not the target, so the critic spends the run chasing scale rather "
            f"than ranking actions. Consider normalising or log-scaling the reward.")

    actions = report["action_counts"]
    if actions:
        total = sum(actions.values())
        top_action, top_count = actions.most_common(1)[0]
        if top_count / total > ACTION_DOMINANCE_WARN:
            notes.append(
                f"**Agent gap: action {top_action} chosen {top_count / total:.0%} of the time.** "
                f"The policy is close to constant, so the state representation may not be "
                f"separating layers. Check the entropy bonus (`ENTROPY_COEF`) and whether the "
                f"action-cost features vary across layers.")

    entropies = report["entropies"]
    if entropies and actions:
        import math
        max_entropy = math.log(max(len(actions), 2))
        if statistics.fmean(entropies[-max(len(entropies) // 3, 1):]) < ENTROPY_COLLAPSE_WARN * max_entropy:
            notes.append("**Policy entropy collapsed** in the final third of training. "
                         "Either the agent converged or exploration died too early; compare "
                         "against the reward trend before deciding.")

    trend = report["return_trend"]
    if trend:
        change = trend["last_third"] - trend["first_third"]
        direction = "improving" if change > 0 else "not improving"
        notes.append(f"Discounted return is **{direction}**: "
                     f"{trend['first_third']:.1f} (first third) -> {trend['last_third']:.1f} (last third).")

    step_seconds = report["step_seconds"]
    if step_seconds:
        mean_step = statistics.fmean(step_seconds)
        if mean_step > SLOW_STEP_WARN:
            slowest = sorted(report["timing"].items(),
                             key=lambda kv: kv[1].get("total", 0), reverse=True)[:3]
            breakdown = ", ".join(f"`{name}` {s['total'] / 60:.1f} min" for name, s in slowest)
            notes.append(f"**Throughput: {mean_step:.0f}s per RL step.** Dominant stages: "
                         f"{breakdown}. A full experiment needs far more episodes, so this is "
                         f"the ceiling on how much data the agent can see.")

    deltas = report["acc_deltas"]
    if deltas:
        mean_delta = statistics.fmean(deltas)
        mean_param = statistics.fmean(report["param_ratios"]) if report["param_ratios"] else None
        mean_flops = statistics.fmean(report["flops_ratios"]) if report["flops_ratios"] else None
        summary = (f"Evaluation: mean accuracy change **{mean_delta:+.2f} pp**"
                   + (f", parameters x{mean_param:.3f}" if mean_param else "")
                   + (f", FLOPs x{mean_flops:.3f}" if mean_flops else "") + ".")
        if mean_delta < -ACC_DROP_WARN:
            summary += (" That exceeds the accuracy budget; revisit `allowed_acc_reduction` "
                        "or the fine-tuning epochs after pruning.")
        notes.append(summary)

    if not report["counts"]["evaluations"]:
        notes.append("No evaluation records: the run stopped before (or during) the evaluation "
                     "phase, so compression quality is still unmeasured.")

    return notes or ["No anomalies detected in the recorded measurements."]


def render_markdown(report: Dict[str, Any]) -> str:
    manifest = report["manifest"]
    lines: List[str] = []
    add = lines.append

    add(f"# SPECTRA run summary — `{os.path.basename(report['run_dir'])}`")
    add("")
    add(f"**Status:** {report['status']}")
    add("")

    overview = [
        ["run id", manifest.get("run_id", "-")],
        ["host", manifest.get("hostname", "-")],
        ["slurm job", manifest.get("slurm_job_id", "-")],
        ["git commit", (manifest.get("git_commit") or "-")[:12]
         + (" (dirty)" if manifest.get("git_dirty") else "")],
        ["torch", f"{manifest.get('torch', '-')} / cuda {manifest.get('cuda_build', '-')}"],
        ["gpus", ", ".join(manifest.get("gpus", [])) or "-"],
        ["started", manifest.get("started_at", "-")],
        ["duration", f"{report['run_end'].get('seconds', 0) / 60:.1f} min"
         if report["run_end"].get("seconds") else "-"],
    ]
    add("## Overview")
    add("")
    lines.extend(_table(["field", "value"], overview))

    counts = report["counts"]
    lines.extend(_table(["episodes", "steps", "evaluations", "prune actions", "issues", "events"],
                        [[counts["episodes"], counts["steps"], counts["evaluations"],
                          counts["prune_actions"], counts["issues"], counts["events"]]]))

    add("## Problems")
    add("")
    if report["crashes"]:
        add("**Crash:** `" + str(report["crashes"][-1].get("error")) + "`")
        add("")
    rows = [[kind, detail[:90] if detail else "-", count]
            for (kind, detail), count in report["issue_counts"].most_common(15)]
    lines.extend(_table(["issue", "detail", "count"], rows))

    add("## Where the time went")
    add("")
    timing_rows = []
    for name, stats in sorted(report["timing"].items(), key=lambda kv: kv[1].get("total", 0),
                              reverse=True)[:15]:
        timing_rows.append([f"`{name}`", stats["count"], f"{stats['total'] / 60:.2f}",
                            _fmt(stats["mean"], 2), _fmt(stats["p95"], 2), _fmt(stats["max"], 2)])
    lines.extend(_table(["stage", "n", "total (min)", "mean (s)", "p95 (s)", "max (s)"], timing_rows))

    add("## Pruning coverage (library health)")
    add("")
    mode_rows = [[mode or "-", count, f"{count / max(sum(report['prune_modes'].values()), 1):.1%}"]
                 for mode, count in report["prune_modes"].most_common()]
    lines.extend(_table(["mode", "count", "share"], mode_rows))
    if report["blocked_reasons"]:
        add("Reasons structural pruning was not possible:")
        add("")
        lines.extend(_table(["reason", "count"],
                            [[r or "-", c] for r, c in report["blocked_reasons"].most_common(10)]))
    if report["masked_by_type"]:
        lines.extend(_table(["masked layer type", "count"],
                            [[t or "-", c] for t, c in report["masked_by_type"].most_common(10)]))

    add("## Agent behaviour")
    add("")
    if report["action_counts"]:
        total_actions = sum(report["action_counts"].values())
        lines.extend(_table(["action index", "count", "share"],
                            [[a, c, f"{c / total_actions:.1%}"]
                             for a, c in sorted(report["action_counts"].items())]))
    if report["rate_counts"]:
        total_rates = sum(report["rate_counts"].values())
        lines.extend(_table(["compression rate", "count", "share"],
                            [[r, c, f"{c / total_rates:.1%}"]
                             for r, c in sorted(report["rate_counts"].items(),
                                                key=lambda kv: (kv[0] is None, kv[0]))]))
    learning_rows = []
    if report["returns"]:
        stats = _stats(report["returns"])
        learning_rows.append(["discounted return", _fmt(stats["mean"], 2), _fmt(stats["max"], 2)])
    if report["entropies"]:
        stats = _stats(report["entropies"])
        learning_rows.append(["policy entropy", _fmt(stats["mean"], 4), _fmt(stats["max"], 4)])
    if report["step_seconds"]:
        stats = _stats(report["step_seconds"])
        learning_rows.append(["seconds / step", _fmt(stats["mean"], 1), _fmt(stats["max"], 1)])
    lines.extend(_table(["metric", "mean", "max"], learning_rows))

    add("## Evaluation results")
    add("")
    eval_rows = [[r["network"][:48], r["mode"], r["pass"], _fmt(r["origin_acc"]),
                  _fmt(r["new_acc"]), _fmt(r["delta_acc"]), _fmt(r["param_ratio"]),
                  _fmt(r["flops_ratio"])] for r in report["eval_rows"][:40]]
    lines.extend(_table(["network", "mode", "pass", "orig acc", "new acc", "delta",
                         "param x", "flops x"], eval_rows))

    add("## What to do next")
    add("")
    for note in recommendations(report):
        add(f"- {note}")
    add("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", help="Run directory (default: newest under runs/)")
    parser.add_argument("--root", default="runs", help="Where to look for runs")
    parser.add_argument("--json", action="store_true", help="Emit a JSON digest instead")
    parser.add_argument("--no-write", action="store_true", help="Do not write summary.md")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run(args.root)
    if not run_dir or not os.path.isdir(run_dir):
        print(f"No run directory found (looked under '{args.root}').", file=sys.stderr)
        return 1

    report = build_report(run_dir)

    if args.json:
        digest = {
            "run_dir": report["run_dir"], "status": report["status"],
            "counts": report["counts"], "masked_fraction": report["masked_fraction"],
            "issues": {f"{k}|{d}": c for (k, d), c in report["issue_counts"].items()},
            "recommendations": recommendations(report),
        }
        print(json.dumps(digest, indent=2))
        return 0

    markdown = render_markdown(report)
    print(markdown)
    if not args.no_write:
        try:
            with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as handle:
                handle.write(markdown)
        except Exception as exc:
            print(f"warning: could not write summary.md ({exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
