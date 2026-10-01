#!/usr/bin/env python
"""
Proxy-fidelity readout: ``proxy_fidelity`` records of ``SPECTRA_EVAL_PROXY_FIDELITY`` walks (CPU).

    python scripts/proxy_fidelity_readout.py <run_dir | events.jsonl> [...]

Per state (net x target) and candidate set: Spearman rho between each proxy's **val** delta (what the
agent's reward sees) and the final fine-tune's **TEST** delta (what the thesis quotes; mean over the
seeds), the noise ceiling rho(final seed a, final seed b) on TEST, and the top-1 regret (TEST pp the
proxy's favourite gives up against the set's best final). Sets: ``crit`` = the five criteria at keep 0.8
on one row; ``where`` = the same share of the network cut from different groups; ``menu`` = identity +
the Stage-4 cut actions (mixed sizes, so it is read as the depth penalty, not as a ranking).

The calls at the end are the ones registered in ``docs/SITTING_GPU_QUEUE.md`` before the run.
"""
import argparse
import glob
import json
import os
import statistics

RHO_BAR = 0.6          # a proxy is valid if mean rho >= max(RHO_BAR, CEIL_FRAC x ceiling) ...
CEIL_FRAC = 0.8
REGRET_BAR_PP = 0.5    # ... and its median top-1 regret is at most this
CEIL_MIN = 0.5         # below this the final fine-tune does not rank the candidates consistently
GAIN_40 = 0.2          # 40x10 over 12x4 by this much releases the held 40/10 train (21940321)
MIN_ITEMS = 3


def event_files(target):
    if os.path.isfile(target):
        return [target]
    return sorted(glob.glob(os.path.join(target, "events", "rank*.jsonl")))


def load(targets):
    rows = []
    for target in targets:
        for path in event_files(target):
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if '"proxy_fidelity"' not in line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("event") == "proxy_fidelity" and not event.get("duplicate_of"):
                        rows.append(event)
    return rows


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0
        i = j + 1
    return out


def spearman(a, b):
    if len(a) < 2:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def proxy_pp(rec, name):
    return (rec["scores"][name]["val"] - rec["val_origin"]) * 100.0


def final_pp(rec, seed=None):
    finals = rec["finals"]
    keys = [seed] if seed is not None else list(finals)
    return statistics.mean((finals[k]["test"] - rec["test_origin"]) * 100.0 for k in keys)


def candidate_sets(recs):
    """``{"crit": [...], "where": [...], "menu": [...]}`` for one state."""
    menu_row = next((r["row"] for r in recs if r["kind"] in ("menu", "crit")), None)
    crit = [r for r in recs if r["kind"] in ("menu", "crit") and abs(r["rate"] - 0.8) < 1e-9
            and r["row"] == menu_row]
    anchor = [r for r in recs if r["kind"] == "menu" and abs(r["rate"] - 0.8) < 1e-9 and r["ranking"] == "l1"]
    where = anchor + [r for r in recs if r["kind"] == "where"]
    menu = [r for r in recs if r["kind"] in ("identity", "menu")]
    return {"crit": crit, "where": where, "menu": menu}


def nan(x):
    return x != x


def summarize(rows):
    states = {}
    for r in rows:
        states.setdefault((os.path.basename(r["network"]), r["target"]), []).append(r)
    proxies = sorted({k for r in rows for k in r["scores"]}, key=lambda k: (k not in ("none", "bn"), k))
    seeds = sorted({k for r in rows for k in r["finals"]})
    table, depth = [], []
    for (net, target), recs in sorted(states.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        for set_name, items in candidate_sets(recs).items():
            if set_name == "menu" or len(items) < MIN_ITEMS:
                continue
            fin = [final_pp(r) for r in items]
            ceiling = (spearman([final_pp(r, seeds[0]) for r in items], [final_pp(r, seeds[1]) for r in items])
                       if len(seeds) >= 2 else float("nan"))
            best = max(fin)
            row = {"net": net, "target": target, "set": set_name, "n": len(items), "ceiling": ceiling,
                   "spread": best - min(fin), "rho": {}, "regret": {}}
            for p in proxies:
                prox = [proxy_pp(r, p) for r in items]
                row["rho"][p] = spearman(prox, fin)
                row["regret"][p] = best - fin[max(range(len(items)), key=lambda i: prox[i])]
            table.append(row)
        by_rank = {}
        for r in recs:
            if r["kind"] == "menu":
                by_rank.setdefault(r["ranking"], {})[round(r["rate"], 3)] = r
        for ranking, pair in by_rank.items():
            if 0.9 in pair and 0.8 in pair:
                d_final = final_pp(pair[0.9]) - final_pp(pair[0.8])
                depth.append({"net": net, "target": target, "ranking": ranking, "final": d_final,
                              "proxy": {p: proxy_pp(pair[0.9], p) - proxy_pp(pair[0.8], p) for p in proxies}})
    return table, depth, proxies, seeds


def mean_or_nan(xs):
    xs = [x for x in xs if not nan(x)]
    return statistics.mean(xs) if xs else float("nan")


def calls(table, proxies):
    """The registered calls, as text lines."""
    if not table:
        return ["no ranked set with >= 3 distinct candidates yet"]
    ceiling = mean_or_nan([t["ceiling"] for t in table])
    rho = {p: mean_or_nan([t["rho"][p] for t in table]) for p in proxies}
    regret = {p: statistics.median([t["regret"][p] for t in table]) for p in proxies}
    bar = max(RHO_BAR, CEIL_FRAC * ceiling) if not nan(ceiling) else RHO_BAR
    valid = {p: (not nan(rho[p])) and rho[p] >= bar and regret[p] <= REGRET_BAR_PP for p in proxies}
    out = [f"mean ceiling rho {ceiling:+.2f} over {len(table)} ranked sets; bar max({RHO_BAR}, {CEIL_FRAC} x ceiling) "
           f"= {bar:.2f}, median regret <= {REGRET_BAR_PP} pp"]
    for p in proxies:
        out.append(f"  {p}: mean rho {rho[p]:+.2f}, median regret {regret[p]:.2f} pp -> "
                   f"{'VALID' if valid[p] else 'not valid'}")
    if not nan(ceiling) and ceiling < CEIL_MIN:
        out.append(f"UNINFORMATIVE: ceiling {ceiling:+.2f} < {CEIL_MIN}; the final fine-tune does not rank these "
                   f"candidates consistently. Widen the cuts before reading any proxy.")
        return out
    short, long_ = "12x4", "40x10"
    if short in valid:
        out.append(f"12x4 (the agent's training budget): {'VALIDATED as the proxy' if valid[short] else 'NOT validated'}")
    if short in valid and long_ in valid and not valid[short]:
        gain = rho[long_] - rho[short]
        release = valid[long_] or gain >= GAIN_40
        out.append(f"40x10 vs 12x4: rho {gain:+.2f} -> "
                   f"{'RELEASE the held 40/10 train (21940321) on Ido' if release else 'no release'}")
    if "bn" in valid and short in rho and not nan(rho["bn"]) and not nan(rho[short]):
        if valid["bn"] and rho["bn"] >= rho[short] - 0.1:
            out.append("bn (re-estimated BN, no fine-tune) ranks within 0.1 of 12x4: a cheap-proxy train arm is a "
                       "sitting question")
    if short in valid and not valid[short] and not valid.get(long_, False):
        out.append("neither fine-tune budget is valid: compare SGD variants on this agreement next (same saved finals)")
    return out


def render(table, depth, proxies, seeds):
    lines = ["| net | target | set | n | final spread pp | ceiling rho | "
             + " | ".join(f"rho {p}" for p in proxies) + " | " + " | ".join(f"regret {p}" for p in proxies) + " |",
             "|---|---|---|---|---|---|" + "---|" * (2 * len(proxies))]
    for t in table:
        lines.append(f"| {t['net'][:34]} | {t['target']:g} | {t['set']} | {t['n']} | {t['spread']:.2f} | "
                     f"{t['ceiling']:+.2f} | " + " | ".join(f"{t['rho'][p]:+.2f}" for p in proxies) + " | "
                     + " | ".join(f"{t['regret'][p]:.2f}" for p in proxies) + " |")
    lines.append("")
    lines.append(f"final seeds: {', '.join(seeds)}. Depth penalty = delta(keep 0.9) - delta(keep 0.8), pp "
                 f"(proxy on val, final on TEST):")
    for d in depth:
        lines.append(f"- {d['net'][:34]} t={d['target']:g} {d['ranking']}: final {d['final']:+.2f}; "
                     + ", ".join(f"{p} {v:+.2f}" for p, v in d["proxy"].items()))
    usable = [d for d in depth if abs(d["final"]) >= 0.2]
    for p in proxies:
        ratios = [d["proxy"][p] / d["final"] for d in usable]
        if ratios:
            lines.append(f"- {p}: median depth penalty {statistics.median(ratios):.2f} x the final fine-tune's "
                         f"({len(ratios)} pairs with |final| >= 0.2 pp)")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("targets", nargs="+", help="run directories or events jsonl files")
    args = parser.parse_args(argv)
    rows = load(args.targets)
    if not rows:
        print("no proxy_fidelity records")
        return 1
    table, depth, proxies, seeds = summarize(rows)
    print(render(table, depth, proxies, seeds))
    print()
    print("\n".join(calls(table, proxies)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
