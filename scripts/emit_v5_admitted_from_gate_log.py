#!/usr/bin/env python3
"""Fill the P5-B3 CIFAR-100 gate table from a completed mild-walk log and emit the admitted catalog.

Used so layer-replacement DRL trains do not start on the C10-only placeholder.

    python scripts/emit_v5_admitted_from_gate_log.py \\
        --log /path/to/spectra_<GATEJOB>.out \\
        --write

Admit rule (configs/v5_p5b3_c100_gate.json): TRAJ val_best kept <= 0.98 and
val Δacc >= -10. Exit 0 if at least two CIFAR-100 nets are admitted; exit 2
if fewer (do not start a one-net C100 mix). Exit 1 on a missing log.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v5_catalog import (  # noqa: E402
    ADMITTED_PATH,
    GATE_PATH,
    INTENDED_PATH,
    admitted_catalog,
    load_gate,
)

VAL_BEST = re.compile(
    r"\[eval\] TRAJ val_best (?P<net>\S+) step=\d+ \| acc [0-9.]+ -> [0-9.]+ \([^)]+\) "
    r"\| params x(?P<kept>[0-9.]+) \| FLOPs x[0-9.]+ \| val Δacc (?P<dacc>[+-]?[0-9.]+) pp"
)


def parse_val_best(text: str) -> dict[str, dict]:
    out = {}
    for m in VAL_BEST.finditer(text):
        name = Path(m.group("net")).name
        out[name] = {
            "val_best_kept": float(m.group("kept")),
            "val_best_delta_pp": float(m.group("dacc")),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--write", action="store_true")
    p.add_argument("--min-admitted", type=int, default=2)
    args = p.parse_args()
    if not args.log.is_file():
        print(f"missing gate log: {args.log}", file=sys.stderr)
        return 1
    text = args.log.read_text(encoding="utf-8", errors="replace")
    hits = parse_val_best(text)
    if not hits:
        print(f"no TRAJ val_best lines in {args.log}", file=sys.stderr)
        return 1
    gate = load_gate()
    c100 = gate.setdefault("c100", {})
    n_admitted = 0
    for name, row in c100.items():
        rec = hits.get(name)
        if rec is None:
            # log basename may be a prefix of the catalog filename
            rec = next((v for k, v in hits.items() if k in name or name in k), None)
        if rec is None:
            row["status"] = "rejected"
            row["probe_job"] = str(args.log)
            row["why_rejected"] = "no TRAJ val_best in gate log"
            print(f"REJECT {name} (no val_best)")
            continue
        kept = rec["val_best_kept"]
        dacc = rec["val_best_delta_pp"]
        ok = kept <= 0.98 and dacc >= -10.0
        row["status"] = "admitted" if ok else "rejected"
        row["probe_job"] = str(args.log)
        row["val_best_kept"] = kept
        row["val_best_delta_pp"] = dacc
        if ok:
            n_admitted += 1
            print(f"ADMIT  {name}  kept={kept:.3f}  val_dacc={dacc:+.2f}")
        else:
            print(f"REJECT {name}  kept={kept:.3f}  val_dacc={dacc:+.2f}")
    gate["stamped"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M UTC (emit_v5_admitted_from_gate_log)")
    intended = json.loads(INTENDED_PATH.read_text(encoding="utf-8"))
    admitted = admitted_catalog(intended, gate)
    n100 = sum(1 for r in admitted.values() if (r[2] if isinstance(r[2], str) else r[2].get("name")) == "cifar-100")
    print(f"admitted catalog n={len(admitted)} cifar-100={n100}")
    if args.write:
        GATE_PATH.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        ADMITTED_PATH.write_text(json.dumps(admitted, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {GATE_PATH}")
        print(f"wrote {ADMITTED_PATH}")
    if n_admitted < args.min_admitted:
        print(f"only {n_admitted} CIFAR-100 net(s) admitted (need {args.min_admitted}); "
              "do not start the mixed train", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
