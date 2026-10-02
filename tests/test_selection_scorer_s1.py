"""CPU tests for the S1 learned NAP-F scorer (scripts/selection_scorer_s1.py).

    python -m pytest tests/test_selection_scorer_s1.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import selection_probe as probe  # noqa: E402
import selection_scorer_s1 as s1  # noqa: E402

pytest.importorskip("sklearn")

COLUMNS = (["group", "channel", "depth", "width", "n_producers"] + list(probe.WEIGHT_CRITERIA + probe.DATA_CRITERIA)
           + ["out_l1"] + [f"w_{s}" for s in probe.NAP_STATS] + [f"g_{s}" for s in probe.NAP_STATS])


def _cell(tmp_path, net, seed, groups=6, width=24, oracle=None, printed=None):
    """An S0-shaped run dir; the oracle is ``oracle(columns_dict)`` (default: noisy L1)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(groups):
        for j in range(width):
            rows.append([g, j, g / max(1, groups - 1), width, 1 + (g % 2)] + list(rng.normal(size=len(COLUMNS) - 5)))
    data = np.asarray(rows, dtype=float)
    col = {c: data[:, i] for i, c in enumerate(COLUMNS)}
    data[:, COLUMNS.index("l1")] = np.abs(col["l1"]) + 0.1
    data[:, COLUMNS.index("l1")][0] = 0.0                         # one dead channel: excluded like S0's alive
    col = {c: data[:, i] for i, c in enumerate(COLUMNS)}
    abl = oracle(col, rng) if oracle else col["l1"] + 0.3 * rng.normal(size=len(data))
    data[:, COLUMNS.index("ablation")] = abl
    res = tmp_path / net / "results"
    res.mkdir(parents=True)
    np.savez_compressed(res / "selection_features.npz", data=data, columns=np.array(COLUMNS),
                        groups=np.array([f"g{g}" for g in range(groups)]), net=np.array(net), dataset=np.array("x"))
    cell = s1.load_cell(str(tmp_path / net))
    tau = {c: round(s1.tau_groups(s1.column(cell, c), s1.column(cell, "ablation"), s1.groups_of(cell)), 4)
           for c in s1.HAND}
    summary = {"kind": "summary", "net": net, "tau_vs_ablation": printed if printed is not None else tau}
    lines = []
    for crit, val in (("l1", 0.0), ("l1", 0.2), ("l1", -0.2), ("l2", 0.1), ("fpgm", 5.0), ("random", -1.0),
                      ("random", -1.4), ("anti_l1", -3.0), ("ablation", -0.5)):
        lines.append({"net": net, "keep": 0.6, "criterion": crit, "budget": "40", "d_val_pp": val, "d_test_pp": val})
    with open(res / "selection_probe.jsonl", "w", encoding="utf-8") as fh:
        for rec in lines + [summary]:
            fh.write(json.dumps(rec) + "\n")
    return str(tmp_path / net)


def test_tau_matches_s0_agreement():
    rng = np.random.default_rng(0)
    widths = (8, 12, 5)
    keys = [(f"g{i}",) for i in range(len(widths))]
    scores = {c: {k: torch.tensor(rng.normal(size=w)) for k, w in zip(keys, widths)} for c in ("ablation", "svd")}
    scores["l1"] = {k: torch.tensor(np.abs(rng.normal(size=w))) for k, w in zip(keys, widths)}
    scores["l1"][keys[0]][[1, 4]] = 0.0
    alive = {k: scores["l1"][k] > 0 for k in keys}
    plan = [(k, None, None) for k in keys]
    want = probe.agreement(scores, alive, plan, "ablation")
    flat = {c: np.concatenate([scores[c][k].numpy() for k in keys]) for c in scores}
    gid = np.concatenate([np.full(w, i) for i, w in enumerate(widths)])
    groups = [np.where((gid == i) & (flat["l1"] > 0))[0] for i in range(len(widths))]
    for crit in ("l1", "svd"):
        assert s1.tau_groups(flat[crit], flat["ablation"], groups) == pytest.approx(want[crit], abs=1e-4)


def test_rank01_ties_and_nan():
    r = s1._rank01(np.array([3.0, 1.0, 3.0, np.nan, 2.0]))
    assert r[1] == 0.0 and r[4] == pytest.approx(1 / 3) and r[0] == r[2] == pytest.approx(5 / 6) and r[3] == 0.5


def test_learned_beats_single_criteria_when_oracle_needs_two(tmp_path):
    def oracle(col, rng):  # no single column carries it
        return np.argsort(np.argsort(col["l1"])) + np.argsort(np.argsort(col["g_std"])) + 0.1 * rng.normal(size=len(col["l1"]))

    runs = [_cell(tmp_path, f"n{i}", i, oracle=oracle) for i in range(3)]
    result = s1.evaluate([s1.load_cell(r) for r in runs])
    assert result["reproduced"] == 3 * len(s1.HAND)
    for fold in result["folds"]:
        assert fold["held_out"] not in fold["train"]
        assert fold["tau"] > fold["best_hand_tau"] + 0.05
    assert result["g1"] and result["g1_passes"] == 3
    assert "PASS" in s1.render(result)


def test_reproduction_mismatch_aborts(tmp_path):
    runs = [_cell(tmp_path, "a", 0), _cell(tmp_path, "b", 1),
            _cell(tmp_path, "c", 2, printed={c: 0.99 for c in s1.HAND})]
    with pytest.raises(SystemExit, match="does not reproduce"):
        s1.evaluate([s1.load_cell(r) for r in runs])


def test_m8_null(tmp_path):
    cell = s1.load_cell(_cell(tmp_path, "m", 0))
    out = s1.m8_reread(cell)
    (b,) = out["budgets"]
    assert out["sigma_ft_pooled"] == pytest.approx(0.2)
    assert b["best_named"] == "fpgm" and b["best_minus_l1"] == pytest.approx(5.0)
    assert b["p_null_max"] < 0.01                                  # 25 sigma above L1
    assert b["l1_minus_random"] == pytest.approx(1.2) and b["l1_minus_anti"] == pytest.approx(3.0)
    assert b["ablation_minus_l1"] == pytest.approx(-0.5)
