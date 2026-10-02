"""CPU tests for S2 (scripts/selection_probe_s2.py): the nap_f hook, the paired seed plan and the G2 readout.

    python -m pytest tests/test_selection_probe_s2.py -v
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tests.test_selection_probe import _batches, _model  # noqa: E402
from tests.test_selection_scorer_s1 import _cell  # noqa: E402

import selection_probe as probe  # noqa: E402
import selection_probe_s2 as s2  # noqa: E402
import selection_scorer_s1 as s1  # noqa: E402

pytest.importorskip("sklearn")


@pytest.fixture
def bundle(tmp_path):
    runs = [_cell(tmp_path, f"n{i}", i) for i in range(3)]
    path = tmp_path / "nap_f.pkl"
    s1.export_model([s1.load_cell(r) for r in runs], str(path))
    with open(path, "rb") as fh:
        return pickle.load(fh)


def test_install_adds_nap_f_pairs_seeds_and_restores(bundle):
    original = (probe.CRITERIA, probe.channel_features, probe.mask_specs)
    restore = s2.install(bundle, oracle_seeds=3)
    try:
        assert "nap_f" in probe.CRITERIA
        specs = probe.mask_specs(["l1", "nap_f", "ablation", "random", "anti_l1"], 3, 5)
        assert [s for s in specs if s[0] == "l1"] == [("l1", 0, s) for s in range(5)]
        assert [s for s in specs if s[0] == "nap_f"] == [("nap_f", 0, s) for s in range(5)]
        assert [s for s in specs if s[0] == "ablation"] == [("ablation", 0, s) for s in range(3)]
        assert [s for s in specs if s[0] == "random"] == [("random", s, s) for s in range(3)]
        assert [s for s in specs if s[0] == "anti_l1"] == [("anti_l1", 0, 0)]
    finally:
        restore()
    assert (probe.CRITERIA, probe.channel_features, probe.mask_specs) == original


def test_nap_f_ranks_live_channels_by_the_scorer_and_cuts_the_same_shape(bundle):
    model = _model(5)
    plan = probe.cut_plan(model)
    batches = _batches(seed=9)
    scores, _ = probe.all_scores(model, plan, batches)
    grads = probe.mean_gradients(model, plan, batches)
    restore = s2.install(bundle)
    try:
        columns, table = probe.channel_features(plan, grads, scores)
    finally:
        restore()
    assert set(scores["nap_f"]) == {key for key, _, _ in plan}
    per_group = s1.score_net(bundle, columns, table)
    assert per_group, "the scorer ranked no group"
    alive = {key: scores["l1"][key] > 0 for key, _, _ in plan}
    for gi, (key, group, _) in enumerate(plan):
        values = scores["nap_f"][key]
        assert values.shape == (group.width,) and torch.isfinite(values).all()
        if gi in per_group:
            rows, pred = per_group[gi]
            channels = table[rows, columns.index("channel")].astype(int)
            np.testing.assert_allclose(values[channels].numpy(), pred)
    shapes = []
    with probe.RankingOverride() as override:
        for spec in (("l1", 0, 0), ("nap_f", 0, 0)):
            tables, fallbacks = probe.score_tables(spec, scores, alive, plan)
            assert fallbacks == 0
            cut_model, _, _ = probe.cut(model, plan, 0.5, tables, override, (3, 32, 32))
            shapes.append(probe.shape_signature(cut_model))
    assert shapes[0] == shapes[1]


def _run(tmp_path, name, nap_f, l1=(0.0, 0.4, -0.4, 0.2, -0.2), budgets=("bn", "40"), oracle=0.0):
    res = tmp_path / name / "results"
    res.mkdir(parents=True)
    rows = []
    for b in budgets:
        for s, v in enumerate(l1):
            rows.append({"net": name, "keep": 0.6, "criterion": "l1", "mask_seed": 0, "ft_seed": s, "budget": b,
                         "d_val_pp": v})
            rows.append({"net": name, "keep": 0.6, "criterion": "nap_f", "mask_seed": 0, "ft_seed": s, "budget": b,
                         "d_val_pp": v + nap_f[b]})
        for s in range(3):
            rows.append({"net": name, "keep": 0.6, "criterion": "ablation", "mask_seed": 0, "ft_seed": s,
                         "budget": b, "d_val_pp": l1[s] + oracle})
            rows.append({"net": name, "keep": 0.6, "criterion": "random", "mask_seed": s, "ft_seed": s, "budget": b,
                         "d_val_pp": l1[s] - 1.0})
        rows.append({"net": name, "keep": 0.6, "criterion": "anti_l1", "mask_seed": 0, "ft_seed": 0, "budget": b,
                     "d_val_pp": -5.0})
    with open(res / "selection_probe.jsonl", "w", encoding="utf-8") as fh:
        for r in rows + [{"kind": "summary", "net": name}]:
            fh.write(json.dumps(r) + "\n")
    return str(tmp_path / name)


@pytest.mark.parametrize("gain,call", [({"bn": 0.0, "40": 0.8}, "PASS"), ({"bn": 0.0, "40": 0.1}, "FAIL"),
                                       ({"bn": 0.9, "40": 0.0}, "CHEAP-FT"), ({"bn": 0.9, "40": -0.5}, "HARM")])
def test_g2_calls(tmp_path, gain, call):
    runs = [_run(tmp_path, "a", gain), _run(tmp_path, "b", gain, oracle=0.5)]
    cells, calls = s2.readout(runs)
    assert calls["call"] == call
    mean, se, n, sd = cells["a"]["40"]["nap_f"]
    assert n == 5 and mean == pytest.approx(gain["40"]) and sd == pytest.approx(np.std([0, .4, -.4, .2, -.2], ddof=1))
    assert cells["b"]["40"]["ablation"][0] == pytest.approx(0.5) and cells["b"]["40"]["ablation"][2] == 3
    assert cells["a"]["40"]["random"][0] == pytest.approx(-1.0) and cells["a"]["40"]["random"][2] == 3
    assert "G2 call" in s2.render_readout(cells, calls)
