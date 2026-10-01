"""CPU tests for the selection headroom probe (scripts/selection_probe.py).

    python -m pytest tests/test_selection_probe.py -v

The NAPv2 cross-check runs when a NAPv2 checkout is found (NAPV2_DIR, default
~/scratch_audit/third_party/NAPv2) and is skipped otherwise.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

import selection_probe as probe  # noqa: E402
import src.pruning as pruning  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402

CPU = torch.device("cpu")
NAP_DIR = os.environ.get("NAPV2_DIR", str(Path.home() / "scratch_audit" / "third_party" / "NAPv2"))


def _model(seed=0):
    torch.manual_seed(seed)
    model = resnet20(10, False, 4).eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.1, 0.1)
                m.running_mean.uniform_(-0.1, 0.1)
                m.running_var.uniform_(0.5, 1.5)
    return model


def _batches(n=2, size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(size, 3, 32, 32, generator=g), torch.randint(0, 10, (size,), generator=g))
            for _ in range(n)]


def _loader(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=16)


@torch.no_grad()
def _mean_loss(model, batches):
    loss = nn.CrossEntropyLoss(reduction="sum")
    return sum(float(loss(model(x), y)) for x, y in batches) / sum(len(x) for x, _ in batches)


def test_override_keeps_each_tables_top_and_every_criterion_cuts_the_same_shape():
    model = _model()
    plan = probe.cut_plan(model)
    assert len(plan) >= 4
    scores = probe.weight_scores(model, plan, _batches())
    alive = {key: scores["l1"][key] > 0 for key, _, _ in plan}
    original = pruning.group_importance
    shapes, masks = [], []
    with probe.RankingOverride() as override:
        for spec in (("l1", 0, 0), ("random", 0, 0), ("anti_l1", 0, 0)):
            tables, fallbacks = probe.score_tables(spec, scores, alive, plan)
            assert fallbacks == 0
            cut_model, modes, kept = probe.cut(model, plan, 0.5, tables, override, (3, 32, 32))
            assert modes == ["structural"] * len(plan)
            for key, idx in kept.items():
                assert set(idx) == set(torch.topk(tables[key], len(idx)).indices.tolist())
            shapes.append(probe.shape_signature(cut_model))
            masks.append(kept)
    assert pruning.group_importance is original
    assert shapes[0] == shapes[1] == shapes[2] != probe.shape_signature(model)
    assert probe.jaccard(masks[0], masks[0]) == 1.0
    assert probe.jaccard(masks[0], masks[2]) == 0.0  # half of every even-width group, opposite ends


def test_ablation_damage_equals_the_loss_after_cutting_that_channel():
    model = _model(1)
    plan = probe.cut_plan(model)
    batches = _batches(seed=3)
    damage, base = probe.ablation_scores(model, plan, batches)
    assert base == pytest.approx(_mean_loss(model, batches), abs=1e-6)
    for entry in (plan[0], plan[1]):  # the stage-1 residual stream, then a block's inner group
        key, group, _ = entry
        j = int(torch.argmax(damage[key].abs()))
        table = torch.arange(1, group.width + 1, dtype=torch.float64)
        table[j] = 0.5
        with probe.RankingOverride() as override:
            cut_model, modes, kept = probe.cut(model, [entry], (group.width - 1) / group.width,
                                               {key: table}, override, (3, 32, 32))
        assert modes == ["structural"] and j not in kept[key] and len(kept[key]) == group.width - 1
        assert _mean_loss(cut_model.eval(), batches) - base == pytest.approx(float(damage[key][j]), abs=1e-4)


def test_activation_scores_cover_every_group():
    model = _model(2)
    plan = probe.cut_plan(model)
    scores = probe.activation_scores(model, plan, _batches(seed=4))
    for name in ("act", "apoz", "hrank"):
        assert set(scores[name]) == {key for key, _, _ in plan}
        for key, group, _ in plan:
            values = scores[name][key]
            assert values.shape == (group.width,) and torch.isfinite(values).all() and (values >= 0).all()


def test_channel_features_have_one_row_per_channel():
    model = _model(3)
    plan = probe.cut_plan(model)
    batches = _batches(seed=5)
    scores, _ = probe.all_scores(model, plan, batches)
    grads = probe.mean_gradients(model, plan, batches)
    columns, table = probe.channel_features(plan, grads, scores)
    assert table.shape == (sum(g.width for _, g, _ in plan), len(columns))
    assert np.isfinite(table[:, columns.index("ablation")]).all()
    assert np.isfinite(table[:, columns.index("g_L2")]).all()


def test_rank_table_orders_alive_channels_and_zeroes_dead_ones():
    raw = torch.tensor([0.3, -1.0, 2.0, 0.0], dtype=torch.float64)
    alive = torch.tensor([True, True, True, False])
    assert probe.rank_table(raw, alive).tolist() == [3.0, 1.0, 4.0, 0.0]
    assert probe.rank_table(raw, alive, largest_first=False).tolist() == [2.0, 4.0, 1.0, 0.0]


def test_recover_returns_accuracies_for_every_budget_kind():
    model = _model(4)
    loaders = (_loader(seed=6), _loader(seed=7), _loader(seed=8))
    for budget in (0, "bn", 1):
        val, test, collector = probe.recover(model, budget, loaders, CPU, seed=0, bn_batches=2)
        assert 0.0 <= val <= 1.0 and 0.0 <= test <= 1.0 and collector is None


def test_snapshotting_loader_steps_the_collector_once_per_batch():
    class Counter:
        n = 0

        def step(self):
            self.n += 1

    counter = Counter()
    wrapped = probe._Snapshotting(_loader(n=64), counter)
    assert len(wrapped) == 4 and len(list(wrapped)) == 4 and counter.n == 4


def test_kendall_tau_and_jaccard():
    assert probe.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert probe.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert probe.jaccard({("a",): [0, 1]}, {("a",): [1, 2]}) == pytest.approx(1 / 3, abs=1e-4)


def test_filter_stats_match_napv2():
    if not Path(NAP_DIR, "nap2").is_dir():
        pytest.skip("NAPv2 checkout not found (set NAPV2_DIR)")
    sys.path.insert(0, NAP_DIR)
    stats = pytest.importorskip("nap2.stats")
    weight = torch.randn(5, 3, 3, 3, generator=torch.Generator().manual_seed(0))
    mine = probe.nap_filter_stats(weight).numpy()
    for j in range(weight.shape[0]):
        ref = stats.extract_layer_stats(weight[j].double().flatten().numpy())
        expected = ([ref[k] for k in ("mean", "variance", "median", "std", "max", "min", "covariance",
                                      "skewness", "kurtosis")]
                    + list(np.asarray(ref["q-th_percentile"]).ravel())
                    + [ref["L1_norm_new"], ref["L2_norm_new"]])
        np.testing.assert_allclose(mine[j], np.asarray(expected, dtype=float), rtol=1e-9, atol=1e-12)
