"""D-IMIT: the state dump (src/state_dump.py) and the imitation probe (scripts/dimit_probe.py)."""

import json
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dimit_probe as dp  # noqa: E402

V10_FLAGS = {"SPECTRA_FORTIFY": "1", "SPECTRA_BUDGET_IN_STATE": "1", "SPECTRA_STATE_SLACK": "1",
             "SPECTRA_STATE_GROUPCOST": "1", "SPECTRA_FIXED_TARGET": "1", "SPECTRA_STATE_SENS": "1"}
LAYOUT = [("base", 0, 38), ("fortify", 38, 42), ("budget", 42, 43), ("slack", 43, 45),
          ("groupcost", 45, 49), ("target", 49, 51), ("sens", 51, 53), ("action", 53, 63)]
CATALOG = ["resnet20-width8_cifar10_thin-res-net_89.74_0.069_10.72.pt",
           "resnet20-width10_cifar10_thin-res-net_91.90_0.107_16.55.pt",
           "resnet56-width6_cifar10_thin-res-net_92.88_0.122_18.60.pt",
           "resnet32_cifar10_chenyaofo_93.53_047_138.24.pt",
           "vgg11_bn_cifar10_chenyaofo_92.79_9.76_306.58.pt",
           "vgg13_bn_cifar10_chenyaofo_94_9.94_457.58.pt",
           "mobilenet-v2x0.5_cifar10_chenyaofo_92.99_0.7_55.94.pt",
           "mobilenet-v2x1_cifar10_chenyaofo_93.79_2.24_175.96.pt",
           "densenet40_cifar10_densenet-cifar_93.17_0.176_74.43.pt",
           "vgg11-bn_svhn_vgg-chenyaofo_96.25_9.756_153.60.pt"]


@pytest.fixture
def v10_env(monkeypatch):
    for key, value in V10_FLAGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SPECTRA_STATE_TOKENS", raising=False)


def toy_state(num_layers=8, width=63, seed=0):
    from src.action_costs import ACTION_FEATURE_DIM
    g = torch.Generator().manual_seed(seed)
    cids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])[:num_layers]
    return {"layer_features": torch.randn(num_layers, width, generator=g),
            "layer_types": torch.tensor([2, 3, 2, 3, 2, 3, 2, 1])[:num_layers],
            "coupling_ids": cids, "block_ids": cids, "target_index": 0,
            "action_costs": torch.rand(5, ACTION_FEATURE_DIM, generator=g)}


def toy_record(name, kappa, seed=0):
    """Four planned groups on eight tokens; the keep rises with the sens percentile channel."""
    state = toy_state(seed=seed)
    token_rows = [0, 0, 1, -1, 2, 2, 3, -1]
    g = torch.Generator().manual_seed(100 + seed)
    pct = torch.rand(4, generator=g)
    keeps = {r: float(0.1 + 0.9 * pct[k]) for k, r in enumerate((0, 1, 2, 3))}
    for i, row in enumerate(token_rows):
        state["layer_features"][i, 51:53] = 0.0
        state["layer_features"][i, 45] = 0.0
        if row >= 0:
            state["layer_features"][i, 52] = float(pct[row])
            state["layer_features"][i, 45] = 0.1 * (row + 1)
    group = {"layer_features": torch.randn(4, 67, generator=g), "layer_types": torch.tensor([2, 2, 2, 2]),
             "coupling_ids": torch.arange(4), "block_ids": torch.arange(4), "target_index": 0,
             "relations": torch.zeros(4, 4, dtype=torch.long), "action_costs": state["action_costs"],
             "token_members": [[0, 1], [2, 3], [4, 5], [6, 7]]}
    return {"net": name, "kappa": kappa, "layer": state, "group": group, "bert": None,
            "layout": LAYOUT, "group_layout": LAYOUT + [("group_extra", 63, 67)],
            "token_rows": token_rows, "group_token_rows": [0, 1, 2, 3],
            "plan": {"rows": [0, 1, 2, 3], "keeps": keeps, "sens": {r: 1.0 for r in keeps},
                     "origin_widths": {0: 16, 1: 32, 2: 32, 3: 64}, "widths": {}, "kept": 0.6}}


def toy_args(**over):
    base = dict(seeds=[0], epochs=2, lr=3e-4, frozen_epochs=2, frozen_lr=1e-3, weight_decay=0.01)
    base.update(over)
    return types.SimpleNamespace(**base)


TOY_NETS = [CATALOG[3], CATALOG[5], CATALOG[8], CATALOG[7]]
TOY_FOLDS = (("resnet32", "vgg13_bn"), ("densenet40", "mobilenet-v2x1"))


def toy_recs():
    return [toy_record(net, k, seed=i * 3 + j) for i, net in enumerate(TOY_NETS)
            for j, k in enumerate((0.4, 0.6, 0.8))]


# ---------------------------------------------------------------- state dump


def test_dump_flags(monkeypatch):
    from src import state_dump
    monkeypatch.delenv("SPECTRA_DUMP_STATES", raising=False)
    assert not state_dump.enabled()
    monkeypatch.setenv("SPECTRA_DUMP_STATES", "1")
    assert state_dump.enabled()
    monkeypatch.delenv("SPECTRA_DUMP_KEEPS", raising=False)
    assert state_dump.keeps() == (0.4, 0.6, 0.8)
    monkeypatch.setenv("SPECTRA_DUMP_KEEPS", "0.5,0.7")
    assert state_dump.keeps() == (0.5, 0.7)


def test_layout_is_v10s_63_columns(v10_env):
    from src import state_dump
    from src.BERTInputModeler import token_feature_dim
    assert token_feature_dim(5) == 63
    assert state_dump.layout(63, 5) == LAYOUT
    assert state_dump.layout(67, 5, group_extra=4)[-1] == ("group_extra", 63, 67)
    with pytest.raises(ValueError):
        state_dump.layout(62, 5)


def test_token_rows_follow_group_of(monkeypatch):
    from src import state_dump
    import src.channel_groups as channel_groups
    g1, g2, g3 = object(), object(), object()
    owner = {"conv1": g1, "bn1": g1, "conv2": g2, "fc": None, "conv3": g3}
    monkeypatch.setattr(channel_groups, "group_of", lambda groups, layer: owner[layer])
    mwr = types.SimpleNamespace(all_layers=["conv1", "bn1", "conv2", "fc", "conv3"])
    plan = [(g1, 0), (g2, 2)]
    assert state_dump.token_rows(mwr, [g1, g2, g3], plan) == [0, 0, 2, -1, -1]
    assert state_dump.token_rows(mwr, None, plan) == [-1] * 5


def test_group_token_rows():
    from src.state_dump import group_token_rows
    rows = [3, 3, -1, 7, 7, -1]
    assert group_token_rows([[0, 1], [2], [3, 4, 5]], rows) == ([3, -1, 7], 0)
    assert group_token_rows([[0, 3]], rows) == ([3], 1)


def test_bert_input_restores_the_encoder_kind():
    import src.BERTInputModeler as bim
    from src.state_dump import bert_input
    old = bim.STATE_ENCODER_KIND
    with bert_input():
        assert bim.STATE_ENCODER_KIND == "bert"
    assert bim.STATE_ENCODER_KIND == old
    with pytest.raises(RuntimeError):
        with bert_input():
            raise RuntimeError("boom")
    assert bim.STATE_ENCODER_KIND == old


def test_to_cpu_nested():
    from src.state_dump import to_cpu
    out = to_cpu({"a": torch.ones(2), "b": [torch.zeros(1), 3], "c": "x"})
    assert torch.equal(out["a"], torch.ones(2)) and out["b"][1] == 3 and out["c"] == "x"


# ---------------------------------------------------------------- probe


def test_catalog_keys_and_folds():
    keys = [dp.net_key(n) for n in CATALOG]
    assert len(set(keys)) == 10
    assert sorted(n for fold in dp.FOLDS for n in fold) == sorted(keys)
    assert set(dp.THIN) <= set(keys)
    with pytest.raises(ValueError):
        dp.net_key("resnet110_cifar10.pt")


def test_spearman_and_ranks():
    assert dp.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert dp.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert dp.spearman([1, 2, 3], [1, 1, 1]) is None
    assert dp.spearman([1, 2], [1, 2]) is None
    assert dp.rank([5, 1, 5, 3]) == [2.5, 0.0, 2.5, 1.0]
    assert dp.spearman([1, 2, 3, 4, 5], [1, 1, 2, 3, 3]) == pytest.approx(0.9486833, abs=1e-6)


def test_effective_rank():
    m = torch.zeros(10, 4)
    m[:5, 0] = 1.0
    m[5:, 1] = 1.0
    erank, srank = dp.effective_rank(m)
    assert erank == pytest.approx(1.0, abs=1e-6) and srank == 1
    erank, srank = dp.effective_rank(torch.randn(400, 8, generator=torch.Generator().manual_seed(0)))
    assert 6.0 < erank <= 8.0 and srank <= 8


@pytest.mark.parametrize("kind", ["transformer", "set"])
def test_token_encoder_returns_the_sequence_the_encoder_pools(kind, monkeypatch):
    monkeypatch.setenv("SPECTRA_ENCODER_DROPOUT", "0")
    from src.Model.StateEncoder import build_state_encoder
    state = toy_state()
    torch.manual_seed(0)
    enc = build_state_encoder(kind, 63).eval()
    pooled = enc(state)
    captured = {}

    def grab(encoded, target_index, n):
        captured["encoded"] = encoded
        return encoded[0, :n]

    enc.pool = grab
    tokens = enc(state)
    assert tokens.shape == (8, enc.output_dim)
    assert torch.allclose(type(enc).pool(enc, captured["encoded"], 0, 8), pooled, atol=1e-6)
    assert torch.equal(dp.token_encoder(enc)(state), tokens)


def test_sample_zeroes_channels_and_weights_by_group_share():
    rec = toy_record(CATALOG[3], 0.6)
    s = dp.Sample(rec, "layer", ("sens",))
    assert torch.count_nonzero(s.state["layer_features"][:, 51:53]) == 0
    assert torch.equal(s.state["layer_features"][:, :51], rec["layer"]["layer_features"][:, :51])
    assert torch.allclose(s.weight, torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert s.token_mask.tolist() == [True, True, True, False, True, True, True, False]
    assert s.token_k.tolist() == [0, 0, 1, 2, 2, 3]
    g = dp.Sample(rec, "group", ("sens",))
    assert g.num_tokens == 4 and g.token_k.tolist() == [0, 1, 2, 3]
    assert "token_members" not in g.state


def test_group_prediction_is_the_token_mean():
    s = dp.Sample(toy_record(CATALOG[3], 0.6))
    pred, valid = dp.group_prediction(torch.arange(8, dtype=torch.float32), s)
    assert valid.all() and pred.tolist() == [0.5, 2.0, 4.5, 6.0]


def test_sens_reference_recovers_the_plan_order():
    refs = dp.references(toy_recs(), folds=TOY_FOLDS)
    assert refs["sens_pct"]["rho"] == pytest.approx(1.0)
    assert set(refs) == {"sens_pct", "depth", "width"}


@pytest.mark.parametrize("arm", ["a", "b", "c", "h", "i", "e0"])
def test_arms_run_end_to_end(arm):
    res = dp.run_arm(arm, toy_recs(), toy_args(), torch.device("cpu"), folds=TOY_FOLDS, log=lambda _l: None)
    assert len(res["folds"]) == 2
    for fold in res["folds"]:
        assert len(fold["items"]) == 6
        assert all(-1.0 <= i["rho"] <= 1.0 for i in fold["items"])
    assert res["rho"] is not None and res["wabs"] is not None


def test_v10_arm_loads_the_actor_encoder(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRA_ENCODER_DROPOUT", "0")
    from src.Model.StateEncoder import SpectraStateEncoder
    torch.manual_seed(1)
    enc = SpectraStateEncoder(feature_dim=63, dropout=0.0)
    sd = {f"state_encoder.{k}": v for k, v in enc.state_dict().items()}
    sd["actor.0.weight"] = torch.zeros(3, 3)
    path = tmp_path / "actor.pt"
    torch.save(sd, path)
    v10 = dp.v10_encoder_state(str(path))
    assert set(v10) == set(enc.state_dict())
    res = dp.run_arm("e", toy_recs(), toy_args(), torch.device("cpu"), folds=TOY_FOLDS, v10_state=v10,
                     log=lambda _l: None)
    assert res["rho"] is not None
    ranks = dp.encoder_ranks(toy_recs(), v10, torch.device("cpu"))
    assert ranks["v10"]["states"] == 12 and ranks["v10"]["tokens"] == 96


def test_calls_and_pairs():
    assert dp.call(0.71) == "SUFFICIENT" and dp.call(0.40) == "INSUFFICIENT" and dp.call(0.55) == "PARTIAL"
    assert dp.call(None) == "NOT RUN"
    fold = lambda rhos: {"folds": [{"rho": r} for r in rhos]}  # noqa: E731
    assert dp.paired(fold([0.8, 0.8, 0.8, 0.8, 0.5]), fold([0.6] * 5))["call"] == "BEATS"
    assert dp.paired(fold([0.8, 0.8, 0.8, 0.5, 0.5]), fold([0.6] * 5))["call"] == "TIES"
    assert dp.paired(fold([0.4] * 5), fold([0.6] * 5))["call"] == "LOSES TO"
    res = {"folds": [{"items": [{"net": "resnet20-width8", "rho": 0.2, "wabs": 0.1},
                                {"net": "vgg11_bn_cifar10", "rho": 0.6, "wabs": 0.2}]}]}
    assert dp.restrict(res, lambda n: n not in dp.THIN)["rho"] == pytest.approx(0.6)


def test_main_writes_results(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    for i, rec in enumerate(toy_recs()):
        torch.save(rec, dump / f"rec{i}.pt")
    out = tmp_path / "out"
    dp.main(["--dump", str(dump), "--out", str(out), "--arms", "a,b", "--seeds", "0",
             "--epochs", "1", "--frozen_epochs", "1"])
    results = json.loads((out / "dimit_results.json").read_text())
    assert set(results["arms"]) == {"a", "b"} and "d" in results["not_run"]
    assert "j_out" in results and "a_vs_b" in results["pairs"]


def test_bert_arm_is_reported_not_run_when_bert_does_not_load(tmp_path, monkeypatch):
    dump = tmp_path / "dump"
    dump.mkdir()
    torch.save(toy_record(CATALOG[3], 0.6), dump / "rec.pt")
    monkeypatch.setattr(dp, "load_bert", lambda device: (None, "stub: no weights"))
    out = tmp_path / "out"
    dp.main(["--dump", str(dump), "--out", str(out), "--arms", "g", "--seeds", "0"])
    results = json.loads((out / "dimit_results.json").read_text())
    assert results["arms"] == {} and results["not_run"]["g"] == "stub: no weights"
