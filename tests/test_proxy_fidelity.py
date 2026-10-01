"""CPU tests for the proxy-fidelity battery (``SPECTRA_EVAL_PROXY_FIDELITY``, default off) and its readout.

    python -m pytest tests/test_proxy_fidelity.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fortify  # noqa: E402
from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from src.Configuration.StaticConf import StaticConf  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
from src.NetworkEnv import NetworkEnv  # noqa: E402
import src.proxy_fidelity as proxy_fidelity  # noqa: E402
import src.utils as utils  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
KEYS = ("SPECTRA_EVAL_PROXY_FIDELITY", "SPECTRA_EVAL_PROXY_BUDGETS", "SPECTRA_EVAL_PROXY_FINAL_SEEDS",
        "SPECTRA_EVAL_PROXY_FINAL_EPOCHS", "SPECTRA_EVAL_PROXY_WHERE_ROWS", "SPECTRA_EVAL_FINAL_FT_BATCH",
        "SPECTRA_FT_REINIT_EDITED", "SPECTRA_FT_RECIPE", "SPECTRA_FT_BN_RECAL", "SPECTRA_FT_GROUP_FIRST")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


def _readout():
    spec = importlib.util.spec_from_file_location("proxy_fidelity_readout",
                                                  REPO / "scripts" / "proxy_fidelity_readout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loader(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)


def _bare_env(model):
    env = NetworkEnv.__new__(NetworkEnv)
    conf = StaticConf.get_instance().conf_values
    env.conf = SimpleNamespace(train_compressed_layer_only=False, device=conf.device, learning_rate=1e-3)
    env.current_model = model
    env.row_idx = 3
    env.train_loader, env.val_loader, env.test_loader = _loader(16, 1), _loader(8, 2), _loader(8, 3)
    env.original_params = utils.calc_num_parameters(model)
    env.original_flops = utils.calc_flops(model, (3, 32, 32))
    handler = ClassificationHandler(model, nn.CrossEntropyLoss())
    env.original_acc = float(handler.evaluate_model(env.val_loader))
    env._origin_test_acc = float(handler.evaluate_model(env.test_loader))
    return env


# ---------------------------------------------------------------- flags

def test_flags_default_off_and_parse(monkeypatch):
    assert fortify.eval_proxy_fidelity() == ()
    assert fortify.eval_proxy_budgets() == ((12, 4), (40, 10))
    assert fortify.eval_proxy_final_seeds() == (0, 1)
    assert fortify.eval_proxy_final_epochs() == 100
    assert fortify.eval_proxy_where_rows() == 4
    monkeypatch.setenv("SPECTRA_EVAL_PROXY_FIDELITY", "0.7, 0.9,1.0,x,0.9")
    monkeypatch.setenv("SPECTRA_EVAL_PROXY_BUDGETS", "3x1,bad,5")
    monkeypatch.setenv("SPECTRA_EVAL_PROXY_FINAL_SEEDS", "2,2,7")
    assert fortify.eval_proxy_fidelity() == (0.9, 0.7)
    assert fortify.eval_proxy_budgets() == ((3, 1), (5, 5))
    assert fortify.eval_proxy_final_seeds() == (2, 7)


def test_final_recipe_keys_match_the_runner():
    import a2c_agent_reinforce_runner as runner
    assert proxy_fidelity.FINAL_ENV_KEYS == runner._FINAL_FT_ENV_KEYS


# ---------------------------------------------------------------- readout

def _rec(kind, row, rate, ranking, proxy_val, final_tests, net="n.pt", target=0.9):
    return {"event": "proxy_fidelity", "network": net, "target": target, "kind": kind, "row": row,
            "rate": rate, "ranking": ranking, "val_origin": 0.9, "test_origin": 0.8,
            "scores": {p: {"val": 0.9 + v / 100.0, "test": 0.0} for p, v in proxy_val.items()},
            "finals": {str(s): {"test": 0.8 + t / 100.0, "val": 0.0} for s, t in enumerate(final_tests)}}


def _state(proxy_of, net="n.pt"):
    """Five criteria at keep 0.8 with known final order; ``proxy_of`` maps the final delta to each proxy."""
    finals = [-1.0, -2.0, -3.0, -4.0, -5.0]
    names = [("menu", "l1"), ("menu", "fpgm"), ("crit", "bn_scale"), ("crit", "svd"), ("crit", "taylor")]
    recs = [_rec(kind, 5, 0.8, rank, proxy_of(f), (f, f - 0.05), net=net) for (kind, rank), f in zip(names, finals)]
    recs.append(_rec("identity", 5, 1.0, "none", proxy_of(0.0), (0.0, 0.0), net=net))
    recs.append(_rec("menu", 5, 0.9, "l1", proxy_of(-0.5), (-0.5, -0.5), net=net))
    return recs


def test_readout_ranks_and_calls():
    ro = _readout()
    assert ro.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert ro.spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)
    faithful = _state(lambda f: {"none": -f, "bn": f, "12x4": f, "40x10": f})
    table, depth, proxies, seeds = ro.summarize(faithful)
    crit = next(t for t in table if t["set"] == "crit")
    assert crit["n"] == 5 and crit["ceiling"] == pytest.approx(1.0)
    assert crit["rho"]["12x4"] == pytest.approx(1.0) and crit["regret"]["12x4"] == pytest.approx(0.0)
    assert crit["rho"]["none"] == pytest.approx(-1.0) and crit["regret"]["none"] == pytest.approx(4.0)
    text = "\n".join(ro.calls(table, proxies))
    assert "12x4 (the agent's training budget): VALIDATED" in text and "none: mean rho -1.00" in text
    assert any(d["ranking"] == "l1" and d["final"] == pytest.approx(-0.5 + 1.025) for d in depth)

    short_fails = _state(lambda f: {"none": f, "bn": f, "12x4": -f, "40x10": f})
    table, _, proxies, _ = ro.summarize(short_fails)
    text = "\n".join(ro.calls(table, proxies))
    assert "12x4 (the agent's training budget): NOT validated" in text
    assert "RELEASE the held 40/10 train (21940321)" in text


def test_readout_skips_duplicates_and_reads_files(tmp_path):
    ro = _readout()
    events = tmp_path / "events"
    events.mkdir()
    recs = _state(lambda f: {"12x4": f})
    recs.append(dict(recs[0], duplicate_of="menu row=5 0.800/l1"))
    (events / "rank0.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    rows = ro.load([str(tmp_path)])
    assert len(rows) == len(recs) - 1
    assert ro.main([str(tmp_path)]) == 0


# ---------------------------------------------------------------- the battery on a real thin ResNet-20

def test_battery_scores_every_set_and_leaves_the_walk_untouched(monkeypatch):
    monkeypatch.setenv("SPECTRA_EVAL_PROXY_BUDGETS", "1x1")
    monkeypatch.setenv("SPECTRA_EVAL_PROXY_FINAL_SEEDS", "0")
    monkeypatch.setenv("SPECTRA_EVAL_PROXY_FINAL_EPOCHS", "1")
    monkeypatch.setenv("SPECTRA_EVAL_PROXY_WHERE_ROWS", "2")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_BATCH", "8")
    import src.run_recorder as recorder
    seen, issues = [], []
    monkeypatch.setattr(recorder, "record", lambda event, **kw: seen.append((event, kw)))
    monkeypatch.setattr(recorder, "issue", lambda kind, detail="", **kw: issues.append((kind, detail)))

    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=8).eval()
    env = _bare_env(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    done = set()
    proxy_fidelity.maybe_run(env, "/x/resnet20-width8_cifar10_t.pt", {"step": 4, "param": 0.88, "flop": 0.86},
                             (0.9, 0.7), done)

    assert done == {0.9} and not issues
    assert env.current_model is model and env.row_idx == 3
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng)
    every = [kw for kind, kw in seen if kind == "proxy_fidelity"]
    scored = [r for r in every if not r.get("duplicate_of")]
    kinds = [r["kind"] for r in every]
    assert kinds.count("identity") == 1 and kinds.count("menu") == 4 and kinds.count("crit") == 3
    assert sum(r["kind"] in ("menu", "crit") for r in scored) >= 4      # duplicates are recorded, not scored
    for r in scored:
        assert set(r["scores"]) == {"none", "bn", "1x1"} and set(r["finals"]) == {"0"}
        assert r["finals"]["0"]["epochs"] == 1 and r["finals"]["0"]["aug"] == "none"
        assert (r["param"] < 1.0) == (r["kind"] != "identity")
    menu_rows = {r["row"] for r in every if r["kind"] in ("menu", "crit", "identity")}
    assert len(menu_rows) == 1
    where = [r["row"] for r in every if r["kind"] == "where"]
    assert where and len(set(where)) == len(where) and not set(where) & menu_rows


def test_battery_refuses_a_non_a_recipe(monkeypatch):
    monkeypatch.setenv("SPECTRA_FT_REINIT_EDITED", "1")
    import src.run_recorder as recorder
    seen = []
    monkeypatch.setattr(recorder, "record", lambda event, **kw: seen.append(event))
    model = resnet20(num_classes=10, large_input=False, width=8).eval()
    assert proxy_fidelity.run_battery(_bare_env(model), "/x/n.pt", 0.9, {"step": 1, "param": 0.9, "flop": 0.9}) == []
    assert seen == []
