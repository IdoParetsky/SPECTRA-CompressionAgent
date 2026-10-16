"""
v23 ``SPECTRA_ALLOC_KIND=randbest`` (default off): the per-network random-search baseline of Li et al. (CVPR 2022,
"Revisiting Random Channel Pruning") in the plan agent's own action space. N plans z_g ~ N(0, σ²) per group are decoded
as the agent's mean plan is (``plan_agent.decode`` at the walk target less the undershoot, k_min, the width floor and
the budget), scored on the val half with the trainer's proxy (``plan_proxies.PlanScorer``, the reward path of
``plan_agent.NetInstance``), and the best val (ties to the lower draw) is the plan the walk follows as it follows the
agent's. The TEST half is never read and every global RNG state is put back after the search. With the kind unused
nothing changes: ``KINDS`` keeps every old kind, the default is still ``sens`` and no rule kind enters the search.

CPU only, no datasets.  python -m pytest tests/test_v23_randbest.py -v
"""

import os
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import RATES, SHAPE, ZOO, _instance, _loader, _randomize_norms  # noqa: E402
from tests.test_v10_fixed_target import _target_env  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, plan_agent, plan_proxies  # noqa: E402
from src.ModelHandlers.ClassificationHandler import ClassificationHandler  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_EVAL_PROXY",
                 "SPECTRA_MIN_WIDTH", "SPECTRA_GROUP_ONCE", "SPECTRA_SEED", "SPECTRA_FT_")
OLD_KINDS = ("uniform", "sens", "inner", "widths", "sample", "agent", "agent_sample", "sens_cost")
NET = "toy.pt"
DRAW_LINE = re.compile(r"^\[alloc\] randbest (\S+) draw=(\d+) score=(\d\.\d{4}) params x(\d\.\d{3}) FLOPs x(\d\.\d{3})$")
SUMMARY_LINE = re.compile(r"^\[alloc\] randbest (\S+): best draw (\d+) of (\d+), val (\d\.\d{4}) "
                          r"\(spread (\d\.\d{4})-(\d\.\d{4}), median (\d\.\d{4})\), proxy (\S+), sigma (\S+), "
                          r"(\d+\.\d) min$")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


class _NeverRead:
    """The TEST loader of the fake env: reading it during the search fails the test."""

    def __iter__(self):
        pytest.fail("the TEST loader was read during the randbest search")

    def __len__(self):
        return 0


class _Counted:
    """A loader that counts how many times it is iterated."""

    def __init__(self, loader, reads):
        self.loader, self.reads = loader, reads

    def __iter__(self):
        self.reads.append(1)
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def _handler(model):
    return ClassificationHandler(model, nn.CrossEntropyLoss())


def _model():
    torch.manual_seed(0)
    model = ZOO["thin_r20_w4"]().eval()
    _randomize_norms(model)
    return model


def _env(model=None, target_keep=0.6, seed=0):
    """A fake env for ``alloc_walk._state``: the thin ResNet-20 origin of ``tests.test_plan_agent._instance`` with its
    toy train / val loaders, a TEST loader that must never be read, and the run seed ``conf.seed``."""
    return types.SimpleNamespace(
        selected_net_path=f"/nets/{NET}", current_model=_model() if model is None else model, target_keep=target_keep,
        conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES, passes=6, seed=seed),
        train_loader=_loader(), val_loader=_loader(seed=1, shuffle=False), test_loader=_NeverRead(),
        create_learning_handler=_handler, _input_shape=lambda: SHAPE,
        _dependency_groups=lambda mwr: channel_groups.build_channel_groups(mwr.model))


def _randbest(monkeypatch, n=4, proxy="cut", **flags):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "randbest")
    monkeypatch.setenv("SPECTRA_ALLOC_RANDBEST_N", str(n))
    if proxy is not None:
        monkeypatch.setenv("SPECTRA_ALLOC_RANDBEST_PROXY", proxy)
    for key, value in flags.items():
        monkeypatch.setenv(key, str(value))


def _run(monkeypatch, env=None, lines=None, records=None):
    """``alloc_walk._state`` on ``env`` with its lines and records captured: ``(entry, lines, records)``."""
    lines, records = [] if lines is None else lines, [] if records is None else records
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
    entry = alloc_walk._state(_env() if env is None else env)
    return entry, lines, records


def _record(records, event):
    found = [k for e, k in records if e == event]
    assert len(found) == 1, [e for e, _k in records]
    return found[0]


def _widths(draw):
    return {int(row): int(width) for row, width in draw["widths"].items()}


def _plan_of(model):
    return group_sensitivity.group_plan(ModelWithRows(model), channel_groups.build_channel_groups(model))


def _rng_state():
    """The global torch / random / numpy state as plain Python values (comparable with ``==``)."""
    npy = np.random.get_state()
    return torch.get_rng_state().tolist(), random.getstate(), (npy[0], npy[1].tolist(), tuple(npy[2:]))


# ------------------------------------------------------------------ flags, the old kinds and the draws


def test_flags_default_parse_and_every_old_kind_stays(monkeypatch):
    assert alloc_walk.kind() == "sens"
    assert set(OLD_KINDS) <= set(alloc_walk.KINDS) and "randbest" in alloc_walk.KINDS
    assert alloc_walk.randbest_n() == 100 and alloc_walk.randbest_sigma() == 1.0
    with pytest.raises(ValueError, match="SPECTRA_ALLOC_RANDBEST_PROXY"):
        alloc_walk.randbest_proxy()                                   # no default: the proxy must be named
    monkeypatch.setenv("SPECTRA_ALLOC_RANDBEST_PROXY", "ft12_4")      # the eval-side name, not a trainer proxy
    with pytest.raises(ValueError):
        alloc_walk.randbest_proxy()
    for raw, want in (("fs32", "fs32"), (" BN8 ", "bn8"), ("cut", "cut"), ("ft1", "ft1")):
        monkeypatch.setenv("SPECTRA_ALLOC_RANDBEST_PROXY", raw)
        assert alloc_walk.randbest_proxy() == want
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "RandBest")
    assert alloc_walk.kind() == "randbest"
    monkeypatch.setenv("SPECTRA_ALLOC_RANDBEST_N", "0")
    with pytest.raises(ValueError):
        alloc_walk.randbest_n()
    monkeypatch.setenv("SPECTRA_ALLOC_RANDBEST_N", "7")
    monkeypatch.setenv("SPECTRA_ALLOC_RANDBEST_SIGMA", "-2")
    assert alloc_walk.randbest_n() == 7 and alloc_walk.randbest_sigma() == 0.0
    env = types.SimpleNamespace(conf=types.SimpleNamespace(seed=5))
    assert alloc_walk.run_seed(env) == 5
    monkeypatch.setenv("SPECTRA_SEED", "42")
    assert alloc_walk.run_seed(env) == 5 and alloc_walk.run_seed(types.SimpleNamespace()) == 42
    monkeypatch.delenv("SPECTRA_SEED")
    assert alloc_walk.run_seed(types.SimpleNamespace()) == 0
    assert len({alloc_walk.randbest_seed(a, b) for a in range(3) for b in range(3)}) == 9


def test_draws_are_seeded_per_group_private_and_a_prefix_of_a_longer_search():
    before = torch.get_rng_state()
    a = alloc_walk.randbest_draws(5, 3, 1.0, 7)
    assert torch.equal(torch.get_rng_state(), before)                 # its own generator: the global stream stands
    assert len(a) == 5 and all(len(row) == 3 for row in a)
    assert a == alloc_walk.randbest_draws(5, 3, 1.0, 7) and a != alloc_walk.randbest_draws(5, 3, 1.0, 8)
    assert alloc_walk.randbest_draws(8, 3, 1.0, 7)[:5] == a           # a longer search extends, never reshuffles
    half = alloc_walk.randbest_draws(5, 3, 0.5, 7)
    assert all(h == pytest.approx(0.5 * v) for ra, rh in zip(a, half) for v, h in zip(ra, rh))
    assert alloc_walk.randbest_draws(2, 3, 0.0, 7) == [[0.0] * 3] * 2
    flat = [v for row in alloc_walk.randbest_draws(200, 10, 1.0, 3) for v in row]
    assert abs(sum(flat) / len(flat)) < 0.1 and 0.9 < (sum(v * v for v in flat) / len(flat)) ** 0.5 < 1.1


# ------------------------------------------------------------------ the decode, the budget and the floors


@pytest.mark.parametrize("budget", ["params", "flops"])
def test_every_draw_is_the_agents_decode_at_the_walk_target_less_the_undershoot(monkeypatch, budget):
    _randbest(monkeypatch, n=5, proxy="cut", SPECTRA_PLAN_MIN_WIDTH="2", SPECTRA_ALLOC_SAMPLE_SEED="3")
    if budget == "flops":
        monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "flops")
        monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", "flop:0.6")
    env = _env(seed=11)
    entry, lines, records = _run(monkeypatch, env)
    plan = _plan_of(env.current_model)
    cm = (plan_agent.FlopModel(env.current_model, plan, SHAPE) if budget == "flops"
          else plan_agent.ParamModel(env.current_model, plan))
    params0 = float(cm.pm.total0 if budget == "flops" else cm.total0)
    rec = _record(records, "alloc_randbest")
    assert rec["n"] == 5 and len(rec["draws"]) == 5 and rec["budget"] == budget and rec["min_width"] == 2
    assert rec["seed"] == alloc_walk.randbest_seed(3, 11) == 3 * 100003 + 11 and rec["k_min"] == 0.1
    assert (rec["sample_seed"], rec["run_seed"], rec["proxy"], rec["sigma"]) == (3, 11, "cut", 1.0)
    assert rec["target"] == pytest.approx(0.58) and "fs_lr" not in rec
    z = alloc_walk.randbest_draws(5, len(cm.rows), 1.0, rec["seed"])
    for i, draw in enumerate(rec["draws"]):
        widths = _widths(draw)
        expect, info = plan_agent.decode(z[i], cm, 0.58, 0.1, min_width=2)   # the agent's own decode, floors and polish
        assert widths == expect and draw["kept"] == pytest.approx(info["kept"])
        assert cm.kept(widths) == pytest.approx(draw["kept"]) and abs(draw["kept"] - 0.58) < 0.04
        assert all(2 <= widths[row] <= cm.widths0[row] for row in cm.rows)
        assert all(widths[row] >= max(1, round(0.1 * cm.widths0[row])) for row in cm.rows)
        assert draw["params"] == pytest.approx(cm.params(widths) / params0)
        assert 0.0 < draw["flops"] <= 1.0 and 0.0 <= draw["score"] <= 1.0
        if budget == "flops":
            assert draw["flops"] == pytest.approx(draw["kept"])
    pick = rec["pick"]
    assert entry["widths"] == _widths(rec["draws"][pick]) and rec["score"] == rec["draws"][pick]["score"]
    unit = "FLOPs" if budget == "flops" else "params"
    plan_line = [line for line in lines if line.startswith(f"[alloc] {NET}: ")]
    assert plan_line == [lines[-1]]
    assert lines[-1].startswith(f"[alloc] {NET}: randbest best draw {pick} of 5 (val {rec['score']:.4f}, proxy cut, "
                                f"sigma 1, seed {rec['seed']}) plan keeps x{rec['draws'][pick]['kept']:.3f} of the "
                                f"{unit} (target x0.580 = walk target")
    assert lines[-1].endswith("; group width floor 2")
    planned = _record(records, "alloc_plan")
    assert planned["kind"] == "randbest" and planned["min_width"] == 2
    assert planned["kept"] == pytest.approx(rec["draws"][pick]["kept"])
    assert planned["randbest"] == {key: rec[key] for key in ("pick", "n", "score", "proxy", "sigma", "seed")}
    assert set(int(r) for r in planned["rows"]) == set(cm.rows)
    assert (planned.get("budget") == "flops") == (budget == "flops")


def test_with_the_agent_of_record_named_the_draws_decode_with_its_k_min_and_floor(monkeypatch, tmp_path):
    path = tmp_path / "plan_agent" / "policy_latest.pt"
    path.parent.mkdir(parents=True)
    torch.save({"state_dict": {}, "feature_dim": 63, "k_min": 0.2, "min_width": 3}, str(path))
    _randbest(monkeypatch, n=3, proxy="cut", SPECTRA_PLAN_AGENT=str(path), SPECTRA_ALLOC_MIN_KEEP="0.25")
    env = _env()
    entry, lines, records = _run(monkeypatch, env)                   # no SPECTRA_PLAN_MIN_WIDTH: the checkpoint's floor
    pm = plan_agent.ParamModel(env.current_model, _plan_of(env.current_model))
    rec = _record(records, "alloc_randbest")
    assert rec["k_min"] == 0.2 and rec["min_width"] == 3 and rec["agent"] == str(path)
    z = alloc_walk.randbest_draws(3, len(pm.rows), 1.0, rec["seed"])
    for i, draw in enumerate(rec["draws"]):
        assert _widths(draw) == plan_agent.decode(z[i], pm, 0.58, 0.2, min_width=3)[0]
        assert all(width >= 3 for width in _widths(draw).values())
    assert all(width >= 3 for width in entry["widths"].values())
    assert "min_width" not in _record(records, "alloc_plan")         # the flag is off: the plan line stays as before
    assert "width floor" not in lines[-1]
    monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "2")                 # the flag wins over the checkpoint, as plan_for_env
    _entry, _lines, records = _run(monkeypatch, _env())
    rec = _record(records, "alloc_randbest")
    assert rec["k_min"] == 0.2 and rec["min_width"] == 2
    monkeypatch.delenv("SPECTRA_PLAN_AGENT")                          # no checkpoint: the alloc flags
    _entry, _lines, records = _run(monkeypatch, _env())
    rec = _record(records, "alloc_randbest")
    assert rec["k_min"] == 0.25 and rec["min_width"] == 2 and "agent" not in rec


# ------------------------------------------------------------------ the seeds


def test_the_same_seeds_give_the_same_plans_and_pick_and_another_seed_other_plans(monkeypatch):
    _randbest(monkeypatch, n=4, proxy="cut", SPECTRA_ALLOC_SAMPLE_SEED="5")
    a, _lines, ra = _run(monkeypatch, _env(seed=2))
    b, _lines, rb = _run(monkeypatch, _env(seed=2))
    ra, rb = _record(ra, "alloc_randbest"), _record(rb, "alloc_randbest")
    ra.pop("minutes"), rb.pop("minutes")
    assert a["widths"] == b["widths"] and ra == rb
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SEED", "6")
    _c, _lines, rc = _run(monkeypatch, _env(seed=2))
    rc = _record(rc, "alloc_randbest")
    assert rc["seed"] != ra["seed"] and [_widths(d) for d in rc["draws"]] != [_widths(d) for d in ra["draws"]]
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SEED", "5")
    _d, _lines, rd = _run(monkeypatch, _env(seed=3))                  # another run seed (SPECTRA_SEED): other plans
    rd = _record(rd, "alloc_randbest")
    assert rd["seed"] != ra["seed"] and [_widths(d) for d in rd["draws"]] != [_widths(d) for d in ra["draws"]]


# ------------------------------------------------------------------ the pick and the lines


def test_the_pick_is_the_argmax_of_the_val_scores_with_ties_to_the_lower_draw(monkeypatch):
    _randbest(monkeypatch, n=5, proxy="fs2")
    scripted = iter([0.30, 0.90, 0.50, 0.90, 0.10])
    seen = []
    monkeypatch.setattr(plan_proxies.PlanScorer, "score",
                        lambda self, widths, proxy, batches: seen.append((dict(widths), proxy, len(batches)))
                        or next(scripted))
    entry, lines, records = _run(monkeypatch)
    rec = _record(records, "alloc_randbest")
    assert rec["pick"] == 1 and rec["score"] == 0.9 and [d["score"] for d in rec["draws"]] == [0.3, 0.9, 0.5, 0.9, 0.1]
    assert entry["widths"] == seen[1][0] == _widths(rec["draws"][1])
    assert all(proxy == "fs2" and n_batches == 2 for _w, proxy, n_batches in seen) and rec["fs_lr"] == 0.01
    draws = [DRAW_LINE.match(line) for line in lines if line.startswith(f"[alloc] randbest {NET} draw=")]
    assert len(draws) == 5 and all(draws) and [int(m.group(2)) for m in draws] == [0, 1, 2, 3, 4]
    assert [float(m.group(3)) for m in draws] == [0.3, 0.9, 0.5, 0.9, 0.1]
    assert [float(m.group(4)) for m in draws] == pytest.approx([d["params"] for d in rec["draws"]], abs=5e-4)
    assert [float(m.group(5)) for m in draws] == pytest.approx([d["flops"] for d in rec["draws"]], abs=5e-4)
    summary = [SUMMARY_LINE.match(line) for line in lines if line.startswith(f"[alloc] randbest {NET}:")]
    assert len(summary) == 1 and summary[0]
    assert summary[0].groups()[1:9] == ("1", "5", "0.9000", "0.1000", "0.9000", "0.5000", "fs2", "1")
    plan_line = [line for line in lines if line.startswith(f"[alloc] {NET}: ")]
    assert plan_line == [lines[-1]]                                   # the gates' line, last, naming the kind
    assert lines[-1].startswith(f"[alloc] {NET}: randbest best draw 1 of 5 (val 0.9000, proxy fs2, sigma 1, seed 0) "
                                "plan keeps x")
    assert lines.index(summary[0].string) == len(lines) - 2 and all(DRAW_LINE.match(line) for line in lines[:5])
    planned = _record(records, "alloc_plan")
    assert planned["kind"] == "randbest" and planned["randbest"]["pick"] == 1 and planned["randbest"]["n"] == 5
    scripted = iter([0.5] * 5)                                        # all tied: the lowest index
    entry, lines, records = _run(monkeypatch)
    rec = _record(records, "alloc_randbest")
    assert rec["pick"] == 0 and entry["widths"] == _widths(rec["draws"][0])
    assert any(line.startswith(f"[alloc] randbest {NET}: best draw 0 of 5, val 0.5000 (spread 0.5000-0.5000, "
                               "median 0.5000), proxy fs2, sigma 1, ") for line in lines)


# ------------------------------------------------------------------ val only, scored as the trainer rewards


@pytest.mark.parametrize("proxy", ["cut", "bn3", "fs2"])
def test_the_search_reads_val_once_never_test_and_scores_as_the_trainer_rewards(monkeypatch, proxy):
    _randbest(monkeypatch, n=3, proxy=proxy, SPECTRA_EVAL_PROXY_SEED="4")
    env = _env()
    reads = []
    env.val_loader = _Counted(env.val_loader, reads)
    _entry, _lines, records = _run(monkeypatch, env)                 # env.test_loader fails the test if iterated
    assert len(reads) == 1
    rec = _record(records, "alloc_randbest")
    assert rec["proxy_seed"] == 4 and len(rec["draws"]) == 3
    inst = _instance()                                                # the trainer's own instance of this origin
    batches = inst.batches(proxy, 4)
    for draw in rec["draws"]:
        r = inst.reward(_widths(draw), proxy, batches)
        assert 100.0 * (draw["score"] - inst.origin_val) == pytest.approx(r, abs=1e-9)


def test_plan_scorer_is_the_trainers_reward_before_the_origin_subtraction():
    inst = _instance()
    scorer = plan_proxies.PlanScorer(inst.model, inst.plan, _loader(seed=1, shuffle=False), "cpu",
                                     train_loader=_loader(), input_shape=SHAPE, name=NET)
    assert scorer.cutter.rows == inst.pm.rows and scorer.widths0 == inst.pm.widths0
    assert len(scorer.val) == len(inst.val) and all(torch.equal(a[0], b[0]) for a, b in zip(scorer.val, inst.val))
    widths, _info = plan_agent.decode([0.1 * k for k in range(inst.n_groups)], inst.pm, 0.6)
    for proxy in ("cut", "bn3", "fs2"):
        ours, theirs = scorer.batches(proxy, 3), inst.batches(proxy, 3)
        assert len(ours) == len(theirs)
        for a, b in zip(ours, theirs):
            if isinstance(a, tuple):
                assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            else:
                assert torch.equal(a, b)
        s = scorer.score(widths, proxy, ours)
        assert 0.0 <= s <= 1.0 and 100.0 * (s - inst.origin_val) == pytest.approx(inst.reward(widths, proxy, theirs),
                                                                                   abs=1e-9)
        assert scorer.score(widths, proxy, ours) == pytest.approx(s)  # no leak from one plan into the next
    assert scorer.batches("ft1", 5) == 5 and scorer.batches("cut", 5) == []
    with pytest.raises(ValueError):
        scorer.score(widths, "fs0", [])


def test_plan_scorer_ft1_is_the_trainers_finetune_reward_and_the_search_accepts_it(monkeypatch):
    inst = _instance()
    inst.create_handler = _handler
    scorer = plan_proxies.PlanScorer(inst.model, inst.plan, _loader(seed=1, shuffle=False), "cpu",
                                     train_loader=inst.train_loader, input_shape=SHAPE, create_handler=_handler, name=NET)
    widths, _info = plan_agent.decode([0.1 * k for k in range(inst.n_groups)], inst.pm, 0.6)
    shown = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: shown.append(str(s)))
    origin = {k: v.clone() for k, v in inst.model.state_dict().items()}
    before = _rng_state()
    s = scorer.score(widths, "ft1", 5)
    assert _rng_state() == before and 0.0 <= s <= 1.0
    assert 100.0 * (s - inst.origin_val) == pytest.approx(inst.reward(widths, "ft1", 5), abs=1e-9)
    assert scorer.recipe_line == inst.recipe_line and scorer.recipe_line.startswith("optim=")
    assert not any("Fine-tune recipe" in line for line in shown)     # the handler's lines stay captured
    for key, value in inst.model.state_dict().items():
        assert torch.equal(value, origin[key]), key                   # the origin is untouched
    for bad in ([], None, 1.5, True):
        with pytest.raises(ValueError, match="seed"):
            scorer.score(widths, "ft1", bad)

    class Silent:
        def __init__(self, model):
            pass

        def train_model(self, *a, **k):
            return None                                               # no recipe line: it never trained

    scorer.create_handler = Silent
    with pytest.raises(RuntimeError, match="did not run"):
        scorer.score(widths, "ft1", 5)
    _randbest(monkeypatch, n=2, proxy="ft1")                          # and through the search, on val only
    entry, lines, records = _run(monkeypatch, _env())
    rec = _record(records, "alloc_randbest")
    assert len(rec["draws"]) == 2 and all(0.0 <= d["score"] <= 1.0 for d in rec["draws"])
    assert entry["widths"] == _widths(rec["draws"][rec["pick"]])
    assert not any("Fine-tune recipe" in line or "Epoch " in line for line in lines)


# ------------------------------------------------------------------ the RNG


def test_the_search_puts_every_global_rng_state_back_and_does_not_depend_on_it(monkeypatch):
    _randbest(monkeypatch, n=3, proxy="fs2")
    env = _env()
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    before = _rng_state()
    a, _lines, ra = _run(monkeypatch, env)
    assert _rng_state() == before
    after = (torch.rand(3), random.random(), np.random.rand())
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    assert torch.equal(torch.rand(3), after[0]) and random.random() == after[1] and np.random.rand() == after[2]
    env = _env()
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    b, _lines, rb = _run(monkeypatch, env)                           # the plans do not depend on where the streams stood
    ra, rb = _record(ra, "alloc_randbest"), _record(rb, "alloc_randbest")
    ra.pop("minutes"), rb.pop("minutes")
    assert a["widths"] == b["widths"] and ra == rb


# ------------------------------------------------------------------ one draw, the errors, the other kinds


def test_n_one_works_and_a_missing_proxy_or_a_residual_raises_before_any_draw(monkeypatch):
    _randbest(monkeypatch, n=1, proxy="cut")
    entry, lines, records = _run(monkeypatch)
    rec = _record(records, "alloc_randbest")
    assert rec["pick"] == 0 and rec["n"] == 1 and len(rec["draws"]) == 1 and entry["widths"] == _widths(rec["draws"][0])
    assert sum(DRAW_LINE.match(line) is not None for line in lines) == 1
    summary = [SUMMARY_LINE.match(line) for line in lines if line.startswith(f"[alloc] randbest {NET}:")]
    assert len(summary) == 1 and summary[0]
    val = f"{rec['score']:.4f}"
    assert summary[0].groups()[1:7] == ("0", "1", val, val, val, val)
    monkeypatch.delenv("SPECTRA_ALLOC_RANDBEST_PROXY")
    lines, records = [], []
    with pytest.raises(ValueError, match="SPECTRA_ALLOC_RANDBEST_PROXY"):
        _run(monkeypatch, lines=lines, records=records)
    assert lines == [] and records == []                              # refused before any draw or line
    monkeypatch.setenv("SPECTRA_ALLOC_RANDBEST_PROXY", "cut")
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "sens")
    with pytest.raises(ValueError, match="SPECTRA_PLAN_RESIDUAL"):
        _run(monkeypatch, lines=lines, records=records)
    assert lines == [] and records == []
    monkeypatch.setenv("SPECTRA_PLAN_RESIDUAL", "off")                # off is as unset
    entry, lines, records = _run(monkeypatch)
    assert _record(records, "alloc_randbest")["pick"] == 0


def test_the_other_kinds_never_enter_the_search_and_the_grid_priority_ranks_randbest_by_z(monkeypatch):
    monkeypatch.setattr(alloc_walk, "randbest_plan", lambda *a, **k: pytest.fail("a rule kind entered the search"))
    monkeypatch.setattr(plan_proxies, "PlanScorer", lambda *a, **k: pytest.fail("a rule kind built a scorer"))
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "uniform")
    monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: [])
    entry, lines, records = _run(monkeypatch)
    assert entry["widths"] and lines == [lines[0]] and lines[0].startswith(f"[alloc] {NET}: uniform alpha=0.5 plan keeps x")
    assert [e for e, _k in records] == ["alloc_plan"] and "randbest" not in _record(records, "alloc_plan")
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "agent")
    monkeypatch.setenv("SPECTRA_PLAN_AGENT", "/x/plan_agent/policy_latest.pt")
    calls = []

    def fake(env, target, path, k_min):                               # the agent path's call, as before
        calls.append((target, path, k_min))
        return {5: 3}, {"kind": "agent", "alpha": 0.0, "target": target, "kept": 0.57, "keeps": {5: 0.5},
                        "origin_widths": {5: 6}, "sens": {5: 0.1}, "held": 0, "policy": "plan_agent/policy_latest.pt"}

    monkeypatch.setattr(plan_agent, "plan_for_env", fake)
    entry, lines, records = _run(monkeypatch)
    assert entry["widths"] == {5: 3} and calls == [(pytest.approx(0.58), "/x/plan_agent/policy_latest.pt", 0.1)]
    assert lines == [lines[0]] and lines[0].startswith(f"[alloc] {NET}: agent plan_agent/policy_latest.pt plan keeps x0.570")
    assert "randbest" not in _record(records, "alloc_plan")
    rows = [1, 2, 3]
    z = {1: 0.4, 2: -1.0, 3: 2.0}
    assert alloc_walk.grid_priority({"kind": "randbest", "sens": z}, rows) == z
    assert alloc_walk.grid_priority({"kind": "agent", "sens": z}, rows) == z
    assert alloc_walk.grid_priority({"kind": "agent_sample", "sens": z}, rows) == z
    assert alloc_walk.grid_priority({"kind": "uniform", "sens": {r: 1.0 for r in rows}, "alpha": 0.5}, rows) is None


# ------------------------------------------------------------------ downstream: the grid round and the real walk


def test_under_the_grid_round_the_best_draw_is_rounded_like_an_agent_plan(monkeypatch):
    _randbest(monkeypatch, n=3, proxy="cut", SPECTRA_ALLOC_GRID_ROUND="1", SPECTRA_GROUP_ONCE_PER_PASS="1")
    env = _env()
    entry, lines, records = _run(monkeypatch, env)
    pm = plan_agent.ParamModel(env.current_model, _plan_of(env.current_model))
    assert "grid" in entry and entry["grid"]["targets"] == entry["widths"]
    assert 0.6 - 0.05 <= pm.kept(entry["widths"]) <= 0.6 + 1e-9        # at or below the walk target (the v18 band
    rec = _record(records, "alloc_randbest")                            # itself is the v18 tests' own check)
    assert set(entry["widths"]) == set(_widths(rec["draws"][rec["pick"]]))
    grid = _record(records, "alloc_grid_round")
    assert grid["kind"] == "randbest" and grid["walk_target"] == 0.6
    assert [e for e, _k in records] == ["alloc_randbest", "alloc_plan", "alloc_grid_round"]
    assert lines[-1].startswith("[alloc] grid round: ") and lines[-2].startswith(f"[alloc] {NET}: randbest best draw ")


def test_the_walk_follows_the_best_draw_as_it_follows_the_agent(monkeypatch):
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
    _randbest(monkeypatch, n=3, proxy="cut", SPECTRA_ALLOC_GRID_ROUND="1")
    env, _seen = _target_env(_model(), target=0.6, accs=[0.9] * 400, passes=6)
    del env._register_group_lock                                      # the env's own group-once lock
    env.conf.compression_rates_dict = RATES
    env.test_loader = _NeverRead()
    lines, records = [], []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
    done, steps = False, 0
    while not done and steps < 300:
        legal = env.legal_action_mask(device="cpu")
        pick = int(alloc_walk.action(env, legal, RATES, "cpu").item())
        assert bool(legal[pick])
        _, _, done = env.step(RATES[pick])
        steps += 1
    kept = env.param_ratio()
    assert done and 0.6 - 0.03 <= kept <= 0.6 + 1e-9
    state = env._alloc_walk[env.selected_net_path]
    assert not state["fallback"]
    rec = _record(records, "alloc_randbest")                           # the search ran once for the net
    assert rec["n"] == 3 and rec["run_seed"] == 0 and rec["network"] == env.selected_net_path
    targets = state["grid"]["targets"]
    now = alloc_walk.group_widths(env.current_model, list(targets))
    assert sum(now[row] < targets[row] for row in targets) <= 1        # only the landing cut may leave the plan
    assert sum(line.startswith("[alloc] tiny_r20w4.pt: randbest best draw ") for line in lines) == 1
