"""
Three default-off evaluation flags (v23 eval gaps), CPU only, no datasets.

    A1  SPECTRA_CUT_RANK=random      random channels at the cuts the allocation walk plays (same widths as L1)
    A2  SPECTRA_EVAL_FINAL_FT_SCRATCH_BUDGET=flops   Scratch-B epochs = round(E / kept FLOPs), optional cap
    A3  SPECTRA_EVAL_THROUGHPUT=1    GPU inference throughput of the final-FT candidates, before any fine-tune

    python -m pytest tests/test_v23_eval_gaps.py -v
"""

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import SHAPE, ZOO, _plan, _randomize_norms  # noqa: E402
from tests.test_v10_fixed_target import _target_env  # noqa: E402
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, fortify  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402
from spectra_models_instantiation.thin_res_net import resnet20  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RATES = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
FLAG_PREFIXES = ("SPECTRA_CUT_", "SPECTRA_ALLOC_", "SPECTRA_PLAN_", "SPECTRA_FIXED_TARGET", "SPECTRA_GROUP_ONCE",
                 "SPECTRA_EVAL_SIZE_", "SPECTRA_EVAL_FINAL_FT", "SPECTRA_EVAL_THROUGHPUT", "SPECTRA_FILTER_IMPORTANCE",
                 "SPECTRA_SEED")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield
    pruning._RANDOM_RANK.update(gen=None, seed=None)       # the random ranking's generator is process-wide


# ------------------------------------------------------------------ A1: the "random" ranking


def _conv(n=8, dead=()):
    torch.manual_seed(0)
    layer = nn.Conv2d(3, n, 3)
    with torch.no_grad():
        for i in dead:
            layer.weight[i] = 0
    return layer


def test_random_mode_is_accepted_and_scores_one_positive_value_per_live_filter():
    assert pruning.normalize_importance_mode("random") == "random" and "random" in pruning.IMPORTANCE_MODES
    assert pruning.normalize_importance_mode("bogus") == "l1"          # the lenient reader is unchanged
    layer = _conv(8, dead=(2, 5))
    scores = pruning.filter_importance(layer, "random")
    assert scores.shape == (8,) and scores.device == layer.weight.device
    assert [float(scores[i]) for i in (2, 5)] == [0.0, 0.0]
    live = [i for i in range(8) if i not in (2, 5)]
    assert all(0.0 < float(scores[i]) <= 1.0 for i in live)
    assert pruning.alive_filters(layer).tolist() == live                # alive_filters reads the default (L1), as before


def test_random_mode_is_deterministic_per_seed_and_advances_on_every_call(monkeypatch):
    layer = _conv(16)
    pruning.reset_random_rank(3)
    a1, a2 = pruning.filter_importance(layer, "random"), pruning.filter_importance(layer, "random")
    pruning.reset_random_rank(3)
    b1, b2 = pruning.filter_importance(layer, "random"), pruning.filter_importance(layer, "random")
    assert torch.equal(a1, b1) and torch.equal(a2, b2) and not torch.equal(a1, a2)
    pruning.reset_random_rank(4)
    assert not torch.equal(pruning.filter_importance(layer, "random"), a1)
    # seed order: SPECTRA_CUT_RANK_SEED, then SPECTRA_SEED, then 0
    assert pruning.reset_random_rank() == 0
    monkeypatch.setenv("SPECTRA_SEED", "11")
    assert pruning.reset_random_rank() == 11 and pruning.random_rank_seed() == 11
    monkeypatch.setenv("SPECTRA_CUT_RANK_SEED", "5")
    assert pruning.reset_random_rank() == 5
    monkeypatch.setenv("SPECTRA_CUT_RANK_SEED", "x")
    with pytest.raises(ValueError):
        pruning.reset_random_rank()


def test_random_mode_leaves_the_global_rng_alone():
    layer = _conv(8)
    torch.manual_seed(123)
    want = torch.rand(3)
    torch.manual_seed(123)
    pruning.reset_random_rank(1)
    pruning.filter_importance(layer, "random")
    assert torch.equal(torch.rand(3), want)


def test_an_unknown_cut_rank_raises_and_unset_leaves_the_ranking_alone(monkeypatch):
    assert fortify.cut_rank() == "" and alloc_walk.cut_ranking("fpgm") == "fpgm" and alloc_walk.cut_ranking(None) is None
    monkeypatch.setenv("SPECTRA_CUT_RANK", " Random ")
    assert fortify.cut_rank() == "random" and alloc_walk.cut_ranking("fpgm") == "random"
    monkeypatch.setenv("SPECTRA_CUT_RANK", "l1")
    assert alloc_walk.cut_ranking(None) == "l1"
    monkeypatch.setenv("SPECTRA_CUT_RANK", "shuffled")
    with pytest.raises(ValueError):
        fortify.cut_rank()
    with pytest.raises(ValueError):
        alloc_walk.cut_ranking(None)
    with pytest.raises(ValueError):
        pruning.strict_importance_mode("shuffled")


# ---- the walk's cuts: kept indices (L1, as today) and kept widths (identical under random)


def _tag_batchnorms(model):
    """Distinct integer BatchNorm weights per channel, so a channel's origin can be read off after any cut."""
    tags = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.weight.data = torch.randperm(module.num_features).float() + 1.0
            tags[name] = module.weight.detach().clone()
    return tags


def _kept(model, tags):
    """``{BatchNorm name: origin channel indices still alive}``."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            alive = set(module.weight.detach().tolist())
            out[name] = tuple(i for i, value in enumerate(tags[name].tolist()) if value in alive)
    return out


def _walk(monkeypatch, cut_rank, lines=None):
    """The eval walk of ``SPECTRA_ALLOC_KIND=uniform`` with the runner's ranking rule: ``env.step(rate, ranking=
    alloc_walk.cut_ranking(...))`` for every cut. Returns ``(kept indices, origin tags)``."""
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "uniform")
    if cut_rank:
        monkeypatch.setenv("SPECTRA_CUT_RANK", cut_rank)
    else:
        monkeypatch.delenv("SPECTRA_CUT_RANK", raising=False)
    if lines is not None:
        monkeypatch.setattr(utils, "print_flush", lambda *a, **k: lines.append(" ".join(str(x) for x in a)))
    pruning.reset_random_rank(7)
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    tags = _tag_batchnorms(model)
    env, _ = _target_env(model, target=0.6, accs=[0.9] * 400, passes=6)
    env.conf.compression_rates_dict = RATES
    done, steps = False, 0
    while not done and steps < 300:
        legal = env.legal_action_mask(device="cpu")
        pick = int(alloc_walk.action(env, legal, RATES, "cpu").item())
        rate = RATES[pick]
        ranking = alloc_walk.cut_ranking(None) if abs(rate - 1.0) >= 1e-9 else None
        _, _, done = env.step(rate, ranking=ranking)
        steps += 1
    assert done
    return _kept(env.current_model, tags), tags


def test_unset_cut_rank_keeps_the_l1_survivors_at_a_cut(monkeypatch):
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    torch.manual_seed(0)
    model = resnet20(num_classes=10, large_input=False, width=4).eval()
    tags = _tag_batchnorms(model)
    mwr = ModelWithRows(model)
    group = channel_groups.group_of(channel_groups.build_channel_groups(model), mwr.all_layers[mwr.row_to_main_layer[0]])
    expect = pruning.select_group_survivors(group, 0.7, mode="l1")
    assert expect.numel() < group.width
    env, _ = _target_env(model, target=0.05, accs=[0.9] * 5)
    env.step(0.7, ranking=alloc_walk.cut_ranking(None))                  # the runner's call with the flag unset
    assert _kept(env.current_model, tags)["embed.1"] == tuple(expect.tolist())


def test_the_walk_with_cut_rank_unset_is_the_l1_walk(monkeypatch):
    unset, _ = _walk(monkeypatch, "")
    explicit, _ = _walk(monkeypatch, "l1")
    assert unset == explicit


def test_random_cuts_keep_the_l1_widths_and_other_channels(monkeypatch):
    lines = []
    l1, tags = _walk(monkeypatch, "")
    rnd, _ = _walk(monkeypatch, "random", lines)
    assert {n: len(v) for n, v in rnd.items()} == {n: len(v) for n, v in l1.items()}      # per-layer widths identical
    assert any(rnd[n] != l1[n] for n in l1)                                              # different survivors somewhere
    assert any(len(v) < len(tags[n]) for n, v in rnd.items())                              # the walk did cut
    said = [ln for ln in lines if ln.startswith("[alloc] cut ranking")]
    assert said == ["[alloc] cut ranking: random (seed 7) for tiny_r20w4.pt"]              # once per network
    rnd2, _ = _walk(monkeypatch, "random")
    assert rnd2 == rnd                                                                     # same seed, same cuts


# ---- the override does not reach the probes, the plan agent or the training environment


def _model(name="thin_r20_w4"):
    torch.manual_seed(0)
    model = ZOO[name]().eval()
    _randomize_norms(model)
    return model


def _batches(n=2, size=8):
    gen = torch.Generator().manual_seed(3)
    return [(torch.randn(size, *SHAPE, generator=gen), torch.randint(0, 10, (size,), generator=gen))
            for _ in range(n)]


def test_group_sensitivity_and_the_plan_are_identical_with_cut_rank_random(monkeypatch):
    model = _model()
    plan, batches = _plan(model), _batches()
    off_costs, on_costs = {}, {}
    off, base_off = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, costs=off_costs)
    off_w, off_info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_cost", 0.6)
    monkeypatch.setenv("SPECTRA_CUT_RANK", "random")
    pruning.reset_random_rank(3)
    state = pruning._RANDOM_RANK["gen"].get_state().clone()
    on, base_on = group_sensitivity.group_sensitivity(model, plan, batches, SHAPE, costs=on_costs)
    on_w, on_info = alloc_walk.plan_targets(model, batches, SHAPE, "sens_cost", 0.6)
    assert on == off and base_on == base_off and on_costs == off_costs
    assert on_w == off_w and on_info["keeps"] == off_info["keeps"] and on_info["sens"] == off_info["sens"]
    assert torch.equal(pruning._RANDOM_RANK["gen"].get_state(), state)       # no probe drew a random score


def test_every_probe_and_the_plan_agent_rank_with_an_explicit_l1():
    """Static guard: the code that measures sensitivities, builds the agent's tokens and decodes its plan names its
    ranking, so neither ``SPECTRA_CUT_RANK`` nor ``SPECTRA_FILTER_IMPORTANCE`` can move it."""
    def calls(path, name):
        text = (REPO / path).read_text(encoding="utf-8")
        out = []
        for m in re.finditer(re.escape(name) + r"\(", text):
            depth, i = 1, m.end()
            while depth:
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
            out.append(text[m.start():i])
        return out

    for path in ("src/group_sensitivity.py", "src/alloc_walk.py"):
        found = calls(path, "prune_current_model")
        assert found and all('importance="l1"' in c for c in found), path
    plan_calls = calls("src/plan_agent.py", "select_group_survivors") + calls("src/plan_agent.py", "group_importance")
    assert plan_calls and all('"l1"' in c for c in plan_calls)
    assert "cut_ranking" not in (REPO / "src/plan_agent.py").read_text(encoding="utf-8")
    for path in ("src/A2C_Agent_Reinforce.py", "src/plan_trainer.py", "src/NetworkEnv.py", "src/group_sensitivity.py"):
        text = (REPO / path).read_text(encoding="utf-8")
        assert "cut_rank" not in text and "SPECTRA_CUT_RANK" not in text, path


# ------------------------------------------------------------------ A2: Scratch-B epochs


def test_scratch_epochs_follow_the_flops_budget(monkeypatch):
    assert fortify.eval_final_ft_scratch_budget() == "" and fortify.eval_final_ft_scratch_max_epochs() == 0
    assert [fortify.scratch_epochs_for(f) for f in (0.5, 1.0, 0.3)] == [200, 200, 200]      # budget unset: E
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH_BUDGET", " Flops ")
    assert fortify.eval_final_ft_scratch_budget() == "flops"
    assert fortify.scratch_epochs_for(0.5) == 400
    assert fortify.scratch_epochs_for(1.0) == 200
    assert fortify.scratch_epochs_for(0.3) == 667
    assert fortify.scratch_epochs_for(0.0) >= 200                                           # floored, not a division by zero
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH_EPOCHS", "30")
    assert fortify.scratch_epochs_for(0.5) == 60
    monkeypatch.delenv("SPECTRA_EVAL_FINAL_FT_SCRATCH_EPOCHS")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH_MAX_EPOCHS", "500")
    assert fortify.eval_final_ft_scratch_max_epochs() == 500
    assert fortify.scratch_epochs_for(0.3) == 500 and fortify.scratch_epochs_for(0.5) == 400
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH_BUDGET", "")
    assert fortify.scratch_epochs_for(0.3) == 200                                            # off again: the cap is moot
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH_BUDGET", "params")
    with pytest.raises(ValueError):
        fortify.eval_final_ft_scratch_budget()
    with pytest.raises(ValueError):
        fortify.scratch_epochs_for(0.5)


# ---- the runner (fixtures of tests/test_v9c_traj_models.py)


def _runner_fixtures(monkeypatch, capture_records=False):
    import a2c_agent_reinforce_runner as runner
    from tests.test_v9c_traj_models import _quiet
    lines, issues = _quiet(monkeypatch, runner)
    records = []
    if capture_records:
        monkeypatch.setattr(runner.run_recorder, "record", lambda kind, **k: records.append(dict(k, kind=kind)))
    return runner, lines, issues, records


def test_scratch_budget_sets_the_epochs_per_candidate_and_marks_the_line(monkeypatch):
    from tests.test_v9c_traj_models import _Env, _candidates, _point
    runner, lines, _issues, records = _runner_fixtures(monkeypatch, capture_records=True)
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH", "1")
    env = _Env()
    runner._run_final_ft(env, "net.pt", {"val_best": _point(5, 0.7, -9.0)}, _candidates(), 100)
    assert [r["epochs"] for r in env.runs] == [100, 200, 100, 200]                  # off: 200 for every scratch run
    assert not any("budget=flops" in ln for ln in lines)
    assert all("scratch_budget" not in r for r in records if r["kind"] == "eval_traj_final_ft")
    lines.clear(), records.clear()
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SCRATCH_BUDGET", "flops")
    env = _Env()
    runner._run_final_ft(env, "net.pt", {"val_best": _point(5, 0.7, -9.0)}, _candidates(), 100)
    # val_best keeps FLOPs x0.7, size_param0.80 keeps x0.8 (tests/test_v9c_traj_models._candidates)
    assert [r["epochs"] for r in env.runs] == [100, 286, 100, 250]
    scratch_lines = [ln for ln in lines if "init=scratch" in ln]
    assert len(scratch_lines) == 2 and all(" init=scratch budget=flops | " in ln for ln in scratch_lines)
    assert not any("budget=flops" in ln for ln in lines if "init=inherit" in ln)
    flagged = [r for r in records if r["kind"] == "eval_traj_final_ft" and r.get("scratch")]
    assert len(flagged) == 2 and all(r["scratch_budget"] == "flops" for r in flagged)
    assert all("scratch_budget" not in r for r in records if r["kind"] == "eval_traj_final_ft" and not r.get("scratch"))


# ------------------------------------------------------------------ A3: throughput


def _tiny_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(4, 2))


def test_throughput_flags_default_off_and_parse(monkeypatch):
    assert not fortify.eval_throughput() and fortify.eval_throughput_batches() == (256, 1)
    monkeypatch.setenv("SPECTRA_EVAL_THROUGHPUT", "1")
    monkeypatch.setenv("SPECTRA_EVAL_THROUGHPUT_BATCHES", "64, 1,x,64,0")
    assert fortify.eval_throughput() and fortify.eval_throughput_batches() == (64, 1)


def test_measure_throughput_returns_medians_and_changes_nothing():
    model = _tiny_net().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    rng = torch.get_rng_state()
    out = fortify.measure_throughput(model, (3, 8, 8), "cpu", batches=(4, 1), warmup_iters=1, timed_iters=2, repeats=3)
    assert set(out) == {4, 1}
    assert all(set(v) == {"img_s", "ms"} and v["img_s"] > 0 and v["ms"] > 0 for v in out.values())
    assert out[4]["img_s"] == pytest.approx(4 * 1000.0 / out[4]["ms"], rel=0.5)             # medians of one run's two views
    assert model.training and all(m.training for m in model.modules())
    after = model.state_dict()
    assert set(after) == set(before) and all(torch.equal(after[k], before[k]) for k in before)
    assert torch.backends.cudnn.benchmark == benchmark and torch.backends.cudnn.deterministic == deterministic
    assert torch.equal(torch.get_rng_state(), rng)                                         # zero inputs: no RNG drawn


def test_throughput_off_never_measures(monkeypatch):
    from tests.test_v9c_traj_models import _Env, _candidates, _point
    runner, lines, _issues, _records = _runner_fixtures(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("throughput measured with SPECTRA_EVAL_THROUGHPUT unset")

    monkeypatch.setattr(fortify, "measure_throughput", boom)
    monkeypatch.setattr(runner, "_measure_candidates_throughput", boom)
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_ORIGIN", "1")
    runner._run_final_ft(_Env(), "net.pt", {"val_best": _point(5, 0.7, -9.0), "origin": _point(-1, 1.0, 0.0)},
                         _candidates(), 10)
    assert not any("THROUGHPUT" in ln for ln in lines)


@pytest.mark.skipif(torch.cuda.is_available(), reason="the no-CUDA branch")
def test_throughput_without_cuda_says_so_once_and_the_fine_tune_goes_on(monkeypatch):
    from tests.test_v9c_traj_models import _Env, _candidates, _point
    runner, lines, _issues, _records = _runner_fixtures(monkeypatch)
    monkeypatch.setenv("SPECTRA_EVAL_THROUGHPUT", "1")
    monkeypatch.setattr(fortify, "measure_throughput", lambda *a, **k: pytest.fail("no CUDA: nothing to measure"))
    env = _Env()
    runner._run_final_ft(env, "net.pt", {"val_best": _point(5, 0.7, -9.0)}, _candidates(), 10)
    assert lines.count("[eval] THROUGHPUT skipped: no CUDA device") == 1
    assert [r["tag"] for r in env.runs] == ["final FT val_best", "final FT size_param0.80"]


def test_throughput_line_record_and_order_with_a_fake_gpu(monkeypatch):
    """The CUDA branch with the device calls faked: one line and one record per distinct step, the origin included,
    all before the first fine-tune."""
    from tests.test_v9c_traj_models import _Env, _candidates, _point
    runner, lines, _issues, records = _runner_fixtures(monkeypatch, capture_records=True)
    monkeypatch.setenv("SPECTRA_EVAL_THROUGHPUT", "1")
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_ORIGIN", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "FakeGPU")
    calls = []

    def fake(model, input_shape, device, batches, **k):
        calls.append((input_shape, str(device), tuple(batches)))
        return {b: {"img_s": 1000.0 / b, "ms": 0.25 * b} for b in batches}

    monkeypatch.setattr(fortify, "measure_throughput", fake)
    env = _Env()
    env.conf = SimpleNamespace(device="cpu")
    order = []
    train_model = env.create_learning_handler

    def handler(model):
        h = train_model(model)
        real = h.train_model
        h.train_model = lambda loader, **kw: (order.append("train"), real(loader, **kw))[1]
        return h

    env.create_learning_handler = handler
    monkeypatch.setattr(runner.utils, "print_flush", lambda msg: (lines.append(str(msg)),
                                                                   order.append("line") if "THROUGHPUT" in str(msg)
                                                                   else None)[0])
    runner._run_final_ft(env, "net.pt", {"val_best": _point(5, 0.7, -9.0), "origin": _point(-1, 1.0, 0.0)},
                         _candidates(), 10)
    det = int(bool(torch.backends.cudnn.deterministic))
    want = [f"[eval] THROUGHPUT val_best net.pt step=5 | params x0.700 | FLOPs x0.700 | bs256 4 img/s 64.000 ms | "
            f"bs1 1000 img/s 0.250 ms | FakeGPU fp32 cudnn.benchmark=1 deterministic={det}",
            f"[eval] THROUGHPUT size_param0.80 net.pt step=3 | params x0.800 | FLOPs x0.800 | bs256 4 img/s 64.000 ms | "
            f"bs1 1000 img/s 0.250 ms | FakeGPU fp32 cudnn.benchmark=1 deterministic={det}",
            f"[eval] THROUGHPUT origin net.pt step=-1 | params x1.000 | FLOPs x1.000 | bs256 4 img/s 64.000 ms | "
            f"bs1 1000 img/s 0.250 ms | FakeGPU fp32 cudnn.benchmark=1 deterministic={det}"]
    assert [ln for ln in lines if "THROUGHPUT" in ln] == want
    assert order[:3] == ["line"] * 3 and "train" in order[3:]                                 # all measured before any train
    assert calls == [((1,), "cpu", (256, 1))] * 3
    recs = [r for r in records if r["kind"] == "eval_throughput"]
    assert [(r["label"], r["step"]) for r in recs] == [("val_best", 5), ("size_param0.80", 3), ("origin", -1)]
    assert recs[0]["gpu"] == "FakeGPU" and recs[0]["results"]["256"] == {"img_s": 1000.0 / 256, "ms": 64.0}
    assert recs[0]["param"] == 0.7 and recs[0]["flop"] == 0.7 and recs[0]["network"] == "net.pt"
