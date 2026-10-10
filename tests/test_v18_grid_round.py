"""
v18 ``SPECTRA_ALLOC_GRID_ROUND`` (``fortify.alloc_grid_round``; ``alloc_walk.walk_grids`` / ``grid_round_widths`` /
``_grid_round_state`` / ``_grid_watch``).

The alloc walk realizes a group's planned width only through ``choose`` chains on the rate grid (from w0, widths like
w0 · 0.9^a · 0.8^b, each cut rounded), so groups that share one keep round the same way together. Uniform on VGG-19
C100 at params 0.6 decoded x0.581: the stem row is identity-only, every 512-wide group stopped at one 0.8 cut (410 for a
planned 390), and the walk stalled at x0.642, where its fallback took the strongest legal cut on the rows that came
next. With the flag the plan goes onto the widths the walk reaches exactly, and single groups step along that grid
until it keeps at most the walk's target and no more than 0.01 below it: the walk lands with no stall and no fallback.
Default off: unset, ``alloc_walk._state`` keeps the decoder's widths, line and record as in tree_v17.

CPU only, no datasets.  python -m pytest tests/test_v18_grid_round.py -v
"""

import copy
import math
import os
import random
import re
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, fortify, plan_agent  # noqa: E402
import src.channel_groups as channel_groups  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.pruning as pruning  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402
from spectra_models_instantiation import thin_res_net, vgg_depgraph  # noqa: E402
from tests.test_v10_fixed_target import _target_env  # noqa: E402

SHAPE = (3, 32, 32)
RATES = {0: 1.0, 1: 0.9, 2: 0.8}
FLOP_MODEL = plan_agent.FlopModel
GRID_LINE = re.compile(r"^\[alloc\] grid round: (\d+) groups moved; predicted x(\d\.\d{3}) of the (params|FLOPs) "
                       r"\(walk target x(\d\.\d{3})\); grid \[1\.0, 0\.9, 0\.8\]$")
FLAG_PREFIXES = ("SPECTRA_ALLOC_", "SPECTRA_PLAN_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_MIN_WIDTH",
                 "SPECTRA_GROUP_ONCE", "SPECTRA_STEM_ROWS", "SPECTRA_WIDTH_LADDER", "SPECTRA_ACTION_",
                 "SPECTRA_PROTECT_STREAMS", "SPECTRA_FORTIFY", "SPECTRA_EVAL_ROLLBACK")
BASE_KEYS = {"widths", "n_rows", "idle", "fallback", "last_kept"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SPECTRA_GROUP_ONCE_PER_PASS", "1")           # every registered alloc cell walks group-once
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


class Net:
    """A fixture net's origin, read once: its plan, cost models and the planned group each walk row produces."""

    def __init__(self, model):
        self.model = model
        self.mwr = ModelWithRows(model)
        self.groups = channel_groups.build_channel_groups(model)
        self.plan = group_sensitivity.group_plan(self.mwr, self.groups)
        self.rows = [row for _group, row in self.plan]
        self.pm = plan_agent.ParamModel(model, self.plan)
        self._fm = None
        first = {id(group): row for group, row in self.plan}
        self.walk_rows = sorted(self.mwr.row_to_main_layer)[:-1]
        self.group_at = {}
        for row in self.walk_rows:
            group = channel_groups.group_of(self.groups, self.mwr.all_layers[self.mwr.row_to_main_layer[row]])
            self.group_at[row] = None if group is None else first.get(id(group))

    def cost(self, budget):
        if budget == "params":
            return self.pm
        if self._fm is None:
            self._fm = FLOP_MODEL(self.model, self.plan, SHAPE)
        return self._fm

    def walks(self, aims=None, passes=6):
        return alloc_walk.walk_grids(self.model, self.plan, RATES, passes, self.groups, aims=aims)


def _built(make):
    with pytest.MonkeyPatch.context() as mp:
        for key in list(os.environ):
            if key.startswith("SPECTRA_"):
                mp.delenv(key)
        torch.manual_seed(0)
        return Net(make().eval())


@pytest.fixture(scope="module")
def vgg():
    """VGG-19-BN in DepGraph's CIFAR layout with a CIFAR-100 head, at random init (16 groups, 64-512 wide)."""
    return _built(lambda: vgg_depgraph.vgg19_bn(num_classes=100))


@pytest.fixture(scope="module")
def thin():
    """The thin ResNet-20 (width 4: groups of 4, 8 and 16, three residual streams)."""
    return _built(lambda: thin_res_net.resnet20(num_classes=10, large_input=False, width=4))


# ------------------------------------------------------------------ plans in the decoders' families


def _scale_plan(cm, kappa, w, kind_name, sens=None):
    """``plan_targets``' family on a cost model: keep = clip(c · w, 0.1, 1), cut to ``pruning.target_width``, c bisected
    to the kept size closest to ``kappa`` (no polish); ``(widths, info)`` in ``plan_targets``' format."""
    lo, hi, best = 0.0, 1.0 / min(w.values()), None
    for _ in range(40):
        c = 0.5 * (lo + hi)
        keeps = {row: min(1.0, max(0.1, c * w[row])) for row in cm.rows}
        widths = {row: w0 if keeps[row] >= 1.0 else pruning.target_width(w0, keeps[row])
                  for row, w0 in cm.widths0.items()}
        kept = cm.kept(widths)
        if best is None or abs(kept - kappa) < abs(best[2] - kappa):
            best = (widths, keeps, kept)
        if kept > kappa:
            hi = c
        else:
            lo = c
    widths, keeps, kept = best
    return widths, {"kind": kind_name, "alpha": 0.5, "target": kappa, "kept": kept, "keeps": keeps,
                    "origin_widths": dict(cm.widths0), "sens": dict(sens or {row: 1.0 for row in cm.rows}), "held": 0}


def _uniform(cm, kappa):
    return _scale_plan(cm, kappa, {row: 1.0 for row in cm.rows}, "uniform")


def _sens(cm, kappa, seed=1):
    rng = random.Random(seed)
    sens = {row: math.exp(rng.gauss(0.0, 1.0)) for row in cm.rows}
    return _scale_plan(cm, kappa, alloc_walk.weights("sens", sens, 0.5), "sens", sens)


def _agent(cm, kappa, seed=2):
    """A random plan through the agent's decoder (random scores z, sigmoid keeps, b bisected, polished)."""
    rng = random.Random(seed)
    z = [rng.gauss(0.0, 1.0) for _ in cm.rows]
    widths, decoded = plan_agent.decode(z, cm, kappa)
    return widths, {"kind": "agent", "alpha": 0.0, "target": kappa, "kept": decoded["kept"],
                    "keeps": {row: widths[row] / float(cm.widths0[row]) for row in cm.rows},
                    "origin_widths": dict(cm.widths0), "sens": dict(zip(cm.rows, z)), "held": 0,
                    "policy": "plan_agent/policy_latest.pt"}


PLANS = {"uniform": _uniform, "sens": _sens, "agent": _agent}


# ------------------------------------------------------------------ the walk, replayed and real


def _walk(net, targets, walk_target, cm, passes=6, group_once=True):
    """``alloc_walk.action``'s loop (choose, idle count, stall fallback) over ``NetworkEnv``'s legal mask, group-once
    lock, landing and fixed-target end, replayed on group widths with ``cm``'s kept size: ``(widths, state, done)``,
    ``state["stall"]`` the kept size when the fallback fired."""
    identity, n_rows = 0, len(net.walk_rows)
    widths = dict(cm.widths0)
    state = {"idle": 0, "fallback": False, "last_kept": 1.0, "stall": None}

    def cut(group, rate):
        trial = dict(widths)
        trial[group] = pruning.target_width(widths[group], rate)
        return trial, cm.kept(trial)

    for _pass in range(passes):
        locked = set()
        for row in net.walk_rows:
            group = net.group_at[row]
            kept_now = cm.kept(widths)
            if kept_now > state["last_kept"] + 1e-9:
                state.update(idle=0, fallback=False)
            state["last_kept"] = kept_now
            legal = fortify.legal_action_mask(RATES, row_index=row, alive_count=widths.get(group, 1), device="cpu",
                                              force_identity=group in locked)
            legal_idx = [int(i) for i in legal.nonzero(as_tuple=False).flatten().tolist()]
            if state["fallback"]:
                pick = alloc_walk.pick_strongest(RATES, legal_idx, identity)
            else:
                pick = identity
                if group in targets:
                    pick = alloc_walk.choose(widths[group], targets[group], RATES, legal_idx, identity)
                state["idle"] = 0 if pick != identity else state["idle"] + 1
                if state["idle"] >= n_rows and kept_now > walk_target:
                    state["fallback"], state["stall"] = True, kept_now
                    pick = alloc_walk.pick_strongest(RATES, legal_idx, identity)
            if pick == identity or group is None:
                continue
            trial, kept = cut(group, RATES[pick])
            if kept <= walk_target + 1e-9:                         # NetworkEnv._land_on_target; the episode ends
                lo, hi = RATES[pick], 1.0
                for _ in range(10):
                    mid = 0.5 * (lo + hi)
                    if cut(group, mid)[1] <= walk_target + 1e-9:
                        lo = mid
                    else:
                        hi = mid
                return cut(group, lo)[0], state, True
            widths = trial
            if group_once:
                locked.add(group)
    return widths, state, False


def _real_walk(monkeypatch, model, target, *, flops=False, passes=6, limit=400):
    """``alloc_walk.action`` on a ``NetworkEnv`` that really cuts (``_target_env``: recovery skipped, val scripted; the
    env's own group-once lock), until the fixed target ends it: ``(env, lines, done)``."""
    monkeypatch.setenv("SPECTRA_FIXED_TARGET", "1")
    env, _seen = _target_env(model, target=target, accs=[0.9] * limit, passes=passes)
    del env._register_group_lock
    env.conf.compression_rates_dict = RATES
    if flops:
        monkeypatch.setenv("SPECTRA_FIXED_TARGET_METRIC", "flop")
        monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "flops")
        env.original_flops = utils.calc_flops(model, SHAPE)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    done, steps = False, 0
    while not done and steps < limit:
        legal = env.legal_action_mask(device="cpu")
        pick = int(alloc_walk.action(env, legal, RATES, "cpu").item())
        assert bool(legal[pick])
        _, _, done = env.step(RATES[pick])
        steps += 1
    return env, lines, done


def _alloc_env(net, target_keep, passes=6):
    return types.SimpleNamespace(selected_net_path="net.pt", current_model=net.model, target_keep=target_keep,
                                 conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES, passes=passes),
                                 train_loader=None, _input_shape=lambda: SHAPE)


def _planned_state(monkeypatch, net, family, budget, walk_target):
    """``alloc_walk._state`` on ``net`` with its decoder replaced by ``PLANS[family]`` on the budget's cost model
    (decoded at the walk target less the undershoot, as ``_state`` asks): ``(entry, plan widths, info, lines,
    records)``."""
    cm = net.cost(budget)
    made = {}

    def decoded(target):
        made["plan"] = PLANS[family](cm, target)
        return made["plan"]

    if budget == "flops":
        monkeypatch.setattr(plan_agent, "FlopModel", lambda *a, **k: cm)
        monkeypatch.setenv("SPECTRA_ALLOC_BUDGET", "flops")
        monkeypatch.setenv("SPECTRA_EVAL_SIZE_MATCH", f"flop:{walk_target}")
    if family == "agent":
        monkeypatch.setenv("SPECTRA_ALLOC_KIND", "agent")
        monkeypatch.setenv("SPECTRA_PLAN_AGENT", "/x/plan_agent/policy_latest.pt")
        monkeypatch.setattr(plan_agent, "plan_for_env", lambda env, target, path, k_min, **kw: decoded(target))
    else:
        monkeypatch.setenv("SPECTRA_ALLOC_KIND", family)
        monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: [])
        monkeypatch.setattr(alloc_walk, "plan_targets", lambda model, batches, shape, name, target, *a, **k:
                            decoded(target))
    lines, records = [], []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
    entry = alloc_walk._state(_alloc_env(net, walk_target))
    widths, info = made["plan"]
    return entry, widths, info, lines, records


def _legal(row_index=1):
    def at(width):
        mask = fortify.legal_action_mask(RATES, row_index=row_index, alive_count=int(width), device="cpu")
        return [int(i) for i in mask.nonzero(as_tuple=False).flatten().tolist()]
    return at


# ------------------------------------------------------------------ the flag and the grid


def test_flag_default_off_and_parse(monkeypatch):
    assert not fortify.alloc_grid_round()
    for raw, want in (("1", True), ("true", True), ("on", True), ("0", False), ("", False), ("no", False)):
        monkeypatch.setenv("SPECTRA_ALLOC_GRID_ROUND", raw)
        assert fortify.alloc_grid_round() is want


def test_walk_to_follows_choose_and_the_grid_is_where_it_lands_exactly():
    legal = _legal()
    assert alloc_walk.walk_to(512, 390, RATES, legal, 6) == (410, 1)   # one 0.8 cut; 369 is no closer to 390
    assert alloc_walk.walk_to(512, 369, RATES, legal, 6) == (369, 2)   # 0.8 then 0.9
    grid = alloc_walk.grid_widths(512, RATES, legal, 6)
    assert {461, 410, 369, 328} <= set(grid) and not {415, 390} & set(grid)
    assert grid[512] == 0 and grid[461] == grid[410] == 1 and grid[369] == 2
    for width, cuts in grid.items():
        assert alloc_walk.walk_to(512, width, RATES, legal, 6) == (width, cuts) and cuts <= 6
    assert alloc_walk.grid_widths(512, RATES, legal, 1) == {512: 0, 461: 1, 410: 1}
    assert alloc_walk.grid_widths(512, RATES, legal, 0) == {512: 0}


def test_the_grid_honours_the_walks_width_floor_and_the_width_ladder(monkeypatch):
    assert alloc_walk.grid_widths(4, RATES, _legal(), 6) == {4: 0, 3: 1, 2: 2}    # width 2 is identity-only
    monkeypatch.setenv("SPECTRA_MIN_WIDTH_FOR_PRUNE", "3")
    assert alloc_walk.grid_widths(4, RATES, _legal(), 6) == {4: 0, 3: 1}
    monkeypatch.setenv("SPECTRA_FORTIFY", "0")
    assert alloc_walk.grid_widths(4, RATES, _legal(), 6) == {4: 0, 3: 1, 2: 2, 1: 3}
    monkeypatch.delenv("SPECTRA_FORTIFY")
    monkeypatch.delenv("SPECTRA_MIN_WIDTH_FOR_PRUNE")
    monkeypatch.setenv("SPECTRA_WIDTH_LADDER", "8")
    assert alloc_walk.walk_cut(6, 0.8) == 4 != pruning.target_width(6, 0.8)      # the ladder takes two channels
    assert alloc_walk.walk_cut(16, 0.8) == pruning.target_width(16, 0.8)         # wider than the ladder
    grid = alloc_walk.grid_widths(6, RATES, _legal(), 6)
    for width, cuts in grid.items():                                               # the grid follows the env's cut
        assert alloc_walk.walk_to(6, width, RATES, _legal(), 6) == (width, cuts)


def test_walk_grids_reads_the_stem_rows_passes_group_once_and_protected_streams(monkeypatch, vgg, thin):
    walks = vgg.walks()
    assert walks[0]["cap"] == 0 and walks[0]["grid"] == {64: 0}      # VGG's stem row is identity-only (fortify)
    assert all(walks[row]["cap"] == 6 for row in vgg.rows[1:])
    monkeypatch.setenv("SPECTRA_STEM_ROWS", "0")
    assert vgg.walks()[0]["cap"] == 6 and 51 in vgg.walks()[0]["grid"]
    monkeypatch.delenv("SPECTRA_STEM_ROWS")
    streams = [row for group, row in thin.plan if len(group.producers) > 1]
    assert len(streams) == 3
    assert all(walk["cap"] == 2 for walk in thin.walks(passes=2).values())       # group-once: a cut per pass
    monkeypatch.delenv("SPECTRA_GROUP_ONCE_PER_PASS")
    walks = thin.walks(passes=2)
    for row in streams:
        producing = [r for r in thin.walk_rows if thin.group_at[r] == row and r >= fortify.stem_rows()]
        assert walks[row]["cap"] == 2 * len(producing) > 2
    monkeypatch.setenv("SPECTRA_PROTECT_STREAMS", "1")
    walks = thin.walks()
    assert all(walks[row]["cap"] == 0 and walks[row]["grid"] == {thin.pm.widths0[row]: 0} for row in streams)
    assert all(walks[row]["cap"] > 0 for row in thin.rows if row not in streams)


def test_grid_priority_reads_the_decoders_weights():
    rows = [1, 2, 3]
    ones = {row: 1.0 for row in rows}
    for name in ("uniform", "inner", "widths"):
        assert alloc_walk.grid_priority({"kind": name, "sens": ones, "alpha": 0.5}, rows) is None
    sens, cost = {1: 0.2, 2: 1.0, 3: 5.0}, {1: 1.0, 2: 10.0, 3: 1.0}
    assert alloc_walk.grid_priority({"kind": "sens", "sens": sens, "alpha": 0.5}, rows) == \
        pytest.approx(alloc_walk.weights("sens", sens, 0.5))
    assert alloc_walk.grid_priority({"kind": "sens_cost", "sens": sens, "alpha": 0.5, "cost": cost}, rows) == \
        pytest.approx(alloc_walk.weights("sens_cost", sens, 0.5, cost=cost))
    noise = alloc_walk.sample_noise(rows, 0.5, 3)
    around = {"around": "uniform", "sigma": 0.5, "seed": 3}
    assert alloc_walk.grid_priority({"kind": "sample", "sens": ones, "alpha": 0.5, "sample": around}, rows) == \
        pytest.approx(noise)
    around = {"around": "sens", "sigma": 0.5, "seed": 3}
    want = {row: alloc_walk.weights("sens", sens, 0.5)[row] * noise[row] for row in rows}
    assert alloc_walk.grid_priority({"kind": "sample", "sens": sens, "alpha": 0.5, "sample": around}, rows) == \
        pytest.approx(want)
    z = {1: -0.3, 2: 0.4, 3: 0.0}
    assert alloc_walk.grid_priority({"kind": "agent", "sens": z}, rows) == z
    assert alloc_walk.grid_priority({"kind": "agent_sample", "sens": z}, rows) == z


def test_grid_round_widths_steps_by_priority_then_back_up_and_never_past_the_target():
    planned = {1: 10, 2: 10, 3: 10}
    grids = {row: [6, 7, 8, 9, 10] for row in planned}
    total = lambda w: sum(w.values()) / 30.0  # noqa: E731
    # lowest weight first, each group once per round: 2, 1, 3, then 2 and 1 again
    widths, report = alloc_walk.grid_round_widths(planned, grids, total, 0.85, {1: 0.5, 2: 0.1, 3: 0.9})
    assert widths == {1: 8, 2: 8, 3: 9} and report["down"] == 5 and report["up"] == 0
    assert report["kept"] == pytest.approx(25 / 30.0) and report["snapped"] == planned
    # one weight for all: the step that leaves the group closest to its plan relative to it
    widths, report = alloc_walk.grid_round_widths({1: 10, 2: 20}, {1: [7, 8, 9, 10], 2: [14, 16, 18, 20]},
                                                  lambda w: (w[1] + w[2]) / 30.0, 0.8)
    assert widths == {1: 8, 2: 16} and report["down"] == 4 and report["kept"] == pytest.approx(0.8)
    # a step past the band comes back up: highest weight first, never above the target
    widths, report = alloc_walk.grid_round_widths({1: 10, 2: 10}, {1: [5, 10], 2: [9, 10]},
                                                  lambda w: (w[1] + w[2]) / 20.0, 0.9, {1: 1.0, 2: 0.0})
    assert widths == {1: 5, 2: 10} and (report["down"], report["up"]) == (2, 1)
    assert report["kept"] == pytest.approx(0.75)
    # fixed rows keep their plan, floors hold, and an unreachable target is reported, not forced
    widths, report = alloc_walk.grid_round_widths(planned, grids, total, 0.5, None, fixed={3}, floors={1: 8})
    assert widths == {1: 8, 2: 6, 3: 10} and report["kept"] == pytest.approx(0.8)
    # the start is the nearest grid width, ties to the wider
    widths, report = alloc_walk.grid_round_widths({1: 8}, {1: [6, 10]}, lambda w: 1.0, 2.0)
    assert widths == {1: 10} == report["snapped"]


# ------------------------------------------------------------------ the defect, flag off


def test_flag_off_the_uniform_vgg_plan_at_params_0p6_stalls_at_0p642_and_falls_back(monkeypatch, vgg):
    entry, widths, info, lines, records = _planned_state(monkeypatch, vgg, "uniform", "params", 0.6)
    assert entry["widths"] is widths and set(entry) == BASE_KEYS
    assert info["kept"] == pytest.approx(0.581, abs=0.002)                       # the cells' plan line
    assert len(lines) == 1 and [event for event, _k in records] == ["alloc_plan"]
    walks = vgg.walks(aims=widths)
    walked = {row: walks[row]["walk"] for row in widths}
    assert walked[0] == 64 and widths[0] < 64                                   # the stem is planned, never cut
    assert all(walked[row] == 410 and widths[row] == 390 for row in vgg.rows if vgg.pm.widths0[row] == 512)
    assert vgg.pm.kept(walked) == pytest.approx(0.642, abs=0.001)               # x0.642: where the cells stalled
    final, state, done = _walk(vgg, widths, 0.6, vgg.pm)
    assert state["fallback"] and state["stall"] == pytest.approx(vgg.pm.kept(walked))


# ------------------------------------------------------------------ flag on: every plan lands in the band


@pytest.mark.parametrize("budget,walk_target", [("params", 0.6), ("params", 0.47), ("flops", 0.6)])
@pytest.mark.parametrize("family", ["uniform", "sens", "agent"])
def test_flag_on_every_plan_lands_in_the_band_and_the_walk_never_falls_back(monkeypatch, vgg, family, budget,
                                                                            walk_target):
    monkeypatch.setenv("SPECTRA_ALLOC_GRID_ROUND", "1")
    entry, widths, info, lines, records = _planned_state(monkeypatch, vgg, family, budget, walk_target)
    cm = vgg.cost(budget)
    rounded = entry["widths"]
    kept = cm.kept(rounded)
    assert walk_target - alloc_walk.GRID_BAND <= kept <= walk_target + 1e-9
    assert set(rounded) == set(widths) and entry["grid"]["targets"] == rounded
    walks = vgg.walks()
    assert all(rounded[row] in walks[row]["grid"] for row in rounded)
    assert all(rounded[row] >= round(alloc_walk.min_keep() * vgg.pm.widths0[row]) for row in rounded)
    found =[GRID_LINE.match(line) for line in lines if line.startswith("[alloc] grid round")]
    assert len(found) == 1 and found[0] and found[0].group(3) == ("FLOPs" if budget == "flops" else "params")
    assert int(found[0].group(1)) == sum(rounded[row] != widths[row] for row in widths)
    assert float(found[0].group(2)) == pytest.approx(kept, abs=5e-4)
    assert float(found[0].group(4)) == pytest.approx(walk_target, abs=5e-4)
    grid_records = [k for event, k in records if event == "alloc_grid_round"]
    assert len(grid_records) == 1 and grid_records[0]["kept"] == pytest.approx(kept)
    assert grid_records[0]["rows"]["0"]["grid"] == 64 and grid_records[0]["walk_target"] == walk_target
    final, state, done = _walk(vgg, rounded, walk_target, cm)
    assert done and not state["fallback"]
    assert walk_target - alloc_walk.GRID_BAND <= cm.kept(final) <= walk_target + 1e-9
    assert all(final[row] >= rounded[row] for row in rounded)                 # no group goes past its grid width


def test_flag_off_is_the_v17_state_bit_for_bit_and_the_flag_only_adds(monkeypatch, vgg):
    outs = []
    for raw in (None, "0", ""):
        if raw is None:
            monkeypatch.delenv("SPECTRA_ALLOC_GRID_ROUND", raising=False)
        else:
            monkeypatch.setenv("SPECTRA_ALLOC_GRID_ROUND", raw)
        entry, widths, _info, lines, records = _planned_state(monkeypatch, vgg, "sens", "params", 0.6)
        assert entry["widths"] is widths and set(entry) == BASE_KEYS
        assert len(lines) == 1 and [event for event, _k in records] == ["alloc_plan"]
        outs.append((dict(widths), lines, records))
    assert outs[0] == outs[1] == outs[2]
    monkeypatch.setenv("SPECTRA_ALLOC_GRID_ROUND", "1")
    entry, widths, _info, lines, records = _planned_state(monkeypatch, vgg, "sens", "params", 0.6)
    assert dict(widths) == outs[0][0] and lines[0] == outs[0][1][0] and records[0] == outs[0][2][0]
    assert len(lines) == 2 and GRID_LINE.match(lines[1])
    assert [event for event, _k in records] == ["alloc_plan", "alloc_grid_round"]
    assert set(entry) == BASE_KEYS | {"grid"} and entry["widths"] is not widths


def test_the_budget_menu_skips_the_round_with_a_warning(monkeypatch, vgg):
    monkeypatch.setenv("SPECTRA_ALLOC_GRID_ROUND", "1")
    monkeypatch.setenv("SPECTRA_ACTION_MENU", "budget")
    entry, widths, _info, lines, _records = _planned_state(monkeypatch, vgg, "uniform", "params", 0.6)
    assert entry["widths"] is widths and "grid" not in entry
    assert lines[-1].startswith("[alloc] WARNING grid round skipped: SPECTRA_ACTION_MENU=budget")


# ------------------------------------------------------------------ the real walk


def test_the_real_vgg_walk_stalls_at_0p642_off_and_lands_the_rounded_plan_on(monkeypatch, vgg):
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "uniform")
    monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: [])
    monkeypatch.setattr(alloc_walk, "plan_targets", lambda m, b, s, name, target, *a, **k: _uniform(vgg.pm, target))
    env, lines, done = _real_walk(monkeypatch, copy.deepcopy(vgg.model), 0.6)
    fallback = [line for line in lines if "strongest legal cut from here" in line]
    assert done and env._alloc_walk[env.selected_net_path]["fallback"] and len(fallback) == 1
    assert "params x0.642 above the target x0.600" in fallback[0]
    monkeypatch.setenv("SPECTRA_ALLOC_GRID_ROUND", "1")
    env, lines, done = _real_walk(monkeypatch, copy.deepcopy(vgg.model), 0.6)
    state = env._alloc_walk[env.selected_net_path]
    assert done and not state["fallback"]
    assert 0.6 - alloc_walk.GRID_BAND <= env.param_ratio() <= 0.6 + 1e-9
    assert not [line for line in lines if "WARNING" in line or "strongest legal cut" in line]
    now = alloc_walk.group_widths(env.current_model, vgg.rows)
    assert all(now[row] >= state["grid"]["targets"][row] for row in vgg.rows)
    assert sum(now[row] != state["grid"]["targets"][row] for row in vgg.rows) <= 1   # only the landing cut differs


@pytest.mark.parametrize("kind_name,target,flops", [("uniform", 0.6, False), ("sens", 0.6, False),
                                                    ("inner", 0.6, False), ("uniform", 0.47, False),
                                                    ("sens", 0.47, False), ("uniform", 0.6, True)])
def test_the_real_walk_on_the_thin_resnet_follows_the_rounded_plan(monkeypatch, thin, kind_name, target, flops):
    monkeypatch.setenv("SPECTRA_ALLOC_GRID_ROUND", "1")
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", kind_name)
    sens = {row: 0.01 * (1 + k) for k, row in enumerate(thin.rows)}
    monkeypatch.setattr(group_sensitivity, "group_sensitivity", lambda m, plan, b, shape, keep=0.5, costs=None:
                        (sens, 1.0))
    env, lines, done = _real_walk(monkeypatch, copy.deepcopy(thin.model), target, flops=flops)
    state = env._alloc_walk[env.selected_net_path]
    kept = env.flops_ratio() if flops else env.param_ratio()
    assert done and not state["fallback"] and kept <= target + 1e-9
    assert not [line for line in lines if "WARNING" in line or "strongest legal cut" in line]
    found = [GRID_LINE.match(line) for line in lines if line.startswith("[alloc] grid round")]
    assert len(found) == 1 and found[0]
    predicted = float(found[0].group(2))
    assert target - alloc_walk.GRID_BAND - 5e-4 <= predicted <= target + 5e-4
    assert predicted - 5e-4 <= kept                                              # the walk ends at or above the plan
    now = alloc_walk.group_widths(env.current_model, thin.rows)
    assert all(now[row] >= state["grid"]["targets"][row] for row in thin.rows)
    if kind_name == "inner":
        assert all(now[row] == thin.pm.widths0[row] == state["grid"]["targets"][row]
                   for group, row in thin.plan if len(group.producers) > 1)


def test_the_plan_floor_holds_through_the_round(monkeypatch, thin):
    monkeypatch.setenv("SPECTRA_ALLOC_GRID_ROUND", "1")
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "uniform")
    monkeypatch.setenv("SPECTRA_PLAN_MIN_WIDTH", "3")
    monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: [])
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    entry = alloc_walk._state(_alloc_env(thin, 0.35))
    assert all(width >= 3 for width in entry["widths"].values())
    assert lines[0].endswith("; group width floor 3") and GRID_LINE.match(lines[1])
    assert thin.pm.kept(entry["widths"]) <= 0.35 + 1e-9


# ------------------------------------------------------------------ the walk-time check


def test_the_watch_warns_once_per_group_below_its_grid_width_and_once_on_a_fallback(monkeypatch, thin):
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    rows = thin.rows
    widths = dict(thin.pm.widths0)
    widths[rows[1]] -= 1
    widths[rows[4]] -= 2
    model = alloc_walk.cut_to(thin.model, thin.plan, plan_agent.rates_of(widths, thin.pm.widths0), SHAPE)
    kept = [0.9]
    env = types.SimpleNamespace(current_model=model, param_ratio=lambda: kept[0])
    targets = dict(thin.pm.widths0)
    targets[rows[1]] = widths[rows[1]]                                           # at its grid width: silent
    entry = {"fallback": False, "grid": {"targets": targets, "checked": 1.0, "below": set(), "fallback": False}}
    alloc_walk._grid_watch(env, entry)
    assert lines == [f"[alloc] WARNING grid round: the walk took the group at row {rows[4]} to width "
                     f"{widths[rows[4]]}, below its grid width {targets[rows[4]]}"]
    alloc_walk._grid_watch(env, entry)                                           # nothing cut since: not re-read
    kept[0] = 0.8
    alloc_walk._grid_watch(env, entry)                                           # re-read; row already reported
    assert len(lines) == 1
    entry["fallback"] = True
    alloc_walk._grid_watch(env, entry)
    alloc_walk._grid_watch(env, entry)
    assert len(lines) == 2 and lines[1].startswith("[alloc] WARNING grid round: the stall fallback fired with 1 "
                                                   f"groups off their grid widths (row {rows[4]}: ")


def test_grid_priority_residual_agent_ranks_by_prior_times_exp_z():
    """A T2 residual plan's effective weight is w_prior * exp(z): the grid steps by log(w) + z, not z alone."""
    import math
    from src import alloc_walk
    rows = [3, 7, 11]
    z = {3: 0.5, 7: 0.0, 11: -0.2}
    prior = {3: 0.1, 7: 1.0, 11: 2.0}
    info = {"kind": "agent", "sens": z, "residual": "sens", "prior": {"weights": prior}}
    assert alloc_walk.grid_priority(info, rows) == {r: math.log(prior[r]) + z[r] for r in rows}
    assert alloc_walk.grid_priority({"kind": "agent", "sens": z}, rows) == z
