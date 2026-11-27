"""
SAMENET (queue row SAMENET, 26 Nov 2026; every piece default off):

* ``SPECTRA_ALLOC_KIND=tilt`` (``alloc_walk.tilt_plan``): a base plan given as a widths table, tilted by one score per group
  (w_g = base keep_g · exp(z_g)), the scale bisected to the base plan's own size with c = 1 first, so z = 0 is the base plan;
* ``SPECTRA_ALLOC_TILT_PARAMS=equal[:tol][:landed|planned]`` (amendments A1 and A2 of the row; ``alloc_walk.tilt_params`` /
  ``TiltHold`` / ``TiltWalk`` / ``_tilt_held``): the tilt also held at the base plan's params, by a λ along the per-group
  FLOPs-per-param direction; what is held is the params of the network the walk is predicted to land on (``landed``, the
  default: the walk run on the cost models) or of the plan it aims at (``planned``); λ is 0 exactly when the tilt is within
  the tolerance as it is, and with the flag off the tilt is the tilt before the amendments;
* ``SPECTRA_EVAL_SAVE_WIDTHS=1`` (``alloc_walk.save_widths``, called by the runner's ``_run_final_ft``): the size point's planned
  and cut widths as tables ``widths`` / ``tilt`` read; the planned table walks to the same network again, bit for bit;
* ``SPECTRA_EVAL_FINAL_FT_SIZE_ONLY=1`` (v34, already in the tree): the size point alone is fine-tuned, also when ``val_best``
  sits on the same step;
* ``SPECTRA_PLAN_MENU_SAMPLE=1`` (``plan_agent.menu_sample``): a menu policy's sampled plan, drawn as its trainer draws;
* the search's arithmetic in ``scripts/_tmp_s26nov_samenet_step.py`` (PG's update, EVO's parent, the equal-size rule, the pick);
* with every flag off, ``alloc_walk`` and ``plan_for_env`` give what the untouched tree gives, for every existing kind.

CPU only, no datasets.  OMP_NUM_THREADS=4 python -m pytest tests/test_samenet.py -v
It takes tens of minutes (real walks, and the last test runs a script in two interpreters): run it on partition cpu, not
on the login node. The search's tests read the driver's step (scripts/_tmp_s26nov_samenet_step.py here, or
s26nov_samenet_step.py beside the tree, or SAMENET_STEP); the last test compares with SAMENET_BASELINE_TREE
(default /home/paretsky/scratch_audit/_samenet_base, tree_v41 + commit 997adbd, untouched) and is skipped without one.
"""

import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows  # noqa: E402
from src import alloc_walk, fortify, plan_agent, plan_trainer  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402
import a2c_agent_reinforce_runner as runner  # noqa: E402
from spectra_models_instantiation import thin_res_net  # noqa: E402
from tests.test_plan_menu import (MENU, MENU_FLAG, MENU_META, RES, RESIDUAL_META, SETTINGS, _hot_policy, _origin, _save,  # noqa: E402
                                  _seed_all, v10_env)  # noqa: E402,F401
from tests.test_v10_fixed_target import _target_env  # noqa: E402
from tests.test_v25_plan_resume import _equal  # noqa: E402

SHAPE = (3, 32, 32)
RATES = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
LABEL = "size_flop0.20"
FLAG_PREFIXES = ("SPECTRA_ALLOC_", "SPECTRA_PLAN_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_", "SPECTRA_MIN_WIDTH",
                 "SPECTRA_GROUP_ONCE", "SPECTRA_STEM_ROWS", "SPECTRA_WIDTH_LADDER", "SPECTRA_ACTION_", "SPECTRA_PROTECT_STREAMS",
                 "SPECTRA_FORTIFY", "SPECTRA_SENS_", "SPECTRA_CUT_RANK")
# the registered cells' walk: a FLOPs budget and target, group-once, the grid rounding, the walk's width floor
CELL_ENV = {"SPECTRA_FIXED_TARGET": "1", "SPECTRA_FIXED_TARGET_METRIC": "flop", "SPECTRA_ALLOC_BUDGET": "flops",
            "SPECTRA_GROUP_ONCE_PER_PASS": "1", "SPECTRA_ALLOC_GRID_ROUND": "1", "SPECTRA_PLAN_MIN_WIDTH": "walk",
            "SPECTRA_EVAL_SIZE_MATCH": "flop:0.2", "SPECTRA_EVAL_POLICY": "alloc"}
_MODELS, _BASES = {}, {}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


# ------------------------------------------------------------------ helpers


def _model(width=12, classes=100):
    """A thin ResNet-20 at random init (width 12 with a CIFAR-100 head is the smokes' architecture), built once."""
    if (width, classes) not in _MODELS:
        torch.manual_seed(0)
        _MODELS[(width, classes)] = thin_res_net.resnet20(num_classes=classes, large_input=False, width=width).eval()
    return _MODELS[(width, classes)]


def _plan(model):
    return group_sensitivity.group_plan(ModelWithRows(model))


def _table(model, widths):
    """``{producer name: width}``: a plan's widths as the table ``widths`` and ``tilt`` read."""
    return alloc_walk.plan_table(model, _plan(model), widths)


def _write(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _fake_sens(model):
    rows = [row for _g, row in _plan(model)]
    return {row: 0.01 * (1 + (7 * k) % 5) for k, row in enumerate(rows)}


def _base_plan(monkeypatch, model, kind_name="sens", target=0.3, budget="flops", min_width=2):
    """A base plan for the unit tests: a rule's own plan (fake sensitivities), as ``(widths, info)``."""
    sens = _fake_sens(model)
    monkeypatch.setattr(group_sensitivity, "group_sensitivity", lambda m, plan, b, shape, keep=0.5, costs=None: (sens, 1.0))
    return alloc_walk.plan_targets(model, [], SHAPE, kind_name, target, budget=budget, min_width=min_width)


def _tilt(monkeypatch, model, base_path, target=0.3, budget="flops", min_width=2, **flags):
    for key in ("SPECTRA_ALLOC_TILT_Z", "SPECTRA_ALLOC_TILT_SIGMA", "SPECTRA_ALLOC_TILT_TARGET", "SPECTRA_ALLOC_SAMPLE_SEED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SPECTRA_ALLOC_TILT_BASE", base_path)
    for key, value in flags.items():
        monkeypatch.setenv(key, str(value))
    return alloc_walk.plan_targets(model, None, SHAPE, "tilt", target, budget=budget, min_width=min_width)


def _walk(origin, flags, seed, target=0.2, limit=1500):
    """The registered cell's walk on a copy of ``origin`` under ``flags`` (``alloc_walk.action`` on a ``NetworkEnv`` that really
    cuts: recovery skipped, val scripted, the env's own group-once lock and landing): ``(env, lines, records, done)``."""
    with pytest.MonkeyPatch.context() as mp:
        for key in list(os.environ):
            if key.startswith(FLAG_PREFIXES):
                mp.delenv(key, raising=False)
        for key, value in {**CELL_ENV, "SPECTRA_EVAL_SIZE_MATCH": f"flop:{target:g}", **flags}.items():
            mp.setenv(key, str(value))
        lines, records = [], []
        mp.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
        mp.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
        torch.manual_seed(seed)
        model = copy.deepcopy(origin)
        env, _seen = _target_env(model, target=target, accs=[0.9] * (limit + 10), passes=6)
        del env._register_group_lock
        env.conf.compression_rates_dict = RATES
        env.conf.seed = seed
        env.original_flops = utils.calc_flops(model, SHAPE)
        env.data_dict = {env.selected_net_path: (copy.deepcopy(origin), None)}
        done, steps = False, 0
        while not done and steps < limit:
            legal = env.legal_action_mask(device="cpu")
            pick = int(alloc_walk.action(env, legal, RATES, "cpu").item())
            assert bool(legal[pick])
            _, _, done = env.step(RATES[pick])
            steps += 1
        env._steps = steps
        return env, lines, records, done


def _plan_only(origin, flags, target=0.2):
    """``alloc_walk._state`` under the registered cell's flags plus ``flags`` on a copy of ``origin``: the plan and its grid
    rounding as the walk's first decision makes them, without the walk. ``(entry, lines, records)``."""
    with pytest.MonkeyPatch.context() as mp:
        for key in list(os.environ):
            if key.startswith(FLAG_PREFIXES):
                mp.delenv(key, raising=False)
        for key, value in {**CELL_ENV, "SPECTRA_EVAL_SIZE_MATCH": f"flop:{target:g}", **flags}.items():
            mp.setenv(key, str(value))
        lines, records = [], []
        mp.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
        mp.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
        env = types.SimpleNamespace(selected_net_path="/zoo/tiny.pt", current_model=copy.deepcopy(origin), target_keep=target,
                                    conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES, passes=6),
                                    train_loader=None, _input_shape=lambda: SHAPE)
        return alloc_walk._state(env), lines, records


def _candidate(env):
    point = {"step": env._steps, "param": float(env.param_ratio()), "flop": float(env.flops_ratio())}
    return {"point": point, "model": copy.deepcopy(env.current_model).cpu(), "key": None}


def _base_walk(tmp_factory, kind_name="sens_thresh", width=12, classes=100, target=0.2):
    """One base walk per setting, shared by the tests: the env after the walk, its lines and the folder of its saved widths."""
    key = (kind_name, width, classes, target)
    if key not in _BASES:
        origin = _model(width, classes)
        env, lines, _records, done = _walk(origin, {"SPECTRA_ALLOC_KIND": kind_name}, seed=1, target=target)
        assert done
        folder = str(tmp_factory.mktemp("samenet_base"))
        with pytest.MonkeyPatch.context() as mp:
            for k, v in {**CELL_ENV, "SPECTRA_ALLOC_KIND": kind_name}.items():
                mp.setenv(k, v)
            mp.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
            meta = alloc_walk.save_widths(env, env.selected_net_path, LABEL, _candidate(env), folder)
        assert meta is not None
        stem = os.path.join(folder, f"{os.path.basename(env.selected_net_path)}__{LABEL}")
        _BASES[key] = types.SimpleNamespace(origin=origin, env=env, lines=lines, folder=folder, meta=meta, target=target,
                                            plan=stem + "__plan.json", cut=stem + "__cut.json", meta_path=stem + "__meta.json")
    return _BASES[key]


def _step_module():
    """The driver's Python step (the search's arithmetic lives there): from the repo's scripts, or where it is deployed."""
    for cand in (os.environ.get("SAMENET_STEP"), ROOT / "scripts" / "_tmp_s26nov_samenet_step.py",
                 ROOT.parent / "s26nov_samenet_step.py", "/home/paretsky/scratch_audit/s26nov_samenet_step.py"):
        if cand and os.path.exists(str(cand)):
            spec = importlib.util.spec_from_file_location("samenet_step", str(cand))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    pytest.skip("scripts/_tmp_s26nov_samenet_step.py is not beside this tree (set SAMENET_STEP=<path>)")


# ------------------------------------------------------------------ the tilt kind: flags, keys, scores


def test_tilt_is_registered_beside_the_pinned_kinds_and_its_flags_parse(monkeypatch, tmp_path):
    assert alloc_walk.TILT_KINDS == ("tilt",) and "tilt" not in alloc_walk.KINDS
    assert alloc_walk.KINDS[-2:] == alloc_walk.MAGNITUDE_KINDS                # the pinned tuple is as it was
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "Tilt")
    assert alloc_walk.kind() == "tilt"
    with pytest.raises(ValueError, match="SPECTRA_ALLOC_TILT_BASE"):
        alloc_walk.tilt_base_table()
    assert alloc_walk.tilt_target() == "base" and alloc_walk.tilt_sigma() is None
    monkeypatch.setenv("SPECTRA_ALLOC_TILT_TARGET", "RUN")
    assert alloc_walk.tilt_target() == "run"
    monkeypatch.setenv("SPECTRA_ALLOC_TILT_TARGET", "walk")
    with pytest.raises(ValueError, match="expected one of"):
        alloc_walk.tilt_target()
    for raw, want in (("0", 0.0), ("0.3", 0.3), (" 1e-1 ", 0.1)):
        monkeypatch.setenv("SPECTRA_ALLOC_TILT_SIGMA", raw)
        assert alloc_walk.tilt_sigma() == pytest.approx(want)
    for raw in ("-0.1", "nan", "inf", "x"):
        monkeypatch.setenv("SPECTRA_ALLOC_TILT_SIGMA", raw)
        with pytest.raises(ValueError, match="at least 0"):
            alloc_walk.tilt_sigma()
    path = _write(tmp_path / "base.json", {"a.conv": 8, "b.conv": "5"})
    monkeypatch.setenv("SPECTRA_ALLOC_TILT_BASE", path)
    assert alloc_walk.tilt_base_table() == {"a.conv": 8, "b.conv": 5} and alloc_walk.tilt_base_path() == path


def test_group_keys_are_the_sorted_producer_names_and_do_not_follow_the_plan_order():
    model = _model(4, 10)
    plan = _plan(model)
    keys = alloc_walk.group_keys(model, plan)
    names = {id(m): n for n, m in model.named_modules()}
    assert list(keys) == [row for _g, row in plan] and len(set(keys.values())) == len(keys)
    for group, row in plan:
        assert keys[row] == "+".join(sorted(names[id(p)] for p in group.producers))
    assert alloc_walk.group_keys(model, list(reversed(plan))) == keys           # a name, not a position
    streams = [keys[row] for group, row in plan if len(group.producers) > 1]
    assert len(streams) == 3 and all("+" in key for key in streams)             # one residual stream per stage
    twin = copy.deepcopy(model)
    assert alloc_walk.group_keys(twin, _plan(twin)) == keys                     # the same names on another copy of the network


def test_tilt_draw_is_seeded_scales_with_sigma_and_leaves_the_global_rng_alone():
    torch.manual_seed(123)
    before = alloc_walk.rng_snapshot()
    a = alloc_walk.tilt_draw(12, 0.3, 7)
    assert _equal(alloc_walk.rng_snapshot(), before)
    assert a == alloc_walk.tilt_draw(12, 0.3, 7) and a != alloc_walk.tilt_draw(12, 0.3, 8)
    unit = alloc_walk.tilt_draw(12, 1.0, 7)
    assert a == pytest.approx([0.3 * v for v in unit]) and alloc_walk.tilt_draw(12, 0.0, 7) == [0.0] * 12
    eps = torch.randn(12, generator=torch.Generator().manual_seed(7), dtype=torch.float64).tolist()
    assert unit == eps                                                          # agent_sample's generator and dtype
    assert abs(float(np.std(alloc_walk.tilt_draw(4000, 0.3, 1))) - 0.3) < 0.02


def test_tilt_scores_come_from_a_list_a_keyed_dict_or_a_seeded_draw_and_nothing_else(monkeypatch, tmp_path):
    model = _model(4, 10)
    plan = _plan(model)
    rows = [row for _g, row in plan]
    keys = alloc_walk.group_keys(model, plan)
    with pytest.raises(ValueError, match="exactly one of"):
        alloc_walk.tilt_scores(rows, keys)                                      # neither
    values = [0.1 * (k - 5) for k in range(len(rows))]
    monkeypatch.setenv("SPECTRA_ALLOC_TILT_Z", _write(tmp_path / "list.json", values))
    z, source = alloc_walk.tilt_scores(rows, keys)
    assert [z[row] for row in rows] == values and source == {"z": "file", "path": str(tmp_path / "list.json")}
    by_key = {keys[row]: v for row, v in zip(reversed(rows), reversed(values))}  # any order in the file
    monkeypatch.setenv("SPECTRA_ALLOC_TILT_Z", _write(tmp_path / "dict.json", by_key))
    assert alloc_walk.tilt_scores(rows, keys)[0] == z
    monkeypatch.setenv("SPECTRA_ALLOC_TILT_SIGMA", "0.3")
    with pytest.raises(ValueError, match="exactly one of"):
        alloc_walk.tilt_scores(rows, keys)                                      # both
    monkeypatch.delenv("SPECTRA_ALLOC_TILT_Z")
    monkeypatch.setenv("SPECTRA_ALLOC_SAMPLE_SEED", "5")
    z, source = alloc_walk.tilt_scores(rows, keys)
    assert [z[row] for row in rows] == alloc_walk.tilt_draw(len(rows), 0.3, 5)
    assert source == {"z": "draw", "sigma": 0.3, "seed": 5}
    monkeypatch.delenv("SPECTRA_ALLOC_TILT_SIGMA")
    for name, doc, match in (("short.json", values[:-1], "scores for"), ("text.json", "0.1", "a JSON list or object"),
                             ("missing.json", dict(list(by_key.items())[1:]), "1 planned groups not named"),
                             ("unknown.json", {**by_key, "nowhere.conv": 0.0}, "1 names that are no planned group"),
                             ("nan.json", [float("nan")] + values[1:], "is not finite")):
        monkeypatch.setenv("SPECTRA_ALLOC_TILT_Z", _write(tmp_path / name, doc))
        with pytest.raises(ValueError, match=match):
            alloc_walk.tilt_scores(rows, keys)


# ------------------------------------------------------------------ the tilt kind: the plan


@pytest.mark.parametrize("budget,min_width", [("flops", 2), ("flops", 1), ("params", 2)])
@pytest.mark.parametrize("kind_name", ["sens", "uniform"])
def test_tilt_at_zero_is_the_base_plan_exactly(monkeypatch, tmp_path, kind_name, budget, min_width):
    model = _model(8, 100)
    base, base_info = _base_plan(monkeypatch, model, kind_name, 0.3, budget, min_width)
    path = _write(tmp_path / "base.json", _table(model, base))
    rows = list(base)
    keys = alloc_walk.group_keys(model, _plan(model))
    for flags in ({"SPECTRA_ALLOC_TILT_SIGMA": "0"}, {"SPECTRA_ALLOC_TILT_SIGMA": "0", "SPECTRA_ALLOC_SAMPLE_SEED": "9"},
                  {"SPECTRA_ALLOC_TILT_Z": _write(tmp_path / "z_list.json", [0.0] * len(rows))},
                  {"SPECTRA_ALLOC_TILT_Z": _write(tmp_path / "z_dict.json", {keys[row]: 0.0 for row in rows})}):
        widths, info = _tilt(monkeypatch, model, path, 0.3, budget, min_width, **flags)
        assert widths == base                                                   # every group, exactly
        tilt = info["tilt"]
        assert tilt["moved"] == 0 and tilt["c"] == 1.0 and tilt["base_widths"] == base and tilt["widths"] == base
        assert info["kept"] == tilt["base_kept"] == info["target"] == tilt["aim"]
        assert info["kept"] == pytest.approx(base_info["kept"], abs=1e-12)      # the rule's own plan, the same cut
        assert info["kind"] == "tilt" and info["held"] == 0 and tilt["unnamed"] == 0 and tilt["target"] == "base"
        assert info["weights"] == tilt["base_keeps"] == {row: base[row] / float(info["origin_widths"][row]) if
                                                          base[row] < info["origin_widths"][row] else 1.0 for row in rows}
        assert set(info["sens"].values()) == {0.0} and info["cost_budget"] == budget
    # the run's own target at z = 0 is the base plan scaled to that target, not the base plan
    run_w, run_info = _tilt(monkeypatch, model, path, 0.2, budget, min_width, SPECTRA_ALLOC_TILT_SIGMA="0",
                            SPECTRA_ALLOC_TILT_TARGET="run")
    assert run_info["target"] == 0.2 and run_info["tilt"]["target"] == "run" and run_info["tilt"]["c"] < 1.0
    assert run_info["kept"] == pytest.approx(0.2, abs=0.03) and run_info["tilt"]["moved"] > 0 and run_w != base
    assert all(run_w[row] <= base[row] for row in rows)


def test_tilt_holds_a_base_width_under_the_floor_and_reads_an_unnamed_group_as_whole(monkeypatch, tmp_path):
    model = _model(8, 100)
    base, _info = _base_plan(monkeypatch, model, "sens", 0.3, "flops", 2)
    rows = list(base)
    origin = alloc_walk.group_widths(model, rows)
    low = dict(base)
    low[rows[1]] = 1                                                            # under both floors (2 channels, keep 0.1)
    table = _table(model, low)
    names = {id(m): n for n, m in model.named_modules()}
    last = _plan(model)[-1][0]
    for producer in last.producers:                                             # the last group is not named at all
        table.pop(names[id(producer)])
    path = _write(tmp_path / "low.json", table)
    widths, info = _tilt(monkeypatch, model, path, 0.3, "flops", 2, SPECTRA_ALLOC_TILT_SIGMA="0")
    assert widths[rows[1]] == 1 and info["tilt"]["floor"][rows[1]] == pytest.approx(1.0 / origin[rows[1]])
    assert info["tilt"]["unnamed"] == 1 and info["tilt"]["base_keeps"][rows[-1]] == 1.0 and widths[rows[-1]] == origin[rows[-1]]
    assert info["tilt"]["moved"] == 0 and all(widths[row] == low[row] for row in rows[:-1])
    assert "unnamed" not in info                                                # not the widths kind's "kept whole" count
    with pytest.raises(ValueError, match="names no planned group"):             # another network's table is refused
        _tilt(monkeypatch, model, _write(tmp_path / "other.json", {"nowhere.conv": 8}), 0.3, "flops", 2,
              SPECTRA_ALLOC_TILT_SIGMA="0")
    # every other group keeps the rule's floor under a tilt that pushes it down hard
    z = [-3.0 if k % 2 else 3.0 for k in range(len(rows))]
    widths, info = _tilt(monkeypatch, model, path, 0.3, "flops", 2, SPECTRA_ALLOC_TILT_Z=_write(tmp_path / "z.json", z))
    for row in rows:
        floor_width = max(1, min(origin[row], 2), int(round(0.1 * origin[row])))
        assert widths[row] >= min(floor_width, low[row] if row != rows[-1] else origin[row])


@pytest.mark.parametrize("budget", ["flops", "params"])
def test_tilt_bisects_to_the_base_plans_size_and_the_weights_are_base_keep_times_exp_z(monkeypatch, tmp_path, budget):
    model = _model(12, 100)
    base, _base_info = _base_plan(monkeypatch, model, "sens", 0.2, budget, 2)
    path = _write(tmp_path / "base.json", _table(model, base))
    rows = list(base)
    size = (lambda m: utils.calc_flops(m, SHAPE)) if budget == "flops" else utils.calc_num_parameters
    for seed in (1, 2, 3, 4):
        widths, info = _tilt(monkeypatch, model, path, 0.2, budget, 2, SPECTRA_ALLOC_TILT_SIGMA="0.3",
                             SPECTRA_ALLOC_SAMPLE_SEED=seed)
        tilt = info["tilt"]
        z = alloc_walk.tilt_draw(len(rows), 0.3, seed)
        assert [info["sens"][row] for row in rows] == z == [tilt["z"][row] for row in rows]
        assert info["weights"] == {row: tilt["base_keeps"][row] * math.exp(v) for row, v in zip(rows, z)}
        assert info["keeps"] == {row: min(1.0, max(tilt["floor"][row], tilt["c"] * info["weights"][row])) for row in rows}
        assert abs(info["kept"] - tilt["base_kept"]) <= 0.01                    # on the base's size (tol 0.003, integer widths)
        assert tilt["moved"] >= len(rows) // 3 and widths != base
        cut = alloc_walk.cut_to(model, _plan(model), info["keeps"], SHAPE)      # the plan is the real one-shot cut
        assert size(cut) / size(model) == pytest.approx(info["kept"], abs=1e-12)
        assert alloc_walk.group_widths(cut, rows) == widths
        assert utils.calc_num_parameters(cut) / utils.calc_num_parameters(model) == pytest.approx(tilt["params"], abs=1e-12)
        assert utils.calc_flops(cut, SHAPE) / utils.calc_flops(model, SHAPE) == pytest.approx(tilt["flops"], abs=1e-12)
        again, info2 = _tilt(monkeypatch, model, path, 0.2, budget, 2, SPECTRA_ALLOC_TILT_SIGMA="0.3",
                             SPECTRA_ALLOC_SAMPLE_SEED=seed)
        assert again == widths and info2["tilt"]["c"] == tilt["c"]              # the same scores, the same plan
        keys = alloc_walk.group_keys(model, _plan(model))
        by_key = _tilt(monkeypatch, model, path, 0.2, budget, 2,
                       SPECTRA_ALLOC_TILT_Z=_write(tmp_path / f"z{seed}.json", {keys[row]: v for row, v in zip(rows, z)}))
        assert by_key[0] == widths and by_key[1]["tilt"]["source"]["z"] == "file"
        # the run's own target instead: sens' loop on the tilted weights, to the target it was handed
        run_w, run_info = _tilt(monkeypatch, model, path, 0.2, budget, 2, SPECTRA_ALLOC_TILT_SIGMA="0.3",
                                SPECTRA_ALLOC_SAMPLE_SEED=seed, SPECTRA_ALLOC_TILT_TARGET="run")
        assert run_info["target"] == 0.2 and abs(run_info["kept"] - 0.2) <= 0.01
    # a single score moves its own group against the others, and the budget stays
    origin = alloc_walk.group_widths(model, rows)
    j = next(k for k, row in enumerate(rows) if base[row] <= 0.6 * origin[row])
    one = [0.0] * len(rows)
    one[j] = 1.0
    widths, info = _tilt(monkeypatch, model, path, 0.2, budget, 2, SPECTRA_ALLOC_TILT_Z=_write(tmp_path / "one.json", one))
    assert widths[rows[j]] > base[rows[j]] and info["tilt"]["c"] < 1.0
    assert all(widths[row] <= base[row] for row in rows if row != rows[j])
    assert abs(info["kept"] - info["tilt"]["base_kept"]) <= 0.01


def test_the_grid_rounding_steps_a_tilt_by_its_weights():
    rows = [1, 2, 3]
    info = {"kind": "tilt", "sens": {1: 0.2, 2: -0.1, 3: 0.0}, "alpha": 0.5, "weights": {1: 0.61, 2: 0.27, 3: 1.0}}
    assert alloc_walk.grid_priority(info, rows) == {1: 0.61, 2: 0.27, 3: 1.0}


# ------------------------------------------------------------------ the walk: one network, again and again


def test_the_state_logs_the_tilt_records_it_and_draws_no_batch(tmp_path_factory, monkeypatch):
    base = _base_walk(tmp_path_factory)
    for key, value in {**CELL_ENV, "SPECTRA_ALLOC_KIND": "tilt", "SPECTRA_ALLOC_TILT_BASE": base.plan,
                       "SPECTRA_ALLOC_TILT_SIGMA": "0.3", "SPECTRA_ALLOC_SAMPLE_SEED": "4"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: pytest.fail("tilt draws no batch"))
    lines, records = [], []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setattr(recorder, "record", lambda event, **k: records.append((event, k)))
    env = types.SimpleNamespace(selected_net_path="/zoo/net.pt", current_model=copy.deepcopy(base.origin), target_keep=0.2,
                                conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES, passes=6),
                                train_loader=None, _input_shape=lambda: SHAPE)
    entry = alloc_walk._state(env)
    name = os.path.basename(base.plan)
    assert lines[0].startswith(f"[alloc] net.pt: tilt of {name}, z drawn sigma=0.3 seed=4 plan keeps x")
    assert "= the base plan's size) over 12 groups; group keep min" in lines[0] and lines[0].endswith("; group width floor 2")
    assert lines[1].startswith(f"[alloc] tilt net.pt: base {name} keeps x") and "z drawn sigma=0.3 seed=4, |z| max " in lines[1]
    assert "bisected to x" in lines[1] and "(the base plan); the plan keeps params x" in lines[1]
    assert " groups moved from the base" in lines[1]
    groups = [line for line in lines if line.startswith("[alloc] tilt net.pt group ")]
    keys = [g["key"] for g in base.meta["groups"]]
    z = alloc_walk.tilt_draw(12, 0.3, 4)
    assert len(groups) == 12
    for i, (line, g) in enumerate(zip(groups, base.meta["groups"])):            # the plan order, the key, z, the keeps, the width
        assert line.startswith(f"[alloc] tilt net.pt group {i} row {g['row']} {keys[i]}: z {z[i]:+.4f} base keep ")
        assert line.endswith(f"(base {g['plan']}, origin {g['origin']})") and " keep " in line and " width " in line
    assert lines[-1].startswith("[alloc] grid round: ")
    events = [event for event, _k in records]
    assert events == ["alloc_plan", "alloc_tilt", "alloc_grid_round"]
    tilt = records[1][1]
    assert tilt["base"] == base.plan and tilt["source"] == {"z": "draw", "sigma": 0.3, "seed": 4} and tilt["target"] == "base"
    assert [g["key"] for g in tilt["groups"]] == keys and [g["z"] for g in tilt["groups"]] == z
    assert [g["base_width"] for g in tilt["groups"]] == [g["plan"] for g in base.meta["groups"]]
    assert records[0][1]["kind"] == "tilt" and set(entry) == {"widths", "n_rows", "idle", "fallback", "last_kept", "grid"}


def test_the_z0_tilt_and_the_widths_final_cut_the_base_network_bit_for_bit_under_other_seeds(tmp_path_factory):
    base = _base_walk(tmp_path_factory)
    env0 = base.env
    assert env0.flops_ratio() <= 0.2 + 1e-9 and env0.flops_ratio() >= 0.2 - 0.02
    assert not env0._alloc_walk[env0.selected_net_path]["fallback"]
    assert not [line for line in base.lines if "strongest legal cut" in line or "WARNING" in line]
    targets = env0._alloc_walk[env0.selected_net_path]["widths"]
    digest = alloc_walk.state_digest(env0.current_model)
    cut = alloc_walk.cut_table(env0.current_model)
    assert base.meta["state_sha1"] == digest and json.load(open(base.cut)) == cut
    for flags, seed in (({"SPECTRA_ALLOC_KIND": "tilt", "SPECTRA_ALLOC_TILT_BASE": base.plan, "SPECTRA_ALLOC_TILT_SIGMA": "0"}, 7),
                        ({"SPECTRA_ALLOC_KIND": "widths", "SPECTRA_ALLOC_WIDTHS": base.plan}, 11),
                        ({"SPECTRA_ALLOC_KIND": "tilt", "SPECTRA_ALLOC_TILT_BASE": base.plan,
                          "SPECTRA_ALLOC_TILT_Z": _zeros(base)}, 23)):
        env, lines, records, done = _walk(base.origin, flags, seed)
        entry = env._alloc_walk[env.selected_net_path]
        assert done and not entry["fallback"]
        assert entry["widths"] == targets                                       # the walk aims at the same widths
        assert alloc_walk.cut_table(env.current_model) == cut                   # every module's width
        assert alloc_walk.state_digest(env.current_model) == digest             # the same channels and weights
        assert env.flops_ratio() == env0.flops_ratio() and env.param_ratio() == env0.param_ratio()
        assert env._steps == env0._steps
        grid = [line for line in lines if line.startswith("[alloc] grid round: ")]
        assert len(grid) == 1 and grid[0].startswith("[alloc] grid round: 0 groups moved;")   # the plan is its own rounding
        assert not [line for line in lines if "strongest legal cut" in line or "WARNING" in line]
        if flags["SPECTRA_ALLOC_KIND"] == "tilt":
            assert " 0 of 12 groups moved from the base" in next(line for line in lines if line.startswith("[alloc] tilt tiny"))


def _zeros(base):
    path = os.path.join(base.folder, "z_zero.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({g["key"]: 0.0 for g in base.meta["groups"]}, fh)
    return path


def test_random_tilts_walk_to_the_target_without_a_fallback_and_differ_from_the_base(tmp_path_factory):
    base = _base_walk(tmp_path_factory)
    digest = alloc_walk.state_digest(base.env.current_model)
    seen = set()
    for seed in (1, 2, 3):
        flags = {"SPECTRA_ALLOC_KIND": "tilt", "SPECTRA_ALLOC_TILT_BASE": base.plan, "SPECTRA_ALLOC_TILT_SIGMA": "0.3",
                 "SPECTRA_ALLOC_SAMPLE_SEED": str(seed)}
        env, lines, _records, done = _walk(base.origin, flags, 100 + seed)
        entry = env._alloc_walk[env.selected_net_path]
        assert done and not entry["fallback"] and not [line for line in lines if "strongest legal cut" in line]
        assert 0.2 - 0.0051 <= env.flops_ratio() <= 0.2 + 1e-9                  # the gate's landing band
        now = alloc_walk.state_digest(env.current_model)
        assert now != digest and now not in seen
        seen.add(now)
        again, _l, _r, _d = _walk(base.origin, flags, 500 + seed)               # the same scores under another seed
        assert alloc_walk.state_digest(again.current_model) == now


# ------------------------------------------------------------------ amendment A1: SPECTRA_ALLOC_TILT_PARAMS (the params hold)


def test_the_params_flag_parses_equal_and_a_tolerance_and_needs_the_flops_budget(monkeypatch):
    assert alloc_walk.tilt_params() is None
    monkeypatch.setenv("SPECTRA_ALLOC_TILT_PARAMS", "  ")
    assert alloc_walk.tilt_params() is None
    assert alloc_walk.tilt_params_mode() is None
    for raw, want, mode in (("equal", 0.01, "landed"), (" EQUAL ", 0.01, "landed"), ("equal:0.02", 0.02, "landed"),
                            ("equal: 0.005", 0.005, "landed"), ("equal:landed", 0.01, "landed"),
                            ("equal:planned", 0.01, "planned"), ("equal:0.02:planned", 0.02, "planned"),
                            ("Equal : 0.03 : Landed", 0.03, "landed")):
        monkeypatch.setenv("SPECTRA_ALLOC_TILT_PARAMS", raw)
        assert alloc_walk.tilt_params() == want and alloc_walk.tilt_params_mode() == mode   # the landing is the default
    for raw in ("1", "on", "flops", "equal:", "equal:0", "equal:1", "equal:-0.1", "equal:x", "equals", "equal:planned:0.02",
                "equal:0.02:0.03", "equal:0.02:walked", "equal:0.02:landed:x", "equal::landed", "landed"):
        monkeypatch.setenv("SPECTRA_ALLOC_TILT_PARAMS", raw)
        with pytest.raises(ValueError, match="expected equal"):
            alloc_walk.tilt_params()
        with pytest.raises(ValueError, match="expected equal"):
            alloc_walk.tilt_params_mode()
    assert alloc_walk.TILT_PARAMS_MODES == ("landed", "planned")
    assert alloc_walk.TILT_PARAMS_TOL == 0.01 and alloc_walk.TILT_LAMBDA_MAX == 2.0 and alloc_walk.TILT_LAMBDA_STEP == 0.05
    with pytest.raises(ValueError, match="needs SPECTRA_ALLOC_BUDGET=flops"):   # a tilt bisected in params holds them already
        alloc_walk.plan_targets(_model(8, 100), None, SHAPE, "tilt", 0.3, budget="params", min_width=2, tilt_hold=object())
    with pytest.raises(ValueError, match="needs SPECTRA_ALLOC_BUDGET=flops"):
        alloc_walk.tilt_hold_for(None, None, 0.2, False)


class _Cost:
    """A constructed cost model with ``plan_agent.ParamModel``'s interface: cost = fixed + Σ per[row] · width."""

    def __init__(self, per, widths0, fixed=0.0, pm=None):
        self.per, self.widths0, self.fixed, self.pm = dict(per), dict(widths0), float(fixed), pm
        self.rows = list(widths0)
        self.total0 = self.cost(widths0)

    def cost(self, widths):
        return self.fixed + sum(self.per[row] * int(widths.get(row, self.widths0[row])) for row in self.rows)

    def kept(self, widths):
        return self.cost(widths) / self.total0


def test_the_direction_is_the_log_flops_per_param_of_a_kept_channel_centred_by_the_flops():
    widths0 = {0: 64, 3: 64, 5: 128, 9: 1, 11: 32}
    flops = _Cost({0: 640.0, 3: 320.0, 5: 80.0, 9: 5.0, 11: 20.0}, widths0, fixed=1000.0)
    params = _Cost({0: 10.0, 3: 20.0, 5: 40.0, 9: 7.0, 11: 160.0}, widths0, fixed=30.0)
    here = {0: 40, 3: 30, 5: 50, 9: 1, 11: 1}                                   # one group at one channel, one that cannot move
    rows = list(widths0)
    d = alloc_walk.tilt_direction(flops, params, here, rows)
    raw = {0: math.log(64.0), 3: math.log(16.0), 5: math.log(2.0), 11: math.log(0.125)}   # row 11: measured from 1 up to 2
    share = {0: 640.0 * 40, 3: 320.0 * 30, 5: 80.0 * 50, 11: 20.0 * 1}          # each group's FLOPs at these widths
    mean = sum(share[row] * raw[row] for row in raw) / sum(share.values())
    assert d[9] == 0.0                                                          # an origin width of 1: no slope, no tilt
    assert all(d[row] == pytest.approx(raw[row] - mean, abs=1e-9) for row in raw)
    assert abs(sum(share[row] * d[row] for row in raw)) < 1e-6                  # lambda d moves no FLOPs to first order
    assert d[0] > d[3] > d[5] > d[11]                                           # FLOPs-heavy channels up, param-heavy ones down
    plain = sum(raw.values()) / len(raw)
    assert abs(mean - plain) > 0.1                                              # not the plain mean
    shifted = alloc_walk.tilt_direction(_Cost({r: 3.0 * v for r, v in flops.per.items()}, widths0, 5.0), params, here, rows)
    assert all(shifted[row] == pytest.approx(d[row], abs=1e-9) for row in rows)  # a unit of FLOPs changes nothing


def test_the_search_takes_zero_when_it_is_within_tol_and_brackets_on_both_sides():
    calls = []

    def ratio(scale, rate):
        def at(lam):
            calls.append(lam)
            return scale * math.exp(-rate * lam)
        return at

    assert alloc_walk.tilt_lambda(ratio(1.009, 1.7), 0.01) == (0.0, 1.009, True) and calls == [0.0]   # as it is: no search
    for scale, rate in ((1.4, 1.7), (0.7, 1.7), (1.4, -0.9), (2.5, 3.0), (0.3, 1.0)):
        del calls[:]
        lam, value, hit = alloc_walk.tilt_lambda(ratio(scale, rate), 0.01)
        assert hit and abs(value - 1.0) <= 0.01 and value == scale * math.exp(-rate * lam)
        assert lam == pytest.approx(math.log(scale) / rate, abs=0.02) and lam != 0.0
        assert len(calls) == len(set(calls)) and max(abs(c) for c in calls) <= alloc_walk.TILT_LAMBDA_MAX   # once per lambda
        assert calls[:3] == [0.0, 0.05, -0.05]                                  # outwards from 0, the positive side first
    lam, value, hit = alloc_walk.tilt_lambda(ratio(1.4, 1.7), 0.01, aim=0.9)   # another aim (the hold's corrected search)
    assert hit and abs(value - 0.9) <= 0.01
    # a ratio that moves in steps and jumps over the band: the closest of those tried, and the miss is told
    lam, value, hit = alloc_walk.tilt_lambda(lambda x: 1.05 if x < 0.123 else 0.97, 0.01)
    assert not hit and value == 0.97 and 0.123 <= lam < 0.1231                  # the smallest lambda found on the closer side
    lam, value, hit = alloc_walk.tilt_lambda(lambda x: 1.05 if x < 0.123 else 0.95, 0.01)
    assert (lam, value, hit) == (0.0, 1.05, False)                              # a tie goes to the smaller |lambda|
    lam, value, hit = alloc_walk.tilt_lambda(lambda x: 1.5 * math.exp(-0.1 * x), 0.01)
    assert not hit and lam == pytest.approx(alloc_walk.TILT_LAMBDA_MAX) and value == pytest.approx(1.5 * math.exp(-0.2))
    lam, value, hit = alloc_walk.tilt_lambda(lambda x: 1.2, 0.01, bound=0.2, step=0.1)
    assert (lam, value, hit) == (0.0, 1.2, False)                               # nothing moves: lambda stays 0
    # the landed hold's scan: fine near 0, where a narrow spot the even steps of 0.05 pass over is found
    scan = alloc_walk.TILT_LANDED_SCAN
    assert len(scan) == 42 and list(scan) == sorted(scan) and scan[0] == 0.0125 and scan[15] == pytest.approx(0.2)
    assert scan[31] == pytest.approx(1.0) and scan[-1] == pytest.approx(2.0) and alloc_walk.TILT_LANDED_ITERS == 6
    del calls[:]

    def spot(x):
        calls.append(x)
        return 1.0 if -0.03 < x < -0.02 else (1.2 if x > -0.5 else 0.7)

    assert alloc_walk.tilt_lambda(spot, 0.01)[::2] == (0.0, False)              # the even scan never lands in the spot
    del calls[:]
    assert alloc_walk.tilt_lambda(spot, 0.01, scan=scan, iters=6) == (-0.025, 1.0, True)
    assert calls == [0.0, 0.0125, -0.0125, 0.025, -0.025]                       # outwards from 0, the positive side first
    lam, value, hit = alloc_walk.tilt_lambda(lambda x: 1.2 if x > -0.61 else 0.7, 0.01, scan=scan, iters=6)
    assert not hit and value == 1.2 and lam == 0.0                              # a jump over the band: the closest, lambda 0


def _toy_scores(seed, n, sigma=0.3):
    """n scores from a small linear congruential draw (no torch, the same on every machine)."""
    state = [seed * 2654435761 % 2 ** 32 or 1]

    def gauss():
        total = 0.0
        for _ in range(12):
            state[0] = (1103515245 * state[0] + 12345) % 2 ** 31
            total += state[0] / 2.0 ** 31
        return total - 6.0
    return [sigma * gauss() for _ in range(n)]


class _ToyCut:
    def __init__(self, widths):
        self.widths = dict(widths)


def _toy_world(monkeypatch, tmp_path, masked=(), walked=False):
    """A constructed network for ``tilt_plan``: 12 groups shaped like a VGG's (FLOPs per channel fall with depth, params per
    channel rise), both costs linear in every width, and every function that would touch a real model replaced by the
    cost models. A group in ``masked`` is one no cut can narrow: the cut leaves its width, as a masked cut does, and the
    base plan has it whole. ``walked`` gives it a walk (``walk``: one row per group, every rate legal, 6 passes, group-once)
    and makes the grids the widths that walk lands on exactly (``alloc_walk.grid_widths``). Returns a namespace with
    ``run(z, hold) -> (widths, info)`` and the pieces."""
    from src import pruning
    n = 12
    rows = list(range(n))
    wide = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
    area = [1024, 1024, 256, 256, 64, 64, 64, 16, 16, 16, 4, 4]
    keep = [0.5, 0.45, 0.4, 0.4, 0.35, 0.3, 0.3, 0.25, 0.2, 0.2, 0.15, 0.15]
    widths0 = {r: wide[r] for r in rows}
    per_p = {r: 9.0 * 0.5 * ((wide[r - 1] * keep[r - 1] if r else 3) + (wide[r + 1] * keep[r + 1] if r + 1 < n else 100))
             for r in rows}
    per_f = {r: per_p[r] * area[r] for r in rows}
    pm = _Cost(per_p, widths0, 5000.0)
    fm = _Cost(per_f, widths0, 200000.0, pm=pm)

    def grid(w0, cuts=6):                                                       # the widths a walk of 6 cuts lands on
        seen, frontier = {w0}, {w0}
        for _ in range(cuts):
            nxt = {pruning.target_width(w, rate) for w in frontier for rate in (0.9, 0.8, 0.7, 0.6)}
            frontier = nxt - seen
            seen |= nxt
        return sorted(seen)

    grids = {r: grid(widths0[r]) for r in rows}
    walk = None
    if walked:
        def legal(_row, width):
            return list(RATES) if width > 1 else [0]

        walk = alloc_walk.TiltWalk([(r, r) for r in rows], RATES, 6, legal, once=True)
        grids = {r: sorted(alloc_walk.grid_widths(widths0[r], RATES, lambda width: legal(0, width), 6)) for r in rows}
    base = {r: widths0[r] if r in masked else min(grids[r], key=lambda w, r=r: abs(w - keep[r] * widths0[r])) for r in rows}
    model = object()

    def of(obj):
        return obj.widths if isinstance(obj, _ToyCut) else widths0

    def cut_to(_model, _plan, keeps, _shape):
        return _ToyCut({r: widths0[r] if r in masked or float(keeps.get(r, 1.0)) >= 1.0 - 1e-9
                        else pruning.target_width(widths0[r], float(keeps[r])) for r in rows})

    monkeypatch.setattr(alloc_walk, "cut_to", cut_to)
    monkeypatch.setattr(alloc_walk, "group_widths", lambda obj, which: {r: of(obj)[r] for r in which})
    monkeypatch.setattr(alloc_walk, "copied_keeps",
                        lambda _m, _p, table: ({r: min(1.0, table[f"g{r}"] / float(widths0[r])) for r in rows}, 0))
    monkeypatch.setattr(alloc_walk, "group_keys", lambda _m, _p: {r: f"g{r}" for r in rows})
    monkeypatch.setattr(utils, "calc_num_parameters", lambda obj: pm.cost(of(obj)))
    monkeypatch.setattr(utils, "calc_flops", lambda obj, *a, **k: fm.cost(of(obj)))
    monkeypatch.setenv("SPECTRA_ALLOC_TILT_BASE", _write(tmp_path / "toy_base.json", {f"g{r}": base[r] for r in rows}))
    plan = [(types.SimpleNamespace(width=widths0[r]), r) for r in rows]

    def run(z, hold):
        monkeypatch.setenv("SPECTRA_ALLOC_TILT_Z", _write(tmp_path / "toy_z.json", list(z)))
        return alloc_walk.tilt_plan(model, plan, rows, SHAPE, 0.2, 0.5, 0.1, 16, 0.003, "flops", 2,
                                    lambda m: utils.calc_flops(m, SHAPE), hold=hold)

    target = fm.kept(base) + 0.004                                              # the base sits inside the grid round's band
    floors = {r: max(2, int(round(0.1 * widths0[r]))) for r in rows}
    return types.SimpleNamespace(rows=rows, widths0=widths0, base=base, pm=pm, fm=fm, grids=grids, target=target, floors=floors,
                                 run=run, per_p=per_p, per_f=per_f, walk=walk)


def test_the_params_hold_reaches_the_tolerance_on_a_constructed_cost_model(monkeypatch, tmp_path):
    toy = _toy_world(monkeypatch, tmp_path)
    rows, base = toy.rows, toy.base
    base_params = toy.pm.kept(base)
    plain = alloc_walk.TiltHold(0.01, toy.fm)
    on_grid = alloc_walk.TiltHold(0.01, toy.fm, toy.grids, toy.target, toy.floors)
    assert not plain.grid and on_grid.grid and plain.aimed(base, {r: 1.0 for r in rows}) == base
    # z = 0: lambda is 0 exactly, nothing is searched, and the plan is the base (on the grid, its own rounding)
    for hold in (plain, on_grid):
        widths, info = toy.run([0.0] * len(rows), hold)
        held = info["tilt"]["hold"]
        assert widths == base and held["aimed"] == base and held["lambda"] == 0.0 and held["ratio"] == 1.0 and held["within"]
        assert held["real_plans"] == 1 and held["model_plans"] == 0 and held["d"] is None and held["ratio0"] == 1.0
        assert info["weights"] == info["tilt"]["base_keeps"] and info["tilt"]["c"] == 1.0
    off_w, off_info = toy.run([0.0] * len(rows), None)
    assert off_w == base and "hold" not in off_info["tilt"]
    direction = alloc_walk.tilt_direction(toy.fm, toy.pm, base, rows)
    searched = 0
    for seed in range(1, 41):
        z = _toy_scores(seed, len(rows))
        unheld, unheld_info = toy.run(z, None)
        for hold in (plain, on_grid):
            widths, info = toy.run(z, hold)
            held = info["tilt"]["hold"]
            ratio0 = toy.pm.kept(hold.aimed(unheld, unheld_info["weights"])) / base_params
            assert held["ratio0"] == pytest.approx(ratio0, abs=1e-12)           # what the tilt keeps as it is
            assert held["ratio"] == pytest.approx(toy.pm.kept(held["aimed"]) / base_params, abs=1e-12)
            assert held["aimed"] == hold.aimed(widths, info["weights"])         # the plan the walk will aim at
            assert held["within"] and abs(held["ratio"] - 1.0) <= 0.01          # the base plan's params within 1 %
            assert abs(info["kept"] - info["tilt"]["base_kept"]) <= 0.003 + 1e-12   # and still the base plan's FLOPs
            assert abs(held["flops"] - toy.fm.kept(held["aimed"])) < 1e-12
            assert held["flops"] <= toy.target + 1e-9 or not hold.grid          # on the grid: at or under the walk target
            assert abs(held["lambda"]) <= alloc_walk.TILT_LAMBDA_MAX and held["bound"] == 2.0 and held["step"] == 0.05
            if abs(ratio0 - 1.0) <= 0.01:                                       # already within: the tilt as it is
                assert held["lambda"] == 0.0 and widths == unheld and info["weights"] == unheld_info["weights"]
                assert held["real_plans"] == 1 and held["d"] is None
            else:
                searched += 1
                assert held["lambda"] != 0.0 and held["d"] == direction and 2 <= held["real_plans"] <= 4
                assert held["model_plans"] >= 2 and (held["lambda"] > 0) == (ratio0 > 1.0)   # more params: towards FLOPs
                assert info["weights"] == {r: info["tilt"]["base_keeps"][r] * math.exp(z[r] + held["lambda"] * direction[r])
                                           for r in rows}
            assert info["sens"] == dict(zip(rows, z))                           # the scores stay the scores
            again, info2 = toy.run(z, hold)
            assert again == widths and info2["tilt"]["hold"]["lambda"] == held["lambda"]   # one z, one plan
    assert searched >= 50                                                       # most draws needed the search


def test_the_params_hold_takes_the_closest_plan_and_says_so_when_the_grid_is_too_coarse(monkeypatch, tmp_path):
    toy = _toy_world(monkeypatch, tmp_path)
    rows, base = toy.rows, toy.base
    base_params = toy.pm.kept(base)
    coarse = {r: [w for i, w in enumerate(toy.grids[r]) if i % 6 == 0 or w == base[r]] for r in rows}
    hold = alloc_walk.TiltHold(0.01, toy.fm, coarse, toy.target, toy.floors)
    missed, hits = 0, 0
    for seed in range(1, 41):
        z = _toy_scores(seed, len(rows))
        unheld, unheld_info = toy.run(z, None)
        ratio0 = toy.pm.kept(hold.aimed(unheld, unheld_info["weights"])) / base_params
        widths, info = toy.run(z, hold)
        held = info["tilt"]["hold"]
        assert held["within"] == (abs(held["ratio"] - 1.0) <= 0.01)
        assert abs(held["ratio"] - 1.0) <= abs(ratio0 - 1.0) + 1e-12            # never further than the tilt as it is
        assert all(held["aimed"][r] in coarse[r] for r in rows)
        hits += held["within"]
        missed += not held["within"]
    assert missed >= 1 and hits >= 30                                           # a coarse grid: some plans cannot reach 1 %
    # a hold that can move nothing: every lambda rounds to one plan, so lambda stays 0 and the miss is told
    stuck = alloc_walk.TiltHold(0.01, toy.fm, {r: [int(round(1.25 * base[r]))] for r in rows}, 1.0, toy.floors)
    widths, info = toy.run(_toy_scores(3, len(rows)), stuck)
    held = info["tilt"]["hold"]
    assert held["lambda"] == 0.0 and not held["within"] and held["real_plans"] == 1 and held["model_plans"] > 80
    assert held["d"] is not None and abs(held["ratio"] - 1.0) > 0.01


def test_the_params_hold_corrects_a_cost_model_that_is_off_by_cutting_the_plan_for_real(monkeypatch, tmp_path):
    toy = _toy_world(monkeypatch, tmp_path)
    rows = toy.rows
    base_params = toy.pm.kept(toy.base)
    off_pm = _Cost({r: toy.per_p[r] * (1.08 if r in (9, 10) else 1.0) for r in rows}, toy.widths0, 5000.0)
    off_fm = _Cost(toy.per_f, toy.widths0, 200000.0, pm=off_pm)                 # the search's params model overstates two groups
    hold = alloc_walk.TiltHold(0.01, off_fm, toy.grids, toy.target, toy.floors)
    rounds = set()
    for seed in range(1, 21):
        widths, info = toy.run(_toy_scores(seed, len(rows)), hold)
        held = info["tilt"]["hold"]
        real = toy.pm.kept(held["aimed"]) / base_params                         # the real cut's params, not the model's
        assert held["ratio"] == pytest.approx(real, abs=1e-12) and held["within"] and abs(real - 1.0) <= 0.01
        rounds.add(held["real_plans"])
    assert rounds <= {1, 2, 3, 4} and max(rounds) >= 3                          # some plans needed a second real cut


def test_the_params_hold_keeps_a_group_no_cut_can_narrow_at_its_width_on_the_cost_models(monkeypatch, tmp_path):
    masked = (1, 6)                                                             # as ShuffleNetV2's streams: the cut is masked
    toy = _toy_world(monkeypatch, tmp_path, masked=masked)
    rows, base, origin = toy.rows, toy.base, toy.widths0
    base_params = toy.pm.kept(base)
    assert all(base[r] == origin[r] for r in masked)
    hold = alloc_walk.TiltHold(0.01, toy.fm, toy.grids, toy.target, toy.floors)
    searched = hits = 0
    for seed in range(1, 31):
        z = _toy_scores(seed, len(rows))
        unheld, unheld_info = toy.run(z, None)
        assert all(unheld[r] == origin[r] for r in masked)                      # whatever its keep, the cut leaves it whole
        widths, info = toy.run(z, hold)
        held = info["tilt"]["hold"]
        assert all(widths[r] == origin[r] and held["aimed"][r] == origin[r] for r in masked)
        assert held["ratio"] == pytest.approx(toy.pm.kept(held["aimed"]) / base_params, abs=1e-12)
        if not held["model_plans"]:                                             # within as it is: nothing was probed
            assert held["lambda"] == 0.0 and held["pinned"] == 0 and held["within"]
            continue
        searched += 1
        assert held["pinned"] == len(masked) and held["unmodelled"] == 0        # the half cut found them
        assert held["mirror"] == 0                                              # and the cost models land where the cut does
        assert all(held["d"][r] == 0.0 for r in masked) and any(held["d"][r] != 0.0 for r in rows)
        assert abs(held["ratio"] - 1.0) <= abs(held["ratio0"] - 1.0) + 1e-12
        hits += held["within"]
    assert searched >= 15 and hits >= searched - 3                              # the search works around the pinned groups


def test_the_walk_on_the_cost_models_cuts_towards_its_targets_and_softens_the_cut_that_crosses():
    origin = {0: 100, 1: 100, 2: 100}
    per = {0: 3.0, 1: 2.0, 2: 1.0}

    def kept(widths):
        return sum(per[g] * widths[g] for g in origin) / 600.0

    def legal(_row, width):
        return list(RATES) if width > 1 else [0]

    targets = {0: 50, 1: 60, 2: 40}
    walk = alloc_walk.TiltWalk([(0, 0), (1, 1), (2, 2)], RATES, 6, legal)
    # pass 1: 100 -> 60 in each group (the cut closest to its target), kept 0.80, 0.667, 0.60; pass 2: group 0 would go
    # 60 -> 48 and cross 0.552 (kept 0.54), so that cut is softened to the mildest that still reaches it: width 50
    assert walk.land(kept, origin, targets, 0.552) == ({0: 50, 1: 60, 2: 60}, "landed")
    assert kept({0: 50, 1: 60, 2: 60}) <= 0.552 < kept({0: 51, 1: 60, 2: 60})   # one channel more would miss the target
    assert walk.land(kept, origin, targets, 0.80) == ({0: 60, 1: 100, 2: 100}, "landed")   # the first cut lands already
    # a goal the first cut overshoots: softened from 60 up to the widest width at or under the goal
    widths, how = walk.land(kept, origin, targets, 0.95)
    assert how == "landed" and widths[1] == widths[2] == 100 and kept(widths) <= 0.95 < kept({**widths, 0: widths[0] + 1})
    # a group whose cut is masked is never cut, and the others carry on
    assert walk.land(kept, origin, targets, 0.85, pinned={0}) == ({0: 100, 1: 60, 2: 90}, "landed")
    assert walk.land(kept, origin, targets, 0.552, pinned={0}) == ({0: 100, 1: 60, 2: 42}, "stalled")   # and cannot reach
    assert alloc_walk.TiltWalk([(0, 0), (1, 1), (2, 2)], RATES, 6, legal, protected={2}).land(
        kept, origin, {0: 100, 1: 100, 2: 40}, 0.5) == (origin, "stalled")       # a protected stream is held whole
    # every group at its target above the goal: the walk stalls (the real one then falls back), nothing is cut
    assert walk.land(kept, origin, dict(origin), 0.5) == (origin, "stalled")
    assert walk.land(kept, origin, {0: 36, 1: 36, 2: 36}, 0.01)[1] == "stalled"   # arrived everywhere, still above
    one_pass = alloc_walk.TiltWalk([(0, 0), (1, 1), (2, 2)], RATES, 1, legal)
    assert one_pass.land(kept, origin, {0: 10, 1: 10, 2: 10}, 0.2) == ({0: 60, 1: 60, 2: 60}, "passes")
    # without the group-once lock every row of a group cuts it again within the pass
    twice = alloc_walk.TiltWalk([(0, 0), (1, 0), (2, 2)], RATES, 1, legal, once=False)
    assert twice.land(kept, origin, {0: 36, 2: 100}, 0.3) == ({0: 36, 1: 100, 2: 100}, "passes")
    once = alloc_walk.TiltWalk([(0, 0), (1, 0), (2, 2)], RATES, 1, legal, once=True)
    assert once.land(kept, origin, {0: 36, 2: 100}, 0.3) == ({0: 60, 1: 100, 2: 100}, "passes")
    # the legal mask decides: a row where only the identity is legal (a stem row) leaves its group alone
    stem = alloc_walk.TiltWalk([(0, 0), (1, 1), (2, 2)], RATES, 6, lambda row, width: [0] if row == 0 else list(RATES))
    assert stem.land(kept, origin, targets, 0.82) == ({0: 100, 1: 60, 2: 72}, "landed")


def test_the_landed_hold_holds_the_params_of_the_predicted_landing_on_a_constructed_cost_model(monkeypatch, tmp_path):
    toy = _toy_world(monkeypatch, tmp_path, walked=True)
    rows, base, origin = toy.rows, toy.base, toy.widths0
    assert all(base[r] in toy.grids[r] for r in rows)                           # a plan the walk can reach, group by group
    landed = alloc_walk.TiltHold(0.01, toy.fm, toy.grids, toy.target, toy.floors, walk=toy.walk)
    planned = alloc_walk.TiltHold(0.01, toy.fm, toy.grids, toy.target, toy.floors)
    assert (landed.mode, planned.mode) == ("landed", "planned")
    base_keeps = {r: base[r] / float(origin[r]) for r in rows}
    base_stop, how = landed.landed(landed.aimed(base, base_keeps))
    assert how == "landed" and toy.fm.kept(base_stop) <= toy.target + 1e-9
    assert any(base_stop[r] > base[r] for r in rows)                            # the walk stops before every group arrives
    base_landed = toy.pm.kept(base_stop)
    # z = 0: lambda 0, the base plan, and the base plan's own landing
    widths, info = toy.run([0.0] * len(rows), landed)
    held = info["tilt"]["hold"]
    assert widths == base and held["lambda"] == 0.0 and held["ratio"] == 1.0 and held["within"] and held["mode"] == "landed"
    assert held["landed"] == base_stop and held["how"] == held["base_how"] == "landed" and held["model_plans"] == 0
    assert held["landed_params"] == pytest.approx(base_landed, abs=1e-12) == pytest.approx(held["base_landed"], abs=1e-12)
    hits = hits_planned = searched = 0
    off = off_planned = off_unheld = 0.0
    for seed in range(1, 31):
        z = _toy_scores(seed, len(rows))
        widths, info = toy.run(z, landed)
        held = info["tilt"]["hold"]
        stop, how = landed.landed(held["aimed"])                                # the walk of the plan the hold returned
        assert held["aimed"] == landed.aimed(widths, info["weights"]) and held["landed"] == stop and how == "landed"
        assert held["ratio"] == pytest.approx(toy.pm.kept(stop) / base_landed, abs=1e-12)
        assert held["planned"] == pytest.approx(toy.pm.kept(held["aimed"]) / toy.pm.kept(base), abs=1e-12)
        assert held["landed_flops"] == pytest.approx(toy.fm.kept(stop), abs=1e-12) and held["landed_flops"] <= toy.target + 1e-9
        assert held["within"] == (abs(held["ratio"] - 1.0) <= 0.01)
        assert abs(held["ratio"] - 1.0) <= abs(held["ratio0"] - 1.0) + 1e-12
        assert abs(info["kept"] - info["tilt"]["base_kept"]) <= 0.003 + 1e-12   # the plan is still landed on the base's FLOPs
        again, info2 = toy.run(z, landed)
        assert again == widths and info2["tilt"]["hold"]["lambda"] == held["lambda"]
        hits += held["within"]
        searched += held["model_plans"] > 0
        off += abs(held["ratio"] - 1.0)
        off_unheld += abs(held["ratio0"] - 1.0)
        # the planned hold of the same scores: its plan is close to the base's, its predicted landing less so
        _w, info_p = toy.run(z, planned)
        held_p = info_p["tilt"]["hold"]
        assert held_p["mode"] == "planned" and "landed" not in held_p
        there = toy.pm.kept(landed.landed(held_p["aimed"])[0]) / base_landed
        hits_planned += abs(there - 1.0) <= 0.01
        off_planned += abs(there - 1.0)
    # the walk's landing moves in coarse steps (which groups arrive), so not every draw reaches 1 %: the closest is
    # taken. Holding the landing still holds it far better than holding the plan does, and than no hold at all
    assert searched >= 20 and hits >= 8 and hits > hits_planned
    assert off < 0.6 * off_planned and off < 0.3 * off_unheld


def _tilt_plan_before_a1(model, plan, rows, input_shape, target, a, keep_floor, iters, tol, budget, min_width, size):
    """``alloc_walk.tilt_plan`` as it was before amendment A1, verbatim (the flag-off reference)."""
    path = alloc_walk.tilt_base_path()
    base_keeps, unnamed = alloc_walk.copied_keeps(model, plan, alloc_walk.tilt_base_table())
    if rows and unnamed == len(rows):
        raise ValueError(f"SPECTRA_ALLOC_TILT_BASE={path}: the table names no planned group of this network")
    keys = alloc_walk.group_keys(model, plan)
    z, source = alloc_walk.tilt_scores(rows, keys)
    w = {row: base_keeps[row] * math.exp(max(-alloc_walk.TILT_Z_CLIP, min(alloc_walk.TILT_Z_CLIP, z[row]))) for row in rows}
    floor = {row: keep_floor for row in rows}
    if int(min_width) > 1:
        for (group, row) in plan:
            floor[row] = max(keep_floor, min(1.0, int(min_width) / float(group.width)))
    floor = {row: min(floor[row], base_keeps[row]) for row in rows}
    size0 = size(model)
    base_cut = alloc_walk.cut_to(model, plan, base_keeps, input_shape)
    base_frac = size(base_cut) / size0
    base_widths = alloc_walk.group_widths(base_cut, rows)
    del base_cut
    mode = alloc_walk.tilt_target()
    aim = float(base_frac) if mode == "base" else float(target)

    def at(c):
        keeps = {row: min(1.0, max(floor[row], c * w[row])) for row in rows}
        cut = alloc_walk.cut_to(model, plan, keeps, input_shape)
        frac = size(cut) / size0
        widths = alloc_walk.group_widths(cut, rows)
        del cut
        return keeps, frac, widths, c

    lo, hi = 0.0, 1.0 / min(w.values())
    best = None
    if mode == "base":
        best = at(1.0)
        if best[1] > aim:
            hi = 1.0
        else:
            lo, hi = 1.0, max(hi, 1.0)
    if best is None or abs(best[1] - aim) > tol:
        for _ in range(iters):
            c = 0.5 * (lo + hi)
            trial = at(c)
            if best is None or abs(trial[1] - aim) < abs(best[1] - aim):
                best = trial
            if abs(trial[1] - aim) <= tol:
                break
            if trial[1] > aim:
                hi = c
            else:
                lo = c
    keeps, frac, widths, c = best
    cut = alloc_walk.cut_to(model, plan, keeps, input_shape)
    params = utils.calc_num_parameters(cut) / float(utils.calc_num_parameters(model))
    flops = utils.calc_flops(cut, input_shape) / float(utils.calc_flops(model, input_shape))
    del cut
    moved = sum(widths.get(row) != base_widths.get(row) for row in rows)
    return widths, {"kind": "tilt", "alpha": float(a), "target": float(aim), "kept": float(frac), "keeps": keeps,
                    "origin_widths": alloc_walk.group_widths(model, rows), "sens": dict(z), "held": 0, "weights": w,
                    "cost_budget": budget,
                    "tilt": {"base": path, "base_keeps": base_keeps, "base_widths": base_widths,
                             "base_kept": float(base_frac), "unnamed": int(unnamed), "rows": list(rows), "keys": keys,
                             "z": dict(z), "source": source, "target": mode, "aim": float(aim), "c": float(c),
                             "floor": floor, "widths": dict(widths), "params": float(params), "flops": float(flops),
                             "moved": int(moved)}}


@pytest.mark.parametrize("budget", ["flops", "params"])
def test_with_the_params_flag_off_the_tilt_is_the_tilt_before_the_amendment(monkeypatch, tmp_path, budget):
    model = _model(12, 100)
    base, _info = _base_plan(monkeypatch, model, "sens", 0.2, budget, 2)
    monkeypatch.setenv("SPECTRA_ALLOC_TILT_BASE", _write(tmp_path / "base.json", _table(model, base)))
    plan = _plan(model)
    rows = [row for _g, row in plan]
    size = (lambda m: utils.calc_flops(m, SHAPE)) if budget == "flops" else utils.calc_num_parameters
    cases = ({"SPECTRA_ALLOC_TILT_SIGMA": "0"}, {"SPECTRA_ALLOC_TILT_SIGMA": "0.3", "SPECTRA_ALLOC_SAMPLE_SEED": "2"},
             {"SPECTRA_ALLOC_TILT_SIGMA": "0.3", "SPECTRA_ALLOC_SAMPLE_SEED": "5", "SPECTRA_ALLOC_TILT_TARGET": "run"},
             {"SPECTRA_ALLOC_TILT_Z": _write(tmp_path / "z.json", [0.4 if k % 3 else -0.6 for k in range(len(rows))])})
    for index, flags in enumerate(cases):
        for key in ("SPECTRA_ALLOC_TILT_Z", "SPECTRA_ALLOC_TILT_SIGMA", "SPECTRA_ALLOC_TILT_TARGET", "SPECTRA_ALLOC_SAMPLE_SEED",
                    "SPECTRA_ALLOC_TILT_PARAMS"):
            monkeypatch.delenv(key, raising=False)
        for key, value in flags.items():
            monkeypatch.setenv(key, value)
        torch.manual_seed(11)
        before = _tilt_plan_before_a1(model, plan, rows, SHAPE, 0.2, 0.5, 0.1, 16, 0.003, budget, 2, size)
        rng_before = torch.get_rng_state()
        torch.manual_seed(11)
        now = alloc_walk.plan_targets(model, None, SHAPE, "tilt", 0.2, budget=budget, min_width=2)
        assert torch.equal(torch.get_rng_state(), rng_before)                   # the same cuts in the same order
        assert now == before and "hold" not in now[1]["tilt"]                   # every width, keep and number
        if index < 2:  # the flag alone moves nothing in plan_targets: the hold is the walk's (_state hands it the walk's grid)
            monkeypatch.setenv("SPECTRA_ALLOC_TILT_PARAMS", "equal")
            torch.manual_seed(11)
            assert alloc_walk.plan_targets(model, None, SHAPE, "tilt", 0.2, budget=budget, min_width=2) == before


def test_the_landed_hold_predicts_where_the_walk_lands_and_z0_is_still_the_base_bit_for_bit(tmp_path_factory):
    base = _base_walk(tmp_path_factory)
    origin, plan = base.origin, _plan(base.origin)
    rows = [row for _g, row in plan]
    w0 = alloc_walk.group_widths(origin, rows)
    total = float(utils.calc_num_parameters(origin))

    def params_of(widths):                                                      # a widths table cut in one shot: its params
        cut = alloc_walk.cut_to(origin, plan, plan_agent.rates_of({row: widths.get(row, w0[row]) for row in rows}, w0), SHAPE)
        return utils.calc_num_parameters(cut) / total

    def held_of(records):
        return next(k for event, k in records if event == "alloc_tilt_hold")

    targets = base.env._alloc_walk[base.env.selected_net_path]["widths"]
    base_params = params_of(targets)
    base_stop = alloc_walk.group_widths(base.env.current_model, rows)           # where the base walk really stopped
    hold_on = {"SPECTRA_ALLOC_KIND": "tilt", "SPECTRA_ALLOC_TILT_BASE": base.plan, "SPECTRA_ALLOC_TILT_PARAMS": "equal"}
    # z = 0 under the flag (the landed hold is its default): lambda 0, the base's widths, the base's network
    env, lines, records, done = _walk(origin, {**hold_on, "SPECTRA_ALLOC_TILT_SIGMA": "0"}, 31)
    entry = env._alloc_walk[env.selected_net_path]
    assert done and not entry["fallback"] and entry["widths"] == targets
    assert alloc_walk.state_digest(env.current_model) == alloc_walk.state_digest(base.env.current_model)
    assert env.param_ratio() == base.env.param_ratio() and env.flops_ratio() == base.env.flops_ratio()
    held = held_of(records)
    assert held["mode"] == "landed" and held["lam"] == 0.0 and held["ratio"] == 1.0 and held["within"] and held["match"]
    assert held["grid"] and held["real_plans"] == 1 and held["model_plans"] == 0 and held["tol"] == 0.01
    assert held["planned"] == 1.0 and held["base_params"] == pytest.approx(base_params, abs=1e-12)
    assert held["how"] == held["base_how"] == "landed" and held["pinned"] == 0 and held["unmodelled"] == 0
    # the walk on the cost models stops where the real base walk stopped: every group's width, the params, the FLOPs
    assert [g["landed"] for g in held["groups"]] == [base_stop[row] for row in rows]
    assert held["landed_params"] == pytest.approx(base.env.param_ratio(), abs=1e-12)
    assert held["base_landed"] == pytest.approx(base.env.param_ratio(), abs=1e-12)
    assert held["landed_flops"] == pytest.approx(base.env.flops_ratio(), abs=1e-9)
    assert [event for event, _k in records if event.startswith("alloc_")] == ["alloc_plan", "alloc_tilt", "alloc_grid_round",
                                                                              "alloc_tilt_hold"]
    line = next(line for line in lines if " params hold: " in line)
    assert line.startswith("[alloc] tilt tiny") and "params hold: lambda=+0.000000 (the tilt as it is, within tol; no search)" in line
    assert "; the walk is predicted to land on params x" in line and "1.0000 of the base plan's predicted x" in line
    assert "(within tol 0.01)" in line and "WARNING" not in line
    assert not [g for g in held["groups"] if g["aimed"] != g["walk"] or "d" in g]
    # drawn tilts under the flag: the hold's predicted landing is the real walk's, and it is what is held
    searched = 0
    for seed in (1, 2, 3):
        flags = {**hold_on, "SPECTRA_ALLOC_TILT_SIGMA": "0.3", "SPECTRA_ALLOC_SAMPLE_SEED": str(seed)}
        env, lines, records, done = _walk(origin, flags, 200 + seed)
        entry = env._alloc_walk[env.selected_net_path]
        held = held_of(records)
        assert done and not entry["fallback"] and not [line for line in lines if "strongest legal cut" in line]
        assert held["mode"] == "landed" and held["match"] and held["grid"] and held["how"] == "landed"
        assert [g["aimed"] for g in held["groups"]] == [entry["widths"][row] for row in rows]
        stop = alloc_walk.group_widths(env.current_model, rows)                 # where this walk really stopped
        assert [g["landed"] for g in held["groups"]] == [stop[row] for row in rows]
        assert held["landed_params"] == pytest.approx(env.param_ratio(), abs=1e-12)
        assert held["landed_flops"] == pytest.approx(env.flops_ratio(), abs=1e-9)
        ratio = env.param_ratio() / base.env.param_ratio()                      # the real landed params over the base walk's
        assert held["ratio"] == pytest.approx(ratio, abs=1e-12) and held["within"] == (abs(ratio - 1.0) <= 0.01)
        assert held["planned"] == pytest.approx(params_of(entry["widths"]) / base_params, abs=1e-12)
        line = next(line for line in lines if " params hold: lambda=" in line)
        assert ("(within tol 0.01)" in line) == held["within"] and ("(MISS: " in line) == (not held["within"])
        assert "WARNING" not in line and f"lambda={held['lam']:+.6f}" in line and "is predicted to land on params x" in line
        assert 0.2 - 0.0051 <= env.flops_ratio() <= 0.2 + 1e-9                  # the gate's landing band, as every tilt
        unheld = {k: v for k, v in flags.items() if k != "SPECTRA_ALLOC_TILT_PARAMS"}   # the same scores without the hold
        off_env, _lines, off_records, off_done = _walk(origin, unheld, 200 + seed)
        assert off_done and not [event for event, _k in off_records if event == "alloc_tilt_hold"]
        ratio0 = off_env.param_ratio() / base.env.param_ratio()
        assert held["ratio0"] == pytest.approx(ratio0, abs=1e-12)               # the hold's start is the unheld walk's landing
        assert abs(ratio - 1.0) <= abs(ratio0 - 1.0) + 1e-12
        tilt = next(k for event, k in records if event == "alloc_tilt")
        if held["lam"] == 0.0:                                                  # the unheld plan itself (within as it is,
            assert entry["widths"] == off_env._alloc_walk[off_env.selected_net_path]["widths"] and tilt["lam"] == 0.0
            assert held["within"] == (abs(ratio0 - 1.0) <= 0.01) and (held["within"] <= ("d" not in tilt["groups"][0]))
            assert alloc_walk.state_digest(env.current_model) == alloc_walk.state_digest(off_env.current_model)
        else:                                                                   # or no lambda came closer: a told miss)
            searched += 1
            assert abs(ratio0 - 1.0) > 0.01 and tilt["lam"] == held["lam"] and held["real_plans"] >= 2
            assert held["pinned"] == 0 and held["unmodelled"] == 0              # every group of a ResNet is cut for real
            assert [g["d"] for g in tilt["groups"]] == [g["d"] for g in held["groups"]]
            assert [line for line in lines if line.startswith("[alloc] tilt tiny") and " group 0 row " in line][0].count(" d ") == 1
    assert searched >= 1                                                        # a drawn tilt moved the landing, and was held
    # the planned hold (amendment A1) is still there: the aimed plan's own params against the base table's
    flags = {**hold_on, "SPECTRA_ALLOC_TILT_PARAMS": "equal:planned", "SPECTRA_ALLOC_TILT_SIGMA": "0.3", "SPECTRA_ALLOC_SAMPLE_SEED": "1"}
    entry, lines, records = _plan_only(origin, flags)
    held = held_of(records)
    assert held["mode"] == "planned" and held["match"] and "landed_params" not in held and "landed" not in held["groups"][0]
    assert held["ratio"] == pytest.approx(params_of(entry["widths"]) / base_params, abs=1e-12) and held["planned"] == held["ratio"]
    line = next(line for line in lines if " params hold: lambda=" in line)
    assert "; the plan the walk aims at keeps params x" in line and "predicted" not in line


def test_the_gate_reads_a_tilts_hold_as_within_the_tolerance_or_a_logged_miss():
    step = _step_module()
    ok_rec = {"mode": "landed", "lam": 0.0625, "ratio": 1.004, "tol": 0.01, "within": True, "match": True, "ratio0": 1.31,
              "planned": 0.97, "landed_params": 0.1811, "model_plans": 9, "real_plans": 2}
    line = "x | [alloc] tilt net.pt params hold: lambda=+0.062500 (...); the walk is predicted to land on params x0.18, 1.0040 of"
    ok, detail, note = step.hold_check("rand", ok_rec, line + " the base plan's predicted x0.18 (within tol 0.01)")
    assert ok and note is None and "lambda +0.062500" in detail and "1.0040" in detail and "0.9700" in detail
    miss = dict(ok_rec, ratio=1.018, within=False)
    ok, detail, note = step.hold_check("pg", miss, line + " (MISS: 0.0080 beyond tol 0.01, the closest lambda found)")
    assert ok and note is not None and "outside tol" in note                   # a logged miss passes, with a note
    assert not step.hold_check("pg", miss, line)[0]                             # a miss the log does not tell
    assert not step.hold_check("rand", None, line)[0]                           # the flag did not act
    assert not step.hold_check("rand", dict(ok_rec, match=False), line)[0]      # the walk aimed at another plan
    assert not step.hold_check("rand", dict(ok_rec, within=False), line)[0]     # a record that contradicts itself
    assert not step.hold_check("rand", dict(ok_rec, tol=0.05), line)[0]         # another tolerance than the registered 1 %
    assert not step.hold_check("rand", dict(ok_rec, mode="planned"), line)[0]   # not the registered hold (amendment A2)
    assert not step.hold_check("rand", ok_rec, "no hold line in the log")[0]
    assert not step.hold_check("rand", {"lam": "x"}, line)[0]
    zero = dict(ok_rec, lam=0.0, ratio=1.0, ratio0=1.0, model_plans=0, real_plans=1)
    assert step.hold_check("tilt0", zero, line)[0] and not step.hold_check("tilt0", ok_rec, line)[0]   # z = 0: lambda 0
    assert step.HOLD_TOL == 0.01 and step.TILT_ENV == {"SPECTRA_ALLOC_KIND": "tilt", "SPECTRA_ALLOC_TILT_PARAMS": "equal"}
    # the run's real landing against the hold's prediction: the amendment stands on the two being the same
    assert step.landing_check(ok_rec, 0.1811) == (True, "the hold predicted landed params x0.1811, the run landed x0.1811")
    assert step.landing_check(ok_rec, 0.1815)[0] and not step.landing_check(ok_rec, 0.1830)[0]
    assert not step.landing_check({"mode": "landed"}, 0.18)[0] and not step.landing_check(None, 0.18)[0]
    assert step.PRED_TOL == 0.005


# ------------------------------------------------------------------ SPECTRA_EVAL_SAVE_WIDTHS


def test_save_widths_writes_the_plan_the_cut_and_the_meta_and_touches_nothing(tmp_path_factory, tmp_path, monkeypatch):
    assert fortify.eval_save_widths() is False
    monkeypatch.setenv("SPECTRA_EVAL_SAVE_WIDTHS", "1")
    assert fortify.eval_save_widths() is True
    base = _base_walk(tmp_path_factory)
    env = base.env
    name = os.path.basename(env.selected_net_path)
    assert sorted(os.listdir(base.folder))[:3] == [f"{name}__{LABEL}__cut.json", f"{name}__{LABEL}__meta.json",
                                                   f"{name}__{LABEL}__plan.json"]
    plan, cut, meta = json.load(open(base.plan)), json.load(open(base.cut)), json.load(open(base.meta_path))
    targets = env._alloc_walk[env.selected_net_path]["widths"]
    origin_plan = _plan(base.origin)
    assert plan == alloc_walk.plan_table(base.origin, origin_plan, targets)     # the widths the walk aimed at
    assert cut == {n: (m.out_channels if isinstance(m, torch.nn.Conv2d) else m.out_features)
                   for n, m in env.current_model.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))}
    assert all(isinstance(v, int) for v in list(plan.values()) + list(cut.values()))
    assert meta == base.meta and meta["plan_sha1"] == alloc_walk.table_digest(plan) and meta["cut_sha1"] == alloc_walk.table_digest(cut)
    assert meta["label"] == LABEL and meta["policy"] == "alloc" and meta["kind"] == "sens_thresh" and meta["budget"] == "flops"
    assert meta["walk_target"] == 0.2 and meta["fallback"] is False and meta["seed"] == 1
    assert meta["flop"] == pytest.approx(env.flops_ratio()) and meta["param"] == pytest.approx(env.param_ratio())
    keys = alloc_walk.group_keys(base.origin, origin_plan)
    assert [g["key"] for g in meta["groups"]] == [keys[row] for _g, row in origin_plan]
    for g, (group, row) in zip(meta["groups"], origin_plan):
        assert g["row"] == row and g["origin"] == group.width and g["plan"] == targets[row]
        assert g["cut"] == cut[g["producers"][0]] and g["producers"] == sorted(g["producers"]) and g["cut"] >= g["plan"]
    # read back by the widths kind, the table is the walk's targets again
    monkeypatch.setenv("SPECTRA_ALLOC_WIDTHS", base.plan)
    widths, info = alloc_walk.plan_targets(base.origin, [], SHAPE, "widths", 0.18, budget="flops", min_width=2)
    assert widths == targets and info["unnamed"] == 0
    # nothing else moves: the RNG, the origin, the candidate
    cand = _candidate(env)
    origin = env.data_dict[env.selected_net_path][0]
    sd0 = {k: v.clone() for k, v in origin.state_dict().items()}
    cand0 = alloc_walk.state_digest(cand["model"])
    for key, value in {**CELL_ENV, "SPECTRA_ALLOC_KIND": "sens_thresh"}.items():
        monkeypatch.setenv(key, value)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    torch.manual_seed(5)
    before = alloc_walk.rng_snapshot()
    again = alloc_walk.save_widths(env, env.selected_net_path, LABEL, cand, str(tmp_path / "w"))
    assert _equal(alloc_walk.rng_snapshot(), before)
    assert again == meta and alloc_walk.state_digest(cand["model"]) == cand0
    assert all(torch.equal(v, origin.state_dict()[k]) for k, v in sd0.items())
    assert len(lines) == 1 and lines[0].startswith(f"[eval] saved widths {LABEL} {name} step={cand['point']['step']} | plan ")
    assert f"sha {meta['plan_sha1'][:12]} | cut " in lines[0] and f"net sha {meta['state_sha1'][:12]}" in lines[0]
    # a walk that is not the alloc policy's has no plan: the cut and the meta alone
    monkeypatch.setenv("SPECTRA_EVAL_POLICY", "mild")
    other = alloc_walk.save_widths(env, env.selected_net_path, "size_flop0.20", cand, str(tmp_path / "m"))
    assert other["plan_sha1"] is None and other["kind"] is None and other["files"]["plan"] is None
    assert sorted(os.listdir(tmp_path / "m")) == [f"{name}__{LABEL}__cut.json", f"{name}__{LABEL}__meta.json"]
    assert all(g["plan"] is None for g in other["groups"]) and other["cut_sha1"] == meta["cut_sha1"]
    # a failed save is one line and no exception
    del lines[:]
    broken = types.SimpleNamespace(selected_net_path="x.pt", data_dict={})
    assert alloc_walk.save_widths(broken, "x.pt", LABEL, cand, str(tmp_path / "b")) is None
    assert len(lines) == 1 and lines[0].startswith(f"[eval] widths save failed for {LABEL} x.pt: KeyError")
    assert _equal(alloc_walk.rng_snapshot(), before)


def test_table_and_state_digests_name_one_plan_and_one_network():
    assert alloc_walk.table_digest({"b": 2, "a": 1}) == alloc_walk.table_digest({"a": 1, "b": 2})
    assert alloc_walk.table_digest({"a": 1, "b": 2}) != alloc_walk.table_digest({"a": 1, "b": 3})
    model = _model(4, 10)
    twin = copy.deepcopy(model)
    assert alloc_walk.state_digest(model) == alloc_walk.state_digest(twin)
    with torch.no_grad():
        next(twin.parameters()).view(-1)[0] += 1e-6
    assert alloc_walk.state_digest(model) != alloc_walk.state_digest(twin)


def _final_ft_fakes(monkeypatch, tmp_path):
    calls, saves, lines = [], [], []
    monkeypatch.setattr(runner, "_final_ft", lambda env, net_path, tag, cand, epochs, **kw: calls.append(tag))
    monkeypatch.setattr(alloc_walk, "save_widths", lambda env, net, label, cand, folder: saves.append((label, cand, folder)))
    monkeypatch.setattr(utils, "print_flush", lambda msg, *a, **k: lines.append(str(msg)))
    monkeypatch.setattr(recorder, "recorder", lambda: types.SimpleNamespace(run_dir=str(tmp_path)))
    return calls, saves, lines


def _cand(step, param, flop):
    point = {"step": step, "param": param, "flop": flop, "val_dacc_pp": -1.0, "test_acc": 0.5}
    return {"point": point, "model": object(), "key": (param, flop, 1.0)}


def test_the_runner_saves_widths_only_under_the_flag_and_only_for_size_points(monkeypatch, tmp_path):
    calls, saves, _lines = _final_ft_fakes(monkeypatch, tmp_path)
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "flop:0.45,0.20")
    labels = ["val_best", "size_flop0.45", "size_flop0.20"]

    def candidates():
        return {"val_best": _cand(9, 0.8, 0.8), "size_flop0.45": _cand(40, 0.4, 0.44), "size_flop0.20": _cand(77, 0.2, 0.199)}

    runner._run_final_ft(types.SimpleNamespace(), "/x/net.pt", {"val_best": {"step": 9}}, candidates(), 1)
    assert calls == labels and saves == []                                      # flag off: the fine-tunes, no save
    del calls[:]
    monkeypatch.setenv("SPECTRA_EVAL_SAVE_WIDTHS", "1")
    cands = candidates()
    runner._run_final_ft(types.SimpleNamespace(), "/x/net.pt", {"val_best": {"step": 9}}, cands, 1)
    assert calls == labels                                                      # the same fine-tunes, in the same order
    assert [(label, folder) for label, _c, folder in saves] == [("size_flop0.45", os.path.join(str(tmp_path), "alloc_widths")),
                                                                 ("size_flop0.20", os.path.join(str(tmp_path), "alloc_widths"))]
    assert saves[0][1] is cands["size_flop0.45"] and saves[1][1] is cands["size_flop0.20"]
    del calls[:], saves[:]
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SIZE_ONLY", "1")                  # with the size-only flag: one point, one save
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "flop:0.20")
    runner._run_final_ft(types.SimpleNamespace(), "/x/net.pt", {"val_best": {"step": 9}},
                         {"val_best": _cand(9, 0.8, 0.8), "size_flop0.20": _cand(77, 0.2, 0.199)}, 1)
    assert calls == ["size_flop0.20"] and [label for label, _c, _f in saves] == ["size_flop0.20"]


# ------------------------------------------------------------------ SPECTRA_EVAL_FINAL_FT_SIZE_ONLY (v34, as it is)


def test_size_only_fine_tunes_the_size_point_under_its_own_label_also_at_val_bests_step(monkeypatch, tmp_path):
    calls, _saves, lines = _final_ft_fakes(monkeypatch, tmp_path)
    monkeypatch.setenv("SPECTRA_EVAL_SIZE_POINTS", "flop:0.20")

    def same_step():
        return {"val_best": _cand(77, 0.2, 0.199), "size_flop0.20": _cand(77, 0.2, 0.199)}

    # flag off: the size point is the val_best point, fine-tuned once under val_best's label, so it has no record of its own
    runner._run_final_ft(types.SimpleNamespace(), "/x/net.pt", {"val_best": {"step": 77}}, same_step(), 20)
    assert calls == ["val_best"]
    assert any(line.startswith("[eval] TRAJ final_ft size_flop0.20 net.pt step=77 | same point as val_best") for line in lines)
    del calls[:], lines[:]
    # flag on: val_best is dropped first, and the size point gets the fine-tune and the record under its own label
    monkeypatch.setenv("SPECTRA_EVAL_FINAL_FT_SIZE_ONLY", "1")
    cands = same_step()
    runner._run_final_ft(types.SimpleNamespace(), "/x/net.pt", {"val_best": {"step": 77}}, cands, 20)
    assert calls == ["size_flop0.20"] and list(cands) == ["size_flop0.20"]
    assert any("SIZE_ONLY drops val_best (step 77)" in line for line in lines)
    assert not any("same point as" in line for line in lines)
    del calls[:], lines[:]
    # two different points: the candidate list is the size point alone, whatever val_best is
    cands = {"val_best": _cand(3, 0.9, 0.9), "size_flop0.20": _cand(77, 0.2, 0.199)}
    runner._run_final_ft(types.SimpleNamespace(), "/x/net.pt", {"val_best": {"step": 3}}, cands, 20)
    assert calls == ["size_flop0.20"] and list(cands) == ["size_flop0.20"]
    del calls[:]
    cands = {"size_flop0.20": _cand(77, 0.2, 0.199)}                            # no val_best copy at all (the origin holds it)
    runner._run_final_ft(types.SimpleNamespace(), "/x/net.pt", {"val_best": {"step": -1}}, cands, 20)
    assert calls == ["size_flop0.20"]


# ------------------------------------------------------------------ the search's arithmetic (the driver's step)


def test_pg_update_is_sigma_times_the_mean_of_advantage_times_eps_on_a_hand_computed_case():
    step = _step_module()
    eps = [[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, -2.0]]                    # two mirrored pairs, two groups
    r = [3.0, 2.0, 1.0, 0.0]
    adv = step.pg_advantages(r, [False] * 4)
    sd = math.sqrt(5.0 / 3.0)                                                   # deviations 1.5, 0.5, -0.5, -1.5; sample SD 1.29099
    assert list(adv) == pytest.approx([d / (sd + 1e-6) for d in (1.5, 0.5, -0.5, -1.5)], abs=1e-12)
    assert list(adv) == pytest.approx([1.161895, 0.387298, -0.387298, -1.161895], abs=1e-5)
    mu = step.pg_update([0.1, -0.2], 0.3, eps, r, [False] * 4)
    #   sum_i A_i eps_i = (A0 - A2, 2 A1 - 2 A3) = (2.0, 4.0) / sd; its mean over the 4 plans = (0.5, 1.0) / sd
    #   delta mu = sigma * that = (0.15, 0.30) / 1.29099 = (0.116190, 0.232379)
    assert list(mu) == pytest.approx([0.1 + 0.15 / (sd + 1e-6), -0.2 + 0.30 / (sd + 1e-6)], abs=1e-12)
    assert mu[0] == pytest.approx(0.1 + 0.116190, abs=1e-5) and mu[1] == pytest.approx(-0.2 + 0.232379, abs=1e-5)
    t = torch.tensor(r, dtype=torch.float64)                                    # the plan trainer's own normalisation
    assert list(adv) == pytest.approx(((t - t.mean()) / (t.std() + 1e-6)).tolist(), abs=1e-12)
    assert list(step.pg_update([0.0, 0.0], 0.3, eps, [2.0] * 4, [False] * 4)) == [0.0, 0.0]   # all tied: the mean stays
    assert list(step.pg_advantages([5.0, 5.0], [False, False])) == [0.0, 0.0]
    with pytest.raises(ValueError):
        step.pg_update([0.0, 0.0], 0.3, eps[:3], r, [False] * 4)
    with pytest.raises(ValueError):
        step.pg_advantages([1.0, float("nan")], [False, False])


def test_an_oversize_plan_takes_the_rounds_lowest_reward_before_the_standardizing():
    step = _step_module()
    r = [9.0, 1.0, 2.0, 0.0]
    adv = step.pg_advantages(r, [True, False, False, False])                    # the best val is over-size: it becomes 0.0
    assert list(adv) == pytest.approx(list(step.pg_advantages([0.0, 1.0, 2.0, 0.0], [False] * 4)), abs=1e-12)
    assert adv[0] == adv[3] == min(adv) and adv[2] == max(adv)
    low = step.pg_advantages([-5.0, 1.0, 2.0, 0.0], [True, False, False, False])   # already the lowest: unchanged
    assert list(low) == pytest.approx(list(step.pg_advantages([-5.0, 1.0, 2.0, 0.0], [False] * 4)), abs=1e-12)
    every = step.pg_advantages([3.0, 1.0, 2.0, 0.0], [True] * 4)                # every plan over-size: no signal
    assert list(every) == [0.0] * 4
    assert step.eligible(0.110, 0.100) and not step.eligible(0.1101, 0.100) and step.CAP == 1.10
    eps = step.mirrored_eps(step.draw_rng(0, "pg", 1), 8, 5)
    mu = step.pg_update([0.0] * 5, 0.3, eps, [1, 2, 3, 4, 5, 6, 7, 8], [False] * 7 + [True])
    want = 0.3 * (np.asarray(step.pg_advantages([1, 2, 3, 4, 5, 6, 7, 1], [False] * 8))[:, None] * eps).mean(axis=0)
    assert list(mu) == pytest.approx(list(want), abs=1e-12)


def test_the_draws_reproduce_are_mirrored_and_follow_the_registered_sigmas():
    step = _step_module()
    g = 7
    z1, eps1 = step.pg_round(2, 1, np.zeros(g), g)
    z1b, eps1b = step.pg_round(2, 1, np.zeros(g), g)
    assert np.array_equal(z1, z1b) and np.array_equal(eps1, eps1b) and z1.shape == (8, g)
    assert np.array_equal(eps1[4:], -eps1[:4]) and np.allclose(z1, 0.3 * eps1) and np.allclose(z1.mean(axis=0), 0.0)
    assert not np.array_equal(eps1, step.pg_round(2, 2, np.zeros(g), g)[1])     # another round, another draw
    assert not np.array_equal(eps1, step.pg_round(3, 1, np.zeros(g), g)[1])     # another network, another draw
    mu = np.arange(g) / 10.0
    z2, eps2 = step.pg_round(2, 2, mu, g)
    assert np.allclose(z2, mu[None, :] + 0.3 * eps2) and np.allclose(z2.mean(axis=0), mu)
    rz = step.rand_z(2, g)
    assert rz.shape == (24, g) and np.array_equal(rz, step.rand_z(2, g)) and abs(float(step.rand_z(2, 4000).std()) - 0.3) < 0.01
    parent = np.linspace(-0.2, 0.2, g)
    for rnd, sigma in ((1, 0.3), (2, 0.2), (3, 0.15)):
        kids = step.evo_children(2, rnd, parent, g)
        assert kids.shape == (8, g) and np.array_equal(kids, step.evo_children(2, rnd, parent, g))
        wide = step.evo_children(2, rnd, np.zeros(3000), 3000)
        assert abs(float(wide.std()) - sigma) < 0.01
        assert np.allclose(kids - parent[None, :], step.evo_children(2, rnd, np.zeros(g), g))
    with pytest.raises(ValueError):
        step.mirrored_eps(step.draw_rng(0, "pg", 1), 7, g)
    assert (step.SIGMA_RAND, step.SIGMA_EVO, step.SIGMA_PG, step.ROUNDS, step.ROUND_N, step.N_EVAL) == (0.3, (0.3, 0.2, 0.15), 0.3, 3, 8, 24)


def test_evos_parent_and_every_pick_are_the_best_eligible_plan_with_ties_to_the_base():
    step = _step_module()
    base = {"id": "base", "val": 60.0, "eligible": True, "base": True}

    def kid(name, val, ok=True):
        return {"id": name, "val": val, "eligible": ok, "base": False}

    assert step.pick_best([base, kid("a", 59.0), kid("b", 61.0), kid("c", 60.5)])["id"] == "b"
    assert step.pick_best([base, kid("a", 62.0, ok=False), kid("b", 61.0)])["id"] == "b"      # over-size: never the parent
    assert step.pick_best([base, kid("a", 62.0, ok=False), kid("b", 59.9)])["id"] == "base"   # nothing eligible is better
    assert step.pick_best([base, kid("a", 60.0), kid("b", 60.0)])["id"] == "base"             # a tie goes to the base
    assert step.pick_best([base, kid("a", 61.0), kid("b", 61.0)])["id"] == "a"                # then to the earliest
    assert step.pick_best([base, kid("a", None), kid("b", float("nan"))])["id"] == "base"     # a missing record cannot win
    assert step.pick_best([kid("a", 61.0, ok=False)]) is None


def test_seed_blocks_are_disjoint_and_the_stages_cover_the_120_evaluations_once():
    step = _step_module()
    assert len(step.NETS) == 8 and step.LABEL == "size_flop0.20" and step.EVAL_EPOCHS == 20 and step.FINAL_EPOCHS == 100
    seeds = {}
    for n, (short, _f, _fam) in enumerate(step.NETS):
        keys = step.all_eval_keys(short)
        assert len(keys) == len(set(keys)) == 120
        by_arm = {arm: [k for k in keys if k.split("|")[1] == arm] for arm in step.ARMS}
        assert {arm: len(v) for arm, v in by_arm.items()} == {"rulek": 24, "rand": 24, "evo": 24, "pg": 24, "agent": 24}
        assert step.stage_keys(short, "A") == [f"{short}|rulek|0|0"] and len(step.stage_keys(short, "B")) == 87
        assert len(step.stage_keys(short, "C")) == len(step.stage_keys(short, "D")) == 16
        assert step.stage_keys(short, "B")[:16] == [f"{short}|evo|1|{i}" for i in range(8)] + [f"{short}|pg|1|{i}" for i in range(8)]
        for key in keys:
            _s, arm, rnd, idx = key.split("|")
            seeds[key] = step.seed_of(n, arm, int(rnd), int(idx))
        for j in range(step.FINAL_SEEDS):
            seeds[f"{short}|final|{j}"] = step.seed_of(n, "final", 0, j)
    assert len(set(seeds.values())) == len(seeds) == 8 * 122                      # no seed is used twice
    assert not set(seeds.values()) & (set(range(42, 49)) | set(range(90000, 90100)))   # the rule's of record, the smokes'
    blocks = {}
    for key, seed in seeds.items():
        net, arm, rnd = key.split("|")[:3] if "final" not in key else (key.split("|")[0], "final", "0")
        blocks.setdefault(seed // 100, set()).add((net, arm, rnd))
    assert all(len(v) == 1 for v in blocks.values())                              # one block of 100 per (network, arm, round)
    counts = {}
    for i in range(24):
        counts[step.agent_policy(i)[0]] = counts.get(step.agent_policy(i)[0], 0) + 1
    assert counts == {"s42": 8, "s43": 8, "s44": 8}                               # 8 samples of each final policy
    assert [p.split("/")[-3] for _t, p, _s in step.AGENT_POLICIES] == ["job24337054", "job24337055", "job24337056"]


def test_the_manifest_is_backed_up_and_rewritten_whole_and_one_call_submits_at_a_time(tmp_path, monkeypatch):
    step = _step_module()
    monkeypatch.setattr(step, "MAN", str(tmp_path / "manifest.tsv"))
    blank = {column: "" for column in step.MAN_COLS}
    for wanted in ("key", "net", "arm", "round", "index", "seed", "z_file", "new_job", "nice", "role"):   # the registered columns
        assert wanted in step.MAN_COLS
    step.man_append(dict(blank, key="v16|rulek|0|0", new_job="1", note="a\tb\nc"))
    assert not os.path.exists(step.MAN + ".bak")
    step.man_append(dict(blank, key="v16|rulek|0|1", new_job="2"))
    step.man_append(dict(blank, key="v16|rulek|0|0", new_job="3", note="resub #1 of 1: PREEMPTED"))
    rows = step.man_rows()
    assert [r["new_job"] for r in rows] == ["1", "2", "3"] and rows[0]["note"] == "a b c"
    assert step.newest(rows)["v16|rulek|0|0"]["new_job"] == "3"                 # the newest row of a key counts
    assert [r["new_job"] for r in step.tsv(step.MAN + ".bak")] == ["1", "2"]    # the file as it was before the last row
    assert not os.path.exists(step.MAN + ".tmp")
    monkeypatch.setattr(step, "LOCK", str(tmp_path / "submit.lock"))
    assert step.take_lock() and not step.take_lock()                            # a second submitting call is refused
    step.drop_lock()
    assert step.take_lock()
    old = time.time() - 1000.0
    os.utime(step.LOCK, (old, old))
    assert step.take_lock()                                                     # a lock a killed call left behind is taken over
    step.drop_lock()
    assert not os.path.exists(step.LOCK)


RULE_LINE = {  # the constant part of the rule jobs' logged env lines (tree_v35, s12-rs-rs-sthF-f20-*), one network's paths
    "SPECTRA_ACTOR_CHECKPOINT_PATH": "/x/tree_v10/runs/job22156116/snapshots/ep0127/latest_best_actor.pt",
    "SPECTRA_ALLOC_BUDGET": "flops", "SPECTRA_ALLOC_GRID_ROUND": "1", "SPECTRA_ALLOC_KIND": "sens_thresh",
    "SPECTRA_BATCH_SIZE": "256", "SPECTRA_DATABASE": "/x/rs_configs/input_rs_r20w12.json", "SPECTRA_DATASET_NAMES": "cifar-100",
    "SPECTRA_EVAL_DETERMINISTIC": "1", "SPECTRA_EVAL_FINAL_FT_BATCH": "256", "SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH": "1",
    "SPECTRA_EVAL_FINAL_FT_EPOCHS": "100", "SPECTRA_EVAL_FINAL_FT_LR": "0.1", "SPECTRA_EVAL_FINAL_FT_SELECT": "last",
    "SPECTRA_EVAL_FT_DETERMINISTIC": "1", "SPECTRA_EVAL_PASSES": "6", "SPECTRA_EVAL_POLICY": "alloc",
    "SPECTRA_EVAL_SIZE_MATCH": "flop:0.20", "SPECTRA_EVAL_SIZE_POINTS": "flop:0.20", "SPECTRA_EVAL_TRAJECTORY": "1",
    "SPECTRA_FIXED_TARGET": "1", "SPECTRA_FIXED_TARGET_METRIC": "flop", "SPECTRA_FORTIFY": "1", "SPECTRA_FT_AUG": "1",
    "SPECTRA_FT_AUG_GPU": "1", "SPECTRA_GROUP_ONCE_PER_PASS": "1", "SPECTRA_INPUT": "/x/rs_configs/input_rs_r20w12.json",
    "SPECTRA_JOB_NAME": "s12-rs-rs-sthF-f20-r20w12-s48", "SPECTRA_NICE": "5", "SPECTRA_NUM_EPOCHS": "0", "SPECTRA_PASSES": "6",
    "SPECTRA_PLAN_MIN_WIDTH": "walk", "SPECTRA_PROFILE": "eval_c10_thin_traj_gonce", "SPECTRA_REPO_DIR": "/x/tree_v35",
    "SPECTRA_RUN_DIR": "/x/tree_v35/runs/job24124961", "SPECTRA_RUN_ID": "job24124961", "SPECTRA_SEED": "48",
    "SPECTRA_SKIP_TRAIN": "1", "SPECTRA_STOP_FILE": "/x/tree_v35/runs/job24124961/SLURM_STOP", "SPECTRA_VAL_FROM_TEST": "1",
    "SPECTRA_WALL": "0-02:00:00"}


def test_the_copied_rule_line_is_the_networks_threshold_rule_job_with_the_lowest_id(tmp_path, monkeypatch):
    step = _step_module()
    fname = step.net_file("r20w12")
    rows = [("other.pt", "size_flop0.20", "sens_thresh", "job1"), (fname, "size_flop0.20", "sens_thresh", "job24203004"),
            (fname, "size_flop0.20", "agent", "job24100000"), (fname, "size_flop0.20", "sens_thresh", "job24124956"),
            (fname, "size_flop0.45", "sens_thresh", "job24000001")]
    path = tmp_path / "plans.tsv"
    path.write_text("group_net\tlabel\tfamily\tplan_id\tkind\torig_job\tcand\tgroup\n"
                    + "".join(f"{n}\t{label}\tf\tp\t{k}\t{j}\t1\tg\n" for n, label, k, j in rows), encoding="utf-8")
    monkeypatch.setattr(step, "PLAN", str(path))
    assert step.rule_jobs() == {"r20w12": "job24124956"}      # this network, this point, the rule's rows only, the lowest id


def test_a_jobs_env_is_the_rule_line_with_the_registered_changes_only():
    step = _step_module()
    step.check_rule_line(dict(RULE_LINE), "r20w12")
    for key, value in (("SPECTRA_ALLOC_KIND", "sens"), ("SPECTRA_EVAL_FINAL_FT_EPOCHS", "40"), ("SPECTRA_EVAL_FINAL_FT_LR", "0.05")):
        with pytest.raises(ValueError):
            step.check_rule_line({**RULE_LINE, key: value}, "r20w12")
    for key in ("SPECTRA_EVAL_FT_TRACE", "SPECTRA_ALLOC_TILT_PARAMS"):
        with pytest.raises(ValueError):
            step.check_rule_line({**RULE_LINE, key: "392"}, "r20w12")
    always = {"SPECTRA_REPO_DIR", "SPECTRA_JOB_NAME", "SPECTRA_NICE", "SPECTRA_WALL", "SPECTRA_SEED",
              "SPECTRA_EVAL_FINAL_FT_SIZE_ONLY", "SPECTRA_EVAL_SAVE_WIDTHS"}

    def changed(spec, orig=RULE_LINE):
        spec = dict({"name": "sn-x", "wall": step.WALL_EVAL, "seed": 123456, "epochs": 20, "role": "cell"}, **spec)
        spec["env"] = step.arm_env(spec)
        env, profile = step.build_env(orig, spec, "", check_paths=False)
        assert profile == "eval_c10_thin_traj_gonce" and env["SBATCH_CONSTRAINT"] == "rtx_4090"
        assert "SPECTRA_RUN_DIR" not in env and "SPECTRA_STOP_FILE" not in env and "SPECTRA_DEPENDENCY" not in env
        assert env["SPECTRA_EVAL_FINAL_FT_LR"] == "0.1" and env["SPECTRA_REPO_DIR"].endswith("/tree_v42")
        return env, {line.split(":")[0] for line in step.env_changes(orig, env)}

    env, keys = changed({"arm": "rulek"})                                       # RULE-K: the rule line itself
    assert keys == always | {"SPECTRA_EVAL_FINAL_FT_EPOCHS"} and env["SPECTRA_ALLOC_KIND"] == "sens_thresh"
    assert env["SPECTRA_EVAL_FINAL_FT_EPOCHS"] == "20" and env["SPECTRA_SEED"] == "123456"
    assert env["SPECTRA_EVAL_FINAL_FT_SIZE_ONLY"] == env["SPECTRA_EVAL_SAVE_WIDTHS"] == "1"
    assert "SPECTRA_ALLOC_TILT_PARAMS" not in env                               # RULE-K is not a tilt: no hold (amendment A1)
    for arm in ("rand", "evo", "pg"):                                           # the tilts, each with the params hold
        env, keys = changed({"arm": arm, "widths_file": "/w/base_plan.json", "z_file": "/w/z.json"})
        assert keys == always | {"SPECTRA_EVAL_FINAL_FT_EPOCHS", "SPECTRA_ALLOC_KIND", "SPECTRA_ALLOC_TILT_BASE", "SPECTRA_ALLOC_TILT_Z",
                                 "SPECTRA_ALLOC_TILT_PARAMS"}
        assert (env["SPECTRA_ALLOC_KIND"], env["SPECTRA_ALLOC_TILT_BASE"], env["SPECTRA_ALLOC_TILT_Z"]) == ("tilt", "/w/base_plan.json", "/w/z.json")
        assert env["SPECTRA_ALLOC_TILT_PARAMS"] == "equal"
        assert "SPECTRA_ALLOC_TILT_SIGMA" not in env and "SPECTRA_ALLOC_TILT_TARGET" not in env
    env, keys = changed({"arm": "tilt0", "widths_file": "/w/base_plan.json"})    # the z = 0 smoke
    assert env["SPECTRA_ALLOC_TILT_SIGMA"] == "0" and "SPECTRA_ALLOC_TILT_Z" not in env and env["SPECTRA_ALLOC_TILT_PARAMS"] == "equal"
    env, keys = changed({"arm": "agent", "policy_path": "/p/policy_latest.pt", "sample_seed": 140007})   # AGENT-S
    assert keys == always | {"SPECTRA_EVAL_FINAL_FT_EPOCHS", "SPECTRA_ALLOC_KIND", "SPECTRA_PLAN_AGENT", "SPECTRA_PLAN_MENU",
                             "SPECTRA_SENS_BN_RECAL", "SPECTRA_PLAN_MENU_SAMPLE", "SPECTRA_PLAN_MENU_SAMPLE_SIGMA_Z",
                             "SPECTRA_PLAN_MENU_SAMPLE_SIGMA_U", "SPECTRA_ALLOC_SAMPLE_SEED"}
    assert env["SPECTRA_ALLOC_KIND"] == "agent_sample" and env["SPECTRA_PLAN_MENU"] == "sens,uniform,sens_cost,sens_thresh"
    assert (env["SPECTRA_SENS_BN_RECAL"], env["SPECTRA_PLAN_MENU_SAMPLE"], env["SPECTRA_ALLOC_SAMPLE_SEED"]) == ("8", "1", "140007")
    assert (float(env["SPECTRA_PLAN_MENU_SAMPLE_SIGMA_Z"]), float(env["SPECTRA_PLAN_MENU_SAMPLE_SIGMA_U"])) == (0.1, 2.0)
    assert "SPECTRA_ALLOC_TILT_PARAMS" not in env                               # AGENT-S is not a tilt either
    env, keys = changed({"arm": "final", "role": "final", "widths_file": "/w/pick.json", "epochs": 100, "wall": step.WALL_FINAL})
    assert keys == always | {"SPECTRA_ALLOC_KIND", "SPECTRA_ALLOC_WIDTHS"}       # a final: the rule's 100 epochs, the widths kind
    assert (env["SPECTRA_ALLOC_KIND"], env["SPECTRA_ALLOC_WIDTHS"], env["SPECTRA_EVAL_FINAL_FT_EPOCHS"]) == ("widths", "/w/pick.json", "100")
    assert "SPECTRA_ALLOC_TILT_PARAMS" not in env                               # a final walks the picked plan's widths as saved
    # the efficiency flags stay as the copied line has them: absent stays absent, set stays set
    assert not {"SPECTRA_EVAL_CACHE_READS", "SPECTRA_EVAL_IDENTITY_MEMO", "SPECTRA_EVAL_SKIP_STATE"} & set(env)
    fast = {**RULE_LINE, "SPECTRA_EVAL_CACHE_READS": "1", "SPECTRA_EVAL_IDENTITY_MEMO": "1", "SPECTRA_EVAL_SKIP_STATE": "1"}
    env, keys = changed({"arm": "rulek"}, fast)
    assert (env["SPECTRA_EVAL_CACHE_READS"], env["SPECTRA_EVAL_IDENTITY_MEMO"], env["SPECTRA_EVAL_SKIP_STATE"]) == ("1", "1", "1")
    assert keys == always | {"SPECTRA_EVAL_FINAL_FT_EPOCHS"}
    env, _p = step.build_env(RULE_LINE, {"name": "n", "wall": "0-01:00:00", "seed": 1, "epochs": 20, "env": {}}, "afterany:5:6",
                             check_paths=False)
    assert env["SPECTRA_DEPENDENCY"] == "afterany:5:6"
    with pytest.raises(ValueError, match="does not exist"):
        step.build_env(RULE_LINE, {"name": "n", "wall": "0-01:00:00", "seed": 1, "epochs": 20, "env": {}}, "")


# ------------------------------------------------------------------ SPECTRA_PLAN_MENU_SAMPLE (the frozen menu agent's sample)


def _menu_agent(monkeypatch, tmp_path, policy=None):
    env, model, plan = _origin(monkeypatch)
    path = _save(tmp_path, policy if policy is not None else _hot_policy(), "menu", **RESIDUAL_META, **MENU_META)
    monkeypatch.setenv("SPECTRA_PLAN_MENU", MENU_FLAG)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    return env, model, plan, path, lines


def test_menu_sample_flag_off_a_sampled_menu_plan_is_refused_exactly_as_before(monkeypatch, tmp_path, v10_env):
    env, _model_, _plan_, path, _lines = _menu_agent(monkeypatch, tmp_path)
    assert plan_agent.menu_sample() is False
    assert plan_agent.menu_sample_sigmas() == (0.10, 2.0) == (plan_agent.MENU_SAMPLE_SIGMA_Z, plan_agent.MENU_SAMPLE_SIGMA_U)
    message = (f"the plan agent at {path} is a menu policy: an ensemble and a sampled plan "
               "(agent_sample) are not defined for it yet; plan with one checkpoint's mean (agent)")
    for raw in (None, "0", "", "no"):
        if raw is None:
            monkeypatch.delenv("SPECTRA_PLAN_MENU_SAMPLE", raising=False)
        else:
            monkeypatch.setenv("SPECTRA_PLAN_MENU_SAMPLE", raw)
        monkeypatch.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: pytest.fail("refused before measuring"))
        with pytest.raises(ValueError) as err:
            plan_agent.plan_for_env(env, 0.58, path, sample=(0.2, 3))
        assert str(err.value) == message
    monkeypatch.setenv("SPECTRA_PLAN_MENU_SAMPLE", "1")
    assert plan_agent.menu_sample() is True
    with pytest.raises(ValueError) as err:                                      # an ensemble of menu policies stays refused
        plan_agent.plan_for_env(env, 0.58, f"{path},{path}", sample=(0.2, 3))
    assert str(err.value) == message
    with pytest.raises(ValueError) as err:
        plan_agent.plan_for_env(env, 0.58, f"{path},{path}")
    assert str(err.value) == message
    for name in ("SPECTRA_PLAN_MENU_SAMPLE_SIGMA_Z", "SPECTRA_PLAN_MENU_SAMPLE_SIGMA_U"):
        for raw in ("-0.1", "nan", "x"):
            monkeypatch.setenv(name, raw)
            with pytest.raises(ValueError, match=name):
                plan_agent.menu_sample_sigmas()
        monkeypatch.setenv(name, "0.25")
    assert plan_agent.menu_sample_sigmas() == (0.25, 0.25)
    # what a checkpoint's trainer ended on: ledger 464's policies read 0.10 and 2.0
    blob = {"config": {"sigma": 0.5, "sigma_floor": 0.2, "sigma_decay": 0.6}, "residual": "sens", "residual_sigma": 0.25,
            "menu_sigma": 2.0}
    assert plan_agent.trained_sigmas(blob) == pytest.approx((0.10, 2.0))
    assert plan_agent.trained_sigmas({"config": {"sigma": 0.5, "sigma_floor": 0.2}}) == (0.2, None)
    assert plan_agent.trained_sigmas({}) == (None, None)


@pytest.mark.parametrize("budget", ["flops", "params"])
def test_menu_sample_at_sigma_zero_is_the_mean_plan_and_a_draw_is_the_trainers(monkeypatch, tmp_path, v10_env, budget):
    env, model, plan, path, lines = _menu_agent(monkeypatch, tmp_path)
    kw = {"budget": "flops"} if budget == "flops" else {}
    kappa = 0.6
    _seed_all(5)
    mean_w, mean_i = plan_agent.plan_for_env(env, kappa, path, kappa=kappa, **kw)
    rng_mean = alloc_walk.rng_snapshot()
    assert min(mean_i["menu"]["pi"]) >= 0.005                                   # no share under the mean plan's floor here
    monkeypatch.setenv("SPECTRA_PLAN_MENU_SAMPLE", "1")
    monkeypatch.setenv("SPECTRA_PLAN_MENU_SAMPLE_SIGMA_Z", "0")
    monkeypatch.setenv("SPECTRA_PLAN_MENU_SAMPLE_SIGMA_U", "0")
    del lines[:]
    _seed_all(5)
    w0, i0 = plan_agent.plan_for_env(env, kappa, path, sample=(0.5, 3), kappa=kappa, **kw)
    assert _equal(alloc_walk.rng_snapshot(), rng_mean)                          # the draw leaves the global generators alone
    assert w0 == mean_w and i0["kept"] == mean_i["kept"] and i0["sens"] == mean_i["sens"] == i0["mu"]
    assert i0["menu"]["pi"] == mean_i["menu"]["pi"] == i0["menu"]["pi_mean"] and i0["menu"]["u"] == i0["menu"]["psi"]
    assert i0["kind"] == "agent_sample" and i0["sample"] == {"around": "agent", "sigma": 0.0, "seed": 3, "dist": 0.0, "sigma_u": 0.0}
    assert i0["prior"] == mean_i["prior"]
    line = [x for x in lines if x.startswith("[alloc] plan agent menu sample: ")]
    assert len(line) == 1 and line[0].startswith("[alloc] plan agent menu sample: sigma_z=0 sigma_u=0 seed=3; shares drawn sens ")
    # the defaults: the final agent's training sigmas; the draw: the scores, then the logits, from one seeded generator
    monkeypatch.delenv("SPECTRA_PLAN_MENU_SAMPLE_SIGMA_Z")
    monkeypatch.delenv("SPECTRA_PLAN_MENU_SAMPLE_SIGMA_U")
    del lines[:]
    _seed_all(5)
    w, info = plan_agent.plan_for_env(env, kappa, path, sample=(0.5, 11), kappa=kappa, **kw)
    assert _equal(alloc_walk.rng_snapshot(), rng_mean)
    rows = list(info["mu"])
    generator = torch.Generator().manual_seed(11)
    eps = torch.randn(len(rows), generator=generator, dtype=torch.float64).tolist()
    eps_u = torch.randn(len(MENU), generator=generator, dtype=torch.float64).tolist()
    mu = [info["mu"][row] for row in rows]
    assert mu == [mean_i["sens"][row] for row in rows] and info["menu"]["psi"] == mean_i["menu"]["psi"]
    z = [info["sens"][row] for row in rows]
    assert z == [m + 0.10 * e for m, e in zip(mu, eps)]                         # sigma_z 0.10, not sample's 0.5
    u = [p + 2.0 * e for p, e in zip(info["menu"]["psi"], eps_u)]
    assert info["menu"]["u"] == u and info["menu"]["sigma_u"] == 2.0
    assert info["menu"]["pi"] == plan_agent.menu_shares(u) and info["menu"]["pi_mean"] == mean_i["menu"]["pi"]
    assert info["menu"]["pi"] != mean_i["menu"]["pi"] and sum(info["menu"]["pi"]) == pytest.approx(1.0)
    assert info["sample"]["sigma"] == 0.10 and info["sample"]["sigma_u"] == 2.0 and info["sample"]["seed"] == 11
    assert info["sample"]["dist"] == pytest.approx(0.10 * math.sqrt(sum(e * e for e in eps)))
    # decoded as the trainer decodes a sampled plan: the same scores and shares through plan_trainer._decode
    _seed_all(5)
    inst = plan_agent.NetInstance.from_env(env, "toy.pt", None, None, zero=("sens",), with_cost=True, with_curves=True)
    cfg = plan_trainer.config()
    at = plan_trainer.menu_at(inst, SETTINGS, RES, kappa, cfg, budget, 2)
    prior = plan_trainer.prior_of(inst, RES, budget)
    want, want_info = plan_trainer._decode(z, inst.cost_model(budget), kappa, cfg, 2, prior, 0.3, (plan_agent.menu_shares(u), at["rules"]))
    assert w == want and info["kept"] == want_info["kept"] and info["prior"]["weights"] == want_info["base"]
    assert w != mean_w
    line = [x for x in lines if x.startswith("[alloc] plan agent menu sample: ")]
    assert len(line) == 1 and line[0].startswith("[alloc] plan agent menu sample: sigma_z=0.1 sigma_u=2 seed=11; shares drawn "
                                                 + " ".join(f"{k} {p:.3f}" for k, p in zip(MENU, info["menu"]["pi"]))
                                                 + "; mean shares " + " ".join(f"{k} {p:.3f}" for k, p in zip(MENU, mean_i["menu"]["pi"])))
    assert line[0].endswith("; WARNING: not the sigma the agent trained at: sigma_u 2 (trained 0.5)")   # the test checkpoint's own
    assert [x for x in lines if x.startswith("[alloc] plan agent residual on ")][0].endswith(
        "; menu pi " + " ".join(f"{k} {p:.3f}" for k, p in zip(MENU, info["menu"]["pi"])))
    assert alloc_walk.grid_priority(info, rows) == {
        row: math.log(max(1e-12, want_info["base"][row])) + plan_agent.trust_score(info["sens"][row], 0.3) for row in rows}
    # seeded: the same seed, the same plan; another seed, another draw
    _seed_all(5)
    again_w, again_i = plan_agent.plan_for_env(env, kappa, path, sample=(0.9, 11), kappa=kappa, **kw)
    assert again_w == w and again_i["sens"] == info["sens"] and again_i["menu"]["u"] == u
    _seed_all(5)
    other_w, other_i = plan_agent.plan_for_env(env, kappa, path, sample=(0.5, 12), kappa=kappa, **kw)
    assert other_i["sens"] != info["sens"] and other_i["menu"]["u"] != u


def test_menu_sample_keeps_the_plain_softmax_where_the_mean_plan_drops_small_shares(monkeypatch, tmp_path, v10_env):
    policy = plan_agent.PlanPolicy(63, menu=len(MENU))                          # the zero-init head: psi = (6, 0, 0, 0)
    env, _model_, _plan_, path, _lines = _menu_agent(monkeypatch, tmp_path, policy)
    monkeypatch.setenv("SPECTRA_PLAN_MENU_SAMPLE", "1")
    monkeypatch.setenv("SPECTRA_PLAN_MENU_SAMPLE_SIGMA_Z", "0")
    monkeypatch.setenv("SPECTRA_PLAN_MENU_SAMPLE_SIGMA_U", "0")
    _seed_all(7)
    _w, mean_i = plan_agent.plan_for_env(env, 0.58, path)
    _seed_all(7)
    _w, info = plan_agent.plan_for_env(env, 0.58, path, sample=(0.5, 1))
    assert mean_i["menu"]["pi"] == [1.0, 0.0, 0.0, 0.0] == info["menu"]["pi_mean"]     # the mean plan: the floor drops 0.25 %
    assert info["menu"]["pi"] == plan_agent.menu_shares([6.0, 0.0, 0.0, 0.0])            # the sample: the trainer's plain softmax
    assert info["menu"]["pi"][1] == pytest.approx(0.00246, abs=1e-5)


def test_a_plain_policys_sample_and_the_walks_plan_line_are_as_before(monkeypatch, tmp_path, v10_env):
    env, _model_, _plan_ = _origin(monkeypatch)
    plain = _save(tmp_path, plan_agent.PlanPolicy(63), "plain", **RESIDUAL_META)
    outs = []
    for raw in (None, "1"):
        if raw is None:
            monkeypatch.delenv("SPECTRA_PLAN_MENU_SAMPLE", raising=False)
        else:
            monkeypatch.setenv("SPECTRA_PLAN_MENU_SAMPLE", raw)
        _seed_all(7)
        widths, info = plan_agent.plan_for_env(env, 0.58, plain, sample=(0.2, 3))
        outs.append((widths, info, alloc_walk.rng_snapshot()))
    assert outs[0][0] == outs[1][0] and outs[0][1] == outs[1][1] and _equal(outs[0][2], outs[1][2])
    assert outs[0][1]["sample"] == {"around": "agent", "sigma": 0.2, "seed": 3, "dist": outs[0][1]["sample"]["dist"]}
    assert "menu" not in outs[0][1]
    # the walk's plan line names sigma_u only when the sample carries one
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    monkeypatch.setenv("SPECTRA_ALLOC_KIND", "agent_sample")
    monkeypatch.setenv("SPECTRA_PLAN_AGENT", "/x/plan_agent/policy_latest.pt")
    model = _model(4, 10)
    base_w, base_i = _base_plan(monkeypatch, model, "sens", 0.5, "params", 1)
    for sample, want in (({"around": "agent", "sigma": 0.5, "seed": 5, "dist": 1.0}, "agent_sample sigma=0.5 seed=5 around"),
                         ({"around": "agent", "sigma": 0.1, "seed": 5, "dist": 1.0, "sigma_u": 2.0},
                          "agent_sample sigma=0.1 sigma_u=2 seed=5 around")):
        fake = dict(base_i, kind="agent_sample", policy="plan_agent/policy_latest.pt", sample=sample)
        monkeypatch.setattr(plan_agent, "plan_for_env", lambda env, target, path, k_min, **kw: (dict(base_w), dict(fake)))
        del lines[:]
        alloc_walk._state(types.SimpleNamespace(
            selected_net_path="net.pt", current_model=model, target_keep=0.5, train_loader=None, _input_shape=lambda: SHAPE,
            conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES, passes=6)))
        assert lines[0].startswith(f"[alloc] net.pt: {want} the mean of plan_agent/policy_latest.pt plan keeps x")


# ------------------------------------------------------------------ flags off: the untouched tree's outputs, byte for byte

DIGEST_SCRIPT = r'''
import copy, hashlib, json, os, sys, types
tree = os.path.abspath(sys.argv[1])
sys.path.insert(0, tree)
os.chdir(tree)
for key in list(os.environ):
    if key.startswith("SPECTRA_"):
        del os.environ[key]
import torch
from tests.test_pruning import _init_static_conf
_init_static_conf()
from src import alloc_walk, plan_agent
import src.group_sensitivity as group_sensitivity
import src.run_recorder as recorder
import src.utils as utils
from spectra_models_instantiation import thin_res_net
from tests.test_v10_fixed_target import _target_env

SHAPE = (3, 32, 32)
RATES = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
out, lines, records = {}, [], []
utils.print_flush = lambda s, *a, **k: lines.append(str(s))
recorder.record = lambda event, **k: records.append([event, json.loads(json.dumps(k, sort_keys=True, default=str))])


def setenv(flags):
    for key in list(os.environ):
        if key.startswith("SPECTRA_"):
            del os.environ[key]
    os.environ.update({k: str(v) for k, v in flags.items()})


def norm(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def batches(seed=3, n=2, classes=10):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(8, 3, 32, 32, generator=g), torch.randint(0, classes, (8,), generator=g)) for _ in range(n)]


def loader(seed=3, classes=10):
    g = torch.Generator().manual_seed(seed)
    x, y = torch.randn(32, 3, 32, 32, generator=g), torch.randint(0, classes, (32,), generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)


torch.manual_seed(0)
nets = {"w4": thin_res_net.resnet20(num_classes=10, large_input=False, width=4).eval(),
        "w8": thin_res_net.resnet20(num_classes=10, large_input=False, width=8).eval()}
RULES = ("uniform", "sens", "inner", "sample", "sens_cost", "sens_cost_at", "sens_curve", "sens_thresh", "global_mag", "lamp")

# 1. plan_targets, every rule kind: the params budget without a width floor, the FLOPs budget with the walk's floor (the
#    cells' setting, on the wider net); then the widths kind on a table
for name, model in nets.items():
    names = {id(m): n for n, m in model.named_modules()}
    plan = group_sensitivity.group_plan(alloc_walk.ModelWithRows(model))
    table = {names[id(p)]: max(2, g.width - 1 - k % 3) for k, (g, row) in enumerate(plan) for p in g.producers}
    path = os.path.join(sys.argv[2], "table_%s.json" % name)
    with open(path, "w") as fh:
        json.dump(table, fh)
    for budget, min_width in ((("params", 1),) if name == "w4" else (("flops", 2),)):
        for kind in RULES:
            setenv({"SPECTRA_ALLOC_SAMPLE_SEED": "4", "SPECTRA_ALLOC_SAMPLE_SIGMA": "0.4"})
            torch.manual_seed(11)
            widths, info = alloc_walk.plan_targets(copy.deepcopy(model), batches(), SHAPE, kind, 0.45, budget=budget,
                                                   min_width=min_width)
            out["plan/%s/%s/%d/%s" % (name, budget, min_width, kind)] = norm([widths, info, torch.rand(1).item()])
        setenv({"SPECTRA_ALLOC_WIDTHS": path})
        widths, info = alloc_walk.plan_targets(copy.deepcopy(model), batches(), SHAPE, "widths", 0.45, budget=budget,
                                               min_width=min_width)
        out["plan/%s/%s/%d/widths" % (name, budget, min_width)] = norm([widths, info])

# 2. _state: the plan line, the records, the entry; every rule kind, the grid rounding off and on, both budgets
def state(model, flags, patch=None):
    setenv(flags)
    del lines[:], records[:]
    env = types.SimpleNamespace(selected_net_path="/zoo/net.pt", current_model=copy.deepcopy(model), target_keep=0.5,
                                conf=types.SimpleNamespace(device="cpu", compression_rates_dict=RATES, passes=6, seed=3),
                                train_loader=loader(), val_loader=loader(5), _input_shape=lambda: SHAPE,
                                _dependency_groups=lambda mwr: None)
    torch.manual_seed(13)
    entry = alloc_walk._state(env)
    return norm([{k: (sorted(v) if isinstance(v, set) else v) for k, v in entry.items()}, list(lines), list(records),
                 torch.rand(1).item()])


for kind in RULES + ("widths",):
    for grid, budget in (("0", "params"), ("1", "flops"), ("1", "params")):
        if (grid, budget) == ("1", "params") and kind not in ("sens", "widths", "sens_thresh"):
            continue
        flags = {"SPECTRA_ALLOC_KIND": kind, "SPECTRA_ALLOC_GRID_ROUND": grid, "SPECTRA_ALLOC_BUDGET": budget,
                 "SPECTRA_GROUP_ONCE_PER_PASS": "1", "SPECTRA_ALLOC_SAMPLE_SEED": "2",
                 "SPECTRA_ALLOC_WIDTHS": os.path.join(sys.argv[2], "table_w4.json")}
        if budget == "flops":
            flags.update({"SPECTRA_EVAL_SIZE_MATCH": "flop:0.5", "SPECTRA_PLAN_MIN_WIDTH": "walk"})
        out["state/%s/%s/%s" % (kind, grid, budget)] = state(nets["w4"], flags)

# 3. _state for the kinds that plan elsewhere (the agent's, randbest, fallback): this module's own lines and records
torch.manual_seed(0)
base_w, base_i = alloc_walk.plan_targets(copy.deepcopy(nets["w4"]), batches(), SHAPE, "sens", 0.48)
rows = list(base_w)
z = {row: 0.1 * (k - 5) for k, row in enumerate(rows)}
fakes = {
    "agent": dict(base_i, kind="agent", alpha=0.0, sens=z, policy="plan_agent/policy_latest.pt", budget="params"),
    "agent_sample": dict(base_i, kind="agent_sample", alpha=0.0, sens=z, policy="plan_agent/policy_latest.pt", budget="params",
                         sample={"around": "agent", "sigma": 0.5, "seed": 2, "dist": 1.25}, mu=z),
    "randbest": dict(base_i, kind="randbest", alpha=0.0, sens=z, budget="params",
                     randbest={"pick": 3, "n": 8, "score": 0.5, "proxy": "cut", "sigma": 1.0, "seed": 7, "minutes": 0.1,
                               "scores": [0.1] * 8}),
    "fallback": dict(base_i, kind="sens", fallback={"pick": "prior", "prior": "sens", "proxies": ["cut"], "margin": 0.0,
                                                     "scores": {"cut": {"agent": 0.1, "prior": 0.2}}, "minutes": 0.1}),
}
real = (plan_agent.plan_for_env, alloc_walk.randbest_plan, alloc_walk.fallback_plan)
for kind, info in fakes.items():
    plan_agent.plan_for_env = lambda env, target, path, k_min, **kw: (dict(base_w), dict(fakes[kind]))
    alloc_walk.randbest_plan = lambda env, target, budget="params": (dict(base_w), dict(fakes["randbest"]))
    alloc_walk.fallback_plan = lambda env, target, flops, agent_kw, plan_kw: (dict(base_w), dict(fakes["fallback"]))
    for grid in ("0", "1"):
        flags = {"SPECTRA_ALLOC_KIND": kind, "SPECTRA_ALLOC_GRID_ROUND": grid, "SPECTRA_GROUP_ONCE_PER_PASS": "1",
                 "SPECTRA_PLAN_AGENT": "/x/plan_agent/policy_latest.pt", "SPECTRA_ALLOC_SAMPLE_SEED": "2",
                 "SPECTRA_ALLOC_RANDBEST_PROXY": "cut", "SPECTRA_ALLOC_FALLBACK_PRIOR": "sens", "SPECTRA_ALLOC_FALLBACK_PROXY": "cut"}
        out["state/%s/%s" % (kind, grid)] = state(nets["w4"], flags)
plan_agent.plan_for_env, alloc_walk.randbest_plan, alloc_walk.fallback_plan = real

# 4. the names kind() takes, and the priorities the grid rounding steps by
for name in alloc_walk.KINDS + alloc_walk.AMC_KINDS:
    setenv({"SPECTRA_ALLOC_KIND": name})
    out["kind/" + name] = alloc_walk.kind()
setenv({})
out["kind/default"] = alloc_walk.kind()
for kind, info in list(fakes.items()) + [("sens", base_i), ("uniform", dict(base_i, kind="uniform")),
                                         ("widths", dict(base_i, kind="widths"))]:
    out["priority/" + kind] = norm(alloc_walk.grid_priority(info, rows))

# 5. a real walk: the picks, the landing and the network, for two rules on a FLOPs budget with the grid rounding
for kind in ("uniform", "sens_thresh", "sens"):
    setenv({"SPECTRA_FIXED_TARGET": "1", "SPECTRA_FIXED_TARGET_METRIC": "flop", "SPECTRA_ALLOC_BUDGET": "flops",
            "SPECTRA_GROUP_ONCE_PER_PASS": "1", "SPECTRA_ALLOC_GRID_ROUND": "1", "SPECTRA_PLAN_MIN_WIDTH": "walk",
            "SPECTRA_EVAL_SIZE_MATCH": "flop:0.3", "SPECTRA_ALLOC_KIND": kind})
    del lines[:], records[:]
    torch.manual_seed(17)
    model = copy.deepcopy(nets["w8"])
    env, _seen = _target_env(model, target=0.3, accs=[0.9] * 900, passes=6)
    del env._register_group_lock
    env.conf.compression_rates_dict = RATES
    env.original_flops = utils.calc_flops(model, SHAPE)
    picks, done = [], False
    while not done and len(picks) < 800:
        legal = env.legal_action_mask(device="cpu")
        pick = int(alloc_walk.action(env, legal, RATES, "cpu").item())
        picks.append(pick)
        _, _, done = env.step(RATES[pick])
    sha = hashlib.sha1()
    for k, v in sorted(env.current_model.state_dict().items()):
        sha.update(k.encode()); sha.update(v.detach().cpu().contiguous().numpy().tobytes())
    out["walk/" + kind] = norm([picks, done, env.flops_ratio(), env.param_ratio(), sha.hexdigest(),
                                [l for l in lines if l.startswith("[alloc]")], [r for r in records if r[0].startswith("alloc")]])

# 6. plan_for_env: a plain residual policy (mean and sample) and a menu policy (mean), as the eval calls it
from tests import test_plan_menu as M
from tests.test_plan_agent import V10_FLAGS
import pathlib
patcher = types.SimpleNamespace(setattr=lambda obj, name, value: setattr(obj, name, value))
setenv(V10_FLAGS)
env, model, plan = M._origin(patcher)
tmp = pathlib.Path(sys.argv[2])
plain = M._save(tmp, plan_agent.PlanPolicy(63), "plain", **M.RESIDUAL_META)
menu = M._save(tmp, M._hot_policy(), "menu", **M.RESIDUAL_META, **M.MENU_META)
for tag, path, kw, flags in (("plain/mean", plain, {}, {}), ("plain/sample", plain, {"sample": (0.2, 3)}, {}),
                             ("plain/flops", plain, {"budget": "flops", "kappa": 0.6}, {}),
                             ("menu/mean", menu, {}, {"SPECTRA_PLAN_MENU": M.MENU_FLAG}),
                             ("menu/flops", menu, {"budget": "flops", "kappa": 0.6}, {"SPECTRA_PLAN_MENU": M.MENU_FLAG})):
    setenv(dict(V10_FLAGS, **flags))
    del lines[:]
    M._seed_all(7)
    widths, info = plan_agent.plan_for_env(env, 0.58, path, **kw)
    info = dict(info)
    if "menu" in info:
        info["menu"] = {k: v for k, v in info["menu"].items() if k != "seconds"}
    out["agent/" + tag] = norm([widths, info, [l.replace(str(tmp), "<tmp>") for l in lines], torch.rand(1).item()])
setenv(dict(V10_FLAGS, SPECTRA_PLAN_MENU=M.MENU_FLAG))
try:
    plan_agent.plan_for_env(env, 0.58, menu, sample=(0.2, 3))
    out["agent/menu/sample"] = "no error"
except ValueError as error:
    out["agent/menu/sample"] = str(error).replace(str(tmp), "<tmp>")

text = json.dumps(out, sort_keys=True)
with open(sys.argv[3], "w") as fh:
    fh.write(text)
print("DIGEST " + hashlib.sha256(text.encode()).hexdigest() + " " + str(len(out)))
'''


def _baseline_tree():
    for cand in (os.environ.get("SAMENET_BASELINE_TREE"), "/home/paretsky/scratch_audit/_samenet_base"):
        if cand and os.path.exists(os.path.join(cand, "src", "alloc_walk.py")):
            return cand
    return None


def test_flags_off_every_existing_kind_plans_logs_and_walks_as_the_untouched_tree_does(tmp_path):
    """The same script in two interpreters: this tree and the untouched one (tree_v41 + commit 997adbd; never tree_v41 itself,
    and no bytecode is written). It plans with every existing rule kind (the params budget on one net; the FLOPs budget
    with the walk's width floor on a wider one), runs ``_state`` for every kind with the grid rounding off and on (the
    agent kinds, randbest and fallback through a fixed fake plan), walks three rules for real, and calls ``plan_for_env``
    for a plain and a menu policy. Every output (widths, info, lines, records, the RNG left behind) must be equal."""
    base = _baseline_tree()
    if base is None:
        pytest.skip("no untouched tree to compare with (set SAMENET_BASELINE_TREE=<tree_v41 + commit 997adbd>)")
    script = tmp_path / "digest.py"
    script.write_text(DIGEST_SCRIPT, encoding="utf-8")
    outs = {}
    for name, tree in (("here", str(ROOT)), ("base", base)):
        work = tmp_path / name
        work.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SPECTRA_", "PYTHON"))}
        run = subprocess.run([sys.executable, "-I", "-B", str(script), tree, str(work), str(work / "out.json")], cwd=tree, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert run.returncode == 0, run.stdout[-3000:]
        digest = [line for line in run.stdout.splitlines() if line.startswith("DIGEST ")]
        assert len(digest) == 1, run.stdout[-3000:]
        outs[name] = (digest[0], json.loads((work / "out.json").read_text()))
    here, there = outs["here"][1], outs["base"][1]
    assert set(here) == set(there) and len(here) >= 85
    differing = sorted(key for key in here if here[key] != there[key])
    assert not differing, f"{len(differing)} outputs differ from the untouched tree, first: {differing[:5]}"
    assert outs["here"][0] == outs["base"][0]
    assert there["agent/menu/sample"].startswith("the plan agent at <tmp>/plan_agent/policy_menu.pt is a menu policy: an ensemble")
