"""
v37 (queue row PICK, default off): the val-picked fallback (``SPECTRA_ALLOC_KIND=fallback``) takes the threshold rule
(``sens_thresh``) and the two magnitude rules (``global_mag``, ``lamp``) as further priors in a list of two or more,
beside ``uniform`` (v32). Only names v36 refused are added: one name is v27, and every v32 list behaves as before.

Reuses the v32 fakes (``tests/test_v32_fallback_judge.py``), with the three rules added to their tables.

CPU only, no GPU, no datasets.  python -m pytest tests/test_v37_pick.py -v
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import test_v32_fallback_judge as v32  # noqa: E402  (builds the static conf and the fakes)
from tests.test_v32_fallback_judge import _clean_env  # noqa: E402,F401  (the autouse fixture, here too)
from src import alloc_walk  # noqa: E402

NEW = ("sens_thresh", "global_mag", "lamp")
SEVEN = "sens,uniform,sens_cost,sens_thresh,lamp,global_mag"   # PICK's list: the prior first, then the other five
ORDER = ["sens", "uniform", "sens_cost", "sens_thresh", "lamp", "global_mag"]


@pytest.fixture
def _tables(monkeypatch):
    """The v32 fake tables with the three rules added (distinct widths, RNG burns and train flags)."""
    for name, w, burn, flag in (("sens_thresh", {1: 1, 2: 6}, 5, False), ("global_mag", {1: 6, 2: 2}, 6, True),
                                ("lamp", {1: 2, 2: 2}, 8, False)):
        monkeypatch.setitem(v32.PRIOR_W, name, w)
        monkeypatch.setitem(v32.BURN, name, burn)
        monkeypatch.setitem(v32.TRAIN_AFTER, name, flag)
    yield


def test_the_new_rules_parse_only_in_a_list_and_old_settings_are_unchanged(monkeypatch):
    assert alloc_walk.FALLBACK_PRIORS == ("sens", "sens_cost")          # the v27 tuple stands
    assert alloc_walk.FALLBACK_MORE == ("uniform",) + NEW
    for raw, want in ((SEVEN, ORDER), ("sens, LAMP", ["sens", "lamp"]), ("sens_thresh,sens", ["sens_thresh", "sens"]),
                      ("sens_cost,uniform", ["sens_cost", "uniform"]), ("sens", ["sens"])):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", raw)
        assert alloc_walk.fallback_priors() == want
        assert alloc_walk.fallback_prior() == want[0]
    for bad in ("sens_thresh", "lamp", "global_mag", "uniform", "sens,inner", "sens,agent", "sens,sens_curve",
                "sens,sens_cost_at", "sens,randbest", "sens,fallback"):
        monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", bad)
        with pytest.raises(ValueError, match="SPECTRA_ALLOC_FALLBACK_PRIOR"):
            alloc_walk.fallback_priors()                                # a lone further rule stays refused
    monkeypatch.setenv("SPECTRA_ALLOC_FALLBACK_PRIOR", "sens,lamp,LAMP")
    with pytest.raises(ValueError, match="duplicate"):
        alloc_walk.fallback_priors()


@pytest.mark.parametrize("want", ["agent", "sens", "sens_thresh", "lamp", "global_mag"])
def test_seven_way_pick_plans_each_rule_from_the_same_start_and_returns_the_picked_arm(monkeypatch, _tables, want):
    v32._configure(monkeypatch, priors=SEVEN, proxy="ft1,cut,bn8")
    base = {"agent": 0.70, **{name: 0.69 for name in ORDER}}
    table = dict(base, **({want: 0.75} if want != "agent" else {}))
    scores = {"ft1": table, "cut": {"agent": 0.1, **{n: 0.9 for n in ORDER}}, "bn8": dict(base)}   # only ft1 decides
    fakes = v32._Fakes(monkeypatch, scores)
    widths, info, lines, records, after, env = v32._run_plan(monkeypatch)
    # every rule is planned from the same start (its own calibration draw first), the agent from the start itself
    v32._seed_all()
    start = v32._peek()
    v32._burn(1)
    after_calibration = v32._peek()
    assert [k for k, _p, _f in fakes.entries] == ORDER + ["agent"]
    assert [p for _k, p, _f in fakes.entries] == [after_calibration] * len(ORDER) + [start]
    assert [f for _k, _p, f in fakes.entries] == [False] * (len(ORDER) + 1)   # the origin's flag put back each time
    assert [c[:2] for c in fakes.calls if c[0] in ("calibration", "plan_targets")] == [
        x for name in ORDER for x in (("calibration",), ("plan_targets", name))]
    # the returned plan, info, RNG and flag are the picked arm's alone
    assert widths is (v32.AGENT_W if want == "agent" else v32.PRIOR_W[want])
    expected = fakes.agent_info if want == "agent" else fakes.prior_infos[want]
    assert {k: v for k, v in info.items() if k != "fallback"} == expected
    assert after == v32._reference_draw(want)
    assert env.current_model.training is (True if want == "agent" else v32.TRAIN_AFTER[want])
    summary = info["fallback"]
    assert summary["pick_name"] == want and summary["pick"] == ("agent" if want == "agent" else "prior")
    assert summary["priors"] == ORDER and summary["prior"] == "sens"
    assert summary["pick2"] == alloc_walk.fallback_pick(table["agent"], table["sens"], 0.0)
    assert summary["scores"]["ft1"]["priors"] == {n: table[n] for n in ORDER}
    # per proxy: one batches draw, the agent, then the six rules in the listed order, all on the same batches
    assert [c[:3] for c in fakes.calls if c[0] in ("batches", "score")] == [
        x for proxy in ("ft1", "cut", "bn8")
        for x in [("batches", proxy, 0), ("score", "agent", proxy)] + [("score", n, proxy) for n in ORDER]]
    # lines: per proxy, the first prior's v27 line and five prior<i> lines; then the decision line
    assert len(lines) == 3 * 6 + 1
    assert lines[5] == (f"[alloc] fallback {v32.NET} ft1: agent {table['agent']:.4f} prior6 global_mag "
                        f"{table['global_mag']:.4f} (prior6 − agent {table['global_mag'] - table['agent']:+.4f})")
    decision = v32.DECISION_MULTI.match(lines[-1])
    assert decision and decision.groups()[:3] == (v32.NET, want, "ft1") and decision.group(7) == "sens"
    event, rec = records[0]
    assert event == "alloc_fallback" and rec["pick_name"] == want and rec["priors"] == ORDER
    assert set(rec["prior_widths_all"]) == set(ORDER)
    assert rec["prior_widths_all"]["lamp"] == {"1": 2, "2": 2}


def test_a_tie_between_rules_goes_to_the_first_listed_and_a_tie_with_the_agent_to_the_agent(monkeypatch, _tables):
    v32._configure(monkeypatch, priors=SEVEN, proxy="ft1")
    v32._Fakes(monkeypatch, {"ft1": {"agent": 0.70, "sens": 0.60, "uniform": 0.60, "sens_cost": 0.60,
                                     "sens_thresh": 0.72, "lamp": 0.72, "global_mag": 0.72}})
    _w, info, *_ = v32._run_plan(monkeypatch)
    assert info["fallback"]["pick_name"] == "sens_thresh"               # listed before lamp and global_mag
    v32._Fakes(monkeypatch, {"ft1": {"agent": 0.72, "sens": 0.60, "uniform": 0.60, "sens_cost": 0.60,
                                     "sens_thresh": 0.72, "lamp": 0.72, "global_mag": 0.72}})
    _w, info, *_ = v32._run_plan(monkeypatch)
    assert info["fallback"]["pick_name"] == "agent"


def test_the_decision_line_names_a_new_rule(monkeypatch, _tables):
    v32._configure(monkeypatch, priors="sens,lamp", proxy="ft1")
    v32._Fakes(monkeypatch, {"ft1": {"agent": 0.50, "sens": 0.40, "lamp": 0.60}})
    _w, info, lines, *_ = v32._run_plan(monkeypatch)
    assert re.match(r"^\[alloc\] fallback toy\.pt: pick lamp on ft1 \(margin 0; \d+\.\d min\) \| two-way pick agent \(sens\)$",
                    lines[-1])
