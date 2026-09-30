"""CPU tests for the two offline readers added with tree_v9c: the paired early-kill read
(``scripts/paired_steps.py``) and the final fine-tune's honest gain (``scripts/final_ft_readout.py``).

    python -m pytest tests/test_v9c_readouts.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import crossfit_readout  # noqa: E402
import final_ft_readout  # noqa: E402
import paired_steps  # noqa: E402


def _step(step, dacc_pp, params_m, net="/nets/resnet56_w4.pt", phase="eval_test", mode="structural"):
    base = 0.9
    return {"event": "step", "phase": phase, "net": net.rsplit("/", 1)[-1], "network": net,
            "step_index": step, "baseline_acc": base, "new_acc": round(base + dacc_pp / 100.0, 5),
            "params_after_m": params_m, "prune_mode": mode}


def _write(path, events):
    (path / "events").mkdir(parents=True, exist_ok=True)
    with open(path / "events" / "rank0.jsonl", "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return str(path)


def _walk(offset, n=20, start=0.0545):
    return [_step(k, -0.3 * k + offset, round(start * (0.98 ** k), 4)) for k in range(n)]


@pytest.mark.parametrize("offset,expected", [(-1.5, "KILL"), (1.5, "ADOPT?"), (0.2, "CONTINUE")])
def test_paired_verdicts_by_step(tmp_path, offset, expected):
    ctrl = _write(tmp_path / "ctrl", _walk(0.0))
    arm = _write(tmp_path / "arm", _walk(offset))
    pairs = paired_steps.pair(paired_steps.load_cuts(arm)["resnet56_w4.pt"],
                              paired_steps.load_cuts(ctrl)["resnet56_w4.pt"])
    assert len(pairs) == 20
    assert paired_steps.verdict(pairs)[0] == expected


def test_too_few_pairs_never_kill(tmp_path):
    ctrl = _write(tmp_path / "ctrl", _walk(0.0))
    arm = _write(tmp_path / "arm", _walk(-3.0, n=10))
    pairs = paired_steps.pair(paired_steps.load_cuts(arm)["resnet56_w4.pt"],
                              paired_steps.load_cuts(ctrl)["resnet56_w4.pt"])
    assert paired_steps.verdict(pairs, min_steps=15)[0] == "CONTINUE"


def test_masked_train_phase_rows_are_ignored_and_later_rows_win(tmp_path):
    events = _walk(0.0, n=3) + [_step(1, -50.0, 0.05, mode="masked"), _step(2, -50.0, 0.05, phase="train"),
                                _step(0, +7.0, 0.0545)]
    cuts = paired_steps.load_cuts(_write(tmp_path / "a", events))["resnet56_w4.pt"]
    assert sorted(cuts) == [0, 1, 2]
    assert cuts[0][0] == pytest.approx(7.0) and cuts[1][0] == pytest.approx(-0.3)


def test_pairing_by_params_respects_the_log_resolution():
    ctrl = {0: (-1.0, 0.0500), 1: (-2.0, 0.0400), 2: (-3.0, 0.0046)}
    arm = {0: (-0.5, 0.0502), 1: (-2.5, 0.0430), 2: (-2.0, 0.0047)}
    pairs = paired_steps.pair(arm, ctrl, by="params")
    assert [p[0] for p in pairs] == [0, 2]                  # 0.0430 vs 0.0400 is 7 % off: not paired


def test_paired_cli_prints_one_verdict_per_network(tmp_path, capsys):
    ctrl = _write(tmp_path / "ctrl", _walk(0.0))
    arm = _write(tmp_path / "arm", _walk(-2.0))
    paired_steps.main([arm, ctrl])
    out = capsys.readouterr().out
    assert out.startswith("resnet56_w4.pt: KILL | 20 paired cuts by step") and "arm better on 0%" in out


# ---------------------------------------------------------------- final FT readout

def _ft(label, step, walk, final, val_final=0.9, param=0.8):
    return {"event": "eval_traj_final_ft", "network": "/nets/r56.pt", "label": label, "step": step,
            "param": param, "flop": param, "test_origin": 0.80, "test_walk": walk, "test_final": final,
            "val_origin": 0.90, "val_walk": 0.9, "val_final": val_final}


def _rows(tmp_path, rows):
    return final_ft_readout.load_rows(_write(tmp_path, rows))["r56.pt"]


def test_honest_gain_subtracts_what_the_recipe_gives_the_unpruned_net(tmp_path):
    rows = _rows(tmp_path, [
        _ft("origin", -1, 0.80, 0.81, 0.905, 1.0),
        _ft("size_param0.80", 3, 0.75, 0.785, 0.88),
        _ft("val_best", 5, 0.78, 0.785),
    ])
    gain, honest = final_ft_readout.honest_gain(rows["size_param0.80"], rows["origin"])
    assert gain == pytest.approx(3.5) and honest == pytest.approx(2.5)
    assert final_ft_readout.verdict(honest) == "ADOPT"
    _, honest_vb = final_ft_readout.honest_gain(rows["val_best"], rows["origin"])
    assert final_ft_readout.verdict(honest_vb) == "CROSS-OFF"
    assert final_ft_readout.verdict(1.0) == "HOLD" and final_ft_readout.verdict(None) == "NO-ORIGIN"
    lines = final_ft_readout.readout(rows)
    size = next(ln for ln in lines if "size_param0.80" in ln)
    assert "honest +2.50 pp ADOPT" in size
    assert "10k -1.75 pp" in size                            # (0.88 + 0.785)/2 − (0.90 + 0.80)/2
    assert "10k n/a (val-selected)" in next(ln for ln in lines if "val_best" in ln)
    assert "origin change +1.00 pp" in next(ln for ln in lines if ln.strip().startswith("origin"))


def test_scratch_rows_read_against_origin_scratch_and_the_inherited_row(tmp_path):
    rows = _rows(tmp_path, [
        _ft("origin", -1, 0.80, 0.81, 0.905, 1.0),
        _ft("origin+scratch", -1, 0.80, 0.805, 0.9, 1.0),
        _ft("size_param0.80", 3, 0.75, 0.785),
        _ft("size_param0.80+scratch", 3, 0.75, 0.78),
    ])
    lines = final_ft_readout.readout(rows)
    scratch = next(ln for ln in lines if "size_param0.80+scratch" in ln)
    assert "honest +2.50 pp ADOPT" in scratch                # 3.0 gain − 0.5 origin+scratch change
    assert "scratch−inherit -0.50 pp" in scratch
    assert [ln.split()[0] for ln in lines] == ["size_param0.80", "size_param0.80+scratch", "origin",
                                               "origin+scratch"]


def test_missing_origin_is_flagged_not_quoted(tmp_path):
    rows = _rows(tmp_path, [_ft("size_param0.80", 3, 0.75, 0.785)])
    assert "honest n/a (no origin row)" in final_ft_readout.readout(rows)[0]


def test_a_recipe_that_hurts_the_origin_is_never_adopted(tmp_path):
    # the 1-epoch smoke 21730499 on r20-w2: origin −5.84 pp, raw gain −0.06 pp, "honest" +5.78
    rows = _rows(tmp_path, [
        _ft("origin", -1, 0.80, 0.7416, 0.85, 1.0),
        _ft("size_param0.80", 3, 0.75, 0.7494, 0.84),
    ])
    size = next(ln for ln in final_ft_readout.readout(rows) if "size_param0.80" in ln)
    assert "honest +5.78 pp ORIGIN-HURT" in size and "ADOPT" not in size
    assert final_ft_readout.verdict(2.5, origin_change=-0.4) == "ADOPT"


# ---------------------------------------------------------------- cross-fit readout

def _tp(step, param, val, test):
    return {"step": step, "param": param, "flop": param, "val_dacc_pp": val, "test_dacc_pp": test}


# val (half A) and TEST (half B) disagree about where the band ends
WALK = [_tp(-1, 1.0, 0.0, 0.0), _tp(0, 0.9, 0.8, 0.4), _tp(1, 0.8, -4.0, -6.0), _tp(2, 0.7, -9.0, -11.0),
        _tp(3, 0.6, -12.0, -9.5), _tp(4, 0.5, -14.0, -13.0)]


def test_crossfit_selects_on_each_half_and_reports_on_the_other():
    a, b, mean = crossfit_readout.crossfit(WALK, 10.0)
    assert a[0]["step"] == 2 and a[1] == pytest.approx(-11.0)      # val picks 0.7, TEST half reads −11
    assert b[0]["step"] == 3 and b[1] == pytest.approx(-12.0)      # TEST half picks 0.6, val half reads −12
    assert mean == pytest.approx(-11.5)


def test_size_points_use_both_halves_and_census_counts_gains():
    lines = crossfit_readout.readout(WALK, (10.0,), (("param", 0.8),))
    assert any("size_param0.80: TEST half -6.00 | val half -4.00 | 10k -5.00 pp" in ln for ln in lines)
    c = crossfit_readout.census(WALK)
    assert (c["cuts"], c["val_up"], c["test_up"], c["both_up"]) == (5, 1, 1, 1)
    assert c["deepest_both_up"]["step"] == 0 and c["max_val_pp"] == pytest.approx(0.8)
