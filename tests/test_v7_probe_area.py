"""
V7 (21 Sep): the selection score the governor freezes on.

The legacy probe score is ``1 − kept`` at the deepest in-band point of the argmax walk. It
saturates at the deepest *legal* walk (on thin probe nets: the mild clone) and is blind to
Δacc, so every 12/4 arm froze at the same 0.262. ``SPECTRA_PROBE_SCORE=area`` scores the
slack-weighted in-band cut area instead: deeper-in-band and kinder-at-equal-depth both win.

    python -m pytest tests/test_v7_probe_area.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from src import fortify  # noqa: E402
from src.NetworkEnv import NetworkEnv  # noqa: E402

TAU = 10.0


def _env():
    env = NetworkEnv.__new__(NetworkEnv)
    env._reset_episode_reward_stats()
    return env


def _walk(env, points):
    """points = [(kept_after_step, cumulative val Δacc pp)]; over-budget steps are not booked."""
    for kept, delta in points:
        if -delta <= TAU:
            env._account_inband_point(kept, delta, TAU)
    return env.episode_val_best_compression(), env.episode_inband_area()


def test_kind_flag_default_and_area(monkeypatch):
    monkeypatch.delenv("SPECTRA_PROBE_SCORE", raising=False)
    assert fortify.probe_score_kind() == "cut"
    monkeypatch.setenv("SPECTRA_PROBE_SCORE", "area")
    assert fortify.probe_score_kind() == "area"
    monkeypatch.setenv("SPECTRA_PROBE_SCORE", "bogus")
    assert fortify.probe_score_kind() == "cut"


def test_legacy_cut_is_blind_to_accuracy_but_area_is_not():
    """Same kept sequence, one walk 3 pp kinder at every step: equal cut score, higher area."""
    harsh = [(0.9, -4.0), (0.8, -7.0), (0.74, -9.5)]
    kind = [(0.9, -1.0), (0.8, -4.0), (0.74, -6.5)]
    cut_h, area_h = _walk(_env(), harsh)
    cut_k, area_k = _walk(_env(), kind)
    assert cut_h == pytest.approx(cut_k) == pytest.approx(0.26)     # the 0.26 ceiling
    assert area_k > area_h
    # hand computation: Σ removed × slack_frac
    assert area_h == pytest.approx(0.1 * 0.6 + 0.1 * 0.3 + 0.06 * 0.05)
    assert area_k == pytest.approx(0.1 * 0.9 + 0.1 * 0.6 + 0.06 * 0.35)


def test_area_rewards_a_deeper_in_band_walk_and_ignores_over_budget():
    shallow = [(0.9, -3.0), (0.85, -5.0)]
    deep = [(0.9, -3.0), (0.85, -5.0), (0.75, -8.0)]
    over = [(0.9, -3.0), (0.85, -5.0), (0.60, -14.0)]        # last step leaves the band
    _, a_shallow = _walk(_env(), shallow)
    cut_deep, a_deep = _walk(_env(), deep)
    cut_over, a_over = _walk(_env(), over)
    assert a_deep > a_shallow
    assert a_over == pytest.approx(a_shallow) and cut_over == pytest.approx(0.15)
    assert cut_deep == pytest.approx(0.25)


def test_recovery_after_an_over_budget_step_counts_from_the_last_in_band_point():
    env = _env()
    env._account_inband_point(0.9, -3.0, TAU)          # in band
    # over-budget step at 0.8 is not booked; a later FT recovers into the band at 0.7
    env._account_inband_point(0.7, -9.0, TAU)
    assert env.episode_val_best_compression() == pytest.approx(0.3)
    assert env.episode_inband_area() == pytest.approx(0.1 * 0.7 + 0.2 * 0.1)


def test_reset_clears_area():
    env = _env()
    env._account_inband_point(0.8, -2.0, TAU)
    assert env.episode_inband_area() > 0
    env._reset_episode_reward_stats()
    assert env.episode_inband_area() == 0.0 and env.episode_val_best_compression() == 0.0
