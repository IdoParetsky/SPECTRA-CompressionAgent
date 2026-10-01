"""O38 reward replay (scripts/reward_replay.py): per-step inputs and returns on a synthetic walk (CPU)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import reward_replay  # noqa: E402


def _ev(step, new_acc, before, after, mode="structural", rate=0.8):
    return {"event": "step", "phase": "eval_test", "network": "/x/net_a.pt", "step_index": step,
            "new_acc": new_acc, "baseline_acc": 0.90, "params_before_m": before, "params_after_m": after,
            "compression_rate": rate if mode == "structural" else 1.0, "prune_mode": mode}


def _walk(tmp_path):
    rows = [
        _ev(0, 0.90, 1.0, 1.0, mode="identity"),
        _ev(1, 0.85, 1.0, 0.8),     # ρ 20, Δ −5: in band
        _ev(2, 0.91, 0.8, 0.72),    # ρ 10, Δ +1: gain
        _ev(3, 0.70, 0.72, 0.648),  # ρ 10, Δ −20: miss
    ]
    path = tmp_path / "rank0.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return reward_replay.load_walks(str(path))["net_a.pt"]


def test_census_counts_arms_and_rho(tmp_path):
    c = reward_replay.census(_walk(tmp_path), tau=10.0)
    assert (c["cuts"], c["gain"], c["band"], c["miss"]) == (3, 1, 1, 1)
    assert abs(c["rho_max"] - 20.0) < 1e-9 and c["rho_gt1"] == 1.0
    assert abs(c["keep_end"] - 0.648) < 1e-9


def test_replay_matches_hand_computed_returns(tmp_path, monkeypatch):
    rows = _walk(tmp_path)
    monkeypatch.setenv("SPECTRA_REWARD_MODE", "neon")
    # live: +20 in band, +10 gain (cube rooted), −10 miss
    ret, (peak, where), share = reward_replay.replay(rows, "structural", "cbrt_cubes", 10.0, 1.0)
    assert abs(ret - 20.0) < 1e-6 and where[0] == 2 and abs(share - 10.0 / 30.0) < 1e-9
    # cubic gain: the gain pays 10³ = 1000
    ret, (peak, where), share = reward_replay.replay(rows, "structural", "cbrt_miss", 10.0, 1.0)
    assert abs(ret - (20.0 + 1000.0 - 10.0)) < 1e-6 and abs(peak - 1020.0) < 1e-6
    # NEON raw on the realised cut: the miss costs 10³ and the walk's peak is still cut 2
    ret, (peak, where), _ = reward_replay.replay(rows, "structural", "raw", 10.0, 1.0)
    assert abs(ret - (20.0 + 1000.0 - 1000.0)) < 1e-6 and where[0] == 2
