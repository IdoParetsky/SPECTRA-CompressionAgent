"""
EXIT (queue row EXIT, 26 Nov 2026): expert iteration in the plan trainer, ``SPECTRA_PLAN_EXIT=cem:R:m:shrink:B:n`` (default
off; the block in ``src/plan_trainer.py`` documents it), and the planted benchmark's EXIT configs and equal-evaluation
table (``scripts/_tmp_s26nov_exit_plant.py``).

* With the flag unset the trainer is the trainer of blob eaaf874e: its ``run`` is executed beside the reference file's own
  (``SPECTRA_EXIT_REF_TRAINER``, default ``/home/paretsky/scratch_audit/_av2_stage/src/plan_trainer.py``, checked against
  the blob) on the three lines of record and on two lines with AGENT-V2 flags on, and every log line, record,
  checkpoint and trainer state is compared.
* With it set: the grammar and every refusal (before any origin is read), round 1 = the flag-off draw, R x K reward
  evaluations per instance, the cross-entropy arithmetic on a scripted reward, the live gradient, the replay buffer
  (its own generator, FIFO), a stopped-and-resumed run equal to the uninterrupted one, and the benchmark script's
  ``run --evals`` / ``table`` / ``check`` (the run on the real cache, when it exists).

CPU only, no datasets.  python -m pytest tests/test_exit.py -v
"""

import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import V10_FLAGS  # noqa: E402,F401  (the v10_env fixture's flags)
from tests.test_plan_menu import MENU_FLAG, T1  # noqa: E402,F401
from tests.test_v25_plan_resume import _equal, _keep_states, _prepare, _steps, _weights  # noqa: E402
from tests.test_v25_plan_resume import _its as _line_its, _run as _job  # noqa: E402
from tests.test_agent_v2 import (AV2_FLAGS, FLAG_PREFIXES, LINES_OF_RECORD, MENU_LINE, SHARD, _clear,  # noqa: E402
                                 _go, _its, _menu_instance, _same_folder, _spy, _timeless)
from src import plan_agent, plan_trainer  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

REF_BLOB = "eaaf874e638df115ca0234c273027a211dcbbba6"
REF_PATH = os.environ.get("SPECTRA_EXIT_REF_TRAINER") or "/home/paretsky/scratch_audit/_av2_stage/src/plan_trainer.py"
SPEC = "cem:3:2:0.6:2048:4"
LINE_FLAGS = {"direct": {}, "residual": dict(T1), "menu": dict(MENU_LINE)}
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "_tmp_s26nov_exit_plant.py"
CACHE = "/home/paretsky/scratch_audit/av2_diag/instances_plant.pt"
CALIB_DIR = "/home/paretsky/scratch_audit/av2_diag/plant"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(FLAG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


@pytest.fixture
def v10_env(monkeypatch):
    for key, value in V10_FLAGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SPECTRA_STATE_TOKENS", raising=False)


def _blob(path):
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.fixture(scope="module")
def ref():
    """The trainer of blob eaaf874e as a second module (the same ``plan_agent`` and ``utils`` underneath)."""
    if not os.path.isfile(REF_PATH):
        pytest.skip(f"no reference trainer at {REF_PATH} (set SPECTRA_EXIT_REF_TRAINER)")
    assert _blob(REF_PATH) == REF_BLOB, f"{REF_PATH} is not blob {REF_BLOB}"
    spec = importlib.util.spec_from_file_location("plan_trainer_eaaf874e", REF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ 1. flag off: the trainer of eaaf874e, to the bit

_AV2_ON = {"SPECTRA_PLAN_EMA": "0.9", "SPECTRA_PLAN_LR_SCHEDULE": "cosine", "SPECTRA_PLAN_ADV": "rank",
           "SPECTRA_PLAN_GNORM": "sqrt", "SPECTRA_PLAN_MIRROR": "1"}
_AV2_MENU = {"SPECTRA_PLAN_EMA": "0.9", "SPECTRA_PLAN_LR_SCHEDULE": "cosine", "SPECTRA_PLAN_MENU_BOUND": "4",
             "SPECTRA_PLAN_MENU_ENT": "0.05", "SPECTRA_PLAN_MENU_KAPPA": "bins", "SPECTRA_PLAN_SPLIT": "1"}
IDENTITY = {**LINES_OF_RECORD,
            "residual_av2": {**LINES_OF_RECORD["residual"], **_AV2_ON},
            "menu_av2": {**LINES_OF_RECORD["menu"], **_AV2_MENU}}


@pytest.mark.parametrize("line", list(IDENTITY))
def test_flag_off_the_trainer_is_the_trainer_of_blob_eaaf874e_to_the_bit(monkeypatch, tmp_path, v10_env, ref, line):
    """One job per trainer (the reference file's ``run`` and this tree's), the same instance, every global generator
    seeded alike (the reward draws from all three): the same log lines, records, checkpoints and resumable state."""
    out = {}
    for name, module in (("ref", ref), ("now", plan_trainer)):
        _clear(monkeypatch)
        inst = _menu_instance("synthetic")
        lines = _prepare(monkeypatch, tmp_path / name, inst, **IDENTITY[line])
        monkeypatch.setattr(module, "out_dir", lambda folder=str(tmp_path / name): folder)
        assert not any(os.environ.get(f"SPECTRA_PLAN_{flag}") for flag in AV2_FLAGS + ("EXIT",)) or "av2" in line
        module.run(None, SHARD)
        out[name] = [_timeless(text, tmp_path / name) for text in lines]
    assert out["ref"] == out["now"]
    assert " exit=" not in out["now"][0] and any(text.startswith("[plan] DONE ") for text in out["now"])
    _same_folder(tmp_path / "ref", tmp_path / "now")
    state = plan_agent.read_blob(str(tmp_path / "now" / "trainer_state.pt"))
    assert "exit" not in state["meta"] and "exit" not in (state.get("av2") or {})
    steps = _its([json.loads(text) for text in open(tmp_path / "now" / "plan_train.jsonl", encoding="utf-8")])
    assert sum(abs(r["loss"]) > 1e-3 for r in steps) >= 4 and not any("exit" in r for r in steps)  # not zeros


def test_the_helpers_keep_their_old_behaviour_without_exit():
    assert plan_trainer.av2_state() is None and plan_trainer.av2_state(None, None, None) is None
    assert plan_trainer.exit_settings(plan_trainer.config()) is None
    ema = plan_trainer.PolicyEma(plan_agent.PlanPolicy(63), 0.9)
    assert set(plan_trainer.av2_state(ema, None)) == {"ema", "adv_scale"}  # no exit key without the buffer
    with pytest.raises(ValueError, match="carries no ema"):
        plan_trainer.restore_av2({"done": 2}, "/s.pt", ema)
    with pytest.raises(ValueError, match="carries no exit"):
        plan_trainer.restore_av2({"done": 2}, "/s.pt", exit=plan_trainer.ExitBuffer(4, 0))


# ------------------------------------------------------------------ 2. the grammar and its refusals


def test_the_spec_parses_and_every_malformed_one_is_refused(monkeypatch):
    cfg = plan_trainer.config()  # K = 8
    assert plan_trainer.exit_settings(cfg, None, {}) is None
    monkeypatch.setenv("SPECTRA_PLAN_EXIT", "   ")
    assert plan_trainer.exit_settings(cfg, None, {}) is None
    good = {SPEC: {"spec": SPEC, "rounds": 3, "elite": 2, "shrink": 0.6, "buffer": 2048, "replay": 4},
            "cem:1:8:1:1:0": {"spec": "cem:1:8:1:1:0", "rounds": 1, "elite": 8, "shrink": 1.0, "buffer": 1, "replay": 0},
            " CEM:2:1:1.0:5:0 ": {"spec": "cem:2:1:1:5:0", "rounds": 2, "elite": 1, "shrink": 1.0, "buffer": 5,
                                  "replay": 0},
            "cem:4:3:0.25:100:16": {"spec": "cem:4:3:0.25:100:16", "rounds": 4, "elite": 3, "shrink": 0.25,
                                    "buffer": 100, "replay": 16}}
    for raw, want in good.items():
        monkeypatch.setenv("SPECTRA_PLAN_EXIT", raw)
        assert plan_trainer.exit_settings(cfg, None, {}) == want, raw
    bad = ["cem", "cem:3:2:0.6:2048", "cem:3:2:0.6:2048:4:1", "es:3:2:0.6:2048:4", "0", "off", "1", "cem::2:0.6:8:1",
           "cem:0:2:0.6:8:1", "cem:3:0:0.6:8:1", "cem:3:9:0.6:8:1", "cem:3:2:0:8:1", "cem:3:2:1.5:8:1",
           "cem:3:2:-0.5:8:1", "cem:3:2:nan:8:1", "cem:3:2:inf:8:1", "cem:3:2:x:8:1", "cem:2.5:2:0.6:8:1",
           "cem:3:two:0.6:8:1", "cem:3:2:0.6:0:1", "cem:3:2:0.6:8:-1", "cem:3:2:0.6:8:1.5", "cem:-3:2:0.6:8:1"]
    for raw in bad:
        monkeypatch.setenv("SPECTRA_PLAN_EXIT", raw)
        with pytest.raises(ValueError, match="SPECTRA_PLAN_EXIT"):
            plan_trainer.exit_settings(cfg, None, {})
    monkeypatch.setenv("SPECTRA_PLAN_K", "4")  # m is bounded by this run's K
    monkeypatch.setenv("SPECTRA_PLAN_EXIT", "cem:3:5:0.6:8:1")
    with pytest.raises(ValueError, match="outside 1..K"):
        plan_trainer.exit_settings(plan_trainer.config(), None, {})
    monkeypatch.setenv("SPECTRA_PLAN_EXIT", "cem:3:4:0.6:8:1")
    assert plan_trainer.exit_settings(plan_trainer.config(), None, {})["elite"] == 4


_INCOMPATIBLE = [("MIRROR", "1", {}), ("ADV", "rank", {}), ("GNORM", "sqrt", {}),
                 ("FS_GUARD", "2", {"SPECTRA_PLAN_PROXY": "fs4"}),
                 ("SPLIT", "1", {**MENU_LINE}), ("MENU_SIGMA_ANNEAL", "1", {**MENU_LINE})]


@pytest.mark.parametrize("flag, value, more", _INCOMPATIBLE)
def test_exit_beside_an_incompatible_flag_is_refused_before_any_origin_is_read(monkeypatch, tmp_path, v10_env, flag,
                                                                               value, more):
    def origin(*args, **kwargs):
        pytest.fail("an origin was read")

    _clear(monkeypatch)
    monkeypatch.setattr(plan_agent.NetInstance, "from_env", classmethod(origin))
    monkeypatch.setattr(plan_trainer, "out_dir", lambda: str(tmp_path))
    for key, val in {"SPECTRA_PLAN_K": "4", "SPECTRA_PLAN_EXIT": "cem:2:2:0.6:8:1", f"SPECTRA_PLAN_{flag}": value,
                     **more}.items():
        monkeypatch.setenv(key, val)
    with pytest.raises(ValueError, match=f"cannot be combined with SPECTRA_PLAN_{flag}"):
        plan_trainer.run(None, SHARD)


def test_a_malformed_spec_stops_the_run_before_any_origin_is_read(monkeypatch, tmp_path, v10_env):
    def origin(*args, **kwargs):
        pytest.fail("an origin was read")

    _clear(monkeypatch)
    monkeypatch.setattr(plan_agent.NetInstance, "from_env", classmethod(origin))
    monkeypatch.setattr(plan_trainer, "out_dir", lambda: str(tmp_path))
    monkeypatch.setenv("SPECTRA_PLAN_EXIT", "cem:3:2:2:8:1")
    with pytest.raises(ValueError, match="shrink=2 is outside"):
        plan_trainer.run(None, SHARD)


def test_the_compatible_flags_are_accepted_beside_exit(monkeypatch):
    cfg = plan_trainer.config()
    monkeypatch.setenv("SPECTRA_PLAN_EXIT", SPEC)
    av2 = {"ema": 0.999, "lr_schedule": "cosine", "menu_bound": 4.0, "menu_entropy": 0.05, "kappa_points": 1,
           "sample": 1.0, "check": "1", "contract": True, "menu_kappa": {}, "menu_share_floor": 0.1}
    assert plan_trainer.exit_settings(cfg, None, av2)["spec"] == SPEC


# ------------------------------------------------------------------ 3. round 1 is the flag-off draw


@pytest.mark.parametrize("line", ["direct", "residual", "menu"])
def test_round_one_draws_what_the_flag_off_run_draws(monkeypatch, tmp_path, v10_env, line):
    draws = _spy(monkeypatch)
    per = 2 if line == "menu" else 1  # the scores, then on the menu line the logits
    _go(monkeypatch, tmp_path / "off", _menu_instance(), instances=2, **LINE_FLAGS[line])
    off = list(draws)
    del draws[:]
    lines, records, _blob_ = _go(monkeypatch, tmp_path / "on", _menu_instance(), instances=2,
                                 SPECTRA_PLAN_EXIT="cem:3:2:0.6:16:0", **LINE_FLAGS[line])
    assert len(off) == 2 * per and len(draws) == 2 * per * 3
    for a, b in zip(off[:per], draws[:per]):  # instance 0: the same draws, from the same mean at the same sigma
        assert a["k"] == b["k"] == 4 and a["sigma"] == b["sigma"] and torch.equal(a["mean"], b["mean"])
        assert torch.equal(a["z"], b["z"])
    assert " exit=cem:3:2:0.6:16:0 (12 evaluations per instance)" in lines[0]
    assert all(len(r["rewards"]) == 12 == len(r["kept"]) for r in _its(records))


# ------------------------------------------------------------------ 4. R x K evaluations per instance


@pytest.mark.parametrize("line", ["residual", "menu"])
def test_an_instance_costs_r_times_k_reward_evaluations(monkeypatch, tmp_path, v10_env, line):
    totals, records = {}, {}
    for name, flags in (("off", {}), ("on", {"SPECTRA_PLAN_EXIT": "cem:3:2:0.6:16:1"})):
        inst = _menu_instance()
        base, calls = inst.reward, [0]

        def counted(widths, proxy, batches, _base=base, _calls=calls):
            _calls[0] += 1
            return _base(widths, proxy, batches)

        inst.reward = counted
        _lines, recs, _b = _go(monkeypatch, tmp_path / name, inst, instances=5, **LINE_FLAGS[line], **flags)
        totals[name], records[name] = calls[0], _its(recs)
    assert [len(r["rewards"]) for r in records["off"]] == [4] * 5
    assert [len(r["rewards"]) for r in records["on"]] == [12] * 5 == [len(r["kept"]) for r in records["on"]]
    # the references, the it = 0 mean plan and the closing summary cost the same under both: the difference is the search
    assert totals["on"] - totals["off"] == 5 * (3 - 1) * 4


# ------------------------------------------------------------------ 5. the cross-entropy arithmetic


@pytest.mark.parametrize("line", ["direct", "menu"])
def test_each_round_centres_on_the_previous_rounds_elite_and_the_target_is_the_last_elite(monkeypatch, tmp_path, v10_env,
                                                                                         line):
    script = [[0.1, 0.9, 0.5, 0.7], [3.0, 1.0, 2.0, 0.0], [5.0, 5.0, 4.0, 6.0]]  # round-major, K = 4
    flat, step = [v for row in script for v in row], [0]
    inst = _menu_instance()
    real_state = inst.state_at

    def state_at(kappa, budget="params"):
        step[0] = 0  # every instance asks for its state before its plans: the script restarts there
        return real_state(kappa, budget)

    def reward(widths, proxy, batches):
        i, step[0] = step[0], step[0] + 1
        return flat[i] if i < len(flat) else 0.0

    inst.state_at, inst.reward = state_at, reward
    draws, calls, real = _spy(monkeypatch), [], plan_agent.log_prob

    def log_prob(z, mu, sigma):
        calls.append((z.detach().clone(), float(sigma)))
        return real(z, mu, sigma)

    monkeypatch.setattr(plan_agent, "log_prob", log_prob)
    lines, records, _b = _go(monkeypatch, tmp_path, inst, instances=1, SPECTRA_PLAN_EXIT="cem:3:2:0.5:8:0",
                             **LINE_FLAGS[line])
    per = 2 if line == "menu" else 1
    zs, us = draws[0::per], (draws[1::per] if line == "menu" else None)
    assert len(zs) == 3
    e1, e2, e3 = [1, 3], [0, 2], [3, 0]  # round 3's tie (5.0, 5.0) goes to the lower index, after the 6.0
    sigma = zs[0]["sigma"]
    assert zs[1]["sigma"] == pytest.approx(sigma * 0.5) and zs[2]["sigma"] == pytest.approx(sigma * 0.25)
    assert torch.allclose(zs[1]["mean"], zs[0]["z"][e1].mean(dim=0), atol=1e-6)
    assert torch.allclose(zs[2]["mean"], zs[1]["z"][e2].mean(dim=0), atol=1e-6)
    target = zs[2]["z"][e3].mean(dim=0)
    assert torch.allclose(calls[0][0].squeeze(0), target, atol=1e-6) and calls[0][1] == sigma  # scored at round 1's sigma
    if us is not None:
        assert us[0]["sigma"] == 2.0 and us[1]["sigma"] == pytest.approx(1.0) and us[2]["sigma"] == pytest.approx(0.5)
        assert torch.allclose(us[1]["mean"], us[0]["z"][e1].mean(dim=0), atol=1e-6)
        assert torch.allclose(us[2]["mean"], us[1]["z"][e2].mean(dim=0), atol=1e-6)
        assert torch.allclose(calls[1][0].squeeze(0), us[2]["z"][e3].mean(dim=0), atol=1e-6) and calls[1][1] == 2.0
    else:
        assert len(calls) == 1
    record = _its(records)[0]
    assert record["rewards"] == flat and len(record["kept"]) == 12
    assert record["exit"]["round_max"] == [0.9, 3.0, 6.0]
    assert record["exit"]["round_mean"] == pytest.approx([0.55, 1.5, 5.0])
    assert record["exit"]["target_r"] == pytest.approx(5.5) and record["exit"]["sigma_last"] == pytest.approx(sigma * 0.25)
    assert " | exit r1 max +0.90 r3 max +6.00 target +5.50" in [t for t in lines if t.startswith("[plan] it=")][0]


# ------------------------------------------------------------------ 6. the live gradient


@pytest.mark.parametrize("line", ["direct", "menu"])
def test_live_gradient_is_minus_the_targets_pull_on_the_mean_over_sigma_squared(monkeypatch, tmp_path, v10_env, line):
    """R = 1, m = K, n = 0: the target is the mean of the K draws and the loss is minus the log-density of it, so
    d loss / d mu = -(z* - mu) / sigma^2 / BATCH (BATCH 2 here), and the same on the menu logits at sigma_u."""
    draws, calls, real = _spy(monkeypatch), [], plan_agent.log_prob

    def log_prob(z, mu, sigma):
        mu.retain_grad()
        calls.append((z.detach().clone(), mu, float(sigma)))
        return real(z, mu, sigma)

    monkeypatch.setattr(plan_agent, "log_prob", log_prob)
    _go(monkeypatch, tmp_path, _menu_instance(), instances=1, SPECTRA_PLAN_EXIT="cem:1:4:1:8:0", **LINE_FLAGS[line])
    per = 2 if line == "menu" else 1
    assert len(calls) == per and len(draws) == per
    for (z, mu, sigma), draw in zip(calls, draws):
        assert torch.allclose(z.squeeze(0), draw["z"].mean(dim=0), atol=1e-6)
        want = -(z.squeeze(0) - mu.detach()) / sigma ** 2 / 2.0
        assert mu.grad is not None and torch.allclose(mu.grad, want, rtol=1e-4, atol=1e-6)
        assert float(mu.grad.abs().sum()) > 0


# ------------------------------------------------------------------ 7. the replay


def test_the_buffer_is_a_fifo_with_a_generator_of_its_own():
    buf = plan_trainer.ExitBuffer(3, 5)
    py, torch_state = random.getstate(), torch.get_rng_state()
    z = torch.ones(2, requires_grad=True)
    for i in range(5):
        buf.add(i, 0.5, "flops", z * i, None if i % 2 else z * 10)
    assert [e[0] for e in buf.entries] == [2, 3, 4] and not buf.entries[0][3].requires_grad
    assert buf.entries[0][3].device.type == "cpu"
    got = [e[0] for e in buf.draw(20)]
    ref = random.Random(5 * 7919 + 13)
    assert got == [[2, 3, 4][ref.randrange(3)] for _ in range(20)]  # uniform with replacement from its own seed
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), torch_state)  # no other generator moved
    other = plan_trainer.ExitBuffer(3, 99)
    other.load(buf.state())
    assert _equal(other.state(), buf.state()) and [e[0] for e in other.draw(7)] == [e[0] for e in buf.draw(7)]
    with pytest.raises(ValueError, match="holds 3 targets"):
        plan_trainer.ExitBuffer(4, 0).load(buf.state())


@pytest.mark.parametrize("line", ["residual", "menu"])
def test_replay_leaves_the_instances_the_kappas_and_round_one_alone_and_reproduces_under_the_seed(monkeypatch, tmp_path,
                                                                                                  v10_env, line):
    draws, runs = _spy(monkeypatch), {}
    per = 2 if line == "menu" else 1
    for name, n in (("n0", "0"), ("n4", "4"), ("n4b", "4")):
        del draws[:]
        _lines, records, blob = _go(monkeypatch, tmp_path / name, _menu_instance(), instances=6,
                                    SPECTRA_PLAN_EXIT=f"cem:2:2:0.6:8:{n}", **LINE_FLAGS[line])
        runs[name] = {"seq": [(r["net"], r["kappa"], r["budget"]) for r in _its(records)], "first": list(draws[:per]),
                      "steps": _steps(records), "weights": blob["state_dict"]}
    assert runs["n0"]["seq"] == runs["n4"]["seq"] and len(set(runs["n0"]["seq"])) > 1  # rng untouched: same kappas
    for a, b in zip(runs["n0"]["first"], runs["n4"]["first"]):  # the first instance's round-1 draws
        assert torch.equal(a["z"], b["z"]) and torch.equal(a["mean"], b["mean"])
    assert not _equal(runs["n0"]["weights"], runs["n4"]["weights"])  # the replay moved the policy
    assert runs["n4"]["steps"] == runs["n4b"]["steps"] and _equal(runs["n4"]["weights"], runs["n4b"]["weights"])


# ------------------------------------------------------------------ 8. resume


def _resume_knobs(line, **more):
    return {**LINE_FLAGS[line], "SPECTRA_PLAN_BUDGET": "flops", "SPECTRA_PLAN_PROXY": "cut", "SPECTRA_PLAN_STATE": "1",
            "SPECTRA_PLAN_EMA": "0.9", "SPECTRA_PLAN_LR_SCHEDULE": "cosine", "SPECTRA_PLAN_EXIT": "cem:2:2:0.6:3:2",
            **more}


@pytest.mark.parametrize("line", ["residual", "menu"])
def test_a_run_stopped_at_a_batch_boundary_and_resumed_is_the_uninterrupted_run(monkeypatch, tmp_path, v10_env, line):
    knobs = _resume_knobs(line)
    fresh = lambda: _menu_instance("synthetic")  # noqa: E731  (a new instance per job: a first use draws from the global generators)
    _clear(monkeypatch)
    _l, full_records = _job(monkeypatch, tmp_path / "full", fresh(), **knobs)
    _clear(monkeypatch)
    a_lines, a_records = _job(monkeypatch, tmp_path / "a", fresh(), SPECTRA_PLAN_MAX_MINUTES="1e-9", **knobs)
    assert "[plan] MAX_MINUTES=1e-09 reached after 2 instances" in a_lines
    state = tmp_path / "a" / "trainer_state.pt"
    kept = plan_agent.read_blob(str(state))
    assert kept["done"] == 2 and len(kept["av2"]["exit"]["entries"]) == 2 and kept["av2"]["ema"]["updates"] == 1
    assert kept["meta"]["exit"] == "cem:2:2:0.6:3:2"
    _clear(monkeypatch)
    b_lines, b_records = _job(monkeypatch, tmp_path / "b", fresh(), SPECTRA_PLAN_RESUME=str(state), **knobs)
    assert f"[plan] resumed from {state} at 2 instances" in b_lines
    assert _steps(a_records) + _steps(b_records) == _steps(full_records)
    for name in ("policy_latest.pt", "policy_ema_latest.pt", "policy_ema_it00006.pt"):
        assert _equal(_weights(tmp_path / "b" / name), _weights(tmp_path / "full" / name)), name
    sb = plan_agent.read_blob(str(tmp_path / "b" / "trainer_state.pt"))
    sf = plan_agent.read_blob(str(tmp_path / "full" / "trainer_state.pt"))
    assert _equal(sb["av2"], sf["av2"]) and _equal(sb["optimizer"], sf["optimizer"]) and _equal(sb["rng"], sf["rng"])
    assert len(sf["av2"]["exit"]["entries"]) == 3  # the buffer holds the last B = 3 of the 6 targets
    # the replay generator is the dedicated one: random.Random(SEED * 7919 + 13) after each step's n x BATCH draws
    sim = random.Random(0 * 7919 + 13)
    for step in (1, 2, 3):
        for _ in range(2 * 2):
            sim.randrange(min(3, 2 * step))
    assert sf["av2"]["exit"]["rng"] == sim.getstate()
    assert not _equal(_weights(tmp_path / "a" / "policy_latest.pt"), _weights(tmp_path / "full" / "policy_latest.pt"))


def test_a_mid_batch_state_and_a_closing_state_resume_with_their_buffers(monkeypatch, tmp_path, v10_env):
    knobs = _resume_knobs("residual", SPECTRA_PLAN_BATCH="4", SPECTRA_PLAN_SAVE_EVERY="3")
    fresh = lambda: _menu_instance("synthetic")  # noqa: E731
    _keep_states(monkeypatch)
    _clear(monkeypatch)
    _l, full_records = _job(monkeypatch, tmp_path / "full", fresh(), **knobs)
    full = _weights(tmp_path / "full" / "policy_latest.pt")
    sf = plan_agent.read_blob(str(tmp_path / "full" / "trainer_state.pt"))
    mid = tmp_path / "full" / "trainer_state_it00003.pt"
    kept = plan_agent.read_blob(str(mid))
    assert kept["done"] == 3 and len(kept["av2"]["exit"]["entries"]) == 3
    _clear(monkeypatch)
    b_lines, b_records = _job(monkeypatch, tmp_path / "b", fresh(), SPECTRA_PLAN_RESUME=str(mid), **knobs)
    assert f"[plan] resumed from {mid} at 3 instances, 3 of 4 into a batch" in b_lines
    assert _steps(b_records) == _steps(full_records)[3:]
    assert _equal(_weights(tmp_path / "b" / "policy_latest.pt"), full)
    assert _equal(plan_agent.read_blob(str(tmp_path / "b" / "trainer_state.pt"))["av2"], sf["av2"])
    closing = tmp_path / "full" / "trainer_state_it00006.pt"  # two into a batch: the closing step, replay included
    _clear(monkeypatch)
    c_lines, c_records = _job(monkeypatch, tmp_path / "c", fresh(), SPECTRA_PLAN_RESUME=str(closing), **knobs)
    assert _line_its(c_lines) == [] and _steps(c_records) == []
    assert _equal(_weights(tmp_path / "c" / "policy_latest.pt"), full)
    assert _equal(plan_agent.read_blob(str(tmp_path / "c" / "trainer_state.pt"))["av2"], sf["av2"])


def test_a_chained_job_under_another_spec_or_without_the_buffer_is_refused(monkeypatch, tmp_path, v10_env):
    knobs = _resume_knobs("residual")
    inst = _menu_instance("synthetic")
    _clear(monkeypatch)
    _job(monkeypatch, tmp_path / "a", inst, SPECTRA_PLAN_MAX_MINUTES="1e-9", **knobs)
    state = str(tmp_path / "a" / "trainer_state.pt")
    _clear(monkeypatch)
    with pytest.raises(ValueError, match=r"exit='cem:2:2:0.6:3:2' \(here 'cem:2:2:0.6:3:0'\)"):
        _job(monkeypatch, tmp_path / "c", inst, SPECTRA_PLAN_RESUME=state, **{**knobs, "SPECTRA_PLAN_EXIT": "cem:2:2:0.6:3:0"})
    _clear(monkeypatch)
    plain = {key: value for key, value in knobs.items() if key != "SPECTRA_PLAN_EXIT"}
    with pytest.raises(ValueError, match=r"exit='cem:2:2:0.6:3:2' \(here None\)"):
        _job(monkeypatch, tmp_path / "d", inst, SPECTRA_PLAN_RESUME=state, **plain)
    _clear(monkeypatch)
    _job(monkeypatch, tmp_path / "e", inst, SPECTRA_PLAN_MAX_MINUTES="1e-9", **plain)  # a state written without EXIT
    _clear(monkeypatch)
    with pytest.raises(ValueError, match=r"exit=None \(here 'cem:2:2:0.6:3:2'\)"):
        _job(monkeypatch, tmp_path / "f", inst, SPECTRA_PLAN_RESUME=str(tmp_path / "e" / "trainer_state.pt"), **knobs)


# ------------------------------------------------------------------ 9. the benchmark script


def _script(args, env_extra, timeout=1800):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SPECTRA_")}
    env.pop("PYTHONPATH", None)
    env.update(env_extra)
    done = subprocess.run([sys.executable, "-I", str(SCRIPT)] + [str(a) for a in args], env=env, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return done.returncode, done.stdout


needs_cache = pytest.mark.skipif(not (os.path.isfile(CACHE) and os.path.isfile(os.path.join(CALIB_DIR, "calib_residual.json"))),
                                 reason="the planted benchmark's cache and calibration live on the cluster")


@needs_cache
@pytest.mark.parametrize("config, evals, per, rounds", [("exit_cem3", 96, 24, 3), ("exit_cem1", 32, 8, 1),
                                                        ("ema_cosine", 32, 8, 1)])
def test_a_tiny_planted_run_records_r_times_k_evaluations_per_instance(tmp_path, config, evals, per, rounds):
    code, text = _script(["run", "--line", "residual", "--config", config, "--seed", 42, "--evals", evals, "--minutes",
                          30, "--tree", Path(SCRIPT).parents[1]],
                         {"EXIT_PLANT": str(tmp_path), "AV2_CACHE": CACHE, "EXIT_CALIB": CALIB_DIR})
    assert code == 0, text[-3000:]
    name = f"residual__{config}__s42__i{evals // per}.json"
    res = json.loads((tmp_path / name).read_text(encoding="utf-8"))
    assert res["evals_per_instance"] == per == res["k"] * rounds and res["evaluations"] == evals
    assert res["instances"] == evals // per and str(res["instances"]) in res["series"]["last"]
    assert "ema" in res["series"] and (" exit=" in res["trainer_line"]) == (config != "ema_cosine")
    if config == "ema_cosine":
        assert res["exit_gain"] is None
    else:
        gain = res["exit_gain"]
        assert gain["instances"] >= 1 and math.isfinite(gain["max_gain"]) and math.isfinite(gain["target_gain"])
        if rounds == 1:
            assert gain["max_gain"] == 0.0
    assert not os.path.exists(os.path.join(CALIB_DIR, name))  # nothing is written into section 475's folder


@needs_cache
def test_evals_that_do_not_divide_into_whole_instances_are_refused(tmp_path):
    code, text = _script(["run", "--line", "residual", "--config", "exit_cem3", "--seed", 42, "--evals", 100,
                          "--tree", Path(SCRIPT).parents[1]],
                         {"EXIT_PLANT": str(tmp_path), "AV2_CACHE": CACHE, "EXIT_CALIB": CALIB_DIR})
    assert code != 0 and "does not divide into whole instances" in text and not list(tmp_path.glob("*.json"))


def test_the_script_never_writes_into_section_475s_folder(tmp_path):
    code, text = _script(["table", "--evals", 1], {"EXIT_PLANT": CALIB_DIR})
    assert code != 0 and "never writes there" in text


def _result(line, config, seed, instances, per, d, which=("last", "ema"), gain=None):
    """A result file as the script writes it: ``d`` = (d / d0 at 25 %, 50 %, 100 %) of the 192,000 evaluations (d0 = 1);
    ``per`` None = an older file without ``evals_per_instance``."""
    keys = [str(round(instances * f)) for f in (0.25, 0.5, 1.0)]
    point = lambda v: {"d_in": v, "d_045": v, "d_020": v, "reward": -v, "share_gap": None}  # noqa: E731
    series = {w: {k: point(v) for k, v in zip(keys, d)} for w in which}
    out = {"line": line, "config": config, "seed": seed, "instances": instances, "k": 8, "noise": 1.0, "noise_sd": 0.5,
           "scale": 10.0, "flat": 0.0, "d0_in": 1.0, "d0_045": 1.0, "d0_020": 1.0, "series": series,
           "top_share_last200": None, "seconds": 600.0, "nets": ["a", "b"], "exit_gain": gain}
    if per is not None:
        out["evals_per_instance"], out["evaluations"] = per, instances * per
    return out


def _put(folder, res):
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{res['line']}__{res['config']}__s{res['seed']}__i{res['instances']}.json"
    (folder / name).write_text(json.dumps(res), encoding="utf-8")


def _table(tmp_path, exit_d, seeds=(42, 43, 44)):
    ref, out = tmp_path / "ref", tmp_path / "out"
    for s in (42, 43, 44):
        _put(ref, _result("residual", "ema_cosine", s, 24000, None, (0.8, 0.6, 0.5)))  # an older file: K per instance
    for s, d in zip(seeds, exit_d):
        _put(out, _result("residual", "exit_cem3", s, 8000, 24, (0.7, 0.55, d), gain={"instances": 2000, "max_gain": 0.25,
                                                                                    "target_gain": 0.5}))
    code, text = _script(["table", "--evals", 192000, "--ref", "residual=ema_cosine,menu=head_ema_cosine", "--ref-dir",
                          ref], {"EXIT_PLANT": str(out), "EXIT_CALIB": str(tmp_path / "none")})
    assert code == 0, text
    return text


def test_the_table_reads_every_run_at_equal_evaluations_and_makes_the_registered_call(tmp_path):
    text = _table(tmp_path / "go", (0.40, 0.40, 0.40))
    row = next(line for line in text.splitlines() if line.startswith("exit_cem3") and " ema " in line)
    assert [t for t in row.split() if not t.startswith("(")][3:6] == ["0.700", "0.550", "0.400"]  # keys 2000 / 4000 / 8000 of the 8000-instance runs
    ref_row = next(line for line in text.splitlines() if line.startswith("ema_cosine") and " ema " in line)
    assert [t for t in ref_row.split() if not t.startswith("(")][3:6] == ["0.800", "0.600", "0.500"]  # keys 6000 / 12000 / 24000 of the 24000-instance runs
    assert "0.800" in row and "3/3" in row and "+0.250 / +0.500" in row
    assert "CALL residual: EXIT-GO" in text and "closer on 3/3 seeds, paired mean ratio 0.800" in text
    assert "BAR residual: exit_cem3 d/d0 at 192000 evaluations" in text and "-> reached" in text
    assert (tmp_path / "go" / "out" / "table_E192000.txt").is_file()
    no_ratio = _table(tmp_path / "ratio", (0.46, 0.46, 0.46))  # closer on every seed, but the ratio 0.92 is above 0.85
    assert "CALL residual: EXIT-NO" in no_ratio and "closer on 3/3 seeds, paired mean ratio 0.920" in no_ratio
    one_seed = _table(tmp_path / "seed", (0.30, 0.30, 0.55))  # mean ratio 0.77, but one seed is further than the bundle
    assert "CALL residual: EXIT-NO" in one_seed and "closer on 2/3 seeds" in one_seed
    assert "-> not reached" in one_seed
    few = _table(tmp_path / "few", (0.40, 0.40), seeds=(42, 43))
    assert "CALL residual: INCOMPLETE" in few


def test_check_prints_repro_exact_or_repro_diff_and_exits_0_or_4(tmp_path):
    ref, out = tmp_path / "ref", tmp_path / "out"
    base = _result("residual", "ema_cosine", 42, 24000, None, (0.8, 0.6, 0.5))
    env = {"EXIT_PLANT": str(out), "EXIT_CALIB": str(tmp_path / "none")}
    code, text = _script(["check", "--ref-dir", ref], env)
    assert code == 5 and "REPRO-PENDING" in text  # nothing run yet
    _put(ref, base)
    _put(out, _result("residual", "ema_cosine", 42, 24000, None, (0.8, 0.6, 0.5 + 5e-10)))
    code, text = _script(["check", "--ref-dir", ref], env)
    assert code == 0 and "REPRO-EXACT" in text
    _put(out, _result("residual", "ema_cosine", 42, 24000, None, (0.8, 0.6, 0.5 + 1e-6)))
    code, text = _script(["check", "--ref-dir", ref], env)
    assert code == 4 and "REPRO-DIFF" in text and "series ema at 24000" in text and "series last at 24000" in text
