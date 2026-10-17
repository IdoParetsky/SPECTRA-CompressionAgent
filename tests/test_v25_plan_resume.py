"""
v25 (default off): the plan trainer stops and resumes across jobs. ``SPECTRA_PLAN_STATE=1`` writes ``trainer_state.pt``
beside every ``policy_latest.pt`` (the policy and optimizer state, the partial batch's gradients, the completed count,
every RNG state, the checkpoint meta with the config, the chain and a format version), and with it on MAX_MINUTES stops
only at a batch boundary. ``SPECTRA_PLAN_RESUME=<trainer_state.pt>`` continues a run from the state under the same
settings (``RESUME_FREE`` = MAX_MINUTES and SAVE_EVERY may differ): a run stopped after M instances and resumed to N
ends bitwise where one uninterrupted run of N does. Nothing changes unless one of the two is set.

CPU only, no datasets.  python -m pytest tests/test_v25_plan_resume.py -v
"""

import json
import os
import random
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import V10_FLAGS, _instance  # noqa: E402
from src import plan_agent, plan_trainer  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

BASE = {"SPECTRA_PLAN_INSTANCES": "6", "SPECTRA_PLAN_K": "4", "SPECTRA_PLAN_BATCH": "2", "SPECTRA_PLAN_PROXY": "bn2",
        "SPECTRA_PLAN_REF_EVERY": "3", "SPECTRA_PLAN_SAVE_EVERY": "2", "SPECTRA_PLAN_SUMMARY_KAPPAS": "0.6",
        "SPECTRA_PLAN_NETS": "toy", "SPECTRA_PLAN_LR": "0.01"}
PER_JOB = ("SPECTRA_PLAN_STATE", "SPECTRA_PLAN_RESUME", "SPECTRA_PLAN_MAX_MINUTES", "SPECTRA_PLAN_INIT")  # INIT too: a job without it must not inherit the last one's
POLICIES = ["policy_it00002.pt", "policy_it00004.pt", "policy_it00006.pt", "policy_latest.pt"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("SPECTRA_PLAN_", "SPECTRA_ALLOC_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(recorder, "record", lambda *a, **k: None)
    yield


@pytest.fixture
def v10_env(monkeypatch):
    for key, value in V10_FLAGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SPECTRA_STATE_TOKENS", raising=False)


@pytest.fixture
def inst(v10_env):
    return _instance()


# ------------------------------------------------------------------ helpers


def _prepare(monkeypatch, folder, inst, **knobs):
    """The trainer on ``inst``, writing to ``folder``, under BASE plus ``knobs`` (the per-job flags unset unless given);
    returns the list the log lines go to. ``plan_trainer.run`` is the caller's. The global generators are seeded as a
    fresh job's are (the trainer seeds only its own ``rng`` and torch's), so every job here starts from one state."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    monkeypatch.setattr(plan_agent.NetInstance, "from_env", classmethod(lambda cls, env, p, m, l, **kw: inst))
    monkeypatch.setattr(plan_trainer, "out_dir", lambda: str(folder))
    for key in PER_JOB:
        monkeypatch.delenv(key, raising=False)
    for key, value in {**BASE, **knobs}.items():
        monkeypatch.setenv(key, str(value))
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    return lines


def _run(monkeypatch, folder, inst, **knobs):
    """One job: ``(lines, records)`` of a ``_prepare``d run."""
    lines = _prepare(monkeypatch, folder, inst, **knobs)
    plan_trainer.run(None, [("toy.pt", (None, None))])
    with open(Path(folder) / "plan_train.jsonl", encoding="utf-8") as fh:
        return lines, [json.loads(line) for line in fh]


def _steps(records):
    """The per-instance records without their wall-clock seconds."""
    return [{k: v for k, v in r.items() if k != "seconds"} for r in records if "it" in r]


def _its(lines):
    """The ``it=N/total`` tokens of the step lines, in order."""
    return [line.split(" ")[1] for line in lines if line.startswith("[plan] it=")]


def _step_lines(lines):
    """The step lines without their trailing seconds."""
    return [line.rsplit(" | ", 1)[0] for line in lines if line.startswith("[plan] it=")]


def _weights(path):
    return plan_agent.read_blob(str(path))["state_dict"]


def _equal(a, b):
    """Bitwise equality of nested dicts / lists / tuples of tensors, arrays and plain values."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _synthetic_reward(inst):
    """A reward the K plans never tie on (the bn2 proxy may, on 64 random samples: a tie is a zero advantage and no
    learning), that also draws from the three global RNGs: every instance then moves the policy, and a resume that put
    any generator back wrongly changes the result."""
    rows, widths0 = inst.pm.rows, inst.pm.widths0

    def reward(widths, proxy, batches):
        kept = sum((k + 1) * widths[row] / widths0[row] for k, row in enumerate(rows)) / len(rows)
        return 10.0 * kept + float(torch.rand(())) + random.random() + float(np.random.rand())

    return reward


def _keep_states(monkeypatch):
    """Keep a copy of every trainer_state.pt as trainer_state_it<done>.pt (the trainer keeps only the newest)."""
    write = plan_trainer.write_state

    def keep(path, state):
        write(path, state)
        shutil.copyfile(path, os.path.join(os.path.dirname(path), f"trainer_state_it{state['done']:05d}.pt"))

    monkeypatch.setattr(plan_trainer, "write_state", keep)


def _checkpoint(folder, seed=1):
    """A plan-agent checkpoint with random head weights, for ``SPECTRA_PLAN_INIT``."""
    torch.manual_seed(seed)
    policy = plan_agent.PlanPolicy(63)
    with torch.no_grad():
        policy.head.weight.normal_()
    path = Path(folder) / "init" / "policy_latest.pt"
    plan_agent.save_policy(policy, str(path), tokens="layer", zero=[], k_min=0.1, budget="params", proxy="bn8",
                           nets=["a.pt"])
    return str(path)


# ------------------------------------------------------------------ (a) the flags unset


def test_flags_unset_write_no_state_and_run_as_before(monkeypatch, tmp_path, v10_env, inst):
    lines, records = _run(monkeypatch, tmp_path / "off", inst)
    assert not (tmp_path / "off" / "trainer_state.pt").exists()
    assert not any(line.startswith(("[plan] state ", "[plan] resumed ")) for line in lines)
    head = [line for line in lines if line.startswith("[plan] trainer ")]
    assert len(head) == 1 and " state=" not in head[0] and " resume=" not in head[0]
    assert _its(lines) == [f"it={i}/6" for i in range(1, 7)]
    assert any(line.startswith("[plan] DONE 6 instances") for line in lines)
    assert sorted(os.listdir(tmp_path / "off")) == ["plan_train.jsonl"] + POLICIES
    blob = plan_agent.read_blob(str(tmp_path / "off" / "policy_latest.pt"))
    assert not any(key in blob for key in ("state", "resume", "chain", "done", "grads"))
    # MAX_MINUTES with the flag off stops after the first instance, into a batch, as before
    stop_lines, stop_records = _run(monkeypatch, tmp_path / "stop", inst, SPECTRA_PLAN_MAX_MINUTES="1e-9")
    assert "[plan] MAX_MINUTES=1e-09 reached after 1 instances" in stop_lines
    assert _its(stop_lines) == ["it=1/6"] and len(_steps(stop_records)) == 1
    assert sorted(os.listdir(tmp_path / "stop")) == ["plan_train.jsonl", "policy_it00001.pt", "policy_latest.pt"]
    # the flag on leaves the trajectory as it is: the same step lines, records and weights, one state file more
    on_lines, on_records = _run(monkeypatch, tmp_path / "on", inst, SPECTRA_PLAN_STATE="1")
    assert _steps(on_records) == _steps(records) and _step_lines(on_lines) == _step_lines(lines)
    assert _equal(_weights(tmp_path / "on" / "policy_latest.pt"), _weights(tmp_path / "off" / "policy_latest.pt"))
    assert sorted(os.listdir(tmp_path / "on")) == ["plan_train.jsonl"] + POLICIES + ["trainer_state.pt"]


# ------------------------------------------------------------------ (b) what the state holds


def test_state_holds_the_documented_keys_and_the_trajectory_before_the_closing_step(monkeypatch, tmp_path, v10_env,
                                                                                    inst):
    inst.reward = _synthetic_reward(inst)
    _keep_states(monkeypatch)
    lines, records = _run(monkeypatch, tmp_path, inst, SPECTRA_PLAN_INSTANCES="5", SPECTRA_PLAN_STATE="1")
    head = [line for line in lines if line.startswith("[plan] trainer ")][0]
    assert head.endswith(" state=1") and " resume=" not in head
    path = tmp_path / "trainer_state.pt"
    state = plan_agent.read_blob(str(path))
    assert set(state) == set(plan_trainer.STATE_KEYS) and state["format"] == plan_trainer.STATE_FORMAT == 1
    assert state["done"] == 5 and state["chain"] == [] and state["saved"] > 0
    cfg = plan_trainer.config()
    assert state["meta"]["config"] == asdict(cfg) and state["meta"]["nets"] == ["toy.pt"]
    assert state["meta"]["proxy"] == "bn2" and state["meta"]["tokens"] == "layer"
    assert set(state["rng"]) == {"trainer", "python", "numpy", "torch", "cuda"}
    assert state["rng"]["trainer"][0] == 3 and len(state["rng"]["trainer"][1]) == 625  # random.Random.getstate()
    assert state["rng"]["python"][0] == 3 and state["rng"]["numpy"][0] == "MT19937"
    assert torch.is_tensor(state["rng"]["torch"]) and state["rng"]["torch"].dtype == torch.uint8
    assert (state["rng"]["cuda"] is None) == (not torch.cuda.is_available())
    policy = plan_agent.PlanPolicy(inst.feature_dim)
    policy.load_state_dict(state["policy"])  # strict: every key and shape of the trainer's policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    optimizer.load_state_dict(state["optimizer"])
    params = list(policy.parameters())
    assert 0 < len(optimizer.state) <= len(params)  # a parameter the forward never uses has none
    # 5 instances in batches of 2: the states at 2 and 4 sit on a boundary (no gradient in flight) and their policy is
    # the checkpoint's; the closing one, one instance into a batch, carries that instance's gradients
    for done in (2, 4):
        kept = plan_agent.read_blob(str(tmp_path / f"trainer_state_it{done:05d}.pt"))
        assert kept["done"] == done and all(g is None or not bool(g.any()) for g in kept["grads"])
        assert _equal(kept["policy"], _weights(tmp_path / f"policy_it{done:05d}.pt"))
    assert len(state["grads"]) == len(params)
    assert all(g is None or (torch.is_tensor(g) and g.shape == p.shape) for g, p in zip(state["grads"], params))
    assert any(torch.is_tensor(g) and float(g.abs().sum()) > 0 for g in state["grads"])
    # the closing state is the trajectory after 5 instances, before the partial step: its policy is the one stepped at
    # 4, while policy_it00005.pt and policy_latest.pt carry that step
    assert _equal(state["policy"], _weights(tmp_path / "policy_it00004.pt"))
    assert _equal(_weights(tmp_path / "policy_it00005.pt"), _weights(tmp_path / "policy_latest.pt"))
    assert not _equal(state["policy"], _weights(tmp_path / "policy_latest.pt"))
    assert [line for line in lines if line.startswith("[plan] state ")] == [
        f"[plan] state {path} at 2 instances", f"[plan] state {path} at 4 instances",
        f"[plan] state {path} at 5 instances, 1 of 2 into a batch"]
    saved = next(line for line in lines if line.startswith("[plan] saved "))
    assert lines.index(f"[plan] state {path} at 5 instances, 1 of 2 into a batch") < lines.index(saved)
    assert not (tmp_path / "trainer_state.pt.tmp").exists()


def test_write_state_replaces_the_file_atomically_and_read_state_checks_the_format(tmp_path):
    path = str(tmp_path / "trainer_state.pt")
    plan_trainer.write_state(path, {"format": plan_trainer.STATE_FORMAT, "done": 3})
    assert os.listdir(tmp_path) == ["trainer_state.pt"] and plan_trainer.read_state(path)["done"] == 3
    plan_trainer.write_state(path, {"format": plan_trainer.STATE_FORMAT, "done": 4})
    assert os.listdir(tmp_path) == ["trainer_state.pt"] and plan_trainer.read_state(path)["done"] == 4
    torch.save({"format": 0, "done": 4}, path)
    with pytest.raises(ValueError, match="format"):
        plan_trainer.read_state(path)
    plan_agent.save_policy(plan_agent.PlanPolicy(63), path, tokens="layer")  # a policy checkpoint is not a state
    with pytest.raises(ValueError, match="not a trainer state"):
        plan_trainer.read_state(path)


def _draws(rng):
    """One draw from each generator the state carries: the trainer's ``rng``, random, numpy and torch."""
    return rng.random(), random.random(), float(np.random.rand()), float(torch.rand(()))


def test_rng_states_round_trip_every_generator():
    rng = random.Random(7)
    _draws(rng)
    states = plan_trainer.rng_states(rng)
    want = _draws(rng)
    _draws(rng)
    plan_trainer.set_rng_states(states, rng)
    assert _draws(rng) == want
    again = plan_trainer.rng_states(rng)
    assert _equal(again["trainer"], rng.getstate()) and _equal(again["torch"], torch.get_rng_state())
    assert (states["cuda"] is None) == (not torch.cuda.is_available())


# ------------------------------------------------------------------ (c) exactness


@pytest.mark.parametrize("reward", ["synthetic", "bn2"])
def test_a_run_stopped_and_resumed_ends_where_the_uninterrupted_run_does(monkeypatch, tmp_path, v10_env, inst, reward):
    if reward == "synthetic":
        inst.reward = _synthetic_reward(inst)
    full_lines, full_records = _run(monkeypatch, tmp_path / "full", inst, SPECTRA_PLAN_STATE="1")
    a_lines, a_records = _run(monkeypatch, tmp_path / "a", inst, SPECTRA_PLAN_STATE="1",
                              SPECTRA_PLAN_MAX_MINUTES="1e-9")
    # with the state on, MAX_MINUTES stops at the batch boundary (2), not after the first instance
    assert "[plan] MAX_MINUTES=1e-09 reached after 2 instances" in a_lines
    assert _its(a_lines) == ["it=1/6", "it=2/6"] and len(_steps(a_records)) == 2
    assert sorted(os.listdir(tmp_path / "a")) == ["plan_train.jsonl", "policy_it00002.pt", "policy_latest.pt",
                                                  "trainer_state.pt"]
    state = tmp_path / "a" / "trainer_state.pt"
    kept = plan_agent.read_blob(str(state))
    assert kept["done"] == 2 and all(g is None or not bool(g.any()) for g in kept["grads"])
    assert _equal(kept["policy"], _weights(tmp_path / "a" / "policy_latest.pt"))  # no partial step was flushed
    b_lines, b_records = _run(monkeypatch, tmp_path / "b", inst, SPECTRA_PLAN_STATE="1",
                              SPECTRA_PLAN_RESUME=str(state))
    assert f"[plan] resumed from {state} at 2 instances" in b_lines
    assert _steps(a_records) + _steps(b_records) == _steps(full_records)
    assert _step_lines(a_lines) + _step_lines(b_lines) == _step_lines(full_lines)
    full = _weights(tmp_path / "full" / "policy_latest.pt")
    assert _equal(_weights(tmp_path / "b" / "policy_latest.pt"), full)
    assert _equal(_weights(tmp_path / "b" / "policy_it00006.pt"), full)
    if reward == "synthetic":  # the chain did move on from the stop
        assert not _equal(_weights(tmp_path / "a" / "policy_latest.pt"), full)
        assert any(float(v.abs().sum()) > 0 for v in full.values())
    # the closing states agree too: the optimizer, the gradients in flight (none at 6) and every generator
    sb = plan_agent.read_blob(str(tmp_path / "b" / "trainer_state.pt"))
    sf = plan_agent.read_blob(str(tmp_path / "full" / "trainer_state.pt"))
    assert sb["done"] == sf["done"] == 6 and sb["chain"] == [str(state)] and sf["chain"] == []
    assert _equal(sb["policy"], sf["policy"]) and _equal(sb["optimizer"], sf["optimizer"])
    assert _equal(sb["grads"], sf["grads"]) and _equal(sb["rng"], sf["rng"])
    assert _equal(sb["meta"], sf["meta"])


def test_a_mid_batch_state_resumes_with_its_gradients_and_a_closing_state_re_finishes(monkeypatch, tmp_path, v10_env,
                                                                                      inst):
    inst.reward = _synthetic_reward(inst)
    _keep_states(monkeypatch)
    knobs = {"SPECTRA_PLAN_STATE": "1", "SPECTRA_PLAN_BATCH": "4", "SPECTRA_PLAN_SAVE_EVERY": "3"}
    full_lines, full_records = _run(monkeypatch, tmp_path / "full", inst, **knobs)
    assert [line for line in full_lines if line.startswith("[plan] state ")] == [
        f"[plan] state {tmp_path / 'full' / 'trainer_state.pt'} at 3 instances, 3 of 4 into a batch",
        f"[plan] state {tmp_path / 'full' / 'trainer_state.pt'} at 6 instances, 2 of 4 into a batch"]
    state = tmp_path / "full" / "trainer_state_it00003.pt"
    kept = plan_agent.read_blob(str(state))
    assert kept["done"] == 3 and any(torch.is_tensor(g) and float(g.abs().sum()) > 0 for g in kept["grads"])
    b_lines, b_records = _run(monkeypatch, tmp_path / "b", inst, SPECTRA_PLAN_RESUME=str(state), **knobs)
    assert f"[plan] resumed from {state} at 3 instances, 3 of 4 into a batch" in b_lines
    assert _its(b_lines) == ["it=4/6", "it=5/6", "it=6/6"]
    assert _steps(b_records) == _steps(full_records)[3:]
    # the step at instance 4 clipped the stored gradients plus its own: the same norm as the uninterrupted run's
    assert _steps(b_records)[0]["grad"] is not None and _steps(b_records)[0]["grad"] == _steps(full_records)[3]["grad"]
    full = _weights(tmp_path / "full" / "policy_latest.pt")
    assert _equal(_weights(tmp_path / "b" / "policy_latest.pt"), full)
    assert _equal(plan_agent.read_blob(str(tmp_path / "b" / "trainer_state.pt"))["policy"],
                  plan_agent.read_blob(str(tmp_path / "full" / "trainer_state.pt"))["policy"])
    # resuming from the closing state (6 of 6, two instances into a batch) runs no instance and flushes the same step
    closing = tmp_path / "full" / "trainer_state_it00006.pt"
    c_lines, c_records = _run(monkeypatch, tmp_path / "c", inst, SPECTRA_PLAN_RESUME=str(closing), **knobs)
    assert f"[plan] resumed from {closing} at 6 instances, 2 of 4 into a batch" in c_lines
    assert _its(c_lines) == [] and _steps(c_records) == []
    assert any(line.startswith("[plan] DONE 6 instances") for line in c_lines)
    assert _equal(_weights(tmp_path / "c" / "policy_latest.pt"), full)
    assert _equal(_weights(tmp_path / "c" / "policy_it00006.pt"), full)


# ------------------------------------------------------------------ (d) the settings must match


@pytest.mark.parametrize("knobs, key", [
    ({"SPECTRA_PLAN_SIGMA": "0.4"}, "config.sigma=0.5 (here 0.4)"),
    ({"SPECTRA_PLAN_INSTANCES": "8"}, "config.instances=6 (here 8)"),
    ({"SPECTRA_PLAN_K": "3"}, "config.k=4 (here 3)"),
    ({"SPECTRA_PLAN_LR": "0.02"}, "config.lr=0.01 (here 0.02)"),
    ({"SPECTRA_PLAN_REF_EVERY": "2"}, "config.ref_every=3 (here 2)"),
    ({"SPECTRA_PLAN_MIN_WIDTH": "2"}, "min_width=None (here 2)"),
    ({"SPECTRA_PLAN_PROXY": "cut"}, "proxy='bn2' (here 'cut')"),
    ({"SPECTRA_PLAN_ZERO": "sens"}, "zero=[] (here ['sens'])"),
    ({"SPECTRA_PLAN_RESIDUAL": "sens"}, "residual=None (here 'sens')"),
])
def test_resume_refuses_a_state_from_other_settings(monkeypatch, tmp_path, v10_env, inst, knobs, key):
    _run(monkeypatch, tmp_path / "a", inst, SPECTRA_PLAN_STATE="1", SPECTRA_PLAN_MAX_MINUTES="1e-9")
    state = str(tmp_path / "a" / "trainer_state.pt")
    lines = _prepare(monkeypatch, tmp_path / "b", inst, SPECTRA_PLAN_RESUME=state, **knobs)
    with pytest.raises(ValueError, match=re.escape(key)) as err:
        plan_trainer.run(None, [("toy.pt", (None, None))])
    assert str(err.value).startswith(f"SPECTRA_PLAN_RESUME={state}: ") and "max_minutes, save_every" in str(err.value)
    assert sum(line.startswith("[plan] net toy.pt") for line in lines) == 1  # refused after the origins, before a step
    assert not any(line.startswith(("[plan] resumed ", "[plan] it=", "[plan] state ")) for line in lines)
    assert os.listdir(tmp_path / "b") == []


def test_resume_accepts_the_free_fields_and_needs_the_same_init(monkeypatch, tmp_path, v10_env, inst):
    full_lines, full_records = _run(monkeypatch, tmp_path / "full", inst)
    _run(monkeypatch, tmp_path / "a", inst, SPECTRA_PLAN_STATE="1", SPECTRA_PLAN_MAX_MINUTES="1e-9")
    state = str(tmp_path / "a" / "trainer_state.pt")
    b_lines, b_records = _run(monkeypatch, tmp_path / "b", inst, SPECTRA_PLAN_RESUME=state,
                              SPECTRA_PLAN_MAX_MINUTES="5", SPECTRA_PLAN_SAVE_EVERY="3")
    assert f"[plan] resumed from {state} at 2 instances" in b_lines
    head = [line for line in b_lines if line.startswith("[plan] trainer ")][0]
    assert '"max_minutes": 5.0' in head and '"save_every": 3' in head and head.endswith(f" resume={state}")
    assert " state=" not in head and not any(line.startswith("[plan] state ") for line in b_lines)  # STATE unset
    assert _its(b_lines) == ["it=3/6", "it=4/6", "it=5/6", "it=6/6"]
    assert sorted(os.listdir(tmp_path / "b")) == ["plan_train.jsonl", "policy_it00003.pt", "policy_it00006.pt",
                                                  "policy_latest.pt"]
    assert _equal(_weights(tmp_path / "b" / "policy_latest.pt"), _weights(tmp_path / "full" / "policy_latest.pt"))
    # a warm start is part of the settings: the chain keeps it, and a job without it is refused
    init = _checkpoint(tmp_path)
    _run(monkeypatch, tmp_path / "i", inst, SPECTRA_PLAN_STATE="1", SPECTRA_PLAN_MAX_MINUTES="1e-9",
         SPECTRA_PLAN_INIT=init)
    istate = str(tmp_path / "i" / "trainer_state.pt")
    assert plan_agent.read_blob(istate)["meta"]["init"] == init
    j_lines, _records = _run(monkeypatch, tmp_path / "j", inst, SPECTRA_PLAN_RESUME=istate, SPECTRA_PLAN_INIT=init)
    assert any(line.startswith(f"[plan] init {init}: policy weights loaded") for line in j_lines)
    assert f"[plan] resumed from {istate} at 2 instances" in j_lines
    assert plan_agent.read_blob(str(tmp_path / "j" / "policy_latest.pt"))["init"] == init
    _prepare(monkeypatch, tmp_path / "k", inst, SPECTRA_PLAN_RESUME=istate)
    with pytest.raises(ValueError, match=re.escape(f"init={init!r} (here None)")):
        plan_trainer.run(None, [("toy.pt", (None, None))])


def test_resume_fails_fast_on_a_missing_or_foreign_file(monkeypatch, tmp_path, v10_env, inst):
    missing = str(tmp_path / "nowhere" / "trainer_state.pt")
    lines = _prepare(monkeypatch, tmp_path / "b", inst, SPECTRA_PLAN_RESUME=missing)
    with pytest.raises(FileNotFoundError):
        plan_trainer.run(None, [("toy.pt", (None, None))])
    assert lines == []  # before the config line and the origin reads
    policy = str(tmp_path / "policy_latest.pt")
    plan_agent.save_policy(plan_agent.PlanPolicy(63), policy, tokens="layer")
    lines = _prepare(monkeypatch, tmp_path / "c", inst, SPECTRA_PLAN_RESUME=policy)
    with pytest.raises(ValueError, match="not a trainer state"):
        plan_trainer.run(None, [("toy.pt", (None, None))])
    assert lines == []


# ------------------------------------------------------------------ (e) numbering and the DONE line


def test_the_resumed_job_continues_the_numbering_and_reports_the_total(monkeypatch, tmp_path, v10_env, inst):
    _run(monkeypatch, tmp_path / "a", inst, SPECTRA_PLAN_STATE="1", SPECTRA_PLAN_MAX_MINUTES="1e-9")
    state = str(tmp_path / "a" / "trainer_state.pt")
    lines, records = _run(monkeypatch, tmp_path / "b", inst, SPECTRA_PLAN_STATE="1", SPECTRA_PLAN_RESUME=state)
    head = [line for line in lines if line.startswith("[plan] trainer ")]
    assert len(head) == 1 and head[0].endswith(f" state=1 resume={state}")
    assert sum(line.startswith("[plan] net toy.pt") for line in lines) == 1  # the origins are read again
    resumed = f"[plan] resumed from {state} at 2 instances"
    assert lines.count(resumed) == 1
    its = [line for line in lines if line.startswith("[plan] it=")]
    assert _its(lines) == ["it=3/6", "it=4/6", "it=5/6", "it=6/6"]
    assert all(line.startswith(f"[plan] it={i}/6 toy.pt k=") for i, line in zip(range(3, 7), its))
    assert lines.index(next(line for line in lines if line.startswith("[plan] check toy.pt"))) < lines.index(resumed)
    assert lines.index(resumed) < lines.index(its[0])
    assert [r["it"] for r in records if "it" in r] == [2, 3, 4, 5]  # 0-based in the records, as before
    assert "mean_plan" in _steps(records)[0] and "refs" in _steps(records)[-1]  # ref_every 3: at it=2 and it=5
    assert not any("MAX_MINUTES" in line for line in lines)
    assert [line for line in lines if line.startswith("[plan] state ")] == [
        f"[plan] state {tmp_path / 'b' / 'trainer_state.pt'} at 4 instances",
        f"[plan] state {tmp_path / 'b' / 'trainer_state.pt'} at 6 instances"]
    saved = [line for line in lines if line.startswith("[plan] saved ")]
    assert len(saved) == 1 and saved[0].startswith(f"[plan] saved {tmp_path / 'b' / 'policy_it00006.pt'} after 6 "
                                                   "instances, ")
    done = [line for line in lines if line.startswith("[plan] DONE ")]
    assert len(done) == 1 and re.match(r"^\[plan\] DONE 6 instances in \d+\.\d min$", done[0])
    assert sorted(os.listdir(tmp_path / "b")) == ["plan_train.jsonl", "policy_it00004.pt", "policy_it00006.pt",
                                                  "policy_latest.pt", "trainer_state.pt"]
    assert len([r for r in records if "summary" in r]) == 1
    blob = plan_agent.read_blob(str(tmp_path / "b" / "policy_latest.pt"))
    assert blob["nets"] == ["toy.pt"] and blob["config"]["instances"] == 6 and "chain" not in blob
