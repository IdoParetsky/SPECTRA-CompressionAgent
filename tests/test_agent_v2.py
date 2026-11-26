"""
AGENT-V2 (queue row AGENT-V2, 26 Nov 2026; ``docs/paper/AGENT_AUDIT_26NOV.md``): the plan trainer's upgrades, each behind
a new default-off flag (the block in ``src/fortify.py`` lists them).

* With every flag unset the trainer is tree_v41's, to the bit: its loop is checked against a copy of tree_v41's own
  (``_run_v41``), and, when ``SPECTRA_AV2_REF_TREE`` names an untouched tree_v41, the same jobs are run from that tree
  and from this one in two processes and every output file compared.
* Each flag's own behaviour: the Δ cross-entropy rewards, rank / global / floor advantages (the rank shaping's
  invariance to the reward's scale), mirrored sampling (the pairs cancel the mean noise), the bounded menu logits
  (within B), the entropy bonus, the budget bins, split credit (separate advantages), averaged weights with cosine
  decay, sqrt(G) normalisation, the budget mixture, family sampling, the fine-tune's divergence guard, the
  masked-vs-real gate and the state contract (a mismatch is refused).

CPU only, no datasets.  python -m pytest tests/test_agent_v2.py -v
"""

import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_pruning import _init_static_conf  # noqa: E402

_init_static_conf()

from tests.test_plan_agent import LAYOUT, SHAPE, V10_FLAGS, toy_state  # noqa: E402
from tests.test_plan_agent import _instance as _toy_instance  # noqa: E402
from tests.test_plan_menu import (MENU, MENU_FLAG, MENU_META, RESIDUAL_META, T1, _hot_policy, _instance,  # noqa: E402
                                  _origin, _reward, _save, _seed_all, _train)
from tests.test_v25_plan_resume import _equal, _prepare, _steps, _synthetic_reward, _weights  # noqa: E402
from tests.test_v25_plan_resume import _run as _job  # noqa: E402
from src import alloc_walk, fortify, plan_agent, plan_trainer, recovery_edits  # noqa: E402
import src.group_sensitivity as group_sensitivity  # noqa: E402
import src.run_recorder as recorder  # noqa: E402
import src.utils as utils  # noqa: E402

FLAG_PREFIXES = ("SPECTRA_PLAN_", "SPECTRA_ALLOC_", "SPECTRA_FIXED_TARGET", "SPECTRA_EVAL_SIZE_", "SPECTRA_MIN_WIDTH",
                 "SPECTRA_STEM_ROWS", "SPECTRA_SENS_BN_RECAL", "SPECTRA_PROBE_BATCHES")
MENU_LINE = {**T1, "SPECTRA_PLAN_MENU": MENU_FLAG, "SPECTRA_PLAN_MENU_SIGMA": "2"}  # the rule-blending line's flags
FROM_ENV = plan_agent.NetInstance.__dict__["from_env"]  # the real reader: the trainer jobs below replace it
SHARD = [("toy.pt", (None, None))]
AV2_FLAGS = ("ADV", "MIRROR", "SPLIT", "GNORM", "EMA", "LR_SCHEDULE", "MENU_BOUND", "MENU_ENT", "MENU_KAPPA",
             "MENU_SIGMA_ANNEAL", "MENU_SHARE_FLOOR", "MENU_LOG", "KAPPA_POINTS", "SAMPLE", "FS_GUARD", "CHECK",
             "CONTRACT")


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


# ------------------------------------------------------------------ helpers


def _clear(monkeypatch):
    """Unset every trainer flag a previous job of the same test set."""
    for key in list(os.environ):
        if key.startswith("SPECTRA_PLAN_"):
            monkeypatch.delenv(key, raising=False)


def _go(monkeypatch, folder, inst, **flags):
    """One trainer job on ``inst`` under ``flags`` alone (K = 4 unless given): ``(lines, records, blob)``."""
    _clear(monkeypatch)
    _seed_all()
    return _train(monkeypatch, folder, inst, **{"SPECTRA_PLAN_K": "4", **flags})


def _its(records):
    return [r for r in records if "it" in r]


def _spy(monkeypatch):
    """Every ``plan_agent.sample_scores`` call of the trainer, in order, with what it was asked for and what it drew."""
    draws, real = [], plan_agent.sample_scores

    def draw(mean, sigma, k, generator=None, **kw):
        out = real(mean, sigma, k, generator, **kw)
        draws.append({"mean": mean.detach().clone(), "sigma": float(sigma), "k": int(k), "z": out.clone(), **kw})
        return out

    monkeypatch.setattr(plan_agent, "sample_scores", draw)
    return draws


def _menu_instance(reward="plain"):
    inst = _instance()
    inst.reward = _reward(inst) if reward == "plain" else _synthetic_reward(inst)
    return inst


# ------------------------------------------------------------------ the flags


def test_every_flag_is_off_by_default_and_parses_strictly(monkeypatch):
    assert fortify.plan_adv() == ("off", None) and not fortify.plan_mirror() and not fortify.plan_split()
    assert fortify.plan_menu_bound() == 0.0 and fortify.plan_menu_entropy() == 0.0
    assert fortify.plan_menu_kappa() is None and not fortify.plan_menu_log()
    assert fortify.plan_menu_share_floor() is None and not fortify.plan_menu_sigma_anneal()
    assert fortify.plan_ema() == 0.0 and fortify.plan_lr_schedule() == "constant" and fortify.plan_gnorm() == "off"
    assert fortify.plan_kappa_points() is None and fortify.plan_sample() == ("net", 0.0)
    assert fortify.plan_fs_guard() == 0.0 and fortify.plan_check() == "off" and fortify.plan_contract() == "off"
    cfg = plan_trainer.config()
    assert plan_trainer.av2_settings(cfg) == {} and plan_trainer.av2_line({}) == "" and plan_trainer.av2_meta({}) == {}
    assert plan_trainer.av2_state() is None
    good = {
        "ADV": [("rank", ("rank", None)), (" RANK ", ("rank", None)), ("global", ("global", 0.99)),
                ("global:0.9", ("global", 0.9)), ("floor:0.3", ("floor", 0.3)), ("off", ("off", None))],
        "MENU_KAPPA": [("bins", ("bins", 5)), ("film:3", ("film", 3)), ("off", None)],
        "MENU_SHARE_FLOOR": [("auto", "auto"), ("0.03", 0.03)],
        "LR_SCHEDULE": [("cosine", "cosine"), ("constant", "constant")],
        "GNORM": [("sqrt", "sqrt"), ("linear", "linear"), ("off", "off")],
        "KAPPA_POINTS": [("0.45,0.2", ((0.45, 0.2), 0.5, 0.05)), ("0.2:1:0.03", ((0.2,), 1.0, 0.03))],
        "SAMPLE": [("family", ("family", 1.0)), ("family:0.5", ("family", 0.5)), ("net", ("net", 0.0))],
        "FS_GUARD": [("1", 1.0), ("10", 10.0), ("0", 0.0)],
        "CHECK": [("1", "warn"), ("warn", "warn"), ("refuse", "refuse"), ("0", "off")],
        "CONTRACT": [("1", "on"), ("warn", "warn"), ("0", "off")],
        "MENU_BOUND": [("4", 4.0)], "MENU_ENT": [("0.02", 0.02)], "EMA": [("0.999", 0.999)],
    }
    bad = {
        "ADV": ["ranks", "rank:2", "floor", "floor:0", "floor:-1", "floor:x", "global:1", "global:0", "global:nan"],
        "MENU_KAPPA": ["bin", "bins:1", "bins:x", "film:"], "MENU_SHARE_FLOOR": ["0", "0.5", "-1", "wide"],
        "LR_SCHEDULE": ["cos", "linear"], "GNORM": ["g", "1"], "CHECK": ["yes", "2"], "CONTRACT": ["refuse"],
        "KAPPA_POINTS": ["1.2", "0.4:0", "0.4:2", "0.4:0.5:0.6", "a,b", "0.4:0.5:0.1:3", ":0.5"],
        "SAMPLE": ["families", "family:2", "family:-1", "family:x"], "FS_GUARD": ["0.5", "-1", "nan", "inf"],
        "MENU_BOUND": ["-1", "inf", "nan", "four"], "MENU_ENT": ["-0.1", "inf"], "EMA": ["1", "-0.1", "1.5"],
    }
    reader = {"ADV": fortify.plan_adv, "MENU_KAPPA": fortify.plan_menu_kappa, "LR_SCHEDULE": fortify.plan_lr_schedule,
              "MENU_SHARE_FLOOR": fortify.plan_menu_share_floor, "GNORM": fortify.plan_gnorm,
              "KAPPA_POINTS": fortify.plan_kappa_points, "SAMPLE": fortify.plan_sample,
              "FS_GUARD": fortify.plan_fs_guard, "CHECK": fortify.plan_check, "CONTRACT": fortify.plan_contract,
              "MENU_BOUND": fortify.plan_menu_bound, "MENU_ENT": fortify.plan_menu_entropy, "EMA": fortify.plan_ema}
    assert set(reader) | {"MIRROR", "SPLIT", "MENU_LOG", "MENU_SIGMA_ANNEAL"} == set(AV2_FLAGS)
    for name, cases in good.items():
        for raw, want in cases:
            monkeypatch.setenv(f"SPECTRA_PLAN_{name}", raw)
            assert reader[name]() == want, (name, raw)
        for raw in bad[name]:
            monkeypatch.setenv(f"SPECTRA_PLAN_{name}", raw)
            with pytest.raises(ValueError, match=f"SPECTRA_PLAN_{name}"):
                reader[name]()
        monkeypatch.delenv(f"SPECTRA_PLAN_{name}")
    for name, read in (("MIRROR", fortify.plan_mirror), ("SPLIT", fortify.plan_split),
                       ("MENU_LOG", fortify.plan_menu_log), ("MENU_SIGMA_ANNEAL", fortify.plan_menu_sigma_anneal)):
        monkeypatch.setenv(f"SPECTRA_PLAN_{name}", "1")
        assert read() is True
        monkeypatch.setenv(f"SPECTRA_PLAN_{name}", "0")
        assert read() is False
        monkeypatch.delenv(f"SPECTRA_PLAN_{name}")


def test_flags_that_cannot_work_are_refused_before_any_origin_is_read(monkeypatch, v10_env):
    monkeypatch.setattr(plan_agent.NetInstance, "from_env",
                        classmethod(lambda cls, *a, **k: pytest.fail("refused before the origins are read")))
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    monkeypatch.setenv("SPECTRA_PLAN_NETS", "toy")
    for flag, value in (("MENU_BOUND", "4"), ("MENU_ENT", "0.1"), ("MENU_KAPPA", "bins"), ("MENU_LOG", "1"),
                        ("MENU_SHARE_FLOOR", "auto"), ("MENU_SIGMA_ANNEAL", "1"), ("SPLIT", "1")):
        monkeypatch.setenv(f"SPECTRA_PLAN_{flag}", value)
        with pytest.raises(ValueError, match=f"SPECTRA_PLAN_{flag} needs the menu"):
            plan_trainer.run(None, SHARD)
        monkeypatch.delenv(f"SPECTRA_PLAN_{flag}")
    monkeypatch.setenv("SPECTRA_PLAN_K", "5")
    monkeypatch.setenv("SPECTRA_PLAN_MIRROR", "1")
    with pytest.raises(ValueError, match="SPECTRA_PLAN_MIRROR needs an even SPECTRA_PLAN_K, not 5"):
        plan_trainer.run(None, SHARD)
    monkeypatch.delenv("SPECTRA_PLAN_MIRROR")
    for key, value in {**MENU_LINE, "SPECTRA_PLAN_BUDGET": "flops", "SPECTRA_PLAN_SPLIT": "1"}.items():
        monkeypatch.setenv(key, value)
    for k, mirror in (("5", "0"), ("2", "0"), ("6", "1")):  # odd; a half of one plan; not divisible by 4 under MIRROR
        monkeypatch.setenv("SPECTRA_PLAN_K", k)
        monkeypatch.setenv("SPECTRA_PLAN_MIRROR", mirror)
        with pytest.raises(ValueError, match="SPECTRA_PLAN_SPLIT needs an even SPECTRA_PLAN_K"):
            plan_trainer.run(None, SHARD)
    _clear(monkeypatch)
    monkeypatch.setenv("SPECTRA_PLAN_NETS", "toy")
    monkeypatch.setenv("SPECTRA_PLAN_FS_GUARD", "1")
    with pytest.raises(ValueError, match="SPECTRA_PLAN_FS_GUARD watches the fs<S> proxy"):
        plan_trainer.run(None, SHARD)
    monkeypatch.setenv("SPECTRA_PLAN_PROXY", "fs4")
    assert plan_trainer.av2_settings(plan_trainer.config()) == {"fs_guard": 1.0}
    # the line and the meta name what is on, in one fixed order
    av2 = {"mirror": True, "adv": ("global", 0.9), "ema": 0.999, "kappa_points": ((0.45, 0.2), 0.5, 0.05),
           "sample": 1.0, "menu_kappa": {"kind": "bins", "n": 5, "lo": 0.35, "hi": 0.85}, "contract": True}
    assert plan_trainer.av2_line(av2) == (" | av2: adv=global:0.9 mirror=1 ema=0.999 menu_kappa=bins:5@0.35-0.85 "
                                          "kappa_points=0.45,0.2:0.5:0.05 sample=family:1 contract=1")
    assert plan_trainer.av2_meta(av2) == {"adv": "global:0.9", "mirror": True, "ema": 0.999, "sample": "family:1",
                                          "menu_kappa": {"kind": "bins", "n": 5, "lo": 0.35, "hi": 0.85},
                                          "kappa_points": [[0.45, 0.2], 0.5, 0.05]}


# ------------------------------------------------------------------ flags off: the loop is tree_v41's, to the bit


def _run_v41(env, shard):
    """``plan_trainer.run`` as tree_v41 has it (blob 17bfac6), copied here statement for statement; only the module
    prefixes are added. The flag-off identity test runs it beside the trainer's own ``run``."""
    cfg = plan_trainer.config()
    floor = plan_trainer.min_width()
    mode, alpha, sigma0 = plan_trainer.residual()
    res = None if mode == "off" else (mode, alpha)
    zmax = plan_trainer.residual_zmax() if res else 0.0
    init = plan_trainer.init_path()
    init_blob = plan_agent.read_blob(init) if init else None
    keep, resume = plan_trainer.state_enabled(), plan_trainer.resume_path()
    state = plan_trainer.read_state(resume) if resume else None
    budgets = plan_agent.BUDGETS if cfg.budget == "mixed" else (cfg.budget,)
    menu = plan_trainer.menu_settings(mode, budgets)
    with_cost = res is not None and "sens_cost" in {plan_agent.prior_kind(mode, b) for b in budgets}
    origin_kw = {}
    if menu is not None:
        with_cost = with_cost or "sens_cost" in menu["kinds"]
        if "sens_thresh" in menu["kinds"]:
            origin_kw["with_curves"] = True
    scale = sigma0 / cfg.sigma if res is not None and cfg.sigma > 0 else 1.0
    started = time.perf_counter()
    utils.print_flush(f"[plan] trainer {json.dumps(asdict(cfg))}" + (f" min_width={floor}" if floor > 1 else "")
                      + (f" residual={mode} alpha={alpha:g} sigma={sigma0:g}" if res else "")
                      + (f" zmax={zmax:g}" if zmax > 0 else "")
                      + (f" menu={','.join(menu['kinds'])} menu_init={menu['init']:g} menu_sigma={menu['sigma']:g}"
                         if menu else "")
                      + (f" fs_lr={plan_agent.fs_lr():g}" if cfg.proxy.startswith("fs") else "")
                      + (f" init={init}" if init else "")
                      + (" state=1" if keep else "") + (f" resume={resume}" if resume else ""))
    if menu is not None:
        shares0 = plan_agent.menu_shares([menu["init"]] + [0.0] * (len(menu["kinds"]) - 1))
        if max(shares0[1:]) >= plan_agent.MENU_SHARE_FLOOR:
            utils.print_flush(f"[plan] menu WARNING: SPECTRA_PLAN_MENU_INIT={menu['init']:g} starts the anchor at a "
                              f"share of {shares0[0]:.4f}, the other rules at or above the mean plan's share floor "
                              f"{plan_agent.MENU_SHARE_FLOOR:g}: the mean plan does not start on the prior's plan")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    instances = []
    for net_path, (net_model, net_loaders) in shard:
        name = os.path.basename(str(net_path))
        if cfg.nets and not any(part in name for part in cfg.nets):
            continue
        inst = plan_agent.NetInstance.from_env(env, net_path, net_model, net_loaders, tokens=cfg.tokens,
                                               zero=cfg.zero, with_cost=with_cost, **origin_kw)
        sens = sorted(inst.sens.values()) if inst.sens else [0.0]
        utils.print_flush(
            f"[plan] net {inst.name}: {inst.n_groups} groups, {int(inst.token_mask.sum())}/{inst.token_mask.numel()} "
            f"tokens mapped, width {inst.feature_dim}, {inst.pm.total0} params, origin val {100 * inst.origin_val:.2f}, "
            f"sens min {sens[0]:.4f} median {statistics.median(sens):.4f} max {sens[-1]:.4f}, "
            f"{len(inst.held)} multi-producer groups; {inst.seconds:.1f}s")
        for b in (budgets if res else ()):
            w = sorted(plan_trainer.prior_of(inst, res, b).values())
            utils.print_flush(
                f"[plan] prior {inst.name}{plan_trainer._budget_tag(b)}: {plan_agent.prior_kind(mode, b)} "
                f"alpha={alpha:g} weights "
                f"min {w[0]:.3f} median {statistics.median(w):.3f} max {w[-1]:.3f}")
        chk = inst.check()
        bad = abs(chk["analytic"] - chk["real"]) > 1e-6 or abs(chk["masked_r"] - chk["real_r"]) > 0.05
        utils.print_flush(
            f"[plan] check {inst.name}: uniform k=0.60 kept analytic x{chk['analytic']:.4f} real x{chk['real']:.4f} | "
            f"cut val Δ masked {chk['masked_r']:+.2f} real {chk['real_r']:+.2f} | dead channels {chk['dead']}"
            + (" | WARNING: the masked reward does not match the real cut" if bad else ""))
        if cfg.budget != "params":
            chk = inst.check_flops()
            utils.print_flush(
                f"[plan] check-flops {inst.name}: uniform k=0.60 FLOPs analytic x{chk['analytic']:.4f} real "
                f"x{chk['real']:.4f} | {chk['probes']} probes in {chk['seconds']:.1f}s"
                + (" | WARNING: the FLOPs model does not match the real cut"
                   if abs(chk["analytic"] - chk["real"]) > 1e-6 else ""))
        if menu is not None:
            tick, parts = time.perf_counter(), []
            for b in budgets:
                at = plan_trainer.menu_at(inst, menu, res, 0.6, cfg, b, floor)
                parts.append(f"k=0.60{plan_trainer._budget_tag(b)} rule plans' params kept "
                             + " ".join(f"x{inst.pm.kept(widths):.3f}" for widths, _info in at["plans"]))
            seconds = float((getattr(inst, "curves", None) or {}).get("seconds", 0.0)) + time.perf_counter() - tick
            utils.print_flush(f"[plan] menu setup {inst.name}: {','.join(menu['kinds'])} | " + " | ".join(parts)
                              + f" | {seconds:.1f}s")
        instances.append(inst)
    if not instances:
        raise RuntimeError(f"SPECTRA_PLAN_NETS={cfg.nets}: no test network matched")
    widths = {inst.feature_dim for inst in instances}
    if len(widths) != 1:
        raise RuntimeError(f"networks disagree on the token width: {sorted(widths)}")
    device = instances[0].device
    head_kw = {"menu": len(menu["kinds"]), "menu_init": menu["init"]} if menu is not None else {}
    policy = plan_agent.PlanPolicy(widths.pop(), cfg.encoder, **head_kw).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    folder = plan_trainer.out_dir()
    log_path = os.path.join(folder, "plan_train.jsonl")
    from src.feature_standardizer import resolve_standardizer_path
    meta = {"tokens": cfg.tokens, "zero": list(cfg.zero), "k_min": cfg.k_min, "proxy": cfg.proxy,
            "budget": cfg.budget, "nets": [inst.name for inst in instances], "config": asdict(cfg),
            "standardizer": resolve_standardizer_path(), "actor": os.environ.get("SPECTRA_ACTOR_CHECKPOINT_PATH", ""),
            **({"min_width": floor} if floor > 1 else {}),
            **({"residual": mode, "residual_alpha": alpha, "residual_sigma": sigma0} if res else {}),
            **({"residual_zmax": zmax} if zmax > 0 else {}),
            **({"menu": list(menu["kinds"]), "menu_anchor": menu["kinds"][0], "menu_init": menu["init"],
                "menu_sigma": menu["sigma"], "menu_share_floor": plan_agent.MENU_SHARE_FLOOR} if menu else {}),
            **({"fs_lr": plan_agent.fs_lr()} if cfg.proxy.startswith("fs") else {}),
            **({"init": init} if init else {})}
    if init_blob is not None:
        plan_trainer.warm_start(policy, init_blob, meta, init)

    def log(record):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def save(tag):
        path = os.path.join(folder, f"policy_{tag}.pt")
        plan_agent.save_policy(policy, path, **meta)
        plan_agent.save_policy(policy, os.path.join(folder, "policy_latest.pt"), **meta)
        return path

    state_path = os.path.join(folder, "trainer_state.pt")
    chain = (list(state.get("chain") or []) + [resume]) if state is not None else []
    last_state = [None]

    def save_state(count):
        plan_trainer.write_state(state_path, plan_trainer.trainer_state(policy, optimizer, rng, count, meta, chain))
        last_state[0] = count
        into = count % cfg.batch
        utils.print_flush(f"[plan] state {state_path} at {count} instances"
                          + (f", {into} of {cfg.batch} into a batch" if into else ""))

    optimizer.zero_grad()
    done = start = plan_trainer.restore(state, resume, cfg, meta, policy, optimizer, rng) if state is not None else 0
    for it in range(start, cfg.instances):
        t0 = time.perf_counter()
        inst = rng.choice(instances)
        kappa = rng.uniform(cfg.kappa_lo, cfg.kappa_hi)
        budget = cfg.budget if cfg.budget != "mixed" else rng.choice(plan_agent.BUDGETS)
        sigma = plan_trainer.sigma_at(cfg, it, scale)
        seed = cfg.seed * 100003 + it
        batches = inst.batches(cfg.proxy, seed)
        mu = policy(inst.state_at(kappa, budget), inst.token_mask, inst.token_k, inst.n_groups)
        generator = torch.Generator(device=mu.device).manual_seed(seed)
        z = plan_agent.sample_scores(mu, sigma, cfg.k, generator)
        rewards, kept = [], []
        cm = inst.cost_model(budget)
        prior = plan_trainer.prior_of(inst, res, budget)
        at, shares = plan_trainer.menu_at(inst, menu, res, kappa, cfg, budget, floor), None
        if at is not None:
            psi = policy.menu_logits(inst.state_at(kappa, budget), at["x"])
            u = plan_agent.sample_scores(psi, menu["sigma"], cfg.k, generator)
            shares = [plan_agent.menu_shares(row) for row in u.tolist()]
        for k in range(cfg.k):
            plan_widths, info = plan_trainer._decode(z[k].tolist(), cm, kappa, cfg, floor, prior, zmax,
                                                     None if shares is None else (shares[k], at["rules"]))
            rewards.append(inst.reward(plan_widths, cfg.proxy, batches))
            kept.append(info["kept"])
        if it == 0 and getattr(inst, "recipe_line", None):
            utils.print_flush(f"[plan] ft recipe {inst.name}: {inst.recipe_line}")
            meta["recipe_line"] = inst.recipe_line
        r = torch.tensor(rewards, device=mu.device, dtype=mu.dtype)
        adv = r - r.mean()
        if cfg.norm_adv:
            adv = adv / (r.std() + 1e-6)
        loss = -(adv.detach() * plan_agent.log_prob(z, mu, sigma)).mean() / cfg.batch
        if at is not None:
            loss = -(adv.detach() * plan_agent.menu_log_prob(z, mu, sigma, u, psi, menu["sigma"])).mean() / cfg.batch
        loss.backward()
        grad = None
        if (it + 1) % cfg.batch == 0:
            grad = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
            optimizer.step()
            optimizer.zero_grad()
        record = {"it": it, "net": inst.name, "kappa": kappa, "budget": budget, "sigma": sigma, "rewards": rewards,
                  "kept": kept, "mu_std": float(mu.detach().std()), "loss": float(loss) * cfg.batch, "grad": grad}
        if res:
            record["residual"], record["prior"] = mode, plan_agent.prior_kind(mode, budget)
        if zmax > 0:
            record["mu_beyond_zmax"] = int((mu.detach().abs() > zmax).sum())
        if shares is not None:
            record["menu_pi"] = [sum(s[j] for s in shares) / len(shares) for j in range(len(menu["kinds"]))]
        line = (f"[plan] it={it + 1}/{cfg.instances} {inst.name} k={kappa:.3f}{plan_trainer._budget_tag(budget)} "
                f"sigma={sigma:.3f} "
                f"| plans mean {statistics.mean(rewards):+.2f} max {max(rewards):+.2f} min {min(rewards):+.2f} kept "
                f"x{statistics.mean(kept):.3f} | mu std {record['mu_std']:.3f}")
        if zmax > 0:
            line += f" | mu beyond zmax {record['mu_beyond_zmax']}/{inst.n_groups}"
        if (it + 1) % cfg.ref_every == 0 or it == 0:
            plan_widths, info = plan_trainer.mean_plan(policy, inst, kappa, cfg, budget, floor, prior, zmax, at)
            record["mean_plan"] = {"r": inst.reward(plan_widths, cfg.proxy, batches), "kept": info["kept"]}
            record["refs"] = plan_trainer.references(inst, kappa, cfg, batches, budget, floor, prior)
            line += f" | mean plan {record['mean_plan']['r']:+.2f}" + "".join(
                f" {name} {ref['r']:+.2f}" for name, ref in record["refs"].items())
        record["seconds"] = time.perf_counter() - t0
        utils.print_flush(line + f" | {record['seconds']:.1f}s")
        log(record)
        done = it + 1
        if done % cfg.save_every == 0:
            save(f"it{done:05d}")
            if keep:
                save_state(done)
        over = cfg.max_minutes and (time.perf_counter() - started) / 60.0 >= cfg.max_minutes
        if over and not (keep and done % cfg.batch):
            utils.print_flush(f"[plan] MAX_MINUTES={cfg.max_minutes:g} reached after {done} instances")
            break
    if keep and last_state[0] != done:
        save_state(done)
    if done % cfg.batch:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    path = save(f"it{done:05d}")
    utils.print_flush(f"[plan] saved {path} after {done} instances, {(time.perf_counter() - started) / 60.0:.1f} min")
    plan_trainer.summary(policy, instances, cfg, log, floor, res, zmax, menu)
    utils.print_flush(f"[plan] DONE {done} instances in {(time.perf_counter() - started) / 60.0:.1f} min")


def _timeless(line, folder):
    """A log line without its wall-clock parts and with the run folder named ``<out>``."""
    line = line.replace(str(folder), "<out>")
    return re.sub(r"\d+\.\d+ min", "<min>", re.sub(r"\d+\.\d+s\b", "<s>", line))


def _same_folder(a, b, roots=None):
    """Every file of two run folders agrees: the same names, the records without their seconds, and every checkpoint
    and state blob bitwise (a state's wall-clock ``saved`` aside; the paths under ``roots``, by default the folders
    themselves, named alike)."""
    roots = roots or (a, b)
    assert sorted(os.listdir(a)) == sorted(os.listdir(b)), (sorted(os.listdir(a)), sorted(os.listdir(b)))
    for name in sorted(os.listdir(a)):
        if name.endswith(".jsonl"):
            ra, rb = [[{k: v for k, v in json.loads(line).items() if k != "seconds"}
                       for line in open(os.path.join(folder, name), encoding="utf-8")] for folder in (a, b)]
            assert ra == rb, name
        elif name.endswith(".pt"):
            blobs = []
            for folder, root in zip((a, b), roots):
                blob = plan_agent.read_blob(os.path.join(folder, name))
                blob.pop("saved", None)
                blobs.append(_named(blob, str(root)))
            assert set(blobs[0]) == set(blobs[1]), (name, sorted(set(blobs[0]) ^ set(blobs[1])))
            for key in blobs[0]:
                assert _equal(blobs[0][key], blobs[1][key]), (name, key)


def _named(obj, folder):
    """``obj`` with every string's ``folder`` prefix replaced by ``<out>`` (a resumed run's chain holds its path)."""
    if isinstance(obj, str):
        return obj.replace(folder, "<out>")
    if isinstance(obj, dict):
        return {key: _named(value, folder) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_named(value, folder) for value in obj]
    return obj


LINES_OF_RECORD = {
    "direct": {"SPECTRA_PLAN_NORM_ADV": "0", "SPECTRA_PLAN_STATE": "1"},
    "residual": {**T1, "SPECTRA_PLAN_BUDGET": "flops", "SPECTRA_PLAN_PROXY": "cut", "SPECTRA_PLAN_STATE": "1"},
    "menu": {**MENU_LINE, "SPECTRA_PLAN_BUDGET": "flops", "SPECTRA_PLAN_PROXY": "cut", "SPECTRA_PLAN_STATE": "1",
             "SPECTRA_PLAN_INSTANCES": "7"},  # an odd count: the closing partial step too
}


@pytest.mark.parametrize("line", list(LINES_OF_RECORD))
def test_flags_off_the_trainer_is_the_tree_v41_loop_to_the_bit(monkeypatch, tmp_path, v10_env, line):
    """The three lines of record on the toy net, every AGENT-V2 flag unset, once through tree_v41's loop and once
    through the trainer's: the same log lines, records, checkpoints and resumable state, bitwise. The reward draws
    from every global generator, so a draw added or moved anywhere in the loop shows."""
    out = {}
    for name, runner in (("v41", _run_v41), ("now", plan_trainer.run)):
        _clear(monkeypatch)
        inst = _menu_instance("synthetic")
        lines = _prepare(monkeypatch, tmp_path / name, inst, **LINES_OF_RECORD[line])
        assert not any(os.environ.get(f"SPECTRA_PLAN_{flag}") for flag in AV2_FLAGS)
        runner(None, SHARD)
        out[name] = [_timeless(text, tmp_path / name) for text in lines]
    assert out["v41"] == out["now"]
    assert " | av2:" not in out["now"][0] and any(text.startswith("[plan] DONE ") for text in out["now"])
    _same_folder(tmp_path / "v41", tmp_path / "now")
    blob = plan_agent.read_blob(str(tmp_path / "now" / "policy_latest.pt"))
    state = plan_agent.read_blob(str(tmp_path / "now" / "trainer_state.pt"))
    assert set(state) == set(plan_trainer.STATE_KEYS) and "av2" not in state
    new_keys = {"adv", "mirror", "split", "gnorm", "ema", "lr_schedule", "menu_bound", "menu_entropy", "menu_kappa",
                "menu_sigma_anneal", "menu_log", "kappa_points", "sample", "fs_guard", "check", "state_contract",
                "weights"}
    assert not new_keys & set(blob) and not new_keys & set(state["meta"])
    assert not any(name.startswith("policy_ema") for name in os.listdir(tmp_path / "now"))
    steps = _its([json.loads(text) for text in open(tmp_path / "now" / "plan_train.jsonl", encoding="utf-8")])
    assert sum(abs(r["loss"]) > 1e-3 for r in steps) >= 4  # the comparison is not of zeros
    assert float(sum(v.abs().sum() for k, v in blob["state_dict"].items() if k.startswith("head."))) > 0


_DRIVER = r'''
import json, os, random, sys
sys.dont_write_bytecode = True
from pathlib import Path
tree, out, jobs = sys.argv[1], Path(sys.argv[2]), json.loads(sys.argv[3])
sys.path.insert(0, tree)
os.chdir(tree)
for key in list(os.environ):
    if key.startswith("SPECTRA_"):
        del os.environ[key]
import numpy as np
import torch
from tests.test_pruning import _init_static_conf
_init_static_conf()
from tests.test_plan_agent import V10_FLAGS
os.environ.update(V10_FLAGS)
from tests.test_plan_menu import MENU_FLAG, _instance, _origin
from tests.test_v25_plan_resume import BASE, _synthetic_reward
from src import plan_agent, plan_trainer
import src.run_recorder as recorder
import src.utils as utils
recorder.record = lambda *a, **k: None
assert os.path.realpath(plan_trainer.__file__).startswith(os.path.realpath(tree)), plan_trainer.__file__
for job in jobs:
    folder = out / job["name"]
    folder.mkdir(parents=True, exist_ok=True)
    for key in list(os.environ):
        if key.startswith("SPECTRA_PLAN_"):
            del os.environ[key]
    for key, value in {**BASE, **job["env"]}.items():
        os.environ[key] = str(value).replace("{out}", str(out))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    inst = _instance()
    if job.get("reward") == "synthetic":
        inst.reward = _synthetic_reward(inst)
    plan_agent.NetInstance.from_env = classmethod(lambda cls, env, p, m, l, **kw: inst)
    plan_trainer.out_dir = lambda: str(folder)
    lines = []
    utils.print_flush = lambda s, *a, **k: lines.append(str(s))
    plan_trainer.run(None, [("toy.pt", (None, None))])
    (out / (job["name"] + ".lines.json")).write_text(json.dumps(lines), encoding="utf-8")


class Patch:
    def setattr(self, target, name, value):
        setattr(target, name, value)


# the frozen agent's side: the menu job's last iterate planned on one origin, as the eval walk asks for it
os.environ["SPECTRA_PLAN_MENU"] = MENU_FLAG
env, _model, _plan = _origin(Patch())
random.seed(5)
np.random.seed(5)
torch.manual_seed(5)
lines = []
utils.print_flush = lambda s, *a, **k: lines.append(str(s))
widths, info = plan_agent.plan_for_env(env, 0.58, str(out / "menu" / "policy_latest.pt"), kappa=0.6, budget="flops")
plan = {"widths": {str(k): int(v) for k, v in widths.items()}, "kept": info["kept"], "menu": info["menu"]["pi"],
        "psi": info["menu"]["psi"], "sens": {str(k): float(v) for k, v in info["sens"].items()},
        "keys": sorted(info), "lines": lines, "rng": torch.get_rng_state().tolist()[:64]}
(out / "plan_for_env.json").write_text(json.dumps(plan), encoding="utf-8")
print("DRIVER-DONE")
'''

_MENU_JOB = {**MENU_LINE, "SPECTRA_PLAN_BUDGET": "flops", "SPECTRA_PLAN_PROXY": "cut", "SPECTRA_PLAN_STATE": "1"}
_TREE_JOBS = [
    {"name": "direct_fs2", "env": {"SPECTRA_PLAN_PROXY": "fs2", "SPECTRA_PLAN_NORM_ADV": "0",
                                   "SPECTRA_PLAN_KAPPA": "0.25,0.85", "SPECTRA_PLAN_STATE": "1"}},
    {"name": "direct_bn2", "env": {}},
    {"name": "direct_syn", "env": {"SPECTRA_PLAN_NORM_ADV": "0", "SPECTRA_PLAN_INSTANCES": "7"}, "reward": "synthetic"},
    {"name": "residual_cut", "env": {**T1, "SPECTRA_PLAN_BUDGET": "flops", "SPECTRA_PLAN_PROXY": "cut"}},
    {"name": "residual_syn", "env": {**T1, "SPECTRA_PLAN_BUDGET": "flops", "SPECTRA_PLAN_PROXY": "cut",
                                     "SPECTRA_PLAN_STATE": "1"}, "reward": "synthetic"},
    {"name": "mixed_auto", "env": {"SPECTRA_PLAN_RESIDUAL": "auto", "SPECTRA_PLAN_BUDGET": "mixed",
                                   "SPECTRA_PLAN_PROXY": "cut"}, "reward": "synthetic"},
    {"name": "menu", "env": _MENU_JOB, "reward": "synthetic"},
    {"name": "menu_stop", "env": {**_MENU_JOB, "SPECTRA_PLAN_MAX_MINUTES": "1e-9"}, "reward": "synthetic"},
    {"name": "menu_resume", "env": {**_MENU_JOB, "SPECTRA_PLAN_RESUME": "{out}/menu_stop/trainer_state.pt"},
     "reward": "synthetic"},
]


@pytest.mark.skipif(not os.environ.get("SPECTRA_AV2_REF_TREE"),
                    reason="set SPECTRA_AV2_REF_TREE to an untouched tree_v41 to compare this tree with it")
def test_flags_off_this_tree_and_the_reference_tree_write_the_same_bytes(tmp_path):
    """The acceptance check of queue row AGENT-V2: the same nine trainer jobs (the direct line on the fs2 and bn2
    proxies, the residual line, a mixed budget, the menu line, and a menu run stopped and resumed) and one frozen-agent
    plan, run from the untouched reference tree and from this tree in two processes, write the same files."""
    reference = os.path.realpath(os.environ["SPECTRA_AV2_REF_TREE"])
    here = str(Path(__file__).resolve().parents[1])
    assert reference != os.path.realpath(here) and os.path.isfile(os.path.join(reference, "src", "plan_trainer.py"))
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("SPECTRA_")}
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # the reference tree is read, never written: no bytecode caches either
    for name, tree in (("ref", reference), ("new", here)):
        done = subprocess.run([sys.executable, str(driver), tree, str(tmp_path / name), json.dumps(_TREE_JOBS)],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                              timeout=1800)
        assert done.returncode == 0 and "DRIVER-DONE" in done.stdout, done.stdout[-3000:]
    for job in _TREE_JOBS:
        folders = [tmp_path / name / job["name"] for name in ("ref", "new")]
        _same_folder(*folders, roots=(tmp_path / "ref", tmp_path / "new"))
        texts = [[_timeless(text, tmp_path / name) for text in
                  json.loads((tmp_path / name / (job["name"] + ".lines.json")).read_text(encoding="utf-8"))]
                 for name in ("ref", "new")]
        assert texts[0] == texts[1], job["name"]
        assert any(text.startswith("[plan] DONE ") for text in texts[1]) and " | av2:" not in texts[1][0]
    plans = [json.loads((tmp_path / name / "plan_for_env.json").read_text(encoding="utf-8")) for name in ("ref", "new")]
    for plan, name in zip(plans, ("ref", "new")):
        plan["lines"] = [_timeless(text, tmp_path / name) for text in plan["lines"]]
    assert plans[0] == plans[1]
    resumed = _weights(tmp_path / "new" / "menu_resume" / "policy_latest.pt")
    assert _equal(resumed, _weights(tmp_path / "new" / "menu" / "policy_latest.pt"))  # and the chain is the one run
    assert float(resumed["menu_head.weight"].abs().sum()) > 0


# ------------------------------------------------------------------ rank 1b: the cross-entropy rewards


def test_cutce_and_bnce_rewards_are_the_val_cross_entropy_change(monkeypatch, tmp_path, v10_env):
    assert plan_agent.parse_proxy("cutce", ce=True) == ("cutce", 0)
    assert plan_agent.parse_proxy(" BNCE8 ", ce=True) == ("bnce", 8)
    for name in ("cutce", "bnce8"):  # every other reader of the grammar refuses the names, as before
        with pytest.raises(ValueError, match="expected cut, bn<N>, fs<S> or ft<E>"):
            plan_agent.parse_proxy(name)
        with pytest.raises(ValueError):
            plan_agent.parse_proxy(name, judge=True)
    for bad in ("bnce", "bnce0", "cutce2", "fsce4", "ce"):
        with pytest.raises(ValueError):
            plan_agent.parse_proxy(bad, ce=True)
    assert plan_agent.parse_proxy("bn8", ce=True) == ("bn", 8) == plan_agent.parse_proxy("bn8")
    inst = _toy_instance()
    widths, _info = plan_agent.decode([0.1 * k for k in range(inst.n_groups)], inst.pm, 0.6)
    real = alloc_walk.cut_to(inst.model, inst.plan, plan_agent.rates_of(widths, inst.pm.widths0), SHAPE).eval()
    loss = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        ce0 = sum(float(loss(inst.model(x), y)) for x, y in inst.val) / sum(int(y.numel()) for _x, y in inst.val)
        ce1 = sum(float(loss(real(x), y)) for x, y in inst.val) / sum(int(y.numel()) for _x, y in inst.val)
    assert plan_agent.val_loss(inst.model, inst.val) == pytest.approx(ce0, rel=1e-6) == inst.origin_ce()
    assert not inst.model.training and inst.origin_ce() is inst._origin_ce          # measured once, mode put back
    r_cut = inst.reward(widths, "cutce", [])
    assert r_cut == pytest.approx(-100.0 * (ce1 - ce0), abs=0.05) and abs(r_cut) > 0.05
    assert inst.reward(dict(inst.pm.widths0), "cutce", []) == pytest.approx(0.0, abs=1e-4)   # the uncut plan
    assert inst.batches("cutce", seed=3) == []
    batches = inst.batches("bnce4", seed=3)
    assert len(batches) == 4 and torch.equal(batches[0], inst.batches("bn4", seed=3)[0])
    r_bn = inst.reward(widths, "bnce4", batches)
    recovery_edits.recalibrate_batchnorm(real, batches, "cpu", 4)
    assert r_bn == pytest.approx(-100.0 * (plan_agent.val_loss(real, inst.val) - ce0), abs=0.05) and r_bn != r_cut
    # the accuracy rewards are untouched, and never read the origin's cross-entropy
    fresh = _toy_instance()
    assert fresh.reward(widths, "cut", []) == pytest.approx(
        100.0 * (plan_agent.accuracy(alloc_walk.cut_to(fresh.model, fresh.plan, plan_agent.rates_of(
            widths, fresh.pm.widths0), SHAPE), fresh.val) - fresh.origin_val))
    assert fresh._origin_ce is None
    # the trainer takes the proxy and trains on it
    monkeypatch.setenv("SPECTRA_PLAN_PROXY", "bnce2")
    assert plan_trainer.config().proxy == "bnce2"
    lines, records, blob = _go(monkeypatch, tmp_path, inst, budget="params", SPECTRA_PLAN_PROXY="cutce", instances=3)
    assert blob["proxy"] == "cutce" and '"proxy": "cutce"' in lines[0] and " | av2:" not in lines[0]
    steps = _its(records)
    assert len(steps) == 3 and all(len(r["rewards"]) == 4 for r in steps)
    assert steps[0]["refs"]["uniform"]["r"] == pytest.approx(
        inst.reward(inst.reference("uniform", steps[0]["kappa"])[0], "cutce", []))
    image = 100.0 / 64.0                                           # accuracy moves in steps of one of the 64 val images
    assert any(abs(v / image - round(v / image)) > 1e-3 for r in steps for v in r["rewards"])


# ------------------------------------------------------------------ rank 2: the advantages


def test_rank_utilities_are_centred_ranks_invariant_to_any_increasing_map_of_the_rewards():
    r = torch.tensor([0.3, -1.2, 5.0, 0.31, 2.0, -7.0, 0.0, 1.5])
    u = plan_agent.rank_utilities(r)
    k = r.numel()
    want = (torch.arange(k, dtype=r.dtype) / (k - 1) - 0.5) * ((k - 1) * math.sqrt(12.0 / (k * (k + 1))))
    torch.testing.assert_close(u[torch.argsort(r)], want)
    assert abs(float(u.sum())) < 1e-5 and float(u.std()) == pytest.approx(1.0, abs=1e-5)   # the SD-normalised scale
    for scale in (1000.0, 1e-3, 3.7, 64.0):                     # the reward's scale and offset leave them unchanged
        assert torch.equal(plan_agent.rank_utilities(scale * r), u)
        assert torch.equal(plan_agent.rank_utilities(scale * r - 4.0), u)
    assert torch.equal(plan_agent.rank_utilities(r ** 3), u) and torch.equal(plan_agent.rank_utilities(torch.exp(r)), u)
    torch.testing.assert_close(plan_agent.rank_utilities(-r), -u)
    # an outlier weighs as the worst plan, no more: the diverged plan of audit F11
    a = plan_agent.rank_utilities(torch.tensor([-1.0, -2.0, -3.0, -90.0]))
    assert torch.equal(a, plan_agent.rank_utilities(torch.tensor([-1.0, -2.0, -3.0, -3.5])))
    # ties share their mean rank; an instance whose plans all read alike gives no gradient at all
    s4 = 3.0 * math.sqrt(12.0 / 20.0)
    assert plan_agent.rank_utilities(torch.tensor([1.0, 1.0, 2.0, 3.0])).tolist() == pytest.approx(
        [(0.5 / 3 - 0.5) * s4, (0.5 / 3 - 0.5) * s4, (2 / 3 - 0.5) * s4, 0.5 * s4])
    assert not bool(plan_agent.rank_utilities(torch.full((8,), -65.3)).any())
    assert plan_agent.rank_utilities(torch.tensor([2.0])).tolist() == [0.0]
    # against the SD division, which scales a one-image difference up to a full-size gradient
    cfg = plan_trainer.config()
    flat = torch.tensor([-65.30, -65.30, -65.30, -65.30, -65.30, -65.30, -65.30, -65.28])
    assert float(plan_trainer.advantages(flat, cfg).abs().max()) > 2.0
    ranked = plan_trainer.advantages(flat, cfg, ("rank", None))
    assert torch.equal(ranked, plan_agent.rank_utilities(flat)) and float(ranked.abs().max()) < 1.5


def test_adv_rank_makes_the_whole_run_invariant_to_the_reward_scale(monkeypatch, tmp_path, v10_env):
    def job(name, scale, **flags):
        inst = _instance()
        base = _reward(inst)
        inst.reward = lambda widths, proxy, batches: scale * base(widths, proxy, batches)
        return _go(monkeypatch, tmp_path / name, inst, **T1, **flags)

    draws = _spy(monkeypatch)
    lines, records, blob = job("a", 1.0, SPECTRA_PLAN_ADV="rank")
    first = list(draws)
    _l, big_records, big = job("b", 1000.0, SPECTRA_PLAN_ADV="rank")
    assert lines[0].endswith(" | av2: adv=rank") and blob["adv"] == "rank" == big["adv"]
    steps, big_steps = _its(records), _its(big_records)
    assert [r["rewards"] for r in big_steps] != [r["rewards"] for r in steps]
    assert [r["loss"] for r in big_steps] == [r["loss"] for r in steps]              # the same losses ...
    assert [r["grad"] for r in big_steps] == [r["grad"] for r in steps]
    assert _equal(big["state_dict"], blob["state_dict"])                             # ... and the same policy, bitwise
    assert float(blob["state_dict"]["head.weight"].abs().sum()) > 0
    for r, d in zip(steps, first):                                 # the loss is the rank utilities' estimator
        rewards = torch.tensor(r["rewards"], dtype=d["mean"].dtype)
        want = float(-(plan_agent.rank_utilities(rewards) * plan_agent.log_prob(d["z"], d["mean"], d["sigma"])).mean())
        assert r["loss"] == pytest.approx(want, rel=1e-4, abs=1e-5)
    # the plain mean baseline, the parameter line's, follows the scale
    _l, plain, _b = job("c", 1.0, SPECTRA_PLAN_NORM_ADV="0")
    _l, plain_big, _b = job("d", 1000.0, SPECTRA_PLAN_NORM_ADV="0")
    assert _its(plain_big)[0]["loss"] == pytest.approx(1000.0 * _its(plain)[0]["loss"], rel=1e-3)


def test_adv_global_and_floor_divide_by_a_shared_or_a_floored_scale(monkeypatch, tmp_path, v10_env):
    cfg = plan_trainer.config()
    r = torch.tensor([1.0, 2.0, 4.0, 9.0])
    assert torch.equal(plan_trainer.advantages(r, cfg), (r - r.mean()) / (r.std() + 1e-6))     # the default, as it was
    monkeypatch.setenv("SPECTRA_PLAN_NORM_ADV", "0")
    assert torch.equal(plan_trainer.advantages(r, plan_trainer.config()), r - r.mean())
    torch.testing.assert_close(plan_trainer.advantages(r, cfg, ("floor", 10.0)), (r - r.mean()) / 10.0)
    torch.testing.assert_close(plan_trainer.advantages(r, cfg, ("floor", 0.1)), (r - r.mean()) / r.std())
    flat = torch.tensor([5.0, 5.0, 5.0, 5.02])                                          # one val image apart
    assert float(plan_trainer.advantages(flat, cfg).abs().max()) > 1.0
    assert float(plan_trainer.advantages(flat, cfg, ("floor", 0.3)).abs().max()) < 0.06
    scale = plan_trainer.RunningScale(0.9)
    assert scale.read() == 0.0
    torch.testing.assert_close(plan_trainer.advantages(r, cfg, ("global", 0.9), scale), (r - r.mean()) / r.std())
    want = (0.9 * 0.1 * float(r.std()) + 0.1 * float(flat.std())) / (1.0 - 0.9 ** 2)
    second = plan_trainer.advantages(flat, cfg, ("global", 0.9), scale)
    assert scale.read() == pytest.approx(want) and scale.steps == 2
    torch.testing.assert_close(second, (flat - flat.mean()) / want)
    assert float(second.abs().max()) < 0.02                              # the flat instance weighs by its own spread
    other = plan_trainer.RunningScale(0.9)
    other.load(scale.state())
    assert other.read() == scale.read() and other.update(float("nan")) == scale.read()   # a one-plan half: no move
    # in the trainer: the running scale is recorded, and it is the instances' SDs
    inst = _menu_instance()
    lines, records, blob = _go(monkeypatch, tmp_path / "g", inst, SPECTRA_PLAN_ADV="global:0.9", **T1)
    assert lines[0].endswith(" | av2: adv=global:0.9") and blob["adv"] == "global:0.9"
    check = plan_trainer.RunningScale(0.9)
    for record in _its(records):
        want = check.update(float(torch.tensor(record["rewards"]).std()))
        assert record["adv_scale"] == pytest.approx(want, rel=1e-5)
    inst = _menu_instance()
    lines, records, blob = _go(monkeypatch, tmp_path / "f", inst, SPECTRA_PLAN_ADV="floor:0.25", **T1)
    assert lines[0].endswith(" | av2: adv=floor:0.25") and blob["adv"] == "floor:0.25"
    assert all("adv_scale" not in record for record in records)


# ------------------------------------------------------------------ rank 3: mirrored sampling


def test_mirrored_pairs_sum_to_twice_the_mean_and_cancel_the_mean_noise():
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(7, generator=g)
    z = plan_agent.sample_scores(mu, 0.25, 8, torch.Generator().manual_seed(5), mirror=True)
    assert z.shape == (8, 7) and not z.requires_grad
    half = torch.randn((4, 7), generator=torch.Generator().manual_seed(5))
    assert torch.equal(z[:4], mu.unsqueeze(0) + 0.25 * half)                   # the first half is the plain draw
    torch.testing.assert_close(z[:4] + z[4:], 2.0 * mu.expand(4, 7), atol=2e-6, rtol=0)   # the pairs sum to 2 mu
    torch.testing.assert_close(z.mean(dim=0), mu, atol=2e-6, rtol=0)           # so the mean noise is gone ...
    plain = plan_agent.sample_scores(mu, 0.25, 8, torch.Generator().manual_seed(5))
    assert float((plain.mean(dim=0) - mu).abs().max()) > 0.02                  # ... which a plain draw leaves in
    assert torch.equal(plain, mu.unsqueeze(0) + 0.25 * torch.randn((8, 7), generator=torch.Generator().manual_seed(5)))
    with pytest.raises(ValueError, match="even number of plans"):
        plan_agent.sample_scores(mu, 0.25, 5, mirror=True)

    def gradient(z, reward):
        """The REINFORCE gradient on the mean for the rewards of ``z`` (mean baseline, no division)."""
        m = mu.clone().requires_grad_(True)
        r = reward(z)
        (-((r - r.mean()) * plan_agent.log_prob(z, m, 0.25)).mean()).backward()
        return m.grad

    # an even reward (a bowl centred on the mean) has no gradient there: the pairs say so exactly, a plain draw does not
    bowl = lambda z: -((z - mu) ** 2).sum(dim=1)  # noqa: E731
    assert float(gradient(z, bowl).abs().max()) < 1e-4 < float(gradient(plain, bowl).abs().max())
    # a slope plus the bowl: the pairs read the slope alone, the central difference of each pair
    slope = torch.linspace(-1.0, 1.0, 7)
    tilted = lambda z: (z * slope).sum(dim=1) - 3.0 * ((z - mu) ** 2).sum(dim=1)  # noqa: E731
    linear = lambda z: (z * slope).sum(dim=1)  # noqa: E731
    torch.testing.assert_close(gradient(z, tilted), gradient(z, linear), atol=1e-4, rtol=1e-4)
    assert float((gradient(plain, tilted) - gradient(plain, linear)).abs().max()) > 1e-2


def test_trainer_mirror_draws_pairs_for_the_scores_and_the_menu_logits(monkeypatch, tmp_path, v10_env):
    draws = _spy(monkeypatch)
    inst = _menu_instance()
    lines, records, blob = _go(monkeypatch, tmp_path, inst, SPECTRA_PLAN_K="8", SPECTRA_PLAN_MIRROR="1", **MENU_LINE)
    assert lines[0].endswith(" | av2: mirror=1") and blob["mirror"] is True
    steps, cfg, fm = _its(records), plan_trainer.config(), inst.cost_model("flops")
    assert len(draws) == 2 * len(steps) and all(d["k"] == 8 and d["mirror"] is True for d in draws)
    for it, r in enumerate(steps):
        scores, logits = draws[2 * it], draws[2 * it + 1]
        g = torch.Generator().manual_seed(cfg.seed * 100003 + it)
        eps = torch.randn((4, inst.n_groups), generator=g)
        assert torch.equal(scores["z"], scores["mean"].unsqueeze(0) + scores["sigma"] * torch.cat([eps, -eps]))
        eps_u = torch.randn((4, len(MENU)), generator=g)                     # the logits' half, after the scores'
        assert torch.equal(logits["z"], logits["mean"].unsqueeze(0) + 2.0 * torch.cat([eps_u, -eps_u]))
        rules = inst.rules_at(MENU, r["kappa"], "flops", 0.5, cfg.k_min, 2)
        kept = [plan_agent.menu_decode(scores["z"][k].tolist(), plan_agent.menu_shares(logits["z"][k].tolist()), rules,
                                       fm, r["kappa"], cfg.k_min, min_width=2, zmax=0.3)[1]["kept"] for k in range(8)]
        assert r["kept"] == kept and len(r["rewards"]) == 8
    # the scores' noise without the menu is the same mirrored draw: the logits are still drawn after it
    del draws[:]
    plain = _menu_instance()
    _go(monkeypatch, tmp_path / "plain", plain, SPECTRA_PLAN_K="8", SPECTRA_PLAN_MIRROR="1", **T1)
    g = torch.Generator().manual_seed(cfg.seed * 100003)
    eps = torch.randn((4, plain.n_groups), generator=g)
    assert torch.equal(draws[0]["z"], draws[0]["sigma"] * torch.cat([eps, -eps]))   # mu is 0 at the first instance


# ------------------------------------------------------------------ rank 4: the menu head


def test_bounded_menu_logits_stay_within_the_bound(v10_env):
    torch.manual_seed(0)
    state, x = toy_state(8), [0.2, -0.1, 0.05, 0.6]
    plain, bounded = plan_agent.PlanPolicy(63, menu=4), plan_agent.PlanPolicy(63, menu=4, menu_bound=4.0)
    assert set(bounded.state_dict()) == set(plain.state_dict()) and bounded.menu_bound == 4.0 == 4.0 + plain.menu_bound
    assert plan_agent.PlanPolicy(63, menu_bound=4.0).menu_bound == 0.0                # no menu, no bound
    with torch.no_grad():
        assert bounded.menu_logits(state, x).tolist() == pytest.approx([4.0 * math.tanh(1.5), 0.0, 0.0, 0.0])
        gen = torch.Generator().manual_seed(1)
        for std in (0.01, 0.1, 1.0, 100.0):
            plain.menu_head.weight.normal_(std=std, generator=gen)
            plain.menu_head.bias.normal_(std=20.0 * std, generator=gen)
            bounded.load_state_dict(plain.state_dict())
            raw, psi = plain.menu_logits(state, x), bounded.menu_logits(state, x)
            assert float(psi.abs().max()) <= 4.0                                        # never past B ...
            torch.testing.assert_close(psi, 4.0 * torch.tanh(raw / 4.0), atol=1e-6, rtol=1e-6)
            assert psi.tolist() == pytest.approx(plan_agent.menu_bounded(raw.tolist(), 4.0), abs=1e-5)
            assert float(psi.max() - psi.min()) <= 8.0                                  # ... so no gap past 2 B
        assert float(raw.abs().max()) > 100.0                                           # the raw head was far past it
        plain.menu_head.weight.zero_()
        plain.menu_head.bias.copy_(torch.tensor([0.3, -0.2, 0.1, 0.0]))
        bounded.load_state_dict(plain.state_dict())
        torch.testing.assert_close(bounded.menu_logits(state, x), plain.menu_logits(state, x), atol=2e-3, rtol=0)
    # the gradient passes the bound, thinned by tanh's slope and never cut off
    bounded.zero_grad()
    psi = bounded.menu_logits(state, x)
    psi.sum().backward()
    torch.testing.assert_close(bounded.menu_head.bias.grad, 1.0 - torch.tanh(plain.menu_head.bias / 4.0) ** 2,
                               atol=1e-5, rtol=1e-5)
    assert plan_agent.menu_bounded([6.0, 0.0], 0.0) == [6.0, 0.0]
    # the mean plan's share floor follows the bounded init, so the untrained agent still starts on the anchor's plan
    assert plan_agent.menu_floor(6.0, 4) == plan_agent.MENU_SHARE_FLOOR == plan_agent.menu_floor(6.0, 4, 0.0, "auto")
    floor = plan_agent.menu_floor(6.0, 4, 4.0)
    start = plan_agent.menu_shares(plan_agent.menu_bounded([6.0, 0.0, 0.0, 0.0], 4.0))
    assert floor == 0.031 and start[1] == pytest.approx(0.02478, abs=5e-5) and start[1] < floor < start[0]
    assert plan_agent.menu_shares(plan_agent.menu_bounded([6.0, 0.0, 0.0, 0.0], 4.0), floor) == [1.0, 0.0, 0.0, 0.0]
    assert plan_agent.menu_floor(6.0, 4, 4.0, 0.05) == 0.05 and plan_agent.menu_floor(6.0, 4, 0.0, 0.01) == 0.01
    assert plan_agent.menu_floor(0.0, 4, 4.0) == plan_agent.MENU_SHARE_FLOOR          # an even start has no such floor


def test_trainer_bound_starts_on_the_prior_logs_psi_and_the_frozen_agent_reads_the_same_head(monkeypatch, tmp_path,
                                                                                            v10_env):
    inst = _menu_instance()
    lines, records, blob = _go(monkeypatch, tmp_path / "b", inst, SPECTRA_PLAN_MENU_BOUND="4",
                               SPECTRA_PLAN_MENU_LOG="1", SPECTRA_PLAN_LR="0.05", **MENU_LINE)
    assert lines[0].endswith(" | av2: menu_bound=4 menu_share_floor=0.031 menu_log=1")
    assert not any(line.startswith("[plan] menu WARNING") for line in lines)
    assert blob["menu_bound"] == 4.0 and blob["menu_share_floor"] == 0.031 and blob["menu_log"] is True
    steps = _its(records)
    assert steps[0]["mean_plan"] == steps[0]["refs"]["prior"]                 # the untrained mean plan: the anchor's
    assert steps[0]["menu_psi"] == pytest.approx([4.0 * math.tanh(1.5), 0.0, 0.0, 0.0], abs=1e-5)
    assert steps[-1]["menu_psi"] != pytest.approx(steps[0]["menu_psi"], abs=1e-3)       # lr 0.05: the head has moved
    for r in steps:
        assert max(abs(v) for v in r["menu_psi"]) <= 4.0
        assert r["menu_psi_range"] == pytest.approx(max(r["menu_psi"]) - min(r["menu_psi"])) and r["menu_psi_range"] <= 8
        assert len(r["menu_shares"]) == 4 and all(sum(row) == pytest.approx(1.0, abs=2e-3) for row in r["menu_shares"])
        assert [sum(row[j] for row in r["menu_shares"]) / 4 for j in range(4)] == pytest.approx(r["menu_pi"], abs=1e-3)
    assert sum(" | psi range " in line for line in lines) == len(steps)
    summaries = [r for r in records if "summary" in r]
    assert summaries and all(sum(s["menu_pi"]) == pytest.approx(1.0) for s in summaries)
    # without the flag the shares' record is as it was
    plain = _menu_instance()
    _l, plain_records, plain_blob = _go(monkeypatch, tmp_path / "p", plain, **MENU_LINE)
    assert plain_blob["menu_share_floor"] == 0.005 and "menu_bound" not in plain_blob
    assert not any(key in r for r in plain_records for key in ("menu_psi", "menu_shares", "menu_psi_range"))
    # the checkpoint alone rebuilds the bounded head, and the frozen agent plans with it
    monkeypatch.setenv("SPECTRA_PLAN_MENU", MENU_FLAG)
    policy, loaded = plan_agent.load_policy(str(tmp_path / "b" / "policy_latest.pt"))
    assert policy.menu_bound == 4.0 and loaded["menu_bound"] == 4.0
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    env, _model, _plan = _origin(monkeypatch)
    hot = _hot_policy()
    with torch.no_grad():
        hot.menu_head.bias.mul_(30.0)                                                    # far past the bound
    bounded = plan_agent.PlanPolicy(63, menu=4, menu_bound=4.0)
    bounded.load_state_dict(hot.state_dict())
    path = _save(tmp_path, bounded, "bound", **RESIDUAL_META, **{**MENU_META, "menu_share_floor": 0.031},
                 menu_bound=4.0)
    raw_path = _save(tmp_path, hot, "raw", **RESIDUAL_META, **MENU_META)
    _seed_all(5)
    _w, info = plan_agent.plan_for_env(env, 0.58, path, kappa=0.6, budget="flops")
    _seed_all(5)
    _w, raw_info = plan_agent.plan_for_env(env, 0.58, raw_path, kappa=0.6, budget="flops")
    psi = info["menu"]["psi"]
    assert max(abs(v) for v in psi) <= 4.0 < max(abs(v) for v in raw_info["menu"]["psi"])
    assert psi == pytest.approx(plan_agent.menu_bounded(raw_info["menu"]["psi"], 4.0), abs=1e-5)
    assert info["menu"]["pi"] == plan_agent.menu_shares(psi, 0.031)
    # a warm start from a head under another bound is another agent
    meta = {**RESIDUAL_META, **MENU_META, "tokens": "layer", "zero": ["sens"], "k_min": 0.1, "budget": "mixed"}
    with pytest.raises(ValueError, match=r"trained with menu_bound=4.0; this run has None"):
        plan_trainer.warm_start(plan_agent.PlanPolicy(63, menu=4), plan_agent.read_blob(path),
                                {**meta, "menu_share_floor": 0.031}, path)


def test_entropy_bonus_spreads_the_mean_shares_and_anneals_to_zero(monkeypatch, tmp_path, v10_env):
    assert float(plan_agent.menu_entropy(torch.zeros(4))) == pytest.approx(math.log(4.0))
    assert float(plan_agent.menu_entropy(torch.tensor([30.0, 0.0, 0.0, 0.0]))) < 1e-6
    psi = torch.tensor([6.0, 0.0, 0.0, 0.0], requires_grad=True)
    p = torch.softmax(psi.detach(), dim=0)
    assert float(plan_agent.menu_entropy(psi)) == pytest.approx(float(-(p * p.log()).sum()), rel=1e-5)
    (-plan_agent.menu_entropy(psi)).backward()                                 # descent on -H lowers the leader's logit
    assert float(psi.grad[0]) > 0 > float(psi.grad[1])
    # in the trainer, with a reward that says nothing (every plan alike), the bonus is the only force on the head
    inst = _instance()
    inst.reward = lambda widths, proxy, batches: -3.0
    lines, records, blob = _go(monkeypatch, tmp_path, inst, SPECTRA_PLAN_MENU_ENT="0.5", **MENU_LINE)
    assert lines[0].endswith(" | av2: menu_entropy=0.5") and blob["menu_entropy"] == 0.5
    steps, cfg = _its(records), plan_trainer.config()
    bias = blob["state_dict"]["menu_head.bias"]
    assert float(bias[0]) < 6.0 and all(float(v) > 0.0 for v in bias[1:])      # the anchor down, the others up
    assert not bool(blob["state_dict"]["head.weight"].any())                   # no advantage: the scores' head stays
    assert steps[0]["menu_entropy"] == pytest.approx(float(plan_agent.menu_entropy(torch.tensor([6.0, 0.0, 0.0, 0.0]))),
                                                     rel=1e-4)
    assert steps[-1]["menu_entropy"] > steps[0]["menu_entropy"] and all(r["menu_sigma"] == 2.0 for r in steps)
    # beta_t falls to 0 over the sigma decay span (0.6 of 5 instances): the loss of a silent reward is -beta_t H
    for it, r in enumerate(steps):
        beta_t = 0.5 * max(0.0, 1.0 - it / (cfg.sigma_decay * cfg.instances))
        assert r["loss"] == pytest.approx(-beta_t * r["menu_entropy"], rel=1e-4, abs=1e-6)
    assert steps[3]["loss"] == 0.0 == steps[4]["loss"]


def test_menu_kappa_bins_and_film_read_the_budget_and_start_as_the_plain_head(monkeypatch, tmp_path, v10_env):
    spec = {"kind": "bins", "n": 5, "lo": 0.35, "hi": 0.85}
    torch.manual_seed(3)
    plain = plan_agent.PlanPolicy(63, menu=4)
    after = torch.get_rng_state()
    torch.manual_seed(3)
    bins = plan_agent.PlanPolicy(63, menu=4, menu_kappa=spec)
    assert torch.equal(torch.get_rng_state(), after)                           # the embedding's draw is put back
    torch.manual_seed(3)
    film = plan_agent.PlanPolicy(63, menu=4, menu_kappa={**spec, "kind": "film"})
    assert torch.equal(torch.get_rng_state(), after)
    base = plain.state_dict()
    assert set(bins.state_dict()) == set(base) | {"kappa_shift.weight"}
    assert set(film.state_dict()) == set(base) | {"kappa_shift.weight", "kappa_scale.weight"}
    assert all(torch.equal(bins.state_dict()[key], value) and torch.equal(film.state_dict()[key], value)
               for key, value in base.items())                                 # the encoder and both heads, as they were
    d = bins.encoder.output_dim
    assert bins.kappa_shift.weight.shape == (5, d) and float(bins.kappa_shift.weight.std()) == pytest.approx(1.0, abs=0.1)
    assert not bool(film.kappa_scale.weight.any()) and bins.menu_kappa == spec and plain.menu_kappa is None
    assert [plan_agent.kappa_bin(k, spec) for k in (0.11, 0.2, 0.36, 0.44, 0.46, 0.6, 0.74, 0.76, 0.84, 0.95)] == \
        [0, 0, 0, 0, 1, 2, 3, 4, 4, 4]                                         # outside the range: the nearest bin
    assert plan_agent.kappa_bin(0.5, {**spec, "hi": 0.35}) == 0
    state = toy_state(8)
    with torch.no_grad():
        for kappa in (0.2, 0.4, 0.6, 0.8, 0.95):                              # untrained: the bias, whatever the budget
            assert bins.menu_logits(state, [0.1, 0.0, -0.1, kappa]).tolist() == [6.0, 0.0, 0.0, 0.0]
            assert film.menu_logits(state, [0.1, 0.0, -0.1, kappa]).tolist() == [6.0, 0.0, 0.0, 0.0]
        bins.menu_head.weight.normal_(std=0.1)
        bins.menu_head.weight[:, -1] = 0.0                                      # the head's own kappa input, set aside
        film.load_state_dict({**bins.state_dict(), "kappa_scale.weight": film.kappa_scale.weight})
        plain.load_state_dict({key: value for key, value in bins.state_dict().items() if key in base})
        at = lambda policy, kappa: policy.menu_logits(state, [0.1, 0.0, -0.1, kappa])  # noqa: E731
        assert torch.equal(at(bins, 0.36), at(bins, 0.44)) and not torch.equal(at(bins, 0.44), at(bins, 0.80))
        assert torch.equal(at(plain, 0.36), at(plain, 0.80))                   # the plain head cannot tell them apart
        encoded = bins.encoder(state).mean(dim=0)
        want = bins.menu_head(torch.cat([encoded + bins.kappa_shift.weight[4], torch.tensor([0.1, 0.0, -0.1, 0.8])]))
        torch.testing.assert_close(at(bins, 0.80), want)
        assert torch.equal(at(film, 0.80), at(bins, 0.80))                     # the scale starts at 1
        film.kappa_scale.weight[4].fill_(0.5)
        want = film.menu_head(torch.cat([encoded * 1.5 + film.kappa_shift.weight[4],
                                         torch.tensor([0.1, 0.0, -0.1, 0.8])]))
        torch.testing.assert_close(at(film, 0.80), want)
        assert torch.equal(at(film, 0.44), at(bins, 0.44))
    bins.zero_grad()
    bins.menu_logits(state, [0.1, 0.0, -0.1, 0.8]).sum().backward()            # only the budget's own bin learns
    grad = bins.kappa_shift.weight.grad
    assert float(grad[4].abs().sum()) > 0 and not bool(grad[:4].any())
    for bad in ({"kind": "bins", "n": 1, "lo": 0.35, "hi": 0.85}, {"kind": "hist", "n": 5, "lo": 0.35, "hi": 0.85},
                {"kind": "bins", "n": 5, "lo": 0.9, "hi": 0.85}, {"kind": "bins", "n": 5}, "bins"):
        with pytest.raises(ValueError, match="menu_kappa"):
            plan_agent.PlanPolicy(63, menu=4, menu_kappa=bad)
    # the trainer: the bins span the budgets it draws, start on the prior, and the checkpoint rebuilds the head
    inst = _menu_instance()
    lines, records, blob = _go(monkeypatch, tmp_path / "bins", inst, SPECTRA_PLAN_MENU_KAPPA="bins:4",
                               SPECTRA_PLAN_KAPPA_POINTS="0.2:0.5:0.05", SPECTRA_PLAN_LR="0.05", **MENU_LINE)
    assert lines[0].endswith(" | av2: menu_kappa=bins:4@0.15-0.85 kappa_points=0.2:0.5:0.05")
    assert blob["menu_kappa"] == {"kind": "bins", "n": 4, "lo": pytest.approx(0.15), "hi": 0.85}
    assert blob["state_dict"]["kappa_shift.weight"].shape == (4, 256) and blob["menu_share_floor"] == 0.005
    assert _its(records)[0]["mean_plan"] == _its(records)[0]["refs"]["prior"]
    monkeypatch.setenv("SPECTRA_PLAN_MENU", MENU_FLAG)
    policy, _loaded = plan_agent.load_policy(str(tmp_path / "bins" / "policy_latest.pt"))
    assert policy.menu_kappa == blob["menu_kappa"] and _equal(dict(policy.state_dict()), blob["state_dict"])
    inst = _menu_instance()
    _l, _r, film_blob = _go(monkeypatch, tmp_path / "film", inst, SPECTRA_PLAN_MENU_KAPPA="film", **MENU_LINE)
    assert film_blob["menu_kappa"] == {"kind": "film", "n": 5, "lo": 0.35, "hi": 0.85}
    assert film_blob["state_dict"]["kappa_scale.weight"].shape == (5, 256)
    # the frozen agent: the bin of the budget the state carries
    monkeypatch.setattr(utils, "print_flush", lambda *a, **k: None)
    env, _model, _plan = _origin(monkeypatch)
    hot = plan_agent.PlanPolicy(63, menu=4, menu_kappa=spec)
    hot.load_state_dict({**_hot_policy().state_dict(), "kappa_shift.weight": hot.kappa_shift.weight.detach()})
    with torch.no_grad():
        hot.menu_head.weight[:, :256].normal_(std=0.05, generator=torch.Generator().manual_seed(2))
    path = _save(tmp_path, hot, "kappa", **RESIDUAL_META, **MENU_META, menu_kappa=spec)
    monkeypatch.setattr(plan_agent.NetInstance, "from_env", FROM_ENV)        # the jobs above left their stand-in
    _seed_all(5)
    inst = plan_agent.NetInstance.from_env(env, "toy.pt", None, None, zero=("sens",), with_cost=True, with_curves=True)
    loaded, _blob = plan_agent.load_policy(path)
    seen = {}
    for kappa in (0.20, 0.40, 0.80):                                # 0.20 is below the range the bins were made for
        _seed_all(5)
        _w, info = plan_agent.plan_for_env(env, 0.58, path, kappa=kappa, budget="flops")
        with torch.no_grad():
            want = loaded.menu_logits(inst.state_at(kappa, "flops"), info["menu"]["x"])
            pooled = loaded.encoder(inst.state_at(kappa, "flops")).mean(dim=0)
            by_hand = loaded.menu_head(torch.cat([pooled + loaded.kappa_shift.weight[0 if kappa < 0.45 else 4],
                                                  torch.tensor(info["menu"]["x"])]))
        assert info["menu"]["psi"] == want.tolist() and info["menu"]["x"][-1] == kappa
        torch.testing.assert_close(want, by_hand)                   # the bin of the budget the state carries
        seen[kappa] = want
    assert float((seen[0.40] - seen[0.80]).abs().max()) > 0.05


def test_menu_sigma_anneal_follows_the_scores_schedule(monkeypatch, tmp_path, v10_env):
    draws = _spy(monkeypatch)
    inst = _menu_instance()
    lines, records, blob = _go(monkeypatch, tmp_path, inst, SPECTRA_PLAN_MENU_SIGMA_ANNEAL="1", **MENU_LINE)
    assert lines[0].endswith(" | av2: menu_sigma_anneal=1") and blob["menu_sigma_anneal"] is True
    steps, cfg = _its(records), plan_trainer.config()
    assert steps[0]["menu_sigma"] == 2.0 and steps[-1]["menu_sigma"] == pytest.approx(2.0 * 0.4)
    for it, r in enumerate(steps):
        scores, logits = draws[2 * it], draws[2 * it + 1]
        assert logits["sigma"] == pytest.approx(2.0 * plan_trainer.sigma_at(cfg, it) / cfg.sigma) == r["menu_sigma"]
        assert logits["sigma"] == pytest.approx(2.0 * scores["sigma"] / 0.25)
        rewards = torch.tensor(r["rewards"], dtype=scores["mean"].dtype)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        want = float(-(adv * (plan_agent.log_prob(scores["z"], scores["mean"], scores["sigma"])
                              + plan_agent.log_prob(logits["z"], logits["mean"], logits["sigma"]))).mean())
        assert r["loss"] == pytest.approx(want, rel=1e-4, abs=1e-5)


# ------------------------------------------------------------------ rank 5: split credit


def test_split_credit_scores_each_half_against_its_own_siblings(monkeypatch, tmp_path, v10_env):
    draws = _spy(monkeypatch)
    inst = _menu_instance()
    lines, records, blob = _go(monkeypatch, tmp_path, inst, SPECTRA_PLAN_K="8", SPECTRA_PLAN_SPLIT="1",
                               SPECTRA_PLAN_MENU_LOG="1", SPECTRA_PLAN_LR="0.05", **MENU_LINE)
    assert lines[0].endswith(" | av2: split=1 menu_log=1") and blob["split"] is True
    steps, cfg, fm = _its(records), plan_trainer.config(), inst.cost_model("flops")
    assert len(draws) == 2 * len(steps) and all(d["k"] == 4 for d in draws)     # four score draws, four logit draws
    separate = 0
    for it, r in enumerate(steps):
        scores, logits = draws[2 * it], draws[2 * it + 1]
        mu, psi, z, u = scores["mean"], logits["mean"], scores["z"], logits["z"]
        g = torch.Generator().manual_seed(cfg.seed * 100003 + it)
        assert torch.equal(z, mu.unsqueeze(0) + scores["sigma"] * torch.randn((4, inst.n_groups), generator=g))
        assert torch.equal(u, psi.unsqueeze(0) + 2.0 * torch.randn((4, len(MENU)), generator=g))
        rules = inst.rules_at(MENU, r["kappa"], "flops", 0.5, cfg.k_min, 2)
        mean_shares = plan_agent.menu_shares(psi.tolist(), plan_agent.MENU_SHARE_FLOOR)
        decode = lambda s, pi: plan_agent.menu_decode(s, pi, rules, fm, r["kappa"], cfg.k_min, min_width=2,  # noqa: E731
                                                      zmax=0.3)
        # the first half: sampled scores at the shares the frozen agent plays; the second: the mean scores at sampled shares
        plans = ([decode(z[k].tolist(), mean_shares) for k in range(4)]
                 + [decode(mu.tolist(), plan_agent.menu_shares(u[k].tolist())) for k in range(4)])
        assert r["kept"] == [info["kept"] for _w, info in plans] and r["split"] == 4
        assert r["rewards"] == [inst.reward(widths, "cut", []) for widths, _i in plans]
        assert r["menu_shares"][:4] == [[round(p, 4) for p in mean_shares]] * 4
        assert r["menu_shares"][4:] == [[round(p, 4) for p in plan_agent.menu_shares(u[k].tolist())] for k in range(4)]
        rewards = torch.tensor(r["rewards"], dtype=mu.dtype)
        adv_z = (rewards[:4] - rewards[:4].mean()) / (rewards[:4].std() + 1e-6)   # each half against its own mean and SD
        adv_u = (rewards[4:] - rewards[4:].mean()) / (rewards[4:].std() + 1e-6)
        assert abs(float(adv_z.sum())) < 1e-4 and abs(float(adv_u.sum())) < 1e-4
        want = float(-(adv_z * plan_agent.log_prob(z, mu, scores["sigma"])).mean()
                     - (adv_u * plan_agent.log_prob(u, psi, 2.0)).mean())
        assert r["loss"] == pytest.approx(want, rel=1e-4, abs=1e-5)
        pooled = (rewards - rewards.mean()) / (rewards.std() + 1e-6)             # one advantage for both parts: not this
        other = float(-(pooled[:4] * plan_agent.log_prob(z, mu, scores["sigma"])).mean()
                      - (pooled[4:] * plan_agent.log_prob(u, psi, 2.0)).mean())
        separate += abs(want - other) > 1e-3
    assert separate >= 3
    assert steps[0]["mean_plan"] == steps[0]["refs"]["prior"]
    state = blob["state_dict"]
    assert float(state["head.weight"].abs().sum()) > 0 and float(state["menu_head.weight"].abs().sum()) > 0
    # each half's reward reaches only its own part: with the logit half silent the menu head does not move
    quiet = _instance()
    base = _reward(quiet)
    calls = {"n": 0}

    def batches(proxy, seed):  # the trainer asks for an instance's batches once, before its eight plans
        calls["n"] = 0
        return []

    def reward(widths, proxy, batches):
        calls["n"] += 1
        return -1.0 if 4 < calls["n"] <= 8 else base(widths, proxy, batches)   # plans 5 to 8: the logit half

    quiet.batches, quiet.reward = batches, reward
    _l, _r, half = _go(monkeypatch, tmp_path / "quiet", quiet, instances=4, SPECTRA_PLAN_K="8",
                       SPECTRA_PLAN_SPLIT="1", SPECTRA_PLAN_LR="0.05", **MENU_LINE)
    assert float(half["state_dict"]["head.weight"].abs().sum()) > 0
    assert not bool(half["state_dict"]["menu_head.weight"].any())
    assert half["state_dict"]["menu_head.bias"].tolist() == [6.0, 0.0, 0.0, 0.0]


# ------------------------------------------------------------------ rank 6: averaged weights and cosine decay


def test_policy_ema_tracks_the_weights_with_a_warm_up(v10_env):
    policy = plan_agent.PlanPolicy(63)
    ema = plan_trainer.PolicyEma(policy, 0.9)
    assert _equal(ema.shadow, dict(policy.state_dict())) and ema.updates == 0
    assert all(ema.shadow[key] is not value for key, value in policy.state_dict().items())   # a copy, not a view
    with torch.no_grad():
        policy.head.weight.fill_(1.0)
    ema.update(policy)                                                          # t = 1: decay min(0.9, 2 / 11)
    torch.testing.assert_close(ema.shadow["head.weight"], torch.full_like(policy.head.weight, 1.0 - 2.0 / 11.0))
    value = 1.0 - 2.0 / 11.0
    for t in range(2, 200):
        d = min(0.9, (1.0 + t) / (10.0 + t))
        value = d * value + (1.0 - d) * 1.0
        ema.update(policy)
    assert float(ema.shadow["head.weight"][0, 0]) == pytest.approx(value, rel=1e-5) and value > 0.999
    torch.testing.assert_close(ema.shadow["encoder.output_norm.weight"],            # a weight that stood still
                               policy.state_dict()["encoder.output_norm.weight"])
    other = plan_trainer.PolicyEma(plan_agent.PlanPolicy(63), 0.9)
    other.load(ema.state())
    assert other.updates == 199 and _equal(other.shadow, ema.shadow)
    with pytest.raises(ValueError, match="not this policy's"):
        plan_trainer.PolicyEma(plan_agent.PlanPolicy(63, menu=4), 0.9).load(ema.state())


def test_trainer_saves_the_averaged_policy_beside_the_last_iterate_and_decays_the_lr(monkeypatch, tmp_path, v10_env):
    inst = _menu_instance("synthetic")
    _clear(monkeypatch)
    lines, records = _job(monkeypatch, tmp_path, inst, SPECTRA_PLAN_EMA="0.9", SPECTRA_PLAN_LR_SCHEDULE="cosine")
    assert lines[0].endswith(" | av2: ema=0.9 lr_schedule=cosine")
    assert sorted(os.listdir(tmp_path)) == ["plan_train.jsonl", "policy_ema_it00006.pt", "policy_ema_latest.pt",
                                            "policy_it00002.pt", "policy_it00004.pt", "policy_it00006.pt",
                                            "policy_latest.pt"]
    last = plan_agent.read_blob(str(tmp_path / "policy_latest.pt"))
    mean = plan_agent.read_blob(str(tmp_path / "policy_ema_latest.pt"))
    assert last["ema"] == 0.9 and last["lr_schedule"] == "cosine" and "weights" not in last
    assert mean["weights"] == "ema" and mean["ema_updates"] == 3 and mean["ema"] == 0.9
    assert {key: value for key, value in mean.items() if key not in ("state_dict", "weights", "ema_updates")} == \
        {key: value for key, value in last.items() if key != "state_dict"}
    assert _equal(plan_agent.read_blob(str(tmp_path / "policy_ema_it00006.pt")), mean)
    assert set(mean["state_dict"]) == set(last["state_dict"]) and not _equal(mean["state_dict"], last["state_dict"])
    # the average, by hand from the three iterates (the heads start at zero): decay min(0.9, (1 + t) / (10 + t))
    for key in ("head.weight", "head.bias"):
        shadow = torch.zeros_like(last["state_dict"][key])
        for t, name in enumerate(("policy_it00002.pt", "policy_it00004.pt", "policy_it00006.pt"), start=1):
            d = min(0.9, (1.0 + t) / (10.0 + t))
            shadow = d * shadow + (1.0 - d) * _weights(tmp_path / name)[key]
        torch.testing.assert_close(mean["state_dict"][key], shadow, rtol=1e-5, atol=1e-8)
    policy, _blob = plan_agent.load_policy(str(tmp_path / "policy_ema_latest.pt"))          # an ordinary checkpoint
    assert _equal(dict(policy.state_dict()), mean["state_dict"])
    # cosine: three updates of six instances in batches of two, lr (1 + cos(pi t / 3)) / 2
    cfg = plan_trainer.config()
    assert [plan_trainer.lr_at(cfg, it) for it in (1, 3, 5)] == pytest.approx([cfg.lr, 0.75 * cfg.lr, 0.25 * cfg.lr])
    steps = _steps(records)
    assert [r.get("lr") for r in steps] == [None, pytest.approx(cfg.lr), None, pytest.approx(0.75 * cfg.lr), None,
                                            pytest.approx(0.25 * cfg.lr)]
    # without the flags: no averaged file, and the same first update (the schedule starts at the full lr)
    plain = _menu_instance("synthetic")
    _clear(monkeypatch)
    _lines, plain_records = _job(monkeypatch, tmp_path / "plain", plain)
    assert sorted(os.listdir(tmp_path / "plain")) == ["plan_train.jsonl", "policy_it00002.pt", "policy_it00004.pt",
                                                      "policy_it00006.pt", "policy_latest.pt"]
    assert _equal(_weights(tmp_path / "policy_it00002.pt"), _weights(tmp_path / "plain" / "policy_it00002.pt"))
    assert not _equal(_weights(tmp_path / "policy_it00004.pt"), _weights(tmp_path / "plain" / "policy_it00004.pt"))
    assert all("lr" not in r for r in plain_records)
    with pytest.raises(ValueError, match="not this policy's"):
        plan_agent.save_weights(plan_agent.PlanPolicy(63), {"head.weight": torch.zeros(1)}, str(tmp_path / "x.pt"))


def test_a_run_with_averaged_weights_and_the_running_scale_resumes_where_it_stopped(monkeypatch, tmp_path, v10_env):
    knobs = {"SPECTRA_PLAN_STATE": "1", "SPECTRA_PLAN_EMA": "0.9", "SPECTRA_PLAN_LR_SCHEDULE": "cosine",
             "SPECTRA_PLAN_ADV": "global:0.9", "SPECTRA_PLAN_MIRROR": "1", "SPECTRA_PLAN_GNORM": "sqrt"}
    inst = _menu_instance("synthetic")
    _clear(monkeypatch)
    _lines, full_records = _job(monkeypatch, tmp_path / "full", inst, **knobs)
    _clear(monkeypatch)
    a_lines, a_records = _job(monkeypatch, tmp_path / "a", inst, SPECTRA_PLAN_MAX_MINUTES="1e-9", **knobs)
    assert "[plan] MAX_MINUTES=1e-09 reached after 2 instances" in a_lines
    state = tmp_path / "a" / "trainer_state.pt"
    kept = plan_agent.read_blob(str(state))
    assert set(kept) == set(plan_trainer.STATE_KEYS) | {"av2"} and kept["done"] == 2
    assert kept["av2"]["ema"]["updates"] == 1 and kept["av2"]["adv_scale"]["steps"] == 2
    assert kept["meta"]["ema"] == 0.9 and kept["meta"]["adv"] == "global:0.9" and kept["meta"]["mirror"] is True
    _clear(monkeypatch)
    b_lines, b_records = _job(monkeypatch, tmp_path / "b", inst, SPECTRA_PLAN_RESUME=str(state), **knobs)
    assert f"[plan] resumed from {state} at 2 instances" in b_lines
    assert _steps(a_records) + _steps(b_records) == _steps(full_records)
    for name in ("policy_latest.pt", "policy_ema_latest.pt", "policy_ema_it00006.pt"):
        assert _equal(_weights(tmp_path / "b" / name), _weights(tmp_path / "full" / name)), name
    sb = plan_agent.read_blob(str(tmp_path / "b" / "trainer_state.pt"))
    sf = plan_agent.read_blob(str(tmp_path / "full" / "trainer_state.pt"))
    assert _equal(sb["av2"], sf["av2"]) and _equal(sb["optimizer"], sf["optimizer"]) and sb["av2"]["ema"]["updates"] == 3
    assert not _equal(_weights(tmp_path / "a" / "policy_ema_latest.pt"), _weights(tmp_path / "full" / "policy_ema_latest.pt"))
    # a chained job under other flags is another run
    for change, text in (({"SPECTRA_PLAN_EMA": "0.5"}, r"ema=0.9 \(here 0.5\)"),
                         ({"SPECTRA_PLAN_ADV": "rank"}, r"adv='global:0.9' \(here 'rank'\)"),
                         ({"SPECTRA_PLAN_MIRROR": "0"}, r"mirror=True \(here None\)")):
        _clear(monkeypatch)
        with pytest.raises(ValueError, match=text):
            _job(monkeypatch, tmp_path / "c", inst, SPECTRA_PLAN_RESUME=str(state), **{**knobs, **change})
    # a state written without the flags cannot feed a run with them
    with pytest.raises(ValueError, match="carries no ema"):
        plan_trainer.restore_av2({"done": 2}, "/s.pt", plan_trainer.PolicyEma(plan_agent.PlanPolicy(63), 0.9))


# ------------------------------------------------------------------ rank 8: the group count


def test_gnorm_divides_the_scores_term_by_the_group_count(monkeypatch, tmp_path, v10_env):
    draws = _spy(monkeypatch)
    for mode, div in (("sqrt", math.sqrt), ("linear", float)):
        del draws[:]
        inst = _menu_instance()
        lines, records, blob = _go(monkeypatch, tmp_path / mode, inst, SPECTRA_PLAN_GNORM=mode, **T1)
        assert lines[0].endswith(f" | av2: gnorm={mode}") and blob["gnorm"] == mode
        assert len(draws) == len(_its(records)) == 5
        for r, d in zip(_its(records), draws):
            rewards = torch.tensor(r["rewards"], dtype=d["mean"].dtype)
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            full = float(-(adv * plan_agent.log_prob(d["z"], d["mean"], d["sigma"])).mean())
            assert r["loss"] == pytest.approx(full / div(inst.n_groups), rel=1e-4, abs=1e-6)
    # under the menu only the scores' term is divided: the logits' is K wide on every network
    del draws[:]
    inst = _menu_instance()
    _l, records, _b = _go(monkeypatch, tmp_path / "menu", inst, SPECTRA_PLAN_GNORM="sqrt", **MENU_LINE)
    for it, r in enumerate(_its(records)):
        scores, logits = draws[2 * it], draws[2 * it + 1]
        rewards = torch.tensor(r["rewards"], dtype=scores["mean"].dtype)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        want = float(-(adv * (plan_agent.log_prob(scores["z"], scores["mean"], scores["sigma"])
                              / math.sqrt(inst.n_groups)
                              + plan_agent.log_prob(logits["z"], logits["mean"], 2.0))).mean())
        assert r["loss"] == pytest.approx(want, rel=1e-4, abs=1e-6)


# ------------------------------------------------------------------ ranks 9 and 12: what the trainer draws


def test_kappa_points_draw_part_of_the_budgets_near_the_registered_points(monkeypatch, tmp_path, v10_env):
    cfg = plan_trainer.config()
    rng = random.Random(3)
    near = [plan_trainer.draw_kappa(rng, cfg, ((0.45, 0.2), 1.0, 0.03)) for _ in range(200)]
    assert all(min(abs(k - 0.45), abs(k - 0.2)) <= 0.03 + 1e-12 for k in near)
    assert 60 < sum(abs(k - 0.2) <= 0.03 for k in near) < 140                   # both points are drawn
    rng = random.Random(3)
    mixed = [plan_trainer.draw_kappa(rng, cfg, ((0.2,), 0.5, 0.03)) for _ in range(400)]
    share = sum(abs(k - 0.2) <= 0.03 for k in mixed) / 400.0
    assert 0.4 < share < 0.6 and all(abs(k - 0.2) <= 0.03 or 0.35 <= k <= 0.85 for k in mixed)
    assert all(0.02 <= plan_trainer.draw_kappa(rng, cfg, ((0.03,), 1.0, 0.4)) <= 0.98 for _ in range(50))
    inst = _menu_instance()
    lines, records, blob = _go(monkeypatch, tmp_path, inst, SPECTRA_PLAN_KAPPA_POINTS="0.2:1:0.03", instances=8, **T1)
    assert lines[0].endswith(" | av2: kappa_points=0.2:1:0.03") and blob["kappa_points"] == [[0.2], 1.0, 0.03]
    steps = _its(records)
    assert len(steps) == 8 and all(0.17 <= r["kappa"] <= 0.23 for r in steps)
    rng = random.Random(cfg.seed)                                               # the trainer's own generator
    for r in steps:
        assert rng.choice([inst]) is inst
        assert r["kappa"] == plan_trainer.draw_kappa(rng, cfg, ((0.2,), 1.0, 0.03))


def test_family_sampling_draws_families_not_networks(monkeypatch, tmp_path, v10_env):
    names = {"densenet40_cifar10_densenet-cifar_93.17_0.176_74.43.pt": "densenet",
             "mobilenet-v2x0.5_cifar100_chenyaofo_70.88_0.82_56.16.pt": "mobilenet-v2",
             "mobilenet-v2x1.4_cifar100_chenyaofo_75.98_4.5_340.46.pt": "mobilenet-v2",
             "preact-resnet56_cifar10_spectra-w1_93.23_0.856_126.28.pt": "preact-resnet",
             "repvgga0_cifar100_chenyaofo_75.22_7.96_978.38.pt": "repvgga",
             "repvgga2_cifar10_chenyaofo_94.98_26.82_3700.2.pt": "repvgga",
             "resnet20-width10_cifar100_thin-res-net_64.77_0.111_16.55.pt": "thin-resnet20",
             "resnet56-width6_cifar10_thin-res-net_92.88_0.122_18.60.pt": "thin-resnet56",
             "resnet20_cifar10_chenyaofo_92.6_0.27_81.62.pt": "resnet",
             "resnet56_cifar100_chenyaofo_72.63_0.86_251.5.pt": "resnet",
             "vgg11_bn_cifar100_chenyaofo_70.78_9.8_306.68.pt": "vgg",
             "vgg19_bn_cifar10_chenyaofo_93.91_20.57_797.32.pt": "vgg",
             "wrn-16-4_cifar10_spectra-w1_93.88_2.749_393.09.pt": "wrn",
             "wrn-28-2_cifar100_spectra-w1_71.20_1.479_214.84.pt": "wrn",
             "shufflenetv2x1.5_cifar100_chenyaofo_73.91_2.58_188.7.pt": "shufflenetv2"}
    for name, family in names.items():
        assert fortify.plan_family(name) == family == fortify.plan_family("/some/dir/" + name), name
    nets = [types.SimpleNamespace(name=name) for name in ("resnet20-width4_a.pt", "resnet20-width6_b.pt",
                                                          "resnet20-width10_c.pt", "vgg11_bn_d.pt")]
    families, weights = plan_trainer.family_weights(nets, 1.0)
    assert families == ["thin-resnet20"] * 3 + ["vgg"] and weights == pytest.approx([1 / 6, 1 / 6, 1 / 6, 0.5])
    assert plan_trainer.family_weights(nets, 0.0)[1] == pytest.approx([0.25] * 4)
    half = plan_trainer.family_weights(nets, 0.5)[1]
    assert sum(half) == pytest.approx(1.0) and 0.25 < half[3] < 0.5
    # the trainer on three toy networks, two of one family: its own generator draws by the weights
    by_path = {}
    for name in ("resnet20-width4_cifar10_a.pt", "resnet20-width6_cifar10_b.pt", "vgg11_bn_cifar10_c.pt"):
        inst = _toy_instance()
        inst.name, inst.reward = name, _reward(inst)
        by_path[name] = inst
    _clear(monkeypatch)
    _seed_all()
    monkeypatch.setattr(plan_agent.NetInstance, "from_env", classmethod(lambda cls, env, p, m, l, **kw: by_path[p]))
    monkeypatch.setattr(plan_trainer, "out_dir", lambda: str(tmp_path))
    for key, value in {"SPECTRA_PLAN_INSTANCES": "40", "SPECTRA_PLAN_K": "2", "SPECTRA_PLAN_PROXY": "cut",
                       "SPECTRA_PLAN_REF_EVERY": "1000", "SPECTRA_PLAN_SAVE_EVERY": "1000",
                       "SPECTRA_PLAN_SUMMARY_KAPPAS": "0.6", "SPECTRA_PLAN_SAMPLE": "family"}.items():
        monkeypatch.setenv(key, value)
    lines = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: lines.append(str(s)))
    plan_trainer.run(None, [(name, (None, None)) for name in by_path])
    assert lines[0].endswith(" | av2: sample=family:1")
    assert "[plan] sample family t=1: thin-resnet20 2 nets, 0.500, vgg 1 nets, 0.500" in lines
    steps = _its([json.loads(text) for text in open(tmp_path / "plan_train.jsonl", encoding="utf-8")])
    cfg, rng, order = plan_trainer.config(), random.Random(0), list(by_path)
    for r in steps:
        assert r["net"] == rng.choices(order, weights=[0.25, 0.25, 0.5])[0]
        assert r["kappa"] == rng.uniform(cfg.kappa_lo, cfg.kappa_hi)
    assert len(steps) == 40 and 9 <= sum(r["net"].startswith("vgg") for r in steps) <= 31   # half, not a third
    assert plan_agent.read_blob(str(tmp_path / "policy_latest.pt"))["sample"] == "family:1"


# ------------------------------------------------------------------ rank 14: the fine-tune's divergence guard


def _fs_batches(n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(16, *SHAPE, generator=g), torch.randint(0, 10, (16,), generator=g)) for _ in range(n)]


def test_fs_guard_flags_a_diverged_fine_tune_and_leaves_a_sound_one_alone(monkeypatch, v10_env):
    inst = _toy_instance()
    make = lambda: inst.cutter.cut({})  # noqa: E731 - the origin's weights on the working copy
    state = lambda model: {key: value.clone() for key, value in model.state_dict().items()}  # noqa: E731
    assert plan_agent.short_finetune(make(), _fs_batches(), 0.01) is None       # off: as before, nothing returned
    plain = state(inst.cutter.work)
    assert plan_agent.short_finetune(make(), _fs_batches(), 0.01, guard=1.0) is False
    assert _equal(state(inst.cutter.work), plain)                               # a sound fine-tune is the same steps
    work = make()
    for p in work.parameters():
        p.requires_grad = False
    assert plan_agent.short_finetune(work, _fs_batches(), 1e9, guard=1.0) is True      # the loss or the weights blow up
    assert not work.training and not any(p.requires_grad for p in work.parameters())   # left as a sound one is
    for p in work.parameters():
        p.requires_grad = True
    # above 1: a loss past that many times the first step's stops the fine-tune before its update
    work = make()
    assert plan_agent.short_finetune(work, _fs_batches(), 50.0, guard=1.0001) is True
    assert all(bool(torch.isfinite(p).all()) for p in work.parameters())        # caught while the weights are finite
    # the reward: the flag is read per call, and says whether this plan's fine-tune diverged
    widths, _info = plan_agent.decode([0.1 * k for k in range(inst.n_groups)], inst.pm, 0.6)
    batches = inst.batches("fs2", seed=3)
    r_off = inst.reward(widths, "fs2", batches)
    assert inst.diverged is False
    monkeypatch.setenv("SPECTRA_PLAN_FS_GUARD", "1")
    assert inst.reward(widths, "fs2", batches) == r_off and inst.diverged is False
    monkeypatch.setenv("SPECTRA_PLAN_FS_GUARD", "10")                           # two steps rarely reach inf: the ratio
    monkeypatch.setenv("SPECTRA_PLAN_FS_LR", "1e9")
    inst.reward(widths, "fs2", batches)
    assert inst.diverged is True
    monkeypatch.setenv("SPECTRA_PLAN_FS_LR", "0.01")
    assert inst.reward(widths, "fs2", batches) == r_off and inst.diverged is False


def test_trainer_gives_a_diverged_plan_the_median_of_its_instance(monkeypatch, tmp_path, v10_env):
    draws = _spy(monkeypatch)
    inst = _toy_instance()
    base = _reward(inst)
    lost = lambda widths: sum(widths.values()) % 3 == 0  # noqa: E731 - which plans' fine-tunes "diverge"

    def reward(widths, proxy, batches):
        inst.diverged = lost(widths)
        return -90.0 if inst.diverged else base(widths, proxy, batches)

    inst.reward = reward
    monkeypatch.setattr(inst, "batches", lambda proxy, seed: [])
    lines, records, blob = _go(monkeypatch, tmp_path, inst, budget="params", instances=12, SPECTRA_PLAN_K="6",
                               SPECTRA_PLAN_PROXY="fs2", SPECTRA_PLAN_FS_GUARD="1", SPECTRA_PLAN_NORM_ADV="0")
    assert lines[0].endswith(" | av2: fs_guard=1") and blob["fs_guard"] == 1.0
    steps, cfg, hit = _its(records), plan_trainer.config(), 0
    for r, d in zip(steps, draws):
        plans = [plan_agent.decode(d["z"][k].tolist(), inst.pm, r["kappa"], cfg.k_min)[0] for k in range(6)]
        gone = [lost(widths) for widths in plans]
        sound = [base(widths, "fs2", []) for widths, bad in zip(plans, gone) if not bad]
        fill = statistics.median(sound) if sound else 0.0
        assert r["diverged"] == sum(gone)
        assert r["rewards"] == [fill if bad else base(widths, "fs2", []) for widths, bad in zip(plans, gone)]
        assert -90.0 not in r["rewards"]
        hit += sum(gone)
    assert 0 < hit < 6 * len(steps)                                             # some diverged, not all
    assert sum(" | diverged " in line for line in lines) == sum(r["diverged"] > 0 for r in steps)


# ------------------------------------------------------------------ rank 16: the masked-vs-real gate


def test_check_gate_passes_a_sound_network_and_leaves_the_run_that_follows_unchanged(monkeypatch, tmp_path, v10_env):
    def job(name, **flags):
        inst = _instance()
        real = inst.reward

        def reward(widths, proxy, batches):  # the cut's own reading; the training reads also draw from each generator
            value = real(widths, proxy, batches)
            return value if proxy == "cut" else value + random.random() + float(torch.rand(())) + float(np.random.rand())

        inst.reward = reward
        _clear(monkeypatch)
        lines, records = _job(monkeypatch, tmp_path / name, inst, SPECTRA_PLAN_BUDGET="flops", **T1, **flags)
        return inst, lines, records

    _inst, off_lines, off_records = job("off")
    inst, lines, records = job("on", SPECTRA_PLAN_CHECK="1")
    assert lines[0].endswith(" | av2: check=warn")
    gate = [line for line in lines if line.startswith("[plan] check3 ")]
    assert len(gate) == 1 and re.fullmatch(
        r"\[plan\] check3 toy\.pt b=flops: even k=0\.60 kept analytic x(\d\.\d{4}) real x\1 cut val Δ masked "
        r"([-+]\d+\.\d\d) real \2 \| prior k=0\.60 kept analytic x(\d\.\d{4}) real x\3 cut val Δ masked "
        r"([-+]\d+\.\d\d) real \4 \| deep sample k=0\.35 kept analytic x(\d\.\d{4}) real x\5 cut val Δ masked "
        r"([-+]\d+\.\d\d) real \6 \| dead filters by group none \| ok", gate[0]), gate
    assert "[plan] check gate: PASS (1 networks, three plans each)" in lines
    assert lines.index(gate[0]) == lines.index(next(x for x in lines if x.startswith("[plan] check-flops "))) + 1
    # the three real cuts draw from the global generators and are put back: the run is the run without the flag
    assert _steps(records) == _steps(off_records)
    assert _equal(_weights(tmp_path / "on" / "policy_latest.pt"), _weights(tmp_path / "off" / "policy_latest.pt"))
    assert sum(abs(r["loss"]) > 1e-3 for r in _steps(records)) >= 3
    rest = [line for line in lines[1:] if not line.startswith(("[plan] check3 ", "[plan] check gate"))]
    assert [_timeless(x, tmp_path / "on") for x in rest] == [_timeless(x, tmp_path / "off") for x in off_lines[1:]]
    # the pieces: one plan cut for real on either budget, and the dead filters per group
    widths, info = inst.reference("uniform", 0.6, budget="flops", min_width=2)
    chk = inst.check_widths(widths, "flops")
    assert chk["analytic"] == pytest.approx(info["kept"]) and chk["real"] == pytest.approx(info["kept"], abs=1e-9)
    assert chk["masked_r"] == pytest.approx(chk["real_r"], abs=1e-6)
    widths, info = inst.reference("uniform", 0.6)
    assert inst.check_widths(widths)["real"] == pytest.approx(inst.check()["real"], abs=1e-12)
    assert inst.dead_by_group() == {}
    group, row = next((g, r) for g, r in inst.cutter.plan if len(g.producers) == 1 and not g.depthwise)
    with torch.no_grad():
        group.producers[0].weight[:2] = 0
    inst.cutter.origin = {key: value.clone() for key, value in inst.cutter.work.state_dict().items()}
    assert inst.dead_by_group() == {int(row): 2}


def test_check_gate_reports_a_mismatch_and_refuse_stops_the_run(monkeypatch, tmp_path, v10_env):
    def broken():
        inst = _instance()
        inst.reward = _reward(inst)          # a masked reading that is not the real cut's: the fault the gate is for
        return inst

    _clear(monkeypatch)
    lines = _prepare(monkeypatch, tmp_path / "refuse", broken(), SPECTRA_PLAN_BUDGET="flops",
                     SPECTRA_PLAN_CHECK="refuse", **T1)
    with pytest.raises(RuntimeError, match=r"SPECTRA_PLAN_CHECK=refuse: .* does not match the real cut for 3 plans: "
                                           r"toy\.pt even k=0\.60 flops; toy\.pt prior k=0\.60 flops; "
                                           r"toy\.pt deep sample k=0\.35 flops"):
        plan_trainer.run(None, SHARD)
    gate = [line for line in lines if line.startswith("[plan] check3 ")]
    assert len(gate) == 1 and gate[0].endswith(" | WARNING: the masked reward or the analytic size does not match the "
                                               "real cut (even k=0.60, prior k=0.60, deep sample k=0.35)")
    assert any(line.startswith("[plan] check gate: FAIL (1 networks, three plans each: toy.pt even k=0.60 flops; ")
               for line in lines)
    assert os.listdir(tmp_path / "refuse") == [] and not any(line.startswith("[plan] it=") for line in lines)
    # warn: the same verdict line, and the run goes on
    _clear(monkeypatch)
    lines, records = _job(monkeypatch, tmp_path / "warn", broken(), SPECTRA_PLAN_BUDGET="flops",
                          SPECTRA_PLAN_CHECK="warn", **T1)
    assert any(line.startswith("[plan] check gate: FAIL ") for line in lines) and len(_steps(records)) == 6
    assert plan_agent.read_blob(str(tmp_path / "warn" / "policy_latest.pt"))["check"] == "warn"
    # without a residual the second plan is the sens rule's; on a params budget the size is the parameter count
    _clear(monkeypatch)
    lines, _records = _job(monkeypatch, tmp_path / "params", _instance(), SPECTRA_PLAN_CHECK="refuse")
    gate = [line for line in lines if line.startswith("[plan] check3 ")]
    assert len(gate) == 1 and gate[0].startswith("[plan] check3 toy.pt: even k=0.60 kept analytic x")
    assert " | sens k=0.60 kept analytic x" in gate[0] and gate[0].endswith(" | ok")
    assert "[plan] check gate: PASS (1 networks, three plans each)" in lines


# ------------------------------------------------------------------ rank 17: the state contract


def test_state_contract_is_written_under_the_flag_and_a_mismatch_is_refused(monkeypatch, tmp_path, v10_env):
    spans = plan_agent.spans_of(LAYOUT)
    contract = plan_agent.state_contract(spans)
    assert contract["layout"] == [list(entry) for entry in LAYOUT]
    assert {key: contract[key] for key in ("fortify", "budget_in_state", "state_slack", "state_groupcost",
                                           "fixed_target", "state_sens")} == dict.fromkeys(
        ("fortify", "budget_in_state", "state_slack", "state_groupcost", "fixed_target", "state_sens"), 1)
    assert (contract["stem_rows"], contract["min_width_for_prune"], contract["state_tokens"]) == (1, 2, "layers")
    assert (contract["state_align"], contract["probe_batches"], contract["sens_bn_recal"]) == ("prev", 2, 0)
    assert contract == {"layout": contract["layout"], **fortify.plan_state_contract()}
    # the trainer writes it only under the flag
    _l, _r, off = _go(monkeypatch, tmp_path / "off", _menu_instance(), **T1)
    assert "state_contract" not in off
    lines, _r, on = _go(monkeypatch, tmp_path / "on", _menu_instance(), SPECTRA_PLAN_CONTRACT="1", **T1)
    assert lines[0].endswith(" | av2: contract=1") and on["state_contract"] == contract
    _l, _r, warn = _go(monkeypatch, tmp_path / "warn", _menu_instance(), SPECTRA_PLAN_CONTRACT="warn", **T1)
    assert "state_contract" not in warn                                         # warn is the reader's word, not the writer's
    path = str(tmp_path / "on" / "policy_latest.pt")
    blob = plan_agent.read_blob(path)
    assert plan_agent.check_contract(blob) == [] == plan_agent.check_contract(blob, spans)
    assert plan_agent.check_contract(off) == [] == plan_agent.check_contract(off, {"base": (0, 5)})   # none recorded
    # the frozen agent on a job under the same contract plans as ever
    _clear(monkeypatch)
    out = []
    monkeypatch.setattr(utils, "print_flush", lambda s, *a, **k: out.append(str(s)))
    env, _model, _plan = _origin(monkeypatch)
    _seed_all(5)
    widths, info = plan_agent.plan_for_env(env, 0.58, path, kappa=0.6, budget="flops")
    _seed_all(5)
    assert plan_agent.plan_for_env(env, 0.58, str(tmp_path / "off" / "policy_latest.pt"), kappa=0.6,
                                   budget="flops")[0] == widths
    # a job whose walk contract differs is refused before anything is measured, each difference named
    with monkeypatch.context() as trap:
        trap.setattr(group_sensitivity, "calibration_batches", lambda *a, **k: pytest.fail("refused before measuring"))
        trap.setenv("SPECTRA_STEM_ROWS", "0")
        with pytest.raises(ValueError, match=r"the plan agent at .*policy_latest\.pt was trained under another state "
                                             r"contract: stem_rows=1 \(here 0\)\. Pin the walk actor"):
            plan_agent.plan_for_env(env, 0.58, path, kappa=0.6, budget="flops")
        trap.setenv("SPECTRA_STATE_SLACK", "0")
        trap.setenv("SPECTRA_SENS_BN_RECAL", "4")
        with pytest.raises(ValueError,
                           match=r"sens_bn_recal=0 \(here 4\); state_slack=1 \(here 0\); stem_rows=1 \(here 0\)"):
            plan_agent.plan_for_env(env, 0.58, path, kappa=0.6, budget="flops")
    # ... while a checkpoint that records no contract is planned as before, whatever the job's flags
    monkeypatch.setenv("SPECTRA_STEM_ROWS", "0")
    _seed_all(5)
    assert plan_agent.plan_for_env(env, 0.58, str(tmp_path / "off" / "policy_latest.pt"), kappa=0.6,
                                   budget="flops")[0] == widths
    # warn: the differences are printed and the plan is made
    monkeypatch.setenv("SPECTRA_PLAN_CONTRACT", "warn")
    del out[:]
    _seed_all(5)
    assert plan_agent.plan_for_env(env, 0.58, path, kappa=0.6, budget="flops")[0] == widths
    warned = [line for line in out if "another state contract" in line]
    assert len(warned) == 1 and warned[0].startswith("[alloc] WARNING: the plan agent at ")
    assert "stem_rows=1 (here 0)" in warned[0]
    monkeypatch.delenv("SPECTRA_PLAN_CONTRACT")
    monkeypatch.delenv("SPECTRA_STEM_ROWS")
    # the layout itself: the same width with its channels in another order is another state
    swapped = plan_agent.spans_of([("base", 0, 38), ("fortify", 38, 42), ("budget", 42, 43), ("slack", 43, 45),
                                   ("groupcost", 45, 49), ("sens", 49, 51), ("target", 51, 53), ("action", 53, 63)])
    with pytest.raises(ValueError, match=r"another state contract: layout=\[\['base', 0, 38\]"):
        plan_agent.check_contract(blob, swapped, "the plan agent at x.pt")
    assert plan_agent.check_contract(blob, spans) == []
    # a warm start and a chained job compare contracts too
    policy = plan_agent.PlanPolicy(63)
    meta = {key: value for key, value in blob.items() if key not in ("state_dict", "feature_dim", "encoder")}
    plan_trainer.warm_start(policy, blob, meta, path)
    plan_trainer.warm_start(policy, blob, {key: value for key, value in meta.items() if key != "state_contract"}, path)
    with pytest.raises(ValueError, match=r"SPECTRA_PLAN_INIT=.* another state contract: stem_rows=1 \(here 3\)"):
        plan_trainer.warm_start(policy, blob, {**meta, "state_contract": {**contract, "stem_rows": 3}}, path)
    with pytest.raises(ValueError, match="state_contract="):
        plan_trainer.check_resume({**meta, "config": {}}, {**meta, "config": {}, "state_contract": None}, "/s.pt")
