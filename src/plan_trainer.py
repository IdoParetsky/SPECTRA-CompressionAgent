"""
Plan-agent trainer (``docs/LEARNING_PROGRAM_OCT8.md`` §5). ``SPECTRA_PLAN_TRAIN=1`` (default off) runs it in
place of the TEST loop, on the job's test networks.

An instance is (net, κ), κ ~ U[κ_lo, κ_hi]. The policy scores the net's groups; K plans z_k = μ + σ ε_k are
decoded to κ, and each is cut once and read through the reward proxy on the val half. All K share one set
of recalibration batches (common random numbers). The update is REINFORCE with the K-plan mean as the
baseline (POMO), A_k = r_k − mean(r), optionally over the K-plan std. Every ``REF_EVERY`` instances the
mean plan and the uniform / sens / inner plans are scored on the same batches; their gaps are the learning
monitors. Nothing here is a TEST: the val half selects nothing and the TEST half is never read.

Knobs, ``SPECTRA_PLAN_<NAME>``: INSTANCES (300), K (8), SIGMA (0.5), SIGMA_FLOOR (0.2), SIGMA_DECAY (0.6, the
share of the instances over which σ falls linearly to the floor), LR (3e-4), BATCH (4 instances per update),
KAPPA (0.35,0.85), KMIN (0.1), PROXY (bn32), TOKENS (layer), ENCODER (transformer), ZERO (channels set to 0,
e.g. sens), NETS (name substrings to keep; all when unset), REF_EVERY (10), SAVE_EVERY (50), SEED (0),
NORM_ADV (1), MAX_MINUTES (0 = no limit; the trainer saves and stops when it is reached), SUMMARY_KAPPAS
(0.4,0.6,0.8: the κ of the closing per-net summary).
"""

import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass

import torch

import src.utils as utils
from src import plan_agent

REFERENCES = ("uniform", "sens", "inner")


def enabled() -> bool:
    return os.environ.get("SPECTRA_PLAN_TRAIN", "0").strip().lower() in ("1", "true", "yes", "on")


def _env(name, default):
    return os.environ.get(f"SPECTRA_PLAN_{name}", default).strip()


@dataclass
class Config:
    instances: int
    k: int
    sigma: float
    sigma_floor: float
    sigma_decay: float
    lr: float
    batch: int
    kappa_lo: float
    kappa_hi: float
    k_min: float
    proxy: str
    tokens: str
    encoder: str
    zero: tuple
    nets: tuple
    ref_every: int
    save_every: int
    seed: int
    norm_adv: bool
    max_minutes: float
    summary_kappas: tuple


def config() -> Config:
    lo, hi = (float(v) for v in _env("KAPPA", "0.35,0.85").split(","))
    cfg = Config(
        instances=int(_env("INSTANCES", "300")), k=int(_env("K", "8")), sigma=float(_env("SIGMA", "0.5")),
        sigma_floor=float(_env("SIGMA_FLOOR", "0.2")), sigma_decay=float(_env("SIGMA_DECAY", "0.6")),
        lr=float(_env("LR", "3e-4")), batch=max(1, int(_env("BATCH", "4"))), kappa_lo=lo, kappa_hi=hi,
        k_min=float(_env("KMIN", "0.1")), proxy=_env("PROXY", "bn32").lower(), tokens=_env("TOKENS", "layer").lower(),
        encoder=_env("ENCODER", "transformer").lower(),
        zero=tuple(s.strip() for s in _env("ZERO", "").split(",") if s.strip()),
        nets=tuple(s.strip() for s in _env("NETS", "").split(",") if s.strip()),
        ref_every=max(1, int(_env("REF_EVERY", "10"))), save_every=max(1, int(_env("SAVE_EVERY", "50"))),
        seed=int(_env("SEED", "0")), norm_adv=_env("NORM_ADV", "1") not in ("0", "false", "no", "off"),
        max_minutes=float(_env("MAX_MINUTES", "0")),
        summary_kappas=tuple(float(v) for v in _env("SUMMARY_KAPPAS", "0.4,0.6,0.8").split(",") if v.strip()))
    plan_agent.parse_proxy(cfg.proxy)
    if cfg.tokens not in plan_agent.TOKEN_KINDS:
        raise ValueError(f"SPECTRA_PLAN_TOKENS={cfg.tokens!r}; expected one of {plan_agent.TOKEN_KINDS}")
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError(f"SPECTRA_PLAN_KAPPA={lo},{hi}: need 0 < lo <= hi < 1")
    if cfg.k < 2:
        raise ValueError("SPECTRA_PLAN_K must be at least 2 (the baseline is the K-plan mean)")
    return cfg


def sigma_at(cfg: Config, it: int) -> float:
    span = max(1.0, cfg.sigma_decay * cfg.instances)
    return max(cfg.sigma_floor, cfg.sigma - (cfg.sigma - cfg.sigma_floor) * min(1.0, it / span))


def out_dir() -> str:
    import src.logging_utils as logging_utils
    path = os.path.join(logging_utils.run_dir() or ".", "plan_agent")
    os.makedirs(path, exist_ok=True)
    return path


def references(inst, kappa, cfg, batches):
    out = {}
    for kind in REFERENCES:
        ref = inst.reference(kind, kappa, cfg.k_min)
        if ref is not None:
            widths, info = ref
            out[kind] = {"r": inst.reward(widths, cfg.proxy, batches), "kept": info["kept"]}
    return out


def mean_plan(policy, inst, kappa, cfg):
    with torch.no_grad():
        mu = policy(inst.state_at(kappa), inst.token_mask, inst.token_k, inst.n_groups)
    return plan_agent.decode(mu.tolist(), inst.pm, kappa, cfg.k_min)


def summary(policy, instances, cfg, log):
    """Mean plan vs references per net at each summary κ on one fixed batch set (val-half proxy reads)."""
    for inst in instances:
        for kappa in cfg.summary_kappas:
            batches = inst.batches(cfg.proxy, cfg.seed)
            widths, info = mean_plan(policy, inst, kappa, cfg)
            r_mu = inst.reward(widths, cfg.proxy, batches)
            refs = references(inst, kappa, cfg, batches)
            utils.print_flush(
                f"[plan] summary {inst.name} k={kappa:.2f}: mean plan {r_mu:+.2f} (x{info['kept']:.3f}) | "
                + " | ".join(f"{name} {ref['r']:+.2f} (x{ref['kept']:.3f})" for name, ref in refs.items())
                + f" | keeps min {min(info['keeps'].values()):.2f} max {max(info['keeps'].values()):.2f}")
            log({"summary": inst.name, "kappa": kappa, "mean": r_mu, "kept": info["kept"],
                 "refs": refs, "widths": {str(r): int(w) for r, w in widths.items()}})


def run(env, shard):
    cfg = config()
    started = time.perf_counter()
    utils.print_flush(f"[plan] trainer {json.dumps(asdict(cfg))}")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    instances = []
    for net_path, (net_model, net_loaders) in shard:
        name = os.path.basename(str(net_path))
        if cfg.nets and not any(part in name for part in cfg.nets):
            continue
        inst = plan_agent.NetInstance.from_env(env, net_path, net_model, net_loaders, tokens=cfg.tokens,
                                               zero=cfg.zero)
        sens = sorted(inst.sens.values()) if inst.sens else [0.0]
        utils.print_flush(
            f"[plan] net {inst.name}: {inst.n_groups} groups, {int(inst.token_mask.sum())}/{inst.token_mask.numel()} "
            f"tokens mapped, width {inst.feature_dim}, {inst.pm.total0} params, origin val {100 * inst.origin_val:.2f}, "
            f"sens min {sens[0]:.4f} median {statistics.median(sens):.4f} max {sens[-1]:.4f}, "
            f"{len(inst.held)} multi-producer groups; {inst.seconds:.1f}s")
        chk = inst.check()
        bad = abs(chk["analytic"] - chk["real"]) > 1e-6 or abs(chk["masked_r"] - chk["real_r"]) > 0.05
        utils.print_flush(
            f"[plan] check {inst.name}: uniform k=0.60 kept analytic x{chk['analytic']:.4f} real x{chk['real']:.4f} | "
            f"cut val Δ masked {chk['masked_r']:+.2f} real {chk['real_r']:+.2f} | dead channels {chk['dead']}"
            + (" | WARNING: the masked reward does not match the real cut" if bad else ""))
        instances.append(inst)
    if not instances:
        raise RuntimeError(f"SPECTRA_PLAN_NETS={cfg.nets}: no test network matched")
    widths = {inst.feature_dim for inst in instances}
    if len(widths) != 1:
        raise RuntimeError(f"networks disagree on the token width: {sorted(widths)}")
    device = instances[0].device
    policy = plan_agent.PlanPolicy(widths.pop(), cfg.encoder).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    folder = out_dir()
    log_path = os.path.join(folder, "plan_train.jsonl")
    from src.feature_standardizer import resolve_standardizer_path
    meta = {"tokens": cfg.tokens, "zero": list(cfg.zero), "k_min": cfg.k_min, "proxy": cfg.proxy,
            "nets": [inst.name for inst in instances], "config": asdict(cfg),
            "standardizer": resolve_standardizer_path(), "actor": os.environ.get("SPECTRA_ACTOR_CHECKPOINT_PATH", "")}

    def log(record):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def save(tag):
        path = os.path.join(folder, f"policy_{tag}.pt")
        plan_agent.save_policy(policy, path, **meta)
        plan_agent.save_policy(policy, os.path.join(folder, "policy_latest.pt"), **meta)
        return path

    optimizer.zero_grad()
    done = 0
    for it in range(cfg.instances):
        t0 = time.perf_counter()
        inst = rng.choice(instances)
        kappa = rng.uniform(cfg.kappa_lo, cfg.kappa_hi)
        sigma = sigma_at(cfg, it)
        seed = cfg.seed * 100003 + it
        batches = inst.batches(cfg.proxy, seed)
        mu = policy(inst.state_at(kappa), inst.token_mask, inst.token_k, inst.n_groups)
        generator = torch.Generator(device=mu.device).manual_seed(seed)
        z = plan_agent.sample_scores(mu, sigma, cfg.k, generator)
        rewards, kept = [], []
        for k in range(cfg.k):
            plan_widths, info = plan_agent.decode(z[k].tolist(), inst.pm, kappa, cfg.k_min)
            rewards.append(inst.reward(plan_widths, cfg.proxy, batches))
            kept.append(info["kept"])
        r = torch.tensor(rewards, device=mu.device, dtype=mu.dtype)
        adv = r - r.mean()
        if cfg.norm_adv:
            adv = adv / (r.std() + 1e-6)
        loss = -(adv.detach() * plan_agent.log_prob(z, mu, sigma)).mean() / cfg.batch
        loss.backward()
        grad = None
        if (it + 1) % cfg.batch == 0:
            grad = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
            optimizer.step()
            optimizer.zero_grad()
        record = {"it": it, "net": inst.name, "kappa": kappa, "sigma": sigma, "rewards": rewards, "kept": kept,
                  "mu_std": float(mu.detach().std()), "loss": float(loss) * cfg.batch, "grad": grad}
        line = (f"[plan] it={it + 1}/{cfg.instances} {inst.name} k={kappa:.3f} sigma={sigma:.3f} | plans mean "
                f"{statistics.mean(rewards):+.2f} max {max(rewards):+.2f} min {min(rewards):+.2f} kept "
                f"x{statistics.mean(kept):.3f} | mu std {record['mu_std']:.3f}")
        if (it + 1) % cfg.ref_every == 0 or it == 0:
            plan_widths, info = mean_plan(policy, inst, kappa, cfg)
            record["mean_plan"] = {"r": inst.reward(plan_widths, cfg.proxy, batches), "kept": info["kept"]}
            record["refs"] = references(inst, kappa, cfg, batches)
            line += f" | mean plan {record['mean_plan']['r']:+.2f}" + "".join(
                f" {name} {ref['r']:+.2f}" for name, ref in record["refs"].items())
        record["seconds"] = time.perf_counter() - t0
        utils.print_flush(line + f" | {record['seconds']:.1f}s")
        log(record)
        done = it + 1
        if done % cfg.save_every == 0:
            save(f"it{done:05d}")
        if cfg.max_minutes and (time.perf_counter() - started) / 60.0 >= cfg.max_minutes:
            utils.print_flush(f"[plan] MAX_MINUTES={cfg.max_minutes:g} reached after {done} instances")
            break
    if done % cfg.batch:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    path = save(f"it{done:05d}")
    utils.print_flush(f"[plan] saved {path} after {done} instances, {(time.perf_counter() - started) / 60.0:.1f} min")
    summary(policy, instances, cfg, log)
    utils.print_flush(f"[plan] DONE {done} instances in {(time.perf_counter() - started) / 60.0:.1f} min")
