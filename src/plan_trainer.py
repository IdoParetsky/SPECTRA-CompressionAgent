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
(0.4,0.6,0.8: the κ of the closing per-net summary), BUDGET (params: what κ is a fraction of; flops decodes every
plan to kept MACs on ``plan_agent.FlopModel`` and flags it in the state, mixed draws params or flops per instance),
MIN_WIDTH (``fortify.plan_min_width``: no plan, sampled or reference, cuts a group below this width, so the reward
only sees plans the eval walk can realize; ``walk`` = the walk's own floor; written to the checkpoint when above 1),
RESIDUAL (``fortify.plan_residual``, off: T2, the agent as a residual on the sens / sens_cost rule, ``auto`` = the
best rule of each instance's budget; every plan, sampled or mean, is then ``plan_agent.residual_decode`` on the net's
prior weights, computed once per budget from the sensitivities and, for sens_cost, the costs of the same cuts; the
prior's own plan joins the references as ``prior`` and the checkpoint carries ``residual`` / ``residual_alpha`` /
``residual_sigma``), RESIDUAL_ALPHA (0.5, the rule's power), RESIDUAL_SIGMA (0.25: the σ schedule scaled by
RESIDUAL_SIGMA / SIGMA, as a unit of z moves a keep by about 100 % relative under the residual against
keep · (1 − keep) under ``decode``), RESIDUAL_ZMAX (``fortify.plan_residual_zmax``, 0 = off, read only with the
residual on: v20's trust region, every plan, sampled or mean, decodes zmax · tanh(z / zmax) in place of z, so no
group's weight leaves e^±zmax of its prior's; z is still what is sampled and scored, each step records how many
groups' mean is past the bound, and the checkpoint carries ``residual_zmax``).
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
    budget: str


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
        summary_kappas=tuple(float(v) for v in _env("SUMMARY_KAPPAS", "0.4,0.6,0.8").split(",") if v.strip()),
        budget=_env("BUDGET", "params").lower())
    plan_agent.parse_proxy(cfg.proxy)
    if cfg.tokens not in plan_agent.TOKEN_KINDS:
        raise ValueError(f"SPECTRA_PLAN_TOKENS={cfg.tokens!r}; expected one of {plan_agent.TOKEN_KINDS}")
    if cfg.budget not in plan_agent.BUDGETS + ("mixed",):
        raise ValueError(f"SPECTRA_PLAN_BUDGET={cfg.budget!r}; expected one of {plan_agent.BUDGETS + ('mixed',)}")
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError(f"SPECTRA_PLAN_KAPPA={lo},{hi}: need 0 < lo <= hi < 1")
    if cfg.k < 2:
        raise ValueError("SPECTRA_PLAN_K must be at least 2 (the baseline is the K-plan mean)")
    return cfg


def sigma_at(cfg: Config, it: int, scale: float = 1.0) -> float:
    """The σ of instance ``it``: linear from SIGMA to SIGMA_FLOOR over the decay span, times ``scale`` (the residual's
    RESIDUAL_SIGMA / SIGMA; 1 leaves the schedule as it was)."""
    span = max(1.0, cfg.sigma_decay * cfg.instances)
    return float(scale) * max(cfg.sigma_floor, cfg.sigma - (cfg.sigma - cfg.sigma_floor) * min(1.0, it / span))


def out_dir() -> str:
    import src.logging_utils as logging_utils
    path = os.path.join(logging_utils.run_dir() or ".", "plan_agent")
    os.makedirs(path, exist_ok=True)
    return path


def min_width() -> int:
    """``SPECTRA_PLAN_MIN_WIDTH`` (``fortify.plan_min_width``): the decoders' width floor; 1 = off, as before."""
    from src import fortify
    return fortify.plan_min_width()


def residual():
    """``(mode, alpha, sigma)`` of ``SPECTRA_PLAN_RESIDUAL`` (``fortify.plan_residual``; off = as before): the rule the
    agent is a residual on, the rule's power and the trainer's starting σ under it."""
    from src import fortify
    return fortify.plan_residual(), fortify.plan_residual_alpha(), fortify.plan_residual_sigma()


def residual_zmax() -> float:
    """``SPECTRA_PLAN_RESIDUAL_ZMAX`` (``fortify.plan_residual_zmax``; 0 = off, as before): the trust region's bound on
    the residual's decoded scores (``plan_agent.residual_decode``); ``run`` reads it once, only with the residual on."""
    from src import fortify
    return fortify.plan_residual_zmax()


def prior_of(inst, res, budget="params"):
    """``inst``'s prior weights under ``res`` = (mode, alpha) on ``budget`` (``NetInstance.prior``); None when off."""
    if res is None:
        return None
    kind = plan_agent.prior_kind(res[0], budget)
    weights = inst.prior(kind, budget, res[1])
    if weights is None:
        raise RuntimeError(f"{inst.name}: the {kind} prior needs the sensitivities"
                           + (" and costs" if kind == "sens_cost" else "") + " of the origin")
    return weights


def _decode(z, cm, kappa, cfg, floor=1, prior=None, zmax=0.0):
    """``plan_agent.decode`` as before, or ``residual_decode`` on ``prior`` when the residual is on, inside the trust
    region when ``zmax`` is above 0."""
    if prior is None:
        return plan_agent.decode(z, cm, kappa, cfg.k_min, min_width=floor)
    return plan_agent.residual_decode(z, prior, cm, kappa, cfg.k_min, min_width=floor, zmax=zmax)


def references(inst, kappa, cfg, batches, budget="params", floor=1, prior=None):
    out = {}
    for kind in REFERENCES:
        ref = inst.reference(kind, kappa, cfg.k_min, budget=budget, min_width=floor)
        if ref is not None:
            widths, info = ref
            out[kind] = {"r": inst.reward(widths, cfg.proxy, batches), "kept": info["kept"]}
    if prior is not None:  # the rule the agent starts from, decoded as its mean plan is: equal to it at z = 0
        widths, info = plan_agent.scale_decode(prior, inst.cost_model(budget), kappa, cfg.k_min, min_width=floor)
        out["prior"] = {"r": inst.reward(widths, cfg.proxy, batches), "kept": info["kept"]}
    return out


def mean_plan(policy, inst, kappa, cfg, budget="params", floor=1, prior=None, zmax=0.0):
    with torch.no_grad():
        mu = policy(inst.state_at(kappa, budget), inst.token_mask, inst.token_k, inst.n_groups)
    return _decode(mu.tolist(), inst.cost_model(budget), kappa, cfg, floor, prior, zmax)


def _budget_tag(budget) -> str:
    """`` b=flops`` after a log line's ``k=`` token; empty under the params budget, whose lines are unchanged."""
    return "" if budget == "params" else f" b={budget}"


def summary(policy, instances, cfg, log, floor=1, res=None, zmax=0.0):
    """Mean plan vs references per net at each summary κ (and budget) on one fixed batch set (val-half proxy reads)."""
    budgets = plan_agent.BUDGETS if cfg.budget == "mixed" else (cfg.budget,)
    for inst in instances:
        for kappa in cfg.summary_kappas:
            for budget in budgets:
                batches = inst.batches(cfg.proxy, cfg.seed)
                prior = prior_of(inst, res, budget)
                widths, info = mean_plan(policy, inst, kappa, cfg, budget, floor, prior, zmax)
                r_mu = inst.reward(widths, cfg.proxy, batches)
                refs = references(inst, kappa, cfg, batches, budget, floor, prior)
                utils.print_flush(
                    f"[plan] summary {inst.name} k={kappa:.2f}{_budget_tag(budget)}: mean plan {r_mu:+.2f} "
                    f"(x{info['kept']:.3f}) | "
                    + " | ".join(f"{name} {ref['r']:+.2f} (x{ref['kept']:.3f})" for name, ref in refs.items())
                    + f" | keeps min {min(info['keeps'].values()):.2f} max {max(info['keeps'].values()):.2f}")
                log({"summary": inst.name, "kappa": kappa, "budget": budget, "mean": r_mu, "kept": info["kept"],
                     "refs": refs, "widths": {str(r): int(w) for r, w in widths.items()}})


def run(env, shard):
    cfg = config()
    floor = min_width()
    mode, alpha, sigma0 = residual()
    res = None if mode == "off" else (mode, alpha)
    zmax = residual_zmax() if res else 0.0  # the trust region; 0 = every residual decode as before
    budgets = plan_agent.BUDGETS if cfg.budget == "mixed" else (cfg.budget,)
    with_cost = res is not None and "sens_cost" in {plan_agent.prior_kind(mode, b) for b in budgets}
    scale = sigma0 / cfg.sigma if res is not None and cfg.sigma > 0 else 1.0
    started = time.perf_counter()
    utils.print_flush(f"[plan] trainer {json.dumps(asdict(cfg))}" + (f" min_width={floor}" if floor > 1 else "")
                      + (f" residual={mode} alpha={alpha:g} sigma={sigma0:g}" if res else "")
                      + (f" zmax={zmax:g}" if zmax > 0 else ""))
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    instances = []
    for net_path, (net_model, net_loaders) in shard:
        name = os.path.basename(str(net_path))
        if cfg.nets and not any(part in name for part in cfg.nets):
            continue
        inst = plan_agent.NetInstance.from_env(env, net_path, net_model, net_loaders, tokens=cfg.tokens,
                                               zero=cfg.zero, with_cost=with_cost)
        sens = sorted(inst.sens.values()) if inst.sens else [0.0]
        utils.print_flush(
            f"[plan] net {inst.name}: {inst.n_groups} groups, {int(inst.token_mask.sum())}/{inst.token_mask.numel()} "
            f"tokens mapped, width {inst.feature_dim}, {inst.pm.total0} params, origin val {100 * inst.origin_val:.2f}, "
            f"sens min {sens[0]:.4f} median {statistics.median(sens):.4f} max {sens[-1]:.4f}, "
            f"{len(inst.held)} multi-producer groups; {inst.seconds:.1f}s")
        for b in (budgets if res else ()):
            w = sorted(prior_of(inst, res, b).values())
            utils.print_flush(
                f"[plan] prior {inst.name}{_budget_tag(b)}: {plan_agent.prior_kind(mode, b)} alpha={alpha:g} weights "
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
            "budget": cfg.budget, "nets": [inst.name for inst in instances], "config": asdict(cfg),
            "standardizer": resolve_standardizer_path(), "actor": os.environ.get("SPECTRA_ACTOR_CHECKPOINT_PATH", ""),
            **({"min_width": floor} if floor > 1 else {}),
            **({"residual": mode, "residual_alpha": alpha, "residual_sigma": sigma0} if res else {}),
            **({"residual_zmax": zmax} if zmax > 0 else {})}

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
        budget = cfg.budget if cfg.budget != "mixed" else rng.choice(plan_agent.BUDGETS)
        sigma = sigma_at(cfg, it, scale)
        seed = cfg.seed * 100003 + it
        batches = inst.batches(cfg.proxy, seed)
        mu = policy(inst.state_at(kappa, budget), inst.token_mask, inst.token_k, inst.n_groups)
        generator = torch.Generator(device=mu.device).manual_seed(seed)
        z = plan_agent.sample_scores(mu, sigma, cfg.k, generator)
        rewards, kept = [], []
        cm = inst.cost_model(budget)
        prior = prior_of(inst, res, budget)
        for k in range(cfg.k):
            plan_widths, info = _decode(z[k].tolist(), cm, kappa, cfg, floor, prior, zmax)
            rewards.append(inst.reward(plan_widths, cfg.proxy, batches))
            kept.append(info["kept"])
        r = torch.tensor(rewards, device=mu.device, dtype=mu.dtype)
        adv = r - r.mean()
        if cfg.norm_adv:
            adv = adv / (r.std() + 1e-6)
        # The action is the raw z ~ N(mu, sigma²), under the trust region too: zmax squashes it only inside the decode,
        # a deterministic map from action to plan, so log_prob(z) is still the exact log-density of what was sampled and
        # the REINFORCE estimate stays unbiased (no tanh Jacobian: no density is taken of the squashed scores).
        loss = -(adv.detach() * plan_agent.log_prob(z, mu, sigma)).mean() / cfg.batch
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
        if zmax > 0:  # groups whose mean is past the bound, where tanh's slope (< 0.42) starts to thin their signal
            record["mu_beyond_zmax"] = int((mu.detach().abs() > zmax).sum())
        line = (f"[plan] it={it + 1}/{cfg.instances} {inst.name} k={kappa:.3f}{_budget_tag(budget)} sigma={sigma:.3f} "
                f"| plans mean {statistics.mean(rewards):+.2f} max {max(rewards):+.2f} min {min(rewards):+.2f} kept "
                f"x{statistics.mean(kept):.3f} | mu std {record['mu_std']:.3f}")
        if zmax > 0:
            line += f" | mu beyond zmax {record['mu_beyond_zmax']}/{inst.n_groups}"
        if (it + 1) % cfg.ref_every == 0 or it == 0:
            plan_widths, info = mean_plan(policy, inst, kappa, cfg, budget, floor, prior, zmax)
            record["mean_plan"] = {"r": inst.reward(plan_widths, cfg.proxy, batches), "kept": info["kept"]}
            record["refs"] = references(inst, kappa, cfg, batches, budget, floor, prior)
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
    summary(policy, instances, cfg, log, floor, res, zmax)
    utils.print_flush(f"[plan] DONE {done} instances in {(time.perf_counter() - started) / 60.0:.1f} min")
