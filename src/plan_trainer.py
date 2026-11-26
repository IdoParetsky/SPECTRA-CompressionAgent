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
KAPPA (0.35,0.85), KMIN (0.1), PROXY (bn32; ``fs<S>``: S SGD steps on the masked cut, lr ``SPECTRA_PLAN_FS_LR``;
``ft<E>``, v23: E epochs of the learning handler's own fine-tune on the real cut, the eval-side ft<E> proxy, under the
instance's seed, ``plan_agent.NetInstance.finetune_reward``; the handler's recipe is logged once per run and written to
the checkpoints as ``recipe_line``), TOKENS (layer), ENCODER (transformer), ZERO (channels set to 0, e.g. sens), NETS
(name substrings to keep; all when unset), REF_EVERY (10), SAVE_EVERY (50), SEED (0),
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
groups' mean is past the bound, and the checkpoint carries ``residual_zmax``), INIT (v23, unset: a plan-agent checkpoint
whose policy weights the run starts from, ``warm_start``; refused when it trained under other settings than this run's,
``plan_agent.ENSEMBLE_KEYS``; no optimizer state exists to load, and the σ schedule stays SIGMA / SIGMA_FLOOR /
SIGMA_DECAY; recorded as ``init`` in the config line and in every checkpoint), STATE (v25, default off: ``1`` writes
``trainer_state.pt`` beside every ``policy_latest.pt``, atomically (a temp file ``os.replace``d over it): the policy and
optimizer state dicts, the partial batch's accumulated gradient per parameter, the completed count, the trainer's own
``rng`` and the global random / numpy / torch (and CUDA) RNG states, the checkpoint meta with the config, the states
this run resumed from and a format version (``STATE_KEYS``); with it on, MAX_MINUTES stops only at a batch boundary,
at most BATCH - 1 instances later, so a stopped job flushes no partial step and its policy_latest.pt is the chain's
policy, and the closing state is written before the end-of-run partial step, so every state is the trajectory state
after ``done`` instances), RESUME (v25, unset: the ``trainer_state.pt`` the run continues from; the origins are read and
the policy built as before, then the state is put back and the loop runs it = done ... INSTANCES with the instance
numbering, the saves and the DONE line of one uninterrupted run, each job logging to its own run folder; refused unless
the state's settings are this run's, every checkpoint meta key and config field but ``RESUME_FREE`` = MAX_MINUTES and
SAVE_EVERY, a differing standardizer path warning as INIT's does; the chain's final weights are bitwise the
uninterrupted run's on CPU, and on CUDA as alike as two uninterrupted runs are), MENU (v36, unset = off:
``plan_agent.menu_kinds``, a comma list of rules such as ``sens,uniform,sens_cost,sens_thresh``; it needs the residual
on, and its first rule, the anchor, must be the residual's prior on every budget the run draws, ``menu_settings``. Each
instance then reads every rule's weights at its κ (``NetInstance.rules_at``; the damage curves behind sens_thresh are
measured once per net) and their plans' costs (``plan_agent.menu_features``); the policy's menu head gives K logits ψ;
each of the K plans draws u = ψ + MENU_SIGMA · ε after the scores' draw, from the same generator, so the scores' noise
is the run's without the menu (as are the policy's starting weights and every global generator: the menu head's and
the curves' draws are put back); its plan is ``plan_agent.menu_decode`` at π = softmax(u), and log π(action) adds u's
Gaussian term. The mean plan decodes the mean scores at softmax(ψ) with the shares under
``plan_agent.MENU_SHARE_FLOOR`` dropped, so the zero-init run starts on the prior's plan exactly. Each step's record
gains ``menu_pi``, each summary a ``[plan] menu summary`` line, each net a ``[plan] menu setup`` line with the seconds
its rules cost, and the checkpoint ``menu`` / ``menu_anchor`` / ``menu_init`` / ``menu_sigma`` /
``menu_share_floor``), MENU_INIT (6.0, the anchor's starting logit), MENU_SIGMA (0.5, constant over the run).

AGENT-V2 (26 Nov 2026, ``docs/paper/AGENT_AUDIT_26NOV.md``; queue row AGENT-V2). Every knob below is new and default
off; with all of them unset the loop, its draws, records, log lines and checkpoints are tree_v41's
(``tests/test_agent_v2.py``). Each is read through its ``fortify.plan_*`` function, which documents it; the ones that
are on are listed after ``| av2:`` in the opening line and written to the checkpoints (``av2_meta``), so a chained job
under other flags is refused. PROXY also takes ``cutce`` / ``bnce<N>`` (the reward is the val cross-entropy change);
ADV (``rank`` | ``global[:decay]`` | ``floor:<s0>``: the advantages, in place of NORM_ADV's division); MIRROR (K / 2
draws and their mirror images); MENU_BOUND (the mean logits are B tanh(psi / B)), MENU_ENT (an annealed entropy bonus
on the mean shares), MENU_KAPPA (``bins`` | ``film``: the menu head reads the budget's bin), MENU_LOG (psi, its range
and the per-plan shares in each record), MENU_SHARE_FLOOR (the mean plan's share floor; ``auto`` under MENU_BOUND),
MENU_SIGMA_ANNEAL (sigma_u follows the scores' schedule); SPLIT (half the plans draw the scores at the mean shares,
half the menu logits at the mean scores, each half with its own advantages); EMA (an averaged copy of the weights,
saved as ``policy_ema_latest.pt`` at every save and ``policy_ema_it<N>.pt`` at the end) and LR_SCHEDULE=cosine;
GNORM (``sqrt`` | ``linear``: the scores' log-density over sqrt(G) or G); KAPPA_POINTS (part of the budgets drawn near
the registered points); SAMPLE=family[:t] (networks drawn by family); FS_GUARD (a diverged ``fs<S>`` fine-tune's plan
takes the median reward of its instance); CHECK (``1`` | ``refuse``: three plans per network cut for real against the
masked cut, one ``[plan] check gate`` verdict line); CONTRACT (the state contract in the checkpoint, which
``plan_agent.plan_for_env`` then refuses on a mismatch). With STATE on, the averaged weights and ADV=global's running
scale travel in ``trainer_state.pt`` under ``av2``.
"""

import json
import math
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
    plan_agent.parse_proxy(cfg.proxy, ce=True)  # AGENT-V2: cutce / bnce<N> too; every other name as before
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


def menu_settings(mode="off", budgets=("params",)):
    """``SPECTRA_PLAN_MENU`` (v36; unset = off: None, and the run is as before): ``{"kinds", "init", "sigma"}`` of the
    menu (``plan_agent.menu_kinds`` / ``menu_init`` / ``menu_sigma``). Refused (ValueError) without the residual, or
    when the anchor is not the residual's prior (``mode``) on one of the ``budgets`` the run draws
    (``plan_agent.check_menu_anchor``): ``auto`` under a mixed budget has two priors and so no anchor."""
    kinds = plan_agent.menu_kinds()
    if kinds is None:
        return None
    for budget in budgets:
        plan_agent.check_menu_anchor(kinds, mode, budget, f"SPECTRA_PLAN_MENU={','.join(kinds)}")
    return {"kinds": kinds, "init": plan_agent.menu_init(), "sigma": plan_agent.menu_sigma()}


def menu_at(inst, menu, res, kappa, cfg, budget="params", floor=1):
    """The menu on ``inst`` at ``kappa`` (v36): ``{"rules", "x", "plans"}``, every rule's weights
    (``NetInstance.rules_at`` with the residual's alpha and the decoders' floors), the menu head's network-level inputs
    and each rule's own plan (``plan_agent.menu_features``). None when the menu is off."""
    if menu is None:
        return None
    rules = inst.rules_at(menu["kinds"], kappa, budget, res[1], cfg.k_min, floor)
    x_net, plans = plan_agent.menu_features(rules, inst.cost_model(budget), kappa, cfg.k_min, floor)
    return {"rules": rules, "x": x_net, "plans": plans}


def init_path() -> str:
    """``SPECTRA_PLAN_INIT`` (v23; empty = off, as before): a plan-agent checkpoint whose policy weights the run starts
    from."""
    return _env("INIT", "")


def warm_start(policy, blob, meta, path):
    """Load ``blob``'s policy weights (``SPECTRA_PLAN_INIT``, v23) into ``policy`` for a run whose checkpoint meta is
    ``meta``. Refused (ValueError) when the checkpoint trained under other settings than this run's: every
    ``plan_agent.ENSEMBLE_KEYS`` entry (feature_dim, encoder, tokens, zero, k_min, min_width, budget, the residual
    settings and, v36, the menu's rules) but the standardizer path, which only warns, as ``plan_for_env`` does. No
    optimizer state is saved, so none is loaded, and the σ schedule stays the run's own."""
    want = {**meta, "feature_dim": policy.feature_dim, "encoder": policy.encoder_kind}
    for key in plan_agent.ENSEMBLE_KEYS:
        if key != "standardizer" and blob.get(key) != want.get(key):
            raise ValueError(f"SPECTRA_PLAN_INIT={path}: trained with {key}={blob.get(key)!r}; this run has "
                             f"{want.get(key)!r}")
    trained, here = blob.get("standardizer") or "", want.get("standardizer") or ""
    if trained and os.path.abspath(trained) != os.path.abspath(here):
        utils.print_flush(f"[plan] WARNING: SPECTRA_PLAN_INIT trained under standardizer {trained}; "
                          f"this run reads {here}")
    # AGENT-V2 (F15): two recorded state contracts must agree; a checkpoint or a run without one is not checked
    if blob.get("state_contract") and meta.get("state_contract") and blob["state_contract"] != meta["state_contract"]:
        old, new = blob["state_contract"], meta["state_contract"]
        bad = [f"{key}={old.get(key)!r} (here {new.get(key)!r})" for key in sorted(set(old) | set(new))
               if old.get(key) != new.get(key)]
        raise ValueError(f"SPECTRA_PLAN_INIT={path}: trained under another state contract: " + "; ".join(bad))
    policy.load_state_dict(blob["state_dict"])
    utils.print_flush(f"[plan] init {path}: policy weights loaded (trained with proxy={blob.get('proxy')!r} "
                      f"budget={blob.get('budget')!r} on {len(blob.get('nets') or [])} nets)")


STATE_FORMAT = 1
STATE_KEYS = ("format", "done", "policy", "optimizer", "grads", "rng", "meta", "chain", "saved")
RESUME_FREE = ("max_minutes", "save_every")  # the config fields a chained job may change


def state_enabled() -> bool:
    """``SPECTRA_PLAN_STATE`` (v25; default off, as before): write ``trainer_state.pt`` beside every
    ``policy_latest.pt``."""
    return _env("STATE", "0").lower() in ("1", "true", "yes", "on")


def resume_path() -> str:
    """``SPECTRA_PLAN_RESUME`` (v25; empty = off, as before): the ``trainer_state.pt`` the run continues from."""
    return _env("RESUME", "")


def rng_states(rng):
    """Every generator the loop or a reward may draw from: the trainer's own ``rng``, Python's ``random``, NumPy's,
    torch's CPU generator and, when CUDA is available, every CUDA generator (None otherwise)."""
    try:
        import numpy as np
        npy = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy is a dependency of the env
        npy = None
    return {"trainer": rng.getstate(), "python": random.getstate(), "numpy": npy, "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_states(states, rng):
    """Put ``rng_states``' generators back; the CUDA ones only onto the same number of devices (a WARNING otherwise)."""
    rng.setstate(states["trainer"])
    random.setstate(states["python"])
    if states.get("numpy") is not None:
        import numpy as np
        np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    cuda, devices = states.get("cuda"), torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cuda is not None and devices and len(cuda) == devices:
        torch.cuda.set_rng_state_all(cuda)
    elif cuda is not None or devices:
        utils.print_flush(f"[plan] WARNING: the state holds {0 if cuda is None else len(cuda)} CUDA RNG states and "
                          f"this run has {devices} CUDA devices; the CUDA generators are not restored")


def trainer_state(policy, optimizer, rng, done, meta, chain, extra=None):
    """The resumable state after ``done`` instances (``STATE_KEYS``): the policy and optimizer state dicts, the partial
    batch's accumulated gradient per policy parameter (None at a batch boundary, and for a parameter the forward
    never uses), ``rng_states``, the checkpoint meta (``config`` included), the states this run resumed from (oldest
    first), the format and the wall-clock time. ``extra`` (AGENT-V2; None with no stateful flag on, and the state is
    then as before) is kept under ``av2``: the averaged weights and the running advantage scale (``av2_state``)."""
    out = {"format": STATE_FORMAT, "done": int(done), "policy": policy.state_dict(),
           "optimizer": optimizer.state_dict(),
           "grads": [None if p.grad is None else p.grad.detach() for p in policy.parameters()],
           "rng": rng_states(rng), "meta": dict(meta), "chain": list(chain), "saved": time.time()}
    if extra:
        out["av2"] = extra
    return out


def write_state(path: str, state):
    """``torch.save`` to a temp file, then ``os.replace`` it over ``path``: a job killed mid-write leaves the old
    state."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def read_state(path: str):
    """The ``trainer_state.pt`` at ``path`` as saved (``plan_agent.read_blob``); refused (ValueError) unless this
    trainer wrote it (``format`` = STATE_FORMAT)."""
    blob = plan_agent.read_blob(path)
    found = blob.get("format") if isinstance(blob, dict) else type(blob).__name__
    if not isinstance(blob, dict) or found != STATE_FORMAT or "done" not in blob:
        raise ValueError(f"SPECTRA_PLAN_RESUME={path}: not a trainer state of format {STATE_FORMAT} (found {found!r})")
    return blob


def check_resume(stored, meta, path):
    """Refuse (ValueError) a state from a run under other settings than this one's: every checkpoint meta key but
    ``recipe_line`` (put back by ``restore``) and ``standardizer`` (a WARNING, as ``warm_start``), and every config
    field but ``RESUME_FREE``; every mismatch is named."""
    bad = []
    for key in sorted((set(stored) | set(meta)) - {"config", "recipe_line", "standardizer"}):
        if stored.get(key) != meta.get(key):
            bad.append(f"{key}={stored.get(key)!r} (here {meta.get(key)!r})")
    old, new = stored.get("config") or {}, meta.get("config") or {}
    for key in sorted((set(old) | set(new)) - set(RESUME_FREE)):
        if old.get(key) != new.get(key):
            bad.append(f"config.{key}={old.get(key)!r} (here {new.get(key)!r})")
    if bad:
        raise ValueError(f"SPECTRA_PLAN_RESUME={path}: the state's run had " + "; ".join(bad)
                         + f"; only {', '.join(RESUME_FREE)} may differ between chained jobs")
    trained, here = stored.get("standardizer") or "", meta.get("standardizer") or ""
    if trained != here and os.path.abspath(trained) != os.path.abspath(here):
        utils.print_flush(f"[plan] WARNING: SPECTRA_PLAN_RESUME trained under standardizer {trained}; this run reads "
                          f"{here}")


def restore(state, path, cfg, meta, policy, optimizer, rng) -> int:
    """Continue from ``state`` (``read_state``): after ``check_resume``, the policy weights, the optimizer state,
    the partial batch's gradients and every RNG state are put back in place (into the fresh run's own policy and
    optimizer, so the optimizer keeps its tensors) and the stored ``recipe_line`` rejoins ``meta``. Returns the
    completed count."""
    stored = state["meta"]
    check_resume(stored, meta, path)
    if stored.get("recipe_line"):
        meta["recipe_line"] = stored["recipe_line"]
    policy.load_state_dict(state["policy"])
    optimizer.load_state_dict(state["optimizer"])
    params = list(policy.parameters())
    if len(state["grads"]) != len(params):
        raise ValueError(f"SPECTRA_PLAN_RESUME={path}: {len(state['grads'])} gradients for {len(params)} parameters")
    for param, grad in zip(params, state["grads"]):
        param.grad = None if grad is None else grad.to(device=param.device, dtype=param.dtype)
    set_rng_states(state["rng"], rng)
    done, into = int(state["done"]), int(state["done"]) % cfg.batch
    utils.print_flush(f"[plan] resumed from {path} at {done} instances"
                      + (f", {into} of {cfg.batch} into a batch" if into else ""))
    return done


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


def _decode(z, cm, kappa, cfg, floor=1, prior=None, zmax=0.0, mix=None):
    """``plan_agent.decode`` as before, or ``residual_decode`` on ``prior`` when the residual is on, inside the trust
    region when ``zmax`` is above 0. ``mix`` = (shares, rules) (v36, the menu) decodes the same scores on the blend of
    the rules instead (``plan_agent.menu_decode``)."""
    if mix is not None:
        return plan_agent.menu_decode(z, mix[0], mix[1], cm, kappa, cfg.k_min, min_width=floor, zmax=zmax)
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


def mean_plan(policy, inst, kappa, cfg, budget="params", floor=1, prior=None, zmax=0.0, at=None, share_floor=None):
    """``(widths, info)`` of the mean plan. ``at`` (``menu_at``; v36) makes it the menu's: the mean scores decoded at
    the mean shares softmax(ψ), a rule under ``plan_agent.MENU_SHARE_FLOOR`` dropped (``info["pi"]``).
    ``share_floor`` (AGENT-V2; None = that constant, as before) is the run's own floor (``plan_agent.menu_floor``)."""
    with torch.no_grad():
        mu = policy(inst.state_at(kappa, budget), inst.token_mask, inst.token_k, inst.n_groups)
        psi = None if at is None else policy.menu_logits(inst.state_at(kappa, budget), at["x"])
    drop = plan_agent.MENU_SHARE_FLOOR if share_floor is None else float(share_floor)
    mix = None if at is None else (plan_agent.menu_shares(psi.tolist(), drop), at["rules"])
    return _decode(mu.tolist(), inst.cost_model(budget), kappa, cfg, floor, prior, zmax, mix)


def _budget_tag(budget) -> str:
    """`` b=flops`` after a log line's ``k=`` token; empty under the params budget, whose lines are unchanged."""
    return "" if budget == "params" else f" b={budget}"


def summary(policy, instances, cfg, log, floor=1, res=None, zmax=0.0, menu=None, share_floor=None):
    """Mean plan vs references per net at each summary κ (and budget) on one fixed batch set (val-half proxy reads).
    Under the menu (v36) each summary also prints the mean plan's shares, one ``[plan] menu summary`` line, and records
    them as ``menu_pi``. ``share_floor`` as ``mean_plan``'s."""
    budgets = plan_agent.BUDGETS if cfg.budget == "mixed" else (cfg.budget,)
    for inst in instances:
        for kappa in cfg.summary_kappas:
            for budget in budgets:
                batches = inst.batches(cfg.proxy, cfg.seed)
                prior = prior_of(inst, res, budget)
                at = menu_at(inst, menu, res, kappa, cfg, budget, floor)
                widths, info = mean_plan(policy, inst, kappa, cfg, budget, floor, prior, zmax, at, share_floor)
                r_mu = inst.reward(widths, cfg.proxy, batches)
                refs = references(inst, kappa, cfg, batches, budget, floor, prior)
                utils.print_flush(
                    f"[plan] summary {inst.name} k={kappa:.2f}{_budget_tag(budget)}: mean plan {r_mu:+.2f} "
                    f"(x{info['kept']:.3f}) | "
                    + " | ".join(f"{name} {ref['r']:+.2f} (x{ref['kept']:.3f})" for name, ref in refs.items())
                    + f" | keeps min {min(info['keeps'].values()):.2f} max {max(info['keeps'].values()):.2f}")
                if at is not None:
                    utils.print_flush(f"[plan] menu summary {inst.name} k={kappa:.2f}{_budget_tag(budget)}: pi "
                                      + " ".join(f"{kind} {p:.3f}" for kind, p in zip(menu["kinds"], info["pi"])))
                log({"summary": inst.name, "kappa": kappa, "budget": budget, "mean": r_mu, "kept": info["kept"],
                     "refs": refs, "widths": {str(r): int(w) for r, w in widths.items()},
                     **({"menu_pi": info["pi"]} if at is not None else {})})


# ------------------------------------------------------------------ AGENT-V2 (26 Nov 2026; every piece default off)

AV2_ORDER = ("adv", "mirror", "split", "gnorm", "ema", "lr_schedule", "menu_bound", "menu_entropy", "menu_kappa",
             "menu_sigma_anneal", "menu_share_floor", "menu_log", "kappa_points", "sample", "fs_guard", "check",
             "contract")
AV2_MENU_ONLY = ("menu_bound", "menu_entropy", "menu_kappa", "menu_log", "menu_share_floor", "menu_sigma_anneal",
                 "split")


def av2_settings(cfg, res=None, menu=None):
    """The AGENT-V2 flags that are on, ``{name: value}`` in ``AV2_ORDER``'s names; ``{}`` with every one unset, and the
    run is then tree_v41's. Refused (ValueError) before any origin is read: a menu flag or SPLIT without the menu, an
    odd K under MIRROR or SPLIT (a K not divisible by 4 under both, or under 4 with SPLIT), and FS_GUARD without an
    ``fs<S>`` proxy."""
    from src import fortify
    out = {}
    mode, value = fortify.plan_adv()
    if mode != "off":
        out["adv"] = (mode, value)
    if fortify.plan_mirror():
        out["mirror"] = True
    if fortify.plan_split():
        out["split"] = True
    if fortify.plan_gnorm() != "off":
        out["gnorm"] = fortify.plan_gnorm()
    if fortify.plan_ema() > 0:
        out["ema"] = fortify.plan_ema()
    if fortify.plan_lr_schedule() != "constant":
        out["lr_schedule"] = fortify.plan_lr_schedule()
    if fortify.plan_menu_bound() > 0:
        out["menu_bound"] = fortify.plan_menu_bound()
    if fortify.plan_menu_entropy() > 0:
        out["menu_entropy"] = fortify.plan_menu_entropy()
    points = fortify.plan_kappa_points()
    kappa = fortify.plan_menu_kappa()
    if kappa is not None:  # the bins span every budget the run draws
        lo, hi = cfg.kappa_lo, cfg.kappa_hi
        if points is not None:
            lo, hi = min(lo, max(0.02, min(points[0]) - points[2])), max(hi, min(0.98, max(points[0]) + points[2]))
        out["menu_kappa"] = {"kind": kappa[0], "n": int(kappa[1]), "lo": float(lo), "hi": float(hi)}
    if fortify.plan_menu_sigma_anneal():
        out["menu_sigma_anneal"] = True
    if fortify.plan_menu_share_floor() is not None:
        out["menu_share_floor"] = fortify.plan_menu_share_floor()
    if fortify.plan_menu_log():
        out["menu_log"] = True
    if points is not None:
        out["kappa_points"] = points
    kind, t = fortify.plan_sample()
    if kind != "net":
        out["sample"] = float(t)
    if fortify.plan_fs_guard() > 0:
        out["fs_guard"] = fortify.plan_fs_guard()
    if fortify.plan_check() != "off":
        out["check"] = fortify.plan_check()
    if fortify.plan_contract() == "on":
        out["contract"] = True
    need_menu = [key for key in AV2_MENU_ONLY if key in out]
    if menu is None and need_menu:
        flag = {"menu_entropy": "MENU_ENT"}.get(need_menu[0], need_menu[0].upper())
        raise ValueError(f"SPECTRA_PLAN_{flag} needs the menu (SPECTRA_PLAN_MENU is unset)")
    if out.get("mirror") and cfg.k % 2:
        raise ValueError(f"SPECTRA_PLAN_MIRROR needs an even SPECTRA_PLAN_K, not {cfg.k}")
    if out.get("split") and (cfg.k % 2 or cfg.k < 4 or (out.get("mirror") and cfg.k % 4)):
        raise ValueError(f"SPECTRA_PLAN_SPLIT needs an even SPECTRA_PLAN_K of at least 4 (divisible by 4 under "
                         f"SPECTRA_PLAN_MIRROR), not {cfg.k}")
    if "fs_guard" in out and plan_agent.parse_proxy(cfg.proxy, ce=True)[0] != "fs":
        raise ValueError(f"SPECTRA_PLAN_FS_GUARD watches the fs<S> proxy's fine-tune; SPECTRA_PLAN_PROXY={cfg.proxy}")
    return out


def _av2_show(key, value):
    if key == "adv":
        return value[0] if value[1] is None else f"{value[0]}:{value[1]:g}"
    if key == "menu_kappa":
        return f"{value['kind']}:{value['n']}@{value['lo']:g}-{value['hi']:g}"
    if key == "kappa_points":
        return ",".join(f"{p:g}" for p in value[0]) + f":{value[1]:g}:{value[2]:g}"
    if key == "sample":
        return f"family:{value:g}"
    if value is True:
        return "1"
    return f"{value:g}" if isinstance(value, float) else str(value)


def av2_line(av2):
    """`` | av2: adv=rank mirror=1 …``: the flags that are on, for the trainer's opening line; empty with none on."""
    if not av2:
        return ""
    return " | av2: " + " ".join(f"{key}={_av2_show(key, av2[key])}" for key in AV2_ORDER if key in av2)


def av2_meta(av2):
    """What the flags that are on add to the checkpoint meta (nothing with none on): one key per flag, with values a
    chained job's ``check_resume`` can compare. ``menu_share_floor`` (the run's resolved floor, an older key) and
    ``state_contract`` are written by ``run``."""
    out = {}
    for key in AV2_ORDER:
        if key not in av2 or key in ("menu_share_floor", "contract"):
            continue
        value = av2[key]
        if key == "menu_entropy":
            out["menu_entropy"] = float(value)
        elif key in ("adv", "sample"):
            out[key] = _av2_show(key, value)
        elif key == "kappa_points":
            out[key] = [list(value[0]), float(value[1]), float(value[2])]
        elif key == "menu_kappa":
            out[key] = dict(value)
        else:
            out[key] = value
    return out


def advantages(r, cfg, adv=None, scale=None):
    """The advantages of an instance's rewards ``r``. ``adv`` None is the trainer's own: r − mean, over the SD + 1e-6
    under NORM_ADV. ``adv`` = ``fortify.plan_adv()`` (AGENT-V2) replaces it: ``rank`` is
    ``plan_agent.rank_utilities``; ``floor`` is r − mean over max(SD, s0); ``global`` is r − mean over ``scale``'s
    running mean of the instances' SDs, this instance's included (``RunningScale``)."""
    if adv is None:
        out = r - r.mean()
        if cfg.norm_adv:
            out = out / (r.std() + 1e-6)
        return out
    mode, value = adv
    if mode == "rank":
        return plan_agent.rank_utilities(r)
    centred = r - r.mean()
    if mode == "floor":
        return centred / torch.clamp(r.std(), min=float(value))
    return centred / max(scale.update(float(r.std())), 1e-6)


class RunningScale:
    """The running mean of the per-instance reward SD (``SPECTRA_PLAN_ADV=global``): an exponential moving average with
    Adam's bias correction, so the first instance divides by its own SD and later ones by the run's."""

    def __init__(self, decay: float):
        self.decay, self.value, self.steps = float(decay), 0.0, 0

    def update(self, sd: float) -> float:
        if math.isfinite(sd):
            self.steps += 1
            self.value = self.decay * self.value + (1.0 - self.decay) * float(sd)
        return self.read()

    def read(self) -> float:
        return self.value / (1.0 - self.decay ** self.steps) if self.steps else 0.0

    def state(self):
        return {"decay": self.decay, "value": self.value, "steps": self.steps}

    def load(self, state):
        self.value, self.steps = float(state["value"]), int(state["steps"])


class PolicyEma:
    """An exponential moving average of a policy's weights (``SPECTRA_PLAN_EMA``; audit F6, Polyak averaging), started
    at the policy's weights and updated after every optimizer step with decay min(d, (1 + t) / (10 + t)), t the update
    count: early on it tracks the policy, and over a run shorter than about 9 / (1 − d) updates it averages roughly the
    last ninth of the trajectory. Integer buffers, if any, are copied."""

    def __init__(self, policy, decay: float):
        self.decay, self.updates = float(decay), 0
        self.shadow = {name: value.detach().clone() for name, value in policy.state_dict().items()}

    @torch.no_grad()
    def update(self, policy):
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for name, value in policy.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[name].mul_(d).add_(value.detach(), alpha=1.0 - d)
            else:
                self.shadow[name].copy_(value)

    def state(self):
        return {"decay": self.decay, "updates": self.updates,
                "shadow": {name: value.clone() for name, value in self.shadow.items()}}

    def load(self, state):
        if set(state["shadow"]) != set(self.shadow):
            raise ValueError("the state's averaged weights are not this policy's (their names differ)")
        self.updates = int(state["updates"])
        for name, value in state["shadow"].items():
            self.shadow[name].copy_(value.to(device=self.shadow[name].device, dtype=self.shadow[name].dtype))


def av2_state(ema=None, adv_scale=None):
    """What ``trainer_state`` keeps under ``av2``: the averaged weights and the running advantage scale; None when
    neither flag is on."""
    if ema is None and adv_scale is None:
        return None
    return {"ema": None if ema is None else ema.state(), "adv_scale": None if adv_scale is None else adv_scale.state()}


def restore_av2(state, path, ema=None, adv_scale=None):
    """Put ``av2_state`` back from a resumed ``state``; a state that lacks what a flag needs is refused."""
    saved = state.get("av2") or {}
    for name, target in (("ema", ema), ("adv_scale", adv_scale)):
        if target is None:
            continue
        if not saved.get(name):
            raise ValueError(f"SPECTRA_PLAN_RESUME={path}: the state carries no {name} (it was written without the flag)")
        target.load(saved[name])


def lr_at(cfg, it: int) -> float:
    """``SPECTRA_PLAN_LR_SCHEDULE=cosine``: the lr of the update that closes instance ``it`` (0-based), LR · (1 +
    cos(π t / T)) / 2 with t = it // BATCH of the run's T = ceil(INSTANCES / BATCH) updates."""
    total = max(1, -(-cfg.instances // cfg.batch))
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * (it // cfg.batch) / total))


def family_weights(instances, t: float = 1.0):
    """``(families, weights)`` of ``SPECTRA_PLAN_SAMPLE=family:<t>``: each instance's family (``fortify.plan_family``
    of its name) and its draw probability, proportional to (the family's size)^−t (t = 1: families equally likely)."""
    from src import fortify
    families = [fortify.plan_family(inst.name) for inst in instances]
    raw = [families.count(family) ** (-float(t)) for family in families]
    total = sum(raw)
    return families, [value / total for value in raw]


def draw_kappa(rng, cfg, points):
    """``SPECTRA_PLAN_KAPPA_POINTS``: with probability p a budget within ±w of one of the listed points (kept inside
    [0.02, 0.98]), else the trainer's uniform draw over KAPPA."""
    centres, prob, width = points
    if rng.random() < prob:
        centre = rng.choice(list(centres))
        return min(0.98, max(0.02, rng.uniform(centre - width, centre + width)))
    return rng.uniform(cfg.kappa_lo, cfg.kappa_hi)


def check_gate(inst, index, cfg, res, budgets, floor=1, zmax=0.0, sigma=0.25):
    """``SPECTRA_PLAN_CHECK`` (audit F14): three plans of ``inst`` cut for real on each budget of the run: the even cut
    and the prior's plan (the sens plan without a residual) at 0.6, and one deep sampled plan at κ 0.35 (scores
    ``sigma`` · ε from a generator seeded by SEED and ``index``, decoded as the untrained agent's sample is). A plan
    fails when its analytic kept size is off the real cut's by more than 1e-6 or the masked cut's Δ val by more than
    0.05 point, the thresholds of the trainer's own check. One line per network and budget, with the dead filters per
    group; returns the failures as ``["<net> <plan> <budget>"]``. Every global generator is put back, so the extra
    cuts leave the run that follows as it is without the flag."""
    from src import alloc_walk
    snapshot = alloc_walk.rng_snapshot()
    failed = []
    try:
        dead = inst.dead_by_group()
        for budget in budgets:
            cm, prior = inst.cost_model(budget), prior_of(inst, res, budget)
            plans = [("even k=0.60", inst.reference("uniform", 0.6, cfg.k_min, budget=budget, min_width=floor)[0])]
            if prior is not None:
                plans.append(("prior k=0.60",
                              plan_agent.scale_decode(prior, cm, 0.6, cfg.k_min, min_width=floor)[0]))
            elif inst.sens is not None:
                plans.append(("sens k=0.60", inst.reference("sens", 0.6, cfg.k_min, budget=budget, min_width=floor)[0]))
            generator = torch.Generator().manual_seed(cfg.seed * 7919 + int(index))
            z = (float(sigma) * torch.randn(inst.n_groups, generator=generator)).tolist()
            plans.append(("deep sample k=0.35", _decode(z, cm, 0.35, cfg, floor, prior, zmax)[0]))
            parts, bad = [], []
            for name, widths in plans:
                chk = inst.check_widths(widths, budget)
                gap_k, gap_r = abs(chk["analytic"] - chk["real"]), abs(chk["masked_r"] - chk["real_r"])
                if gap_k > 1e-6 or gap_r > 0.05:
                    bad.append(name)
                    failed.append(f"{inst.name} {name} {budget}")
                parts.append(f"{name} kept analytic x{chk['analytic']:.4f} real x{chk['real']:.4f} cut val Δ masked "
                             f"{chk['masked_r']:+.2f} real {chk['real_r']:+.2f}")
            utils.print_flush(
                f"[plan] check3 {inst.name}{_budget_tag(budget)}: " + " | ".join(parts) + " | dead filters by group "
                + (" ".join(f"{row}:{n}" for row, n in sorted(dead.items())) or "none")
                + (" | WARNING: the masked reward or the analytic size does not match the real cut (" + ", ".join(bad)
                   + ")" if bad else " | ok"))
    finally:
        alloc_walk.rng_restore(snapshot)
    return failed


def run(env, shard):
    cfg = config()
    floor = min_width()
    mode, alpha, sigma0 = residual()
    res = None if mode == "off" else (mode, alpha)
    zmax = residual_zmax() if res else 0.0  # the trust region; 0 = every residual decode as before
    init = init_path()
    init_blob = plan_agent.read_blob(init) if init else None  # read now: a wrong path fails before the origins are read
    keep, resume = state_enabled(), resume_path()
    state = read_state(resume) if resume else None  # read now too: a wrong path or file fails before the origins
    budgets = plan_agent.BUDGETS if cfg.budget == "mixed" else (cfg.budget,)
    menu = menu_settings(mode, budgets)  # v36; None = off: a wrong menu fails here, before the origins are read
    av2 = av2_settings(cfg, res, menu)  # AGENT-V2; {} = every flag unset: a wrong flag fails here too
    share_floor = plan_agent.MENU_SHARE_FLOOR  # what the mean plan drops; the run's own under MENU_BOUND / SHARE_FLOOR
    if menu is not None and ("menu_bound" in av2 or "menu_share_floor" in av2):
        share_floor = plan_agent.menu_floor(menu["init"], len(menu["kinds"]), av2.get("menu_bound", 0.0),
                                            av2.get("menu_share_floor"))
        av2["menu_share_floor"] = share_floor
    with_cost = res is not None and "sens_cost" in {plan_agent.prior_kind(mode, b) for b in budgets}
    origin_kw = {}  # what the menu's rules add to each origin's read; nothing without a menu
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
                      + (" state=1" if keep else "") + (f" resume={resume}" if resume else "")
                      + av2_line(av2))
    if menu is not None:
        psi0 = [menu["init"]] + [0.0] * (len(menu["kinds"]) - 1)
        if "menu_bound" in av2:  # AGENT-V2: the untrained head's logits under the bound
            psi0 = plan_agent.menu_bounded(psi0, av2["menu_bound"])
        shares0 = plan_agent.menu_shares(psi0)
        if max(shares0[1:]) >= share_floor:  # the default init is under the floor: no line
            utils.print_flush(f"[plan] menu WARNING: SPECTRA_PLAN_MENU_INIT={menu['init']:g} starts the anchor at a "
                              f"share of {shares0[0]:.4f}, the other rules at or above the mean plan's share floor "
                              f"{share_floor:g}: the mean plan does not start on the prior's plan")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    instances = []
    gate_failed = []  # AGENT-V2 (CHECK): the plans whose real cut does not match, over every network
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
        if "check" in av2:  # AGENT-V2 (F14): three plans cut for real; the job's generators are put back after them
            gate_failed += check_gate(inst, len(instances), cfg, res, budgets, floor, zmax, sigma_at(cfg, 0, scale))
        if menu is not None:  # every rule's weights and plan once per budget: a missing measurement fails here
            tick, parts = time.perf_counter(), []
            for b in budgets:
                at = menu_at(inst, menu, res, 0.6, cfg, b, floor)
                parts.append(f"k=0.60{_budget_tag(b)} rule plans' params kept "
                             + " ".join(f"x{inst.pm.kept(widths):.3f}" for widths, _info in at["plans"]))
            seconds = float((getattr(inst, "curves", None) or {}).get("seconds", 0.0)) + time.perf_counter() - tick
            utils.print_flush(f"[plan] menu setup {inst.name}: {','.join(menu['kinds'])} | " + " | ".join(parts)
                              + f" | {seconds:.1f}s")
        instances.append(inst)
    if not instances:
        raise RuntimeError(f"SPECTRA_PLAN_NETS={cfg.nets}: no test network matched")
    if "check" in av2:  # AGENT-V2: one verdict line for the gates to read; refuse stops the run on a failure
        utils.print_flush(f"[plan] check gate: {'FAIL' if gate_failed else 'PASS'} ({len(instances)} networks, "
                          f"three plans each" + (": " + "; ".join(gate_failed) if gate_failed else "") + ")")
        if gate_failed and av2["check"] == "refuse":
            raise RuntimeError(f"SPECTRA_PLAN_CHECK=refuse: the masked cut or the analytic size does not match the "
                               f"real cut for {len(gate_failed)} plans: " + "; ".join(gate_failed))
    widths = {inst.feature_dim for inst in instances}
    if len(widths) != 1:
        raise RuntimeError(f"networks disagree on the token width: {sorted(widths)}")
    device = instances[0].device
    head_kw = {"menu": len(menu["kinds"]), "menu_init": menu["init"]} if menu is not None else {}
    for key in ("menu_bound", "menu_kappa"):  # AGENT-V2: the bounded logits and the budget bins, when on
        if key in av2:
            head_kw[key] = av2[key]
    policy = plan_agent.PlanPolicy(widths.pop(), cfg.encoder, **head_kw).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    folder = out_dir()
    log_path = os.path.join(folder, "plan_train.jsonl")
    from src.feature_standardizer import resolve_standardizer_path
    meta = {"tokens": cfg.tokens, "zero": list(cfg.zero), "k_min": cfg.k_min, "proxy": cfg.proxy,
            "budget": cfg.budget, "nets": [inst.name for inst in instances], "config": asdict(cfg),
            "standardizer": resolve_standardizer_path(), "actor": os.environ.get("SPECTRA_ACTOR_CHECKPOINT_PATH", ""),
            **({"min_width": floor} if floor > 1 else {}),
            **({"residual": mode, "residual_alpha": alpha, "residual_sigma": sigma0} if res else {}),
            **({"residual_zmax": zmax} if zmax > 0 else {}),
            **({"menu": list(menu["kinds"]), "menu_anchor": menu["kinds"][0], "menu_init": menu["init"],
                "menu_sigma": menu["sigma"], "menu_share_floor": share_floor} if menu else {}),
            **({"fs_lr": plan_agent.fs_lr()} if cfg.proxy.startswith("fs") else {}),
            **({"init": init} if init else {}),
            **av2_meta(av2)}  # AGENT-V2: one key per flag that is on; nothing with none on
    if av2.get("contract"):  # AGENT-V2 (F15): the contract the states above were built under
        meta["state_contract"] = plan_agent.state_contract(instances[0].spans)
    if init_blob is not None:
        warm_start(policy, init_blob, meta, init)

    def log(record):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def save(tag, final=False):
        path = os.path.join(folder, f"policy_{tag}.pt")
        plan_agent.save_policy(policy, path, **meta)
        plan_agent.save_policy(policy, os.path.join(folder, "policy_latest.pt"), **meta)
        if ema is not None:  # AGENT-V2: the averaged weights beside the last iterate (the tagged copy at the end only)
            averaged = {**meta, "weights": "ema", "ema_updates": ema.updates}
            if final:
                plan_agent.save_weights(policy, ema.shadow, os.path.join(folder, f"policy_ema_{tag}.pt"), **averaged)
            plan_agent.save_weights(policy, ema.shadow, os.path.join(folder, "policy_ema_latest.pt"), **averaged)
        return path

    state_path = os.path.join(folder, "trainer_state.pt")
    chain = (list(state.get("chain") or []) + [resume]) if state is not None else []
    last_state = [None]  # the count trainer_state.pt was last written at

    def save_state(count):
        write_state(state_path, trainer_state(policy, optimizer, rng, count, meta, chain, av2_state(ema, adv_scale)))
        last_state[0] = count
        into = count % cfg.batch
        utils.print_flush(f"[plan] state {state_path} at {count} instances"
                          + (f", {into} of {cfg.batch} into a batch" if into else ""))

    optimizer.zero_grad()
    done = start = restore(state, resume, cfg, meta, policy, optimizer, rng) if state is not None else 0
    # AGENT-V2: what the flags that are on need in the loop; with none on every name below is None / False / off and
    # the loop runs the statements it ran before
    adv_mode = av2.get("adv")  # None, or fortify.plan_adv()'s (mode, value)
    adv_scale = RunningScale(adv_mode[1]) if adv_mode and adv_mode[0] == "global" else None
    ema = PolicyEma(policy, av2["ema"]) if "ema" in av2 else None
    if state is not None and (ema is not None or adv_scale is not None):
        restore_av2(state, resume, ema, adv_scale)
    mirror, split, gnorm = bool(av2.get("mirror")), bool(av2.get("split")), av2.get("gnorm")
    beta, anneal, menu_log = av2.get("menu_entropy", 0.0), bool(av2.get("menu_sigma_anneal")), bool(av2.get("menu_log"))
    guard, cosine, points = "fs_guard" in av2, av2.get("lr_schedule") == "cosine", av2.get("kappa_points")
    own_loss = split or gnorm is not None or beta > 0 or anneal  # the loss is built part by part below
    pick = None
    if "sample" in av2:
        families, pick = family_weights(instances, av2["sample"])
        sizes = {family: families.count(family) for family in sorted(set(families))}
        utils.print_flush(f"[plan] sample family t={av2['sample']:g}: " + ", ".join(
            f"{family} {n} nets, {sum(w for f, w in zip(families, pick) if f == family):.3f}"
            for family, n in sizes.items()))
    for it in range(start, cfg.instances):
        t0 = time.perf_counter()
        inst = rng.choice(instances) if pick is None else rng.choices(instances, weights=pick)[0]
        kappa = rng.uniform(cfg.kappa_lo, cfg.kappa_hi) if points is None else draw_kappa(rng, cfg, points)
        budget = cfg.budget if cfg.budget != "mixed" else rng.choice(plan_agent.BUDGETS)
        sigma = sigma_at(cfg, it, scale)
        seed = cfg.seed * 100003 + it
        batches = inst.batches(cfg.proxy, seed)
        mu = policy(inst.state_at(kappa, budget), inst.token_mask, inst.token_k, inst.n_groups)
        generator = torch.Generator(device=mu.device).manual_seed(seed)
        kz = cfg.k // 2 if split else cfg.k  # the plans that draw the scores (SPLIT: the first half)
        if not mirror:
            z = plan_agent.sample_scores(mu, sigma, kz, generator)
        else:
            z = plan_agent.sample_scores(mu, sigma, kz, generator, mirror=True)
        rewards, kept = [], []
        cm = inst.cost_model(budget)
        prior = prior_of(inst, res, budget)
        at, shares = menu_at(inst, menu, res, kappa, cfg, budget, floor), None  # None off the menu: all as before
        sigma_u = menu["sigma"] if menu is not None else None
        if at is not None:  # v36: the menu logits, drawn after the scores so the scores' noise is the run's without it
            psi = policy.menu_logits(inst.state_at(kappa, budget), at["x"])
            if anneal:  # AGENT-V2: sigma_u follows the scores' schedule
                sigma_u = menu["sigma"] * sigma / sigma_at(cfg, 0, scale)
            if not (mirror or split or anneal):
                u = plan_agent.sample_scores(psi, menu["sigma"], cfg.k, generator)
                shares = [plan_agent.menu_shares(row) for row in u.tolist()]
            else:
                u = plan_agent.sample_scores(psi, sigma_u, cfg.k - kz if split else cfg.k, generator, mirror=mirror)
                shares = [plan_agent.menu_shares(row) for row in u.tolist()]
                if split:  # the score half plays the shares the frozen agent plays: the mean plan's
                    shares = [plan_agent.menu_shares(psi.tolist(), share_floor)] * kz + shares
        mean_scores = mu.detach().tolist() if split else None  # SPLIT: what the logit half decodes
        flagged = []  # FS_GUARD: the plans whose fine-tune diverged
        for k in range(cfg.k):
            plan_widths, info = _decode(z[k].tolist() if k < kz else mean_scores, cm, kappa, cfg, floor, prior, zmax,
                                        None if shares is None else (shares[k], at["rules"]))
            rewards.append(inst.reward(plan_widths, cfg.proxy, batches))
            kept.append(info["kept"])
            if guard and getattr(inst, "diverged", False):
                flagged.append(k)
        if it == 0 and getattr(inst, "recipe_line", None):  # the ft<E> reward's handler recipe, once per run
            utils.print_flush(f"[plan] ft recipe {inst.name}: {inst.recipe_line}")
            meta["recipe_line"] = inst.recipe_line
        if flagged:  # AGENT-V2 (F11): a diverged plan takes the median of its instance's other plans (0 gradient
            sound = sorted(rewards[k] for k in range(cfg.k) if k not in flagged)  # weight under a mean baseline)
            fill = statistics.median(sound) if sound else 0.0
            rewards = [fill if k in flagged else value for k, value in enumerate(rewards)]
        r = torch.tensor(rewards, device=mu.device, dtype=mu.dtype)
        if adv_mode is None and not split:
            adv = r - r.mean()
            if cfg.norm_adv:
                adv = adv / (r.std() + 1e-6)
        elif not split:  # AGENT-V2: SPECTRA_PLAN_ADV's shaping in place of the division
            adv = advantages(r, cfg, adv_mode, adv_scale)
        else:  # AGENT-V2 (F3): each half is scored against its own siblings only
            adv = torch.cat([advantages(r[:kz], cfg, adv_mode, adv_scale), advantages(r[kz:], cfg, adv_mode, adv_scale)])
        # The action is the raw z ~ N(mu, sigma²), under the trust region too: zmax squashes it only inside the decode,
        # a deterministic map from action to plan, so log_prob(z) is still the exact log-density of what was sampled and
        # the REINFORCE estimate stays unbiased (no tanh Jacobian: no density is taken of the squashed scores).
        if not own_loss:
            loss = -(adv.detach() * plan_agent.log_prob(z, mu, sigma)).mean() / cfg.batch
            if at is not None:  # v36: the action is (z, u), drawn independently; the softmax sits inside the decode too
                loss = -(adv.detach() * plan_agent.menu_log_prob(z, mu, sigma, u, psi, menu["sigma"])).mean() / cfg.batch
        else:  # AGENT-V2: the same estimator with the scores' term over sqrt(G) or G (GNORM), the two halves scored
            # apart (SPLIT), the logits' term at the annealed sigma_u, and the entropy bonus on the mean shares
            lp_z = plan_agent.log_prob(z, mu, sigma)
            if gnorm is not None:
                lp_z = lp_z / (math.sqrt(inst.n_groups) if gnorm == "sqrt" else float(inst.n_groups))
            adv = adv.detach()
            if at is None:
                loss = -(adv * lp_z).mean() / cfg.batch
            elif not split:
                loss = -(adv * (lp_z + plan_agent.log_prob(u, psi, sigma_u))).mean() / cfg.batch
            else:
                loss = (-(adv[:kz] * lp_z).mean() - (adv[kz:] * plan_agent.log_prob(u, psi, sigma_u)).mean()) / cfg.batch
            if beta > 0 and at is not None:
                beta_t = beta * max(0.0, 1.0 - it / max(1.0, cfg.sigma_decay * cfg.instances))
                entropy = plan_agent.menu_entropy(psi)
                loss = loss - beta_t * entropy / cfg.batch
        loss.backward()
        grad = None
        if (it + 1) % cfg.batch == 0:
            grad = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
            if cosine:  # AGENT-V2: the lr falls to 0 over the run's updates
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(cfg, it)
            optimizer.step()
            if ema is not None:
                ema.update(policy)
            optimizer.zero_grad()
        record = {"it": it, "net": inst.name, "kappa": kappa, "budget": budget, "sigma": sigma, "rewards": rewards,
                  "kept": kept, "mu_std": float(mu.detach().std()), "loss": float(loss) * cfg.batch, "grad": grad}
        if res:
            record["residual"], record["prior"] = mode, plan_agent.prior_kind(mode, budget)
        if zmax > 0:  # groups whose mean is past the bound, where tanh's slope (< 0.42) starts to thin their signal
            record["mu_beyond_zmax"] = int((mu.detach().abs() > zmax).sum())
        if shares is not None:  # v36: the sampled plans' mean share per rule, in the menu's order
            record["menu_pi"] = [sum(s[j] for s in shares) / len(shares) for j in range(len(menu["kinds"]))]
        line = (f"[plan] it={it + 1}/{cfg.instances} {inst.name} k={kappa:.3f}{_budget_tag(budget)} sigma={sigma:.3f} "
                f"| plans mean {statistics.mean(rewards):+.2f} max {max(rewards):+.2f} min {min(rewards):+.2f} kept "
                f"x{statistics.mean(kept):.3f} | mu std {record['mu_std']:.3f}")
        if zmax > 0:
            line += f" | mu beyond zmax {record['mu_beyond_zmax']}/{inst.n_groups}"
        if av2:  # AGENT-V2: what the flags that are on add to the record and the line (nothing with none on)
            if split:
                record["split"] = kz
            if guard:
                record["diverged"] = len(flagged)
                line += f" | diverged {len(flagged)}" if flagged else ""
            if adv_scale is not None:
                record["adv_scale"] = adv_scale.read()
            if cosine and grad is not None:
                record["lr"] = optimizer.param_groups[0]["lr"]
            if at is not None and (anneal or beta > 0):
                record["menu_sigma"] = sigma_u
            if at is not None and beta > 0:
                record["menu_entropy"] = float(entropy)
            if at is not None and menu_log:  # F2: the mean logits, their range and every sampled plan's shares
                logits = [float(v) for v in psi.detach().tolist()]
                record["menu_psi"], record["menu_psi_range"] = logits, max(logits) - min(logits)
                record["menu_shares"] = [[round(float(p), 4) for p in row] for row in shares]
                line += f" | psi range {record['menu_psi_range']:.2f}"
        if (it + 1) % cfg.ref_every == 0 or it == 0:
            plan_widths, info = mean_plan(policy, inst, kappa, cfg, budget, floor, prior, zmax, at, share_floor)
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
            if keep:
                save_state(done)
        over = cfg.max_minutes and (time.perf_counter() - started) / 60.0 >= cfg.max_minutes
        if over and not (keep and done % cfg.batch):  # with the state on, only at a batch boundary: no partial step
            utils.print_flush(f"[plan] MAX_MINUTES={cfg.max_minutes:g} reached after {done} instances")
            break
    if keep and last_state[0] != done:  # the trajectory state before the closing partial step: a resume re-finishes it
        save_state(done)
    if done % cfg.batch:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if cosine and done:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(cfg, done - 1)
        optimizer.step()
        if ema is not None:
            ema.update(policy)
        optimizer.zero_grad()
    path = save(f"it{done:05d}", final=True)
    utils.print_flush(f"[plan] saved {path} after {done} instances, {(time.perf_counter() - started) / 60.0:.1f} min")
    summary(policy, instances, cfg, log, floor, res, zmax, menu, share_floor)
    utils.print_flush(f"[plan] DONE {done} instances in {(time.perf_counter() - started) / 60.0:.1f} min")
