"""
Allocation-following eval walk (``SPECTRA_EVAL_POLICY=alloc``; no actor, no training).

A0 (ledger §201 / §204 / §205) cut every coupled group once to a sensitivity-weighted keep and
recovered for 40 epochs: at equal kept parameters that beat a uniform allocation on both ResNet-56
cells. This policy asks whether the lever survives the walk. Once per network, on the origin, it
fixes a target width for every group — ``SPECTRA_ALLOC_KIND=uniform`` (one fraction for all),
``sens`` (keep ∝ (s / median s)^α, s = the calibration-loss rise when the group alone is cut to
half, the measurement behind the v10 sensitivity channels) or ``inner`` (every coupled group with
more than one producer, i.e. a residual stream, held at full width and one fraction for the rest:
the structure the sens plans take on the thin ResNets, ledger §229) — with the scale bisected so that
cutting every group once to its target keeps ``target − SPECTRA_ALLOC_UNDERSHOOT`` of the
parameters. ``widths`` bisects nothing: every group takes the width ``SPECTRA_ALLOC_WIDTHS`` (JSON
``{module name: out channels}``, e.g. another method's pruned net) names for its producers, so a run
isolates that allocation under this walk's ranking and recovery; the fixed target then only says
where the walk stops. ``sample`` draws one plan the way the plan-as-action agent will: the weights
of ``SPECTRA_ALLOC_SAMPLE_AROUND`` (sens or uniform) times exp(σ ε) per group, ε ~ N(0, 1) seeded by
``SPECTRA_ALLOC_SAMPLE_SEED``, σ = ``SPECTRA_ALLOC_SAMPLE_SIGMA``, then bisected like ``sens``.
``agent`` takes the mean plan of the frozen plan agent at ``SPECTRA_PLAN_AGENT`` (``src/plan_agent.py``).
``agent_sample`` draws one plan around that mean as the agent samples in training, z = μ + σ ε
with ε ~ N(0, I) seeded by ``SPECTRA_ALLOC_SAMPLE_SEED`` and σ = ``SPECTRA_ALLOC_SAMPLE_SIGMA``,
decoded like the mean (D-PROXY-2).
``randbest`` (v23) is the per-network random-search baseline of Li et al. (CVPR 2022, "Revisiting Random Channel
Pruning") in the agent's own action space, set beside the frozen agent: N = ``SPECTRA_ALLOC_RANDBEST_N`` (100) plans
z_g ~ N(0, σ²) per group, σ = ``SPECTRA_ALLOC_RANDBEST_SIGMA`` (1.0), from a CPU generator seeded by
``SPECTRA_ALLOC_SAMPLE_SEED`` and the run's seed (``randbest_seed``), each decoded as ``agent`` decodes the mean plan
(``plan_agent.decode`` at the same target, k_min, width floor and budget) and scored on the val half with the proxy
``SPECTRA_ALLOC_RANDBEST_PROXY`` (no default: cut, bn<N>, fs<S> or ft<E>, the trainer's reward through
``plan_proxies.PlanScorer``, on the batches ``SPECTRA_EVAL_PROXY_SEED`` draws). The best val (ties to the lower draw)
is the plan the walk then follows exactly as it follows the agent's. The TEST half is never read, and the global RNG is
put back after the search, so the walk and the final fine-tune draw as they do under ``agent``.
``fallback`` (v27) plans twice on the origin, with the frozen agent exactly as ``agent`` does and with a hand rule,
``SPECTRA_ALLOC_FALLBACK_PRIOR`` (``sens`` or ``sens_cost``, no default; "the prior") exactly as that rule's own row does. It
scores both plans on the val half with ``SPECTRA_ALLOC_FALLBACK_PROXY`` (a comma-separated list of cut, bn<N>, fs<S> or
ft<E>, no default; ``plan_proxies.PlanScorer`` on the batches ``SPECTRA_EVAL_PROXY_SEED`` draws) and follows the agent's plan
unless the prior's scores higher on the FIRST proxy by more than ``SPECTRA_ALLOC_FALLBACK_MARGIN`` (val accuracy as a
fraction, default 0.0; a tie goes to the agent). Every proxy is scored and reported, the first one decides. The contract is
exactness: each plan is made from the same global RNG state and the same origin weights as its own row makes it (the
origin's weights are checked after each plan and after the scoring), the RNG is then put back to the state the picked arm
alone leaves, and the picked plan is returned unchanged (``info["kind"]`` stays the arm's own, one key ``fallback`` is
added), so the walk and the final fine-tune run as under that arm alone and, under ``SPECTRA_EVAL_FT_DETERMINISTIC=1``, a
fallback row equals the picked arm's row seed for seed. The TEST half is never read. Any effect ``plan_for_env`` has on
the env itself stays even when the prior is picked, so the exactness against the prior's own row is a check registered on
the GPU, not a guarantee of this code.
``sens_cost_at`` (v30) plans as ``sens_cost`` does, then measures every group again at the depth that plan cuts it
(``min(planned keep, 0.5)``, so a group kept at half width or more is probed at half as before), repeats
``SPECTRA_ALLOC_PROBE_ROUNDS`` times and bisects the cost-aware weights of the last measurement.
``sens_curve`` (v30) measures each group's loss rise and saving at the keeps ``SPECTRA_ALLOC_CURVE_KEEPS``, makes the
curve monotone, picks every group's keep by one multiplier λ that minimises loss − λ · saving, and lands the result
with the same bisection as every rule.
``sens_thresh`` (v31) measures the same curves and picks every group's keep by one global threshold τ on the loss rise
(the smallest keep whose loss is at most τ: Li et al. 2017's per-layer sensitivity rule, as PaddleSlim and NNI's
SensitivityPruner run it), with τ chosen so the estimated saving meets the target, and lands the result the same way.
``SPECTRA_ALLOC_SENS_KEEP`` (unset) moves the keep at which ``sens`` and ``sens_cost`` measure every group (v31).
``global_mag`` is the global magnitude ranking (one importance threshold for every layer, the field's usual default;
Torch-Pruning's ``global_pruning=True``) as an allocation, with no data and no probe. A channel's magnitude is the score
the plan's own L1 cut ranks it by (``pruning.group_importance``: the L1 norm of its filter in each layer that produces
the group's channels, over that layer's largest, summed over those layers; the BN scales and the consumers' input
slices are not read, because the cut does not rank by them), divided by the mean over its group (Torch-Pruning's
``normalizer="mean"``). One threshold τ serves all groups: a group keeps its channels that score above τ, at the
smallest keep on ``sens_thresh``'s grid that removes no channel above τ. τ is chosen so that the budget's cost model
(``plan_agent.ParamModel`` / ``FlopModel``) keeps the target at those widths, and the result is landed with the same
bisection as every rule.
``lamp`` is LAMP (Lee et al., ICLR 2021) in its structured form, per channel of a coupled group as Torch-Pruning's
``LAMPImportance`` applies it: the same magnitudes squared and rescaled within each group (sorted ascending,
score_u = w_u / Σ_{v ≥ u} w_v, their eq. 1), then one global threshold exactly as ``global_mag``. Neither score changes
when all of a group's filters are rescaled.
``amc`` is AMC's per-network search (He et al., ECCV 2018) as a comparator, ported from its released code in
``src/amc_search.py``, which is imported only under this kind (queue row AMC, part C). Once per network, on the origin,
a DDPG agent decides one keep per planned group in plan order; AMC's budget rule bounds each keep on
``plan_agent.FlopModel`` so the FLOPs target stays reachable (``SPECTRA_ALLOC_BUDGET=flops`` only, any other budget is
refused); an episode's reward is its plan's val-half accuracy under ``SPECTRA_ALLOC_AMC_PROXY`` (bn8,
``plan_proxies.PlanScorer``). The best episode's keeps (ties to the earlier) are landed by ``_land`` like every rule's
and walked like any plan. ``SPECTRA_ALLOC_AMC_EPISODES`` / ``SPECTRA_ALLOC_AMC_WARMUP`` (unset: AMC's 800 / 100) shorten
a smoke. The TEST half is never read, and the global RNG is put back after the search.
``tilt`` (queue row SAMENET) perturbs a plan this walk has already followed. The base plan is the table
``SPECTRA_ALLOC_TILT_BASE`` (JSON ``{module name: out channels}``, read as ``widths`` reads its own; the ``plan`` file
that ``SPECTRA_EVAL_SAVE_WIDTHS`` writes), and a group's weight is its base keep times exp(z_g). z comes from the JSON
file ``SPECTRA_ALLOC_TILT_Z`` (a list in plan order, or ``{group key: z}``, the key being the group's producer names
sorted and joined by ``+``) or is drawn, z_g = σ ε_g with σ = ``SPECTRA_ALLOC_TILT_SIGMA`` and ε from a CPU generator
seeded by ``SPECTRA_ALLOC_SAMPLE_SEED``. The scale is bisected as for ``sens`` and ``sample`` (clip(c · w, floor, 1),
the same floor, iterations and tolerance) to the size the base plan itself keeps under the same one-shot cut, with
c = 1 tried first: a tilt moves the budget between groups, and z = 0 returns the base plan's widths exactly.
``SPECTRA_ALLOC_TILT_TARGET=run`` bisects to the run's own plan target instead, exactly as ``sens`` does. No batch is
drawn, and the scores' draw has its own generator (the global RNG moves only as every plan's trial cuts move it).
``SPECTRA_ALLOC_TILT_PARAMS=equal[:tol]`` (amendment A1 of the row; unset: off; FLOPs budget only) holds a tilt at the
base plan's params as well as at its FLOPs. The weights become base keep times exp(z_g + λ d_g), with d_g the log of the
group's FLOPs per kept channel over its params per kept channel at the base widths, from the cost models, centred by
its FLOPs-weighted mean. Each λ is landed exactly as the tilt is, and λ is searched on both sides of 0 (|λ| up to 2)
until the plan the walk aims at, the tilt after any grid rounding, keeps within ``tol`` (0.01) of the params the base
table keeps, relative to them. λ is 0 exactly when the tilt as it is lies within ``tol``, so z = 0 is still the base
plan; when whole widths or the floors keep every λ outside ``tol``, the closest one is taken and the miss is logged.
The search runs on the cost models and the λ it finds is cut for real; a group whose cut is masked (its width stays)
is found by one real half cut and kept at its width on the cost models.
What is held is the params of the network the walk is predicted to land on (``equal[:tol]:landed``, the default;
amendment A2): the walk stops at the FLOPs target before every group has arrived, so the landed params are not the
plan's, and ``TiltWalk`` runs the walk on the cost models to say where it stops, for the tilt and for the base plan
alike. That landing jumps with λ in narrow spots, so this hold scans λ more finely near 0 (step 0.0125 to 0.2, then
0.05 to 1, then 0.1 to 2). ``equal[:tol]:planned`` holds the aimed plan's own one-shot params instead (amendment A1,
with its even scan of 0.05).
``SPECTRA_ALLOC_BUDGET=flops`` (default params) measures the plan's kept fraction, its target and the stall
check in MACs (``utils.calc_flops``) instead of parameters, the T0-F variant.
Each decision then plays the legal cut whose resulting width is closest to its group's
target (ties to the milder cut, identity once it is there). Group-once, the recovery and the
fixed-target landing are the walk's own, so a row differs from greedy / mild only in which groups
keep being cut. The undershoot makes the walk cross the target before every group has arrived;
should it still stall above the target for a whole pass, it falls back to the strongest legal cut.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import statistics
import time

import numpy as np
import torch

import src.channel_groups as channel_groups
import src.fortify as fortify
import src.group_sensitivity as group_sensitivity
import src.pruning as pruning
import src.utils as utils
from NetworkFeatureExtraction.src.ModelWithRows import ModelWithRows

KINDS = ("uniform", "sens", "inner", "widths", "sample", "agent", "agent_sample", "sens_cost", "randbest", "fallback",
         "sens_cost_at", "sens_curve", "sens_thresh", "global_mag", "lamp")
MAGNITUDE_KINDS = ("global_mag", "lamp")
# ``amc`` (queue row AMC, part C; ``src/amc_search.py``) is registered beside ``KINDS``, not appended to it:
# ``tests/test_alloc_lamp_global.py`` pins the tuple's tail to the magnitude kinds.
AMC_KINDS = ("amc",)
# ``tilt`` (queue row SAMENET) is registered beside ``KINDS`` for the same reason: a kind only its own flag reaches.
TILT_KINDS = ("tilt",)
FALLBACK_PRIORS = ("sens", "sens_cost")
# the further rules a list of two or more priors may name: ``uniform`` (v32, FB-BENCH) and, v37 (queue row PICK, default
# off: only names v36 refused), the threshold rule and the two magnitude rules. Each is planned exactly as its own row
# plans it (the row's calibration draw, then ``plan_targets``), so the fallback's exactness contract is unchanged.
FALLBACK_MORE = ("uniform", "sens_thresh", "global_mag", "lamp")
SAMPLE_AROUND = ("sens", "uniform")
BUDGETS = ("params", "flops")


def kind() -> str:
    """``SPECTRA_ALLOC_KIND`` (sens): the allocation the walk follows."""
    name = os.environ.get("SPECTRA_ALLOC_KIND", "sens").strip().lower() or "sens"
    if name not in KINDS and name not in AMC_KINDS and name not in TILT_KINDS:
        raise ValueError(f"SPECTRA_ALLOC_KIND={name!r}: expected one of {KINDS + AMC_KINDS + TILT_KINDS}")
    return name


def budget() -> str:
    """``SPECTRA_ALLOC_BUDGET`` (params): what the plan's kept fraction and the walk's target are measured in."""
    name = os.environ.get("SPECTRA_ALLOC_BUDGET", "params").strip().lower() or "params"
    if name not in BUDGETS:
        raise ValueError(f"SPECTRA_ALLOC_BUDGET={name!r}: expected one of {BUDGETS}")
    return name


def cut_ranking(ranking):
    """
    The ranking ``env.step`` takes for a cut the walk plays: ``SPECTRA_CUT_RANK`` (``fortify.cut_rank``; ``random`` is
    the control that keeps the walk's widths and picks the surviving channels at random) when it is set, else
    ``ranking`` unchanged. The eval loop applies it to its own ``env.step`` calls only, under
    ``SPECTRA_EVAL_POLICY=alloc``: ``plan_targets`` / ``cut_to`` / ``group_sensitivity`` and the plan agent cut with an
    explicit ``l1``, so the plan (and the agent's tokens, which carry sensitivities measured with the ranking in force)
    never see it.
    """
    mode = fortify.cut_rank()
    return mode if mode else ranking


def alpha() -> float:
    """``SPECTRA_ALLOC_ALPHA`` (0.5): sensitivity power of ``sens``, A0's default."""
    return float(os.environ.get("SPECTRA_ALLOC_ALPHA", "0.5"))


def undershoot() -> float:
    """``SPECTRA_ALLOC_UNDERSHOOT`` (0.02): the plan keeps this much less than the walk's target."""
    return max(0.0, float(os.environ.get("SPECTRA_ALLOC_UNDERSHOOT", "0.02")))


def min_keep() -> float:
    """``SPECTRA_ALLOC_MIN_KEEP`` (0.1): no group's planned keep goes below this, A0's floor."""
    return min(1.0, max(0.01, float(os.environ.get("SPECTRA_ALLOC_MIN_KEEP", "0.1"))))


def curve_keeps() -> tuple:
    """``SPECTRA_ALLOC_CURVE_KEEPS`` (0.75,0.5,0.25,0.1): the keeps ``sens_curve`` probes every group at, each in
    (0, 1); a tuple sorted descending without duplicates."""
    raw = os.environ.get("SPECTRA_ALLOC_CURVE_KEEPS", "").strip() or "0.75,0.5,0.25,0.1"
    try:
        values = [float(item) for item in raw.split(",") if item.strip()]
    except ValueError:
        raise ValueError(f"SPECTRA_ALLOC_CURVE_KEEPS={raw!r}: expected comma-separated floats in (0, 1)") from None
    if not values or not all(0.0 < v < 1.0 for v in values):
        raise ValueError(f"SPECTRA_ALLOC_CURVE_KEEPS={raw!r}: expected comma-separated floats in (0, 1)")
    return tuple(sorted(set(values), reverse=True))


def probe_rounds() -> int:
    """``SPECTRA_ALLOC_PROBE_ROUNDS`` (1): how many times ``sens_cost_at`` re-measures at the depth of its last plan."""
    n = int(os.environ.get("SPECTRA_ALLOC_PROBE_ROUNDS", "1"))
    if n < 1:
        raise ValueError(f"SPECTRA_ALLOC_PROBE_ROUNDS={n}: need at least one round")
    return n


def sens_keep():
    """``SPECTRA_ALLOC_SENS_KEEP`` (unset): the scalar keep at which the rules ``sens`` and ``sens_cost`` measure every
    group's loss rise (``group_sensitivity.SENS_KEEP``, 0.5, when unset); a float in (0, 1). v31: PROBE's scq arm (the
    budget's average keep, e.g. sqrt(0.15) = 0.387 at params 0.15). It also moves the prior an agent's residual or a
    fallback plans with, so set it on rule arms only. ``sens_cost_at``'s first plan (a ``sens_cost`` plan) also uses it;
    its own re-probe keeps ``group_sensitivity.SENS_KEEP``. None when unset or blank."""
    raw = os.environ.get("SPECTRA_ALLOC_SENS_KEEP", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"SPECTRA_ALLOC_SENS_KEEP={raw!r}: expected a float in (0, 1)") from None
    if not 0.0 < value < 1.0:
        raise ValueError(f"SPECTRA_ALLOC_SENS_KEEP={raw!r}: expected a float in (0, 1)")
    return value


def sample_around() -> str:
    """``SPECTRA_ALLOC_SAMPLE_AROUND`` (sens): the plan whose weights ``sample`` perturbs."""
    name = os.environ.get("SPECTRA_ALLOC_SAMPLE_AROUND", "sens").strip().lower() or "sens"
    if name not in SAMPLE_AROUND:
        raise ValueError(f"SPECTRA_ALLOC_SAMPLE_AROUND={name!r}: expected one of {SAMPLE_AROUND}")
    return name


def sample_sigma() -> float:
    """``SPECTRA_ALLOC_SAMPLE_SIGMA`` (0.5): std of the per-group noise, on the log-weights of ``sample`` and on
    the scores of ``agent_sample``."""
    return max(0.0, float(os.environ.get("SPECTRA_ALLOC_SAMPLE_SIGMA", "0.5")))


def sample_seed() -> int:
    """``SPECTRA_ALLOC_SAMPLE_SEED`` (0): seed of the noise of ``sample`` and ``agent_sample``."""
    return int(os.environ.get("SPECTRA_ALLOC_SAMPLE_SEED", "0"))


def sample_noise(rows, sigma, seed):
    """``{row: exp(sigma * eps)}``, eps ~ N(0, 1) drawn in ``rows`` order from ``random.Random(seed)``."""
    rng = random.Random(int(seed))
    return {row: math.exp(float(sigma) * rng.gauss(0.0, 1.0)) for row in rows}


def agent_path() -> str:
    """``SPECTRA_PLAN_AGENT``: the plan-agent checkpoint ``agent`` / ``agent_sample`` follows
    (``src/plan_agent.py``)."""
    path = os.environ.get("SPECTRA_PLAN_AGENT", "").strip()
    if not path:
        raise ValueError("SPECTRA_ALLOC_KIND=agent or agent_sample needs SPECTRA_PLAN_AGENT=<plan-agent checkpoint>")
    return path


def randbest_n() -> int:
    """``SPECTRA_ALLOC_RANDBEST_N`` (100): how many random plans ``randbest`` draws and scores."""
    n = int(os.environ.get("SPECTRA_ALLOC_RANDBEST_N", "100"))
    if n < 1:
        raise ValueError(f"SPECTRA_ALLOC_RANDBEST_N={n}: need at least one draw")
    return n


def randbest_sigma() -> float:
    """``SPECTRA_ALLOC_RANDBEST_SIGMA`` (1.0): std of each draw's per-group scores, z_g ~ N(0, σ²)."""
    return max(0.0, float(os.environ.get("SPECTRA_ALLOC_RANDBEST_SIGMA", "1.0")))


def randbest_proxy() -> str:
    """``SPECTRA_ALLOC_RANDBEST_PROXY`` (no default): the proxy every draw is scored with, one of the plan trainer's
    (``plan_agent.parse_proxy``: cut, bn<N>, fs<S>, ft<E>), the one the agent of record trained on."""
    name = os.environ.get("SPECTRA_ALLOC_RANDBEST_PROXY", "").strip().lower()
    if not name:
        raise ValueError("SPECTRA_ALLOC_KIND=randbest needs SPECTRA_ALLOC_RANDBEST_PROXY=<cut | bn<N> | fs<S> | ft<E>>, "
                         "the proxy the plan agent trained on (SPECTRA_PLAN_PROXY)")
    from src import plan_agent
    plan_agent.parse_proxy(name)
    return name


def run_seed(env) -> int:
    """The run's seed (``--seed``, ``SPECTRA_SEED`` in the submit scripts): ``env.conf.seed`` when the env carries one,
    else ``SPECTRA_SEED``, else 0."""
    seed = getattr(getattr(env, "conf", None), "seed", None)
    if seed is None:
        seed = os.environ.get("SPECTRA_SEED", "").strip() or 0
    return int(seed)


def randbest_seed(sample: int, run: int) -> int:
    """One generator seed from ``SPECTRA_ALLOC_SAMPLE_SEED`` and the run's seed: the same pair, the same N plans."""
    return int(sample) * 100003 + int(run)


def randbest_draws(n, n_groups, sigma, seed):
    """``[[z_g]]``: the scores of ``n`` draws, z_g ~ N(0, σ²) independently per group in plan order, from a CPU
    generator seeded by ``seed`` (float64, as ``plan_agent.plan_for_env`` draws ``agent_sample``'s ε); the global RNG
    is untouched. One ``randn`` call per draw, so draw i is the same under any N (a longer search extends a shorter
    one; one call for all N would reorder the stream past torch's 16-element vectorised path)."""
    generator = torch.Generator().manual_seed(int(seed))
    return [(float(sigma) * torch.randn(int(n_groups), generator=generator, dtype=torch.float64)).tolist()
            for _ in range(int(n))]


def fallback_priors() -> list:
    """``SPECTRA_ALLOC_FALLBACK_PRIOR`` (no default): the hand rules ``fallback`` plans with beside the agent, a
    comma-separated list in order (blanks stripped, lower-cased, empty items dropped, no duplicates), each one of
    ``FALLBACK_PRIORS`` (``sens``, ``sens_cost``) and, in a list of two or more, ``FALLBACK_MORE`` too (``uniform``, v32,
    FB-BENCH; ``sens_thresh``, ``global_mag`` and ``lamp``, v37, PICK). One name is the v27 setting exactly (a lone
    further rule stays refused, as ``uniform`` alone was)."""
    raw = os.environ.get("SPECTRA_ALLOC_FALLBACK_PRIOR", "")
    names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    allowed = FALLBACK_PRIORS + (FALLBACK_MORE if len(names) > 1 else ())
    if not names or any(n not in allowed for n in names):
        bad = next((n for n in names if n not in allowed), raw.strip().lower())
        raise ValueError(f"SPECTRA_ALLOC_KIND=fallback needs SPECTRA_ALLOC_FALLBACK_PRIOR set to one of "
                         f"{allowed}{' (a comma-separated list)' if len(names) > 1 else ''}, got {bad!r}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"SPECTRA_ALLOC_FALLBACK_PRIOR={raw!r}: duplicate priors {repeated}")
    return names


def fallback_prior() -> str:
    """The first of ``fallback_priors``: the rule ``fallback`` plans with beside the agent when one is set, the two-way
    pick's prior when several are."""
    return fallback_priors()[0]


def fallback_detail() -> bool:
    """``SPECTRA_ALLOC_FALLBACK_DETAIL=1`` (default off): ``fallback`` also logs, for every proxy and arm, the val mean
    cross-entropy, the per-image correctness bits and each draw's accuracy (``PlanScorer.score_detail``)."""
    return utils.env_flag("SPECTRA_ALLOC_FALLBACK_DETAIL")


def fallback_proxies() -> list:
    """``SPECTRA_ALLOC_FALLBACK_PROXY`` (no default): the proxies ``fallback`` scores both plans with, a comma-separated
    list in order (blanks stripped, lower-cased, empty items dropped, no duplicates), each one of the plan trainer's
    (``plan_agent.parse_proxy``: cut, bn<N>, fs<S>, ft<E>; v32 adds the judge names ``rc<E>`` and the repeat suffix
    ``ft<E>x<k>`` / ``rc<E>x<k>``, ``parse_proxy(name, judge=True)``). The first decides the pick; every one is scored
    and reported."""
    raw = os.environ.get("SPECTRA_ALLOC_FALLBACK_PROXY", "")
    names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    if not names:
        raise ValueError("SPECTRA_ALLOC_KIND=fallback needs SPECTRA_ALLOC_FALLBACK_PROXY=<cut | bn<N> | fs<S> | ft<E>"
                         "[,...]>, the first one decides the pick")
    from src import plan_agent
    for name in names:
        plan_agent.parse_proxy(name, judge=True)
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"SPECTRA_ALLOC_FALLBACK_PROXY={raw!r}: duplicate proxies {repeated}")
    return names


def fallback_margin() -> float:
    """``SPECTRA_ALLOC_FALLBACK_MARGIN`` (0.0): how far above the agent's val accuracy (a fraction, the scores
    ``PlanScorer`` returns) the prior's plan must score for ``fallback`` to walk the prior's plan."""
    margin = float(os.environ.get("SPECTRA_ALLOC_FALLBACK_MARGIN", "0.0"))
    if not margin >= 0.0:
        raise ValueError(f"SPECTRA_ALLOC_FALLBACK_MARGIN={margin}: need a margin of at least 0")
    return margin


def widths_table() -> dict:
    """``SPECTRA_ALLOC_WIDTHS``: the JSON ``{module name: out channels}`` that ``widths`` copies."""
    path = os.environ.get("SPECTRA_ALLOC_WIDTHS", "").strip()
    if not path:
        raise ValueError("SPECTRA_ALLOC_KIND=widths needs SPECTRA_ALLOC_WIDTHS=<json {module name: out channels}>")
    with open(path, encoding="utf-8") as fh:
        return {str(name): int(width) for name, width in json.load(fh).items()}


def copied_keeps(model, plan, table):
    """
    ``({row: keep}, unnamed)``: a group keeps the narrowest width ``table`` names for its producers, over
    its own width. A group none of whose producers is named stays whole and is counted in ``unnamed``.
    """
    names = {id(module): name for name, module in model.named_modules()}
    keeps, unnamed = {}, 0
    for group, row in plan:
        named = [table[names[id(p)]] for p in group.producers if names.get(id(p)) in table]
        unnamed += not named
        keeps[row] = min(1.0, min(named) / float(group.width)) if named else 1.0
    return keeps, unnamed


def weights(kind_name, sens, a=0.5, cost=None):
    """Relative keep per row before scaling: 1 (uniform, inner) or (s / median)^α, s floored at 5 % of the median.

    ``sens_cost`` (v16) uses (v / median v)^α with v = s / c, s and the cost c (saved by the same half cut) each
    floored at 5 % of its median; ``cost`` is its ``{row: saved}`` and must name the rows of ``sens``.
    """
    keys = list(sens)
    if kind_name in ("uniform", "inner"):
        return {key: 1.0 for key in keys}
    if kind_name == "sens_cost":
        if cost is None or set(cost) != set(keys):
            raise ValueError("weights('sens_cost') needs cost={row: saved} naming the same rows as sens")

        def floored(raw):
            positive = [max(0.0, float(raw[key])) for key in keys]
            floor = max(1e-6, 0.05 * statistics.median(positive))
            return {key: max(float(raw[key]), floor) for key in keys}

        s, c = floored(sens), floored(cost)
        v = {key: s[key] / c[key] for key in keys}
        mid = statistics.median(v.values())
        return {key: (v[key] / mid) ** a for key in keys}
    positive = [max(0.0, float(sens[key])) for key in keys]
    floor = max(1e-6, 0.05 * statistics.median(positive))
    s = {key: max(float(sens[key]), floor) for key in keys}
    mid = statistics.median(s.values())
    return {key: (s[key] / mid) ** a for key in keys}


def cut_to(model, plan, keeps, input_shape):
    """A copy of ``model`` with every planned group cut once (L1) to its keep; 1 leaves it alone."""
    from src.NetworkEnv import prune_current_model
    mwr = ModelWithRows(copy.deepcopy(model))
    for _group, row in plan:
        rate = float(keeps.get(row, 1.0))
        if rate >= 1.0 - 1e-9:
            continue
        mwr = prune_current_model(mwr, rate, row, quiet=True, record=False, input_shape=input_shape,
                                  importance="l1")
        mwr = ModelWithRows(mwr.model)
    return mwr.model


def group_widths(model, rows):
    """``{row: width}`` of the coupled group each row's layer produces."""
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    out = {}
    for row in rows:
        group = channel_groups.group_of(groups, mwr.all_layers[mwr.row_to_main_layer[row]])
        if group is not None:
            out[row] = int(group.width)
    return out


def _land(model, plan, rows, w, floor, target, size, size0, input_shape, iters, tol):
    """
    ``(keeps, frac, widths)``: the bisection loop of ``plan_targets`` (keeps clip(c · w, floor, 1), c bisected until
    the one-shot cut keeps ``target`` of ``size0``, the closest c found), with no group held. Used by the v30 kinds
    (``sens_cost_at``, ``sens_curve``) only; the other kinds run their own copy in ``plan_targets``.
    """
    lo, hi = 0.0, 1.0 / min(w.values())
    best = None
    for _ in range(iters):
        c = 0.5 * (lo + hi)
        keeps = {row: min(1.0, max(floor[row], c * w[row])) for row in rows}
        cut = cut_to(model, plan, keeps, input_shape)
        frac = size(cut) / size0
        if best is None or abs(frac - target) < abs(best[1] - target):
            best = (keeps, frac, group_widths(cut, rows))
        del cut
        if abs(frac - target) <= tol:
            break
        if frac > target:
            hi = c
        else:
            lo = c
    keeps, frac, widths = best
    return keeps, frac, widths


def _monotone_curve(points):
    """``[(keep, loss, cost)]`` from keep 1 downward with loss and cost each replaced by the running maximum of its
    positive part: a deeper cut never lowers the damage or the saving. ``points`` are ordered by keep, descending."""
    out, loss, cost = [], 0.0, 0.0
    for k, l, c in points:
        loss, cost = max(loss, max(0.0, float(l))), max(cost, max(0.0, float(c)))
        out.append((float(k), loss, cost))
    return out


def _curve_at(curve, k):
    """``(loss, cost)`` of a monotone ``curve`` at keep ``k``: linear in k between its points, the first point's
    values at or above its keep and the last point's below it."""
    if k >= curve[0][0]:
        return curve[0][1], curve[0][2]
    for (k1, l1, c1), (k2, l2, c2) in zip(curve, curve[1:]):
        if k >= k2:
            t = (k1 - k) / (k1 - k2)
            return l1 + t * (l2 - l1), c1 + t * (c2 - c1)
    return curve[-1][1], curve[-1][2]


def _curve_lambda(curve, floor, size0, target):
    """
    ``(lam, kept, picks)``: the multiplier λ ≥ 0 at which every group's keep k_r(λ) = argmin over its grid (1.00 down
    to 0.01 in steps of 0.01 and its floor, from the floor up) of loss − λ · cost / size0 (ties to the larger k) gives
    the estimated kept fraction 1 − Σ cost / size0 closest to ``target``; ``picks`` is ``{row: k_r(λ)}``.
    """
    grid = [round(1.0 - 0.01 * i, 2) for i in range(0, 100)]
    table = {}
    for row, points in curve.items():
        ks = sorted({g for g in grid if g >= floor[row]} | {floor[row]}, reverse=True)
        table[row] = [(k,) + _curve_at(points, k) for k in ks]

    def pick(lam):
        out = {}
        for row, rows_k in table.items():
            best = rows_k[0]
            for entry in rows_k[1:]:
                if entry[1] - lam * entry[2] / size0 < best[1] - lam * best[2] / size0:
                    best = entry
            out[row] = best
        return out

    def kept(lam):
        return 1.0 - sum(entry[2] for entry in pick(lam).values()) / size0

    if kept(0.0) <= target:
        lam = 0.0
    else:
        hi = 1.0
        for _ in range(60):
            if kept(hi) <= target:
                break
            hi *= 2.0
        lo, lam = 0.0, hi
        gap = abs(kept(hi) - target)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            f = kept(mid)
            if abs(f - target) < gap:
                gap, lam = abs(f - target), mid
            if f > target:
                lo = mid
            else:
                hi = mid
    picks = pick(lam)
    return lam, kept(lam), {row: entry[0] for row, entry in picks.items()}


def _curve_threshold(curve, floor, size0, target):
    """
    ``(tau, kept, picks)``: the threshold τ ≥ 0 at which every group's keep k_r(τ) = the smallest k on its grid (1.00
    down to 0.01 in steps of 0.01 and its floor, from the floor up) whose loss on its monotone curve is at most τ gives
    the estimated kept fraction 1 − Σ cost / size0 closest to ``target``; ``picks`` is ``{row: k_r(τ)}``. Unlike
    ``_curve_lambda`` it ignores the saving when it chooses a group's depth.
    """
    grid = [round(1.0 - 0.01 * i, 2) for i in range(0, 100)]
    table = {}
    for row, points in curve.items():
        ks = sorted({g for g in grid if g >= floor[row]} | {floor[row]}, reverse=True)
        table[row] = [(k,) + _curve_at(points, k) for k in ks]

    def pick(tau):
        out = {}
        for row, rows_k in table.items():
            best = rows_k[0]
            for entry in rows_k[1:]:
                if entry[1] > tau + 1e-12:
                    break
                best = entry
            out[row] = best
        return out

    def kept(tau):
        return 1.0 - sum(entry[2] for entry in pick(tau).values()) / size0

    if kept(0.0) <= target:
        tau = 0.0
    else:
        hi = max(entry[1] for rows_k in table.values() for entry in rows_k)
        if kept(hi) > target:
            tau = hi
        else:
            lo, tau = 0.0, hi
            gap = abs(kept(hi) - target)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                f = kept(mid)
                if abs(f - target) < gap:
                    gap, tau = abs(f - target), mid
                if f > target:
                    lo = mid
                else:
                    hi = mid
    picks = pick(tau)
    return tau, kept(tau), {row: entry[0] for row, entry in picks.items()}


def curve_probes(model, plan, batches, input_shape):
    """
    ``(qs, loss, raw)``: the measurement behind ``sens_curve`` and ``sens_thresh``, as ``plan_targets`` takes it. Every
    planned group is cut alone to each keep of ``curve_keeps()`` (``qs``, descending); ``loss[row][q]`` is its
    calibration-loss rise and ``raw[row][q]`` the ``(params saved, MACs saved)`` of that same cut
    (``group_sensitivity.group_sensitivity`` with ``costs``). The plan agent's menu reads it once per network
    (``plan_agent.menu_rules``), so its ``sens_thresh`` rule is this code's.
    """
    rows = [row for _group, row in plan]
    qs = curve_keeps()
    loss, raw = {row: {} for row in rows}, {row: {} for row in rows}
    for q in qs:
        raw_q = {}
        s_q, _base = group_sensitivity.group_sensitivity(model, plan, batches, input_shape, keep=q, costs=raw_q)
        for row in rows:
            loss[row][q] = s_q[row]
            raw[row][q] = raw_q[row]
    return qs, loss, raw


def probe_curves(rows, qs, loss, raw, budget="params"):
    """``{row: monotone curve}`` of ``curve_probes``' measurement (``_monotone_curve``, from keep 1 downward), the
    saving priced in ``budget``: the params saved, or the MACs saved under ``flops``."""
    which = 1 if budget == "flops" else 0
    return {row: _monotone_curve([(1.0, 0.0, 0.0)] + [(q, loss[row][q], raw[row][q][which]) for q in qs])
            for row in rows}


def curve_weights(kind_name, curve, floor, size0, target, rows=None):
    """
    ``(knob, value, est, w)``: the weights ``sens_thresh`` (one threshold τ, ``_curve_threshold``; knob ``tau``) or
    ``sens_curve`` (one multiplier λ, ``_curve_lambda``; knob ``lambda``) hands the landing bisection at ``target``:
    each group's pick on its curve, at least its floor. ``value`` is the knob's and ``est`` the kept fraction the
    curves' savings predict at the picks. The weights depend on ``target``; the curves do not.
    """
    if kind_name == "sens_thresh":
        knob, (value, est, picks) = "tau", _curve_threshold(curve, floor, size0, target)
    else:
        knob, (value, est, picks) = "lambda", _curve_lambda(curve, floor, size0, target)
    return knob, value, est, {row: max(picks[row], floor[row]) for row in (curve if rows is None else rows)}


# ------------------------------------------------------------------ magnitude kinds (global_mag, lamp)


def _group_magnitudes(group):
    """One magnitude per channel of a coupled ``group``, or None: ``pruning.group_importance(group, "l1")``, the score
    whose largest ``cut_to`` keeps. Each layer that produces the group's channels (a producer or a depthwise
    convolution) votes with its filters' L1 norms over its own largest, and the votes are summed; the BN scales and the
    consumers' input slices do not vote. None when the cut has no group score either (a layer that owns only part of
    the group): it then masks the row's own layer and saves nothing."""
    importance = pruning.group_importance(group, "l1")
    return None if importance is None else [float(v) for v in importance.detach().cpu().tolist()]


def _lamp_scores(w):
    """The LAMP scores (Lee et al., ICLR 2021, eq. 1) of non-negative ``w``, in ``w``'s order: with ``w`` sorted
    ascending, score_u = w_u / Σ_{v ≥ u} w_v. The largest scores 1, a rescaled ``w`` scores the same, and a zero scores
    0. Equal values rank in the order given, the earlier as the larger."""
    scores, tail = [0.0] * len(w), 0.0
    for i in sorted(range(len(w)), key=lambda j: float(w[j]), reverse=True):
        tail += float(w[i])
        scores[i] = float(w[i]) / tail if tail > 0.0 else 0.0
    return scores


def _channel_scores(kind_name, magnitudes):
    """One group's score per channel under a magnitude kind, in the channels' order: ``global_mag`` divides the
    magnitudes by their mean (Torch-Pruning's ``normalizer="mean"``), ``lamp`` takes ``_lamp_scores`` of their squares.
    Either way a larger magnitude never scores lower, a zero scores 0, and multiplying every magnitude by one factor
    changes nothing."""
    m = [max(0.0, float(v)) for v in magnitudes]
    if kind_name == "lamp":
        return _lamp_scores([v * v for v in m])
    mean = sum(m) / len(m) if m else 0.0
    return [v / mean if mean > 0.0 else 0.0 for v in m]


def _threshold_keeps(floor):
    """The keeps at which ``_curve_threshold`` reads the curve of a group with keep floor ``floor``, descending: its
    grid (1.00 down to 0.01 in steps of 0.01) from the floor up, and the floor."""
    grid = [round(1.0 - 0.01 * i, 2) for i in range(0, 100)]
    return sorted({g for g in grid if g >= floor} | {floor}, reverse=True)


def _threshold_steps(scores, keeps):
    """``[(keep, width, loss)]`` of one group at each of ``keeps`` (descending from 1): the width ``cut_to`` leaves at
    that keep (every channel at keep 1, else ``pruning.target_width`` of the channels that score above 0, as the cut
    counts the filters that are not all zero) and the largest score among the channels it removes, 0 when it removes
    none. A cut removes only channels that score at most τ exactly when its loss is at most τ, and the loss never
    falls as the keep does."""
    ranked = sorted((float(s) for s in scores), reverse=True)
    alive = sum(1 for s in ranked if s > 0.0)
    steps, loss = [], 0.0
    for k in keeps:
        width = len(ranked) if k >= 1.0 - 1e-9 else pruning.target_width(alive, k)
        loss = max(loss, ranked[width] if width < len(ranked) else 0.0)
        steps.append((k, width, loss))
    return steps


def _threshold_costs(steps, cost_of, total0):
    """``{row: [saved]}``: the budget each of a group's ``steps`` (``{row: _threshold_steps}``) saves, shared out so
    that at every threshold τ the savings of the steps ``_curve_threshold`` picks add up to what the cost model saves
    at all the picked widths together, ``total0 − cost_of({row: width})`` (``plan_agent.ParamModel.cost`` /
    ``FlopModel.cost``). A threshold picks every step whose loss is at most τ, so the steps of all groups are walked
    once in order of loss (ties: plan order, then depth) and each is given what it adds to the saving of those before
    it. Savings measured one group at a time would count a layer between two cut groups twice: τ would stop too early
    and the landing would scale every keep down."""
    rows = list(steps)
    widths = {row: steps[row][0][1] for row in rows}
    saved = {row: [0.0] * len(steps[row]) for row in rows}
    order = sorted((steps[row][i][2], n, i) for n, row in enumerate(rows) for i in range(1, len(steps[row])))
    total = 0.0
    for _loss, n, i in order:
        row = rows[n]
        saved[row][i] = saved[row][i - 1]
        if steps[row][i][1] != widths[row]:
            widths[row] = steps[row][i][1]
            now = max(total, float(total0 - cost_of(widths)))
            saved[row][i] += now - total
            total = now
    return saved


def _magnitude_curves(plan, kind_name, floor, cost_of, total0):
    """``(curve, half)`` of the magnitude kinds (``global_mag``, ``lamp``), read off the weights alone. ``curve[row]``
    is the group's monotone curve for ``_curve_threshold``: at each keep that function reads (``_threshold_keeps``) the
    loss is the largest score the cut to that keep removes (``_threshold_steps`` of the group's ``_channel_scores``)
    and the cost is that step's share of the saving (``_threshold_costs``); it is flat between those keeps, with the
    deeper keep's values, so ``_curve_at`` returns each step's own numbers and interpolates nothing. ``half[row]`` is
    the largest score a cut to half width removes. A group without magnitudes (``_group_magnitudes``) gets no curve
    and 0 in ``half``."""
    steps, half = {}, {}
    for group, row in plan:
        magnitudes = _group_magnitudes(group)
        half[row] = 0.0
        if magnitudes is None:
            continue
        scores = _channel_scores(kind_name, magnitudes)
        steps[row] = _threshold_steps(scores, _threshold_keeps(floor[row]))
        half[row] = _threshold_steps(scores, [1.0, 0.5])[1][2]
    saved = _threshold_costs(steps, cost_of, total0)
    curve = {}
    for row, row_steps in steps.items():
        points = [(row_steps[0][0], 0.0, 0.0)]
        for i in range(1, len(row_steps)):
            keep, _width, loss = row_steps[i]
            points += [(row_steps[i - 1][0], loss, saved[row][i]), (keep, loss, saved[row][i])]
        curve[row] = points
    return curve, half


# ------------------------------------------------------------------ tilt (SPECTRA_ALLOC_KIND=tilt, queue row SAMENET)

TILT_TARGETS = ("base", "run")
TILT_Z_CLIP = 30.0  # |z| beyond this is clipped before exp, as ``plan_agent.residual_decode`` clips its scores


def tilt_base_path() -> str:
    """``SPECTRA_ALLOC_TILT_BASE``: the path of the JSON ``{module name: out channels}`` that ``tilt`` perturbs."""
    path = os.environ.get("SPECTRA_ALLOC_TILT_BASE", "").strip()
    if not path:
        raise ValueError("SPECTRA_ALLOC_KIND=tilt needs SPECTRA_ALLOC_TILT_BASE=<json {module name: out channels}>")
    return path


def tilt_base_table() -> dict:
    """The table at ``tilt_base_path``, read as ``widths_table`` reads ``SPECTRA_ALLOC_WIDTHS``."""
    with open(tilt_base_path(), encoding="utf-8") as fh:
        return {str(name): int(width) for name, width in json.load(fh).items()}


def tilt_target() -> str:
    """``SPECTRA_ALLOC_TILT_TARGET`` (base): the size ``tilt`` bisects its scale to. ``base`` is what the base plan
    itself keeps under the same one-shot cut, with c = 1 tried first; ``run`` is the run's own plan target (the walk's
    target less ``SPECTRA_ALLOC_UNDERSHOOT``), bisected exactly as ``sens`` and ``sample`` bisect."""
    name = os.environ.get("SPECTRA_ALLOC_TILT_TARGET", "base").strip().lower() or "base"
    if name not in TILT_TARGETS:
        raise ValueError(f"SPECTRA_ALLOC_TILT_TARGET={name!r}: expected one of {TILT_TARGETS}")
    return name


def tilt_sigma():
    """``SPECTRA_ALLOC_TILT_SIGMA`` (unset): σ of ``tilt``'s seeded draw, z_g = σ ε_g, a finite float of at least 0
    (0 is the base plan itself). None when unset or blank."""
    raw = os.environ.get("SPECTRA_ALLOC_TILT_SIGMA", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"SPECTRA_ALLOC_TILT_SIGMA={raw!r}: expected a float of at least 0") from None
    if not 0.0 <= value < float("inf"):
        raise ValueError(f"SPECTRA_ALLOC_TILT_SIGMA={raw!r}: expected a float of at least 0")
    return value


def group_keys(model, plan) -> dict:
    """``{row: key}``: a name for every planned group of ``model`` that does not depend on the plan's order, its
    producers' module names sorted and joined by ``+`` (``row<r>`` for a group with no named producer). A producer
    belongs to one group, so no two groups share a key."""
    names = {id(module): name for name, module in model.named_modules()}
    keys = {}
    for group, row in plan:
        named = sorted(names[id(p)] for p in group.producers if id(p) in names)
        keys[row] = "+".join(named) if named else f"row{int(row)}"
    if len(set(keys.values())) != len(keys):
        raise ValueError("group_keys: two planned groups share their producers' names")
    return keys


def tilt_draw(n_groups, sigma, seed):
    """``[z_g]``: σ ε_g with ε ~ N(0, 1) per planned group in plan order, one float64 ``randn`` call on a CPU generator
    seeded by ``seed`` (as ``plan_agent.plan_for_env`` draws ``agent_sample``'s ε); the global RNG is untouched.
    σ = 0 gives exact zeros."""
    eps = torch.randn(int(n_groups), generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float64).tolist()
    return [float(sigma) * e if float(sigma) != 0.0 else 0.0 for e in eps]


def tilt_scores(rows, keys):
    """
    ``({row: z}, source)`` of ``tilt``, from exactly one of two flags. ``SPECTRA_ALLOC_TILT_Z`` names a JSON file that
    holds either a list, one score per planned group in plan order, or ``{group key: z}`` (``group_keys``), which must
    name every planned group and nothing else. ``SPECTRA_ALLOC_TILT_SIGMA`` draws the scores (``tilt_draw`` with the
    seed ``SPECTRA_ALLOC_SAMPLE_SEED``). ``source`` says which: ``{"z": "file", "path"}`` or
    ``{"z": "draw", "sigma", "seed"}``. A score that is not finite is refused.
    """
    path = os.environ.get("SPECTRA_ALLOC_TILT_Z", "").strip()
    sigma = tilt_sigma()
    if bool(path) == (sigma is not None):
        raise ValueError("SPECTRA_ALLOC_KIND=tilt needs exactly one of SPECTRA_ALLOC_TILT_Z=<json: a list in plan "
                         "order or {group key: z}> and SPECTRA_ALLOC_TILT_SIGMA=<std of a draw seeded by "
                         "SPECTRA_ALLOC_SAMPLE_SEED>")
    if path:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            want = [keys[row] for row in rows]
            missing, unknown = [k for k in want if k not in raw], sorted(set(raw) - set(want))
            if missing or unknown:
                raise ValueError(f"SPECTRA_ALLOC_TILT_Z={path}: {len(missing)} planned groups not named"
                                 + (f" (first: {missing[0]})" if missing else "")
                                 + f", {len(unknown)} names that are no planned group"
                                 + (f" (first: {unknown[0]})" if unknown else ""))
            z = {row: float(raw[keys[row]]) for row in rows}
        elif isinstance(raw, list):
            if len(raw) != len(rows):
                raise ValueError(f"SPECTRA_ALLOC_TILT_Z={path}: {len(raw)} scores for {len(rows)} planned groups")
            z = {row: float(value) for row, value in zip(rows, raw)}
        else:
            raise ValueError(f"SPECTRA_ALLOC_TILT_Z={path}: expected a JSON list or object")
        source = {"z": "file", "path": path}
    else:
        seed = sample_seed()
        z = dict(zip(rows, tilt_draw(len(rows), sigma, seed)))
        source = {"z": "draw", "sigma": float(sigma), "seed": int(seed)}
    bad = [keys[row] for row in rows if not math.isfinite(z[row])]
    if bad:
        raise ValueError(f"SPECTRA_ALLOC_KIND=tilt: the score of {bad[0]} is not finite")
    return z, source


# ---- SPECTRA_ALLOC_TILT_PARAMS (amendments A1 and A2 of queue row SAMENET): a tilt at the base plan's params as well

TILT_PARAMS_TOL = 0.01  # the default tol: the held params within 1 % of the base plan's
TILT_PARAMS_MODES = ("landed", "planned")  # what is held: the predicted landing (A2, the default) or the plan (A1)
TILT_LAMBDA_MAX = 2.0  # the hold searches |λ| up to this bound
TILT_LAMBDA_STEP = 0.05  # the step of its outward scan
TILT_LAMBDA_ITERS = 12  # halvings between two neighbouring scan points that lie on opposite sides of the aim
TILT_HOLD_ROUNDS = 3  # the plans it cuts for real beyond λ = 0, at most
# The landed hold's scan: the predicted landing jumps with λ (which groups have arrived when the walk stops), in spots
# far narrower than the planned params' steps, so its |λ| values are fine near 0 and coarser beyond, with fewer
# halvings.
TILT_LANDED_SCAN = tuple([0.0125 * k for k in range(1, 17)] + [0.2 + 0.05 * k for k in range(1, 17)]
                         + [1.0 + 0.1 * k for k in range(1, 11)])
TILT_LANDED_ITERS = 6


def _tilt_params():
    """``(tol, mode)`` of ``SPECTRA_ALLOC_TILT_PARAMS=equal[:<tol>][:landed|planned]``, or None when unset or blank."""
    raw = os.environ.get("SPECTRA_ALLOC_TILT_PARAMS", "").strip().lower()
    if not raw:
        return None
    message = (f"SPECTRA_ALLOC_TILT_PARAMS={raw!r}: expected equal[:tol][:landed|planned] with 0 < tol < 1 (the tol "
               f"before the mode)")
    parts = [part.strip() for part in raw.split(":")]
    if parts[0] != "equal" or len(parts) > 3 or not all(parts):
        raise ValueError(message)
    tol, mode, rest = TILT_PARAMS_TOL, TILT_PARAMS_MODES[0], parts[1:]
    if rest and rest[-1] in TILT_PARAMS_MODES:
        mode = rest.pop()
    if len(rest) > 1:
        raise ValueError(message)
    if rest:
        try:
            tol = float(rest[0])
        except ValueError:
            raise ValueError(message) from None
        if not 0.0 < tol < 1.0:
            raise ValueError(message)
    return tol, mode


def tilt_params():
    """``SPECTRA_ALLOC_TILT_PARAMS`` (unset: off): ``equal[:<tol>][:landed|planned]`` makes ``_state`` hold a ``tilt``
    plan's params at the base plan's (``TiltHold``, ``_tilt_held``), within ``tol`` of them, relative to them (0.01,
    i.e. 1 %, when no tol is given). Returns ``tol``, or None when unset or blank; ``tilt_params_mode`` says what is
    held."""
    parsed = _tilt_params()
    return None if parsed is None else parsed[0]


def tilt_params_mode():
    """What ``SPECTRA_ALLOC_TILT_PARAMS`` holds, or None when it is off. ``landed`` (the default; amendment A2 of the
    row): the params of the network the walk is predicted to land on, from the walk run on the cost models
    (``TiltWalk``), against the base plan's predicted the same way. ``planned`` (amendment A1): the one-shot params
    of the plan the walk aims at, against the base table's."""
    parsed = _tilt_params()
    return None if parsed is None else parsed[1]


def tilt_direction(fm, pm, widths, rows, skip=()):
    """
    ``{row: d_g}``: the direction the params hold moves a tilt along, d_g = log(FLOPs per kept channel of g / params
    per kept channel of g) at ``widths`` (``{row: width}``, the base plan's). Each is the cost model's slope in g's
    own width with every other group at its ``widths``: the cost at ``widths`` less the cost with g alone at one
    channel, over the channels between (``fm.cost`` in MACs, ``pm.cost`` in params; a group that is at one channel is
    measured up to two). A cost model is linear in one group's width up to the rounding of a partly read group, so the
    long span is its marginal without that rounding. d is centred by its mean over the groups weighted by each
    group's FLOPs at ``widths`` (its slope times its width), not by the plain mean: to first order λ · d then moves
    no FLOPs, only params against FLOPs. A group without a positive slope on both models (an origin width of 1)
    gets 0, and so does every row in ``skip`` (the groups a cut cannot narrow, which no weight moves).
    """
    here = {row: int(widths[row]) for row in rows}
    flops, params = float(fm.cost(here)), float(pm.cost(here))
    raw, share = {}, {}
    for row in rows:
        width, origin = here[row], int(pm.widths0[row])
        ref = 1 if width >= 2 else (2 if origin >= 2 else None)
        if ref is None or row in skip:
            continue
        other = dict(here)
        other[row] = ref
        slope_f = (flops - float(fm.cost(other))) / float(width - ref)
        slope_p = (params - float(pm.cost(other))) / float(width - ref)
        if slope_f > 0.0 and slope_p > 0.0:
            raw[row] = math.log(slope_f / slope_p)
            share[row] = slope_f * width
    total = sum(share.values())
    mean = sum(share[row] * raw[row] for row in raw) / total if total > 0.0 else 0.0
    return {row: (raw[row] - mean) if row in raw else 0.0 for row in rows}


def tilt_lambda(ratio_at, tol, aim=1.0, bound=TILT_LAMBDA_MAX, step=TILT_LAMBDA_STEP, iters=TILT_LAMBDA_ITERS,
                scan=None):
    """
    ``(λ, ratio, hit)``: a λ in [−``bound``, ``bound``] whose ``ratio_at(λ)`` lies within ``tol`` of ``aim``, searched
    from 0 outwards on both sides. 0 comes first and is taken exactly when it is within ``tol``; then ±``step``,
    ±2 ``step``, ... (the positive side first), and wherever two neighbouring scan points lie on opposite sides of
    ``aim``, ``iters`` halvings between them. The first λ found within ``tol`` is returned with ``hit`` True. A plan
    moves in whole widths, so the ratio moves in steps and may jump over the band: then the λ whose ratio is closest
    to ``aim`` of all those tried is returned with ``hit`` False (ties to the smaller |λ|, then to the positive side).
    ``ratio_at`` is called once per λ. ``scan`` (ascending |λ| values) replaces the even steps to ``bound``.
    """
    tried = {}

    def at(lam):
        lam = float(lam)
        if lam not in tried:
            tried[lam] = float(ratio_at(lam))
        return tried[lam]

    if abs(at(0.0) - aim) <= tol:
        return 0.0, at(0.0), True
    points = ([k * step for k in range(1, int(math.floor(bound / step + 1e-9)) + 1)] if scan is None
              else [float(point) for point in scan])
    for k, point in enumerate(points):
        for sign in (1.0, -1.0):
            near, far = sign * (points[k - 1] if k else 0.0), sign * point
            if abs(at(far) - aim) <= tol:
                return far, at(far), True
            if (at(near) - aim) * (at(far) - aim) < 0.0:
                lo, hi = near, far
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    if abs(at(mid) - aim) <= tol:
                        return mid, at(mid), True
                    if (at(mid) - aim) * (at(lo) - aim) > 0.0:
                        lo = mid
                    else:
                        hi = mid
    lam = min(tried, key=lambda value: (abs(tried[value] - aim), abs(value), -value))
    return lam, tried[lam], False


class TiltWalk:
    """
    The walk on the cost models, for the landed hold of ``tilt`` (``SPECTRA_ALLOC_TILT_PARAMS``, amendment A2 of queue
    row SAMENET): where the allocation-following walk stops when it aims at a plan. ``order`` is
    ``[(row, group row)]`` for every row the walk visits whose group is planned, in walk order; ``rates`` the env's
    ``{index: keep rate}``; ``passes`` its passes; ``legal_at(row, width)`` the legal action indices of a
    ``width``-wide group at that row (``fortify.legal_action_mask``); ``once`` the group-once lock
    (``SPECTRA_GROUP_ONCE_PER_PASS``); ``protected`` the group rows the env holds whole
    (``SPECTRA_PROTECT_STREAMS``).
    """

    def __init__(self, order, rates, passes, legal_at, once=True, protected=()):
        self.order = [(int(row), int(group)) for row, group in order]
        self.rates = {int(index): float(rate) for index, rate in rates.items()}
        self.passes, self.legal_at, self.once = max(1, int(passes)), legal_at, bool(once)
        self.protected = set(protected)
        self.identity = next((i for i, r in self.rates.items() if abs(r - 1.0) < 1e-9), 0)

    def land(self, kept_of, origin, targets, goal, pinned=()):
        """
        ``(widths, how)``: the group widths the walk leaves when it starts from ``origin`` (``{group row: width}``)
        and aims at ``targets``, with ``kept_of(widths)`` the kept size the env stops on and ``goal`` its target.

        Every pass visits the rows in order. At a row whose group is not at its target, ``action``'s pick is played
        (``choose``: the legal cut that brings the group closest), cut as ``walk_cut`` cuts it; under ``once`` the
        group is then locked for the rest of the pass, as ``NetworkEnv._register_group_lock`` locks it. The first cut
        that takes ``kept_of`` to ``goal`` or below is softened as ``NetworkEnv._land_on_target`` softens it (ten
        halvings of the keep rate towards 1, the mildest that still reaches ``goal``) and ends the walk: ``how`` is
        ``landed``. Which cuts are made does not depend on the kept size, only where the walk stops does, and every
        cut lowers it: so the cuts are listed first and the stopping one is found by halving over them. A group in
        ``pinned`` (its cut is masked and leaves its width) or in ``protected`` is never cut. A pass with no cut
        above ``goal`` is ``stalled`` (the real walk then falls back to the strongest legal cut, which this does not
        follow), and running out of passes above ``goal`` is ``passes``. What the walk only meets at run time is not
        modelled: an eval rollback lock, an eval size floor.
        """
        widths = {group: int(width) for group, width in origin.items()}
        skip = set(pinned) | self.protected
        cuts, how = [], "passes"  # every cut the walk would make if nothing stopped it: (group, width, new width, keep)
        for _pass in range(self.passes):
            locked, made = set(), 0
            for row, group in self.order:
                if group in locked or group in skip or group not in targets:
                    continue
                width = widths[group]
                pick = choose(width, int(targets[group]), self.rates, self.legal_at(row, width), self.identity)
                if pick == self.identity:
                    continue
                keep = float(self.rates[pick])
                if fortify.width_ladder_max() > 0 and 0.0 < keep < 1.0:  # the env's width ladder, as ``walk_cut``
                    keep, _stop, feasible = fortify.effective_rates({0: keep}, 0.0, group_width=width)[0]
                    if not feasible:
                        continue
                new = pruning.target_width(width, float(keep))
                if new >= width:
                    continue
                cuts.append((group, width, new, float(keep)))
                widths[group] = new
                made += 1
                if self.once:
                    locked.add(group)
            if not made:
                how = "stalled"
                break

        def after(count):  # the widths once the first ``count`` cuts are made
            out = {group: int(width) for group, width in origin.items()}
            for group, _width, new, _keep in cuts[:count]:
                out[group] = new
            return out

        if not cuts or kept_of(widths) > goal + 1e-9:
            return widths, how
        lo, hi = 0, len(cuts)  # every cut lowers the kept size: the first one at or under the goal, by halving
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if kept_of(after(mid)) <= goal + 1e-9:
                hi = mid
            else:
                lo = mid
        group, width, _new, keep = cuts[hi - 1]
        widths = after(hi - 1)
        mild, full = 1.0, float(keep)  # ``_land_on_target``: the mildest keep rate that still reaches the goal
        for _halving in range(10):
            mid = 0.5 * (full + mild)
            if kept_of({**widths, group: pruning.target_width(width, mid)}) <= goal + 1e-9:
                full = mid
            else:
                mild = mid
        widths[group] = pruning.target_width(width, full)
        return widths, "landed"


class TiltHold:
    """
    What ``tilt_plan`` holds a plan's params with (``SPECTRA_ALLOC_TILT_PARAMS``; ``_state`` builds it with
    ``tilt_hold_for``): ``tol``, the cost models the search for λ runs on (``fm``, a ``plan_agent.FlopModel``, and
    ``pm``, its ``ParamModel``) and ``aimed``, the plan the walk aims at. With ``grids`` (``{row: the widths the walk
    lands on exactly}``), ``target`` (the walk target) and ``floors`` that plan is the tilt on the walk's grid, as
    ``_grid_round_state`` rounds it; without them it is the tilt itself. With ``walk`` (a ``TiltWalk``) the hold is
    the ``landed`` one (amendment A2): ``landed`` says where the walk is predicted to stop, and that network's params
    are what is held. Without it the hold is the ``planned`` one (amendment A1): the aimed plan's own params.
    """

    def __init__(self, tol, fm, grids=None, target=None, floors=None, seconds=0.0, walk=None):
        self.tol = float(tol)
        self.fm, self.pm = fm, fm.pm
        self.grids, self.target, self.floors, self.seconds = grids, target, floors, float(seconds)
        self.grid = grids is not None
        self.walk = walk
        self.mode = "landed" if walk is not None else "planned"

    def aimed(self, widths, weights):
        """``{row: width}`` the walk aims at when it is given the plan ``widths`` of a tilt whose groups weigh
        ``weights``: ``grid_round_widths`` with ``_grid_round_state``'s arguments for a ``tilt`` plan (the cost
        model's kept FLOPs, the walk target, the weights as the stepping order, no group held, the plan floors)."""
        planned = {row: int(widths[row]) for row in self.fm.rows if row in widths}
        if self.grids is None:
            return planned
        rounded, _report = grid_round_widths(planned, {row: list(self.grids[row]) for row in planned}, self.fm.kept,
                                             self.target, {row: float(weights[row]) for row in self.fm.rows},
                                             floors=self.floors)
        return rounded

    def landed(self, aimed, pinned=()):
        """``(widths, how)``: where the walk is predicted to stop when it aims at ``aimed``: ``TiltWalk.land`` from
        the origin widths on the cost model's kept FLOPs, to the walk target."""
        origin = {row: int(self.fm.widths0[row]) for row in self.fm.rows}
        return self.walk.land(self.fm.kept, origin, aimed, self.target, pinned)


def tilt_hold_for(env, model, walk_target, flops):
    """
    The ``TiltHold`` of ``_state``'s network under ``SPECTRA_ALLOC_TILT_PARAMS``: a ``FlopModel`` of the origin built
    as ``_grid_round_state`` builds its own and, when ``SPECTRA_ALLOC_GRID_ROUND`` will round the plan, the walk's
    grids, target and floors exactly as that function passes them to ``grid_round_widths``. Under the ``landed`` mode
    it also carries the walk itself (``TiltWalk``): the rows in walk order with their groups, as ``walk_grids`` maps
    them, the env's rates and passes, its legal mask per row and width, the group-once lock and the protected
    streams. The flag needs ``SPECTRA_ALLOC_BUDGET=flops``: under the params budget a tilt is already bisected to the
    base plan's params.
    """
    from src import plan_agent
    if not flops:
        raise ValueError("SPECTRA_ALLOC_TILT_PARAMS holds the params of a tilt that is bisected in FLOPs: it needs "
                         "SPECTRA_ALLOC_BUDGET=flops")
    started = time.perf_counter()
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    plan = group_sensitivity.group_plan(mwr, groups)
    fm = plan_agent.FlopModel(model, plan, env._input_shape(), env.conf.device)
    rates = env.conf.compression_rates_dict
    passes = max(1, int(getattr(env.conf, "passes", 1) or 1))
    walk = None
    if tilt_params_mode() == "landed":
        if fortify.action_menu() == "budget":
            raise ValueError("SPECTRA_ALLOC_TILT_PARAMS in its landed mode follows the walk's cuts on the cost models; "
                             "SPECTRA_ACTION_MENU=budget prices each cut on the net as the walk finds it. Use "
                             "equal[:tol]:planned")
        first = {id(group): row for group, row in plan}
        order = []
        for row in sorted(mwr.row_to_main_layer)[:-1]:
            group = channel_groups.group_of(groups, mwr.all_layers[mwr.row_to_main_layer[row]])
            if group is not None and id(group) in first:
                order.append((row, first[id(group)]))
        masks = {}

        def legal_at(row, width):
            key = (int(row), int(width))
            if key not in masks:
                mask = fortify.legal_action_mask(rates, row_index=int(row), alive_count=int(width), device="cpu")
                masks[key] = [int(i) for i in mask.nonzero(as_tuple=False).flatten().tolist()]
            return masks[key]

        protected = {row for group, row in plan if fortify.protect_streams() and fortify.is_residual_stream(group)}
        walk = TiltWalk(order, rates, passes, legal_at, once=fortify.group_once_per_pass(), protected=protected)
    grids = floors = None
    if fortify.alloc_grid_round() and fortify.action_menu() != "budget":
        walks = walk_grids(model, plan, rates, passes, groups)
        floor = fortify.plan_min_width()
        floors = {row: max(1, int(round(min_keep() * w0)), min(w0, floor)) for row, w0 in fm.widths0.items()}
        grids = {row: list(walks[row]["grid"]) for row in walks}
    return TiltHold(tilt_params(), fm, grids, float(walk_target), floors, seconds=time.perf_counter() - started,
                    walk=walk)


def _tilt_held(hold, model, plan, rows, input_shape, z, base_keeps, base_widths, base_params, w, best, land):
    """
    ``(weights, landing, held)``: ``tilt_plan``'s plan with its params held (``SPECTRA_ALLOC_TILT_PARAMS``).

    A plan's ratio is what the hold measures, over the base plan's. Under the ``planned`` hold (amendment A1) it is
    the params its aimed plan keeps (``hold.aimed`` of its landed widths, cut for real in one shot) over
    ``base_params``, what the base table keeps under the same cut. Under the ``landed`` hold (amendment A2, the
    flag's default) it is the params of the network the walk is predicted to stop on (``hold.landed`` of the aimed
    plan: the walk on the cost models; that widths table cut for real in one shot) over the same for the base table
    walked as a tilt of z = 0. ``w`` and ``best`` are the tilt as it is (λ = 0) and its landing; when its ratio is
    within ``hold.tol`` of 1 they are returned unchanged, and no direction is measured. Otherwise the weights become
    base keep_g · exp(z_g + λ d_g) with d ``tilt_direction`` at the base widths, each λ landed by ``land`` exactly as
    the tilt lands (the same bisection of c to the same FLOPs aim), and λ is searched by ``tilt_lambda`` on the cost
    models: ``land`` measuring ``hold.fm`` at the widths the cut would leave (``pruning.target_width``) and the ratio
    from ``hold.pm``; the landed hold scans λ more finely near 0 (``TILT_LANDED_SCAN``, ``TILT_LANDED_ITERS``), since
    the predicted landing jumps in narrower spots than the plan does. The λ it finds is then landed and cut for real.
    Where the real ratio is off ``tol`` (the cost models are not exact on every network), the search is run again with
    its aim moved by the difference, up to
    ``TILT_HOLD_ROUNDS`` real plans. The real plan closest to 1 is returned (the only one within ``tol`` when there
    is one; ties to the smaller |λ|), and ``held`` says whether it is ``within``: the mode, λ, the ratio, the aimed
    plan with its kept params and FLOPs and its own ``planned`` ratio, the unheld tilt's ratio, d (None when λ = 0
    was taken as it is), the bound and step of the search, the plans tried on the cost models and for real, and the
    seconds this took; under the landed hold also the predicted landing with its params and FLOPs, the base plan's
    predicted landed params and how both walks ended. The map from z to the plan stays deterministic.

    A cut that cannot remove a group's channels masks them and leaves the group's width
    (``NetworkEnv.prune_current_model``: the streams of ShuffleNetV2), so every group is cut once for real to half
    its width (before the search; under the landed hold before anything, since the walk must skip such a group): a
    group that keeps its origin width is ``pinned``, held at that width on the cost models and left out of d.
    ``held`` counts them, the groups that landed on neither width (``unmodelled``) and the groups in which the cost
    models' landing of the tilt as it is differs from the real one (``mirror``; 0 when the cost models follow the
    cuts).
    """
    from src import plan_agent
    started = time.perf_counter()
    fm, pm = hold.fm, hold.pm
    walked = hold.mode == "landed"
    origin = {row: int(pm.widths0[row]) for row in pm.widths0}
    params0, flops0 = float(utils.calc_num_parameters(model)), float(utils.calc_flops(model, input_shape))
    base = {row: int(base_widths.get(row, origin[row])) for row in rows}
    pinned, probed = set(), {}

    def probe():  # which groups a cut really narrows: one real cut of every group to half its width
        if not probed:
            cut = cut_to(model, plan, {row: 0.5 for row in rows}, input_shape)
            left = group_widths(cut, rows)
            del cut
            pinned.update(row for row in rows if left.get(row) == origin[row])
            probed["stray"] = sum(row not in pinned and left.get(row) != pruning.target_width(origin[row], 0.5)
                                  for row in rows)
        return probed["stray"]

    def sizes(table):  # a widths table cut for real in one shot: its shares of the origin's params and FLOPs
        full = {row: int(table.get(row, origin[row])) for row in origin}
        cut = cut_to(model, plan, plan_agent.rates_of(full, origin), input_shape)
        out = utils.calc_num_parameters(cut) / params0, utils.calc_flops(cut, input_shape) / flops0
        del cut
        return out

    def measure(keeps):  # the cost model's one-shot cut: the widths ``cut_to`` leaves and their kept FLOPs
        widths = {row: origin[row] if row in pinned or keeps[row] >= 1.0 - 1e-9
                  else pruning.target_width(origin[row], keeps[row]) for row in rows}
        return fm.kept(widths), widths

    stray, base_stop, base_how, base_landed = 0, None, None, None
    if walked:  # the base plan's own predicted landing: the base table walked as the tilt of z = 0 is walked
        stray = probe()
        base_stop, base_how = hold.landed(hold.aimed(base, base_keeps), pinned)
        base_landed = sizes(base_stop)[0]

    def real(lam, weights, scaled):  # a landed scale's aimed plan (and, walked, its predicted stop), cut for real
        aimed = hold.aimed(scaled[2], weights)
        params, flops = sizes(aimed)
        out = {"lambda": float(lam), "planned": params / base_params, "params": float(params), "flops": float(flops),
               "aimed": aimed, "weights": weights, "scaled": scaled}
        if walked:
            stop, how = hold.landed(aimed, pinned)
            landed_params, landed_flops = sizes(stop)
            out.update(ratio=landed_params / base_landed, landed=stop, landed_params=float(landed_params),
                       landed_flops=float(landed_flops), how=how)
        else:
            out["ratio"] = out["planned"]
        return out

    plans = [real(0.0, w, best)]
    d, seen, mirror = None, {}, 0
    if abs(plans[0]["ratio"] - 1.0) > hold.tol:
        stray = probe()
        d = tilt_direction(fm, pm, base, rows, skip=pinned)
        model_base = pm.kept(base_stop if walked else base)

        def weights_at(lam):
            if lam == 0.0:
                return w
            return {row: base_keeps[row] * math.exp(max(-TILT_Z_CLIP, min(TILT_Z_CLIP, z[row] + lam * d[row])))
                    for row in rows}

        def model_of(scaled, weights):  # the same ratio on the cost models
            aimed = hold.aimed(scaled, weights)
            return pm.kept(hold.landed(aimed, pinned)[0] if walked else aimed) / model_base

        def model_ratio(lam):
            if lam not in seen:
                weights = weights_at(lam)
                seen[lam] = model_of(land(weights, measure)[2], weights)
            return seen[lam]

        search = {"scan": TILT_LANDED_SCAN, "iters": TILT_LANDED_ITERS} if walked else {}
        scaled = land(w, measure)[2]  # the tilt as it is on the cost models: the real landing's widths when they follow
        mirror = sum(width != best[2].get(row) for row, width in scaled.items())
        seen[0.0] = model_of(scaled, w)
        bias = plans[0]["ratio"] - seen[0.0]
        for _ in range(TILT_HOLD_ROUNDS):
            lam, _ratio, _hit = tilt_lambda(model_ratio, hold.tol, aim=1.0 - bias, **search)
            if any(p["lambda"] == lam for p in plans):
                break
            weights = weights_at(lam)
            plans.append(real(lam, weights, land(weights)))
            if abs(plans[-1]["ratio"] - 1.0) <= hold.tol:
                break
            bias = plans[-1]["ratio"] - model_ratio(lam)
    pick = min(plans, key=lambda p: (abs(p["ratio"] - 1.0), abs(p["lambda"]), -p["lambda"]))
    held = {"tol": hold.tol, "mode": hold.mode, "lambda": pick["lambda"], "ratio": float(pick["ratio"]),
            "within": bool(abs(pick["ratio"] - 1.0) <= hold.tol), "base_params": float(base_params),
            "params": pick["params"], "flops": pick["flops"], "planned": float(pick["planned"]),
            "aimed": dict(pick["aimed"]), "grid": bool(hold.grid), "ratio0": float(plans[0]["ratio"]), "d": d,
            "bound": TILT_LAMBDA_MAX, "step": TILT_LANDED_SCAN[0] if walked else TILT_LAMBDA_STEP,
            "model_plans": len(seen), "real_plans": len(plans),
            "pinned": len(pinned), "unmodelled": int(stray), "mirror": int(mirror),
            "build_seconds": float(hold.seconds), "seconds": time.perf_counter() - started}
    if walked:
        held.update(landed=dict(pick["landed"]), landed_params=pick["landed_params"],
                    landed_flops=pick["landed_flops"], base_landed=float(base_landed), how=pick["how"],
                    base_how=base_how, planned0=float(plans[0]["planned"]))
    return pick["weights"], pick["scaled"], held


def tilt_plan(model, plan, rows, input_shape, target, a, keep_floor, iters, tol, budget, min_width, size, hold=None):
    """
    ``(widths, info)`` of ``SPECTRA_ALLOC_KIND=tilt`` in ``plan_targets``' format, which calls it with its own
    arguments and ``size`` (the budget's measure of a model).

    The base plan is the table ``tilt_base_table`` names, read per group as ``widths`` reads its table
    (``copied_keeps``: the narrowest width named for the group's producers over its own width; a group that is not
    named has base keep 1, and a table that names no planned group at all is refused: it is another network's).
    The scores are ``tilt_scores``' and a group's weight is w_g = base keep_g · exp(z_g).
    Keeps are clip(c · w_g, floor_g, 1), with ``sens``' floor (``keep_floor``, lifted to ``min_width`` over the group's
    width) but never above the base plan's own keep, so a base width under the floor is held and never lifted. c is
    bisected with ``sens``' loop (``iters``, ``tol``, the closest c found, each c measured on the real one-shot cut):

    * ``tilt_target() == "base"`` (default): to the fraction the base plan itself keeps under the same cut, c = 1
      first. The base plan is then the answer at z = 0 (c = 1, every keep the base's, the same widths), and any z
      whose unscaled tilt lands within ``tol`` of the base's size is taken as it is.
    * ``"run"``: to ``target``, exactly as ``sens`` and ``sample`` (no c = 1 step; z = 0 then scales the base plan to
      the run's plan target and is not the base plan).

    ``info`` is ``sens``' with ``sens`` holding the scores (as the agent kinds hold theirs), ``weights`` the w_g that
    ``grid_priority`` steps by, ``target`` the size bisected to, and ``tilt`` what ``_state`` logs and records:
    the base's path, keeps, widths and kept fraction, the scores, their source and the groups' keys in plan order, c,
    the plan's widths with its kept params and FLOPs, and how many groups moved from the base. No batch is read, and
    the scores never come from the global RNG (the trial cuts build modules, which draws from it, as under every kind).

    ``hold`` (None: all of the above, unchanged) is ``_state``'s ``TiltHold`` under ``SPECTRA_ALLOC_TILT_PARAMS``
    (amendment A1 of the row). The plan is then also held at the base plan's params: the weights become
    base keep_g · exp(z_g + λ d_g) for the λ ``_tilt_held`` finds, landed by the same bisection to the same aim, and
    λ is 0 exactly, the plan above, whenever that plan is already within the tolerance. ``info["weights"]`` are the
    weights the plan was landed with, ``info["sens"]`` stays z, and ``info["tilt"]["hold"]`` is ``_tilt_held``'s
    record. It needs the FLOPs budget.
    """
    if hold is not None and budget != "flops":
        raise ValueError("SPECTRA_ALLOC_TILT_PARAMS holds the params of a tilt that is bisected in FLOPs: it needs "
                         "SPECTRA_ALLOC_BUDGET=flops")
    path = tilt_base_path()
    base_keeps, unnamed = copied_keeps(model, plan, tilt_base_table())
    if rows and unnamed == len(rows):
        raise ValueError(f"SPECTRA_ALLOC_TILT_BASE={path}: the table names no planned group of this network")
    keys = group_keys(model, plan)
    z, source = tilt_scores(rows, keys)
    w = {row: base_keeps[row] * math.exp(max(-TILT_Z_CLIP, min(TILT_Z_CLIP, z[row]))) for row in rows}
    floor = {row: keep_floor for row in rows}
    if int(min_width) > 1:
        for (group, row) in plan:
            floor[row] = max(keep_floor, min(1.0, int(min_width) / float(group.width)))
    floor = {row: min(floor[row], base_keeps[row]) for row in rows}
    size0 = size(model)
    base_cut = cut_to(model, plan, base_keeps, input_shape)
    base_frac = size(base_cut) / size0
    base_widths = group_widths(base_cut, rows)
    base_params = None
    if hold is not None:  # amendment A1: what the base table keeps of the params under the same one-shot cut
        base_params = utils.calc_num_parameters(base_cut) / float(utils.calc_num_parameters(model))
    del base_cut
    mode = tilt_target()
    aim = float(base_frac) if mode == "base" else float(target)

    def land(w, measure=None):
        """The bisection of c for the weights ``w``: ``(keeps, kept fraction, widths, c)``, each c measured on the
        real one-shot cut, or by ``measure`` (``keeps -> (kept fraction, widths)``, the hold's cost model)."""
        def at(c):
            keeps = {row: min(1.0, max(floor[row], c * w[row])) for row in rows}
            if measure is not None:
                frac, widths = measure(keeps)
                return keeps, frac, widths, c
            cut = cut_to(model, plan, keeps, input_shape)
            frac = size(cut) / size0
            widths = group_widths(cut, rows)
            del cut
            return keeps, frac, widths, c

        lo, hi = 0.0, 1.0 / min(w.values())
        best = None
        if mode == "base":
            best = at(1.0)
            if best[1] > aim:
                hi = 1.0
            else:
                lo, hi = 1.0, max(hi, 1.0)
        if best is None or abs(best[1] - aim) > tol:
            for _ in range(iters):
                c = 0.5 * (lo + hi)
                trial = at(c)
                if best is None or abs(trial[1] - aim) < abs(best[1] - aim):
                    best = trial
                if abs(trial[1] - aim) <= tol:
                    break
                if trial[1] > aim:
                    hi = c
                else:
                    lo = c
        return best

    best = land(w)
    held = None
    if hold is not None:  # amendment A1: λ along the FLOPs-per-param direction, to the base plan's params
        w, best, held = _tilt_held(hold, model, plan, rows, input_shape, z, base_keeps, base_widths, base_params, w,
                                   best, land)
    keeps, frac, widths, c = best
    cut = cut_to(model, plan, keeps, input_shape)
    params = utils.calc_num_parameters(cut) / float(utils.calc_num_parameters(model))
    flops = utils.calc_flops(cut, input_shape) / float(utils.calc_flops(model, input_shape))
    del cut
    moved = sum(widths.get(row) != base_widths.get(row) for row in rows)
    return widths, {"kind": "tilt", "alpha": float(a), "target": float(aim), "kept": float(frac), "keeps": keeps,
                    "origin_widths": group_widths(model, rows), "sens": dict(z), "held": 0, "weights": w,
                    "cost_budget": budget,
                    "tilt": {"base": path, "base_keeps": base_keeps, "base_widths": base_widths,
                             "base_kept": float(base_frac), "unnamed": int(unnamed), "rows": list(rows), "keys": keys,
                             "z": dict(z), "source": source, "target": mode, "aim": float(aim), "c": float(c),
                             "floor": floor, "widths": dict(widths), "params": float(params), "flops": float(flops),
                             "moved": int(moved), **({"hold": held} if held is not None else {})}}


def _tilt_report(net, info, flops):
    """``_state``'s lines and record for a ``tilt`` plan: one summary line, one line per planned group in plan order
    (its key, score, base keep, keep and width) and an ``alloc_tilt`` record that carries the same. Under
    ``SPECTRA_ALLOC_TILT_PARAMS`` the summary and the record also name the hold's λ, and each group its d once a
    direction was measured (``_tilt_hold_report`` says the rest after the grid rounding)."""
    t = info["tilt"]
    name = os.path.basename(str(net))
    rows, src = t["rows"], t["source"]
    held = t.get("hold")
    d = held["d"] if held else None
    drawn = (f"z from {os.path.basename(src['path'])}" if src["z"] == "file"
             else f"z drawn sigma={src['sigma']:g} seed={src['seed']}")
    utils.print_flush(
        f"[alloc] tilt {name}: base {os.path.basename(t['base'])} keeps x{t['base_kept']:.4f} of the "
        f"{'FLOPs' if flops else 'params'}; {drawn}, |z| max {max(abs(float(t['z'][row])) for row in rows):.4f}; "
        f"scale c={t['c']:.4f} bisected to x{t['aim']:.4f} "
        f"({'the base plan' if t['target'] == 'base' else 'the run target'}); the plan keeps params "
        f"x{t['params']:.4f} FLOPs x{t['flops']:.4f}; {t['moved']} of {len(rows)} groups moved from the base"
        + (f"; {t['unnamed']} groups not named in the base table, base keep 1" if t["unnamed"] else "")
        + (f"; params held, weights base keep x exp(z + lambda d) with lambda={held['lambda']:+.6f}" if held else ""))
    for i, row in enumerate(rows):
        utils.print_flush(
            f"[alloc] tilt {name} group {i} row {row} {t['keys'][row]}: z {float(t['z'][row]):+.4f} base keep "
            f"{t['base_keeps'][row]:.4f} keep {info['keeps'][row]:.4f} width {t['widths'].get(row)} "
            f"(base {t['base_widths'].get(row)}, origin {info['origin_widths'].get(row)})"
            + (f" d {float(d[row]):+.4f}" if d else ""))
    try:
        import src.run_recorder as run_recorder
        run_recorder.record(
            "alloc_tilt", network=str(net), base=t["base"], base_kept=t["base_kept"], source=src, target=t["target"],
            aim=t["aim"], c=t["c"], kept=info["kept"], params=t["params"], flops=t["flops"], moved=t["moved"],
            unnamed=t["unnamed"], budget="flops" if flops else "params",
            **({"lam": float(held["lambda"])} if held else {}),
            groups=[{"row": int(row), "key": t["keys"][row], "z": float(t["z"][row]),
                     "base_keep": float(t["base_keeps"][row]), "keep": float(info["keeps"][row]),
                     "base_width": t["base_widths"].get(row), "width": t["widths"].get(row),
                     "origin": info["origin_widths"].get(row), **({"d": float(d[row])} if d else {})} for row in rows])
    except Exception:  # noqa: BLE001 - the record is a convenience
        pass


def _tilt_hold_report(net, entry, info):
    """
    ``_state``'s line and ``alloc_tilt_hold`` record for the params hold of a ``tilt`` plan
    (``SPECTRA_ALLOC_TILT_PARAMS``), after any grid rounding: λ and how it was found, the params the plan the walk
    aims at keeps against the base plan's (within the tolerance, or the MISS and by how much), its FLOPs, what the
    unheld tilt kept, the seconds the hold took, and whether ``entry``'s widths, the ones the walk will aim at, are
    the plan the hold measured (a WARNING when they are not: the hold would then have measured another plan). Under
    the landed hold the ratio is that of the predicted landing, the aimed plan's own ratio stands beside it, and the
    record carries the predicted landed params and widths for the run's real landing to be held against.
    """
    t, h = info["tilt"], info["tilt"]["hold"]
    name = os.path.basename(str(net))
    off = [row for row, width in h["aimed"].items() if entry["widths"].get(row) != width]
    over = abs(h["ratio"] - 1.0) - h["tol"]
    verdict = (f"(within tol {h['tol']:g})" if h["within"] else
               f"(MISS: {over:.4f} beyond tol {h['tol']:g}, the closest lambda found)")
    if h["mode"] == "landed":  # amendment A2: the params of the network the walk is predicted to stop on
        what = (f"; the walk is predicted to land on params x{h['landed_params']:.4f}, {h['ratio']:.4f} of the base "
                f"plan's predicted x{h['base_landed']:.4f} {verdict}, at FLOPs x{h['landed_flops']:.4f}; the plan it "
                f"aims at keeps {h['planned']:.4f} of the base plan's planned params; the unheld tilt would land on "
                f"{h['ratio0']:.4f}"
                + ("" if h["how"] == "landed" and h["base_how"] == "landed" else
                   f"; WARNING: the walk on the cost models did not land (this plan {h['how']}, the base plan "
                   f"{h['base_how']})"))
    else:
        what = (f"; the plan the walk aims at keeps params x{h['params']:.4f}, {h['ratio']:.4f} of the base plan's "
                f"x{h['base_params']:.4f} {verdict}, and FLOPs x{h['flops']:.4f}; the unheld tilt: {h['ratio0']:.4f} "
                f"of the base plan's params")
    utils.print_flush(
        f"[alloc] tilt {name} params hold: lambda={h['lambda']:+.6f} "
        + ("(the tilt as it is, within tol; no search)" if h["model_plans"] == 0 else
           f"(|lambda| <= {h['bound']:g}, scan step {h['step']:g}"
           + (" to 0.2, 0.05 to 1, 0.1 beyond" if h["mode"] == "landed" else "")
           + f"; {h['model_plans']} plans on the cost models, {h['real_plans']} cut)")
        + what + f"; {h['build_seconds'] + h['seconds']:.1f} s"
        + (f"; {h['pinned']} groups a cut cannot narrow, held at their width on the cost models" if h["pinned"] else "")
        + (f"; WARNING: the cost models land the unheld tilt on other widths than the real cut in {h['mirror']} groups"
           f" ({h['unmodelled']} groups followed neither width at the half cut)"
           if h["mirror"] or h["unmodelled"] else "")
        + (f"; WARNING: the walk's plan differs from the plan the hold measured in {len(off)} groups" if off else ""))
    try:
        import src.run_recorder as run_recorder
        run_recorder.record(
            "alloc_tilt_hold", network=str(net), tol=h["tol"], mode=h["mode"], lam=h["lambda"], ratio=h["ratio"],
            within=h["within"], base_params=h["base_params"], params=h["params"], flops=h["flops"],
            planned=h["planned"], ratio0=h["ratio0"], grid=h["grid"],
            **({key: h[key] for key in ("landed_params", "landed_flops", "base_landed", "how", "base_how", "planned0")}
               if h["mode"] == "landed" else {}),
            match=not off, bound=h["bound"], step=h["step"], model_plans=h["model_plans"],
            real_plans=h["real_plans"], pinned=h["pinned"], unmodelled=h["unmodelled"], mirror=h["mirror"],
            build_seconds=h["build_seconds"], seconds=h["seconds"],
            groups=[{"row": int(row), "key": t["keys"][row], "aimed": h["aimed"].get(row),
                     "walk": entry["widths"].get(row), **({"d": float(h["d"][row])} if h["d"] else {}),
                     **({"landed": h["landed"].get(row)} if h["mode"] == "landed" else {})}
                    for row in t["rows"]])
    except Exception:  # noqa: BLE001 - the record is a convenience
        pass


# ------------------------------------------------------------------ SPECTRA_EVAL_SAVE_WIDTHS (queue row SAMENET)


def cut_table(model) -> dict:
    """``{module name: out channels}`` of every Conv2d and Linear of ``model`` (a Linear's out features): a pruned
    network written as the table ``widths_table`` / ``tilt_base_table`` read."""
    table = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            table[name] = int(module.out_channels)
        elif isinstance(module, torch.nn.Linear):
            table[name] = int(module.out_features)
    return table


def plan_table(model, plan, widths) -> dict:
    """``{producer name: width}`` for every planned group of ``model`` (the origin) that ``widths`` (``{row: width}``,
    the walk's targets) covers. ``widths`` (the kind) reads it back as the same target for every group."""
    names = {id(module): name for name, module in model.named_modules()}
    table = {}
    for group, row in plan:
        if row in widths:
            for producer in group.producers:
                if id(producer) in names:
                    table[names[id(producer)]] = int(widths[row])
    return table


def table_digest(table) -> str:
    """sha1 of a widths table in a canonical form (sorted names, no spaces): one name for one plan."""
    text = json.dumps({str(k): int(v) for k, v in table.items()}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def state_digest(model) -> str:
    """sha1 over the names, dtypes, shapes and bytes of ``model.state_dict()`` in name order: two pruned copies get
    the same digest exactly when they are the same network bit for bit."""
    digest = hashlib.sha1()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path, doc):
    """``doc`` as JSON at ``path``, written to a temp file and renamed."""
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(path + ".tmp", path)


def save_widths(env, net_path, label, candidate, folder):
    """
    ``SPECTRA_EVAL_SAVE_WIDTHS=1`` (``fortify.eval_save_widths``, default off; called by the runner's
    ``_run_final_ft`` for every size-point candidate): the candidate's widths as JSON files under ``folder`` (the run
    directory's ``alloc_widths/``), named ``<net>__<label>__plan.json``, ``…__cut.json`` and ``…__meta.json``.

    * ``plan`` (alloc walks only): ``plan_table`` of the widths the walk aimed at, after any grid rounding
      (``_state``'s entry). Read back through ``SPECTRA_ALLOC_KIND=widths``, or as ``SPECTRA_ALLOC_TILT_BASE`` with
      z = 0, the walk aims at the same widths, so it makes the same cuts in the same order and lands on the same
      network.
    * ``cut``: ``cut_table`` of the pruned candidate, the network the final fine-tune recovers. It is the record of
      what was cut, not a plan to walk again: the walk's last cut is softened onto the target
      (``NetworkEnv._land_on_target``), which leaves that group at a width between two rate steps, and groups the
      walk had not reached keep a width above their target.
    * ``meta``: the network, label, step, landed params and FLOPs ratios, seed, policy, kind and budget; the digests
      of both tables (``table_digest``) and of the candidate's weights (``state_digest``); whether the stall fallback
      fired; and per planned group, in plan order, its key (``group_keys``), row, producers, and origin, planned and
      cut widths.

    The names come from the plan of a copy of the origin (``env.data_dict``'s network, which is not touched). Returns
    the meta, or None after one line when anything fails: a save must not cost the fine-tune. The global RNG is left
    as it was.
    """
    name = os.path.basename(str(net_path))
    rng = rng_snapshot()
    try:
        import src.fortify as fortify_mod
        point, model = candidate["point"], candidate["model"]
        origin = copy.deepcopy(env.data_dict[env.selected_net_path][0])
        plan = group_sensitivity.group_plan(ModelWithRows(origin))
        keys = group_keys(origin, plan)
        names = {id(module): n for n, module in origin.named_modules()}
        cut = cut_table(model)
        policy = fortify_mod.eval_policy_name()
        entry = (getattr(env, "_alloc_walk", None) or {}).get(env.selected_net_path) if policy == "alloc" else None
        targets = dict(entry["widths"]) if entry else {}
        planned = plan_table(origin, plan, targets) if entry else None
        groups = []
        for group, row in plan:
            producers = sorted(names[id(p)] for p in group.producers if id(p) in names)
            groups.append({"key": keys[row], "row": int(row), "producers": producers, "origin": int(group.width),
                           "plan": int(targets[row]) if row in targets else None,
                           "cut": min((cut[p] for p in producers if p in cut), default=None)})
        os.makedirs(folder, exist_ok=True)
        stem = os.path.join(folder, f"{name}__{label}")
        meta = {"network": str(net_path), "label": str(label), "step": int(point["step"]),
                "param": float(point["param"]), "flop": float(point["flop"]), "seed": run_seed(env),
                "policy": policy, "kind": kind() if policy == "alloc" else None, "budget": budget(),
                "walk_target": None if getattr(env, "target_keep", None) is None else float(env.target_keep),
                "fallback": bool(entry["fallback"]) if entry else None,
                "plan_sha1": table_digest(planned) if planned is not None else None, "cut_sha1": table_digest(cut),
                "state_sha1": state_digest(model),
                "files": {"plan": os.path.basename(stem) + "__plan.json" if planned is not None else None,
                          "cut": os.path.basename(stem) + "__cut.json"},
                "groups": groups}
        if planned is not None:
            _write_json(stem + "__plan.json", planned)
        _write_json(stem + "__cut.json", cut)
        _write_json(stem + "__meta.json", meta)
        utils.print_flush(
            f"[eval] saved widths {label} {name} step={int(point['step'])} | "
            + (f"plan {len(planned)} names sha {meta['plan_sha1'][:12]}" if planned is not None else "no plan")
            + f" | cut {len(cut)} modules sha {meta['cut_sha1'][:12]} | net sha {meta['state_sha1'][:12]} | "
            f"{sum(g['plan'] is not None and g['cut'] != g['plan'] for g in groups)} of {len(groups)} groups off "
            f"their planned width | {folder}")
        return meta
    except Exception as error:  # noqa: BLE001 - the save must not cost the eval
        utils.print_flush(f"[eval] widths save failed for {label} {name}: {type(error).__name__}: {error}; continuing")
        return None
    finally:
        rng_restore(rng)


def plan_targets(model, batches, input_shape, kind_name, target, a=0.5, keep_floor=0.1, iters=16, tol=0.003,
                 budget="params", min_width=1, tilt_hold=None):
    """
    ``(widths, info)``: ``widths[row]`` is the target width of the group whose first walk row is
    ``row``. Keeps are clip(c · weight, keep_floor, 1), or 1 for a group ``inner`` holds; c is
    bisected until the one-shot cut keeps ``target`` of the parameters, or of the MACs under
    ``budget="flops"`` (widths are integers, so the closest c found). ``widths`` copies its table
    instead and ignores ``target``. ``min_width`` above 1 (``fortify.plan_min_width``) lifts every
    group's keep floor to ``min_width`` / its origin width, the walk's legal floor. The magnitude kinds
    (``global_mag``, ``lamp``) read no ``batches`` and run no probe: ``_magnitude_curves`` and
    ``_curve_threshold`` pick the keeps, ``_land`` lands them, and ``info`` carries ``tau``, ``est_kept``
    (the cost model's kept fraction at the picked widths), ``weights`` (the picks) and, as ``sens``, the
    largest score a half cut of each group removes. ``tilt_hold`` (None) is read by ``tilt`` alone: ``_state``'s
    ``TiltHold`` under ``SPECTRA_ALLOC_TILT_PARAMS``, handed to ``tilt_plan``.
    """
    size = (lambda m: utils.calc_flops(m, input_shape)) if budget == "flops" else utils.calc_num_parameters
    plan = group_sensitivity.group_plan(ModelWithRows(model))
    rows = [row for _group, row in plan]
    if kind_name == "widths":
        keeps, unnamed = copied_keeps(model, plan, widths_table())
        cut = cut_to(model, plan, keeps, input_shape)
        frac = size(cut) / size(model)
        widths = group_widths(cut, rows)
        del cut
        return widths, {"kind": kind_name, "alpha": float(a), "target": float(target), "kept": float(frac),
                        "keeps": keeps, "origin_widths": group_widths(model, rows),
                        "sens": {row: 1.0 for row in rows}, "held": 0, "unnamed": unnamed}
    if kind_name in TILT_KINDS:  # queue row SAMENET: a base plan's table tilted by one score per group
        return tilt_plan(model, plan, rows, input_shape, target, a, keep_floor, iters, tol, budget, min_width, size,
                         hold=tilt_hold)
    if kind_name in MAGNITUDE_KINDS:
        from src import plan_agent
        floor = {row: keep_floor for row in rows}
        if int(min_width) > 1:
            for (group, row) in plan:
                floor[row] = max(keep_floor, min(1.0, int(min_width) / float(group.width)))
        size0 = size(model)
        cost = (plan_agent.FlopModel(model, plan, input_shape) if budget == "flops"
                else plan_agent.ParamModel(model, plan))
        curve, half = _magnitude_curves(plan, kind_name, floor, cost.cost, cost.total0)
        tau, est, picks = _curve_threshold(curve, floor, cost.total0, target) if curve else (0.0, 1.0, {})
        w = {row: max(picks.get(row, 1.0), floor[row]) for row in rows}
        keeps, frac, widths = _land(model, plan, rows, w, floor, target, size, size0, input_shape, iters, tol)
        utils.print_flush(
            f"[alloc] {kind_name}: one threshold tau={tau:.6g} on the channel scores of {len(curve)} of {len(rows)} "
            f"groups; cost model x{est:.3f} of the {'FLOPs' if budget == 'flops' else 'params'} at the picked widths "
            f"(target x{float(target):.3f}), landed x{frac:.3f}")
        return widths, {"kind": kind_name, "alpha": float(a), "target": float(target), "kept": float(frac),
                        "keeps": keeps, "origin_widths": group_widths(model, rows), "sens": half, "held": 0,
                        "tau": float(tau), "est_kept": float(est), "cost_budget": budget, "weights": w}
    if kind_name in ("sens_cost_at", "sens_curve", "sens_thresh"):
        floor = {row: keep_floor for row in rows}
        if int(min_width) > 1:
            for (group, row) in plan:
                floor[row] = max(keep_floor, min(1.0, int(min_width) / float(group.width)))
        size0 = size(model)
        if kind_name == "sens_cost_at":
            _widths0, info0 = plan_targets(model, batches, input_shape, "sens_cost", target, a, keep_floor, iters, tol,
                                           budget, min_width)
            half, keeps = info0["sens"], info0["keeps"]
            for _ in range(probe_rounds()):
                probe = {row: min(float(keeps[row]), group_sensitivity.SENS_KEEP) for row in rows}
                raw = {}
                sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, input_shape, keep=probe,
                                                                  costs=raw)
                cost = {row: raw[row][1] if budget == "flops" else raw[row][0] for row in rows}
                w = weights("sens_cost", sens, a, cost=cost)
                keeps, frac, widths = _land(model, plan, rows, w, floor, target, size, size0, input_shape, iters, tol)
            return widths, {"kind": kind_name, "alpha": float(a), "target": float(target), "kept": float(frac),
                            "keeps": keeps, "origin_widths": group_widths(model, rows), "sens": sens, "held": 0,
                            "cost": cost, "cost_budget": budget, "probe_keeps": probe, "sens_half": half,
                            "keeps_half": info0["keeps"], "rounds": probe_rounds()}
        qs, loss, raw = curve_probes(model, plan, batches, input_shape)
        curve = probe_curves(rows, qs, loss, raw, budget)
        knob, est_knob, est, w = curve_weights(kind_name, curve, floor, size0, target, rows)
        keeps, frac, widths = _land(model, plan, rows, w, floor, target, size, size0, input_shape, iters, tol)
        return widths, {"kind": kind_name, "alpha": float(a), "target": float(target), "kept": float(frac),
                        "keeps": keeps, "origin_widths": group_widths(model, rows),
                        "sens": {row: loss[row][0.5] if 0.5 in loss[row] else loss[row][qs[0]] for row in rows},
                        "held": 0, "curve": {row: [[k, l, c] for (k, l, c) in curve[row]] for row in rows},
                        "curve_keeps": list(qs), knob: float(est_knob), "est_kept": float(est),
                        "cost_budget": budget, "weights": w}
    base_kind = sample_around() if kind_name == "sample" else kind_name
    cost = None
    probe = sens_keep() if base_kind in ("sens", "sens_cost") else None  # v31: unset runs the calls exactly as before
    if base_kind == "sens_cost":
        raw = {}
        if probe is None:
            sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, input_shape, costs=raw)
        else:
            sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, input_shape, keep=probe, costs=raw)
        cost = {row: raw[row][1] if budget == "flops" else raw[row][0] for row in rows}
        w = weights(base_kind, sens, a, cost=cost)
    else:
        if base_kind == "sens" and probe is not None:
            sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, input_shape, keep=probe)
        elif base_kind == "sens":
            sens, _base = group_sensitivity.group_sensitivity(model, plan, batches, input_shape)
        else:
            sens = {row: 1.0 for row in rows}
        w = weights(base_kind, sens, a)
    sample = None
    if kind_name == "sample":
        sample = {"around": base_kind, "sigma": sample_sigma(), "seed": sample_seed()}
        noise = sample_noise(rows, sample["sigma"], sample["seed"])
        w = {row: w[row] * noise[row] for row in rows}
    held = {row for group, row in plan if kind_name == "inner" and len(group.producers) > 1}
    floor = {row: keep_floor for row in rows}
    if int(min_width) > 1:
        for (group, row) in plan:
            floor[row] = max(keep_floor, min(1.0, int(min_width) / float(group.width)))
    size0 = size(model)
    lo, hi = 0.0, 1.0 / min(w.values())
    best = None
    for _ in range(iters):
        c = 0.5 * (lo + hi)
        keeps = {row: 1.0 if row in held else min(1.0, max(floor[row], c * w[row])) for row in rows}
        cut = cut_to(model, plan, keeps, input_shape)
        frac = size(cut) / size0
        if best is None or abs(frac - target) < abs(best[1] - target):
            best = (keeps, frac, group_widths(cut, rows))
        del cut
        if abs(frac - target) <= tol:
            break
        if frac > target:
            hi = c
        else:
            lo = c
    keeps, frac, widths = best
    info = {"kind": kind_name, "alpha": float(a), "target": float(target), "kept": float(frac),
            "keeps": keeps, "origin_widths": group_widths(model, rows), "sens": sens, "held": len(held)}
    if cost is not None:
        info["cost"] = cost
        info["cost_budget"] = budget
    if sample is not None:
        info["sample"] = sample
    if probe is not None:
        info["probe_keep"] = probe
    return widths, info


def choose(width, target, rates, legal_idx, identity):
    """The legal action whose resulting group width is closest to ``target``; ties go to the milder cut."""
    best, best_key = identity, (abs(width - target), -1.0)
    for idx in legal_idx:
        rate = float(rates[int(idx)])
        if rate >= 1.0 - 1e-9:
            continue
        kept = pruning.target_width(width, rate)
        key = (abs(kept - target), -rate)
        if key < best_key:
            best, best_key = int(idx), key
    return best


def flops_walk_target(env) -> float:
    """The kept MACs a ``SPECTRA_ALLOC_BUDGET=flops`` walk heads for: the episode's fixed target when that is measured
    in FLOPs (``fortify.fixed_target_metric``), else the flop ``SPECTRA_EVAL_SIZE_MATCH``, else the deepest flop
    ``SPECTRA_EVAL_SIZE_POINTS`` entry, else 0.6."""
    if env.target_keep is not None and fortify.fixed_target_metric() == "flop":
        return float(env.target_keep)
    match = fortify.eval_size_match()
    if match is not None and match[0] == "flop":
        return float(match[1])
    points = [t for kind_name, t in fortify.eval_size_points() if kind_name == "flop"]
    return float(min(points)) if points else 0.6


# ------------------------------------------------------------------ v18 grid rounding (SPECTRA_ALLOC_GRID_ROUND)

GRID_BAND = 0.01  # a rounded plan keeps at most the walk target and, where the grid allows, no more than this less


def walk_cut(width, rate):
    """The width ``NetworkEnv.step`` leaves when a ``width``-wide group is cut at ``rate``: the width ladder's keep
    (``fortify.effective_rates``, as ``NetworkEnv._ladder_keep_rate``) when the ladder is on, then
    ``pruning.target_width``. A rate the ladder cannot pay for is an identity step."""
    keep = float(rate)
    if fortify.width_ladder_max() > 0 and 0.0 < keep < 1.0:
        keep, _stop, feasible = fortify.effective_rates({0: keep}, 0.0, group_width=int(width))[0]
        if not feasible:
            return int(width)
    return pruning.target_width(int(width), float(keep))


def walk_to(w0, target, rates, legal_at, max_cuts):
    """``(width, cuts)`` where the walk's ``choose`` chain from ``w0`` aimed at ``target`` stops: ``legal_at(width)``
    gives the legal action indices at that width, ``walk_cut`` the cut, and at most ``max_cuts`` cuts are made."""
    identity = next((i for i, r in rates.items() if abs(float(r) - 1.0) < 1e-9), 0)
    width, cuts = int(w0), 0
    while cuts < int(max_cuts):
        pick = choose(width, int(target), rates, legal_at(width), identity)
        if pick == identity:
            break
        new = walk_cut(width, rates[pick])
        if new >= width:
            break
        width, cuts = new, cuts + 1
    return width, cuts


def grid_widths(w0, rates, legal_at, max_cuts):
    """``{width: cuts}``: every width the walk lands on exactly from ``w0`` (``walk_to`` aimed at it stops there) and
    the cuts it takes; ``w0`` itself at none."""
    out = {int(w0): 0}
    for target in range(int(w0) - 1, 0, -1):
        width, cuts = walk_to(w0, target, rates, legal_at, max_cuts)
        if width == target:
            out[target] = cuts
    return out


def walk_grids(model, plan, rates, passes, groups=None, aims=None):
    """
    ``{row: {"grid": {width: cuts}, "cap": max cuts, "walk": width or None}}`` for each planned group of ``model``
    (the origin): the widths the walk reaches exactly (``grid_widths``) under the legal mask the env builds
    (``fortify.legal_action_mask``: identity only at alive <= 1, on stem rows and at or below ``min_width_for_prune``
    under fortify, no-op rates illegal; every row identity under ``SPECTRA_PROTECT_STREAMS`` for a residual stream), the
    env's cut (``walk_cut``) and the cuts ``passes`` passes allow (one per pass under ``SPECTRA_GROUP_ONCE_PER_PASS``,
    else one per non-stem row producing the group). ``walk`` is where the walk aimed at ``aims[row]`` stops. Eval
    rollback locks, masked cuts and the eval size floors act at walk time only (``_grid_watch`` reports them).
    """
    mwr = ModelWithRows(model)
    if groups is None:
        groups = channel_groups.build_channel_groups(mwr.model) or []
    first = {id(group): row for group, row in plan}
    visits = {row: [] for _group, row in plan}
    for row in sorted(mwr.row_to_main_layer)[:-1]:
        group = channel_groups.group_of(groups, mwr.all_layers[mwr.row_to_main_layer[row]])
        if group is not None and id(group) in first:
            visits[first[id(group)]].append(row)
    stem = fortify.stem_rows() if fortify.fortify_enabled() else 0
    masks = {}

    def legal_for(row_index):
        def at(width):
            key = (row_index, int(width))
            if key not in masks:
                mask = fortify.legal_action_mask(rates, row_index=row_index, alive_count=int(width), device="cpu")
                masks[key] = [int(i) for i in mask.nonzero(as_tuple=False).flatten().tolist()]
            return masks[key]
        return at

    out = {}
    for group, row in plan:
        protected = fortify.protect_streams() and fortify.is_residual_stream(group)
        cuttable = [] if protected else [r for r in visits[row] if r >= stem]
        cap =int(passes) * (1 if fortify.group_once_per_pass() else len(cuttable)) if cuttable else 0
        legal_at = legal_for(cuttable[0] if cuttable else row)
        w0 = int(group.width)
        aim = None if aims is None else aims.get(row)
        out[row] = {"grid": grid_widths(w0, rates, legal_at, cap), "cap": cap,
                    "walk": None if aim is None else walk_to(w0, aim, rates, legal_at, cap)[0]}
    return out


def grid_priority(info, rows):
    """``{row: score}`` the grid rounding steps groups by (lowest first down, highest first up): the decoder's weight
    for the rules (``weights`` of ``info``'s sensitivities, with its cost under ``sens_cost`` and its noise under
    ``sample``), the score z for the agent (log prior weight + z for a T2 residual plan); None for one weight on every group (uniform, inner, widths), which steps
    by closeness to the plan instead."""
    kind_name = info.get("kind")
    sens = info.get("sens") or {}
    if kind_name in ("agent", "agent_sample", "randbest"):
        z = {row: float(sens.get(row, 0.0)) for row in rows}
        prior = (info.get("prior") or {}).get("weights") if info.get("residual") else None
        if prior:  # T2: a residual plan ranks groups by log(prior weight) + z (plan_agent.residual_decode)
            zmax = float((info.get("prior") or {}).get("zmax", 0.0))
            if zmax > 0:  # v20 trust region: the decoder exponentiated z_eff = zmax · tanh(z / zmax), so rank by that
                from src import plan_agent
                z = {row: plan_agent.trust_score(z[row], zmax) for row in rows}
            return {row: math.log(max(1e-12, float(prior.get(row, 1.0)))) + z[row] for row in rows}
        return z
    if kind_name in ("sens_curve", "sens_thresh"):  # v30/v31: the decoder's weight is each group's pick on its curve
        return {row: float(info["weights"][row]) for row in rows}
    if kind_name in MAGNITUDE_KINDS:  # as sens_thresh: the decoder's weight is each group's pick at the threshold
        return {row: float(info["weights"][row]) for row in rows}
    if kind_name == "amc":  # the landing's weight is each group's keep in the best episode
        return {row: float(info["weights"][row]) for row in rows}
    if kind_name in TILT_KINDS:  # SAMENET: the decoder's weight is the base plan's keep times exp(z) (exp(z + λ d), A1)
        return {row: float(info["weights"][row]) for row in rows}
    if kind_name == "sens_cost_at":  # v30: the cost-aware weights of the last (deep) measurement
        kind_name = "sens_cost"
    sample = info.get("sample") if kind_name == "sample" else None
    base = sample["around"] if sample else kind_name
    if base not in ("sens", "sens_cost", "sample") and sample is None:
        return None
    keyed = {row: float(sens.get(row, 1.0)) for row in rows}
    cost = {row: float(info["cost"][row]) for row in rows} if base == "sens_cost" else None
    w = weights(base, keyed, float(info.get("alpha", alpha())), cost=cost)
    if sample:
        noise = sample_noise(rows, sample["sigma"], sample["seed"])
        w = {row: w[row] * noise[row] for row in rows}
    return w


def grid_round_widths(planned, grids, kept_of, target, priority=None, fixed=(), floors=None, band=GRID_BAND):
    """
    ``(widths, report)``: the plan ``planned`` (``{row: width}``) on the walk's grid (``grids[row]``: the widths it
    lands on exactly). Each group starts at the grid width nearest its plan (ties to the wider); then, while the kept
    size ``kept_of(widths)`` is above ``target``, one group steps down to its next grid width, and once it is at or
    below, single groups step back up while it is more than ``band`` below, never above ``target``. A step goes by
    ``priority`` (lowest score first down, highest first up, each group once per round, ties by row order) or, with
    ``priority`` None, to the group the step leaves closest to its plan relative to it. ``fixed`` rows keep their
    planned width; no group steps below ``floors[row]`` (or its start, when that is lower). ``report``: kept, the
    starting widths and the steps down / up.
    """
    rows = list(planned)
    order = {row: k for k, row in enumerate(rows)}
    fixed, floors = set(fixed), floors or {}
    ladder = {row: sorted(grids.get(row) or [planned[row]]) for row in rows}
    widths = {row: int(planned[row]) if row in fixed
              else min(ladder[row], key=lambda w, p=planned[row]: (abs(w - p), -w)) for row in rows}
    snapped = dict(widths)
    lo = {row: min(widths[row], int(floors.get(row, 1))) for row in rows}

    def neighbour(row, sign):
        if sign < 0:
            below = [w for w in ladder[row] if lo[row] <= w < widths[row]]
            return max(below) if below else None
        above = [w for w in ladder[row] if w > widths[row]]
        return min(above) if above else None

    def key(row, width, steps, sign):
        if priority is None:
            return abs(width - planned[row]) / float(max(1, planned[row])), order[row]
        return steps[row], sign * float(priority[row]), order[row]

    kept, down, up = float(kept_of(widths)), {row: 0 for row in rows}, {row: 0 for row in rows}
    while kept > target + 1e-9:
        moves = [(key(row, w, down, 1), row, w) for row in rows if row not in fixed
                 for w in [neighbour(row, -1)] if w is not None]
        if not moves:
            break
        _key, row, w = min(moves)
        widths[row], down[row] = w, down[row] + 1
        kept = float(kept_of(widths))
    while kept < target - band - 1e-9:
        moves = sorted((key(row, w, up, -1), row, w) for row in rows if row not in fixed
                       for w in [neighbour(row, 1)] if w is not None)
        for _key, row, w in moves:
            trial = dict(widths)
            trial[row] = w
            k = float(kept_of(trial))
            if k <= target + 1e-9:
                widths, kept, up[row] = trial, k, up[row] + 1
                break
        else:
            break
    return widths, {"kept": kept, "snapped": snapped, "down": sum(down.values()), "up": sum(up.values())}


def _grid_round_state(env, net, entry, model, info, walk_target, flops):
    """``SPECTRA_ALLOC_GRID_ROUND``: ``entry``'s plan on the walk's grid (``walk_grids``, ``grid_round_widths``) on the
    budget's cost model (``plan_agent.ParamModel`` / ``FlopModel``), measured on the origin ``model``; logged, recorded,
    and kept in ``entry["grid"]`` for ``_grid_watch``. Inner's held streams stay whole, as does every group a widths
    table leaves at its origin width; no group steps below the plan floors (``min_keep``,
    ``fortify.plan_min_width``)."""
    from src import plan_agent
    if fortify.action_menu() == "budget":
        utils.print_flush("[alloc] WARNING grid round skipped: SPECTRA_ACTION_MENU=budget prices each cut on the net "
                          "as the walk finds it, which no offline grid follows")
        return
    rates = env.conf.compression_rates_dict
    passes = max(1, int(getattr(env.conf, "passes", 1) or 1))
    mwr = ModelWithRows(model)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    plan = group_sensitivity.group_plan(mwr, groups)
    widths = entry["widths"]
    planned = {row: int(widths[row]) for _group, row in plan if row in widths}
    cost = (plan_agent.FlopModel(model, plan, env._input_shape(), env.conf.device) if flops
            else plan_agent.ParamModel(model, plan))
    walks = walk_grids(model, plan, rates, passes, groups, aims=planned)
    held = {row for group, row in plan if info["kind"] == "inner" and len(group.producers) > 1 and row in planned}
    if info["kind"] == "widths":
        held |= {row for row, w in planned.items() if w >= cost.widths0[row]}
    floor = fortify.plan_min_width()
    floors = {row: max(1, int(round(min_keep() * w0)), min(w0, floor)) for row, w0 in cost.widths0.items()}
    rounded, report = grid_round_widths(planned, {row: list(walks[row]["grid"]) for row in planned}, cost.kept,
                                        walk_target, grid_priority(info, list(cost.rows)), fixed=held, floors=floors)
    entry["widths"] = {**widths, **rounded}
    entry["grid"] = {"targets": dict(rounded), "checked": 1.0, "below": set(), "fallback": False}
    moved = sum(rounded[row] != planned[row] for row in planned)
    grid = [float(rates[i]) for i in sorted(rates)]
    unit = "FLOPs" if flops else "params"
    utils.print_flush(
        f"[alloc] grid round: {moved} groups moved; predicted x{report['kept']:.3f} of the {unit} (walk target "
        f"x{walk_target:.3f}); grid {grid}"
        + ("" if report["kept"] <= walk_target + 1e-9 else
           "; WARNING: no grid plan at or below the walk target, the stall fallback stays the safety net"))
    try:
        import src.run_recorder as run_recorder
        run_recorder.record(
            "alloc_grid_round", network=str(net), kind=info["kind"], budget="flops" if flops else "params",
            walk_target=walk_target, kept=report["kept"], plan_kept=float(cost.kept(planned)),
            walk_kept=float(cost.kept({row: walks[row]["walk"] for row in planned})), moved=moved,
            down=report["down"], up=report["up"], rates=grid, passes=passes,
            rows={str(r): {"origin": cost.widths0[r], "plan": planned[r], "walk": walks[r]["walk"],
                           "grid": rounded[r], "cuts": walks[r]["grid"].get(rounded[r])} for r in planned})
    except Exception:  # noqa: BLE001 - the record is a convenience
        pass


def _grid_watch(env, entry):
    """Walk-time check of a grid-rounded plan: a WARNING once per group the walk takes below its grid width, and once
    if the stall fallback fires anyway. Either means a legality the offline grid did not see (an eval rollback lock, a
    masked cut, an eval size floor)."""
    grid = entry["grid"]
    try:
        if entry["fallback"] and not grid["fallback"]:
            grid["fallback"] = True
            now = group_widths(env.current_model, list(grid["targets"]))
            off = [f"row {r}: {now.get(r)} vs {t}" for r, t in grid["targets"].items() if now.get(r) != t]
            utils.print_flush(f"[alloc] WARNING grid round: the stall fallback fired with {len(off)} groups off their "
                              f"grid widths ({'; '.join(off[:8])}{'; ...' if len(off) > 8 else ''})")
        kept = float(env.flops_ratio() if budget() == "flops" else env.param_ratio())
        if kept >= grid["checked"] - 1e-12:
            return
        grid["checked"] = kept
        now = group_widths(env.current_model, list(grid["targets"]))
        for row, target in grid["targets"].items():
            width = now.get(row)
            if width is not None and width < target and row not in grid["below"]:
                grid["below"].add(row)
                utils.print_flush(f"[alloc] WARNING grid round: the walk took the group at row {row} to width "
                                  f"{width}, below its grid width {target}")
    except Exception as error:  # noqa: BLE001 - a diagnostic must not end the walk
        if not grid.get("error"):
            grid["error"] = True
            utils.print_flush(f"[alloc] grid watch unavailable ({type(error).__name__}: {error})")


# ------------------------------------------------------------------ v23 randbest (SPECTRA_ALLOC_KIND=randbest)


def randbest_plan(env, target, budget="params"):
    """
    ``(widths, info)`` of the best of N random plans in the plan agent's action space, in ``plan_agent.plan_for_env``'s
    format with ``kind`` randbest, so ``_state`` logs, records, grid-rounds and walks it exactly as the agent's mean
    plan: the per-network random-search baseline of Li et al. (CVPR 2022) scored with the agent's own training proxy.

    Draw i's scores are z_g ~ N(0, σ²) per planned group (``randbest_draws``; σ ``randbest_sigma``, N ``randbest_n``,
    the generator seeded by ``randbest_seed`` of ``SPECTRA_ALLOC_SAMPLE_SEED`` and the run's seed), decoded by
    ``plan_agent.decode`` at ``target`` on the ``budget``'s cost model (``ParamModel`` / ``FlopModel``) with the k_min
    and width floor ``plan_for_env`` would decode the agent with: the checkpoint's ``k_min`` and, when
    ``fortify.plan_min_width`` is off, its ``min_width`` if ``SPECTRA_PLAN_AGENT`` names the agent of record (read
    without building it, so the RNG stands), else ``min_keep`` and the flag's floor. Each plan is scored by
    ``plan_proxies.PlanScorer`` with ``randbest_proxy`` on ``env.val_loader`` only, on the batches
    ``SPECTRA_EVAL_PROXY_SEED`` draws once for all draws (common random numbers; ``SPECTRA_PLAN_FS_LR`` for fs<S>); the
    highest val accuracy wins, ties to the lower draw index (indices are 0-based). The search runs inside
    ``plan_proxies.seeded`` on its own seed, so the Python, NumPy, torch CPU and CUDA RNG states are put back after it
    and the walk and final fine-tune draw as they do under ``agent``. Only the per-draw scores, decoded widths and
    landed params / FLOPs ratios are kept (the FLOPs through a ``FlopModel``, built for the report under a params
    budget), never N models. One line per draw, a summary line and an ``alloc_randbest`` record carry every draw.
    ``SPECTRA_PLAN_RESIDUAL`` is refused: the draws live in the plain decoder's space, not a residual on a prior.
    """
    from src import plan_agent, plan_proxies
    if budget not in BUDGETS:
        raise ValueError(f"budget={budget!r}; expected one of {BUDGETS}")
    n, sigma, proxy = randbest_n(), randbest_sigma(), randbest_proxy()
    residual = fortify.plan_residual(default=None)
    if residual not in (None, "off"):
        raise ValueError(f"SPECTRA_ALLOC_KIND=randbest draws in the plain decoder's space (plan_agent.decode); "
                         f"SPECTRA_PLAN_RESIDUAL={residual} would decode the agent on a prior instead")
    seeds = (sample_seed(), run_seed(env))
    seed, proxy_seed = randbest_seed(*seeds), plan_proxies.proxy_seed()
    net = os.path.basename(str(env.selected_net_path))
    device = env.conf.device
    k_min, min_width = min_keep(), fortify.plan_min_width()
    agent = os.environ.get("SPECTRA_PLAN_AGENT", "").strip()
    if agent:  # the agent of record's checkpoint: its k_min, and its floor when no flag sets one, as plan_for_env reads
        blob = plan_agent.read_blob(plan_agent.ensemble_paths(agent)[0])
        k_min = float(blob.get("k_min", k_min))
        if min_width <= 1:
            min_width = max(1, int(blob.get("min_width", 1)))
    flops = budget == "flops"
    started = time.perf_counter()
    draws, best = [], None
    with plan_proxies.seeded(seed):  # the search's own stream; every global RNG state is put back after it
        model = env.current_model.to(device)
        mwr = ModelWithRows(model)
        plan = group_sensitivity.group_plan(mwr, env._dependency_groups(mwr))
        shape = env._input_shape()
        pm = plan_agent.FlopModel(model, plan, shape, device) if flops else plan_agent.ParamModel(model, plan)
        fm = pm if flops else plan_agent.FlopModel(model, plan, shape, device)
        params0 = float(pm.pm.total0 if flops else pm.total0)
        z = randbest_draws(n, len(pm.rows), sigma, seed)
        scorer = plan_proxies.PlanScorer(model, plan, env.val_loader, device, train_loader=env.train_loader,
                                         input_shape=shape, create_handler=getattr(env, "create_learning_handler", None),
                                         name=net)
        batches = scorer.batches(proxy, proxy_seed)
        for i in range(n):
            widths, decoded = plan_agent.decode(z[i], pm, target, k_min, min_width=min_width)
            score = float(scorer.score(widths, proxy, batches))
            p, f = pm.params(widths) / params0, float(fm.kept(widths))
            draws.append({"score": score, "kept": float(decoded["kept"]), "params": float(p), "flops": f,
                          "widths": widths})
            utils.print_flush(f"[alloc] randbest {net} draw={i} score={score:.4f} params x{p:.3f} FLOPs x{f:.3f}")
            if best is None or score > draws[best]["score"]:
                best = i
        del scorer, batches
    minutes = (time.perf_counter() - started) / 60.0
    scores = [d["score"] for d in draws]
    utils.print_flush(
        f"[alloc] randbest {net}: best draw {best} of {n}, val {scores[best]:.4f} (spread {min(scores):.4f}-"
        f"{max(scores):.4f}, median {statistics.median(scores):.4f}), proxy {proxy}, sigma {sigma:g}, {minutes:.1f} min")
    try:
        import src.run_recorder as run_recorder
        run_recorder.record(
            "alloc_randbest", network=str(env.selected_net_path), proxy=proxy, sigma=sigma, n=n, seed=seed,
            sample_seed=seeds[0], run_seed=seeds[1], proxy_seed=proxy_seed, budget=budget, target=float(target),
            k_min=k_min, **({"min_width": min_width} if min_width > 1 else {}), **({"agent": agent} if agent else {}),
            **({"fs_lr": plan_agent.fs_lr()} if proxy.startswith("fs") else {}),
            pick=best, score=scores[best], minutes=round(minutes, 3),
            draws=[{"score": d["score"], "kept": d["kept"], "params": d["params"], "flops": d["flops"],
                    "widths": {str(r): int(w) for r, w in d["widths"].items()}} for d in draws])
    except Exception:  # noqa: BLE001 - the record is a convenience
        pass
    widths = draws[best]["widths"]
    summary = {"pick": best, "n": n, "score": scores[best], "proxy": proxy, "sigma": sigma, "seed": seed,
               "minutes": minutes, "scores": scores}
    return widths, {"kind": "randbest", "alpha": 0.0, "target": float(target), "kept": draws[best]["kept"],
                    "budget": budget, "keeps": {row: widths[row] / float(pm.widths0[row]) for row in pm.rows},
                    "origin_widths": dict(pm.widths0), "sens": {row: float(v) for row, v in zip(pm.rows, z[best])},
                    "held": 0, "randbest": summary, **({"min_width": int(min_width)} if min_width > 1 else {})}


# ------------------------------------------------------------------ v27 fallback (SPECTRA_ALLOC_KIND=fallback)


def fallback_pick(agent_score: float, prior_score: float, margin: float) -> str:
    """``"prior"`` iff ``prior_score > agent_score + margin``, else ``"agent"``: a tie goes to the agent."""
    return "prior" if float(prior_score) > float(agent_score) + float(margin) else "agent"


def fallback_pick_multi(agent_score: float, prior_scores, margin: float) -> str:
    """The name of the best-scoring prior of ``prior_scores`` (``[(name, score), ...]`` in the listed order; a tie
    between priors goes to the first listed) iff its score is above ``agent_score + margin`` (as ``fallback_pick``), else
    ``"agent"``: a tie with the agent's bar goes to the agent."""
    best_name, best = None, None
    for name, score in prior_scores:
        if best is None or float(score) > best:
            best_name, best = name, float(score)
    return best_name if best is not None and best > float(agent_score) + float(margin) else "agent"


def rng_snapshot() -> dict:
    """The Python, NumPy, torch CPU and (when CUDA is available) all CUDA RNG states, for ``rng_restore``."""
    state = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def rng_restore(state: dict) -> None:
    """Put back the generators ``rng_snapshot`` read."""
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["np"])
    random.setstate(state["py"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _state_dict_equal(model, snapshot) -> bool:
    """True iff ``model.state_dict()`` has ``snapshot``'s keys and every tensor is ``torch.equal`` to its copy;
    ``snapshot`` is ``{k: v.detach().clone() for k, v in model.state_dict().items()}``."""
    now = model.state_dict()
    if set(now) != set(snapshot):
        return False
    return all(torch.equal(v.detach(), snapshot[k]) for k, v in now.items())


def _modes(model) -> dict:
    """``{module name: training flag}`` of every module of ``model``, for ``_set_modes``."""
    return {name: bool(module.training) for name, module in model.named_modules()}


def _set_modes(model, modes) -> None:
    """Put back the train / eval flags ``_modes`` read, module by module (``Module.train(mode)`` would recurse)."""
    for name, module in model.named_modules():
        if name in modes:
            module.training = modes[name]


def fallback_plan(env, target, flops, agent_kw, plan_kw):
    """
    ``(widths, info)`` of the plan ``_state`` walks under ``SPECTRA_ALLOC_KIND=fallback``: the frozen agent's plan
    unless the hand rule ``fallback_prior`` plans better on the val half by more than ``fallback_margin`` on the first of
    ``fallback_proxies``, in which case the rule's plan. ``target``, ``flops``, ``agent_kw`` and ``plan_kw`` are the
    values ``_state`` computed for its other kinds.

    The prior is planned by exactly the rule arm's calls (``group_sensitivity.calibration_batches`` and ``plan_targets``
    with ``plan_kw``), then the RNG is put back to its starting state and the agent is planned by exactly the agent arm's
    call (``plan_agent.plan_for_env`` with ``agent_kw``), each from the same starting RNG state and origin as its own row
    makes it; the state after each plan is kept. Both plans are then scored by ``plan_proxies.PlanScorer`` (the agent
    first) with every proxy, on the val half only, on the batches ``SPECTRA_EVAL_PROXY_SEED`` draws once per proxy and
    shares between the two plans (common random numbers), inside ``plan_proxies.seeded``, which puts the RNG back to the
    agent plan's state on exit. The scored widths are the decoders' own, before any grid rounding, as randbest scores
    its draws. The first proxy decides (``fallback_pick``); the others are reported. The origin's weights are checked
    after each plan and after the scoring (a RuntimeError names the step that changed them), and its modules' train /
    eval flags are put back to the start before the agent plans, and to the state the picked arm's plan left after the
    scoring. The RNG is then set to the
    state the picked arm alone leaves, so the grid round, the walk and the final fine-tune draw exactly as under that
    arm, and the picked arm's ``(widths, info)`` is returned unchanged except that a shallow copy of ``info`` gains one
    key, ``fallback`` (pick, prior, proxies, margin, scores, minutes): ``info["kind"]`` stays the arm's own, so
    ``grid_priority``, ``_grid_round_state`` and ``_state``'s plan line treat the plan as that arm's. Nothing is refused
    about ``SPECTRA_PLAN_RESIDUAL``: the agent of record is a residual agent and runs as ``agent`` runs it. One line per
    proxy, a decision line and an ``alloc_fallback`` record carry every score and both plans' widths. An effect
    ``plan_for_env`` has on the env itself stays when the prior is picked. So does a DataLoader iterator that a plan
    first creates: with persistent workers (``utils.DATALOADER_WORKERS`` > 0, every loader but the device-resident
    ``GpuCropFlipLoader`` of CIFAR under GPU augmentation), the second plan reuses an iterator its own row would create, and
    its RNG draws can differ. The equality with the picked arm's row is therefore a check run on the GPU before any read.

    v32 (default off; ``SPECTRA_ALLOC_FALLBACK_PRIOR`` a list, ``fallback_priors``): each prior in the listed order is
    planned from the starting RNG state and origin (the first as above, the others after putting both back), each with
    the calibration batches its own row draws; the agent is then planned from the starting state as above. Every proxy
    scores the agent first and then the priors in the listed order, on the same batches, inside the same ``seeded``
    block. ``fallback_pick_multi`` picks the best-scoring prior if it beats the agent by the margin on the first proxy
    (``info["fallback"]["pick_name"]``; ``pick`` stays ``agent`` or ``prior``), and ``pick2`` is the two-way pick
    (``fallback_pick``) of the agent against the first prior on the same scores. One extra line per further prior and
    proxy, the decision line names both picks, and the record gains ``priors``, ``pick_name``, ``pick2`` and
    ``prior_widths_all``; ``scores[proxy]`` gains ``priors`` ({name: score}). ``SPECTRA_ALLOC_FALLBACK_DETAIL=1`` also logs
    every proxy's val loss, correctness bits and draws per arm (``PlanScorer.score_detail``; the record's ``details``
    and one line per proxy and arm). With one prior and the detail off, every line, record key and the info are v27's.
    """
    from src import plan_agent, plan_proxies
    priors, proxies, margin = fallback_priors(), fallback_proxies(), fallback_margin()
    prior, multi, detail = priors[0], len(priors) > 1, fallback_detail()
    net = os.path.basename(str(env.selected_net_path))
    started = time.perf_counter()
    model = env.current_model.to(env.conf.device)
    m0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    modes0 = _modes(model)
    s0 = rng_snapshot()

    def intact(step):
        if not _state_dict_equal(model, m0):
            raise RuntimeError(f"{net}: the {step} changed the origin's weights")

    planned = {}  # prior name -> (widths, info, rng snapshot after its plan, modes after its plan)
    for i, name in enumerate(priors):
        if i:
            _set_modes(model, modes0)
            rng_restore(s0)
        batches = group_sensitivity.calibration_batches(env.train_loader, group_sensitivity.CALIB_BATCHES,
                                                        env.conf.device)
        widths_p, info_p = plan_targets(model, batches, env._input_shape(), name, target, alpha(), min_keep(),
                                        **plan_kw)
        planned[name] = (widths_p, info_p, rng_snapshot(), _modes(model))
        intact(f"{name} plan")
    widths_p = planned[prior][0]
    _set_modes(model, modes0)
    rng_restore(s0)
    widths_a, info_a = plan_agent.plan_for_env(env, target, agent_path(), min_keep(), **agent_kw)
    s_a, modes_a = rng_snapshot(), _modes(model)
    intact("agent plan")
    scores, details = {}, {}
    with plan_proxies.seeded(plan_proxies.proxy_seed()):
        mwr = ModelWithRows(model)
        plan = group_sensitivity.group_plan(mwr, env._dependency_groups(mwr))
        scorer = plan_proxies.PlanScorer(model, plan, env.val_loader, env.conf.device, train_loader=env.train_loader,
                                         input_shape=env._input_shape(),
                                         create_handler=getattr(env, "create_learning_handler", None), name=net)
        for proxy in proxies:
            b = scorer.batches(proxy, plan_proxies.proxy_seed())
            reads = {}
            for arm, arm_widths in (("agent", widths_a), *((name, planned[name][0]) for name in priors)):
                if detail:
                    reads[arm] = scorer.score_detail(arm_widths, proxy, b)
                    reads[arm]["acc"] = float(reads[arm]["acc"])
                else:
                    reads[arm] = {"acc": float(scorer.score(arm_widths, proxy, b))}
            scores[proxy] = {"agent": reads["agent"]["acc"], "prior": reads[prior]["acc"],
                             **({"priors": {name: reads[name]["acc"] for name in priors}} if multi else {})}
            if detail:
                details[proxy] = {arm: dict(read) for arm, read in reads.items()}
            del b
        del scorer
    intact("proxy scoring")
    first = scores[proxies[0]]
    pick2 = fallback_pick(first["agent"], first["prior"], margin)
    if multi:
        pick_name = fallback_pick_multi(first["agent"], [(name, first["priors"][name]) for name in priors], margin)
        pick = "agent" if pick_name == "agent" else "prior"
    else:
        pick = pick2
        pick_name = prior if pick == "prior" else "agent"
    if pick == "prior":
        widths_pick, info_pick, s_pick, modes_pick = planned[pick_name]
    else:
        widths_pick, info_pick, s_pick, modes_pick = widths_a, info_a, s_a, modes_a
    _set_modes(model, modes_pick)
    rng_restore(s_pick)
    minutes = (time.perf_counter() - started) / 60.0
    for proxy in proxies:
        a, p = scores[proxy]["agent"], scores[proxy]["prior"]
        utils.print_flush(f"[alloc] fallback {net} {proxy}: agent {a:.4f} prior {prior} {p:.4f} "
                          f"(prior − agent {p - a:+.4f})")
        for i, name in enumerate(priors[1:], start=2):
            p = scores[proxy]["priors"][name]
            utils.print_flush(f"[alloc] fallback {net} {proxy}: agent {a:.4f} prior{i} {name} {p:.4f} "
                              f"(prior{i} − agent {p - a:+.4f})")
    for proxy in (proxies if detail else ()):
        for arm, read in details[proxy].items():
            utils.print_flush(f"[alloc] fallback {net} {proxy} {arm}: acc {read['acc']:.4f} loss {read['loss']:.4f}")
    if multi:
        utils.print_flush(f"[alloc] fallback {net}: pick {pick_name} on {proxies[0]} (margin {margin:g}; "
                          f"{minutes:.1f} min) | two-way pick {pick2} ({prior})")
    else:
        utils.print_flush(f"[alloc] fallback {net}: pick {pick} on {proxies[0]} (margin {margin:g}; {minutes:.1f} min)")
    try:
        import src.run_recorder as run_recorder
        run_recorder.record(
            "alloc_fallback", network=str(env.selected_net_path), prior=prior, proxies=list(proxies), margin=margin,
            pick=pick, budget="flops" if flops else "params", target=float(target),
            proxy_seed=plan_proxies.proxy_seed(), scores=scores,
            agent_widths={str(r): int(w) for r, w in widths_a.items()},
            prior_widths={str(r): int(w) for r, w in widths_p.items()}, minutes=round(minutes, 3),
            **({"fs_lr": plan_agent.fs_lr()} if any(p.startswith("fs") for p in proxies) else {}),
            **({"priors": list(priors), "pick_name": pick_name, "pick2": pick2,
                "prior_widths_all": {name: {str(r): int(w) for r, w in planned[name][0].items()} for name in priors}}
               if multi else {}),
            **({"details": details} if detail else {}))
    except Exception:  # noqa: BLE001 - the record is a convenience
        pass
    info = dict(info_pick)
    info["fallback"] = {"pick": pick, "prior": prior, "proxies": list(proxies), "margin": margin, "scores": scores,
                        "minutes": minutes,
                        **({"priors": list(priors), "pick_name": pick_name, "pick2": pick2} if multi else {})}
    return widths_pick, info


def _state(env):
    """Per-network plan, measured on the origin the first time the network comes round; under
    ``SPECTRA_ALLOC_GRID_ROUND`` put on the walk's grid (``_grid_round_state``) and checked at every later call
    (``_grid_watch``)."""
    cache = getattr(env, "_alloc_walk", None)
    if cache is None:
        cache = env._alloc_walk = {}
    net = env.selected_net_path
    if net not in cache:
        model = env.current_model.to(env.conf.device)
        flops = budget() == "flops"
        walk_target = float(env.target_keep if env.target_keep is not None else 0.6)
        if flops:
            walk_target = flops_walk_target(env)
        target = walk_target - undershoot()
        # Under the params budget every call below is the one ``tests/test_plan_agent.py`` fakes (no extra arguments).
        agent_kw = {"budget": "flops", "kappa": walk_target} if flops else {}
        plan_kw = {"budget": "flops"} if flops else {}
        floor = fortify.plan_min_width()
        if floor > 1:
            # plan_for_env reads SPECTRA_PLAN_MIN_WIDTH (or the policy blob's floor) itself; only the rule plans take it.
            plan_kw["min_width"] = floor
        if kind() in ("agent", "agent_sample"):
            from src import plan_agent
            if kind() == "agent_sample":
                widths, info = plan_agent.plan_for_env(env, target, agent_path(), min_keep(),
                                                       sample=(sample_sigma(), sample_seed()), **agent_kw)
            else:
                widths, info = plan_agent.plan_for_env(env, target, agent_path(), min_keep(), **agent_kw)
        elif kind() == "randbest":
            widths, info = randbest_plan(env, target, budget="flops" if flops else "params")
        elif kind() == "fallback":
            widths, info = fallback_plan(env, target, flops, agent_kw, plan_kw)
        elif kind() == "amc":
            from src import amc_search  # imported under this kind only
            widths, info = amc_search.amc_plan(env, target, budget="flops" if flops else "params")
        elif kind() in TILT_KINDS:  # SAMENET: the plan reads the base table and the scores; no batch is drawn
            if tilt_params() is not None:  # amendment A1 (default off): the aimed plan held at the base plan's params
                plan_kw = dict(plan_kw, tilt_hold=tilt_hold_for(env, model, walk_target, flops))
            widths, info = plan_targets(model, None, env._input_shape(), kind(), target, alpha(), min_keep(),
                                        **plan_kw)
        else:
            batches = group_sensitivity.calibration_batches(
                env.train_loader, group_sensitivity.CALIB_BATCHES, env.conf.device)
            widths, info = plan_targets(model, batches, env._input_shape(), kind(), target, alpha(), min_keep(),
                                        **plan_kw)
        mwr = ModelWithRows(model)
        cache[net] = {"widths": widths, "n_rows": max(1, len(mwr.row_to_main_layer) - 1),
                      "idle": 0, "fallback": False, "last_kept": 1.0}
        keeps = sorted(info["keeps"].values())
        source = (f"widths of {os.path.basename(os.environ.get('SPECTRA_ALLOC_WIDTHS', ''))}"
                  if info["kind"] == "widths" else f"{info['kind']} alpha={info['alpha']:g}")
        if info["kind"] == "agent":
            source = f"agent {info['policy']}"
        if info["kind"] == "agent_sample":
            s = info["sample"]
            source = (f"agent_sample sigma={s['sigma']:g}" + (f" sigma_u={s['sigma_u']:g}" if "sigma_u" in s else "")
                      + f" seed={s['seed']} around the mean of {info['policy']}")
        if info["kind"] == "randbest":
            r = info["randbest"]
            source = (f"randbest best draw {r['pick']} of {r['n']} (val {r['score']:.4f}, proxy {r['proxy']}, "
                      f"sigma {r['sigma']:g}, seed {r['seed']})")
        if info["kind"] == "amc":
            r = info["amc"]
            source = (f"amc best episode {r['pick']} of {r['episodes']} (val {r['score']:.4f}, proxy {r['proxy']}, "
                      f"warm-up {r['warmup']}, seed {r['seed']})")
        if info["kind"] == "sample" and info.get("sample"):
            sample = info["sample"]
            source = (f"sample around {sample['around']} sigma={sample['sigma']:g} seed={sample['seed']} "
                      f"alpha={info['alpha']:g}")
        aimed = f"(target x{target:.3f} = walk target − {undershoot():g})"
        if info["kind"] in TILT_KINDS:  # SAMENET: the base table, where the scores come from, the size bisected to
            tilt = info["tilt"]
            source = (f"tilt of {os.path.basename(tilt['base'])}, "
                      + (f"z from {os.path.basename(tilt['source']['path'])}" if tilt["source"]["z"] == "file"
                         else f"z drawn sigma={tilt['source']['sigma']:g} seed={tilt['source']['seed']}"))
            if tilt["target"] == "base":
                aimed = f"(target x{info['target']:.3f} = the base plan's size)"
        utils.print_flush(
            f"[alloc] {os.path.basename(str(net))}: {source} plan keeps "
            f"x{info['kept']:.3f} of the {'FLOPs' if flops else 'params'} {aimed} over "
            f"{len(widths)} groups; group keep min {keeps[0]:.2f} median {statistics.median(keeps):.2f} "
            f"max {keeps[-1]:.2f}" + (f"; {info['held']} coupled groups held at full width" if info["held"] else "")
            + (f"; {info['unnamed']} groups not named in the table, kept whole" if info.get("unnamed") else "")
            + (f"; group width floor {floor}" if floor > 1 else ""))
        if fortify.cut_rank():
            utils.print_flush(
                f"[alloc] cut ranking: {fortify.cut_rank()}"
                + (f" (seed {pruning.random_rank_seed()})" if fortify.cut_rank() == "random" else "")
                + f" for {os.path.basename(str(net))}")
        try:
            import src.run_recorder as run_recorder
            run_recorder.record(
                "alloc_plan", network=str(net), kind=info["kind"], alpha=info["alpha"], target=target,
                kept=info["kept"], **({"sample": info["sample"]} if info.get("sample") else {}),
                **({"randbest": {key: info["randbest"][key] for key in ("pick", "n", "score", "proxy", "sigma", "seed")}}
                   if info.get("randbest") else {}),
                **({"amc": {key: info["amc"][key] for key in ("pick", "episodes", "warmup", "score", "proxy", "seed")}}
                   if info.get("amc") else {}),
                **({"budget": "flops"} if flops else {}), **({"min_width": floor} if floor > 1 else {}),
                rows={str(r): {"origin": info["origin_widths"].get(r), "target": widths[r],
                                                  "keep": round(info["keeps"][r], 4),
                                                  "sens": float(info["sens"][r])} for r in widths})
        except Exception:  # noqa: BLE001 - the record is a convenience
            pass
        if info["kind"] in TILT_KINDS:  # SAMENET: the group order, the scores, the keeps, the plan's params and FLOPs
            _tilt_report(net, info, flops)
        if fortify.alloc_grid_round():
            _grid_round_state(env, net, cache[net], model, info, walk_target, flops)
        if info["kind"] in TILT_KINDS and info["tilt"].get("hold"):  # amendment A1: λ, the params held, the check
            _tilt_hold_report(net, cache[net], info)
    elif "grid" in cache[net]:
        _grid_watch(env, cache[net])
    return cache[net]


def action(env, legal, rates, device):
    """``SPECTRA_EVAL_POLICY=alloc``: one decision of the allocation-following walk."""
    identity = next((i for i, r in rates.items() if abs(float(r) - 1.0) < 1e-9), 0)
    legal_idx = [int(i) for i in legal.nonzero(as_tuple=False).flatten().tolist()]
    state = _state(env)
    flops = budget() == "flops"
    kept_now = float(env.flops_ratio() if flops else env.param_ratio())
    if kept_now > state["last_kept"] + 1e-9:
        state.update(idle=0, fallback=False)
    state["last_kept"] = kept_now
    if state["fallback"]:
        return torch.tensor([pick_strongest(rates, legal_idx, identity)], device=device)
    mwr = ModelWithRows(env.current_model)
    row = max(0, (env.row_idx or 1) - 1)
    groups = channel_groups.build_channel_groups(mwr.model) or []
    group = channel_groups.group_of(groups, mwr.all_layers[mwr.row_to_main_layer[row]])
    pick = identity
    if group is not None:
        first = {id(g): r for g, r in group_sensitivity.group_plan(mwr, groups)}
        target = state["widths"].get(first.get(id(group)))
        if target is not None:
            pick = choose(int(group.width), int(target), rates, legal_idx, identity)
    state["idle"] = 0 if pick != identity else state["idle"] + 1
    if state["idle"] >= state["n_rows"] and env.target_keep is not None and kept_now > float(env.target_keep):
        state["fallback"] = True
        utils.print_flush(f"[alloc] every group at its target with {'FLOPs' if flops else 'params'} x{kept_now:.3f} "
                          f"above the target x{float(env.target_keep):.3f}; strongest legal cut from here")
        pick = pick_strongest(rates, legal_idx, identity)
    return torch.tensor([pick], device=device)


def pick_strongest(rates, legal_idx, identity):
    cuts = [i for i in legal_idx if float(rates[i]) < 1.0 - 1e-9]
    return min(cuts, key=lambda i: float(rates[i])) if cuts else identity
