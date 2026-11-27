"""
NAP (``SPECTRA_EVAL_FT_NAP=1``, default off; needs ``SPECTRA_EVAL_FT_TRACE`` > 0): the NAP2-v2 logger of the registered
study NAPJ. :class:`NapLogger` rides TRACE's hook (:class:`src.ft_trace.FtTrace`, keyword ``nap``) along the final
fine-tune and extracts the feature families online; no model state is written. Family and feature ids (X1, P2, S5, ...)
are those of the NAP2-v2 feature catalogue, as fixed in science's build spec (26 Nov 2026).

Schedule (absolute optimizer steps)
    reads        ``READ_STEPS`` = 0 1 2 3 5 8 13 20 39 59 78 118 157 196 274, and the last step (from ``on_end``)
    checkpoints  ``CKPT_STEPS`` = 0 5 8 20 39 78 157 196 274, and the last step
    windows      the steps between consecutive checkpoints, (w_{k-1}, w_k]
    projection   every 3rd step (0, 3, 6, ...)

Records (``recorder.record``, as ``eval_ft_trace``)
    ``eval_ft_nap0``  once, at ``on_start``: network, label, step 0, seconds and the dicts of ``NAP0_KEYS``
        H5    feature-form kernel-ridge forecast (LOO accuracy and MSE, per side, P_tr -> P_ho fit), for the cut with
              its stale and its recalibrated BatchNorm statistics and for the origin
        Z3    Z3.1 mean squared activation and Z3.2 pair cosine per convolution (stale, recal); Z3.3 backward profile
              and Z3.4 input-output Jacobian spectrum (recal copy)
        F7    ``F7pre``: channels whose mean |output| is at most 1 % (and 2.5 %) of the layer's mean, pre-activation
        F2    effective rank, 99 % rank and participation ratio of each pooled convolution output and the penultimate
        H1    alignment of each pooled representation's linear kernel with the label kernel (P_tr)
        PDIV  Jensen-Shannon divergence and argmax agreement between the cut's and the origin's probe outputs
        Z4    grad norm, SNIP, GraSP, Fisher, SynFlow at the cut (4 minibatches of 64, recal copy)
        BN    stale against recalibrated statistics: accuracy difference on P_ho, flips, divergence, per-layer shift
    ``eval_ft_nap``   one per read: network, label, step, seconds and the dicts of ``NAP_KEYS``; every key is in every
        event, ``{}`` where the family has nothing at that step
        X     running summaries of the train loss (the reader computes X1-X6 from the series in the array file)
        P     P1 churn, P2 function-space step, P3 confidence, P4 calibration, P6 class dispersion, origin divergence;
              at step 0 ``P.stale`` and ``P.recal``
        S R Y V K J F0   at checkpoints only: weight-trajectory geometry (S1-S9), relative change (R3), movement from
              the cut (Y1-Y5), optimizer state (V1 V2 V5 V7 V8), gradient noise over the window (K1-K4 K7 K8, head-only
              K9 K10), the empirical Fisher of the window (``J4emp``, J15), NAP2's family 0 of weights and gradients
        E     E1 E2 E5 E6 E7, at the last step only
        H     H5 at the last step only (and its change from step 0)
    A value that is absent or not finite is written as None and named, with its reason, in the dict's ``missing`` list;
    no NaN or infinity is ever written as a number. Keys ending in ``_L`` hold the fixed-length per-layer form
    ``FORM16``: the value at 8 relative depths, then mean, sd, min, max, first, last, slope and depth of the maximum.
    An event is written when the next read starts (or at ``on_end``), so that the last step's event carries E and H;
    its log line is printed at once:

    [eval] FT_NAP <label> <net> step=<s> | probe acc ho <a> tr <a> | churn <c> | js0 <j> | <sec> s

Array file ``<run dir>/nap/<label>__<network basename>.npz``, written once at ``on_end`` (``meta`` is a JSON string):
the per-step series, the probe logits of every read (float16) with the probe's indices and labels, the per-layer
arrays of every checkpoint, the family-0 arrays and the projections. Arrays hold finite numbers only; where a value
is missing the array holds 0 and its ``<name>__ok`` mask holds 0.

Probe (the same images for every network and plan of a dataset; never TEST)
    P_ho  the first ``probe_n`` (500) of a class-stratified order of the val dataset, seeded by (dataset, "probe_ho")
    P_tr  the first 500 of a class-stratified order of the train dataset, seeded by (dataset, "probe_tr"); its first
          2,000 are the BatchNorm calibration set
    Both are read without augmentation. The fine-tune's augmentation never enters the stored images: the train loader
    either crops and flips on the GPU from the dataset's raw uint8 array (``utils.GpuCropFlipLoader``) or carries
    RandomCrop+Flip in the dataset's transform (``utils.final_ft_train_loader``). P_tr is therefore read item by item
    from a shallow copy of the train dataset whose transform is the val dataset's own (ToTensor + Normalize, the
    normalisation val is read with); a val transform that augments, or a dataset without one, stops NAP with a line.
    The probe is built under a save and restore of every random state and its draws use ``numpy.random.default_rng``
    with zlib.crc32 seeds, so the global generators never move.

Decisions and deviations from the catalogue
    - The per-layer unit is each ``torch.nn.Conv2d`` in module order (its output, pre-activation, for representation
      features; its weight for weight features) plus the final ``torch.nn.Linear`` (its input, the penultimate vector,
      and its weight).
    - The gradient the hook sees is clipped; every gradient feature uses the pre-clip gradient, recovered as
      ``p.grad * max(1, grad_norm + 1e-6)`` (``clip_grad_norm_``'s own factor, max_norm 1).
    - K and J come from the training steps' own minibatch gradients in a window (a moving parameter vector, train-mode
      BatchNorm, augmented batches); the window length is recorded. K1 is the per-layer and overall ratio
      ``|mean g|^2 / (V + 1e-12)``, ``V = n/(n-1) (mean |g|^2 - |mean g|^2)``. K3 has no per-example gradients: its
      minibatch form is reported (``K3mb``). The pairwise quantities of K3mb, K4, K7 and K8 are read from a 4,096-dim
      sketch of each step's gradient. J4 is the empirical Fisher of those gradients (``J4emp``).
    - V2 is the cosine between the previous update and the gradient, which for SGD with momentum is the cosine between
      the momentum buffer and the gradient; the hook has no optimizer. V1 is the mean update norm over the window
      divided by the checkpoint's weight norm.
    - S5's projection is a CountSketch to 4,096 dimensions with seeded signs and a seeded permutation of 64-coordinate
      blocks (a gather, a product and a sum: deterministic on CUDA, where ``index_add_`` would sort); it is applied to
      ``theta_t - theta_0``. S8's weight mask is over all trainable parameters.
    - Z4.1-Z4.3, Z4.5 and Z3.3 are computed on each of the 4 minibatches and averaged (the SD is beside the mean);
      GraSP's vector is the minibatch's own gradient. Z4.5 follows zero-cost-nas, ``0.5 mean_n (sum_hw a g)^2`` per
      channel of each convolution's own output, summed plainly and weighted by the channel's weight count. Z4.4
      (SynFlow) opens every clipped ReLU (``ReLU6``) of its copy to a ReLU, beside setting BatchNorm to the
      identity; the count is recorded. Z3.4 is exact: the Jacobian of 16 probe inputs from class-replicated
      backward passes, its singular values from the C x C Gram matrix (the ratio is missing below rank 10).
    - Y4 uses exact spectra and is computed for layers whose smaller matrix side is at most 256 in the origin network
      (the same layers for every plan of a network). Y5's linear probe is fitted once on the recalibrated cut's
      penultimate features of P_tr (L-BFGS, L2 1e-3). F2 reads P_256.
    - The per-step series holds loss, lr, pre-clip gradient norm, update norm and clip factor. The minibatch accuracy,
      max-probability and margin are absent: TRACE's hook does not pass the logits.
    - Rates per log-step use log(1 + step). Probabilities, entropies and divergences are in nats.

Training is unchanged by the logger. At ``on_start`` every forward, hook and gradient runs on a deep copy of the cut
network and of the origin; the live model is only read (its parameters). Later reads run the live model in eval mode
under ``torch.no_grad()``, register their forward hooks inside the read and remove them before it returns, restore
each module's ``training`` attribute (set directly) and every random state. ``on_step`` only reads ``p.grad`` and the
parameters into the logger's own device buffers, with no host sync; they are copied to the host at reads. A failure
inside the logger prints one line, records an issue and stops the logger only.
"""

import copy
import json
import math
import os
import random
import time
import zipfile
import zlib

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

import src.utils as utils

READ_STEPS = (0, 1, 2, 3, 5, 8, 13, 20, 39, 59, 78, 118, 157, 196, 274)
CKPT_STEPS = (0, 5, 8, 20, 39, 78, 157, 196, 274)
PROJ_EVERY = 3
SKETCH_DIM = 4096
SKETCH_BLOCK = 64
PROBE_N = 500
CALIB_N = 2000
PROBE_CHUNK = 500
F2_ROWS = 128  # per side: P_256
Z4_BATCHES, Z4_BATCH = 4, 64
JAC_INPUTS = 16
JAC_REPLICAS = 256
Y4_MAX_DIM = 256
Y4_RANK_TOL = 1e-6
Y5_L2 = 1e-3
K1_EPS = 1e-12
F0_CHUNK = 4 * 1024 * 1024
CKA_CHUNK_BYTES = 48 * 1024 * 1024
NAP_KEYS = ("X", "P", "S", "R", "Y", "V", "K", "J", "F0", "E", "H")
NAP0_KEYS = ("H5", "Z3", "F7", "F2", "H1", "PDIV", "Z4", "BN")
FORM16 = ("d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "mean", "sd", "min", "max", "first", "last", "slope",
          "argmax_depth")
F0_STATS = ("mean", "var", "median", "std", "max", "min", "cov", "skew", "kurt", "p0", "p25", "p75", "p50", "p100",
            "l1", "l2")
F0_GROUPS = ("layer", "filters_mean", "filters_sd", "filters_q10", "filters_q50", "filters_q90")
# how each family-0 column scales when its tensor is multiplied by a positive factor (1, 2 or 0 powers of it)
_F0_POWER = np.array([1, 2, 1, 1, 1, 1, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
_F0_EVENT = (("mean", 0), ("std", 3), ("skew", 7), ("kurt", 8), ("l2", 15), ("fl2_mean", 16 + 15),
             ("fl2_sd", 32 + 15), ("fkurt_mean", 16 + 8))
STEP_COLS = ("loss", "lr", "gnorm_pre", "upd_norm", "clip_scale")
_DEPTHS = np.linspace(0.0, 1.0, 8)
_PROBE_CACHE = {}
_ORIGIN_CACHE = {}


def seed_of(*parts):
    """zlib.crc32 of the joined parts: (dataset, feature) for the probe, (dataset, network, feature) otherwise."""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0xFFFFFFFF


class _RngGuard:
    """Save every random state on entry and put it back on exit (torch CPU and CUDA, ``random``, numpy)."""

    def __enter__(self):
        self.states = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                       random.getstate(), np.random.get_state())
        return self

    def __exit__(self, exc_type, exc, tb):
        torch_state, cuda_state, py_state, np_state = self.states
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        random.setstate(py_state)
        np.random.set_state(np_state)
        return False


# --- records: finite numbers only ---------------------------------------------------------------------------------

def _num(value):
    """A finite float rounded to 6 significant digits, else None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float("%.6g" % number)


def fixed_form(values):
    """
    The fixed-length per-layer form (``FORM16``): the values at 8 relative depths by linear interpolation, then mean,
    sd, min, max, first, last, the slope of a linear fit against relative depth and the relative depth of the maximum.
    Returns ``(list of 16 floats or None, number of non-finite layers)``; non-finite layers are left out.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    count = int(v.size)
    if count == 0:
        return None, 0
    depth = np.linspace(0.0, 1.0, count) if count > 1 else np.zeros(1)
    ok = np.isfinite(v)
    bad = int(count - ok.sum())
    if not ok.any():
        return None, bad
    d, x = depth[ok], v[ok]
    interp = np.interp(_DEPTHS, d, x)
    slope = argmax_depth = float("nan")  # missing below two finite layers: a slope and a depth of the maximum need two depths
    if x.size >= 2 and float(np.ptp(d)) > 0:
        dc = d - d.mean()
        slope = float((dc * (x - x.mean())).sum() / (dc * dc).sum())
        argmax_depth = d[int(np.argmax(x))]
    out = list(interp) + [x.mean(), x.std(), x.min(), x.max(), x[0], x[-1], slope, argmax_depth]
    return [float(o) for o in out], bad


class _Fam(dict):
    """One family's record. ``put`` and ``form`` store finite numbers; anything else is None and named in ``missing``."""

    def put(self, name, value, why=None):
        number = _num(value)
        self[name] = number
        if number is None:  # ``why`` explains an absent value (None); a value that is not finite is named as such
            self.miss(name, (why or "not available") if value is None else "non-finite")
        return number

    def form(self, name, values, why=None):
        if values is None:
            self[name] = None
            self.miss(name, why or "not available")
            return
        count = int(np.asarray(values).size)
        form, bad = fixed_form(values)
        if form is None:
            self[name] = None
            self.miss(name, why or ("no layer" if count == 0 else "no finite layer"))
            return
        self[name] = [_num(x) for x in form]
        if bad:
            self.miss(name, "%d of %d layers non-finite, left out of the form" % (bad, count))

    def miss(self, name, why):
        self.setdefault("missing", []).append([str(name), str(why)])


def _clean(obj, path, bad):
    """JSON-safe copy of ``obj``; a non-finite number becomes None and its path is appended to ``bad``."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        number = float(obj)
        if math.isfinite(number):
            return number
        bad.append(path or "/")
        return None
    if isinstance(obj, dict):
        return {str(k): _clean(v, path + "/" + str(k), bad) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, "%s[%d]" % (path, i), bad) for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist(), path, bad)
    return str(obj)


# --- host numerics (numpy, float64) ---------------------------------------------------------------------------------

def _log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _moments(x):
    """mean, sd (ddof 0), skewness and excess kurtosis (both biased; 0 for a constant)."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean()
    c = x - mean
    m2 = (c * c).mean()
    if not m2 > 0:
        return mean, 0.0, 0.0, 0.0
    return mean, math.sqrt(m2), float((c ** 3).mean() / m2 ** 1.5), float((c ** 4).mean() / m2 ** 2 - 3.0)


class _Side:
    """One probe side at one read: the per-example quantities the P, E and K features are built from."""

    def __init__(self, logits, labels):
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        n, c = z.shape
        idx = np.arange(n)
        self.z, self.y, self.n, self.c = z, y, n, c
        self.logp = _log_softmax(z)
        self.p = np.exp(self.logp)
        self.pred = z.argmax(axis=1)
        self.correct = self.pred == y
        self.py = self.p[idx, y]
        self.ce = -self.logp[idx, y]
        if c > 1:
            rest = z.copy()
            rest[idx, y] = -np.inf
            self.margin = z[idx, y] - rest.max(axis=1)
        else:
            self.margin = np.full(n, np.nan)


def js_divergence(logp, logq):
    """Per-example Jensen-Shannon divergence (nats) between two rows of log-probabilities."""
    p, q = np.exp(logp), np.exp(logq)
    logm = np.log(np.maximum(0.5 * (p + q), 1e-300))
    js = 0.5 * ((p * (logp - logm)).sum(axis=1) + (q * (logq - logm)).sum(axis=1))
    return np.where(np.isfinite(js), np.maximum(js, 0.0), js)


def temperature(logits, labels, iters=60):
    """
    P4's temperature: ``argmin_T NLL(softmax(logits / T))``, T held in [0.01, 100]. The NLL is convex in b = 1/T, so
    its derivative has one root: Newton steps from b = 1, kept inside a bracket (else a bisection in log b).
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    idx = np.arange(z.shape[0])
    zc = z - z.max(axis=1, keepdims=True)
    zy = zc[idx, y]

    def slope(b):
        a = b * zc
        p = np.exp(a)
        p /= p.sum(axis=1, keepdims=True)
        ez = (p * zc).sum(axis=1)
        return float((ez - zy).mean()), float(((p * zc * zc).sum(axis=1) - ez * ez).mean())

    lo, hi = 0.01, 100.0
    g_hi, _ = slope(hi)
    if not math.isfinite(g_hi):
        return float("nan")
    if g_hi <= 0.0:
        return 1.0 / hi  # the loss still falls at the sharpest temperature allowed
    g_lo, _ = slope(lo)
    if g_lo >= 0.0:
        return 1.0 / lo
    b = 1.0
    for _ in range(iters):
        g, h = slope(b)
        if abs(g) < 1e-13:
            break
        if g < 0.0:
            lo = b
        else:
            hi = b
        nb = b - g / h if h > 0.0 else 0.0
        if math.isfinite(nb) and abs(nb - b) <= 1e-12 * b:
            b = nb
            break
        if not lo < nb < hi:  # a NaN step trips this too
            nb = math.sqrt(lo * hi)
        b = nb
    return 1.0 / b


def _p_static(ho, tr, origin=None):
    """P3, P4, P6 and the origin divergence at one read (``origin``: the origin's ``(_Side ho, _Side tr)``)."""
    f = _Fam()
    with np.errstate(all="ignore"):
        for tag, s in (("ho", ho), ("tr", tr)):
            f.put("acc_" + tag, s.correct.mean())
            f.put("loss_" + tag, s.ce.mean())
            f.put("maxprob_" + tag, s.p.max(axis=1).mean())
            f.put("entropy_" + tag, -(s.p * s.logp).sum(axis=1).mean())
            q = np.percentile(s.margin, [10, 50, 90]) if np.isfinite(s.margin).all() else [np.nan] * 3
            for name, value in zip(("margin_p10_", "margin_p50_", "margin_p90_"), q):
                f.put(name + tag, value)
            f.put("lnorm_" + tag, np.sqrt((s.z * s.z).sum(axis=1)).mean())
        f.put("gap_acc", (f["acc_tr"] - f["acc_ho"]) if None not in (f["acc_tr"], f["acc_ho"]) else None)
        f.put("gap_loss", (f["loss_tr"] - f["loss_ho"]) if None not in (f["loss_tr"], f["loss_ho"]) else None)
        for name in ("maxprob", "entropy", "margin_p10", "margin_p50", "margin_p90", "lnorm"):
            a, b = f[name + "_tr"], f[name + "_ho"]
            f.put("d_" + name, (a - b) if None not in (a, b) else None)
        # P4 on P_ho: 15 equal-width confidence bins, Brier (not divided by C), temperature
        conf = ho.p.max(axis=1)
        bins = np.minimum((conf * 15).astype(np.int64), 14)
        ece = 0.0
        for b in range(15):
            sel = bins == b
            if sel.any():
                ece += sel.mean() * abs(ho.correct[sel].mean() - conf[sel].mean())
        f.put("ece_ho", ece if np.isfinite(conf).all() else np.nan)
        onehot = np.zeros_like(ho.p)
        onehot[np.arange(ho.n), ho.y] = 1.0
        f.put("brier_ho", ((ho.p - onehot) ** 2).sum(axis=1).mean())
        f.put("temp_ho", temperature(ho.z, ho.y) if np.isfinite(ho.z).all() else np.nan)
        # P6 on P_ho
        count = np.bincount(ho.y, minlength=ho.c).astype(np.float64)
        hit = np.bincount(ho.y, weights=ho.correct.astype(np.float64), minlength=ho.c)
        present = count > 0
        class_acc = hit[present] / count[present]
        f.put("cls_sd_ho", class_acc.std())
        f.put("cls_worst_ho", class_acc.min())
        hist = np.bincount(ho.pred, minlength=ho.c).astype(np.float64) / max(ho.n, 1)
        nz = hist[hist > 0]
        f.put("pred_entropy_ho", -(nz * np.log(nz)).sum())
        confusion = np.zeros((ho.c, ho.c))
        np.add.at(confusion, (ho.y, ho.pred), 1.0)
        rows = confusion[present] / count[present][:, None]
        f.put("conf_row_entropy_ho", (-(rows * np.log(np.where(rows > 0, rows, 1.0))).sum(axis=1)).mean())
        if origin is not None:
            for tag, s, o in (("ho", ho, origin[0]), ("tr", tr, origin[1])):
                f.put("js_origin_" + tag, js_divergence(s.logp, o.logp).mean())
                f.put("agree_origin_" + tag, (s.pred == o.pred).mean())
        else:
            for tag in ("ho", "tr"):
                f.put("js_origin_" + tag, None, "origin not read")
                f.put("agree_origin_" + tag, None, "origin not read")
    return f


def e_history(steps, correct, margin, py):
    """
    E1, E2 and E6 on one side from the reads so far: ``correct`` [reads, n] bool, ``margin`` and ``py`` [reads, n],
    ``steps`` [reads]. Read 0 (the cut) is an event; the fraction correct there and E1 on the inputs wrong there
    are reported beside it. An input is learned at the first read where it is correct at that read and the next.
    """
    f = _Fam()
    steps = np.asarray(steps, dtype=np.float64)
    reads = correct.shape[0]
    last = float(steps[-1])
    with np.errstate(all="ignore"):
        f.put("correct0", correct[0].mean())
        if reads >= 2 and last > 0:
            both = correct[:-1] & correct[1:]
            learned = both.any(axis=0)
            tau = np.where(learned, steps[both.argmax(axis=0)], np.nan) / last
            f.put("E1_learned", learned.mean())
            pct = np.percentile(tau[learned], [10, 25, 50, 75, 90]) if learned.any() else [None] * 5
            for name, value in zip(("p10", "p25", "p50", "p75", "p90"), pct):
                f.put("E1_tau_" + name, value, "no input learned")
            wrong0 = ~correct[0]
            relearned = learned & wrong0
            f.put("E1_relearned", (relearned.sum() / wrong0.sum()) if wrong0.any() else None, "no input wrong at step 0")
            pct = np.percentile(tau[relearned], [10, 25, 50, 75, 90]) if relearned.any() else [None] * 5
            for name, value in zip(("p10", "p25", "p50", "p75", "p90"), pct):
                f.put("E1_retau_" + name, value, "no input relearned")
            forget = (correct[:-1] & ~correct[1:]).sum(axis=0)
            f.put("E2_mean", forget.mean())
            f.put("E2_any", (forget >= 1).mean())
            f.put("E2_p90", np.percentile(forget, 90))
            f.put("E2_never", (learned & (forget == 0)).mean())
        else:
            for name in ("E1_learned", "E2_mean", "E2_any", "E2_p90", "E2_never"):
                f.put(name, None, "fewer than two reads")
        aum = margin.mean(axis=0)
        confidence = py.mean(axis=0)
        variability = py.std(axis=0)
        f.put("E6_aum_mean", aum.mean())
        f.put("E6_aum_p10", np.percentile(aum, 10) if np.isfinite(aum).all() else np.nan)
        f.put("E6_aum_neg", (aum < 0).mean() if np.isfinite(aum).all() else np.nan)
        f.put("E6_conf_mean", confidence.mean())
        f.put("E6_conf_low", (confidence < 0.5).mean() if np.isfinite(confidence).all() else np.nan)
        f.put("E6_var_mean", variability.mean())
        f.put("E6_var_p67", np.percentile(variability, 200.0 / 3.0) if np.isfinite(variability).all() else np.nan)
    return f


def gmm2(x, iters=100):
    """
    E7: a two-component 1-D Gaussian mixture by EM, started at the quartiles. Returns ``(weight of the high-mean
    component, |mu2 - mu1| / sqrt((var1 + var2) / 2))`` or None when the data are degenerate.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4 or not np.isfinite(x).all() or not x.var() > 0:
        return None
    lo, hi = np.percentile(x, [25, 75])
    if not hi > lo:
        lo, hi = x.min(), x.max()
    mu = np.array([lo, hi])
    var = np.full(2, x.var())
    w = np.array([0.5, 0.5])
    floor = 1e-6 * x.var() + 1e-300
    with np.errstate(all="ignore"):
        for _ in range(iters):
            logp = -0.5 * ((x[:, None] - mu) ** 2 / var + np.log(2.0 * np.pi * var)) + np.log(w)
            r = np.exp(logp - logp.max(axis=1, keepdims=True))
            r /= r.sum(axis=1, keepdims=True)
            nk = r.sum(axis=0) + 1e-300
            mu_new = (r * x[:, None]).sum(axis=0) / nk
            var_new = np.maximum((r * (x[:, None] - mu_new) ** 2).sum(axis=0) / nk, floor)
            w_new = np.maximum(nk / nk.sum(), 1e-300)
            moved = np.abs(mu_new - mu).max() + np.abs(var_new - var).max()
            mu, var, w = mu_new, var_new, w_new
            if moved <= 1e-12 * (1.0 + np.abs(mu).max()):
                break
    if not (np.isfinite(mu).all() and np.isfinite(var).all() and np.isfinite(w).all()):
        return None
    high = int(np.argmax(mu))
    return float(w[high]), float(abs(mu[1] - mu[0]) / math.sqrt(0.5 * (var[0] + var[1])))


def k1_snr(sum_norm2, step_norm2, n, eps=K1_EPS):
    """
    K1 from a window's accumulators: ``|mean g|^2 / (V + eps)`` with ``V = n/(n-1) (mean |g|^2 - |mean g|^2)``.
    ``sum_norm2`` is ``|sum_t g_t|^2`` and ``step_norm2`` is ``sum_t |g_t|^2`` (scalars or per-layer arrays).
    """
    n = float(n)
    mean2 = np.asarray(sum_norm2, dtype=np.float64) / (n * n)
    spread = np.maximum(np.asarray(step_norm2, dtype=np.float64) / n - mean2, 0.0) * (n / (n - 1.0))
    return mean2 / (spread + eps)


def head_gradient(tr, pen, bias, weight_norm):
    """
    The closed-form head gradients on P_tr: ``e_i = p_i - y_i``, ``grad_W l_i = e_i h_i^T``. Returns EL2N, GraNd-head
    (E5), ``gamma_i = cos(-grad_W l_i, W)`` (K9) and K10's three numbers.
    """
    e = tr.p.copy()
    e[np.arange(tr.n), tr.y] -= 1.0
    e_norm = np.sqrt((e * e).sum(axis=1))
    h = np.asarray(pen, dtype=np.float64)
    h_norm = np.sqrt((h * h).sum(axis=1))
    wh = tr.z - (np.asarray(bias, dtype=np.float64)[None, :] if bias is not None else 0.0)
    with np.errstate(all="ignore"):
        gamma = -(e * wh).sum(axis=1) / (e_norm * h_norm * weight_norm)
        mean_grad = math.sqrt(float(((e.T @ h / max(tr.n, 1)) ** 2).sum()))
        h_fro = math.sqrt(float((h * h).sum()))
        k10 = (mean_grad, mean_grad / weight_norm if weight_norm > 0 else np.nan,
               mean_grad / h_fro if h_fro > 0 else np.nan)
    return e_norm, e_norm * h_norm, gamma, k10


def _write_npz(path, arrays, stored=()):
    """An ``.npz`` (``numpy.load`` reads it): deflated members, except the noise-like ones in ``stored``. Atomic."""
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if name in stored else zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.asanyarray(value), allow_pickle=False)
    os.replace(tmp, path)
    return os.path.getsize(path)


# --- device numerics (torch, float64) ---------------------------------------------------------------------------------

def _norms(tensors):
    """The 2-norm of every tensor of a list, as one 1-D tensor."""
    if not tensors:
        return None
    foreach = getattr(torch, "_foreach_norm", None)  # one fused call on CUDA, as clip_grad_norm_ itself uses
    if foreach is not None:
        try:
            return torch.stack(list(foreach(tensors)))
        except (RuntimeError, TypeError):
            pass
    return torch.stack([torch.linalg.vector_norm(t) for t in tensors])


def _eigh(matrix, vectors=False):
    """Eigenvalues (ascending) of a symmetric matrix on the host, with the eigenvectors on request; None when the
    matrix is empty or not finite, or the solver fails."""
    if isinstance(matrix, np.ndarray):
        matrix = np.ascontiguousarray(matrix)
    m = torch.as_tensor(matrix, dtype=torch.float64).cpu()
    if m.numel() == 0 or not bool(torch.isfinite(m).all()):
        return None
    try:
        return torch.linalg.eigh(m) if vectors else torch.linalg.eigvalsh(m)
    except RuntimeError:
        return None


def _spd_inverse(matrix):
    chol, info = torch.linalg.cholesky_ex(matrix)
    if int(info) == 0:
        return torch.cholesky_inverse(chol)
    return torch.linalg.inv(matrix)


def h5_forecast(features, labels, n_classes, n_ho=None, ridge=1e-3, return_loo=False):
    """
    H5, feature form: kernel ridge regression of the one-hot labels (centred) on the linear kernel of the standardised
    features, ``lam = ridge * tr(K) / n``, with the closed-form leave-one-out prediction
    ``yhat_i = y_i - [(K + lam I)^-1 Y]_i / [(K + lam I)^-1]_ii``. ``features`` [n, d], ``labels`` [n] (class indices),
    rows ``[:n_ho]`` the held-out side and the rest the train side. Returns a dict (LOO accuracy; LOO squared error
    summed over classes and averaged over inputs; each side's LOO accuracy; the accuracy on the held-out side of the
    fit on the train side), or None when the kernel is empty. With ``return_loo`` also the [n, C] LOO predictions.
    """
    h = features.detach().double()
    n = int(h.shape[0])
    labels = labels.to(h.device).long()
    mean = h.mean(dim=0, keepdim=True)
    sd = h.std(dim=0, unbiased=False, keepdim=True)
    live = sd > 1e-12 * sd.max().clamp_min(1e-300)
    z = torch.where(live, (h - mean) / torch.where(live, sd, torch.ones_like(sd)), torch.zeros_like(h))
    kernel = z @ z.t()
    trace = float(kernel.diagonal().sum())
    if n < 3 or not (math.isfinite(trace) and trace > 0):
        return (None, None) if return_loo else None
    eye = torch.eye(n, dtype=h.dtype, device=h.device)
    onehot = torch.nn.functional.one_hot(labels, int(n_classes)).to(h.dtype)
    ybar = onehot.mean(dim=0, keepdim=True)
    yc = onehot - ybar
    lam = ridge * trace / n
    try:
        inverse = _spd_inverse(kernel + lam * eye)
    except RuntimeError:
        return (None, None) if return_loo else None
    residual = (inverse @ yc) / inverse.diagonal()[:, None]
    loo = yc - residual + ybar
    hit = (loo.argmax(dim=1) == labels).double()
    out = {"loo_acc": float(hit.mean()), "loo_mse": float((residual * residual).sum(dim=1).mean()),
           "n": n, "d": int(h.shape[1]), "lam": float(lam)}
    if n_ho is not None and 0 < int(n_ho) < n:
        n_ho = int(n_ho)
        out["loo_acc_ho"] = float(hit[:n_ho].mean())
        out["loo_acc_tr"] = float(hit[n_ho:].mean())
        zt, zh = z[n_ho:], z[:n_ho]
        ktt = zt @ zt.t()
        tr_trace = float(ktt.diagonal().sum())
        if tr_trace > 0:
            n_tr = n - n_ho
            yt = onehot[n_ho:]
            ytbar = yt.mean(dim=0, keepdim=True)
            try:
                beta = _spd_inverse(ktt + (ridge * tr_trace / n_tr) * torch.eye(n_tr, dtype=h.dtype, device=h.device)) @ (yt - ytbar)
                fit = (zh @ zt.t()) @ beta + ytbar
                out["fit_acc_tr2ho"] = float((fit.argmax(dim=1) == labels[:n_ho]).double().mean())
            except RuntimeError:
                pass
    return (out, loo) if return_loo else out


def _hsic_parts(x):
    """``x`` [g, n, d] float64: the zero-diagonal linear kernel of the centred rows, its row sums and its total."""
    x = x - x.mean(dim=1, keepdim=True)
    kernel = x @ x.transpose(1, 2)
    kernel.diagonal(dim1=1, dim2=2).zero_()
    rows = kernel.sum(dim=2)
    return kernel, rows, rows.sum(dim=1)


def _hsic(a, b, n):
    t1 = (a[0] * b[0]).sum(dim=(1, 2))
    t2 = a[2] * b[2] / ((n - 1) * (n - 2))
    t3 = (a[1] * b[1]).sum(dim=1) * (2.0 / (n - 2))
    return (t1 + t2 - t3) / (n * (n - 3))


def cka_unbiased(x, y):
    """Linear CKA with the unbiased HSIC estimator; ``x`` [n, d1] and ``y`` [n, d2], or batched [g, n, d]."""
    single = x.dim() == 2
    if single:
        x, y = x[None], y[None]
    n = int(x.shape[1])
    if n < 4:
        out = torch.full((x.shape[0],), float("nan"), dtype=torch.float64, device=x.device)
    else:
        a, b = _hsic_parts(x.double()), _hsic_parts(y.double())
        out = _hsic(a, b, n) / (_hsic(a, a, n) * _hsic(b, b, n)).sqrt()
    return out[0] if single else out


def row_stats(v, lengths):
    """
    NAP2's 16 statistics (``F0_STATS``) of every row of ``v`` ([rows, width], zero-padded past each row's length), and
    the power sums a pooled statistic needs: ``([rows, 16], [rows, 4])`` (sum, sum of squares, of cubes, of 4th powers).
    Constant rows get zero skewness and kurtosis.
    """
    v = v.double()
    n = lengths.to(v.dtype)
    mask = torch.arange(v.shape[1], device=v.device)[None, :] < lengths[:, None]
    total = v.sum(dim=1)
    mean = total / n
    c = (v - mean[:, None]) * mask
    c2 = c * c
    ss = c2.sum(dim=1)
    m2 = ss / n
    flat = m2 <= 0
    safe = torch.where(flat, torch.ones_like(m2), m2)
    zero = torch.zeros_like(m2)
    skew = torch.where(flat, zero, (c2 * c).sum(dim=1) / n / safe.pow(1.5))
    kurt = torch.where(flat, zero, (c2 * c2).sum(dim=1) / n / safe.pow(2) - 3.0)
    cov = ss / (n - 1).clamp_min(1)
    ordered = v.masked_fill(~mask, float("inf")).sort(dim=1).values

    def level(q):
        pos = q * (n - 1)
        lo = pos.floor()
        a = ordered.gather(1, lo.long()[:, None])[:, 0]
        b = ordered.gather(1, pos.ceil().long()[:, None])[:, 0]
        return a + (b - a) * (pos - lo)

    p0, p100 = level(0.0), level(1.0)
    p25, p50, p75 = level(0.25), level(0.5), level(0.75)
    sq = v * v
    sum2 = sq.sum(dim=1)
    stats = torch.stack([mean, m2, p50, m2.sqrt(), p100, p0, cov, skew, kurt, p0, p25, p75, p50, p100,
                         v.abs().sum(dim=1), sum2.sqrt()], dim=1)
    sums = torch.stack([total, sum2, (sq * v).sum(dim=1), (sq * sq).sum(dim=1)], dim=1)
    return stats, sums


def spectrum_ranks(gram):
    """
    F2 from the Gram matrix of a centred representation (either side): effective rank ``exp(H(sigma / sum sigma))``,
    the smallest k whose leading singular values hold 99 % of their sum, and the participation ratio of the
    eigenvalues. ``(erank, srank, pr)`` or None for a zero matrix.
    """
    lam = _eigh(gram)
    if lam is None:
        return None
    lam = lam.clamp_min(0).flip(0)
    sigma = lam.sqrt()
    total = float(sigma.sum())
    if not (math.isfinite(total) and total > 0):
        return None
    q = sigma / total
    nz = q[q > 0]
    erank = math.exp(float(-(nz * nz.log()).sum()))
    srank = int((q.cumsum(0) < 0.99 - 1e-12).sum()) + 1
    return erank, min(srank, int(q.numel())), float(lam.sum() ** 2 / (lam * lam).sum())


def fit_linear_probe(features, labels, n_classes, l2=Y5_L2, max_iter=30):
    """Y5's reference head: multinomial logistic regression (L-BFGS from zero, L2 on the weight). Returns W, b."""
    x = features.detach().double()
    labels = labels.to(x.device).long()
    weight = torch.zeros(int(n_classes), x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    bias = torch.zeros(int(n_classes), dtype=x.dtype, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=int(max_iter), history_size=10,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ weight.t() + bias, labels) + 0.5 * l2 * (weight * weight).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
    return weight.detach(), bias.detach()


# --- the probe ----------------------------------------------------------------------------------------------------------

def _resolve(dataset):
    """Unwrap ``Subset`` chains: ``(base dataset, base indices or None)``."""
    indices = None
    while isinstance(dataset, torch.utils.data.Subset):
        outer = list(dataset.indices)
        indices = outer if indices is None else [outer[i] for i in indices]
        dataset = dataset.dataset
    return dataset, indices


def dataset_key(dataset):
    """A dataset's canonical name from its class (``CIFAR100`` -> ``cifar-100``): the probe's seed key."""
    base, _ = _resolve(dataset)
    return utils.canonical_dataset_name(type(base).__name__)


def _labels_of(base, indices):
    if isinstance(base, torch.utils.data.TensorDataset):
        values = base.tensors[1]
        if values.dim() > 1:
            values = values.argmax(dim=1)
        values = values.detach().cpu().numpy()
    else:
        values = None
        for attr in ("targets", "labels"):
            if getattr(base, attr, None) is not None:
                values = getattr(base, attr)
                break
        if values is None:
            raise ValueError("%s exposes neither targets nor labels" % type(base).__name__)
        values = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    values = np.asarray(values).astype(np.int64).reshape(-1)
    return values if indices is None else values[np.asarray(indices, dtype=np.int64)]


def _augments(transform):
    """The names of a transform's random steps (empty: it does not augment)."""
    steps = getattr(transform, "transforms", None)
    steps = list(steps) if steps is not None else ([transform] if transform is not None else [])
    marks = ("Random", "AutoAugment", "RandAugment", "TrivialAugment", "AugMix", "ColorJitter")
    return [type(s).__name__ for s in steps if any(m in type(s).__name__ for m in marks)]


def _images(base, base_indices, transform=None, swap=False):
    """The images at ``base_indices`` as one float tensor; ``swap`` reads through ``transform`` instead of the base's."""
    if isinstance(base, torch.utils.data.TensorDataset):
        return base.tensors[0][torch.as_tensor(np.asarray(base_indices, dtype=np.int64))].detach().float().cpu()
    view = utils._with_transform(base, transform) if swap else base
    items = [view[int(i)][0] for i in base_indices]
    if not items or not all(torch.is_tensor(item) for item in items):
        raise ValueError("%s items are not tensors" % type(base).__name__)
    return torch.stack(items).float()


def stratified_order(labels, seed, limit=None):
    """
    A class-stratified order of ``range(len(labels))``: round r takes the r-th image of a per-class shuffle from every
    class that still has one, the classes in a fresh random order, so every prefix is as balanced as it can be.
    Draws only from ``numpy.random.default_rng(seed)``. Stops after ``limit`` indices.
    """
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(int(seed))
    classes = np.unique(labels)
    per_class = [rng.permutation(np.flatnonzero(labels == c)) for c in classes]
    want = labels.size if limit is None else min(int(limit), labels.size)
    order, level = [], 0
    while len(order) < want:
        ready = [i for i, members in enumerate(per_class) if members.size > level]
        if not ready:
            break
        for i in rng.permutation(len(ready)):
            order.append(int(per_class[ready[i]][level]))
        level += 1
    return np.asarray(order[:want], dtype=np.int64)


def build_probe(val_dataset, train_dataset, dataset=None, probe_n=PROBE_N, calib_n=CALIB_N, use_cache=True):
    """
    The fixed probe of a dataset (module docstring): ``x_ho`` / ``y_ho`` / ``idx_ho`` (the first ``probe_n`` of the
    val order, positions in ``val_dataset``) and ``x_cal`` / ``y_cal`` / ``idx_tr`` (the first ``calib_n`` of the
    train order; its first ``probe_n`` are P_tr). Images are float CPU tensors, read without augmentation. No global
    random state moves. Cached per (val dataset, train dataset) object for the process.
    """
    name = dataset or dataset_key(val_dataset)
    key = (id(val_dataset), id(train_dataset), str(name), int(probe_n), int(calib_n))
    hit = _PROBE_CACHE.get(key) if use_cache else None
    if hit is not None and hit["_pins"][0] is val_dataset and hit["_pins"][1] is train_dataset:
        return hit
    with _RngGuard():
        val_base, val_idx = _resolve(val_dataset)
        tr_base, tr_idx = _resolve(train_dataset)
        tensor_val = isinstance(val_base, torch.utils.data.TensorDataset)
        tensor_tr = isinstance(tr_base, torch.utils.data.TensorDataset)
        eval_transform = None
        if not tensor_val:
            if not hasattr(val_base, "transform"):
                raise ValueError("the val dataset (%s) has no transform to read un-augmented images with"
                                 % type(val_base).__name__)
            eval_transform = val_base.transform
            random_steps = _augments(eval_transform)
            if random_steps:
                raise ValueError("the val transform augments (%s)" % ", ".join(random_steps))
        if not tensor_tr and (tensor_val or not hasattr(tr_base, "transform")):
            raise ValueError("no un-augmented path for the train dataset (%s)" % type(tr_base).__name__)
        y_val, y_tr = _labels_of(val_base, val_idx), _labels_of(tr_base, tr_idx)
        order_ho = stratified_order(y_val, seed_of(name, "probe_ho"), limit=probe_n)
        order_tr = stratified_order(y_tr, seed_of(name, "probe_tr"), limit=calib_n)
        base_ho = order_ho if val_idx is None else np.asarray(val_idx, dtype=np.int64)[order_ho]
        base_tr = order_tr if tr_idx is None else np.asarray(tr_idx, dtype=np.int64)[order_tr]
        probe = {
            "dataset": str(name),
            "x_ho": _images(val_base, base_ho), "y_ho": y_val[order_ho], "idx_ho": order_ho, "base_ho": base_ho,
            "x_cal": _images(tr_base, base_tr, eval_transform, swap=not tensor_tr), "y_cal": y_tr[order_tr],
            "idx_tr": order_tr, "base_tr": base_tr,
            "how_tr": "tensors" if tensor_tr else "train items through the val dataset's transform",
            "_pins": (val_dataset, train_dataset),
        }
    if use_cache:
        while len(_PROBE_CACHE) >= 4:
            _PROBE_CACHE.pop(next(iter(_PROBE_CACHE)))
        _PROBE_CACHE[key] = probe
    return probe


# --- the logger ---------------------------------------------------------------------------------------------------------

class _Flat:
    """One zero-padded flat buffer the size of the parameters, its unpadded part and its per-tensor views."""

    __slots__ = ("buf", "main", "views")

    def __init__(self, padded, total, sizes, dtype, device):
        self.buf = torch.zeros(padded, dtype=dtype, device=device)
        self.main = self.buf[:total]
        self.views = list(self.main.split(sizes))


class _Collector:
    """What one probe read keeps: logits, the penultimate vector and, per convolution, pooled outputs and statistics."""

    def __init__(self, mode, n_conv):
        self.mode = mode
        self.side, self.first, self.seen = "ho", True, set()
        self.logits = {"ho": [], "tr": []}
        self.pen = {"ho": [], "tr": []}
        self.pooled = {"ho": [[] for _ in range(n_conv)], "tr": [[] for _ in range(n_conv)]}
        self.sq = [[] for _ in range(n_conv)]
        self.absm = [[] for _ in range(n_conv)]
        self.pair = [[] for _ in range(n_conv)]

    def begin(self, side, first):
        self.side, self.first, self.seen = side, first, set()

    def pre_head(self, module, args):
        if "head" in self.seen or not args:
            return None
        self.seen.add("head")
        h = args[0].detach().float()
        if h.dim() > 2:
            h = h.flatten(1, -2).mean(dim=1)
        self.pen[self.side].append(h)
        return None

    def conv(self, index, module, inputs, output):
        if index in self.seen or (self.mode == "drift" and self.side != "ho"):
            return None
        self.seen.add(index)
        a = output.detach()
        if a.dim() < 3:
            return None
        a = a.float()
        self.pooled[self.side][index].append(a.flatten(2).mean(dim=2))
        if self.mode != "full":
            return None
        n = int(a.shape[0])
        self.sq[index].append((a.pow(2).mean() * n, n))
        if self.side == "tr":
            self.absm[index].append((a.abs().flatten(2).mean(dim=2).sum(dim=0), n))
        if self.first:
            k = min(F2_ROWS, n) // 2
            if k >= 1:
                u, v = a[:k].flatten(1), a[k:2 * k].flatten(1)
                den = (u.norm(dim=1) * v.norm(dim=1)).clamp_min(1e-30)
                self.pair[index].append((((u * v).sum(dim=1) / den).sum(), k))
        return None

    def pooled_side(self, side):
        return [torch.cat(parts) if parts else None for parts in self.pooled[side]]

    def logits_all(self):
        return torch.cat(self.logits["ho"] + self.logits["tr"])

    def pen_all(self):
        parts = self.pen["ho"] + self.pen["tr"]
        return torch.cat(parts) if parts and len(parts) == len(self.logits["ho"]) + len(self.logits["tr"]) else None


class NapLogger:
    """
    The NAP2-v2 logger (module docstring). ``origin`` is the unpruned network (never modified: it is deep-copied before
    it is read); ``val_dataset`` and ``train_dataset`` are the val and train loaders' datasets; ``out_dir`` is the
    array file's directory (a string, or a callable returning one, resolved at ``on_end``). ``batch_size``,
    ``momentum`` and ``weight_decay`` are the fine-tune's (K2, V5's momentum form, V8); None leaves those missing.
    """

    def __init__(self, origin, val_dataset, train_dataset, n_classes, device, label, network, recorder=None,
                 printer=None, out_dir=None, dataset=None, batch_size=None, momentum=None, weight_decay=None,
                 probe_n=PROBE_N, calib_n=CALIB_N, read_steps=READ_STEPS, ckpt_steps=CKPT_STEPS):
        self.origin = origin
        self.val_dataset, self.train_dataset = val_dataset, train_dataset
        self.n_classes = None if n_classes is None else int(n_classes)
        self.device = device
        self.label = str(label)
        self.network = str(network)
        self.name = os.path.basename(self.network)
        self.recorder, self.printer = recorder, printer
        self.out_dir = out_dir
        self.dataset = dataset
        self.batch_size = None if batch_size is None else int(batch_size)
        self.momentum = None if momentum is None else float(momentum)
        self.weight_decay = None if weight_decay is None else float(weight_decay)
        self.probe_n, self.calib_n = int(probe_n), int(calib_n)
        self.read_set = set(int(s) for s in read_steps)
        self.ckpt_set = set(int(s) for s in ckpt_steps)
        self.started = self.done = self.dead = False
        self.failure = None
        self.records, self.record0 = [], None
        self.file_path, self.file_bytes = None, 0
        self.model = None
        self.n_steps = 0
        self.t_step = self.t_reads = self.t_start = 0.0
        self.mem_peak = 0
        self.notes = []
        self._pending = None

    # --- the hook's three calls: a failure stops the logger, never the fine-tune ------------------------------------

    def on_start(self, model):
        if self.dead or self.started:
            return
        try:
            self._start(model)
        except Exception as error:  # noqa: BLE001
            self._fail("on_start", error)

    def on_step(self, step, loss, grad_norm, lr):
        if self.dead or not self.started or self.done:
            return
        try:
            self._step(int(step), loss, grad_norm, lr)
        except Exception as error:  # noqa: BLE001
            self._fail("step %s" % step, error)

    def on_end(self, step):
        if self.dead or not self.started or self.done:
            return
        try:
            self._end(int(step))
        except Exception as error:  # noqa: BLE001
            self._fail("on_end", error)

    def _fail(self, where, error):
        self.dead = True
        self.failure = "%s: %s: %s" % (where, type(error).__name__, error)
        self._release()
        text = "[eval] FT_NAP %s %s: stopped at %s (%s: %s); the fine-tune continues" % (
            self.label, self.name, where, type(error).__name__, str(error)[:300])
        try:
            if self.printer is not None:
                self.printer(text)
            issue = getattr(self.recorder, "issue", None)
            if issue is not None:
                issue("ft_nap_failed", self.failure[:500], network=self.network, label=self.label)
        except Exception:  # noqa: BLE001
            pass

    def _release(self):
        for name in ("G", "T0", "C1", "C2", "S", "X1", "X2", "upd", "prv", "cur", "rows", "gsk", "s5", "sign",
                     "bperm", "mask_prev", "bn_mask_prev", "x_ho", "x_tr", "ref_stale", "ref_recal", "_pen_dev",
                     "y_all_dev", "sizes_t", "norm0", "_pen0", "_f0_q", "_zeros", "model", "params", "convs", "head",
                     "bns", "layer_mods"):
            if hasattr(self, name):
                setattr(self, name, None)
        self._pending = None

    # --- memory and time bookkeeping ------------------------------------------------------------------------------

    def _mem_begin(self):
        if getattr(self.dev, "type", "cpu") != "cuda":
            return None
        return torch.cuda.memory_allocated(self.dev), torch.cuda.max_memory_allocated(self.dev)

    def _mem_end(self, mark):
        if mark is None:
            return
        peak = torch.cuda.max_memory_allocated(self.dev)
        extra = (peak - mark[0]) if peak > mark[1] else (torch.cuda.memory_allocated(self.dev) - mark[0])
        self.mem_peak = max(self.mem_peak, int(extra))

    def _held_bytes(self):
        total = 0
        for name in ("G", "T0", "C1", "C2", "S", "X1", "X2", "upd", "prv", "cur"):
            flat = getattr(self, name, None)
            if flat is not None:
                total += flat.buf.numel() * flat.buf.element_size()
        for name in ("rows", "gsk", "s5", "sign", "bperm", "mask_prev", "x_ho", "x_tr"):
            tensor = getattr(self, name, None)
            if torch.is_tensor(tensor):
                total += tensor.numel() * tensor.element_size()
        for name in ("ref_stale", "ref_recal"):
            for tensor in getattr(self, name, None) or []:
                if torch.is_tensor(tensor):
                    total += tensor.numel() * tensor.element_size()
        return total

    # --- on_start ---------------------------------------------------------------------------------------------------

    def _layout(self, model):
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        if not self.params:
            raise RuntimeError("no trainable parameter")
        self.dev, self.dtype = self.params[0].device, self.params[0].dtype
        if any(p.dtype != self.dtype or p.device != self.dev for p in self.params):
            raise RuntimeError("parameters of mixed dtype or device")
        self.shapes = [p.shape for p in self.params]
        self.sizes = [int(p.numel()) for p in self.params]
        self.sizes_np = np.asarray(self.sizes, dtype=np.float64)
        self.N, self.P = len(self.params), int(sum(self.sizes))
        names = {id(m): n for n, m in model.named_modules()}
        self.convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        self.head = linears[-1] if linears else None
        self.bns = [m for m in model.modules() if isinstance(m, _BatchNorm) and m.weight is not None
                    and m.running_mean is not None]
        index = {id(p): i for i, p in enumerate(self.params)}
        self.layer_mods = self.convs + ([self.head] if self.head is not None else [])
        self.layer_names = [names.get(id(m), "?") for m in self.layer_mods]
        self.lcols = np.asarray([index.get(id(m.weight), -1) for m in self.layer_mods], dtype=np.int64)
        self.layer_shapes = [tuple(int(d) for d in m.weight.shape) for m in self.layer_mods]
        self.matrix_like = np.asarray([p.dim() >= 2 for p in self.params], dtype=bool)
        self.Lc, self.Lw, self.Lb = len(self.convs), len(self.layer_mods), len(self.bns)
        self.bn_sizes = [int(m.weight.numel()) for m in self.bns]
        use_cuda = self.dev.type == "cuda"
        self._amp = bool(utils.env_flag("SPECTRA_AMP") and use_cuda)
        self._cl = bool(utils.env_flag("SPECTRA_CHANNELS_LAST") and use_cuda)

    def _buffers(self):
        self.m = max(1, -(-(self.P + 1) // SKETCH_DIM))  # at least one padding slot
        self.P_pad = self.m * SKETCH_DIM
        self.nb = self.P_pad // SKETCH_BLOCK
        rng = np.random.default_rng(seed_of(self.dataset, self.name, "S5"))
        self.bperm = torch.as_tensor(rng.permutation(self.nb).astype(np.int64), device=self.dev)
        sign = rng.integers(0, 2, size=self.P_pad, dtype=np.int8).astype(np.float32) * 2.0 - 1.0
        self.sign = torch.as_tensor(sign, device=self.dev).to(self.dtype).view(self.m, SKETCH_DIM)

        def flat():
            return _Flat(self.P_pad, self.P, self.sizes, self.dtype, self.dev)

        self.G, self.T0, self.C1, self.C2, self.S, self.X1, self.X2 = (flat() for _ in range(7))
        self.upd, self.prv, self.cur = flat(), flat(), flat()
        self.sizes_t = torch.as_tensor(self.sizes, dtype=torch.long, device=self.dev)
        self._cap = 512
        self._width = 3 + 3 * self.N
        self.rows = torch.zeros(self._cap, self._width, dtype=self.dtype, device=self.dev)
        self.gsk = torch.zeros(self._cap, SKETCH_DIM, dtype=self.dtype, device=self.dev)
        self.s5 = torch.zeros(self._cap // PROJ_EVERY + 2, SKETCH_DIM, dtype=self.dtype, device=self.dev)
        self.s5_steps = [0]  # row 0: theta_0 - theta_0
        self._zeros = {}
        with torch.no_grad():
            torch.cat([p.detach().reshape(-1) for p in self.params], out=self.prv.main)
            self.T0.buf.copy_(self.prv.buf)
            self.C1.buf.copy_(self.prv.buf)
            self.norm0 = _norms(self.T0.views)
        self.mask_prev = self.bn_mask_prev = None
        self.ck_steps, self.ck_dist, self.ck_prev = [], [], None
        self._win_start = 0
        self._pulled = 0
        self.h_rows = np.zeros((0, self._width), dtype=np.float64)
        self.h_lr, self.h_a = [], []
        self._ema = [None, None]
        self._x_last = 0
        self.j_hist = []

    def _fwd(self, model, x):
        if self._cl and x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        if self._amp:
            with torch.cuda.amp.autocast(enabled=True):
                return model(x)
        return model(x)

    def _probe_read(self, model, mode, convs, head):
        """Both probe sides through ``model`` in eval mode, no grad; the hooks live only inside this call."""
        col = _Collector(mode, len(convs))
        handles = []
        try:
            if head is not None:
                handles.append(head.register_forward_pre_hook(col.pre_head))
            if mode != "plain":
                for index, conv in enumerate(convs):
                    handles.append(conv.register_forward_hook(
                        lambda module, inputs, output, index=index: col.conv(index, module, inputs, output)))
            model.eval()
            with torch.no_grad():
                for side, x in (("ho", self.x_ho), ("tr", self.x_tr)):
                    for start in range(0, int(x.shape[0]), PROBE_CHUNK):
                        col.begin(side, start == 0)
                        col.logits[side].append(self._fwd(model, x[start:start + PROBE_CHUNK]).detach().float())
        finally:
            for handle in handles:
                handle.remove()
        return col

    def _live_read(self, mode):
        """A probe read of the live model: module flags and every random state are put back."""
        model = self.model
        flags = [(m, m.training) for m in model.modules()]
        with _RngGuard():
            try:
                return self._probe_read(model, mode, self.convs, self.head)
            finally:
                for module, training in flags:
                    module.training = training  # the attribute, not train(): train() recurses

    def _start(self, model):
        t0 = time.perf_counter()
        with _RngGuard():
            self._layout(model)
            mark = self._mem_begin()
            probe = build_probe(self.val_dataset, self.train_dataset, self.dataset, self.probe_n, self.calib_n)
            self.dataset = probe["dataset"]
            self._probe_meta = {k: probe[k] for k in ("idx_ho", "idx_tr", "base_ho", "base_tr", "how_tr")}
            n_ho, n_tr = int(probe["x_ho"].shape[0]), min(self.probe_n, int(probe["x_cal"].shape[0]))
            if n_ho < 4 or n_tr < 4:
                raise RuntimeError("the probe has %d val and %d train images" % (n_ho, n_tr))
            self.n_ho, self.n_tr = n_ho, n_tr
            self.x_ho = probe["x_ho"].to(self.dev, self.dtype)
            self.x_tr = probe["x_cal"][:n_tr].to(self.dev, self.dtype)
            self.y_ho, self.y_tr = probe["y_ho"].astype(np.int64), probe["y_cal"][:n_tr].astype(np.int64)
            self.y_all = np.concatenate([self.y_ho, self.y_tr])
            self.y_all_dev = torch.as_tensor(self.y_all, device=self.dev)
            self._buffers()
            work = copy.deepcopy(model)
            work_convs = [m for m in work.modules() if isinstance(m, torch.nn.Conv2d)]
            work_linears = [m for m in work.modules() if isinstance(m, torch.nn.Linear)]
            work_head = work_linears[-1] if work_linears else None
            stale = self._probe_read(work, "full", work_convs, work_head)
            logits0 = stale.logits_all()
            if logits0.dim() != 2 or logits0.shape[0] != n_ho + n_tr:
                raise RuntimeError("the model's output has shape %s" % (tuple(logits0.shape),))
            classes = int(logits0.shape[1])
            if self.n_classes is not None and self.n_classes != classes:
                self.notes.append("n_classes given as %d, the model has %d outputs" % (self.n_classes, classes))
            self.n_classes = classes
            if int(self.y_all.max()) >= classes or int(self.y_all.min()) < 0:
                raise RuntimeError("probe labels outside the model's %d classes" % classes)
            bn_shift = self._recalibrate(work, probe["x_cal"])
            recal = self._probe_read(work, "full", work_convs, work_head)
            origin = self._origin_read()
            rec0 = self._cut_record(stale, recal, origin, bn_shift)
            self._cut_gradients(rec0, work, work_convs, work_head, probe)
            del work
            self._y4_reference()
            rec = self._first_record(stale, recal)
            self._mem_end(mark)
        seconds = time.perf_counter() - t0
        self.t_start = seconds
        rec0["seconds"] = round(seconds, 3)
        self.started = True
        self._emit("eval_ft_nap0", rec0)
        self.record0 = rec0
        self._pending = rec
        h5 = rec0["H5"]
        self._say("[eval] FT_NAP %s %s cut | H5 loo stale %s recal %s origin %s | acc ho stale %s recal %s | %.1f s" % (
            self.label, self.name, _fmt((h5.get("stale") or {}).get("loo_acc")), _fmt((h5.get("recal") or {}).get("loo_acc")),
            _fmt((h5.get("origin") or {}).get("loo_acc")), _fmt(rec0["BN"].get("stale_acc_ho")),
            _fmt(rec0["BN"].get("recal_acc_ho")), seconds))
        self._say_read(rec, seconds)

    def _recalibrate(self, work, x_cal):
        """Re-estimate the copy's BatchNorm running statistics from the calibration images (train-mode BatchNorm
        only, cumulative average, no grad). Returns the per-layer shift between the stale and the new statistics."""
        bns = [m for m in work.modules() if isinstance(m, _BatchNorm) and m.track_running_stats
               and m.running_mean is not None]
        work.eval()
        if not bns:
            return None
        old = [(m.running_mean.detach().clone(), m.running_var.detach().clone()) for m in bns]
        for m in bns:
            m.reset_running_stats()
            m.momentum = None
            m.training = True
        with torch.no_grad():
            for start in range(0, int(x_cal.shape[0]), PROBE_CHUNK):
                chunk = x_cal[start:start + PROBE_CHUNK]
                if chunk.shape[0] >= 2:
                    self._fwd(work, chunk.to(self.dev, self.dtype))
        work.eval()
        mean_shift, var_shift = [], []
        for m, (mean0, var0) in zip(bns, old):
            scale = (var0.double() + m.eps).sqrt()
            mean_shift.append(((m.running_mean.double() - mean0.double()).abs() / scale).mean())
            var_shift.append(((m.running_var.double() + m.eps).log() - (var0.double() + m.eps).log()).abs().mean())
        return torch.stack(mean_shift + var_shift).cpu().numpy().reshape(2, len(bns))

    def _origin_read(self):
        """The origin's probe logits and H5, once per process and network (a deep copy is read; the origin is not)."""
        if self.origin is None:
            return None
        key = (id(self.origin), self.network, id(self.val_dataset), id(self.train_dataset), self.probe_n)
        hit = _ORIGIN_CACHE.get(key)
        if hit is not None and hit["pin"] is self.origin:
            return hit
        out = {"pin": self.origin, "logits": None, "h5": None, "why": None}
        try:
            work = copy.deepcopy(self.origin).to(self.dev).float()
            linears = [m for m in work.modules() if isinstance(m, torch.nn.Linear)]
            col = self._probe_read(work, "plain", [], linears[-1] if linears else None)
            logits = col.logits_all()
            if logits.dim() != 2 or logits.shape[1] != self.n_classes:
                out["why"] = "the origin has %s outputs" % (tuple(logits.shape[1:]),)
            else:
                out["logits"] = logits.cpu().numpy()
                pen = col.pen_all()
                if pen is not None:
                    out["h5"] = h5_forecast(pen, self.y_all_dev, self.n_classes, self.n_ho)
            del work, col
        except Exception as error:  # noqa: BLE001 - the origin's features go missing, with the reason
            out["why"] = "%s: %s" % (type(error).__name__, str(error)[:200])
        while len(_ORIGIN_CACHE) >= 4:
            _ORIGIN_CACHE.pop(next(iter(_ORIGIN_CACHE)))
        _ORIGIN_CACHE[key] = out
        return out

    def _sides(self, logits):
        logits = np.asarray(logits)
        return _Side(logits[:self.n_ho], self.y_ho), _Side(logits[self.n_ho:], self.y_tr)

    def _state_features(self, col):
        """Z3.1, Z3.2, F7pre, F2 and H1 of one full read (one device-to-host copy)."""
        dev, f64 = self.dev, torch.float64
        nan = torch.full((), float("nan"), dtype=f64, device=dev)
        pooled_ho, pooled_tr = col.pooled_side("ho"), col.pooled_side("tr")
        pen = col.pen_all()
        yc = torch.nn.functional.one_hot(torch.as_tensor(self.y_tr, device=dev), self.n_classes).to(f64)
        yc = yc - yc.mean(dim=0, keepdim=True)
        yy = (yc.t() @ yc).norm()
        scal, grams, gram_shapes, dorm_counts, widths = [], [], [], [], 0
        reps = [(pooled_ho[i], pooled_tr[i]) for i in range(len(pooled_ho))]
        reps.append((pen[:self.n_ho], pen[self.n_ho:]) if pen is not None else (None, None))
        for index, (ho, tr) in enumerate(reps):
            is_conv = index < len(pooled_ho)
            q = c = d01 = d025 = dead = nan
            if is_conv and col.sq[index]:
                q = sum(v for v, _ in col.sq[index]).double() / sum(n for _, n in col.sq[index])
            if is_conv and col.pair[index]:
                c = sum(v for v, _ in col.pair[index]).double() / sum(k for _, k in col.pair[index])
            if is_conv and col.absm[index]:
                mean_abs = sum(v for v, _ in col.absm[index]).double() / sum(n for _, n in col.absm[index])
                level = mean_abs.mean()
                counts = torch.stack([(mean_abs <= 0.01 * level).sum(), (mean_abs <= 0.025 * level).sum(),
                                      (mean_abs <= 0).sum()]).double()
                dorm_counts.append(counts)
                widths += int(mean_abs.numel())
                d01, d025, dead = (counts / mean_abs.numel()).unbind(0)
            align = nan
            if tr is not None and tr.shape[0] >= 2:
                hc = tr.double()
                hc = hc - hc.mean(dim=0, keepdim=True)
                align = (hc.t() @ yc).pow(2).sum() / ((hc.t() @ hc).norm() * yy)
            scal.append(torch.stack([q, c, d01, d025, dead, align]))
            if ho is not None and tr is not None:
                x = torch.cat([ho[:F2_ROWS], tr[:F2_ROWS]]).double()
                x = x - x.mean(dim=0, keepdim=True)
                gram = (x.t() @ x) if x.shape[1] <= x.shape[0] else (x @ x.t())
                grams.append(gram.reshape(-1))
                gram_shapes.append(int(gram.shape[0]))
            else:
                gram_shapes.append(0)
        totals = torch.stack(dorm_counts).sum(dim=0) if dorm_counts else torch.zeros(3, dtype=f64, device=dev)
        packed = torch.cat([torch.stack(scal).reshape(-1), totals] + grams).cpu().numpy()
        n_rep = len(reps)
        table = packed[:n_rep * 6].reshape(n_rep, 6)
        totals = packed[n_rep * 6:n_rep * 6 + 3]
        ranks = np.full((n_rep, 3), np.nan)
        offset = n_rep * 6 + 3
        for index, side in enumerate(gram_shapes):
            if side:
                got = spectrum_ranks(packed[offset:offset + side * side].reshape(side, side))
                offset += side * side
                if got is not None:
                    ranks[index] = got
        return {"q": table[:self.Lc, 0], "c": table[:self.Lc, 1], "d01": table[:self.Lc, 2],
                "d025": table[:self.Lc, 3], "dead": table[:self.Lc, 4], "align": table[:, 5], "ranks": ranks,
                "dorm_all": (totals / widths) if widths else np.full(3, np.nan),
                "has_pen": pen is not None}

    def _cut_record(self, stale, recal, origin, bn_shift):
        """The ``eval_ft_nap0`` dicts that need no gradient; keeps what later reads compare against."""
        rec = {"network": self.network, "label": self.label, "step": 0, "seconds": 0.0}
        logits = {"stale": stale.logits_all().cpu().numpy(), "recal": recal.logits_all().cpu().numpy()}
        self.logits_recal0 = logits["recal"]
        self.sides0 = {name: self._sides(value) for name, value in logits.items()}
        self.origin_logits = None if origin is None else origin["logits"]
        self.origin_sides = None if self.origin_logits is None else self._sides(self.origin_logits)
        origin_why = "no origin model" if origin is None else (origin["why"] or "origin not read")
        pens = {"stale": stale.pen_all(), "recal": recal.pen_all()}
        self._pen0 = pens
        depth = np.linspace(0.0, 1.0, self.Lc) if self.Lc > 1 else np.zeros(max(self.Lc, 1))
        h5, z3, f7, f2, h1, pdiv = _Fam(), _Fam(), _Fam(form="F7pre"), _Fam(rows=2 * F2_ROWS), _Fam(), _Fam()
        self.cut_arrays = {}
        self.h5_0 = {}
        for name, col in (("stale", stale), ("recal", recal)):
            got = h5_forecast(pens[name], self.y_all_dev, self.n_classes, self.n_ho) if pens[name] is not None else None
            self.h5_0[name] = got
            sub = _Fam()
            if got is None:
                sub.miss("H5", "no final Linear" if pens[name] is None else "empty kernel")
            else:
                _put_h5(sub, got)
            h5[name] = sub
            feats = self._state_features(col)
            for key in ("q", "c", "d01", "d025", "dead"):
                self.cut_arrays["%s_%s" % (name, key)] = feats[key]
            self.cut_arrays["%s_align" % name] = feats["align"]
            for i, key in enumerate(("erank", "srank", "pr")):
                self.cut_arrays["%s_%s" % (name, key)] = feats["ranks"][:, i]
            sub = _Fam()
            with np.errstate(all="ignore"):
                sub.form("q_L", feats["q"])
                logq = np.log(feats["q"])
                ok = np.isfinite(logq)
                slope = np.polyfit(depth[ok], logq[ok], 1)[0] if ok.sum() >= 2 else None
                sub.put("q_logslope", slope, "fewer than two layers")
                sub.put("q_ratio", (feats["q"][-1] / feats["q"][0]) if self.Lc else None, "no convolution")
                sub.form("c_L", feats["c"])
                sub.put("c_last", feats["c"][-1] if self.Lc else None, "no convolution")
                ok = np.isfinite(feats["c"])
                sub.put("c_slope", np.polyfit(depth[ok], feats["c"][ok], 1)[0] if ok.sum() >= 2 else None,
                        "fewer than two layers")
            z3[name] = sub
            sub = _Fam()
            sub.form("dormant_L", feats["d01"])
            sub.form("dormant025_L", feats["d025"])
            sub.put("dormant_all", feats["dorm_all"][0])
            sub.put("dormant025_all", feats["dorm_all"][1])
            sub.put("dead_all", feats["dorm_all"][2])
            f7[name] = sub
            sub = _Fam()
            for i, key in enumerate(("erank", "srank", "pr")):
                sub.form(key + "_L", feats["ranks"][:, i])
                sub.put(key + "_pen", feats["ranks"][-1, i] if feats["has_pen"] else None, "no final Linear")
            f2[name] = sub
            sub = _Fam()
            sub.form("align_L", feats["align"])
            sub.put("align_pen", feats["align"][-1] if feats["has_pen"] else None, "no final Linear")
            h1[name] = sub
            sub = _Fam()
            ho, tr = self.sides0[name]
            if self.origin_sides is not None:
                with np.errstate(all="ignore"):
                    for tag, side, ref in (("ho", ho, self.origin_sides[0]), ("tr", tr, self.origin_sides[1])):
                        sub.put("js_" + tag, js_divergence(side.logp, ref.logp).mean())
                        sub.put("agree_" + tag, (side.pred == ref.pred).mean())
            else:
                sub.miss("PDIV", origin_why)
            pdiv[name] = sub
        sub = _Fam()
        if origin is not None and origin["h5"] is not None:
            _put_h5(sub, origin["h5"])
        else:
            sub.miss("H5", origin_why)
        h5["origin"] = sub
        if self.origin_sides is not None:
            for tag, side in (("ho", self.origin_sides[0]), ("tr", self.origin_sides[1])):
                pdiv.put("origin_acc_" + tag, side.correct.mean())
                pdiv.put("origin_loss_" + tag, side.ce.mean())
        bn = _Fam(n_bn=0 if bn_shift is None else int(bn_shift.shape[1]), calib_n=int(min(self.calib_n, self._calib_count())))
        s_ho, s_tr = self.sides0["stale"]
        r_ho, r_tr = self.sides0["recal"]
        with np.errstate(all="ignore"):
            bn.put("stale_acc_ho", s_ho.correct.mean())
            bn.put("recal_acc_ho", r_ho.correct.mean())
            bn.put("stale_vs_recal_ho_acc", s_ho.correct.mean() - r_ho.correct.mean())
            for tag, a, b in (("ho", s_ho, r_ho), ("tr", s_tr, r_tr)):
                bn.put("flip_" + tag, (a.pred != b.pred).mean())
                bn.put("js_" + tag, js_divergence(a.logp, b.logp).mean())
        if bn_shift is None:
            bn.miss("shift", "no BatchNorm with running statistics: recal equals stale")
        else:
            bn.form("shift_mean_L", bn_shift[0])
            bn.form("shift_var_L", bn_shift[1])
            self.cut_arrays["bn_shift_mean"], self.cut_arrays["bn_shift_var"] = bn_shift[0], bn_shift[1]
        self.ref_stale, self.ref_recal = stale.pooled_side("ho"), recal.pooled_side("ho")
        rec.update({"H5": h5, "Z3": z3, "F7": f7, "F2": f2, "H1": h1, "PDIV": pdiv, "Z4": _Fam(), "BN": bn})
        return rec

    def _calib_count(self):
        try:
            return len(self.train_dataset)
        except TypeError:
            return self.calib_n

    def _cut_gradients(self, rec0, work, convs, head, probe):
        """Z4.1-Z4.5, Z3.3 and Z3.4 on the recalibrated copy (float32, eval mode, autocast off), then SynFlow, which
        rewrites the copy. A failure leaves these features missing with its reason."""
        z4, z3 = rec0["Z4"], rec0["Z3"]
        z4["state"] = "recal"
        layers = convs + ([head] if head is not None else [])
        weights = [m.weight for m in layers]
        params = [p for p in work.parameters() if p.requires_grad]
        depth = np.linspace(0.0, 1.0, len(convs)) if len(convs) > 1 else np.zeros(max(len(convs), 1))
        x_cal, y_cal = probe["x_cal"], probe["y_cal"]
        work.eval().float()
        grad_sub, jac_sub = _Fam(), _Fam()
        z3["grad"], z3["jac"] = grad_sub, jac_sub
        try:
            pick = np.random.default_rng(seed_of(self.dataset, self.name, "Z4")).permutation(int(x_cal.shape[0]))
            pick = pick[:Z4_BATCHES * Z4_BATCH]
            acts, handles = {}, []

            def keep(index):
                def hook(module, inputs, output):
                    if index in acts:  # a module called twice: its first output is the layer's
                        return None
                    acts[index] = output
                    return output.clone()  # the convolution's own output stays a node no later in-place op rewrites
                return hook

            for index, conv in enumerate(convs):
                handles.append(conv.register_forward_hook(keep(index)))
            scal, per_layer, fisher, act_grad, grasp_layers = [], [], [], [], []
            grasp_why = None

            def gradients(x, y, second):
                """One minibatch: the gradients of the loss at the parameters and at each convolution's output, and
                (``second``) the Hessian-vector product along the gradient, by a double backward."""
                acts.clear()
                loss = torch.nn.functional.cross_entropy(work(x).float(), y)
                order = sorted(acts)
                outs = [acts[i] for i in order]
                grads = torch.autograd.grad(loss, params + outs, create_graph=second, allow_unused=True)
                gp, ga = grads[:len(params)], grads[len(params):]
                hg = None
                if second:
                    z = sum((g.detach() * g).sum() for g in gp if g is not None)
                    hg = torch.autograd.grad(z, weights, allow_unused=True)
                return order, outs, gp, ga, hg

            try:
                for b in range(Z4_BATCHES):
                    idx = pick[b * Z4_BATCH:(b + 1) * Z4_BATCH]
                    if idx.size < 2:
                        break
                    idx_t = torch.as_tensor(np.sort(idx))
                    x = x_cal[idx_t].to(self.dev, torch.float32)
                    y = torch.as_tensor(y_cal[np.sort(idx)], device=self.dev)
                    with torch.enable_grad():
                        try:
                            order, outs, gp, ga, hg = gradients(x, y, grasp_why is None)
                        except Exception as error:  # noqa: BLE001 - no double backward here: first order only
                            if grasp_why is not None:
                                raise
                            grasp_why = "%s: %s" % (type(error).__name__, str(error)[:160])
                            order, outs, gp, ga, hg = gradients(x, y, False)
                    by_id = {id(p): g for p, g in zip(params, gp)}
                    lg = [by_id.get(id(w)) for w in weights]
                    zero = torch.zeros((), dtype=torch.float64, device=self.dev)
                    gnorm = torch.stack([g.detach().double().pow(2).sum() for g in gp if g is not None]).sum().sqrt()
                    layer_norm = torch.stack([g.detach().double().norm() if g is not None else zero for g in lg])
                    snip = torch.stack([(w.detach() * g.detach()).abs().double().sum() if g is not None else zero
                                        for w, g in zip(weights, lg)])
                    scal.append(torch.stack([gnorm, layer_norm.sum(), snip.sum()]))
                    if hg is not None:
                        grasp_layers.append(torch.stack([-(w.detach() * h).double().sum() if h is not None else zero
                                                         for w, h in zip(weights, hg)]))
                    fis, agn, fis_param = [zero] * len(convs), [zero] * len(convs), zero
                    for index, a, g in zip(order, outs, ga):
                        if g is None or a.dim() < 3:
                            continue
                        prod = (a.detach() * g.detach()).double().flatten(2).sum(dim=2)
                        channel = 0.5 * prod.pow(2).mean(dim=0)
                        fis[index] = channel.sum()
                        fis_param = fis_param + channel.sum() * (convs[index].weight[0].numel())
                        agn[index] = g.detach().double().norm()
                    per_layer.append(snip)
                    if convs:
                        fisher.append(torch.cat([torch.stack(fis), fis_param.reshape(1)]))
                        act_grad.append(torch.stack(agn))
                    del gp, ga, lg, hg, outs
            finally:
                for handle in handles:
                    handle.remove()
                acts.clear()
            if not scal:
                raise RuntimeError("no minibatch of at least 2 calibration images")
            scal = torch.stack(scal).cpu().numpy()
            z4["batches"], z4["batch"] = int(scal.shape[0]), Z4_BATCH
            for i, name in enumerate(("gnorm", "gnorm_layersum", "snip")):
                z4.put(name, scal[:, i].mean())
                z4.put(name + "_sd", scal[:, i].std())
            snip_layers = torch.stack(per_layer).mean(dim=0).cpu().numpy()
            z4.form("snip_L", snip_layers)
            self.cut_arrays["z4_snip"] = snip_layers
            if grasp_layers:
                grasp = torch.stack(grasp_layers).cpu().numpy()
                z4.put("grasp", grasp.sum(axis=1).mean())
                z4.put("grasp_sd", grasp.sum(axis=1).std())
                z4.form("grasp_L", grasp.mean(axis=0))
                self.cut_arrays["z4_grasp"] = grasp.mean(axis=0)
            else:
                z4.miss("grasp", grasp_why or "no Hessian-vector product")
            if fisher:
                fis = torch.stack(fisher).mean(dim=0).cpu().numpy()
                z4.put("fisher_ch", fis[:-1].sum())
                z4.put("fisher_param", fis[-1])
                z4.form("fisher_L", fis[:-1])
                self.cut_arrays["z4_fisher"] = fis[:-1]
                agn = torch.stack(act_grad).mean(dim=0).cpu().numpy()
                self.cut_arrays["z33_gnorm"] = agn
                with np.errstate(all="ignore"):
                    grad_sub.form("g_L", agn)
                    logg = np.log(agn)
                    ok = np.isfinite(logg)
                    grad_sub.put("g_logslope", np.polyfit(depth[ok], logg[ok], 1)[0] if ok.sum() >= 2 else None,
                                 "fewer than two layers")
                    grad_sub.put("g_ratio", agn[0] / agn[-1])
            else:
                z4.miss("fisher", "no convolution")
                grad_sub.miss("Z3.3", "no convolution")
        except Exception as error:  # noqa: BLE001
            why = "%s: %s" % (type(error).__name__, str(error)[:200])
            z4.miss("Z4", why)
            grad_sub.miss("Z3.3", why)
        try:
            self._jacobian(jac_sub, work)
        except Exception as error:  # noqa: BLE001
            jac_sub.miss("Z3.4", "%s: %s" % (type(error).__name__, str(error)[:200]))
        try:
            self._synflow(z4, work, weights)
        except Exception as error:  # noqa: BLE001
            z4.miss("synflow", "%s: %s" % (type(error).__name__, str(error)[:200]))

    def _jacobian(self, sub, work):
        """Z3.4, exact: the C x D input-output Jacobian of JAC_INPUTS probe inputs from one backward pass per chunk
        of class-replicated inputs; its singular values from the C x C Gram matrix."""
        classes = self.n_classes
        if classes > JAC_REPLICAS:
            sub.miss("Z3.4", "%d classes: over the %d-replica cap" % (classes, JAC_REPLICAS))
            return
        half = JAC_INPUTS // 2
        xs = torch.cat([self.x_ho[:half], self.x_tr[:half]]).float()
        per = max(1, JAC_REPLICAS // classes)
        grams = []
        for start in range(0, int(xs.shape[0]), per):
            xb = xs[start:start + per]
            count = int(xb.shape[0])
            xr = xb.repeat_interleave(classes, dim=0).clone().requires_grad_(True)
            with torch.enable_grad():
                out = work(xr).float()
                picked = out.view(count, classes, classes).diagonal(dim1=1, dim2=2).sum()
                (grad,) = torch.autograd.grad(picked, xr)
            jac = grad.reshape(count, classes, -1).double()
            grams.append(jac @ jac.transpose(1, 2))
            dim = int(jac.shape[2])
        lam = torch.linalg.eigvalsh(torch.cat(grams).cpu()).clamp_min(0).flip(-1).numpy()
        sigma = np.sqrt(lam)
        tenth = min(10, classes) - 1
        with np.errstate(all="ignore"):
            sub.put("s1_mean", sigma[:, 0].mean())
            full = sigma[:, tenth] > Y4_RANK_TOL * sigma[:, 0]  # below this a Gram's singular value is noise
            sub.put("s1_over_s10", (sigma[full, 0] / sigma[full, tenth]).mean() if full.any() else None,
                    "the Jacobian's rank is below %d at every input" % (tenth + 1))
            sub.put("msv", (lam.sum(axis=1) / min(classes, dim)).mean())
        sub["n_inputs"], sub["tenth"], sub["rank10_inputs"] = int(sigma.shape[0]), tenth + 1, int(full.sum())

    def _synflow(self, z4, work, weights):
        """Z4.4 on the copy: absolute weights, normalisation as identity, all-ones input, float64; logs are stored.
        A clipped ReLU (ReLU6) is opened to a ReLU: SynFlow needs a positively homogeneous network, and with absolute
        weights an all-ones input saturates the clip, which zeroes every layer's score but the head's."""
        opened = 0
        with torch.no_grad():
            for p in work.parameters():
                p.abs_()
            for m in work.modules():
                if isinstance(m, _BatchNorm):
                    if m.weight is not None:
                        m.weight.fill_(1.0)
                        m.bias.zero_()
                    if m.running_mean is not None:
                        m.running_mean.zero_()
                        m.running_var.fill_(1.0)
                    m.eps = 0.0
                elif isinstance(m, torch.nn.Hardtanh) and m.min_val == 0 and math.isfinite(m.max_val):
                    m.max_val = float("inf")
                    opened += 1
        z4["synflow_relu6_opened"] = opened
        work.double().eval()
        ones = torch.ones((1,) + tuple(self.x_ho.shape[1:]), dtype=torch.float64, device=self.dev)
        with torch.enable_grad():
            total = work(ones).sum()
            grads = torch.autograd.grad(total, weights, allow_unused=True)
        zero = torch.zeros((), dtype=torch.float64, device=self.dev)
        scores = torch.stack([(w.detach() * g).sum() if g is not None else zero for w, g in zip(weights, grads)])
        values = torch.cat([scores, total.detach().reshape(1)]).cpu().numpy()
        with np.errstate(all="ignore"):
            z4.put("synflow_log", np.log(values[:-1].sum()))
            z4.put("synflow_logR", np.log(values[-1]))
            log_layers = np.log(values[:-1])
        z4.form("synflow_L", log_layers)
        self.cut_arrays["z4_synflow_log"] = log_layers

    # --- on_step ----------------------------------------------------------------------------------------------------

    def _sketch(self, padded):
        return (padded.view(self.nb, SKETCH_BLOCK).index_select(0, self.bperm).view(self.m, SKETCH_DIM) * self.sign).sum(dim=0)

    def _grow(self):
        self._cap *= 2
        for name, rows in (("rows", self._cap), ("gsk", self._cap), ("s5", self._cap // PROJ_EVERY + 2)):
            old = getattr(self, name)
            new = torch.zeros(rows, old.shape[1], dtype=old.dtype, device=old.device)
            new[:old.shape[0]] = old
            setattr(self, name, new)

    def _zero_like(self, index):
        if index not in self._zeros:
            self._zeros[index] = torch.zeros_like(self.params[index])
        return self._zeros[index]

    def _step(self, step, loss, grad_norm, lr):
        t0 = time.perf_counter()
        params = self.params
        if [p.shape for p in params] != self.shapes:
            raise RuntimeError("a parameter changed shape during the fine-tune")
        row = self.n_steps
        if step != row + 1:
            raise RuntimeError("step %d follows %d logged steps" % (step, row))
        if row >= self._cap:
            self._grow()
        grads = [p.grad for p in params]
        if any(g is None for g in grads):
            grads = [g if g is not None else self._zero_like(i) for i, g in enumerate(grads)]
        cur = self.cur
        with torch.no_grad():
            torch.cat([g.reshape(-1) for g in grads], out=self.G.main)
            torch.cat([p.detach().reshape(-1) for p in params], out=cur.main)
            norm = grad_norm.detach().to(self.dtype).reshape(())
            scale = torch.clamp(norm + 1e-6, min=1.0)  # 1 / clip_grad_norm_'s factor: the pre-clip gradient is scale * p.grad
            a = 3.0 * float(lr) if lr is not None and float(lr) > 0 else 1.0
            self.upd.buf.add_(self.G.buf, alpha=a)  # -(previous update) + a * g
            self.prv.buf.sub_(cur.buf)              # theta_{t-1} - theta_t = -(this update)
            norms = _norms(self.G.views + self.upd.views + self.prv.views)
            head = torch.stack([loss.detach().to(self.dtype).reshape(()), norm, scale])
            self.rows[row] = torch.cat([head, norms.to(self.dtype)])
            self.S.buf.addcmul_(self.G.buf, scale)
            self.gsk[row] = self._sketch(self.G.buf)
            if step % PROJ_EVERY == 0:
                self.s5[len(self.s5_steps)] = self._sketch(cur.buf - self.T0.buf)
                self.s5_steps.append(step)
        self.upd, self.prv, self.cur = self.prv, cur, self.upd
        self.h_a.append(a)
        self.h_lr.append(None if lr is None else float(lr))
        self.n_steps = row + 1
        self.t_step += time.perf_counter() - t0
        if step in self.read_set:
            self._read(step, step in self.ckpt_set)

    # --- reads --------------------------------------------------------------------------------------------------------

    def _pull(self):
        """Copy the step rows logged since the last read to the host (the one sync of the per-step log)."""
        if self.n_steps > self._pulled:
            new = self.rows[self._pulled:self.n_steps].double().cpu().numpy()
            self.h_rows = np.concatenate([self.h_rows, new])
            self._pulled = self.n_steps

    def _series(self):
        """Host views of the step rows: loss, pre-clip norm, clip factor, per-tensor |g| (clipped), |-(prev update) + a g|
        and |update|."""
        rows, n = self.h_rows, self.N
        return rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3:3 + n], rows[:, 3 + n:3 + 2 * n], rows[:, 3 + 2 * n:]

    def _read(self, step, ckpt):
        t0 = time.perf_counter()
        mark = self._mem_begin()
        self._flush()
        with _RngGuard():
            col = self._live_read("drift" if ckpt else "plain")
            rec = self._base_record(step, col)
            if ckpt:
                self._fill_ckpt(rec, step, col)
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        self._pending = rec
        self._mem_end(mark)
        self.t_reads += time.perf_counter() - t0
        self._say_read(rec, time.perf_counter() - t0)

    def _say(self, text):
        if self.printer is not None:
            self.printer(text)

    def _say_read(self, rec, seconds):
        p = rec["P"].get("stale", rec["P"])
        self._say("[eval] FT_NAP %s %s step=%d | probe acc ho %s tr %s | churn %s | js0 %s | %.2f s" % (
            self.label, self.name, rec["step"], _fmt(p.get("acc_ho")), _fmt(p.get("acc_tr")),
            _fmt(rec["P"].get("churn_ho")), _fmt(rec["P"].get("js_init_ho")), seconds))

    def _emit(self, event, rec):
        bad = []
        rec = _clean(rec, "", bad)
        if bad:
            rec.setdefault("missing", []).extend([[path, "non-finite"] for path in bad[:50]])
        if event == "eval_ft_nap":
            self.records.append(rec)
        if self.recorder is not None:
            self.recorder.record(event, **rec)
        return rec

    def _flush(self):
        if self._pending is not None:
            self._emit("eval_ft_nap", self._pending)
            self._pending = None

    def _new_record(self, step):
        rec = {"network": self.network, "label": self.label, "step": int(step), "seconds": 0.0, "ckpt": False}
        for key in NAP_KEYS:
            rec[key] = _Fam()
        return rec

    def _remember(self, step, logits, ho, tr, pen_dev):
        """A read's history: logits for the array file, the per-example arrays E needs, the last outputs for P1/P2."""
        if not hasattr(self, "hist"):
            self.hist = {"steps": [], "logits": [], "correct": [], "margin": [], "py": [], "churn": {"ho": [], "tr": []}}
        h = self.hist
        h["steps"].append(int(step))
        h["logits"].append(np.asarray(logits, dtype=np.float32))
        h["correct"].append(np.concatenate([ho.correct, tr.correct]))
        h["margin"].append(np.concatenate([ho.margin, tr.margin]))
        h["py"].append(np.concatenate([ho.py, tr.py]))
        self._prev = (int(step), ho, tr)
        self._pen_dev = pen_dev

    def _first_record(self, stale, recal):
        """Step 0's ``eval_ft_nap``: P in both BatchNorm states and the checkpoint families at the cut."""
        rec = self._new_record(0)
        origin = self.origin_sides
        rec["P"]["stale"] = _p_static(self.sides0["stale"][0], self.sides0["stale"][1], origin)
        rec["P"]["recal"] = _p_static(self.sides0["recal"][0], self.sides0["recal"][1], origin)
        ho, tr = self.sides0["stale"]
        self._first = (ho, tr)
        self._remember(0, stale.logits_all().cpu().numpy(), ho, tr, self._pen0["stale"])
        self._fit_probe()
        self._fill_ckpt(rec, 0, stale)
        return rec

    def _fit_probe(self):
        """Y5's reference: the linear probe on the recalibrated cut's penultimate features of P_tr, fitted once."""
        self.y5 = None
        pen = self._pen0["recal"]
        if pen is None:
            return
        try:
            weight, bias = fit_linear_probe(pen[self.n_ho:], torch.as_tensor(self.y_tr, device=self.dev), self.n_classes)
            scores = pen[:self.n_ho].double() @ weight.t() + bias
            acc = float((scores.argmax(dim=1).cpu().numpy() == self.y_ho).mean())
            self.y5 = {"weight": weight.cpu().numpy(), "acc_ho": acc}
        except Exception as error:  # noqa: BLE001
            self.notes.append("Y5 probe fit failed: %s: %s" % (type(error).__name__, str(error)[:160]))

    def _base_record(self, step, col):
        """X and P of a read after step 0."""
        rec = self._new_record(step)
        logits = col.logits_all().cpu().numpy()
        ho, tr = self._sides(logits)
        self._pull()
        loss, gnorm, scale, g, _b, u = self._series()
        x = rec["X"]
        n = int(loss.shape[0])
        with np.errstate(all="ignore"):
            for value in loss[self._x_last_ema():]:
                for i, beta in enumerate((0.9, 0.99)):
                    self._ema[i] = float(value) if self._ema[i] is None else beta * self._ema[i] + (1.0 - beta) * float(value)
            self._ema_n = n
            since = slice(self._x_last, n)
            x["n"] = n
            x.put("loss_last", loss[since].mean() if n > self._x_last else None, "no step since the last read")
            x.put("loss_all", loss.mean() if n else None, "no step yet")
            x.put("ema09", self._ema[0], "no step yet")
            x.put("ema099", self._ema[1], "no step yet")
            x.put("lr", self.h_lr[-1] if self.h_lr else None, "no step yet")
            x.put("gnorm_last", gnorm[since].mean() if n > self._x_last else None, "no step since the last read")
            x.put("upd_last", np.sqrt((u[since] ** 2).sum(axis=1)).mean() if n > self._x_last else None,
                  "no step since the last read")
            x.put("clip_frac_last", (scale[since] > 1.0).mean() if n > self._x_last else None,
                  "no step since the last read")
            self._x_last = n
            p = _p_static(ho, tr, self.origin_sides)
            prev_step, prev_ho, prev_tr = self._prev
            dlog = math.log1p(step) - math.log1p(prev_step)
            js_all = []
            for tag, now, before, first in (("ho", ho, prev_ho, self._first[0]), ("tr", tr, prev_tr, self._first[1])):
                churn = float((now.pred != before.pred).mean())
                history = self.hist["churn"][tag]
                history.append(churn)
                p.put("churn_" + tag, churn)
                p.put("churn3_" + tag, float(np.mean(history[-3:])))
                p.put("churn_rate_" + tag, churn / dlog if dlog > 0 else None, "no log-step between the reads")
                js = js_divergence(now.logp, before.logp)
                js_all.append(js)
                p.put("js_prev_" + tag, js.mean())
                p.put("js_init_" + tag, js_divergence(now.logp, first.logp).mean())
            js_mean = float(np.concatenate(js_all).mean())
            p.put("js_prev_all", js_mean)
            p.put("js_rate_all", js_mean / dlog if dlog > 0 else None, "no log-step between the reads")
        rec["P"] = p
        self._remember(step, logits, ho, tr, col.pen_all())
        return rec

    def _x_last_ema(self):
        return getattr(self, "_ema_n", 0)

    # --- checkpoints --------------------------------------------------------------------------------------------------

    def _take(self, values):
        """A per-tensor array at the weight layers (NaN where a layer's weight is not trained)."""
        values = np.asarray(values, dtype=np.float64)
        out = np.full(self.Lw, np.nan)
        ok = self.lcols >= 0
        out[ok] = values[self.lcols[ok]]
        return out

    def _f0_plan(self):
        """Family 0's buckets: weight layers grouped by ceil(log2(row length)), cut into chunks of bounded size."""
        if hasattr(self, "_f0_chunks"):
            return
        buckets = {}
        self._f0_rows = []
        for layer, col in enumerate(self.lcols):
            if col < 0:
                self._f0_rows.append((0, 0))
                continue
            shape = self.layer_shapes[layer]
            out, width = int(shape[0]), int(self.sizes[col] // max(int(shape[0]), 1))
            self._f0_rows.append((out, width))
            buckets.setdefault(max(0, int(math.ceil(math.log2(max(width, 1))))), []).append(layer)
        self._f0_chunks = []
        for key in sorted(buckets):
            width = max(self._f0_rows[layer][1] for layer in buckets[key])
            per_chunk = max(1, F0_CHUNK // max(width, 1))
            chunk, used = [], 0
            for layer in buckets[key]:
                out, start = self._f0_rows[layer][0], 0
                while start < out:
                    take = min(out - start, per_chunk - used)
                    chunk.append((layer, start, start + take))
                    used += take
                    start += take
                    if used >= per_chunk:
                        self._f0_chunks.append((width, chunk))
                        chunk, used = [], 0
            if chunk:
                self._f0_chunks.append((width, chunk))
        self._f0_q = []
        for layer, col in enumerate(self.lcols):
            if col < 0:
                self._f0_q.append(None)
                continue
            last = self.sizes[col] - 1
            pos = [q * last for q in (0.25, 0.5, 0.75)]
            index = [v for p in pos for v in (int(math.floor(p)), int(math.ceil(p)))]
            self._f0_q.append((torch.as_tensor(index, dtype=torch.long, device=self.dev),
                               torch.as_tensor([p - math.floor(p) for p in pos], dtype=torch.float64, device=self.dev)))

    def _f0_device(self, views):
        """Family 0 on the device for one tensor kind: the tensors to pack (row statistics per chunk, layer quartiles)."""
        self._f0_plan()
        pack = []
        for width, chunk in self._f0_chunks:
            mats, lengths = [], []
            for layer, start, stop in chunk:
                out, row = self._f0_rows[layer]
                mat = views[self.lcols[layer]].reshape(out, row)[start:stop]
                mats.append(torch.nn.functional.pad(mat, (0, width - row)) if row < width else mat)
                lengths.append(torch.full((stop - start,), row, dtype=torch.long, device=self.dev))
            stats, sums = row_stats(torch.cat(mats), torch.cat(lengths))
            pack.append(torch.cat([stats, sums], dim=1).reshape(-1))
        quart = []
        for layer, col in enumerate(self.lcols):
            if col < 0:
                continue
            index, frac = self._f0_q[layer]
            ordered = views[col].sort().values.index_select(0, index).double().view(3, 2)
            quart.append(ordered[:, 0] + (ordered[:, 1] - ordered[:, 0]) * frac)
        if quart:
            pack.append(torch.stack(quart).reshape(-1))
        return pack

    def _f0_host(self, flat, factor=1.0):
        """Family 0 on the host: [layers, 96] = the layer's 16 statistics, then mean, sd, q10, q50, q90 over its filters."""
        out = np.full((self.Lw, 96), np.nan)
        per_layer = {}
        offset = 0
        for width, chunk in self._f0_chunks:
            rows = sum(stop - start for _, start, stop in chunk)
            block = flat[offset:offset + rows * 20].reshape(rows, 20)
            offset += rows * 20
            at = 0
            for layer, start, stop in chunk:
                per_layer.setdefault(layer, []).append((start, block[at:at + stop - start]))
                at += stop - start
        layers = [layer for layer, col in enumerate(self.lcols) if col >= 0]
        quart = flat[offset:offset + 3 * len(layers)].reshape(len(layers), 3)
        with np.errstate(all="ignore"):
            for k, layer in enumerate(layers):
                block = np.concatenate([part for _, part in sorted(per_layer[layer], key=lambda item: item[0])])
                stats, sums = block[:, :16], block[:, 16:]
                rows, width = self._f0_rows[layer]
                count = float(rows * width)
                m1, m2r, m3r, m4r = (sums[:, i].sum() / count for i in range(4))
                var = m2r - m1 * m1
                third = m3r - 3.0 * m1 * m2r + 2.0 * m1 ** 3
                fourth = m4r - 4.0 * m1 * m3r + 6.0 * m1 * m1 * m2r - 3.0 * m1 ** 4
                flat_layer = not var > 0
                var = max(var, 0.0)
                skew = 0.0 if flat_layer else third / var ** 1.5
                kurt = 0.0 if flat_layer else fourth / var ** 2 - 3.0
                low, high = stats[:, 5].min(), stats[:, 4].max()
                out[layer, :16] = [m1, var, quart[k, 1], math.sqrt(var), high, low, var * count / max(count - 1.0, 1.0),
                                   skew, kurt, low, quart[k, 0], quart[k, 2], quart[k, 1], high, stats[:, 14].sum(),
                                   math.sqrt(sums[:, 1].sum())]
                out[layer, 16:32] = stats.mean(axis=0)
                out[layer, 32:48] = stats.std(axis=0)
                out[layer, 48:96] = np.percentile(stats, [10, 50, 90], axis=0).reshape(-1)
        if factor != 1.0:
            out = out * np.tile(float(factor) ** _F0_POWER, 6)[None, :]
        return out

    def _y4_plan(self):
        """Y4's layers: those whose smaller matrix side is at most Y4_MAX_DIM in the origin (else in the cut)."""
        dims = None
        if self.origin is not None:
            try:
                mods = [m for m in self.origin.modules() if isinstance(m, torch.nn.Conv2d)]
                linears = [m for m in self.origin.modules() if isinstance(m, torch.nn.Linear)]
                mods = mods + ([linears[-1]] if linears and self.head is not None else [])
                if len(mods) == self.Lw:
                    dims = [min(int(m.weight.shape[0]), int(m.weight[0].numel())) for m in mods]
            except Exception:  # noqa: BLE001
                dims = None
        self.y4_by = "origin" if dims is not None else "cut"
        self.y4_layers = []
        for layer, col in enumerate(self.lcols):
            if col < 0:
                continue
            out = int(self.layer_shapes[layer][0])
            width = int(self.sizes[col] // max(out, 1))
            ref = dims[layer] if dims is not None else min(out, width)
            if ref <= Y4_MAX_DIM and min(out, width) >= 1:
                self.y4_layers.append((layer, int(col), out, width, out > width))

    def _y4_device(self, views, with_delta):
        """Gram matrices of each Y4 layer's weight (and of its change from step 0, and the cross matrix a tall layer
        needs), on its smaller side, flattened for the pack."""
        pack = []
        for layer, col, out, width, tall in self.y4_layers:
            w = views[col].reshape(out, width).double()
            w0 = self.T0.views[col].reshape(out, width).double()
            if tall:
                pack.append((w.t() @ w).reshape(-1))
                pack.append((w0.t() @ w).reshape(-1))
            else:
                pack.append((w @ w.t()).reshape(-1))
            if with_delta:
                d = w - w0
                pack.append(((d.t() @ d) if tall else (d @ d.t())).reshape(-1))
        return pack

    def _y4_reference(self):
        """Step 0: every Y4 layer's singular values and vectors (smaller side), kept on the host."""
        self._y4_plan()
        self.y4_ref = {}
        if not self.y4_layers:
            return
        with torch.no_grad():
            flat = torch.cat(self._y4_device(self.T0.views, False)).cpu()
        offset = 0
        for layer, col, out, width, tall in self.y4_layers:
            side = min(out, width)
            gram = flat[offset:offset + side * side].reshape(side, side)
            offset += side * side * (2 if tall else 1)
            got = _eigh(gram, vectors=True)
            if got is None:
                self.y4_ref[layer] = (np.zeros(0), np.zeros((side, 0)))
                continue
            sigma = got[0].clamp_min(0).sqrt().flip(0)
            vec = got[1].flip(1)
            # a Gram matrix resolves singular values down to about 1e-8 of the largest: keep those well above it
            keep = sigma > Y4_RANK_TOL * float(sigma[0]) if float(sigma[0]) > 0 else torch.zeros_like(sigma, dtype=torch.bool)
            self.y4_ref[layer] = (sigma[keep].numpy(), vec[:, keep].numpy())

    def _y4_host(self, flat, with_delta):
        erank, top, intruders = np.full(self.Lw, np.nan), np.full(self.Lw, np.nan), np.full(self.Lw, np.nan)
        offset = 0
        for layer, col, out, width, tall in self.y4_layers:
            side = min(out, width)
            size = side * side
            gram_w = flat[offset:offset + size].reshape(side, side)
            offset += size
            cross = None
            if tall:
                cross = flat[offset:offset + size].reshape(side, side)
                offset += size
            sigma0, vec0 = self.y4_ref[layer]
            if with_delta:
                gram_d = flat[offset:offset + size].reshape(side, side)
                offset += size
                lam_d = _eigh(gram_d)
                if lam_d is not None:
                    sig_d = lam_d.clamp_min(0).sqrt().flip(0).numpy()
                    total = sig_d.sum()
                    if total > 0:
                        q = sig_d[sig_d > 0] / total
                        erank[layer] = math.exp(float(-(q * np.log(q)).sum()))
                    top[layer] = sig_d[0] / sigma0[0] if sigma0.size and sigma0[0] > 0 else np.nan
            else:
                top[layer] = 0.0
            if not sigma0.size:
                continue
            if not with_delta:
                intruders[layer] = 0.0
                continue
            got = _eigh(gram_w, vectors=True)
            if got is None:
                continue
            sig_w = got[0].clamp_min(0).sqrt().flip(0).numpy()
            vec = got[1].flip(1).numpy()
            lead = int(min(10, (sig_w > Y4_RANK_TOL * sig_w[0]).sum())) if sig_w[0] > 0 else 0
            if lead == 0:
                continue
            if tall:
                cosine = np.abs(vec0.T @ cross @ vec[:, :lead]) / (sigma0[:, None] * sig_w[None, :lead])
            else:
                cosine = np.abs(vec0.T @ vec[:, :lead])
            intruders[layer] = float((cosine.max(axis=0) < 0.5).sum())
        return erank, top, intruders

    def _cka_drift(self, live, ref):
        """1 - CKA between each convolution's pooled P_ho output now and at step 0, batched over layers of similar
        width. Returns a device tensor [convs] (NaN where a layer was not read)."""
        out = torch.full((self.Lc,), float("nan"), dtype=torch.float64, device=self.dev)
        layers = [i for i in range(self.Lc) if live[i] is not None and ref[i] is not None
                  and live[i].shape == ref[i].shape and live[i].shape[0] >= 4]
        layers.sort(key=lambda i: int(live[i].shape[1]))
        start = 0
        while start < len(layers):
            n = int(live[layers[start]].shape[0])
            stop, width = start, 0
            while stop < len(layers) and stop - start < 24:
                wide = max(width, int(live[layers[stop]].shape[1]))
                if stop > start and ((stop - start + 1) * n * wide * 8 > CKA_CHUNK_BYTES or live[layers[stop]].shape[0] != n):
                    break
                width, stop = wide, stop + 1
            group = layers[start:stop]
            x = torch.zeros(len(group), n, width, dtype=torch.float64, device=self.dev)
            y = torch.zeros_like(x)
            for k, i in enumerate(group):
                x[k, :, :live[i].shape[1]] = live[i]
                y[k, :, :ref[i].shape[1]] = ref[i]
            out.index_copy_(0, torch.as_tensor(group, dtype=torch.long, device=self.dev), 1.0 - cka_unbiased(x, y))
            start = stop
        return out

    def _fill_ckpt(self, rec, step, col):
        """The checkpoint families of ``rec`` (S R Y V K J F0) at ``step``; closes the gradient window."""
        dev, f64 = self.dev, torch.float64
        k = len(self.ck_steps)
        n_steps = self.n_steps
        w0 = self._win_start
        th = self.prv  # theta at this step
        nan_row = torch.full((self.N,), float("nan"), dtype=f64, device=dev)
        pack = []
        with torch.no_grad():
            n_th = _norms(th.views).double()
            torch.sub(th.buf, self.T0.buf, out=self.X1.buf)
            n_d0 = _norms(self.X1.views).double()
            norm0 = self.norm0.double()
            inv_t = torch.where(n_th > 0, 1.0 / n_th, torch.zeros_like(n_th)).to(self.dtype)
            inv_0 = torch.where(norm0 > 0, 1.0 / norm0, torch.zeros_like(norm0)).to(self.dtype)
            self.X2.main.copy_(th.main * torch.repeat_interleave(inv_t, self.sizes_t)
                               - self.T0.main * torch.repeat_interleave(inv_0, self.sizes_t))
            rot = _norms(self.X2.views).double()
            self.X2.main.copy_(torch.sign(th.main) != torch.sign(self.T0.main))
            s0 = _norms(self.X2.views).double()
            n_d1 = s1 = n_dp = n_sum = same = nan_row
            if k >= 1:
                torch.sub(th.buf, self.C1.buf, out=self.X1.buf)
                n_d1 = _norms(self.X1.views).double()
                self.X2.main.copy_(torch.sign(th.main) != torch.sign(self.C1.main))
                s1 = _norms(self.X2.views).double()
            if k >= 2:
                torch.sub(self.C1.buf, self.C2.buf, out=self.X2.buf)
                n_dp = _norms(self.X2.views).double()
                agree = torch.sign(self.X1.main) == torch.sign(self.X2.main)
                self.X2.buf.add_(self.X1.buf)
                n_sum = _norms(self.X2.views).double()
                self.X2.main.copy_(agree)
                same = _norms(self.X2.views).double()
            n_s = _norms(self.S.views).double()
            pack.append(torch.stack([n_th, n_d0, rot, s0, n_d1, s1, n_dp, n_sum, same, n_s]).reshape(-1))
            # S8: the half of the parameters with the larger magnitude, and the same for the BatchNorm scales
            magnitude = th.main.abs()
            mask = magnitude > magnitude.median()
            ham = (mask != self.mask_prev).double().mean() if self.mask_prev is not None else torch.full((), float("nan"), dtype=f64, device=dev)
            self.mask_prev = mask
            ham_bn = torch.full((), float("nan"), dtype=f64, device=dev)
            if self.bns:
                gamma = torch.cat([m.weight.detach().reshape(-1) for m in self.bns]).double()
                bn_mask = gamma.abs() > gamma.abs().median()
                if self.bn_mask_prev is not None:
                    ham_bn = (bn_mask != self.bn_mask_prev).double().mean()
                self.bn_mask_prev = bn_mask
                pack.append(torch.cat([gamma, torch.cat([m.running_mean.detach().reshape(-1) for m in self.bns]).double(),
                                       torch.cat([m.running_var.detach().reshape(-1) for m in self.bns]).double()]))
            pack.append(torch.stack([ham, ham_bn]))
            # the window's sketch Gram and the projections' centred Gram
            n_win = n_steps - w0
            if n_win >= 1:
                sk = self.gsk[w0:n_steps].double() * self.rows[w0:n_steps, 2].double()[:, None]
                pack.append((sk @ sk.t()).reshape(-1))
            n_proj = len(self.s5_steps)
            proj = self.s5[:n_proj].double()
            proj = proj - proj.mean(dim=0, keepdim=True)
            pack.append((proj @ proj.t()).reshape(-1))
            # Y2
            live = col.pooled_side("ho")
            pack.append(self._cka_drift(live, self.ref_stale))
            pack.append(self._cka_drift(live, self.ref_recal))
            # Y4, the head, family 0
            pack.extend(self._y4_device(th.views, step > 0))
            head_w = head_b = None
            if self.head is not None:
                head_w = self.head.weight.detach().double()
                head_b = self.head.bias.detach().double() if self.head.bias is not None else None
                pack.append(head_w.reshape(-1))
                if head_b is not None:
                    pack.append(head_b.reshape(-1))
            f0_w = self._f0_device(th.views)
            f0_g = self._f0_device(self.G.views) if n_steps > 0 else []
            pack.extend(f0_w)
            pack.extend(f0_g)
            sizes = [int(t.numel()) for t in pack]
            flat = torch.cat([t.reshape(-1).double() for t in pack]).cpu().numpy()
            pen = col.pen_all()
            pen_host = pen[self.n_ho:].cpu().numpy() if pen is not None else None
            # roll the checkpoints and reset the window
            self.C1, self.C2 = self.C2, self.C1
            self.C1.buf.copy_(th.buf)
            self.S.buf.zero_()
        self._win_start = n_steps
        parts, offset = [], 0
        for size in sizes:
            parts.append(flat[offset:offset + size])
            offset += size
        it = iter(parts)
        table = next(it).reshape(10, self.N)
        n_th, n_d0, rot, s0, n_d1, s1, n_dp, n_sum, same, n_s = table
        bn_flat = next(it) if self.bns else None
        ham, ham_bn = next(it)
        gram_win = next(it).reshape(n_win, n_win) if n_win >= 1 else None
        gram_proj = next(it).reshape(n_proj, n_proj)
        drift_stale, drift_recal = next(it), next(it)
        y4_count = sum((2 if tall else 1) + (1 if step > 0 else 0) for *_, tall in self.y4_layers)
        y4_flat = np.concatenate([next(it) for _ in range(y4_count)]) if y4_count else np.zeros(0)
        head_w_host = head_b_host = None
        if head_w is not None:
            head_w_host = next(it).reshape(tuple(head_w.shape))
            if head_b is not None:
                head_b_host = next(it)
        f0_w_host = np.concatenate([next(it) for _ in f0_w]) if f0_w else np.zeros(0)
        f0_g_host = np.concatenate([next(it) for _ in f0_g]) if f0_g else None

        self._pull()
        loss, gnorm, scale, g, b, u = self._series()
        win = slice(w0, n_steps)
        lr = np.asarray([np.nan if v is None else v for v in self.h_lr], dtype=np.float64)
        a = np.asarray(self.h_a, dtype=np.float64)
        norm0 = self.norm0.double().cpu().numpy() if not hasattr(self, "_norm0_host") else self._norm0_host
        self._norm0_host = norm0
        lw = self.lcols >= 0
        arrays = {}
        S, R, Y, V, K, J, F0 = (rec[key] for key in ("S", "R", "Y", "V", "K", "J", "F0"))
        with np.errstate(all="ignore"):
            # --- S, R ---
            dist = math.sqrt(float((n_d0 ** 2).sum()))
            total0 = math.sqrt(float((norm0 ** 2).sum()))
            total = math.sqrt(float((n_th ** 2).sum()))
            S.put("S1_dist", dist)
            arrays["S1_dist"] = self._take(n_d0)
            S.form("S1_L", arrays["S1_dist"])
            lr_sum = float(lr.sum()) if lr.size else 0.0
            S.put("S1_per_lr", dist / lr_sum if lr_sum > 0 and math.isfinite(lr_sum) else None,
                  "no learning rate summed yet")
            rotation = np.where((n_th > 0) & (norm0 > 0), 0.5 * rot ** 2, np.nan)
            arrays["S2_rot"] = self._take(rotation)
            S.form("S2_L", arrays["S2_rot"])
            finite = arrays["S2_rot"][np.isfinite(arrays["S2_rot"])]
            for name, fn in (("S2_mean", np.mean), ("S2_min", np.min), ("S2_sd", np.std)):
                S.put(name, fn(finite) if finite.size else None, "no layer")
            u_glob = np.sqrt((u ** 2).sum(axis=1))
            path = float(u_glob.sum())
            S.put("S3_path", path)
            S.put("S3_direct", dist / path if path > 0 else None, "no step yet")
            if k >= 2:
                dot = 0.5 * (n_sum ** 2 - n_d1 ** 2 - n_dp ** 2)
                for tag, sel in (("w", self.matrix_like), ("o", ~self.matrix_like)):
                    if sel.any():
                        den = math.sqrt(float((n_d1[sel] ** 2).sum()) * float((n_dp[sel] ** 2).sum()))
                        S.put("S4_cos_" + tag, float(np.clip(dot[sel].sum() / den, -1.0, 1.0)) if den > 0 else None,
                              "a zero displacement")
                        S.put("S4_same_" + tag, float(np.round(same[sel] ** 2).sum() / self.sizes_np[sel].sum()))
                    else:
                        S.put("S4_cos_" + tag, None, "no such parameter")
                        S.put("S4_same_" + tag, None, "no such parameter")
            else:
                for name in ("S4_cos_w", "S4_same_w", "S4_cos_o", "S4_same_o"):
                    S.put(name, None, "needs three checkpoints")
            S["S5_n"] = int(n_proj)
            lam = _eigh(gram_proj) if n_proj >= 3 else None
            lam = None if lam is None else lam.clamp_min(0).flip(0).numpy()
            if lam is not None and lam.sum() > 0:
                share = np.cumsum(lam) / lam.sum()
                S.put("S5_pc1", share[0])
                S.put("S5_pc2", share[1])
                S.put("S5_pc3", share[2])
                S.put("S5_pr", lam.sum() ** 2 / (lam * lam).sum())
            else:
                for name in ("S5_pc1", "S5_pc2", "S5_pc3", "S5_pr"):
                    S.put(name, None, "fewer than three projections" if n_proj < 3 else "no movement")
            self.ck_steps.append(int(step))
            self.ck_dist.append(dist)
            xs = np.asarray([s for s, d in zip(self.ck_steps, self.ck_dist) if s > 0 and d > 0], dtype=np.float64)
            ys = np.asarray([d for s, d in zip(self.ck_steps, self.ck_dist) if s > 0 and d > 0], dtype=np.float64)
            S.put("S6_slope", np.polyfit(np.log(xs), np.log(ys), 1)[0] if xs.size >= 2 else None,
                  "fewer than two checkpoints after step 0")
            flips0 = np.round(s0 ** 2)
            S.put("S7_sign0", flips0.sum() / self.P)
            arrays["S7_sign0"] = self._take(flips0 / self.sizes_np)
            S.form("S7_sign0_L", arrays["S7_sign0"])
            if k >= 1:
                flips1 = np.round(s1 ** 2)
                S.put("S7_signprev", flips1.sum() / self.P)
                arrays["S7_signprev"] = self._take(flips1 / self.sizes_np)
                S.form("S7_signprev_L", arrays["S7_signprev"])
                dlog = math.log1p(step) - math.log1p(self.ck_prev["step"])
                growth = (np.log(n_th) - np.log(self.ck_prev["n_th"])) / dlog if dlog > 0 else np.full(self.N, np.nan)
                arrays["S9_growth"] = self._take(growth)
                S.form("S9_L", arrays["S9_growth"])
                S.put("S9_all", (math.log(total) - math.log(self.ck_prev["total"])) / dlog
                      if dlog > 0 and total > 0 and self.ck_prev["total"] > 0 else None, "no log-step or a zero norm")
            else:
                S.put("S7_signprev", None, "first checkpoint")
                S.form("S7_signprev_L", None, "first checkpoint")
                S.form("S9_L", None, "first checkpoint")
                S.put("S9_all", None, "first checkpoint")
            S.put("S8_mask", ham if k >= 1 else None, "first checkpoint")
            S.put("S8_bnmask", ham_bn if (k >= 1 and self.bns) else None, "first checkpoint" if self.bns else "no BatchNorm")
            rel = np.where(norm0 > 0, n_d0 / norm0, np.nan)
            arrays["R3_rel"] = self._take(rel)
            R.form("R3_L", arrays["R3_rel"])
            R.put("R3_all", dist / total0 if total0 > 0 else None, "zero weights")
            # --- Y1, Y2, Y3, Y4, Y5 ---
            if self.head is not None and self.Lc >= 1:
                conv_r3, conv_s2 = arrays["R3_rel"][:self.Lc], arrays["S2_rot"][:self.Lc]
                r3_mean, s2_mean = np.nanmean(conv_r3), np.nanmean(conv_s2)
                Y.put("Y1_head_r3_ratio", arrays["R3_rel"][-1] / r3_mean if r3_mean > 0 else None, "no movement yet")
                Y.put("Y1_head_s2_ratio", arrays["S2_rot"][-1] / s2_mean if s2_mean > 0 else None, "no movement yet")
            else:
                Y.put("Y1_head_r3_ratio", None, "no final Linear or no convolution")
                Y.put("Y1_head_s2_ratio", None, "no final Linear or no convolution")
            arrays["Y2_cka_stale"], arrays["Y2_cka_recal"] = drift_stale, drift_recal
            Y.form("Y2_stale_L", drift_stale)
            Y.form("Y2_recal_L", drift_recal)
            pre = g * scale[:, None]  # pre-clip per-tensor gradient norms
            if n_win >= 1:
                mean_g = pre[win].mean(axis=0)
                arrays["Y3_gnorm"] = self._take(mean_g)
                Y.form("Y3_L", arrays["Y3_gnorm"])
                Y.put("Y3_ratio", arrays["Y3_gnorm"][0] / arrays["Y3_gnorm"][-1] if self.Lw else None, "no layer")
            else:
                Y.form("Y3_L", None, "no step in the window")
                Y.put("Y3_ratio", None, "no step in the window")
            if self.y4_layers:
                y4_erank, y4_top, y4_intr = self._y4_host(y4_flat, step > 0)
                arrays["Y4_erank"], arrays["Y4_top"], arrays["Y4_intruders"] = y4_erank, y4_top, y4_intr
                Y.form("Y4_erank_L", y4_erank, "no movement at step 0")
                Y.form("Y4_top_L", y4_top)
                Y.form("Y4_intr_L", y4_intr)
                Y.put("Y4_intr_sum", np.nansum(y4_intr) if np.isfinite(y4_intr).any() else None, "no layer")
                Y["Y4_layers"] = len(self.y4_layers)
            else:
                Y.miss("Y4", "no layer within the %d cap" % Y4_MAX_DIM)
            ho, tr = self._prev[1], self._prev[2]
            if self.y5 is not None and head_w_host is not None and head_w_host.shape == self.y5["weight"].shape:
                den = math.sqrt(float((head_w_host ** 2).sum()) * float((self.y5["weight"] ** 2).sum()))
                Y.put("Y5_cos", float((head_w_host * self.y5["weight"]).sum() / den) if den > 0 else None, "zero head")
                Y.put("Y5_gap", float(ho.correct.mean()) - self.y5["acc_ho"])
                Y.put("Y5_probe_acc_ho", self.y5["acc_ho"])
            else:
                Y.miss("Y5", "no linear probe")
            # --- V ---
            V["n"] = int(n_win)
            w_layer = self._take(n_th)
            if n_win >= 1:
                arrays["V1_upd_ratio"] = self._take(u[win].mean(axis=0)) / w_layer
                V.form("V1_L", arrays["V1_upd_ratio"])
                V.put("V1_all", u_glob[win].mean() / total if total > 0 else None, "zero weights")
                rows = np.arange(max(w0, 1), n_steps)
                if rows.size:
                    dot = (b[rows] ** 2 - u[rows - 1] ** 2 - (a[rows, None] * g[rows]) ** 2) / (2.0 * a[rows, None])
                    cosine = np.clip(dot / (u[rows - 1] * g[rows]), -1.0, 1.0)
                    arrays["V2_cos"] = self._take(np.nanmean(np.where(np.isfinite(cosine), cosine, np.nan), axis=0))
                    V.form("V2_L", arrays["V2_cos"])
                    den = np.sqrt((u[rows - 1] ** 2).sum(axis=1) * (g[rows] ** 2).sum(axis=1))
                    overall = np.clip(dot.sum(axis=1) / den, -1.0, 1.0)
                    V.put("V2_all", np.nanmean(overall) if np.isfinite(overall).any() else None, "a zero update or gradient")
                else:
                    V.form("V2_L", None, "no previous update in the window")
                    V.put("V2_all", None, "no previous update in the window")
                arrays["V5_grad_ratio"] = arrays["Y3_gnorm"] / w_layer
                V.form("V5_L", arrays["V5_grad_ratio"])
                V.put("V5_ratio", arrays["V5_grad_ratio"][0] / arrays["V5_grad_ratio"][-1] if self.Lw else None, "no layer")
                last_lr = lr[n_steps - 1]
                if self.momentum is not None and math.isfinite(last_lr) and last_lr > 0:
                    arrays["V5m_buf_ratio"] = (1.0 - self.momentum) * self._take(u[n_steps - 1]) / last_lr / w_layer
                    V.form("V5m_L", arrays["V5m_buf_ratio"])
                else:
                    V.form("V5m_L", None, "momentum or learning rate unknown")
                if self.weight_decay is not None:
                    arrays["V8_wd_ratio"] = self.weight_decay * w_layer / arrays["Y3_gnorm"]
                    V.form("V8_L", arrays["V8_wd_ratio"])
                else:
                    V.form("V8_L", None, "weight decay unknown")
            else:
                for name in ("V1_L", "V2_L", "V5_L", "V5m_L", "V8_L"):
                    V.form(name, None, "no step in the window")
                for name in ("V1_all", "V2_all", "V5_ratio"):
                    V.put(name, None, "no step in the window")
            if self.bns:
                edges = np.concatenate([[0], np.cumsum(self.bn_sizes)])
                count = int(edges[-1])
                gamma, run_mean, run_var = bn_flat[:count], bn_flat[count:2 * count], bn_flat[2 * count:]
                starts = edges[:-1]
                sizes_bn = np.asarray(self.bn_sizes, dtype=np.float64)
                g_mean = np.add.reduceat(gamma, starts) / sizes_bn
                g_sd = np.sqrt(np.maximum(np.add.reduceat(gamma * gamma, starts) / sizes_bn - g_mean ** 2, 0.0))
                small = np.add.reduceat((np.abs(gamma) < 0.01).astype(np.float64), starts)
                arrays["V7_gamma_mean"], arrays["V7_gamma_sd"], arrays["V7_gamma_small"] = g_mean, g_sd, small / sizes_bn
                V.form("V7_mean_L", g_mean)
                V.form("V7_sd_L", g_sd)
                V.form("V7_small_L", small / sizes_bn)
                V.put("V7_small_all", small.sum() / count)
                if self.ck_prev is not None and self.ck_prev.get("bn") is not None:
                    old_mean, old_var = self.ck_prev["bn"]
                    rm = np.sqrt(np.add.reduceat((run_mean - old_mean) ** 2, starts) / np.add.reduceat(old_mean ** 2, starts))
                    rv = np.sqrt(np.add.reduceat((run_var - old_var) ** 2, starts) / np.add.reduceat(old_var ** 2, starts))
                    arrays["V7_rm_change"], arrays["V7_rv_change"] = rm, rv
                    V.form("V7_rm_L", rm)
                    V.form("V7_rv_L", rv)
                else:
                    V.form("V7_rm_L", None, "first checkpoint")
                    V.form("V7_rv_L", None, "first checkpoint")
                bn_now = (run_mean.copy(), run_var.copy())
            else:
                V.miss("V7", "no BatchNorm")
                bn_now = None
            # --- K, J ---
            K["n"], J["n"] = int(n_win), int(n_win)
            if n_win >= 2:
                q = pre[win] ** 2
                sum2, step2 = n_s ** 2, q.sum(axis=0)
                all_sum2, all_step2 = float(sum2.sum()), float(step2.sum())
                K.put("K1", float(k1_snr(all_sum2, all_step2, n_win)))
                arrays["K1_snr"] = self._take(k1_snr(sum2, step2, n_win))
                K.form("K1_L", arrays["K1_snr"])
                mean2, gbar2 = all_step2 / n_win, all_sum2 / n_win ** 2
                if self.batch_size:
                    small_b, big_b = float(self.batch_size), float(self.batch_size) * n_win
                    g2 = (big_b * gbar2 - small_b * mean2) / (big_b - small_b)
                    tr_sigma = (mean2 - gbar2) / (1.0 / small_b - 1.0 / big_b)
                    ok = g2 > 0 and tr_sigma > 0
                    K.put("K2_logB", math.log(tr_sigma / g2) if ok else None, "non-positive |G|^2 or tr(Sigma) estimate")
                    K.put("K2_ratio", tr_sigma / g2 / small_b if ok else None, "non-positive |G|^2 or tr(Sigma) estimate")
                else:
                    K.put("K2_logB", None, "batch size unknown")
                    K.put("K2_ratio", None, "batch size unknown")
                K.put("K3mb_alpha", gbar2 / mean2 if mean2 > 0 else None, "zero gradients")
                arrays["K3_alpha"] = self._take(np.where(step2 > 0, sum2 / n_win ** 2 / (step2 / n_win), np.nan))
                K.form("K3mb_alpha_L", arrays["K3_alpha"])
                K.put("K4_div", all_step2 / all_sum2 if all_sum2 > 0 else None, "zero mean gradient")
                arrays["K4_diversity"] = self._take(np.where(sum2 > 0, step2 / sum2, np.nan))
                K.form("K4_div_L", arrays["K4_diversity"])
                diag = np.diag(gram_win)
                upper = np.triu_indices(n_win, 1)
                cosines = (gram_win / np.sqrt(np.outer(diag, diag)))[upper]
                if np.isfinite(cosines).all():
                    K.put("K3mb_cbar", 2.0 * cosines.sum() / (n_win * (n_win - 1)))
                    K.put("K4_cos_min", cosines.min())
                    K.put("K4_cos_p05", np.percentile(cosines, 5))
                else:
                    for name in ("K3mb_cbar", "K4_cos_min", "K4_cos_p05"):
                        K.put(name, None, "a zero gradient in the window")
                gaps = np.sqrt(np.maximum(diag[:, None] + diag[None, :] - 2.0 * gram_win, 0.0))[upper]
                K.put("K7_disp", gaps.mean())
                K.put("K7_disp_rel", gaps.mean() / math.sqrt(gbar2) if gbar2 > 0 else None, "zero mean gradient")
                centred = gram_win - gram_win.mean(axis=0, keepdims=True)
                centred = centred - centred.mean(axis=1, keepdims=True)
                eig = _eigh(centred / n_win)
                eig = None if eig is None else eig.clamp_min(0).numpy()
                K.put("K8_top", None if eig is None else eig.max(), "non-finite gradients")
                K.put("K8_trace", (all_step2 - all_sum2 / n_win) / n_win)
                K.put("K8_trace_sk", None if eig is None else eig.sum(), "non-finite gradients")
                K.put("K8_pr", eig.sum() ** 2 / (eig * eig).sum() if eig is not None and eig.sum() > 0 else None, "no spread")
            else:
                why = "no step in the window" if n_win == 0 else "a window of one step"
                for name in ("K1", "K2_logB", "K2_ratio", "K3mb_alpha", "K3mb_cbar", "K4_div", "K4_cos_min",
                             "K4_cos_p05", "K7_disp", "K7_disp_rel", "K8_top", "K8_trace", "K8_trace_sk", "K8_pr"):
                    K.put(name, None, why)
                for name in ("K1_L", "K3mb_alpha_L", "K4_div_L"):
                    K.form(name, None, why)
            if pen_host is not None and head_w_host is not None:
                e_norm, grand, gamma_i, k10 = head_gradient(tr, pen_host, head_b_host, math.sqrt(float((head_w_host ** 2).sum())))
                self._head_now = (e_norm, grand)
                mean, sd, _skew, kurt = _moments(gamma_i) if np.isfinite(gamma_i).all() else (np.nan,) * 4
                K.put("K9_mean", mean)
                K.put("K9_sd", sd)
                K.put("K9_kurt", kurt)
                K.put("K9_neg", (gamma_i < 0).mean() if np.isfinite(gamma_i).all() else np.nan)
                K.put("K9_gwa", mean / (kurt + 1.2) if abs(kurt + 1.2) > 1e-12 else None, "kurtosis + 1.2 is zero")
                for name, value in zip(("K10", "K10_w", "K10_h"), k10):
                    K.put(name, value)
            else:
                self._head_now = None
                K.miss("K9", "no final Linear")
                K.miss("K10", "no final Linear")
            if n_win >= 1:
                q = pre[win] ** 2
                j_layers = q.mean(axis=0)
                j_all = float(j_layers.sum())
                J.put("J4emp", j_all)
                arrays["J4emp"] = self._take(j_layers)
                J.form("J4emp_L", arrays["J4emp"])
                self.j_hist.append((int(step), j_all))
                best = max(self.j_hist, key=lambda item: item[1])
                J.put("J15_max", best[1])
                J["J15_step"] = int(best[0])
                J.put("J15_ratio", j_all / best[1] if best[1] > 0 else None, "zero gradients")
            else:
                J.put("J4emp", None, "no step in the window")
                J.form("J4emp_L", None, "no step in the window")
            # --- family 0 ---
            f0 = {"w": self._f0_host(f0_w_host) if self.Lw and lw.any() else None,
                  "g": self._f0_host(f0_g_host, float(scale[n_steps - 1])) if f0_g_host is not None and lw.any() else None}
            for kind in ("w", "g"):
                sub = _Fam()
                if f0[kind] is None:
                    sub.miss("F0", "no weight layer" if kind == "w" or n_steps else "no training step before step 0")
                else:
                    for name, column in _F0_EVENT:
                        sub.form(name + "_L", f0[kind][:, column])
                F0[kind] = sub
        self.ck_prev = {"step": int(step), "n_th": n_th.copy(), "total": total, "bn": bn_now}
        if not hasattr(self, "ck_arrays"):
            self.ck_arrays, self.ck_f0 = [], []
        self.ck_arrays.append(arrays)
        self.ck_f0.append(f0)
        rec["ckpt"] = True

    # --- on_end -------------------------------------------------------------------------------------------------------

    def _end(self, step):
        t0 = time.perf_counter()
        if step != self.n_steps:
            raise RuntimeError("on_end(%d) after %d logged steps" % (step, self.n_steps))
        mark = self._mem_begin()
        pending = self._pending
        with _RngGuard():
            if pending is None or pending["step"] != step:
                self._read(step, True)
            elif not pending.get("ckpt"):
                col = self._live_read("drift")  # the last step is always a checkpoint: add its families to the read
                self._fill_ckpt(pending, step, col)
            rec = self._pending
            self._last_families(rec)
            if pending is rec:
                rec["seconds"] = round(float(rec["seconds"]) + time.perf_counter() - t0, 3)
            self._flush()
            self._mem_end(mark)
            self._write_file(step)
        seconds = time.perf_counter() - t0
        held = self._held_bytes()
        self._say("[eval] FT_NAP %s %s end step=%d | %d reads | file %.2f MB | mem held %.0f MB peak %.0f MB | "
                  "step %.2f ms | start %.1f s reads %.1f s | %.1f s" % (
                      self.label, self.name, step, len(self.records), self.file_bytes / 2 ** 20, held / 2 ** 20,
                      max(self.mem_peak, held) / 2 ** 20, 1e3 * self.t_step / max(self.n_steps, 1), self.t_start,
                      self.t_reads, seconds))
        self.done = True
        self._release()

    def _last_families(self, rec):
        """E (E1 E2 E5 E6 E7) and H (H5 and its change from step 0) of the last step's event."""
        E, H = rec["E"], rec["H"]
        hist = self.hist
        steps = np.asarray(hist["steps"], dtype=np.float64)
        correct, margin, py = np.stack(hist["correct"]), np.stack(hist["margin"]), np.stack(hist["py"])
        n_ho = self.n_ho
        E["reads"] = int(steps.size)
        for tag, sel in (("ho", slice(0, n_ho)), ("tr", slice(n_ho, None))):
            E[tag] = e_history(steps, correct[:, sel], margin[:, sel], py[:, sel])
        tr = self._prev[2]
        with np.errstate(all="ignore"):
            if getattr(self, "_head_now", None) is not None:
                for name, values in (("E5_el2n", self._head_now[0]), ("E5_grand", self._head_now[1])):
                    mean, sd, skew, _kurt = _moments(values) if np.isfinite(values).all() else (np.nan,) * 4
                    E.put(name + "_mean", mean)
                    E.put(name + "_sd", sd)
                    E.put(name + "_skew", skew)
                    E.put(name + "_p90", np.percentile(values, 90) if np.isfinite(values).all() else np.nan)
            else:
                E.miss("E5", "no final Linear")
            mixture = gmm2(tr.ce)
            E.put("E7_weight_high", None if mixture is None else mixture[0], "degenerate losses")
            E.put("E7_separation", None if mixture is None else mixture[1], "degenerate losses")
        got = None
        if self._pen_dev is not None:
            got = h5_forecast(self._pen_dev, self.y_all_dev, self.n_classes, self.n_ho)
        if got is None:
            H.miss("H5", "no final Linear" if self._pen_dev is None else "empty kernel")
            return
        _put_h5(H, got)
        for name in ("stale", "recal"):
            base = self.h5_0.get(name)
            H.put("R7_loo_acc_" + name, got["loo_acc"] - base["loo_acc"] if base is not None else None,
                  "no H5 at step 0")

    def _write_file(self, step):
        """The array file (module docstring). Finite numbers only; ``<name>__ok`` masks mark what is missing."""
        out_dir = self.out_dir() if callable(self.out_dir) else self.out_dir
        if not out_dir:
            self.notes.append("no array file: no output directory")
            return
        os.makedirs(out_dir, exist_ok=True)
        safe = "".join(ch if (ch.isalnum() or ch in "._+-=") else "_" for ch in self.label)
        path = os.path.join(out_dir, "%s__%s.npz" % (safe, self.name))
        arrays, masked = {}, []

        def add(name, values, dtype=np.float32):
            values = np.asarray(values, dtype=np.float64)
            ok = np.isfinite(values)
            if not ok.all():
                arrays[name + "__ok"] = ok.astype(np.uint8)
                masked.append(name)
                values = np.where(ok, values, 0.0)
            limit = float(np.finfo(dtype).max)
            arrays[name] = np.clip(values, -limit, limit).astype(dtype)

        self._pull()
        loss, gnorm, scale, g, _b, u = self._series()
        lr = np.asarray([np.nan if v is None else v for v in self.h_lr], dtype=np.float64)
        add("step_series", np.stack([loss, lr, gnorm, np.sqrt((u ** 2).sum(axis=1)), scale], axis=1)
            if loss.size else np.zeros((0, len(STEP_COLS))))
        cols = self.lcols[self.lcols >= 0]
        add("step_layer_gnorm", (g * scale[:, None])[:, cols])
        add("step_layer_unorm", u[:, cols])
        hist = self.hist
        logits = np.stack(hist["logits"]).astype(np.float64)
        add("probe_logits", logits, np.float16)
        arrays["probe_steps"] = np.asarray(hist["steps"], dtype=np.int32)
        arrays["probe_labels"] = self.y_all.astype(np.int16 if self.n_classes < 32768 else np.int32)
        arrays["probe_index_ho"] = np.asarray(self._probe_meta["idx_ho"], dtype=np.int64)
        arrays["probe_index_tr"] = np.asarray(self._probe_meta["idx_tr"][:self.n_tr], dtype=np.int64)
        arrays["probe_base_index_ho"] = np.asarray(self._probe_meta["base_ho"], dtype=np.int64)
        arrays["probe_base_index_tr"] = np.asarray(self._probe_meta["base_tr"][:self.n_tr], dtype=np.int64)
        add("probe_logits_recal0", self.logits_recal0, np.float16)
        if self.origin_logits is not None:
            add("probe_logits_origin", self.origin_logits, np.float16)
        arrays["ckpt_steps"] = np.asarray(self.ck_steps, dtype=np.int32)
        names = sorted(set(name for item in self.ck_arrays for name in item))
        for name in names:
            width = max(int(np.asarray(item[name]).size) for item in self.ck_arrays if name in item)
            table = np.full((len(self.ck_arrays), width), np.nan)
            for row, item in enumerate(self.ck_arrays):
                if name in item:
                    table[row, :int(np.asarray(item[name]).size)] = np.asarray(item[name], dtype=np.float64)
            add("layer_" + name, table)
        for kind in ("w", "g"):
            table = np.full((len(self.ck_f0), self.Lw, 96), np.nan)
            for row, item in enumerate(self.ck_f0):
                if item[kind] is not None:
                    table[row] = item[kind]
            add("f0_" + kind, table)
        for name, values in sorted(self.cut_arrays.items()):
            add("cut_" + name, values)
        n_proj = len(self.s5_steps)
        add("s5_proj", self.s5[:n_proj].double().cpu().numpy())
        arrays["s5_steps"] = np.asarray(self.s5_steps, dtype=np.int32)
        meta = {
            "schema": 1, "network": self.network, "label": self.label, "dataset": self.dataset,
            "n_classes": int(self.n_classes), "steps": int(step), "n_ho": int(self.n_ho), "n_tr": int(self.n_tr),
            "probe_rows": "rows [0, n_ho) are P_ho, the rest P_tr; probe_index_* are positions in the val / train "
                          "dataset, probe_base_index_* in its base dataset",
            "probe_tr_read": self._probe_meta["how_tr"],
            "read_steps": sorted(self.read_set), "ckpt_steps": sorted(self.ckpt_set), "proj_every": PROJ_EVERY,
            "step_cols": list(STEP_COLS),
            "step_missing": {"mb_acc": "the hook does not pass the logits", "mb_maxprob": "the hook does not pass the logits",
                             "mb_margin": "the hook does not pass the logits"},
            "layers": self.layer_names, "layer_shapes": [list(s) for s in self.layer_shapes],
            "layers_note": "weight layers: every Conv2d in module order, then the final Linear; layer_* arrays are "
                           "[checkpoints, layers] (Y2_*: convolutions only; V7_*: BatchNorm layers)",
            "f0": {"stats": list(F0_STATS), "groups": list(F0_GROUPS),
                   "note": "f0_w / f0_g are [checkpoints, layers, 96]: 6 groups of 16; g is the pre-clip gradient of the "
                           "checkpoint's own step"},
            "cut_note": "cut_<state>_{q,c,d01,d025,dead} and cut_z33_gnorm: per convolution; cut_<state>_{align,erank,srank,pr}: "
                        "per convolution, then the penultimate vector; cut_z4_*: weight layers; cut_bn_shift_*: BatchNorm "
                        "layers with running statistics; <state> is stale or recal",
            "form16": list(FORM16), "masked": masked,
            "missing_note": "an array is finite everywhere; where <name>__ok exists, 0 marks a missing value stored as 0",
            "sketch": {"dim": SKETCH_DIM, "block": SKETCH_BLOCK, "of": "theta_t - theta_0",
                       "seed": seed_of(self.dataset, self.name, "S5")},
            "y4": {"max_dim": Y4_MAX_DIM, "by": self.y4_by, "layers": [item[0] for item in self.y4_layers]},
            "recipe": {"batch_size": self.batch_size, "momentum": self.momentum, "weight_decay": self.weight_decay},
            "params": int(self.P), "tensors": int(self.N), "notes": self.notes,
            "seconds": {"start": round(self.t_start, 3), "reads": round(self.t_reads, 3), "steps": round(self.t_step, 3)},
            "torch": torch.__version__,
        }
        arrays["meta"] = np.asarray(json.dumps(meta))
        self.file_bytes = _write_npz(path, arrays, stored=("probe_logits", "s5_proj", "probe_logits_recal0",
                                                           "probe_logits_origin"))
        self.file_path = path


def _put_h5(family, result):
    for key, value in result.items():
        if key in ("n", "d"):
            family[key] = int(value)
        else:
            family.put(key, value)


def _fmt(value):
    return "-" if value is None else "%.4f" % value
