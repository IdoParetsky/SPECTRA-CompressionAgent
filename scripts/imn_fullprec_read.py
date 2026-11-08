"""
IMN-FP (queue row ``IMN-FP``): a full-precision, per-image re-read of the eight ImageNet ResNet-50 networks of
ledger 427-428, with paired differences. EVALUATION ONLY: no training, no new arm, no new point.

    python scripts/imn_fullprec_read.py --out <dir> [--cells-root /home/paretsky/scratch_audit/imn_cells]
        [--cells original,origin-0,...] [--batch 250] [--workers 8] [--limit N] [--device cuda] [--force]

Networks (fixed order): ``original`` (the torchvision ResNet-50 the pipeline loads from
``resnet50_imagenet1kv1_torchvision_76.13_25.56.pth``), ``origin-0`` (the unpruned control after the 90-epoch
recovery), ``t4f-48 / sensF-48 / uniF-48`` and ``t4f-56 / sensF-56 / uniF-56`` (agent, rule, even cut at FLOPs
0.48 and 0.56). Each recovered network is the single ``<cells-root>/run-<cell>/recovery__*.pt`` (written by
``src/imagenet_fast.py:save_checkpoint``), rebuilt strictly from its saved arch and weights.

What is mirrored (file:line as of tree_v34)
-------------------------------------------
* Original network: ``src/utils.py:parse_input_argument`` (480) -> ``instantiate_networks_and_load_datasets``
  (521-566) -> ``load_model_from_script(arch, dataset, script, net_path, {}, num_classes, input_shape)`` (738-860),
  with the config entry ``[arch, script, dataset]`` of ``configs/input_imagenet_tv_resnet50_v1.json``,
  ``num_classes = utils.infer_num_classes(train_split)`` (436) and ``input_shape`` the (C, H, W) of one test sample
  (``utils.get_input_shape``, 2115, reads the same shape from a loader batch).
* Recovered networks: ``src/traj_models.py:rebuild_from_state_dict`` (73) on ``copy.deepcopy(template).cpu()``, as
  ``a2c_agent_reinforce_runner.py:_rebuild_origin`` (456-500) does for a saved model; the checkpoint is checked
  against ``imagenet_fast.CKPT_VERSION`` (67) and the final state (epoch 90, step 0, recipe epochs 90).
* Data: ``utils.load_cnn_dataset`` (1381), branch ``SPECTRA_VAL_FROM_TRAIN`` (1406-1409) ->
  ``utils._val_from_train_datasets(name_or_path, options, canonical, 10)`` (1552-1574) with ``name_or_path`` =
  "imagenet1k", ``options`` = {} (``utils.parse_dataset_spec``, 1197), canonical "imagenet1k", per_class 10, split
  seed ``SPECTRA_SPLIT_SEED`` unset = 0 (``_class_balanced_holdout``, 124). TEST = the official 50,000-image val
  (``_large_spatial_split``, 1180), val = the 10,000 held-out train images; both with the eval transform
  ``utils.build_transform(..., train=False)`` (1218): Resize(256), CenterCrop(224), ToTensor, ImageNet mean / std.
  ``shuffle=False`` loaders as ``load_cnn_dataset`` builds them (1546-1547).
* The TEST / val read (the number printed as ``TRAJ final_ft ... | acc 0.761 -> 0.752``):
  ``a2c_agent_reinforce_runner.py:_final_ft_imagenet`` (235-293; reads at 258-262) calls
  ``ClassificationHandler.evaluate_model(loader)`` (``src/ModelHandlers/ClassificationHandler.py`` 134-181):
  ``model.eval()``, ``torch.no_grad()``, batches as loaded (NCHW; the batch is channels_last only under
  ``SPECTRA_CHANNELS_LAST``, which the IMN submit script does not set), ``torch.cuda.amp.autocast(enabled=
  SPECTRA_AMP)`` (fp16 autocast; the runs had ``SPECTRA_AMP=1``), prediction ``torch.argmax(logits, 1)``. The
  recovered model reaches that read with channels_last WEIGHTS (``imagenet_fast.recover`` 1135-1136 converts the
  model and leaves it so, 1268); the ``original`` is read as ``NetworkEnv.score_test_loader`` (498-507) reads the
  origin (TEST) and ``NetworkEnv.reset`` (401-403) reads it for val: the loaded model, contiguous weights. So the
  default ``--channels-last auto`` gives channels_last weights to the seven recovered networks, contiguous weights
  to ``original``, and NCHW inputs to all.
* Determinism and numerics of the process: ``SPECTRA_EVAL_FT_DETERMINISTIC=1`` through
  ``fortify.apply_ft_determinism`` (772-799; runner 1189) before the first CUDA use (``--no-deterministic`` turns it
  off, then ``cudnn.benchmark = True`` as the runner 1208-1209), ``torch.set_float32_matmul_precision("high")``
  (runner 1201). ``SPECTRA_EVAL_DETERMINISTIC`` only changes the walk's policy and has no effect here.
* The runs' top-5 numbers (``eval_topk`` / ``utils.evaluate_topk``, 2090) were read in fp32 without autocast; here
  top-1 and top-5 come from the one autocast pass that mirrors the TEST / val read, so top-5 is the same pass as
  top-1 (the top-1 of a pass is the registered check).

Per image: ``pred1 = argmax(logits)`` (as the runs), ``top5`` = membership in ``topk(logits, 5)`` OR'd with
``pred1 == y`` (so top-1 is always inside top-5 whatever the fp16 tie order of ``topk``), ``top1 = pred1 == y``.

Outputs under ``--out``: ``<cell>__test.npz`` and ``<cell>__val.npz`` (keys top1 uint8, top5 uint8, pred1 int16),
``targets__test.npy`` and ``targets__val.npy`` (int16), ``<cell>__meta.json``, ``summary.json``, ``summary.md``.
A network whose two npz files exist (and have the right length) is skipped unless ``--force`` (resume after
preemption). Exit 0 if every check passes (or under ``--limit``), 3 if any network is NOT-REPRODUCED.

The statistics (``paired_counts``, ``mcnemar_exact_p``, ``paired_diff_ci``, ``check_test``, ``check_val``) are pure
numpy / math and import without torch; torch, torchvision and ``src.*`` are imported lazily inside the functions
that need them.
"""

import argparse
import copy
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
if not sys.path or sys.path[0] != ROOT:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------------------------------------------
# Registered constants (queue row IMN-FP)
# ---------------------------------------------------------------------------------------------------------------
CELLS = ("original", "origin-0", "t4f-48", "sensF-48", "uniF-48", "t4f-56", "sensF-56", "uniF-56")
RECOVERED = CELLS[1:]
LOGGED_TEST_TOP1 = {  # fractions, three decimals as logged (ledger 427-428)
    "original": 0.761, "origin-0": 0.769,
    "t4f-48": 0.752, "sensF-48": 0.753, "uniF-48": 0.754,
    "t4f-56": 0.757, "sensF-56": 0.759, "uniF-56": 0.757,
}
LOGGED_VAL_CHANGE_PP = {  # val change from the original, pp, two decimals as logged
    "origin-0": -6.48,
    "t4f-48": -8.73, "sensF-48": -8.90, "uniF-48": -8.59,
    "t4f-56": -8.43, "sensF-56": -8.08, "uniF-56": -8.09,
}
TEST_TOL = 0.0006
VAL_TOL_PP = 0.02
Z95 = 1.959963984540054
SPLITS = ("test", "val")
POINTS = ("48", "56")
PAIRS = (  # (label, a-arm, b-arm): a - b
    ("agent - rule", "t4f", "sensF"),
    ("agent - even cut", "t4f", "uniF"),
    ("rule - even cut", "sensF", "uniF"),
)
ARMS = ("t4f", "sensF", "uniF")
DEFAULT_CELLS_ROOT = "/home/paretsky/scratch_audit/imn_cells"
DEFAULT_CONFIG = "configs/input_imagenet_tv_resnet50_v1.json"
EXPECTED_N = {"test": 50000, "val": 10000}
VAL_PER_CLASS = 10
REPORT_HEADER = ("Report only (queue row IMN-FP). No call (one network, one seed). The paired interval measures "
                 "the TEST set's sampling, not seed-to-seed variance.")


# ---------------------------------------------------------------------------------------------------------------
# Statistics (pure numpy / math)
# ---------------------------------------------------------------------------------------------------------------
def paired_counts(a, b):
    """``{n, a_only, b_only, both, neither}`` of two equal-length correctness arrays (a right / b wrong = a_only)."""
    a = np.asarray(a).astype(bool).reshape(-1)
    b = np.asarray(b).astype(bool).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("paired_counts: length mismatch (%d vs %d)" % (a.shape[0], b.shape[0]))
    return {
        "n": int(a.shape[0]),
        "a_only": int(np.count_nonzero(a & ~b)),
        "b_only": int(np.count_nonzero(~a & b)),
        "both": int(np.count_nonzero(a & b)),
        "neither": int(np.count_nonzero(~a & ~b)),
    }


def mcnemar_exact_p(a_only, b_only):
    """
    Two-sided exact McNemar p: ``min(1, 2 * P(X <= min(a_only, b_only)))``, ``X ~ Bin(a_only + b_only, 0.5)``;
    1.0 when both counts are 0. Computed in log space (lgamma + log-sum-exp), so totals of thousands do not underflow.
    """
    a_only, b_only = int(a_only), int(b_only)
    if a_only < 0 or b_only < 0:
        raise ValueError("mcnemar_exact_p: negative count")
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    log_half_n = n * math.log(0.5)
    log_n_fact = math.lgamma(n + 1)
    logs = [log_n_fact - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n for i in range(k + 1)]
    top = max(logs)
    log_tail = top + math.log(sum(math.exp(v - top) for v in logs))
    return float(min(1.0, 2.0 * math.exp(log_tail)))


def paired_diff_ci(a, b, z=Z95):
    """``(diff_pp, lo_pp, hi_pp)``: ``d = (a_only - b_only) / n``, ``se = sqrt((a_only + b_only) - (a_only - b_only)^2 / n) / n``,
    interval ``d -/+ z * se``; all times 100."""
    c = paired_counts(a, b)
    n = c["n"]
    if n == 0:
        raise ValueError("paired_diff_ci: empty arrays")
    diff = c["a_only"] - c["b_only"]
    d = diff / float(n)
    var = (c["a_only"] + c["b_only"]) - (diff * diff) / float(n)
    se = math.sqrt(max(0.0, var)) / n
    return 100.0 * d, 100.0 * (d - z * se), 100.0 * (d + z * se)


def check_test(x, logged, tol=TEST_TOL):
    """TEST top-1 fraction ``x`` lies within ``tol`` of the logged value (inclusive, with a 1e-12 float guard)."""
    return bool(abs(float(x) - float(logged)) <= float(tol) + 1e-12)


def check_val(d, logged, tol=VAL_TOL_PP):
    """Val change ``d`` (pp) lies within ``tol`` pp of the logged change (inclusive, with a 1e-12 float guard)."""
    return bool(abs(float(d) - float(logged)) <= float(tol) + 1e-12)


# ---------------------------------------------------------------------------------------------------------------
# Cell discovery, atomic writes
# ---------------------------------------------------------------------------------------------------------------
def find_checkpoint(cells_root, cell):
    """The single ``<cells_root>/run-<cell>/recovery__*.pt``; zero or several matches raise with a clear message."""
    pattern = os.path.join(glob.escape(os.path.join(str(cells_root), "run-%s" % cell)), "recovery__*.pt")
    found = sorted(glob.glob(pattern))
    if not found:
        raise FileNotFoundError("cell %r: no recovery checkpoint matches %s" % (cell, pattern))
    if len(found) > 1:
        raise RuntimeError("cell %r: %d recovery checkpoints match %s (expected exactly one): %s"
                           % (cell, len(found), pattern, ", ".join(found)))
    return found[0]


def _atomic_write_bytes(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _atomic_write_text(path, text):
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_savez(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _atomic_save_npy(path, array):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def npz_path(out, cell, split):
    return os.path.join(out, "%s__%s.npz" % (cell, split))


def meta_path(out, cell):
    return os.path.join(out, "%s__meta.json" % cell)


def targets_path(out, split):
    return os.path.join(out, "targets__%s.npy" % split)


def load_cell_files(out, cell):
    """``{"test": {top1, top5, pred1}, "val": {...}}`` from the saved npz files, or None when absent / unreadable."""
    res = {}
    for split in SPLITS:
        path = npz_path(out, cell, split)
        if not os.path.isfile(path):
            return None
        try:
            with np.load(path) as z:
                res[split] = {k: np.asarray(z[k]) for k in ("top1", "top5", "pred1")}
        except Exception:  # noqa: BLE001 - a torn file is recomputed
            return None
    return res


def load_meta(out, cell):
    try:
        with open(meta_path(out, cell), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------------------------------------------
# Summary (pure numpy)
# ---------------------------------------------------------------------------------------------------------------
def _acc(arr):
    arr = np.asarray(arr)
    return int(np.count_nonzero(arr)), int(arr.shape[0])


def build_summary(results, params=None, secs=None, limited=False):
    """
    ``results[cell][split]`` = ``{"top1": array, "top5": array}``. Returns a JSON-able dict with the per-network
    values, the checks, the paired differences per point / pair / split, and the raw / honest values per arm.
    """
    params = params or {}
    secs = secs or {}
    present = [c for c in CELLS if c in results]
    nets = {}
    base_params = params.get("original")
    for cell in present:
        row = {"params": params.get(cell),
               "param_ratio": (float(params[cell]) / float(base_params)
                               if params.get(cell) and base_params else None),
               "secs": secs.get(cell)}
        for split in SPLITS:
            for key in ("top1", "top5"):
                c, n = _acc(results[cell][split][key])
                row["%s_%s_correct" % (split, key)] = c
                row["%s_%s_n" % (split, key)] = n
                row["%s_%s" % (split, key)] = (c / float(n)) if n else None
        nets[cell] = row

    checks = {}
    for cell in present:
        test_x = nets[cell]["test_top1"]
        row = {"test": test_x, "test_logged": LOGGED_TEST_TOP1[cell],
               "test_ok": None if limited else check_test(test_x, LOGGED_TEST_TOP1[cell])}
        if cell == "original":
            row.update({"val_change_pp": None, "val_logged_pp": None, "val_ok": None, "val_note": "reference"})
        elif "original" in nets:
            d = 100.0 * (nets[cell]["val_top1"] - nets["original"]["val_top1"])
            row.update({"val_change_pp": d, "val_logged_pp": LOGGED_VAL_CHANGE_PP[cell],
                        "val_ok": None if limited else check_val(d, LOGGED_VAL_CHANGE_PP[cell]), "val_note": ""})
        else:
            row.update({"val_change_pp": None, "val_logged_pp": LOGGED_VAL_CHANGE_PP[cell], "val_ok": None,
                        "val_note": "no original read"})
        checks[cell] = row
    failed = [c for c in present if (checks[c]["test_ok"] is False or checks[c]["val_ok"] is False)]
    if limited:
        verdict = "SKIPPED (limit)"
    elif failed:
        verdict = "NOT-REPRODUCED: " + ", ".join(failed)
    elif len(present) == len(CELLS):
        verdict = "ALL REPRODUCED"
    else:
        verdict = "PARTIAL: %d of %d networks read, none failed" % (len(present), len(CELLS))

    def _paired(a_cell, b_cell, split):
        a, b = results[a_cell][split]["top1"], results[b_cell][split]["top1"]
        c = paired_counts(a, b)
        diff, lo, hi = paired_diff_ci(a, b)
        return {"diff_pp": diff, "lo_pp": lo, "hi_pp": hi, "a_only": c["a_only"], "b_only": c["b_only"],
                "both": c["both"], "neither": c["neither"], "n": c["n"],
                "p": mcnemar_exact_p(c["a_only"], c["b_only"])}

    pairs = []
    for point in POINTS:
        for label, a_arm, b_arm in PAIRS:
            a_cell, b_cell = "%s-%s" % (a_arm, point), "%s-%s" % (b_arm, point)
            if a_cell not in results or b_cell not in results:
                continue
            for split in SPLITS:
                row = {"point": point, "pair": label, "a": a_cell, "b": b_cell, "split": split}
                row.update(_paired(a_cell, b_cell, split))
                pairs.append(row)

    arms = []
    for cell in ("origin-0",) + tuple("%s-%s" % (arm, pt) for pt in POINTS for arm in ARMS):
        if cell not in results:
            continue
        row = {"arm": cell}
        if "original" in results:
            row["raw"] = _paired(cell, "original", "test")
        if cell != "origin-0" and "origin-0" in results:
            row["honest"] = _paired(cell, "origin-0", "test")
        arms.append(row)

    return {"report": REPORT_HEADER, "limited": bool(limited), "networks": nets, "checks": checks,
            "verdict": verdict, "failed": failed, "pairs": pairs, "arms": arms}


def _frac(x):
    return "n/a" if x is None else "%.5f" % x


def _pp(x):
    return "n/a" if x is None else "%+.2f" % x


def _p(x):
    return "n/a" if x is None else "%.3g" % x


def _ci(row):
    return "%s [%s, %s]" % (_pp(row["diff_pp"]), _pp(row["lo_pp"]), _pp(row["hi_pp"]))


def render_markdown(summary):
    """The ``summary.md`` text."""
    lines = [REPORT_HEADER, ""]
    lines += ["## Networks", "",
              "| network | params | ratio to original | TEST top-1 | TEST top-5 | val top-1 | val top-5 |",
              "|---|---|---|---|---|---|---|"]
    for cell, r in summary["networks"].items():
        lines.append("| %s | %s | %s | %d/%d = %s | %d/%d = %s | %d/%d = %s | %d/%d = %s |" % (
            cell, ("%d" % r["params"]) if r["params"] else "n/a", _frac(r["param_ratio"]),
            r["test_top1_correct"], r["test_top1_n"], _frac(r["test_top1"]),
            r["test_top5_correct"], r["test_top5_n"], _frac(r["test_top5"]),
            r["val_top1_correct"], r["val_top1_n"], _frac(r["val_top1"]),
            r["val_top5_correct"], r["val_top5_n"], _frac(r["val_top5"])))

    def _ok(v):
        return "SKIPPED (limit)" if v is None and summary["limited"] else ("n/a" if v is None else ("OK" if v else "FAIL"))

    lines += ["", "## Checks against the logged values", "",
              "| network | TEST top-1 re-read | logged | TEST check | val change from original (pp) | logged (pp) | val check |",
              "|---|---|---|---|---|---|---|"]
    for cell, r in summary["checks"].items():
        lines.append("| %s | %s | %.3f | %s | %s | %s | %s |" % (
            cell, _frac(r["test"]), r["test_logged"], _ok(r["test_ok"]),
            ("%+.3f" % r["val_change_pp"]) if r["val_change_pp"] is not None else "n/a",
            ("%+.2f" % r["val_logged_pp"]) if r["val_logged_pp"] is not None else "n/a",
            _ok(r["val_ok"]) if not r["val_note"] else r["val_note"]))
    lines += ["", "Verdict: **%s**" % summary["verdict"], ""]

    lines += ["## Paired differences (a - b; a_only = images only a gets right)", "",
              "| point | pair | split | diff pp [95 % interval] | a_only | b_only | exact two-sided p |",
              "|---|---|---|---|---|---|---|"]
    for r in summary["pairs"]:
        lines.append("| %s | %s (%s - %s) | %s | %s | %d | %d | %s |" % (
            r["point"], r["pair"], r["a"], r["b"], r["split"], _ci(r), r["a_only"], r["b_only"], _p(r["p"])))

    lines += ["", "## Raw and honest, TEST (raw = arm - original; honest = arm - origin-0, i.e. raw minus the control's change)", "",
              "| arm | raw pp [95 % interval] | raw a_only / b_only | raw p | honest pp [95 % interval] | honest a_only / b_only | honest p |",
              "|---|---|---|---|---|---|---|"]
    for r in summary["arms"]:
        raw, hon = r.get("raw"), r.get("honest")
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r["arm"],
            _ci(raw) if raw else "n/a", ("%d / %d" % (raw["a_only"], raw["b_only"])) if raw else "n/a",
            _p(raw["p"]) if raw else "n/a",
            _ci(hon) if hon else "n/a", ("%d / %d" % (hon["a_only"], hon["b_only"])) if hon else "n/a",
            _p(hon["p"]) if hon else "n/a"))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Pipeline pieces (torch / torchvision / src.* imported lazily)
# ---------------------------------------------------------------------------------------------------------------
def resolve_config(config_path):
    """``(config_path, net_path, arch, script_path, dataset_spec, optional_kwargs)`` of the config's single entry,
    as ``utils.instantiate_networks_and_load_datasets`` (541-547) reads it."""
    path = config_path
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(ROOT, path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if len(cfg) != 1:
        raise ValueError("%s: expected exactly one network, found %d" % (path, len(cfg)))
    net_path, values = next(iter(cfg.items()))
    if len(values) == 3:
        arch, script_path, dataset = values
        kwargs = {}
    else:
        arch, script_path, dataset, kwargs = values
    return path, net_path, arch, script_path, dataset, dict(kwargs or {})


def load_data(dataset_spec, limit):
    import src.utils as utils
    from torch.utils.data import Subset

    if os.environ.get("SPECTRA_SPLIT_SEED", "0").strip() not in ("", "0"):
        raise RuntimeError("SPECTRA_SPLIT_SEED=%r: the runs used no split seed (0); unset it"
                           % os.environ["SPECTRA_SPLIT_SEED"])
    name_or_path, options = utils.parse_dataset_spec(dataset_spec)
    canonical = utils.canonical_dataset_name(name_or_path)
    if canonical != "imagenet1k":
        raise ValueError("the config's dataset is %r (canonical %r); this reader is for imagenet1k"
                         % (name_or_path, canonical))
    train_data, val_data, test_data = utils._val_from_train_datasets(name_or_path, options, canonical, VAL_PER_CLASS)
    num_classes = utils.infer_num_classes(train_data)
    input_shape = tuple(test_data[0][0].shape)
    if not limit:
        if len(val_data) != EXPECTED_N["val"] or len(test_data) != EXPECTED_N["test"]:
            raise AssertionError("split sizes: val %d (want %d), test %d (want %d)"
                                 % (len(val_data), EXPECTED_N["val"], len(test_data), EXPECTED_N["test"]))
    else:
        val_data = Subset(val_data, list(range(min(int(limit), len(val_data)))))
        test_data = Subset(test_data, list(range(min(int(limit), len(test_data)))))
    return name_or_path, val_data, test_data, num_classes, input_shape


def make_loader(dataset, batch, workers, device):
    from torch.utils.data import DataLoader

    kwargs = {"num_workers": int(workers), "pin_memory": device.type == "cuda"}
    if int(workers) > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    return DataLoader(dataset, batch_size=int(batch), shuffle=False, **kwargs)


def load_original(net_path, arch, script_path, dataset_name, kwargs, num_classes, input_shape):
    """The template: the pipeline's own load of the config's network."""
    import src.utils as utils
    return utils.load_model_from_script(arch, dataset_name, script_path, net_path, kwargs, num_classes, input_shape)


def load_recovered(path, template):
    """A recovered network: the epoch-90 checkpoint rebuilt strictly into a CPU copy of the template."""
    import torch
    import src.imagenet_fast as imagenet_fast
    import src.traj_models as traj_models

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != imagenet_fast.CKPT_VERSION:
        raise RuntimeError("%s: checkpoint version %r, expected %r" % (path, obj.get("version"), imagenet_fast.CKPT_VERSION))
    if int(obj["epoch"]) != 90 or int(obj["step"]) != 0:
        raise RuntimeError("%s: saved at epoch %s step %s, expected the final state (epoch 90, step 0)"
                           % (path, obj["epoch"], obj["step"]))
    if int(obj["recipe"]["epochs"]) != 90:
        raise RuntimeError("%s: recipe epochs %s, expected 90" % (path, obj["recipe"]["epochs"]))
    model = traj_models.rebuild_from_state_dict(copy.deepcopy(template).cpu(), obj["model"], obj["arch"])
    return model, {"label": obj.get("label"), "job": obj.get("job"), "saved_at": obj.get("saved_at")}


def count_params(model):
    import src.utils as utils
    return int(utils.calc_num_parameters(model))


def read_split(model, loader, device, use_amp):
    """
    One pass over ``loader``: ``(top1 uint8, top5 uint8, pred1 int16, targets int16)``. The loop is
    ``ClassificationHandler.evaluate_model`` (eval mode, no_grad, NCHW batches, fp16 autocast under SPECTRA_AMP,
    ``argmax`` prediction), with the per-image outcome kept and the top-5 added.
    """
    import torch

    model.eval()
    top1, top5, pred1, targets = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if y.dim() > 1 and y.shape[1] > 1:  # one-hot targets, as evaluate_model
                y = torch.argmax(y, dim=1)
            y = y.long()
            with torch.autocast(device_type=device.type, enabled=bool(use_amp)):
                logits = model(x)
            arg = torch.argmax(logits, dim=1)
            k5 = torch.topk(logits, 5, dim=1).indices
            hit1 = arg == y
            hit5 = (k5 == y[:, None]).any(dim=1) | hit1
            top1.append(hit1.to(torch.uint8).cpu().numpy())
            top5.append(hit5.to(torch.uint8).cpu().numpy())
            pred1.append(arg.to(torch.int16).cpu().numpy())
            targets.append(y.to(torch.int16).cpu().numpy())
    return (np.concatenate(top1), np.concatenate(top5), np.concatenate(pred1), np.concatenate(targets))


def save_targets(out, split, y):
    """Write ``targets__<split>.npy`` once; later networks must reproduce it exactly."""
    path = targets_path(out, split)
    if os.path.isfile(path):
        old = np.load(path)
        if old.shape != y.shape or not np.array_equal(old, y):
            raise RuntimeError("%s: this pass's targets differ from the saved ones (data or order changed)" % path)
        return
    _atomic_save_npy(path, y.astype(np.int16))


def cell_is_done(out, cell, expected):
    """Both npz files exist and have the expected number of images (so a --limit run never resumes a full one)."""
    res = load_cell_files(out, cell)
    if res is None:
        return None
    for split in SPLITS:
        if res[split]["top1"].shape[0] != expected[split]:
            return None
    return res


def read_cell(cell, args, device, use_amp, template, ckpt_dir_cells, val_loader, test_loader):
    """Build one network, read TEST then val, write its files; returns ``(results, params, secs)``."""
    import torch

    t0 = time.perf_counter()
    if cell == "original":  # a copy: the template stays on the CPU, untouched, for the recovered networks' rebuilds
        model, info, path = copy.deepcopy(template).cpu(), {}, None
    else:
        path = find_checkpoint(ckpt_dir_cells, cell)
        model, info = load_recovered(path, template)
    params = count_params(model)
    model = model.to(device)
    cl_weights = (args.channels_last == "on") or (args.channels_last == "auto" and cell != "original")
    if cl_weights and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    meta = {"cell": cell, "params": params, "checkpoint": path, "limit": args.limit, "channels_last_weights": bool(cl_weights),
            "amp": bool(use_amp), "batch": int(args.batch)}
    meta.update(info)
    _atomic_write_text(meta_path(args.out, cell), json.dumps(meta, indent=1, default=str))
    res = {}
    for split, loader in (("test", test_loader), ("val", val_loader)):
        t1, t5, p1, y = read_split(model, loader, device, use_amp)
        save_targets(args.out, split, y)
        _atomic_savez(npz_path(args.out, cell, split), top1=t1, top5=t5, pred1=p1)
        res[split] = {"top1": t1, "top5": t5, "pred1": p1}
    secs = time.perf_counter() - t0
    meta["secs"] = secs
    _atomic_write_text(meta_path(args.out, cell), json.dumps(meta, indent=1, default=str))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res, params, secs


def fp_line(cell, res, secs):
    c1, n1 = _acc(res["test"]["top1"])
    c5, n5 = _acc(res["test"]["top5"])
    cv, nv = _acc(res["val"]["top1"])
    return ("[fp] %s TEST top-1 %d/%d = %.5f | top-5 %d/%d = %.5f | val top-1 %d/%d = %.5f | %.0f s"
            % (cell, c1, n1, c1 / float(n1), c5, n5, c5 / float(n5), cv, nv, cv / float(nv), secs))


def print_checks(summary):
    for cell, r in summary["checks"].items():
        if summary["limited"]:
            val_part = "val change %s vs logged %s: SKIPPED (limit)" % (
                ("%+.3f" % r["val_change_pp"]) if r["val_change_pp"] is not None else "n/a",
                ("%+.2f" % r["val_logged_pp"]) if r["val_logged_pp"] is not None else "n/a")
            print("[fp-check] %s TEST %.5f vs logged %.3f: SKIPPED (limit) | %s" % (cell, r["test"], r["test_logged"], val_part))
            continue
        test_part = "TEST %.5f vs logged %.3f: %s" % (r["test"], r["test_logged"], "OK" if r["test_ok"] else "FAIL")
        if r["val_change_pp"] is None:
            val_part = "val change n/a (%s)" % (r["val_note"] or "not computed")
        else:
            val_part = "val change %+.3f vs logged %+.2f: %s" % (
                r["val_change_pp"], r["val_logged_pp"], "OK" if r["val_ok"] else "FAIL")
        print("[fp-check] %s %s | %s" % (cell, test_part, val_part))
    print("[fp-check] %s" % summary["verdict"])


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="IMN-FP: full-precision per-image re-read of the eight ImageNet networks")
    ap.add_argument("--cells-root", default=DEFAULT_CELLS_ROOT)
    ap.add_argument("--net", default=None, help="override the original network's .pth (default: the config's)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default=",".join(CELLS), help="comma list; default all eight")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N images of each split (a smoke); checks are skipped")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="re-read networks whose files exist")
    ap.add_argument("--channels-last", choices=("auto", "on", "off"), default="auto",
                    help="auto: channels_last weights for the recovered networks only (as the runs' reads); inputs stay NCHW")
    ap.add_argument("--no-amp", action="store_true", help="read in fp32 (the runs used SPECTRA_AMP=1 autocast)")
    ap.add_argument("--no-deterministic", action="store_true", help="skip SPECTRA_EVAL_FT_DETERMINISTIC's settings")
    args = ap.parse_args(argv)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    unknown = [c for c in cells if c not in CELLS]
    if unknown:
        ap.error("unknown cells %s (known: %s)" % (unknown, ", ".join(CELLS)))
    args.cells = [c for c in CELLS if c in cells]
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    # Process settings of the runs, before the first CUDA use (runner 1189, 1201, 1208-1209).
    if not args.no_deterministic:
        os.environ.setdefault("SPECTRA_EVAL_FT_DETERMINISTIC", "1")
    import torch
    from src import fortify as fortify_mod
    if args.no_deterministic:
        torch.backends.cudnn.benchmark = True
    else:
        fortify_mod.apply_ft_determinism(print)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    print("[fp] device %s | amp %d | channels_last %s | batch %d | workers %d | limit %s | cells %s"
          % (device, int(use_amp), args.channels_last, args.batch, args.workers, args.limit or "none", ",".join(args.cells)))

    config_path, net_path, arch, script_path, dataset_spec, kwargs = resolve_config(args.config)
    if args.net:
        net_path = args.net
    dataset_name, val_data, test_data, num_classes, input_shape = load_data(dataset_spec, args.limit)
    expected = {"test": len(test_data), "val": len(val_data)}
    print("[fp] config %s | network %s | arch %s | num_classes %d | input %s | test n=%d val n=%d"
          % (config_path, net_path, arch, num_classes, input_shape, expected["test"], expected["val"]))
    test_loader = make_loader(test_data, args.batch, args.workers, device)
    val_loader = make_loader(val_data, args.batch, args.workers, device)

    template = load_original(net_path, arch, script_path, dataset_name, kwargs, num_classes, input_shape)

    results, params, secs = {}, {}, {}
    for cell in args.cells:
        done = None if args.force else cell_is_done(args.out, cell, expected)
        if done is not None:
            meta = load_meta(args.out, cell)
            results[cell], params[cell], secs[cell] = done, meta.get("params"), meta.get("secs", 0.0)
            print("[fp] %s: both npz files exist in %s; skipped (--force re-reads)" % (cell, args.out))
            print(fp_line(cell, done, secs[cell]))
            continue
        res, n_params, seconds = read_cell(cell, args, device, use_amp, template, args.cells_root, val_loader, test_loader)
        results[cell], params[cell], secs[cell] = res, n_params, seconds
        print(fp_line(cell, res, seconds))
        base = params.get("original")
        ratio = ("%.4f of original" % (n_params / float(base))) if base else "ratio n/a (original not read)"
        print("[fp] %s params %d (%s)" % (cell, n_params, ratio))
        sys.stdout.flush()

    limited = bool(args.limit)
    summary = build_summary(results, params, secs, limited=limited)
    _atomic_write_text(os.path.join(args.out, "summary.json"), json.dumps(summary, indent=1))
    _atomic_write_text(os.path.join(args.out, "summary.md"), render_markdown(summary))
    print_checks(summary)
    print("[fp] wrote %s and %s" % (os.path.join(args.out, "summary.json"), os.path.join(args.out, "summary.md")))
    return 3 if summary["failed"] and not limited else 0


if __name__ == "__main__":
    sys.exit(main())
