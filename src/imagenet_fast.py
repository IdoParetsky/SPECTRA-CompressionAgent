"""
The ImageNet recovery path (v33, 5 Nov 2026; queue row IMN). Default off: nothing here runs unless ``SPECTRA_IMN_FT=1``.

With the flag on, ``a2c_agent_reinforce_runner._final_ft`` hands a candidate's final recovery to :func:`recover`
instead of ``ClassificationHandler.train_model``:

* **the registered recipe as flags** - SGD momentum ``SPECTRA_IMN_FT_MOMENTUM`` (0.9), lr ``SPECTRA_EVAL_FINAL_FT_LR``
  (0.01), weight decay ``SPECTRA_IMN_FT_WD`` (1e-4; the stock final recipe hard-codes 5e-4), the step schedule
  ``SPECTRA_IMN_FT_MILESTONES`` (``30,60``) x ``SPECTRA_IMN_FT_GAMMA`` (0.1) stepped per epoch, batch
  ``SPECTRA_EVAL_FINAL_FT_BATCH``, mixed precision under ``SPECTRA_AMP=1``, channels_last (``SPECTRA_IMN_CHANNELS_LAST``,
  on), the last epoch kept, no early stop, no gradient clipping unless ``SPECTRA_IMN_FT_CLIP`` > 0, no KD / mixup /
  label smoothing (the counterparts' flow: torchvision's reference recipe as DepGraph's repository runs it);
* **a train loader** chosen by ``SPECTRA_IMN_DATA``: ``nfs`` (the dataset's own PIL DataLoader over the NFS tree, as
  tree_v29 reads it), ``pack`` (one node-local packed copy of the train JPEGs, staged once per job, PIL decode in the
  workers), ``pack-tv`` (the pack with torchvision's libjpeg-turbo decode and a tensor RandomResizedCrop in the
  workers; uint8 batches normalised on the GPU), ``pack-gpu`` (the pack with nvjpeg decode and the crop on the GPU in a
  producer thread), ``pack-dali`` (the pack through NVIDIA DALI's mixed decoder, only where ``nvidia.dali`` imports).
  Every path draws the same augmentation: RandomResizedCrop(224, scale (0.08, 1), ratio (3/4, 4/3)) + horizontal flip
  + ImageNet mean/std. The eval reads (val, TEST) keep the dataset's Resize(256)+CenterCrop(224) loaders;
* **preemption-safe checkpoints** under ``SPECTRA_IMN_CKPT_DIR`` (one directory per cell): the decided candidate
  (``decision/``, written by the runner through :func:`save_decision`) and, every epoch and every
  ``SPECTRA_IMN_CKPT_EVERY_STEPS`` steps, the model, optimizer, scaler, scheduler, epoch, step and the RNG states
  (``recovery__<label>.pt``). A job that finds them restores the decision (the runner's hook) and resumes at the saved
  step with the same sample order; the augmentation is seeded per (run seed, epoch, sample), so a resumed epoch sees
  the crops an uninterrupted one would have;
* **timing lines** (``[imn] ...``): the decision's stages (summed from the run's recorded events), the staging, every
  epoch's wall time, images per second and data wait, the TEST and val reads;
* ``SPECTRA_IMN_ORIGIN_ONLY=1``: the origin control as its own job (the runner skips the walk);
* ``SPECTRA_IMN_STOP_AT=e<E>s<S>``: a forced stop right after the checkpoint at that step (the smoke's stop / resume);
* ``SPECTRA_IMN_MAX_STEPS_PER_EPOCH``: an iteration cap per epoch (the smoke's 2,000).

Python 3.8 (the cluster's spectra env).
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import math
import os
import queue
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import src.run_recorder as run_recorder
import src.traj_models as traj_models
import src.utils as utils

DATA_PATHS = ("nfs", "pack", "pack-tv", "pack-gpu", "pack-dali")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CROP = 224
SCALE = (0.08, 1.0)
RATIO = (3.0 / 4.0, 4.0 / 3.0)
CKPT_VERSION = 1
_NETWORK_FS = ("nfs", "nfs4", "cifs", "smb3", "lustre", "gpfs", "beegfs", "fuse.sshfs", "glusterfs")


# ----------------------------------------------------------------------------------------------------------- flags


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name}={raw!r}: expected an integer") from error


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name}={raw!r}: expected a number") from error


def enabled() -> bool:
    """``SPECTRA_IMN_FT=1`` (default off): the final recovery runs through :func:`recover`."""
    return utils.env_flag("SPECTRA_IMN_FT")


def ckpt_dir() -> str:
    """``SPECTRA_IMN_CKPT_DIR=<dir>``: the cell's directory for the decision and the recovery checkpoints ("" = none)."""
    return os.environ.get("SPECTRA_IMN_CKPT_DIR", "").strip()


def data_path() -> str:
    """``SPECTRA_IMN_DATA`` (default ``nfs``): one of :data:`DATA_PATHS`; anything else raises ValueError."""
    raw = (os.environ.get("SPECTRA_IMN_DATA", "") or "nfs").strip().lower()
    if raw not in DATA_PATHS:
        raise ValueError(f"SPECTRA_IMN_DATA={raw!r}: expected one of {', '.join(DATA_PATHS)}")
    return raw


def local_dir() -> str:
    """``SPECTRA_IMN_LOCAL_DIR`` (default ``$TMPDIR``, else ``/tmp``): where a job stages its pack (node-local)."""
    return (os.environ.get("SPECTRA_IMN_LOCAL_DIR", "").strip() or os.environ.get("TMPDIR", "").strip() or "/tmp")


def pack_src() -> str:
    """``SPECTRA_IMN_PACK_SRC`` (default the dataset's own train root): the tree the pack is read from."""
    return os.environ.get("SPECTRA_IMN_PACK_SRC", "").strip()


def pack_limit() -> int:
    """``SPECTRA_IMN_PACK_LIMIT`` (0 = every file): pack only the first N files (loader probes and tests)."""
    return max(0, _int("SPECTRA_IMN_PACK_LIMIT", 0))


def stage_threads() -> int:
    return max(1, _int("SPECTRA_IMN_STAGE_THREADS", 16))


def workers() -> int:
    """``SPECTRA_IMN_WORKERS``: DataLoader workers of the recovery loader; default ``min(10, cpus_per_task - 2)`` under
    Slurm, else ``SPECTRA_DATALOADER_WORKERS``."""
    raw = os.environ.get("SPECTRA_IMN_WORKERS", "").strip()
    if raw:
        return max(0, int(raw))
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0").strip() or 0)
    if cpus > 0:
        return max(1, min(10, cpus - 2))
    return int(utils.DATALOADER_WORKERS)


def read_threads() -> int:
    """``SPECTRA_IMN_READ_THREADS`` (8): file-read threads of the ``pack-gpu`` / ``pack-dali`` producers."""
    return max(1, _int("SPECTRA_IMN_READ_THREADS", 8))


def weight_decay() -> float:
    return max(0.0, _float("SPECTRA_IMN_FT_WD", 1e-4))


def momentum() -> float:
    return _float("SPECTRA_IMN_FT_MOMENTUM", 0.9)


def nesterov() -> bool:
    return utils.env_flag("SPECTRA_IMN_FT_NESTEROV")


def milestones() -> List[int]:
    """``SPECTRA_IMN_FT_MILESTONES`` (``30,60``): the epochs at which the lr is multiplied by :func:`gamma`; "" = none."""
    raw = os.environ.get("SPECTRA_IMN_FT_MILESTONES", "30,60")
    out = []
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return sorted(out)


def gamma() -> float:
    return _float("SPECTRA_IMN_FT_GAMMA", 0.1)


def channels_last() -> bool:
    """``SPECTRA_IMN_CHANNELS_LAST`` (default on): NHWC weights and batches on the GPU."""
    return utils.env_flag("SPECTRA_IMN_CHANNELS_LAST", "1")


def grad_clip() -> float:
    """``SPECTRA_IMN_FT_CLIP`` (0 = none): a max gradient norm; the stock final recipe clips at 1.0, the counterparts do not."""
    return max(0.0, _float("SPECTRA_IMN_FT_CLIP", 0.0))


def ckpt_every_steps() -> int:
    return max(0, _int("SPECTRA_IMN_CKPT_EVERY_STEPS", 0))


def max_steps_per_epoch() -> int:
    return max(0, _int("SPECTRA_IMN_MAX_STEPS_PER_EPOCH", 0))


def stop_at() -> Optional[Tuple[int, int]]:
    """``SPECTRA_IMN_STOP_AT=e<E>s<S>``: stop (exit 0) right after the checkpoint at epoch E (0-based), S completed steps."""
    raw = os.environ.get("SPECTRA_IMN_STOP_AT", "").strip().lower()
    if not raw:
        return None
    m = re.fullmatch(r"e(\d+)s(\d+)", raw)
    if not m:
        raise ValueError(f"SPECTRA_IMN_STOP_AT={raw!r}: expected e<epoch>s<step>")
    return int(m.group(1)), int(m.group(2))


def origin_only() -> bool:
    """``SPECTRA_IMN_ORIGIN_ONLY=1`` (with the master flag): the origin control as its own job, no walk."""
    return enabled() and utils.env_flag("SPECTRA_IMN_ORIGIN_ONLY")


def prestage_enabled() -> bool:
    """``SPECTRA_IMN_PRESTAGE`` (default on): a pack path stages in a background thread while the decision runs."""
    return utils.env_flag("SPECTRA_IMN_PRESTAGE", "1")


def amp() -> bool:
    return utils.env_flag("SPECTRA_AMP")


def summary() -> str:
    """One token for the ``[eval] policy=`` banner."""
    stop = stop_at()
    return (f"data={data_path()},wd={weight_decay():g},milestones={'/'.join(str(m) for m in milestones()) or 'none'},"
            f"gamma={gamma():g},cl={int(channels_last())},amp={int(amp())},clip={grad_clip():g},"
            f"ckpt={'on' if ckpt_dir() else 'off'},every={ckpt_every_steps() or 'epoch'},"
            f"cap={max_steps_per_epoch() or 'none'},workers={workers()},origin_only={int(origin_only())}"
            f"{',stop=e%ds%d' % stop if stop else ''}")


def run_seed(env) -> int:
    seed = getattr(getattr(env, "conf", None), "seed", None)
    if seed is None:
        seed = os.environ.get("SPECTRA_SEED", "").strip() or 0
    return int(seed)


# ------------------------------------------------------------------------------------------- decision checkpoint


def decision_dir() -> str:
    return os.path.join(ckpt_dir(), "decision") if ckpt_dir() else ""


def decision_saved(net_name: str) -> bool:
    """True when the cell's directory holds a saved decision for ``net_name`` (flag on, directory set)."""
    d = decision_dir()
    if not enabled() or not d:
        return False
    return bool(glob.glob(os.path.join(d, f"{glob.escape(net_name)}__*__step*.json")))


def save_decision(net_path: str, label: str, candidate: dict, timing: Optional[dict] = None):
    """
    The decided candidate (its cut model and TRAJ point) under ``decision/`` through ``traj_models.save_candidate``,
    once per label; the origin control saves nothing (there is no decision). Returns the stem or None.
    """
    d = decision_dir()
    if not d or label == "origin":
        return None
    os.makedirs(d, exist_ok=True)
    name = os.path.basename(net_path)
    point = candidate["point"]
    stem = traj_models.candidate_stem(d, name, label, point["step"])
    if os.path.exists(stem + ".json") and os.path.exists(stem + ".pt"):
        return stem
    meta = {"network": net_path, "label": label, "decision_timing": timing,
            "saved_by_job": os.environ.get("SLURM_JOB_ID", ""), "saved_at": datetime.now().isoformat(timespec="seconds")}
    out = traj_models.save_candidate(candidate["model"], stem, point, meta)
    if out:
        utils.print_flush(f"[ckpt] decision saved {os.path.basename(stem)}: step {point['step']} params x{float(point['param']):.3f} "
                          f"FLOPs x{float(point['flop']):.3f} walk acc {float(point['test_acc']):.3f}")
    return out


def restored_line(net_name: str) -> str:
    """The ``[ckpt] decision restored`` line for the runner's resume hook."""
    d = decision_dir()
    parts = []
    for path in sorted(glob.glob(os.path.join(d, f"{glob.escape(net_name)}__*__step*.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        point = doc.get("point") or {}
        timing = doc.get("decision_timing") or {}
        total = timing.get("total")
        parts.append(f"{doc.get('label', '?')} step {point.get('step', '?')} params x{float(point.get('param', 0)):.3f} "
                     f"FLOPs x{float(point.get('flop', 0)):.3f} (saved by job {doc.get('saved_by_job') or '?'}"
                     f"{', decided in %.1f min' % (float(total) / 60.0) if total else ''})")
    return f"[ckpt] decision restored from {d}: {'; '.join(parts) or 'no candidate'}"


# ------------------------------------------------------------------------------------------------ decision timing

_STAGE_LABELS = (("reset.group_sensitivity", "sensitivities"), ("reset.feature_extraction", "state"),
                 ("reset.baseline_accuracy", "origin val"), ("step.decide", "plan"), ("step.prune", "cut"),
                 ("step.evaluate", "step val"), ("step.feature_extraction", "state refresh"), ("step.finetune", "step ft"))


def decision_timing(env=None, events_path: Optional[str] = None) -> Dict[str, float]:
    """
    Seconds per decision phase, summed from the run's recorded ``stage`` events (``logging_utils.stage`` records every
    stage whatever its log level; ``step.decide`` needs ``SPECTRA_TIME_DECIDE=1``), plus ``total`` since the runner's
    ``env.t_start`` and ``other`` (the TEST / val / top-5 reads and everything unstaged).
    """
    path = events_path
    if path is None:
        try:
            path = run_recorder.recorder().path
        except Exception:  # noqa: BLE001 - a missing recorder costs the line, never the run
            path = None
    sums: Dict[str, float] = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("event") == "stage" and ev.get("stage"):
                    sums[ev["stage"]] = sums.get(ev["stage"], 0.0) + float(ev.get("seconds", 0.0) or 0.0)
    out = {label: sums.get(key, 0.0) for key, label in _STAGE_LABELS}
    t_start = getattr(env, "t_start", None) if env is not None else None
    if t_start is not None:
        out["total"] = max(0.0, time.perf_counter() - float(t_start))
        out["other"] = max(0.0, out["total"] - sum(out[label] for _k, label in _STAGE_LABELS))
    return out


def timing_line(net_name: str, timing: Dict[str, float]) -> str:
    bits = [f"{label} {timing.get(label, 0.0):.1f} s" for _k, label in _STAGE_LABELS if timing.get(label, 0.0) > 0]
    if "other" in timing:
        bits.append(f"other {timing['other']:.1f} s")
    if "total" in timing:
        bits.append(f"total {timing['total']:.1f} s ({timing['total'] / 60.0:.1f} min)")
    return f"[imn] decision timing {net_name}: " + (" | ".join(bits) if bits else "no stage recorded")


# ---------------------------------------------------------------------------------------------- the train files


def scan_tree(root: str, limit_classes: int = 0, limit_files: int = 0):
    """
    ``(rel_paths, targets, classes)`` of an ImageFolder tree in torchvision's order (class directories sorted, file
    names sorted within each). ``limit_classes`` / ``limit_files`` cut the listing (probes and tests).
    """
    classes = sorted(e.name for e in os.scandir(root) if e.is_dir())
    if limit_classes > 0:
        classes = classes[:limit_classes]
    rel, targets = [], []
    for ci, c in enumerate(classes):
        names = sorted(e.name for e in os.scandir(os.path.join(root, c)) if e.is_file())
        for n in names:
            rel.append(f"{c}/{n}")
            targets.append(ci)
            if limit_files > 0 and len(rel) >= limit_files:
                return rel, targets, classes
    return rel, targets, classes


def unwrap_subset(dataset):
    """``(base, indices)``: the innermost dataset of nested Subsets and the composed indices (None = all)."""
    indices = None
    while isinstance(dataset, Subset):
        outer = list(dataset.indices)
        indices = outer if indices is None else [outer[i] for i in indices]
        dataset = dataset.dataset
    return dataset, indices


def train_root_of(base) -> str:
    """The directory the base dataset's ``samples`` live under (``split_folder`` of ``datasets.ImageNet``, else ``root``)."""
    root = getattr(base, "split_folder", None) or getattr(base, "root", None)
    if not root:
        raise ValueError(f"{type(base).__name__} has no root: not an ImageFolder-shaped dataset")
    return str(root)


def rel_paths_of(base, root: str) -> List[str]:
    """``samples`` paths relative to ``root`` (a prefix strip, then ``os.path.relpath`` where the prefix does not match)."""
    prefix = root.rstrip("/") + "/"
    out = []
    for path, _target in base.samples:
        path = str(path)
        out.append(path[len(prefix):] if path.startswith(prefix) else os.path.relpath(path, root))
    return out


# ------------------------------------------------------------------------------------------------------- the pack


def _mount_type(path: str) -> str:
    """The filesystem type of the mount holding ``path`` (from /proc/mounts; "" where unknown)."""
    try:
        path = os.path.realpath(path)
        best, best_type = "", ""
        with open("/proc/mounts", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt, fstype = parts[1], parts[2]
                if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                    best, best_type = mnt, fstype
        return best_type
    except OSError:
        return ""


def refuse_home_nfs(path: str) -> None:
    """Raise when ``path`` sits on the network-mounted home: a copy of ImageNet is never written there (quota)."""
    if utils.env_flag("SPECTRA_IMN_ALLOW_HOME_PACK"):
        return
    home = os.path.expanduser("~")
    try:
        same = os.stat(path).st_dev == os.stat(home).st_dev
    except OSError:
        return
    if same and _mount_type(home) in _NETWORK_FS:
        raise RuntimeError(f"[imn] refusing to stage the pack at {path}: it is on the network home ({_mount_type(home)}); "
                           f"set SPECTRA_IMN_LOCAL_DIR to a node-local directory")


def pack_digest(src_root: str, rel: Sequence[str]) -> str:
    """A short fingerprint of a pack's content list (the source root, the count and a few entries)."""
    h = hashlib.sha1()
    h.update(str(src_root).encode())
    h.update(str(len(rel)).encode())
    for i in (0, len(rel) // 3, len(rel) // 2, (2 * len(rel)) // 3, len(rel) - 1):
        if 0 <= i < len(rel):
            h.update(str(rel[i]).encode())
    return h.hexdigest()[:16]


class Pack:
    """
    ``<stem>.pack`` (every file's bytes back to back) + ``<stem>.idx.npz`` (``offset`` / ``length`` per entry, in the
    order the pack was built with) + ``<stem>.meta.json`` (``done`` with ``n``, ``bytes``, ``digest``). ``read(i)`` is
    one ``os.pread``: safe from DataLoader workers (forked, sharing the descriptor) and from threads.
    """

    def __init__(self, stem: str):
        self.stem = stem
        with np.load(stem + ".idx.npz") as z:
            self.offset = z["offset"].astype(np.int64)
            self.length = z["length"].astype(np.int64)
        self.n = int(self.offset.shape[0])
        self.bytes = int(self.length.sum())
        self._fd = None

    @staticmethod
    def meta_path(stem: str) -> str:
        return stem + ".meta.json"

    @staticmethod
    def complete(stem: str, n: int, digest: str) -> bool:
        try:
            with open(Pack.meta_path(stem), encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return False
        return bool(meta.get("done")) and int(meta.get("n", -1)) == int(n) and meta.get("digest") == digest \
            and os.path.exists(stem + ".pack") and os.path.exists(stem + ".idx.npz")

    def open(self):
        if self._fd is None:
            self._fd = os.open(self.stem + ".pack", os.O_RDONLY)
        return self._fd

    def close(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def read(self, i: int) -> bytes:
        fd = self.open()
        length, offset = int(self.length[i]), int(self.offset[i])
        out = os.pread(fd, length, offset)
        if len(out) != length:
            raise IOError(f"short read of pack entry {i}: {len(out)} of {length} bytes")
        return out

    def read_many(self, indices, pool: Optional[ThreadPoolExecutor] = None) -> List[bytes]:
        idx = [int(i) for i in indices]
        if pool is None or len(idx) < 8:
            return [self.read(i) for i in idx]
        return list(pool.map(self.read, idx))

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fd"] = None  # a pickled copy (spawned worker) opens its own descriptor
        return state


def build_pack(src_root: str, rel: Sequence[str], stem: str, threads: int = 16, log=None, chunk: int = 128,
               progress_every: float = 30.0, est_bytes_per_file: int = 120 * 1024) -> Pack:
    """
    Write ``<stem>.pack`` from the files ``rel`` under ``src_root`` (read with ``threads`` threads, bounded in flight,
    written in order), then the index and the meta. Reuses a complete pack with the same digest. Refuses the network
    home and a disk without room. Progress lines every ``progress_every`` seconds. Returns the :class:`Pack`.
    """
    log = log or utils.print_flush
    rel = list(rel)
    n = len(rel)
    digest = pack_digest(src_root, rel)
    if Pack.complete(stem, n, digest):
        pack = Pack(stem)
        log(f"[imn] pack reused at {stem}.pack: {pack.n} files, {pack.bytes / 2 ** 30:.1f} GiB")
        return pack
    directory = os.path.dirname(stem) or "."
    os.makedirs(directory, exist_ok=True)
    refuse_home_nfs(directory)
    need = int(n * est_bytes_per_file * 1.05)
    free = shutil.disk_usage(directory).free
    if free < need:
        raise RuntimeError(f"[imn] not enough room for the pack at {directory}: {free / 2 ** 30:.1f} GiB free, "
                           f"about {need / 2 ** 30:.1f} GiB needed for {n} files")
    log(f"[imn] staging {n} files from {src_root} to {stem}.pack with {threads} threads "
        f"({free / 2 ** 30:.0f} GiB free on {_mount_type(directory) or 'local'} at {directory})")
    tmp = stem + ".pack.tmp"
    offsets = np.zeros(n, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    chunks = [rel[i:i + chunk] for i in range(0, n, chunk)]

    def read_chunk(paths):
        return [_read_file(os.path.join(src_root, p)) for p in paths]

    t0 = time.perf_counter()
    last = t0
    written = 0
    done_files = 0
    pos = 0
    with ThreadPoolExecutor(max_workers=threads) as pool, open(tmp, "wb", buffering=8 * 2 ** 20) as out:
        in_flight = []
        next_chunk = 0
        ahead = max(2, 2 * threads)
        while next_chunk < len(chunks) and len(in_flight) < ahead:
            in_flight.append(pool.submit(read_chunk, chunks[next_chunk]))
            next_chunk += 1
        while in_flight:
            fut = in_flight.pop(0)
            blobs = fut.result()
            for blob in blobs:
                out.write(blob)
                offsets[pos] = written
                lengths[pos] = len(blob)
                written += len(blob)
                pos += 1
            done_files += len(blobs)
            if next_chunk < len(chunks):
                in_flight.append(pool.submit(read_chunk, chunks[next_chunk]))
                next_chunk += 1
            now = time.perf_counter()
            if now - last >= progress_every:
                rate = done_files / (now - t0)
                log(f"[imn] staging {done_files}/{n} files, {written / 2 ** 30:.1f} GiB, {rate:.0f} files/s, "
                    f"{written / (now - t0) / 2 ** 20:.0f} MiB/s, ETA {(n - done_files) / max(rate, 1e-9) / 60:.1f} min")
                last = now
    seconds = time.perf_counter() - t0
    os.replace(tmp, stem + ".pack")
    np.savez(stem + ".idx.npz.tmp.npz", offset=offsets, length=lengths)
    os.replace(stem + ".idx.npz.tmp.npz", stem + ".idx.npz")
    meta = {"done": True, "n": n, "bytes": written, "digest": digest, "src": src_root, "seconds": round(seconds, 2),
            "threads": threads, "job": os.environ.get("SLURM_JOB_ID", ""), "built_at": datetime.now().isoformat(timespec="seconds")}
    with open(Pack.meta_path(stem) + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(Pack.meta_path(stem) + ".tmp", Pack.meta_path(stem))
    log(f"[imn] staged train pack: {n} files, {written / 2 ** 30:.1f} GiB in {seconds:.0f} s "
        f"({n / max(seconds, 1e-9):.0f} files/s, {written / max(seconds, 1e-9) / 2 ** 20:.0f} MiB/s) at {stem}.pack")
    return Pack(stem)


def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def pack_stem(tag: str = "train") -> str:
    """``<local_dir>/spectra_imn_<job or pid>/<tag>``: one pack per job (Slurm's tmpfs namespace removes it)."""
    job = os.environ.get("SLURM_JOB_ID", "").strip() or f"pid{os.getpid()}"
    return os.path.join(local_dir(), f"spectra_imn_{job}", tag)


# ------------------------------------------------------------------------------------------------- augmentation


def sample_seed(seed: int, epoch: int, index: int) -> int:
    """A 62-bit seed that is a function of (run seed, epoch, sample): the crop draw follows the sample, not the worker."""
    h = hashlib.blake2b(f"{int(seed)}:{int(epoch)}:{int(index)}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & ((1 << 62) - 1)


def rrc_params(height: int, width: int, gen: torch.Generator, scale=SCALE, ratio=RATIO) -> Tuple[int, int, int, int]:
    """``(top, left, h, w)`` as torchvision's ``RandomResizedCrop.get_params`` draws them, from ``gen``."""
    area = float(height * width)
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * (scale[0] + (scale[1] - scale[0]) * float(torch.rand(1, generator=gen).item()))
        aspect = math.exp(log_ratio[0] + (log_ratio[1] - log_ratio[0]) * float(torch.rand(1, generator=gen).item()))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= width and 0 < h <= height:
            top = int(torch.randint(0, height - h + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, width - w + 1, (1,), generator=gen).item())
            return top, left, h, w
    in_ratio = float(width) / float(height)
    if in_ratio < ratio[0]:
        w = width
        h = int(round(w / ratio[0]))
    elif in_ratio > ratio[1]:
        h = height
        w = int(round(h * ratio[1]))
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


def decode_rgb(raw: bytes, device="cpu") -> torch.Tensor:
    """A (3, H, W) uint8 tensor from JPEG bytes: torchvision's decoder (libjpeg-turbo on the CPU, nvjpeg on CUDA),
    with PIL as the fallback for what it refuses (CMYK, a PNG under a .JPEG name, a truncated file)."""
    import torchvision.io as tio
    try:
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        return tio.decode_jpeg(data, mode=tio.ImageReadMode.RGB, device=str(device))
    except Exception:  # noqa: BLE001 - the fallback decodes what nvjpeg / libjpeg refuse
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.asarray(img)
        return torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().to(device)


def crop_flip_uint8(img: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """RandomResizedCrop(224) (antialiased bilinear) + a p=0.5 horizontal flip of a (3, H, W) uint8 tensor, from ``gen``."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as F2
    top, left, h, w = rrc_params(int(img.shape[-2]), int(img.shape[-1]), gen)
    out = F2.resized_crop(img, top, left, h, w, [CROP, CROP], interpolation=InterpolationMode.BILINEAR, antialias=True)
    if float(torch.rand(1, generator=gen).item()) < 0.5:
        out = out.flip(-1)
    return out


def normalize_batch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, H, W) -> float32 normalised; a float batch passes through."""
    if x.dtype == torch.uint8:
        return (x.float().div_(255.0) - mean) / std
    return x


def epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    """The epoch's permutation of ``range(n)``: a function of (seed, epoch) only, so a resumed epoch keeps its order."""
    gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(epoch) + 1)
    return torch.randperm(int(n), generator=gen).numpy()


class SeededAug(Dataset):
    """``base[i]`` read with torch's CPU RNG seeded by (run seed, epoch, i), so a PIL transform's draw follows the sample."""

    def __init__(self, base, seed: int):
        self.base, self.seed, self.epoch = base, int(seed), 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        torch.manual_seed(sample_seed(self.seed, self.epoch, i))
        return self.base[i]


class PackedImages(Dataset):
    """
    Items of a :class:`Pack` for the train split: ``positions[k]`` is the pack entry, ``targets[positions[k]]`` its
    class. ``decode='pil'`` returns ``transform(PIL image)`` (the dataset's own RRC + flip + ToTensor + Normalize);
    ``decode='tv'`` returns a (3, 224, 224) uint8 tensor cropped and flipped from torchvision's decoder, normalised
    later on the GPU. Both draw from (run seed, epoch, pack position).
    """

    def __init__(self, pack: Pack, targets, positions=None, decode: str = "tv", transform=None, seed: int = 0):
        if decode not in ("pil", "tv"):
            raise ValueError(f"decode={decode!r}: expected 'pil' or 'tv'")
        self.pack = pack
        self.targets = np.asarray(targets, dtype=np.int64)
        pos = np.arange(pack.n) if positions is None else np.asarray(list(positions), dtype=np.int64)
        self.positions = pos[pos < pack.n]  # a limited pack (probe, test) holds only its first files
        self.decode = decode
        self.transform = transform
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return int(self.positions.shape[0])

    def __getitem__(self, k):
        p = int(self.positions[k])
        raw = self.pack.read(p)
        target = int(self.targets[p])
        if self.decode == "pil":
            from PIL import Image
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            torch.manual_seed(sample_seed(self.seed, self.epoch, p))
            return (self.transform(img) if self.transform is not None else img), target
        gen = torch.Generator().manual_seed(sample_seed(self.seed, self.epoch, p))
        return crop_flip_uint8(decode_rgb(raw, "cpu"), gen), target


# ------------------------------------------------------------------------------------------------- train data


class _Batches:
    """A batch sampler over one epoch's order from ``start_step`` on (the DataLoader's ``batch_sampler``)."""

    def __init__(self, order: np.ndarray, batch: int, start_step: int = 0):
        self.order, self.batch, self.start = order, int(batch), int(start_step)
        self.n_steps = int(math.ceil(len(order) / float(batch)))

    def __len__(self):
        return max(0, self.n_steps - self.start)

    def __iter__(self):
        for step in range(self.start, self.n_steps):
            yield self.order[step * self.batch:(step + 1) * self.batch].tolist()


class WorkerData:
    """
    DataLoader-backed train data (``nfs``, ``pack``, ``pack-tv``): ``iterate(epoch, start_step)`` yields
    ``(step, x, y)`` over the epoch's order from ``start_step`` (one DataLoader per epoch, so a resumed epoch starts at
    its step). Nothing to checkpoint beyond (epoch, step): the augmentation is seeded per sample.
    """

    def __init__(self, dataset, batch: int, num_workers: int, seed: int, label: str, pin: bool = False, prefetch: int = 4):
        self.dataset, self.batch, self.num_workers, self.seed, self.label = dataset, int(batch), int(num_workers), int(seed), label
        self.pin, self.prefetch = bool(pin), int(prefetch)
        self.images = len(dataset)
        self.steps_per_epoch = int(math.ceil(self.images / float(self.batch)))

    def iterate(self, epoch: int, start_step: int = 0):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        order = epoch_order(self.images, self.seed, epoch)
        sampler = _Batches(order, self.batch, start_step)
        kwargs = dict(batch_sampler=sampler, num_workers=self.num_workers, pin_memory=self.pin)
        if self.num_workers > 0:
            kwargs.update(prefetch_factor=self.prefetch, persistent_workers=False)
        loader = DataLoader(self.dataset, **kwargs)
        step = start_step
        for x, y in loader:
            yield step, x, y
            step += 1

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


class GpuDecodeData:
    """
    ``pack-gpu``: a producer thread reads each batch's bytes from the pack (a thread pool), decodes them on the device
    one by one (``torchvision.io.decode_jpeg(device='cuda')``: this torchvision has no batched CUDA decode), crops and
    flips on the device from a generator seeded per (run seed, epoch, pack position) - the same draw as ``pack-tv``,
    and nothing to checkpoint although the producer runs ahead of the step - and queues uint8 (B, 3, 224, 224)
    batches on a side stream; the consumer waits on the batch's event. On a CPU device the same code runs with the
    CPU decoder (tests).
    """

    def __init__(self, pack: Pack, targets, positions, batch: int, seed: int, device, threads: int = 8, prefetch: int = 3):
        self.pack, self.batch, self.seed = pack, int(batch), int(seed)
        self.targets = np.asarray(targets, dtype=np.int64)
        pos = np.asarray(list(positions), dtype=np.int64) if positions is not None else np.arange(pack.n)
        self.positions = pos[pos < pack.n]
        self.device = torch.device(device)
        self.threads, self.prefetch = int(threads), int(prefetch)
        self.images = int(self.positions.shape[0])
        self.steps_per_epoch = int(math.ceil(self.images / float(self.batch)))
        self.label = "pack-gpu"
        self.decode_seconds = 0.0

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None

    def _produce(self, epoch, start_step, q, stop):
        order = epoch_order(self.images, self.seed, epoch)
        use_cuda = self.device.type == "cuda"
        stream = torch.cuda.Stream(device=self.device) if use_cuda else None
        try:
            with ThreadPoolExecutor(max_workers=self.threads) as pool:
                for step in range(start_step, self.steps_per_epoch):
                    if stop.is_set():
                        break
                    idx = order[step * self.batch:(step + 1) * self.batch]
                    pos = self.positions[idx]
                    raws = self.pack.read_many(pos, pool)
                    y = torch.from_numpy(self.targets[pos])
                    t0 = time.perf_counter()
                    ctx = torch.cuda.stream(stream) if use_cuda else _nullcontext()
                    with ctx:
                        imgs = []
                        for p, raw in zip(pos, raws):
                            gen = torch.Generator().manual_seed(sample_seed(self.seed, epoch, int(p)))
                            imgs.append(crop_flip_uint8(decode_rgb(raw, self.device), gen))
                        x = torch.stack(imgs)
                        event = torch.cuda.Event() if use_cuda else None
                        if event is not None:
                            event.record(stream)
                    self.decode_seconds += time.perf_counter() - t0
                    q.put((step, x, y, event))
        except BaseException as error:  # noqa: BLE001 - handed to the consumer
            q.put(error)
        finally:
            q.put(None)

    def iterate(self, epoch: int, start_step: int = 0):
        q: "queue.Queue" = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        thread = threading.Thread(target=self._produce, args=(epoch, start_step, q, stop), name="imn-gpu-decode", daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                step, x, y, event = item
                if event is not None:
                    torch.cuda.current_stream(self.device).wait_event(event)
                    x.record_stream(torch.cuda.current_stream(self.device))
                yield step, x, y
        finally:
            stop.set()
            while thread.is_alive():
                try:
                    q.get(timeout=0.1)
                except queue.Empty:
                    pass
            thread.join(timeout=5.0)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _grad_scaler(use_amp: bool):
    """A GradScaler that is inert off the GPU (``torch.amp.GradScaler`` where this torch has it)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def dali_available() -> bool:
    try:
        import nvidia.dali  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class DaliData:
    """
    ``pack-dali``: the pack's bytes fed to a DALI pipeline (``fn.decoders.image_random_crop`` on the mixed backend,
    ``fn.resize`` to 224 with antialiasing, ``fn.crop_mirror_normalize``), one pipeline per epoch seeded by (seed, epoch).
    The sample order is this module's; DALI's own crop draws are not restored across a mid-epoch resume.
    """

    def __init__(self, pack: Pack, targets, positions, batch: int, seed: int, device, threads: int = 4, read_threads: int = 8):
        if not dali_available():
            raise RuntimeError("SPECTRA_IMN_DATA=pack-dali needs nvidia.dali (run the job with the DALI venv's python)")
        self.pack, self.batch, self.seed = pack, int(batch), int(seed)
        self.targets = np.asarray(targets, dtype=np.int64)
        pos = np.asarray(list(positions), dtype=np.int64) if positions is not None else np.arange(pack.n)
        self.positions = pos[pos < pack.n]
        self.device = torch.device(device)
        self.threads, self.read_threads = int(threads), int(read_threads)
        self.images = int(self.positions.shape[0])
        self.steps_per_epoch = int(math.ceil(self.images / float(self.batch)))
        self.label = "pack-dali"

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None

    def _pipeline(self, epoch: int):
        from nvidia.dali import fn, pipeline_def, types
        device_id = self.device.index if self.device.index is not None else 0
        seed = self.seed * 1_000_003 + epoch + 11

        @pipeline_def(batch_size=self.batch, num_threads=self.threads, device_id=device_id, seed=seed, prefetch_queue_depth=2)
        def pipe():
            jpegs = fn.external_source(name="raw", device="cpu")
            imgs = fn.decoders.image_random_crop(jpegs, device="mixed", output_type=types.RGB,
                                                 random_area=[SCALE[0], SCALE[1]], random_aspect_ratio=[RATIO[0], RATIO[1]])
            imgs = fn.resize(imgs, resize_x=CROP, resize_y=CROP, interp_type=types.INTERP_TRIANGULAR, antialias=True)
            out = fn.crop_mirror_normalize(imgs, dtype=types.FLOAT, output_layout="CHW",
                                           mean=[m * 255.0 for m in MEAN], std=[s * 255.0 for s in STD],
                                           mirror=fn.random.coin_flip(probability=0.5))
            return out

        p = pipe()
        p.build()
        return p

    def iterate(self, epoch: int, start_step: int = 0):
        from nvidia.dali.plugin.pytorch import feed_ndarray
        order = epoch_order(self.images, self.seed, epoch)
        pipe = self._pipeline(epoch)
        with ThreadPoolExecutor(max_workers=self.read_threads) as pool:
            for step in range(start_step, self.steps_per_epoch):
                idx = order[step * self.batch:(step + 1) * self.batch]
                pos = self.positions[idx]
                raws = [np.frombuffer(b, dtype=np.uint8) for b in self.pack.read_many(pos, pool)]
                if len(raws) < self.batch:  # DALI's batch is fixed: pad the short last batch, then drop the padding
                    raws = raws + [raws[-1]] * (self.batch - len(raws))
                pipe.feed_input("raw", raws)
                out = pipe.run()
                images = out[0].as_tensor()
                x = torch.empty(tuple(images.shape()), dtype=torch.float32, device=self.device)
                feed_ndarray(images, x, cuda_stream=torch.cuda.current_stream(self.device).cuda_stream)
                x = x[:len(pos)]
                y = torch.from_numpy(self.targets[pos])
                yield step, x, y


# ---------------------------------------------------------------------------------------- staging in the background

_PRESTAGE: Dict[str, object] = {}


def _pack_plan(train_loader):
    """``(src_root, rel_paths, base, indices)`` of the loader's train dataset, honouring ``SPECTRA_IMN_PACK_SRC`` / ``_LIMIT``."""
    base, indices = unwrap_subset(train_loader.dataset)
    root = pack_src() or train_root_of(base)
    rel = rel_paths_of(base, train_root_of(base))
    limit = pack_limit()
    if limit > 0:
        rel = rel[:limit]
    return root, rel, base, indices


def prestage(loaders, log=None) -> bool:
    """
    Start staging the train pack in a background thread (``SPECTRA_IMN_DATA`` a pack path, ``SPECTRA_IMN_PRESTAGE`` on),
    so it overlaps the decision. ``loaders`` is the network's ``(train, val, test)``. Returns True when started.
    """
    log = log or utils.print_flush
    if not enabled() or not data_path().startswith("pack") or not prestage_enabled() or _PRESTAGE.get("thread") is not None:
        return False
    try:
        root, rel, _base, _indices = _pack_plan(loaders[0])
    except Exception as error:  # noqa: BLE001 - staging then happens at recovery time, with the error visible
        log(f"[imn] prestage skipped ({type(error).__name__}: {error})")
        return False
    stem = pack_stem()
    _PRESTAGE.update(stem=stem, started=time.perf_counter(), error=None, pack=None)

    def run():
        try:
            _PRESTAGE["pack"] = build_pack(root, rel, stem, threads=stage_threads(), log=log)
        except BaseException as error:  # noqa: BLE001 - re-raised by staged_pack
            _PRESTAGE["error"] = error

    thread = threading.Thread(target=run, name="imn-prestage", daemon=True)
    _PRESTAGE["thread"] = thread
    thread.start()
    log(f"[imn] prestage started in the background: {len(rel)} files -> {stem}.pack")
    return True


def staged_pack(train_loader, log=None) -> Pack:
    """The job's train pack: the prestaged one (waited for), else built now."""
    log = log or utils.print_flush
    thread = _PRESTAGE.get("thread")
    if thread is not None:
        t0 = time.perf_counter()
        thread.join()
        waited = time.perf_counter() - t0
        if _PRESTAGE.get("error") is not None:
            raise _PRESTAGE["error"]
        log(f"[imn] prestage joined after waiting {waited:.0f} s (staging took "
            f"{time.perf_counter() - float(_PRESTAGE['started']):.0f} s from its start)")
        return _PRESTAGE["pack"]
    root, rel, _base, _indices = _pack_plan(train_loader)
    return build_pack(root, rel, pack_stem(), threads=stage_threads(), log=log)


def build_train_data(env, batch: int, seed: int, log=None):
    """The recovery's train data for ``env.train_loader`` under ``SPECTRA_IMN_DATA``."""
    log = log or utils.print_flush
    path = data_path()
    device = torch.device(env.conf.device)
    use_cuda = device.type == "cuda"
    dataset = env.train_loader.dataset
    if path == "nfs":
        data = WorkerData(SeededAug(dataset, seed), batch, workers(), seed, "nfs", pin=use_cuda)
    else:
        pack = staged_pack(env.train_loader, log)
        base, indices = unwrap_subset(dataset)
        targets = np.asarray(utils._targets_of(base), dtype=np.int64)
        positions = np.arange(len(base)) if indices is None else np.asarray(indices, dtype=np.int64)
        if path == "pack":
            data = WorkerData(PackedImages(pack, targets, positions, "pil", getattr(base, "transform", None), seed),
                              batch, workers(), seed, "pack", pin=use_cuda)
        elif path == "pack-tv":
            data = WorkerData(PackedImages(pack, targets, positions, "tv", None, seed), batch, workers(), seed, "pack-tv",
                              pin=use_cuda)
        elif path == "pack-gpu":
            data = GpuDecodeData(pack, targets, positions, batch, seed, device, threads=read_threads())
        else:
            data = DaliData(pack, targets, positions, batch, seed, device, threads=max(1, workers()), read_threads=read_threads())
    log(f"[imn] train data: {data.label} | {data.images} images | {data.steps_per_epoch} steps of {batch} per epoch | "
        f"workers {getattr(data, 'num_workers', 'n/a')} | device {device}")
    return data


# ------------------------------------------------------------------------------------------------------ recovery


def recovery_path(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(label))
    return os.path.join(ckpt_dir(), f"recovery__{safe}.pt") if ckpt_dir() else ""


class _State:
    def __init__(self):
        self.epoch = 0          # the epoch the next step belongs to
        self.step = 0           # steps completed in that epoch
        self.global_step = 0
        self.timing: List[dict] = []
        self.loss_sum = 0.0
        self.loss_n = 0
        self.epoch_images = 0
        self.epoch_seconds = 0.0
        self.epoch_wait = 0.0
        self.resumed = 0


def _rng_state(data, use_cuda: bool) -> dict:
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if use_cuda else None,
            "numpy": np.random.get_state(), "python": random.getstate(), "data": data.state_dict()}


def _rng_restore(rng: dict, data, use_cuda: bool) -> None:
    torch.set_rng_state(rng["torch"])
    if use_cuda and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    data.load_state_dict(rng.get("data") or {})


def save_checkpoint(path: str, label: str, model, opt, sched, scaler, data, state: _State, arch: dict, recipe: dict,
                    use_cuda: bool) -> float:
    """Atomic: ``<path>.tmp`` then ``os.replace``; a kill mid-write leaves the previous file whole. Returns seconds."""
    t0 = time.perf_counter()
    if use_cuda:
        torch.cuda.synchronize()
    obj = {
        "version": CKPT_VERSION, "label": label, "arch": arch, "recipe": recipe,
        "epoch": state.epoch, "step": state.step, "global_step": state.global_step, "timing": state.timing,
        "loss_sum": state.loss_sum, "loss_n": state.loss_n, "epoch_images": state.epoch_images,
        "epoch_seconds": state.epoch_seconds, "epoch_wait": state.epoch_wait, "resumed": state.resumed,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
        "rng": _rng_state(data, use_cuda),
        "job": os.environ.get("SLURM_JOB_ID", ""), "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)
    side = {k: obj[k] for k in ("version", "label", "epoch", "step", "global_step", "timing", "resumed", "job", "saved_at", "recipe")}
    with open(path[:-3] + ".json.tmp", "w", encoding="utf-8") as fh:
        json.dump(side, fh, indent=1, default=str)
    os.replace(path[:-3] + ".json.tmp", path[:-3] + ".json")
    return time.perf_counter() - t0


def load_checkpoint(path: str, model, opt, sched, scaler, data, state: _State, arch: dict, use_cuda: bool, log) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if int(obj.get("version", -1)) != CKPT_VERSION:
        raise RuntimeError(f"[ckpt] {path}: version {obj.get('version')} is not {CKPT_VERSION}")
    if obj.get("arch") != arch:
        raise RuntimeError(f"[ckpt] {path}: the saved architecture differs from this candidate's; refusing to resume")
    model.load_state_dict(obj["model"])
    opt.load_state_dict(obj["optimizer"])
    sched.load_state_dict(obj["scheduler"])
    scaler.load_state_dict(obj["scaler"])
    _rng_restore(obj["rng"], data, use_cuda)
    state.epoch, state.step, state.global_step = int(obj["epoch"]), int(obj["step"]), int(obj["global_step"])
    state.timing = list(obj.get("timing") or [])
    state.loss_sum, state.loss_n = float(obj.get("loss_sum", 0.0)), int(obj.get("loss_n", 0))
    state.epoch_images, state.epoch_seconds = int(obj.get("epoch_images", 0)), float(obj.get("epoch_seconds", 0.0))
    state.epoch_wait = float(obj.get("epoch_wait", 0.0))
    state.resumed = int(obj.get("resumed", 0)) + 1
    log(f"[ckpt] resumed at epoch {state.epoch} step {state.step} (global step {state.global_step}, "
        f"lr {opt.param_groups[0]['lr']:g}, saved by job {obj.get('job') or '?'} at {obj.get('saved_at')}, "
        f"{len(state.timing)} epoch rows, resume #{state.resumed})")


def recover(env, model, label: str, point: dict, epochs: int, lr: float, batch: int, name: str, log=None) -> dict:
    """
    The registered recovery of ``model`` (a candidate's copy, every parameter trainable) on ``env.train_loader``'s
    images: ``epochs`` epochs of SGD under the flags above, the last epoch kept, checkpointed under the cell's
    directory and resumed from it. Returns a dict for the runner's TRAJ line and record.
    """
    log = log or utils.print_flush
    device = torch.device(env.conf.device)
    use_cuda = device.type == "cuda"
    use_amp = amp() and use_cuda
    use_cl = channels_last() and use_cuda
    seed = run_seed(env)
    epochs = int(epochs)
    t_build = time.perf_counter()
    data = build_train_data(env, batch, seed, log)
    build_seconds = time.perf_counter() - t_build
    model.to(device)
    if use_cl:
        model.to(memory_format=torch.channels_last)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("[imn] no trainable parameter")
    opt = torch.optim.SGD(params, lr=float(lr), momentum=momentum(), weight_decay=weight_decay(), nesterov=nesterov())
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones(), gamma=gamma())
    scaler = _grad_scaler(use_amp)
    autocast = (lambda: torch.autocast("cuda", enabled=True)) if use_amp else _nullcontext
    loss_fn = torch.nn.CrossEntropyLoss()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    arch = traj_models.module_arch(model)
    recipe = {"optim": "sgd", "lr": float(lr), "momentum": momentum(), "nesterov": nesterov(), "wd": weight_decay(),
              "milestones": milestones(), "gamma": gamma(), "batch": int(batch), "epochs": epochs, "amp": bool(use_amp),
              "channels_last": bool(use_cl), "clip": grad_clip(), "data": data.label, "seed": seed, "select": "last",
              "cap": max_steps_per_epoch()}
    state = _State()
    path = recovery_path(label)
    if path and os.path.exists(path):
        load_checkpoint(path, model, opt, sched, scaler, data, state, arch, use_cuda, log)
        model.to(device)
        if use_cl:
            model.to(memory_format=torch.channels_last)
        model.train()
    elif not path:
        log("[imn] WARNING: SPECTRA_IMN_CKPT_DIR unset; the recovery is not checkpointed")
    cap = max_steps_per_epoch()
    n_steps = min(cap, data.steps_per_epoch) if cap > 0 else data.steps_per_epoch
    every = ckpt_every_steps()
    stop = stop_at()
    clip = grad_clip()
    log(f"[imn] recovery {label} {name}: {epochs} epochs x {n_steps} steps of {batch} ({data.images} images; "
        f"{'capped from %d' % data.steps_per_epoch if n_steps < data.steps_per_epoch else 'full epochs'}) | "
        f"sgd lr={float(lr):g} m={momentum():g} wd={weight_decay():g} milestones={milestones()} gamma={gamma():g} "
        f"amp={int(use_amp)} channels_last={int(use_cl)} clip={clip:g} | data={data.label} | "
        f"ckpt={path or 'off'} every={every or 'epoch'}{' stop=e%ds%d' % stop if stop else ''} | "
        f"loader built in {build_seconds:.0f} s | start epoch {state.epoch} step {state.step}")
    job_seconds = 0.0
    for epoch in range(state.epoch, epochs):
        start = state.step if (epoch == state.epoch and state.step > 0) else 0
        if start == 0:
            state.loss_sum, state.loss_n, state.epoch_images = 0.0, 0, 0
            state.epoch_seconds, state.epoch_wait = 0.0, 0.0
        loss_acc = torch.zeros((), device=device)
        loss_cnt = 0
        images = 0
        wait = 0.0
        ckpt_s = 0.0
        t_epoch = time.perf_counter()
        t_fetch = time.perf_counter()
        lr_before = opt.param_groups[0]["lr"]
        for step, x, y in data.iterate(epoch, start):
            wait += time.perf_counter() - t_fetch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = normalize_batch(x, mean, std)
            if use_cl:
                x = x.contiguous(memory_format=torch.channels_last)
            if x.size(0) >= 2:
                with autocast():
                    out = model(x)
                    loss = loss_fn(out, y.long())
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=clip)
                scaler.step(opt)
                scaler.update()
                loss_acc += loss.detach().float()
                loss_cnt += 1
                images += int(x.size(0))
            state.step = step + 1
            state.global_step += 1
            done_epoch = state.step >= n_steps
            want_ckpt = bool(path) and ((every > 0 and state.step % every == 0 and not done_epoch)
                                        or (stop is not None and stop == (epoch, state.step)))
            if want_ckpt:
                if use_cuda:
                    torch.cuda.synchronize()
                segment = time.perf_counter() - t_epoch
                state.loss_sum += float(loss_acc.item())
                state.loss_n += loss_cnt
                loss_acc.zero_()
                loss_cnt = 0
                state.epoch_images += images
                state.epoch_seconds += segment
                state.epoch_wait += wait
                job_seconds += segment
                images, wait = 0, 0.0
                secs = save_checkpoint(path, label, model, opt, sched, scaler, data, state, arch, recipe, use_cuda)
                ckpt_s += secs
                t_epoch = time.perf_counter()  # the write is not training time
                log(f"[ckpt] saved epoch {epoch} step {state.step} ({secs:.1f} s)")
            if stop is not None and stop == (epoch, state.step):
                log(f"[ckpt] forced stop at epoch {epoch} step {state.step} (SPECTRA_IMN_STOP_AT); exiting 0, the resume job continues")
                try:
                    run_recorder.record("run_end", status="forced_stop", epoch=epoch, step=state.step)
                except Exception:  # noqa: BLE001
                    pass
                sys.stdout.flush()
                raise SystemExit(0)
            if done_epoch:
                break
            t_fetch = time.perf_counter()
        if use_cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t_epoch
        state.loss_sum += float(loss_acc.item())
        state.loss_n += loss_cnt
        state.epoch_images += images
        state.epoch_seconds += seconds
        state.epoch_wait += wait
        job_seconds += seconds
        sched.step()
        lr_after = opt.param_groups[0]["lr"]
        avg = state.loss_sum / max(1, state.loss_n)
        rate = state.epoch_images / max(state.epoch_seconds, 1e-9)
        row = {"epoch": epoch, "seconds": round(state.epoch_seconds, 2), "images": state.epoch_images, "img_s": round(rate, 1),
               "wait_s": round(state.epoch_wait, 2), "loss": round(avg, 5), "lr": lr_before, "lr_after": lr_after,
               "ckpt_s": round(ckpt_s, 2), "job": os.environ.get("SLURM_JOB_ID", ""), "resumed_at": start}
        state.timing.append(row)
        state.epoch, state.step = epoch + 1, 0
        state.loss_sum, state.loss_n, state.epoch_images, state.epoch_seconds, state.epoch_wait = 0.0, 0, 0, 0.0, 0.0
        log(f"[imn] epoch {epoch + 1}/{epochs} {label} | loss {avg:.4f} | lr {lr_before:g} -> {lr_after:g} | "
            f"{row['seconds']:.1f} s | {row['images']} img | {rate:.0f} img/s | data wait {row['wait_s']:.1f} s "
            f"({100.0 * row['wait_s'] / max(row['seconds'], 1e-9):.0f} %) | ckpt {ckpt_s:.1f} s"
            f"{' | resumed at step %d' % start if start else ''}")
        if path:
            secs = save_checkpoint(path, label, model, opt, sched, scaler, data, state, arch, recipe, use_cuda)
            log(f"[ckpt] saved epoch {epoch + 1} ({secs:.1f} s)")
    model.eval()
    full = [r for r in state.timing if int(r.get("images", 0)) >= 0.95 * data.images]
    total = sum(float(r["seconds"]) for r in state.timing)
    mean_rate = (sum(float(r["img_s"]) for r in full) / len(full)) if full else (
        state.timing[-1]["img_s"] if state.timing else 0.0)
    log(f"[imn] recovery summary {label} {name}: {len(state.timing)} epoch rows | train {total / 3600:.2f} h "
        f"(this job {job_seconds / 3600:.2f} h, earlier jobs {(total - job_seconds) / 3600:.2f} h) | "
        f"mean {mean_rate:.0f} img/s over {len(full)} full epochs | resumed {state.resumed} time(s) | data {data.label}")
    return {"epochs_ran": len(state.timing), "resumed": state.resumed, "data": data.label, "img_s": mean_rate,
            "train_minutes": total / 60.0, "job_minutes": job_seconds / 60.0, "steps_per_epoch": n_steps,
            "workers": getattr(data, "num_workers", 0), "recipe": recipe, "timing": state.timing}


def recipe_text(out: dict) -> str:
    """The recipe fragment of the TRAJ final_ft line under the flag."""
    r = out["recipe"]
    steps = ",".join(str(m) for m in r["milestones"]) or "none"
    return (f"sgd lr={r['lr']:g} m={r['momentum']:g} wd={r['wd']:g} step{steps} g{r['gamma']:g} e{r['epochs']} keep=last "
            f"bs={r['batch']} aug=imn:{r['data']} amp={int(r['amp'])} cl={int(r['channels_last'])} clip={r['clip']:g} "
            f"kd=0 init=inherit{' cap=%d' % r['cap'] if r['cap'] else ''}"
            f"{' resumed=%d' % out['resumed'] if out['resumed'] else ''}")


class timed:
    """``with timed(log, 'TEST read', name, n_images):`` prints ``[imn] <what> <name>: N images in S s (R img/s)``."""

    def __init__(self, log, what: str, name: str, n_images: int):
        self.log, self.what, self.name, self.n = log, what, name, int(n_images)

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        secs = time.perf_counter() - self.t0
        self.log(f"[imn] {self.what} {self.name}: {self.n} images in {secs:.1f} s ({self.n / max(secs, 1e-9):.0f} img/s)")
        return False
