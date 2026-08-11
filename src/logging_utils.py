"""
Timestamped, rank-aware, crash-proof logging for SPECTRA.

Why this exists
---------------
Runs happen on a SLURM node that nobody is watching, and the only artefact left behind is a
log file. The previous mechanism was a single ``print_flush`` that (a) printed only from rank
0, so a crash on rank 1 vanished, (b) had no severity levels, so a fatal error looked exactly
like a progress message, (c) had no context, so "Layer 12: masked 30/64 filters" did not say
*which network* or *which episode*, and (d) never recorded tracebacks.

What this module guarantees
---------------------------
* Every line carries a millisecond timestamp, a level, the rank, and the active context
  (episode / network / step / layer), so a log line is self-describing.
* Every rank writes its own complete DEBUG-level file; the console stays readable (rank 0
  informational messages, plus warnings and errors from every rank).
* A separate ``errors.log`` collects WARNING and above, so triaging a failed run starts with
  one short file instead of a 100k-line transcript.
* Uncaught exceptions (main thread, worker threads, and ``atexit``) are logged with full
  tracebacks before the process dies.
* A heartbeat thread periodically reports wall-clock, GPU/CPU memory and the active context,
  which is what distinguishes "still training" from "hung" in an unattended run.

Nothing here may raise: a logging failure must never take down an experiment.
"""

import atexit
import contextlib
import contextvars
import logging
import os
import platform
import socket
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import src.distributed as ddp

# Fields carried on every log record, set by `context()` / `set_context()`
_CONTEXT: "contextvars.ContextVar[Dict[str, Any]]" = contextvars.ContextVar(
    "spectra_log_context", default={})

_LOGGER_NAME = "spectra"
_SETUP_DONE = False
_SETUP_LOCK = threading.Lock()
_RUN_DIR: Optional[str] = None
_RUN_ID: Optional[str] = None
_HEARTBEAT_STOP = threading.Event()

DEFAULT_LEVEL = os.environ.get("SPECTRA_LOG_LEVEL", "INFO").strip().upper()
HEARTBEAT_SECONDS = float(os.environ.get("SPECTRA_HEARTBEAT_SECONDS", "120"))


# --------------------------------------------------------------------------- formatting

class _ContextFilter(logging.Filter):
    """Attach the active context and rank to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _CONTEXT.get()
        record.rank = ddp.get_rank()
        record.context = " ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else "-"
        return True


class _Formatter(logging.Formatter):
    """``2026-08-11 09:31:02.123 | INFO     | r0 | ep=12 net=resnet20 | message``"""

    def formatTime(self, record, datefmt=None):  # noqa: N802 (stdlib signature)
        created = datetime.fromtimestamp(record.created)
        return created.strftime("%Y-%m-%d %H:%M:%S") + f".{int(record.msecs):03d}"

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        head = (f"{self.formatTime(record)} | {record.levelname:<8} | "
                f"r{getattr(record, 'rank', 0)} | {getattr(record, 'context', '-')} | ")
        text = head + record.message
        if record.exc_info:
            text += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
        return text


# --------------------------------------------------------------------------- setup

def default_run_id() -> str:
    """Stable, sortable identifier: SLURM job id when present, else a timestamp."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job = os.environ.get("SLURM_JOB_ID")
    return f"job{job}_{stamp}" if job else f"local_{stamp}"


def run_dir() -> str:
    """Directory holding this run's logs, events and results."""
    if _RUN_DIR is None:
        setup()
    return _RUN_DIR


def run_id() -> str:
    if _RUN_ID is None:
        setup()
    return _RUN_ID


def setup(run_directory: Optional[str] = None, level: Optional[str] = None,
          identifier: Optional[str] = None) -> str:
    """
    Configure logging once per process. Safe to call repeatedly.

    Args:
        run_directory: Destination for logs/events. Defaults to ``$SPECTRA_RUN_DIR`` or
                       ``runs/<run_id>``.
        level:         Console/file level name. Defaults to ``$SPECTRA_LOG_LEVEL`` or INFO.
        identifier:    Run id. Defaults to ``$SPECTRA_RUN_ID`` or ``default_run_id()``.

    Returns:
        str: The resolved run directory.
    """
    global _SETUP_DONE, _RUN_DIR, _RUN_ID

    with _SETUP_LOCK:
        if _SETUP_DONE:
            return _RUN_DIR

        _RUN_ID = identifier or os.environ.get("SPECTRA_RUN_ID") or default_run_id()
        _RUN_DIR = (run_directory or os.environ.get("SPECTRA_RUN_DIR")
                    or os.path.join("runs", _RUN_ID))
        log_dir = os.path.join(_RUN_DIR, "logs")

        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(logging.DEBUG)  # handlers do the filtering
        logger.handlers.clear()
        logger.propagate = False
        logger.addFilter(_ContextFilter())

        formatter = _Formatter()
        console_level = getattr(logging, (level or DEFAULT_LEVEL), logging.INFO)
        rank = ddp.get_rank()

        # Console: rank 0 reports progress; other ranks stay quiet unless something is wrong,
        # which keeps the SLURM stdout readable without hiding secondary-rank failures.
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(console_level if rank == 0 else logging.WARNING)
        console.setFormatter(formatter)
        logger.addHandler(console)

        try:
            os.makedirs(log_dir, exist_ok=True)

            # Full-fidelity per-rank transcript, always DEBUG regardless of console level
            detail = logging.FileHandler(os.path.join(log_dir, f"rank{rank}.log"), encoding="utf-8")
            detail.setLevel(logging.DEBUG)
            detail.setFormatter(formatter)
            logger.addHandler(detail)

            # Triage file: everything that went wrong, from every rank, in one place
            errors = logging.FileHandler(os.path.join(log_dir, "errors.log"), encoding="utf-8")
            errors.setLevel(logging.WARNING)
            errors.setFormatter(formatter)
            logger.addHandler(errors)
        except Exception as exc:  # a read-only filesystem must not kill the run
            logger.warning(f"File logging unavailable ({exc}); console only")

        _SETUP_DONE = True

    install_excepthooks()
    return _RUN_DIR


def get_logger() -> logging.Logger:
    if not _SETUP_DONE:
        setup()
    return logging.getLogger(_LOGGER_NAME)


# --------------------------------------------------------------------------- context

def set_context(**fields) -> None:
    """Merge fields into the logging context for the current execution context."""
    merged = dict(_CONTEXT.get())
    for key, value in fields.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    _CONTEXT.set(merged)


def current_context() -> Dict[str, Any]:
    return dict(_CONTEXT.get())


@contextlib.contextmanager
def context(**fields):
    """Scope extra fields (``episode=3``, ``net='resnet20'``) onto every log line inside."""
    token = _CONTEXT.set({**_CONTEXT.get(), **{k: v for k, v in fields.items() if v is not None}})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


# --------------------------------------------------------------------------- emit

def debug(msg, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
    """Log an error together with the traceback of the exception being handled."""
    kwargs.setdefault("exc_info", True)
    get_logger().error(msg, *args, **kwargs)


# --------------------------------------------------------------------------- timing

@contextlib.contextmanager
def stage(name: str, level: int = logging.INFO, record: bool = True, **fields):
    """
    Time a block of work, logging entry, exit and duration; failures are logged and re-raised.

    Durations recorded here are what a post-mortem uses to answer "where did the time go",
    which is the first question when an experiment is too slow to finish.
    """
    logger = get_logger()
    started = time.perf_counter()
    with context(**fields):
        logger.log(level, f"[{name}] start")
        try:
            yield
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.error(f"[{name}] FAILED after {elapsed:.2f}s: {type(exc).__name__}: {exc}",
                         exc_info=True)
            _record_stage(name, elapsed, ok=False, error=f"{type(exc).__name__}: {exc}",
                          enabled=record)
            raise
        elapsed = time.perf_counter() - started
        logger.log(level, f"[{name}] done in {elapsed:.2f}s")
        _record_stage(name, elapsed, ok=True, error=None, enabled=record)


def _record_stage(name: str, seconds: float, ok: bool, error, enabled: bool):
    if not enabled:
        return
    try:
        from src.run_recorder import RunRecorder
        RunRecorder.instance().record("stage", stage=name, seconds=round(seconds, 4),
                                      ok=ok, error=error)
    except Exception:
        pass  # recording is best-effort; never let it break the run


class Timer:
    """``with Timer() as t: ...`` then ``t.seconds``. No logging, for hot paths."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False


# --------------------------------------------------------------------------- diagnostics

def resource_snapshot() -> Dict[str, Any]:
    """GPU/CPU memory and process uptime; all fields best-effort."""
    snapshot: Dict[str, Any] = {}
    try:
        import torch
        if torch.cuda.is_available():
            index = ddp.get_local_rank()
            snapshot["gpu_alloc_gb"] = round(torch.cuda.memory_allocated(index) / 1e9, 3)
            snapshot["gpu_reserved_gb"] = round(torch.cuda.memory_reserved(index) / 1e9, 3)
            snapshot["gpu_max_alloc_gb"] = round(torch.cuda.max_memory_allocated(index) / 1e9, 3)
    except Exception:
        pass
    try:
        import resource
        # ru_maxrss is KiB on Linux
        snapshot["host_rss_gb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
    except Exception:
        pass
    return snapshot


def environment_summary() -> Dict[str, Any]:
    """Everything needed to reproduce or explain a run, gathered once at startup."""
    summary: Dict[str, Any] = {
        "run_id": run_id(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "rank": ddp.get_rank(),
        "world_size": ddp.get_world_size(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "SLURM_NTASKS",
                "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(key):
            summary[key.lower()] = os.environ[key]
    summary["spectra_env"] = {k: v for k, v in os.environ.items() if k.startswith("SPECTRA_")}
    try:
        import torch
        summary["torch"] = torch.__version__
        summary["cuda_build"] = torch.version.cuda
        summary["cuda_available"] = torch.cuda.is_available()
        summary["device_count"] = torch.cuda.device_count()
        summary["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        summary["torch_error"] = str(exc)
    try:
        import subprocess
        summary["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        summary["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        pass
    return summary


def log_environment():
    """Emit the environment summary to the log (and the run manifest)."""
    summary = environment_summary()
    info("Environment: " + ", ".join(
        f"{k}={v}" for k, v in summary.items() if k not in ("spectra_env", "gpus")))
    if summary.get("gpus"):
        info(f"GPUs: {summary['gpus']}")
    if summary.get("spectra_env"):
        info(f"SPECTRA_* env: {summary['spectra_env']}")
    return summary


# --------------------------------------------------------------------------- heartbeat

def start_heartbeat(interval: Optional[float] = None):
    """
    Periodically log liveness, elapsed time, memory and the active context.

    In an unattended run this is the difference between "slow" and "deadlocked": if the
    heartbeat keeps ticking while the step counter does not move, the job is stuck rather
    than dead, and the context tells you in which stage.
    """
    seconds = interval if interval is not None else HEARTBEAT_SECONDS
    if seconds <= 0:
        return None

    started = time.perf_counter()

    def beat():
        while not _HEARTBEAT_STOP.wait(seconds):
            try:
                elapsed = time.perf_counter() - started
                snapshot = resource_snapshot()
                ctx = current_context()
                info(f"[heartbeat] up {elapsed / 60:.1f} min | "
                     + (" ".join(f"{k}={v}" for k, v in ctx.items()) or "idle")
                     + (" | " + " ".join(f"{k}={v}" for k, v in snapshot.items()) if snapshot else ""))
            except Exception:
                pass

    thread = threading.Thread(target=beat, name="spectra-heartbeat", daemon=True)
    thread.start()
    atexit.register(_HEARTBEAT_STOP.set)
    return thread


def stop_heartbeat():
    _HEARTBEAT_STOP.set()


# --------------------------------------------------------------------------- crash capture

def install_excepthooks():
    """Route uncaught exceptions (main and worker threads) through the logger."""

    def handle(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        try:
            get_logger().error("UNCAUGHT EXCEPTION - run is terminating",
                               exc_info=(exc_type, exc_value, exc_tb))
            from src.run_recorder import RunRecorder
            RunRecorder.instance().record(
                "crash", error=f"{exc_type.__name__}: {exc_value}",
                traceback="".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            RunRecorder.instance().close()
        except Exception:
            traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = handle

    def handle_thread(args):
        if issubclass(args.exc_type, SystemExit):
            return
        try:
            get_logger().error(f"UNCAUGHT EXCEPTION in thread {args.thread.name if args.thread else '?'}",
                               exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        except Exception:
            pass

    if hasattr(threading, "excepthook"):
        threading.excepthook = handle_thread
