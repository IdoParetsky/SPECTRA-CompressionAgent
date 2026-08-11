"""
Structured results gathering for SPECTRA runs.

Logs answer "what happened, in order". This module answers "what did we measure", in a form
that can be aggregated without re-reading prose: one JSON object per event, appended to
``<run_dir>/events/rank<N>.jsonl``.

Event types currently emitted
-----------------------------
``run_start`` / ``run_end``  process-level metadata and exit status
``stage``                    a timed block (feature extraction, fine-tune, evaluation, ...)
``prune``                    one compression action: structural vs masked, widths, reason
``step``                     one RL transition: action, reward, accuracy before/after
``episode``                  one A2C update: return, losses, entropy, duration
``eval``                     one evaluation record (accuracy / params / FLOPs, original vs pruned)
``standardizer``             feature-standardisation fit statistics
``issue``                    a recoverable problem worth counting (fallbacks, degraded features)
``crash``                    uncaught exception with traceback

The point of the ``issue`` and ``prune`` events is diagnostic coverage: they turn "the agent
scored poorly" into "structured pruning fell back to masking on 41% of steps, all in
DenseNet concat groups", which is actionable against the library rather than the agent.

Analyse with:  python scripts/summarize_run.py <run_dir>
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import src.distributed as ddp


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of torch/numpy/pathlib values into JSON primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    # torch tensors / numpy scalars expose .item() or .tolist()
    for attr in ("item", "tolist"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _jsonable(method())
            except Exception:
                pass
    return str(value)


class RunRecorder:
    """Append-only JSONL event sink, one file per rank."""

    _instance: Optional["RunRecorder"] = None
    _lock = threading.Lock()

    def __init__(self, directory: Optional[str] = None):
        from src import logging_utils

        self.run_dir = directory or logging_utils.run_dir()
        self.rank = ddp.get_rank()
        self.started = time.perf_counter()
        self._file = None
        self._write_lock = threading.Lock()
        self._counters: Dict[str, int] = {}

        try:
            events_dir = os.path.join(self.run_dir, "events")
            os.makedirs(events_dir, exist_ok=True)
            self.path = os.path.join(events_dir, f"rank{self.rank}.jsonl")
            self._file = open(self.path, "a", encoding="utf-8")
        except Exception as exc:  # recording must never break a run
            logging_utils.warning(f"Event recording disabled ({exc})")
            self.path = None

    @classmethod
    def instance(cls) -> "RunRecorder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance.close()
        cls._instance = None

    # ------------------------------------------------------------------ writing

    def record(self, event: str, **fields) -> None:
        """Append one event. Context fields from logging_utils are merged in automatically."""
        if self._file is None:
            return
        try:
            from src import logging_utils
            payload = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "t": round(time.perf_counter() - self.started, 4),
                "rank": self.rank,
                "event": event,
            }
            payload.update(_jsonable(logging_utils.current_context()))
            payload.update({k: _jsonable(v) for k, v in fields.items()})

            line = json.dumps(payload, ensure_ascii=False)
            with self._write_lock:
                self._file.write(line + "\n")
                self._file.flush()  # a crashed run must not lose its last events
            self._counters[event] = self._counters.get(event, 0) + 1
        except Exception:
            pass

    def issue(self, kind: str, detail: str = "", **fields) -> None:
        """
        Record a recoverable problem and warn once per (kind, detail) pair.

        Repeated identical issues are counted rather than re-logged, so a systematic fallback
        does not bury the console while still being visible in the summary.
        """
        from src import logging_utils

        key = f"issue::{kind}::{detail}"
        seen = self._counters.get(key, 0)
        self._counters[key] = seen + 1
        if seen == 0:
            logging_utils.warning(f"[{kind}] {detail}")
        self.record("issue", kind=kind, detail=detail, occurrence=seen + 1, **fields)

    def counters(self) -> Dict[str, int]:
        return dict(self._counters)

    def close(self):
        try:
            if self._file is not None:
                with self._write_lock:
                    self._file.flush()
                    self._file.close()
                self._file = None
        except Exception:
            pass

    # ------------------------------------------------------------------ manifest

    def write_manifest(self, extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Persist run metadata (environment, config, arguments) next to the events.

        Rank 0 only: the manifest describes the run, not the process.
        """
        if not ddp.is_main_process():
            return None
        try:
            from src import logging_utils
            manifest = logging_utils.environment_summary()
            manifest["run_dir"] = self.run_dir
            if extra:
                manifest.update(_jsonable(extra))
            path = os.path.join(self.run_dir, "manifest.json")
            os.makedirs(self.run_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
            return path
        except Exception as exc:
            from src import logging_utils
            logging_utils.warning(f"Could not write manifest ({exc})")
            return None


# --------------------------------------------------------------------------- helpers

def recorder() -> RunRecorder:
    return RunRecorder.instance()


def record(event: str, **fields) -> None:
    RunRecorder.instance().record(event, **fields)


def issue(kind: str, detail: str = "", **fields) -> None:
    RunRecorder.instance().issue(kind, detail, **fields)


def config_snapshot(conf) -> Dict[str, Any]:
    """
    Serialise ConfigurationValues without dragging in loaded models or DataLoaders.

    ``input_dict`` / ``database_dict`` map paths to (model, loaders) tuples, which are neither
    JSON-serialisable nor informative; only their keys are kept.
    """
    snapshot: Dict[str, Any] = {}
    for key, value in vars(conf).items():
        if key in ("input_dict", "database_dict"):
            try:
                snapshot[key] = sorted(value.keys()) if value else []
                snapshot[f"{key}_size"] = len(value) if value else 0
            except Exception:
                snapshot[key] = "<unserialisable>"
        elif key == "dataloaders_dict":
            snapshot[key] = "<registry>"
        else:
            snapshot[key] = _jsonable(value)
    return snapshot
