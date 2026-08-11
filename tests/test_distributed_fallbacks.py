"""
The distributed helpers must behave sensibly when no process group exists, which is the
default for a single-GPU run. Every one of these used to raise or mis-report.

    python -m pytest tests/test_distributed_fallbacks.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.distributed as ddp  # noqa: E402


def test_queries_are_safe_without_a_process_group():
    assert ddp.is_distributed() is False
    assert ddp.get_world_size() == 1
    assert ddp.get_rank() == 0
    assert ddp.is_main_process() is True
    ddp.barrier()  # must be a no-op rather than an error


def test_collectives_degrade_to_identity():
    assert ddp.broadcast_flag(True) is True
    assert ddp.broadcast_flag(False) is False
    assert ddp.all_agree(True) is True
    assert ddp.all_agree(False) is False


def test_distributed_is_not_initialised_without_a_launcher(monkeypatch):
    """A bare `python a2c_agent_reinforce_runner.py ...` must never touch NCCL."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    assert ddp.launcher_world_size() == 1
    assert ddp.maybe_init_distributed() is False


def test_launcher_world_size_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert ddp.launcher_world_size() == 2
    monkeypatch.delenv("WORLD_SIZE")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    assert ddp.launcher_world_size() == 2


def test_unwrap_handles_plain_and_nested_wrappers():
    import torch
    from torch import nn

    model = nn.Linear(4, 4)
    assert ddp.unwrap(model) is model
    assert ddp.unwrap(nn.DataParallel(model)) is model
    assert ddp.unwrap(nn.DataParallel(nn.DataParallel(model))) is model


def _capture_spectra_records():
    """
    Collect records emitted on the 'spectra' logger.

    Neither capsys nor caplog can see them: the console handler binds sys.stdout at setup
    time (before capsys replaces it), and the logger sets propagate=False so records never
    reach caplog's root handler. Attaching our own handler tests the real path and behaves
    identically across pytest versions.
    """
    import logging
    import src.logging_utils as logging_utils

    logging_utils.setup()
    logger = logging_utils.get_logger()

    class _Collector(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.DEBUG)
            self.records = []

        def emit(self, record):
            self.records.append(record)

    return logger, _Collector()


def test_print_flush_does_not_require_a_process_group():
    """print_flush must work with no process group, and must actually emit a record."""
    import src.utils as utils

    logger, collector = _capture_spectra_records()
    logger.addHandler(collector)
    try:
        utils.print_flush("hello")
    finally:
        logger.removeHandler(collector)

    assert any("hello" in record.getMessage() for record in collector.records)


def test_logging_carries_timestamp_rank_and_context():
    """A log line must be self-describing: when, how severe, which rank, which episode."""
    import src.logging_utils as logging_utils

    logger, collector = _capture_spectra_records()
    logger.addHandler(collector)
    try:
        with logging_utils.context(ep=7, net="resnet20"):
            logging_utils.info("probe")
    finally:
        logger.removeHandler(collector)

    # The context filter already stamped the record at emit time, inside the scope; running
    # it again here would re-read the (now-empty) context and erase what we are testing.
    line = logging_utils._Formatter().format(collector.records[-1])

    assert "INFO" in line and "r0" in line and "ep=7" in line and "net=resnet20" in line
    assert line.count(":") >= 2  # HH:MM:SS timestamp present


def test_context_is_restored_after_the_scope_exits():
    """A leaked context would mislabel every later log line with a stale episode."""
    import src.logging_utils as logging_utils

    before = logging_utils.current_context()
    with logging_utils.context(ep=99):
        assert logging_utils.current_context().get("ep") == 99
    assert logging_utils.current_context() == before


def test_stage_logs_and_reraises_failures():
    """A stage that raises must be recorded as failed, not swallowed."""
    import src.logging_utils as logging_utils

    logger, collector = _capture_spectra_records()
    logger.addHandler(collector)
    try:
        with pytest.raises(ValueError):
            with logging_utils.stage("unit.explode", record=False):
                raise ValueError("boom")
    finally:
        logger.removeHandler(collector)

    messages = [r.getMessage() for r in collector.records]
    assert any("FAILED" in m and "unit.explode" in m for m in messages)
    assert any(r.exc_info for r in collector.records), "traceback must be attached"


def test_run_recorder_writes_jsonl_events(tmp_path):
    """Events must survive as parseable JSON so summaries never re-parse prose logs."""
    import json
    import src.logging_utils as logging_utils
    from src.run_recorder import RunRecorder

    RunRecorder.reset_instance()
    recorder = RunRecorder(directory=str(tmp_path))
    with logging_utils.context(ep=3):
        recorder.record("step", reward=1.5, compression_rate=0.8)
    recorder.issue("prune_fallback_masked", "concat offset unresolved")
    recorder.close()

    lines = [json.loads(l) for l in open(recorder.path, encoding="utf-8").read().splitlines()]
    assert lines[0]["event"] == "step" and lines[0]["reward"] == 1.5 and lines[0]["ep"] == 3
    assert lines[0]["ts"] and lines[0]["t"] >= 0
    assert lines[1]["event"] == "issue" and lines[1]["kind"] == "prune_fallback_masked"
    RunRecorder.reset_instance()


def test_summarizer_reports_a_masked_fallback_run(tmp_path):
    """The summary must flag library gaps, not just print counts."""
    import json
    import importlib.util

    events_dir = tmp_path / "events"
    events_dir.mkdir()
    records = [{"event": "prune", "mode": "masked", "reason": "concat offset", "layer_type": "Conv2d"}
               for _ in range(8)]
    records += [{"event": "prune", "mode": "structural", "layer_type": "Conv2d"} for _ in range(2)]
    records.append({"event": "run_end", "status": "ok", "seconds": 12.0})
    with open(events_dir / "rank0.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    spec = importlib.util.spec_from_file_location(
        "summarize_run", Path(__file__).resolve().parents[1] / "scripts" / "summarize_run.py")
    summarize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summarize)

    report = summarize.build_report(str(tmp_path))
    assert report["status"] == "ok"
    assert abs(report["masked_fraction"] - 0.8) < 1e-9
    assert any("fell back to masking" in note for note in summarize.recommendations(report))
    assert "Pruning coverage" in summarize.render_markdown(report)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
