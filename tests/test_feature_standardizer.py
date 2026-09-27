"""Feature-standardizer cache path (no GPU)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_standardizer import (
    FeatureStandardizer,
    cache_path_from_actor,
    ensure_fitted,
    resolve_standardizer_path,
)


def test_cache_path_from_actor():
    actor = "/home/paretsky/SPECTRA-CompressionAgent/runs/job21168759/agent_checkpoints/latest_best_actor.pt"
    assert cache_path_from_actor(actor).replace("\\", "/").endswith(
        "runs/job21168759/standardizer.pt")
    assert cache_path_from_actor("") == ""


def test_resolve_prefers_existing_actor_cache(monkeypatch, tmp_path):
    run = tmp_path / "job1"
    ckpt = run / "agent_checkpoints"
    ckpt.mkdir(parents=True)
    cache = run / "standardizer.pt"
    cache.write_text("x")
    actor = ckpt / "latest_best_actor.pt"
    actor.write_text("y")
    monkeypatch.delenv("SPECTRA_STANDARDIZER_PATH", raising=False)
    monkeypatch.setenv("SPECTRA_ACTOR_CHECKPOINT_PATH", str(actor))
    got = resolve_standardizer_path(for_write=False)
    assert Path(got) == cache


def test_eval_only_without_cache_stays_unfitted(monkeypatch):
    FeatureStandardizer.reset_instance()
    monkeypatch.delenv("SPECTRA_SKIP_STANDARDIZER", raising=False)
    monkeypatch.delenv("SPECTRA_STANDARDIZER_PATH", raising=False)
    monkeypatch.delenv("SPECTRA_ACTOR_CHECKPOINT_PATH", raising=False)
    std = ensure_fitted({}, "cpu", 38, load_only=True)
    assert std.is_fitted is False
