"""
IMN-FP reader: the statistics and the cell discovery (no torch needed).

    python -m pytest tests/test_imn_fullprec.py -v
"""

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("imn_fullprec_read", str(ROOT / "scripts" / "imn_fullprec_read.py"))
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)


def test_mcnemar_exact_p_known_values():
    assert fp.mcnemar_exact_p(0, 0) == 1.0
    assert fp.mcnemar_exact_p(5, 5) == 1.0
    assert abs(fp.mcnemar_exact_p(10, 2) - 158.0 / 4096.0) < 1e-12
    assert abs(fp.mcnemar_exact_p(0, 7) - 2 * 0.5 ** 7) < 1e-12
    assert fp.mcnemar_exact_p(2000, 2000) == 1.0
    p = fp.mcnemar_exact_p(2100, 1900)
    assert 0.001 < p < 0.003


def test_mcnemar_exact_p_is_symmetric():
    assert abs(fp.mcnemar_exact_p(10, 2) - fp.mcnemar_exact_p(2, 10)) < 1e-15


def test_paired_counts_and_ci():
    a = [1, 1, 0, 0, 1]
    b = [1, 0, 1, 0, 0]
    c = fp.paired_counts(a, b)
    assert c["n"] == 5
    assert c["a_only"] == 2
    assert c["b_only"] == 1
    assert c["both"] == 1
    assert c["neither"] == 1

    diff, lo, hi = fp.paired_diff_ci(a, b)
    z = 1.959963984540054
    se_pp = 100.0 * math.sqrt(3 - 1.0 / 5) / 5
    assert abs(diff - 20.0) < 1e-9
    assert abs(lo - (20.0 - z * se_pp)) < 1e-9
    assert abs(hi - (20.0 + z * se_pp)) < 1e-9


def test_paired_counts_length_mismatch_raises():
    with pytest.raises(ValueError):
        fp.paired_counts([1, 0, 1], [1, 0])


def test_checks_are_inclusive_at_the_registered_bounds():
    assert fp.check_test(0.7526, 0.752) is True
    assert fp.check_test(0.7527, 0.752) is False
    assert fp.check_val(-8.71, -8.73) is True
    assert fp.check_val(-8.70, -8.73) is False


def test_find_checkpoint_zero_one_two(tmp_path):
    run = tmp_path / "run-x"
    run.mkdir()
    with pytest.raises(Exception):
        fp.find_checkpoint(str(tmp_path), "x")  # zero matches

    first = run / "recovery__a.pt"
    first.write_bytes(b"")
    assert Path(fp.find_checkpoint(str(tmp_path), "x")) == first

    (run / "recovery__b.pt").write_bytes(b"")
    with pytest.raises(Exception):
        fp.find_checkpoint(str(tmp_path), "x")  # two matches
