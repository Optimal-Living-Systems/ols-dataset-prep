"""Tests for Stage 4 — Validate"""

import pytest
from datasets import Dataset

from ols_dataset_prep.validator import validate_dataset, ValidationReport


def _make_dataset(n=10, include_pii=False):
    rows = {
        "instruction": [f"Instruction {i}" for i in range(n)],
        "response": [f"Response {i}" for i in range(n)],
    }
    if include_pii:
        rows["response"][0] = "Contact me at test@example.com or 555-123-4567"
    return Dataset.from_dict(rows)


def test_valid_dataset_passes():
    """A clean dataset passes all checks."""
    report = validate_dataset(_make_dataset(10), required_columns=["instruction", "response"], min_rows=8)
    assert report.passed is True
    assert len(report.errors) == 0


def test_missing_column_fails():
    """Missing required column is a hard failure."""
    ds = Dataset.from_dict({"instruction": ["Q1"]})
    report = validate_dataset(ds, required_columns=["instruction", "response"], min_rows=1)
    assert report.passed is False
    assert report.checks["required_columns_present"] is False


def test_below_min_rows_fails():
    """Row count below 80% threshold is a hard failure."""
    report = validate_dataset(_make_dataset(5), required_columns=["instruction", "response"], min_rows=9)
    assert report.passed is False
    assert report.checks["min_row_count"] is False


def test_degenerate_pairs_warns():
    """instruction == response is a warning, not a hard failure."""
    ds = Dataset.from_dict({
        "instruction": ["same text", "different"],
        "response": ["same text", "response"],
    })
    report = validate_dataset(ds, required_columns=["instruction", "response"], min_rows=1)
    assert report.passed is True  # warning, not failure
    assert any("degenerate" in w.lower() or "instruction == response" in w.lower() for w in report.warnings)


def test_pii_detected_is_warning():
    """PII detection is a warning, not a hard failure."""
    report = validate_dataset(
        _make_dataset(10, include_pii=True),
        required_columns=["instruction", "response"],
        min_rows=8,
    )
    assert report.passed is True   # PII is a warning, not a blocker
    assert any("PII" in w or "pii" in w.lower() for w in report.warnings)


def test_report_never_raises():
    """validate_dataset must not raise exceptions under any input."""
    try:
        validate_dataset(Dataset.from_dict({}), required_columns=[], min_rows=0)
    except Exception as e:
        pytest.fail(f"validate_dataset raised an exception: {e}")
