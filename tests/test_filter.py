"""Tests for Stage 2 — Filter"""

import pytest
from datasets import Dataset

from ols_dataset_prep.filter import filter_dataset


def _make_dataset(n=20):
    return Dataset.from_dict({
        "question": [f"Q{i}" for i in range(n)],
        "answer": [f"A{i}" for i in range(n)],
        "extra_col": [f"extra{i}" for i in range(n)],
    })


def test_column_selection():
    """Only specified columns are retained."""
    ds = filter_dataset(_make_dataset(), columns_keep=["question", "answer"], rows=100)
    assert set(ds.column_names) == {"question", "answer"}
    assert "extra_col" not in ds.column_names


def test_row_sampling():
    """Rows are sampled down to `rows` when dataset is larger."""
    ds = filter_dataset(_make_dataset(100), columns_keep=["question", "answer"], rows=10)
    assert len(ds) == 10


def test_reproducible_sampling():
    """Same random_seed produces same sample."""
    ds1 = filter_dataset(_make_dataset(100), columns_keep=["question"], rows=10, random_seed=42)
    ds2 = filter_dataset(_make_dataset(100), columns_keep=["question"], rows=10, random_seed=42)
    assert ds1["question"] == ds2["question"]


def test_missing_column_skipped():
    """Missing columns are skipped without raising an error."""
    ds = filter_dataset(
        _make_dataset(),
        columns_keep=["question", "nonexistent_col"],
        rows=100,
    )
    assert "question" in ds.column_names
    assert "nonexistent_col" not in ds.column_names


def test_null_rows_removed():
    """Rows where all kept columns are null/empty are removed."""
    raw = Dataset.from_dict({
        "question": ["Q1", "", None, "Q4"],
        "answer": ["A1", "", None, "A4"],
    })
    ds = filter_dataset(raw, columns_keep=["question", "answer"], rows=100)
    assert len(ds) == 2  # only Q1/A1 and Q4/A4 survive
