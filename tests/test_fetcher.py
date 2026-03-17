"""Tests for Stage 1 — Fetch"""

import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset

from ols_dataset_prep.fetcher import fetch_dataset, DatasetFetchError


def _mock_dataset(n=10):
    return Dataset.from_dict({
        "question": [f"Question {i}" for i in range(n)],
        "answer": [f"Answer {i}" for i in range(n)],
    })


def test_fetch_returns_dataset():
    """fetch_dataset returns a Dataset when load_dataset succeeds."""
    with patch("ols_dataset_prep.fetcher.load_dataset", return_value=_mock_dataset(100)):
        result = fetch_dataset("fake/dataset", rows=10)
    assert isinstance(result, Dataset)


def test_fetch_samples_to_rows():
    """fetch_dataset samples down to `rows` when dataset is larger."""
    with patch("ols_dataset_prep.fetcher.load_dataset", return_value=_mock_dataset(500)):
        result = fetch_dataset("fake/dataset", rows=50)
    assert len(result) == 50


def test_fetch_keeps_all_when_smaller():
    """fetch_dataset keeps all rows when dataset has fewer than `rows`."""
    with patch("ols_dataset_prep.fetcher.load_dataset", return_value=_mock_dataset(5)):
        result = fetch_dataset("fake/dataset", rows=100)
    assert len(result) == 5


def test_fetch_raises_on_404():
    """fetch_dataset raises DatasetFetchError with human-readable message on 404."""
    with patch("ols_dataset_prep.fetcher.load_dataset", side_effect=Exception("404 not found")):
        with pytest.raises(DatasetFetchError) as exc_info:
            fetch_dataset("nonexistent/dataset")
    assert "not found" in str(exc_info.value).lower()


def test_fetch_raises_on_auth_error():
    """fetch_dataset raises DatasetFetchError with auth hint on 401."""
    with patch("ols_dataset_prep.fetcher.load_dataset", side_effect=Exception("401 authentication")):
        with pytest.raises(DatasetFetchError) as exc_info:
            fetch_dataset("private/dataset")
    assert "HF_TOKEN" in str(exc_info.value)


def test_fetch_error_never_raw_traceback():
    """DatasetFetchError message must be human-readable, not a raw exception type."""
    with patch("ols_dataset_prep.fetcher.load_dataset", side_effect=Exception("some internal error xyz")):
        with pytest.raises(DatasetFetchError) as exc_info:
            fetch_dataset("fake/dataset")
    # Should contain a suggested fix, not just the raw exception
    assert "→" in str(exc_info.value)
