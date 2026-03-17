"""
Stage 2 — Filter

Trims a fetched dataset to only the columns and rows the pipeline needs.
Runs after fetch, before augmentation.

Operations in order:
  1. Column whitelist — drop everything not in columns_keep
  2. Null/empty row removal — drop rows where any kept column is null or blank
  3. Row sampling — if dataset is still larger than `rows`, sample with seed
"""

import logging
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


def filter_dataset(
    dataset: Dataset,
    columns_keep: list[str],
    rows: int,
    random_seed: int = 42,
) -> Dataset:
    """
    Filter a Dataset to the specified columns and row count.

    Args:
        dataset: Input Dataset (already fetched)
        columns_keep: Column names to retain. Others are dropped.
        rows: Target row count. Samples if dataset is larger.
        random_seed: Seed for reproducible sampling.

    Returns:
        Filtered Dataset.
    """
    original_len = len(dataset)

    # ── Step 1: Column selection ───────────────────────────────────────────
    available = set(dataset.column_names)
    requested = set(columns_keep)
    missing = requested - available

    if missing:
        logger.warning(
            f"Columns not found in dataset (will be skipped): {sorted(missing)}\n"
            f"  → Available columns: {sorted(available)}"
        )

    to_keep = [c for c in columns_keep if c in available]
    to_drop = [c for c in dataset.column_names if c not in to_keep]

    if to_drop:
        dataset = dataset.remove_columns(to_drop)
        logger.debug(f"Dropped {len(to_drop)} columns: {to_drop}")

    # ── Step 2: Remove null/empty rows ────────────────────────────────────
    def _row_is_valid(row):
        for col in to_keep:
            val = row.get(col)
            if val is None:
                return False
            if isinstance(val, str) and val.strip() == "":
                return False
        return True

    before_null_drop = len(dataset)
    dataset = dataset.filter(_row_is_valid)
    dropped_nulls = before_null_drop - len(dataset)
    if dropped_nulls:
        logger.info(f"Dropped {dropped_nulls} null/empty rows")

    # ── Step 3: Row count enforcement ─────────────────────────────────────
    if len(dataset) > rows:
        dataset = dataset.shuffle(seed=random_seed).select(range(rows))
        logger.debug(f"Sampled {rows} rows from {len(dataset) + rows} (seed={random_seed})")
    elif len(dataset) < rows:
        logger.warning(
            f"Dataset has fewer rows than requested: {len(dataset)} < {rows}\n"
            f"  → Using all available rows. This is fine for small datasets."
        )

    logger.info(
        f"Filter complete: {original_len} → {len(dataset)} rows, "
        f"{len(to_keep)} columns retained"
    )
    return dataset
