"""
Pipeline Example — Text to SQL

Demonstrates how to use ols-dataset-prep to produce text-to-SQL training
data for downstream fine-tuning workflows.

What this target does:
  Trains a model to translate natural language questions into SQL queries
  given a database schema context.

Datasets that work well with this target:
  - b-mc2/sql-create-context   (question + context + answer columns)
  - gretelai/synthetic_text_to_sql
  - any dataset with (question, schema, sql) structure

Runtime target label: `text_to_sql`

Usage:
  python -m pipelines.text_to_sql
  # or via CLI:
  ols-prep run --recipe text_to_sql
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ols_dataset_prep.fetcher import fetch_dataset
from ols_dataset_prep.filter import filter_dataset
from ols_dataset_prep.augmentor import augment_dataset
from ols_dataset_prep.validator import validate_dataset
from ols_dataset_prep.deliverer import deliver_dataset
from ols_dataset_prep import manifest as manifest_module

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────

HF_REPO = "b-mc2/sql-create-context"
SPLIT = "train"
ROWS = 2000
COLUMNS_KEEP = ["question", "context", "answer"]
AUGMENTATION = "none"             # no LLM augmentation needed for this recipe
AUGMENTATION_LLM = None
OUTPUT_NAME = "your-hf-username/ols-sql-context-2k"
LOCAL_SUBDIR = "05-text-to-sql"
DATASET_ID = "sql-create-context"


def run():
    print(f"[text_to_sql] Fetching {HF_REPO}...")
    dataset = fetch_dataset(
        hf_repo=HF_REPO,
        split=SPLIT,
        rows=ROWS,
        random_seed=42,
    )
    print(f"  Fetched {len(dataset)} rows")

    dataset = filter_dataset(dataset, columns_keep=COLUMNS_KEEP, rows=ROWS)
    print(f"  Filtered → {len(dataset)} rows, columns: {dataset.column_names}")

    # No augmentation for text-to-SQL — the dataset already has paired examples
    dataset = augment_dataset(dataset, augmentation_type=AUGMENTATION)

    report = validate_dataset(
        dataset=dataset,
        required_columns=COLUMNS_KEEP,
        min_rows=int(ROWS * 0.8),
    )
    print(f"  Validation: {'PASSED' if report.passed else 'FAILED'}")
    if not report.passed:
        for err in report.errors:
            print(f"  ERROR: {err}")
        return

    result = deliver_dataset(
        dataset=dataset,
        dataset_id=DATASET_ID,
        output_name=OUTPUT_NAME,
        local_subdir=LOCAL_SUBDIR,
        hf_repo=HF_REPO,
        augmentation=AUGMENTATION,
        recipe_target="text_to_sql",
        ols_project="my-project",
        push_to_hub=bool(os.getenv("HF_TOKEN")),
    )

    manifest_module.update_manifest(DATASET_ID, {
        "hf_repo": HF_REPO,
        "output_name": OUTPUT_NAME,
        "local_path": result["local_path"],
        "hf_url": result["hf_url"],
        "rows_out": result["rows"],
        "validation_passed": report.passed,
        "processed_at": result["processed_at"],
        "recipe_target": "text_to_sql",
    })
    print(f"  Delivered → {result['local_path']}")
    print(f"  HF Hub   → {result['hf_url']}")


if __name__ == "__main__":
    run()
