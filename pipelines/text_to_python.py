"""
Pipeline Example — Text to Python

Demonstrates how to use ols-dataset-prep to produce text-to-code training
data for downstream fine-tuning workflows.

What this target does:
  Trains a model to generate Python code from natural language instructions.
  Useful for code-generation fine-tuning on domain-specific tasks.

Datasets that work well with this target:
  - iamtarun/python_code_instructions_18k_alpaca  (instruction + input + output)
  - flytech/python-codes-25k
  - any dataset with (instruction, code) pairs in Alpaca format

Runtime target label: `text_to_python`

Usage:
  python -m pipelines.text_to_python
  # or via CLI:
  ols-prep run --recipe text_to_python
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

HF_REPO = "iamtarun/python_code_instructions_18k_alpaca"
SPLIT = "train"
ROWS = 2000
COLUMNS_KEEP = ["instruction", "input", "output"]
AUGMENTATION = "none"             # dataset already has clean instruction/code pairs
AUGMENTATION_LLM = None
OUTPUT_NAME = "your-hf-username/ols-python-code-2k"
LOCAL_SUBDIR = "04-text-to-python"
DATASET_ID = "python-code-instructions"


def run():
    print(f"[text_to_python] Fetching {HF_REPO}...")
    dataset = fetch_dataset(
        hf_repo=HF_REPO,
        split=SPLIT,
        rows=ROWS,
        random_seed=42,
    )
    print(f"  Fetched {len(dataset)} rows")

    dataset = filter_dataset(dataset, columns_keep=COLUMNS_KEEP, rows=ROWS)
    print(f"  Filtered → {len(dataset)} rows, columns: {dataset.column_names}")

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
        recipe_target="text_to_python",
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
        "recipe_target": "text_to_python",
    })
    print(f"  Delivered → {result['local_path']}")
    print(f"  HF Hub   → {result['hf_url']}")


if __name__ == "__main__":
    run()
