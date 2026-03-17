"""
Pipeline Example — Structured Output (Recipe 6)

Demonstrates how to use ols-dataset-prep to produce a dataset
for Unsloth Studio's "Structured Outputs (Jinja)" recipe.

What this recipe does:
  Trains a model to produce JSON-formatted responses following a schema.
  Useful when you need the model to return structured data (dicts, lists)
  rather than free-form text.

Datasets that work well with this recipe:
  - demelin/moral_stories     (scenario components as structured fields)
  - any dataset with multiple structured columns that map to a JSON schema

Target in Unsloth Studio: "Structured Outputs (Jinja)" recipe block.

Usage:
  python -m pipelines.structured_output
  # or via CLI:
  ols-prep run --recipe structured_outputs_jinja
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

HF_REPO = "demelin/moral_stories"
SPLIT = "train"
ROWS = 2000
COLUMNS_KEEP = [
    "norm", "situation", "intention",
    "moral_action", "immoral_action",
    "moral_consequence", "immoral_consequence",
]
AUGMENTATION = "structured_output"
AUGMENTATION_LLM = "anthropic"
OUTPUT_NAME = "your-hf-username/ols-moral-stories-2k"
LOCAL_SUBDIR = "06-structured-outputs"
DATASET_ID = "moral-stories"


def run():
    print(f"[structured_output] Fetching {HF_REPO}...")
    dataset = fetch_dataset(
        hf_repo=HF_REPO,
        split=SPLIT,
        rows=ROWS,
        random_seed=42,
        trust_remote_code=True,   # needed — legacy .py script
    )
    print(f"  Fetched {len(dataset)} rows")

    dataset = filter_dataset(dataset, columns_keep=COLUMNS_KEEP, rows=ROWS)
    print(f"  Filtered → {len(dataset)} rows, columns: {dataset.column_names}")

    # Phase 3: augmentation will format rows as structured JSON
    dataset = augment_dataset(dataset, augmentation_type=AUGMENTATION, augmentation_llm=AUGMENTATION_LLM)

    report = validate_dataset(
        dataset=dataset,
        required_columns=COLUMNS_KEEP,
        min_rows=int(ROWS * 0.8),
        output_type="structured",
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
        recipe_target="structured_outputs_jinja",
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
        "recipe_target": "structured_outputs_jinja",
    })
    print(f"  Delivered → {result['local_path']}")
    print(f"  HF Hub   → {result['hf_url']}")


if __name__ == "__main__":
    run()
