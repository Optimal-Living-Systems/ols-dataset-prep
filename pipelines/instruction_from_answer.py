"""
Pipeline Example — Instruction from Answer (Recipe 1)

Demonstrates how to use ols-dataset-prep to produce a dataset
for Unsloth Studio's "Instruction from Answer" recipe.

What this recipe does:
  Given a seed column of existing answers/responses, generate matching
  instructions so the model learns to follow that style of request.
  Useful for any dataset where you have good responses but no instructions.

Datasets that work well with this recipe:
  - allenai/prosocial-dialog      (context + response columns)
  - allenai/social_bias_frames    (post + framing columns)
  - Any Q&A dataset with answer columns

Target in Unsloth Studio: "Instruction from Answer" recipe block.

Usage:
  python -m pipelines.instruction_from_answer
  # or via CLI:
  ols-prep run --recipe instruction_from_answer
"""

import os
import sys
from pathlib import Path

# Allow running from repo root without installing
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
# Edit these values to point at your dataset and HF account.

HF_REPO = "allenai/prosocial-dialog"
SPLIT = "train"
ROWS = 2000
COLUMNS_KEEP = ["context", "response", "rots", "safety_label"]
AUGMENTATION = "instruction_from_answer"
AUGMENTATION_LLM = "ollama"               # "ollama" | "anthropic" | "litellm"
OUTPUT_NAME = "your-hf-username/ols-prosocial-dialog-2k"
LOCAL_SUBDIR = "01-instruction-from-answer"
DATASET_ID = "prosocial-dialog"


def run():
    print(f"[instruction_from_answer] Fetching {HF_REPO}...")
    dataset = fetch_dataset(
        hf_repo=HF_REPO,
        split=SPLIT,
        rows=ROWS,
        random_seed=42,
        trust_remote_code=True,
    )
    print(f"  Fetched {len(dataset)} rows")

    dataset = filter_dataset(dataset, columns_keep=COLUMNS_KEEP, rows=ROWS)
    print(f"  Filtered → {len(dataset)} rows, columns: {dataset.column_names}")

    # Phase 3: augmentation will generate instructions from the response column
    dataset = augment_dataset(dataset, augmentation_type=AUGMENTATION, augmentation_llm=AUGMENTATION_LLM)

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
        recipe_target="instruction_from_answer",
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
        "recipe_target": "instruction_from_answer",
    })
    print(f"  Delivered → {result['local_path']}")
    print(f"  HF Hub   → {result['hf_url']}")


if __name__ == "__main__":
    run()
