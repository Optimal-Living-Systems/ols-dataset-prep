"""
Stage 5 — Deliver

Saves the processed dataset as snappy-compressed parquet locally,
then pushes to HuggingFace Hub. Always snappy — never zstd, never gzip.
That is the entire reason this pipeline exists.

Also generates a dataset card (README) for the HF Hub repo with OLS
attribution, source dataset credit, and CC-BY-4.0 license.
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from datasets import Dataset
from huggingface_hub import DatasetCard, DatasetCardData

logger = logging.getLogger(__name__)


def deliver_dataset(
    dataset: Dataset,
    dataset_id: str,
    output_name: str,
    local_subdir: str,
    hf_repo: str,
    augmentation: str,
    recipe_target: str,
    ols_project: str,
    hf_token: Optional[str] = None,
    push_to_hub: bool = True,
    local_base: Optional[str] = None,
) -> dict:
    """
    Save dataset as snappy parquet locally and push to HuggingFace Hub.

    Args:
        dataset: Validated Dataset from Stage 4.
        dataset_id: Short ID from datasets.yaml (e.g. "mmlu-sociology").
        output_name: HF Hub destination (e.g. "your-hf-username/ols-mmlu-sociology").
        local_subdir: Subdirectory under OUTPUT_BASE_DIR (e.g. "01-instruction-from-answer").
        hf_repo: Source HF repo for attribution in dataset card.
        augmentation: Augmentation type applied (for card and manifest).
        recipe_target: Unsloth recipe this feeds.
        ols_project: OLS project this belongs to.
        hf_token: HF token. Reads HF_TOKEN env var if None.
        push_to_hub: Whether to push to HF Hub (set False for local-only runs).
        local_base: Override for OUTPUT_BASE_DIR env var.

    Returns:
        dict with local_path, hf_url, rows, processed_at.
    """
    token = hf_token or os.getenv("HF_TOKEN")
    base_dir = local_base or os.getenv("OUTPUT_BASE_DIR", str(Path.home() / "ols-pipeline" / "unsloth-datasets"))

    # ── Step 1: Save locally as snappy parquet ────────────────────────────
    output_dir = Path(base_dir) / local_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / f"{dataset_id}.parquet"

    logger.info(f"Saving snappy parquet → {local_path}")
    dataset.to_parquet(str(local_path), compression="snappy")
    logger.info(f"Saved {len(dataset)} rows ({local_path.stat().st_size / 1024:.1f} KB)")

    processed_at = datetime.now(timezone.utc).isoformat()
    hf_url = f"https://huggingface.co/datasets/{output_name}"

    # ── Step 2: Push to HuggingFace Hub ───────────────────────────────────
    # token=None is fine: huggingface_hub falls back to cached login
    # (huggingface-cli login) automatically when token is not provided.
    if push_to_hub:
        if not token:
            logger.info("HF_TOKEN not in env — attempting push with cached huggingface-cli token")
        logger.info(f"Pushing to HuggingFace Hub → {output_name}")
        try:
            dataset.push_to_hub(
                output_name,
                token=token,
                commit_message=f"OLS Dataset Prep: {dataset_id} processed {processed_at[:10]}",
            )
            _push_dataset_card(
                output_name=output_name,
                dataset_id=dataset_id,
                hf_repo=hf_repo,
                augmentation=augmentation,
                recipe_target=recipe_target,
                ols_project=ols_project,
                rows=len(dataset),
                processed_at=processed_at,
                token=token,
            )
            logger.info(f"Pushed → {hf_url}")
        except Exception as e:
            logger.error(
                f"Hub push failed for {output_name}\n"
                f"  → Cause: {e}\n"
                f"  → Local file saved at {local_path} — you can push manually later\n"
                f"  → Run: ols-prep push {dataset_id}"
            )

    return {
        "local_path": str(local_path),
        "hf_url": hf_url,
        "rows": len(dataset),
        "processed_at": processed_at,
    }


def _push_dataset_card(
    output_name: str,
    dataset_id: str,
    hf_repo: str,
    augmentation: str,
    recipe_target: str,
    ols_project: str,
    rows: int,
    processed_at: str,
    token: str,
) -> None:
    """Generate and push a dataset card to HF Hub."""
    card_content = f"""---
license: cc-by-4.0
tags:
  - ols
  - optimal-living-systems
  - {ols_project.lower()}
  - {recipe_target}
---

# {output_name.split("/")[-1]}

Processed by the [OLS Dataset Preparation Pipeline](https://github.com/Optimal-Living-Systems/ols-dataset-prep).

## Dataset Details

| Field | Value |
|-------|-------|
| Source | [{hf_repo}](https://huggingface.co/datasets/{hf_repo}) |
| Rows | {rows} |
| Augmentation | {augmentation} |
| Recipe Target | {recipe_target} |
| OLS Project | {ols_project} |
| Processed | {processed_at[:10]} |
| License | CC-BY-4.0 |

## About OLS

[Optimal Living Systems](https://github.com/Optimal-Living-Systems) is a mutual aid nonprofit
building open-source AI infrastructure for community organizations.

This dataset is part of the OLS AI Lab training data stack.
"""
    try:
        card = DatasetCard(card_content)
        card.push_to_hub(output_name, token=token)
    except Exception as e:
        # Card push failure is non-fatal — dataset itself was already pushed
        logger.warning(f"Dataset card push failed (non-fatal): {e}")
