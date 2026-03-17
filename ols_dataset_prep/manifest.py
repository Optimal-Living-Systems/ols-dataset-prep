"""
Manifest

Reads and writes manifest.json at the repo root. Tracks every dataset
the pipeline has processed so every run is reproducible and auditable.

Never overwrites existing entries — always merges. Use --force at the CLI
level to re-process a dataset that already has a complete entry.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Resolve manifest path relative to this file's package root (one level up)
_MANIFEST_PATH = Path(__file__).parent.parent / "manifest.json"


def load_manifest(manifest_path: Optional[Path] = None) -> dict:
    """Load manifest.json, returning {} if it doesn't exist or is empty."""
    path = manifest_path or _MANIFEST_PATH
    if not path.exists():
        return {}
    try:
        text = path.read_text().strip()
        return json.loads(text) if text else {}
    except json.JSONDecodeError as e:
        logger.warning(f"manifest.json is malformed, starting fresh: {e}")
        return {}


def save_manifest(manifest: dict, manifest_path: Optional[Path] = None) -> None:
    """Write manifest to disk as pretty-printed JSON."""
    path = manifest_path or _MANIFEST_PATH
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def update_manifest(
    dataset_id: str,
    entry: dict[str, Any],
    manifest_path: Optional[Path] = None,
) -> None:
    """
    Merge a new entry into manifest.json for the given dataset_id.

    If an entry already exists, it is updated (not replaced) so that
    fields added by previous runs are preserved.
    """
    manifest = load_manifest(manifest_path)
    existing = manifest.get(dataset_id, {})
    existing.update(entry)
    manifest[dataset_id] = existing
    save_manifest(manifest, manifest_path)
    logger.debug(f"Manifest updated for {dataset_id}")


def is_complete(dataset_id: str, manifest_path: Optional[Path] = None) -> bool:
    """Return True if this dataset has already been processed successfully."""
    manifest = load_manifest(manifest_path)
    entry = manifest.get(dataset_id, {})
    return entry.get("validation_passed") is True and entry.get("processed_at") is not None


def get_entry(dataset_id: str, manifest_path: Optional[Path] = None) -> Optional[dict]:
    """Return the manifest entry for a dataset_id, or None if not found."""
    return load_manifest(manifest_path).get(dataset_id)
