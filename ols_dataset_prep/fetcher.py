"""
Stage 1 — Fetch

Universal HuggingFace dataset loader. Handles every format the pipeline
encounters without requiring the caller to know what format they're dealing with:
  - Standard parquet (no issues)
  - zstd-compressed parquet (HF's new default — not all downstream tools read it directly)
  - Legacy .py loading scripts — fallback: discover and load raw data files directly
  - Multi-config datasets (subset parameter)
  - Gated/private datasets (HF_TOKEN)
  - Very large datasets (streaming mode, take first N rows)

All errors surface as DatasetFetchError with a human-readable message
and a suggested fix. Never a raw traceback as user-facing output.
"""

import os
import logging
from typing import Optional

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

# Raw file extensions we can load directly when a .py script is blocking us
_LOADABLE_EXTENSIONS = {
    ".jsonl": "json",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
}

# Preference order when multiple data files exist — "full" beats split-specific
_FILE_PREFERENCE = ["full", "train", "all"]


class DatasetFetchError(Exception):
    """Raised when a dataset cannot be fetched. Message is human-readable."""
    pass


def fetch_dataset(
    hf_repo: str,
    split: str = "train",
    subset: Optional[str] = None,
    rows: int = 2000,
    random_seed: int = 42,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = True,
    streaming: bool = False,
    data_url: Optional[str] = None,
    data_url_file: Optional[str] = None,
) -> Dataset:
    """
    Fetch a HuggingFace dataset and return a Dataset object.

    Primary path: datasets.load_dataset() — handles zstd transparently.
    Fallback path: when a legacy .py script blocks loading, list the repo's
    raw data files and load them directly (json/jsonl/csv/parquet).

    Args:
        hf_repo: HuggingFace dataset repo ID (e.g. "allenai/prosocial-dialog")
        split: Dataset split to load ("train", "test", "validation")
        subset: Config/subset name for multi-config datasets (e.g. "sociology")
        rows: Maximum rows to return. Samples if dataset is larger.
        random_seed: Seed for reproducible row sampling.
        hf_token: HuggingFace token. Reads HF_TOKEN env var if None.
        trust_remote_code: Kept for API compatibility; ignored in datasets >= 4.4.
        streaming: Use streaming mode for very large datasets (> 500k rows).
        data_url: Direct download URL for datasets with no HF-hosted data files.
            Used as last-resort fallback when both load_dataset and raw-file
            approaches fail. Supports .tgz archives containing TSV/CSV/JSONL.
        data_url_file: Filename to extract from a .tgz archive (e.g. "data.tsv").

    Returns:
        Dataset with at most `rows` rows.

    Raises:
        DatasetFetchError: With a human-readable message and suggested fix.
    """
    token = hf_token or os.getenv("HF_TOKEN")

    load_kwargs = {
        "path": hf_repo,
        "split": split,
        "token": token,
        "streaming": streaming,
    }
    if subset:
        load_kwargs["name"] = subset

    logger.debug(f"Loading {hf_repo} (split={split}, subset={subset})")

    try:
        raw = load_dataset(**load_kwargs)
    except Exception as primary_exc:
        msg = str(primary_exc)

        # Legacy .py script: try loading raw data files from the repo directly
        if "scripts are no longer supported" in msg or "loading script" in msg.lower():
            logger.info(
                f"{hf_repo} has a legacy .py loading script — "
                f"falling back to direct raw file load"
            )
            try:
                raw = _load_raw_files(hf_repo, token, primary_exc)
            except DatasetFetchError:
                # No raw files in repo — try external data_url if provided
                if data_url:
                    logger.info(
                        f"No raw files in repo — falling back to external data_url: {data_url}"
                    )
                    raw = _load_from_url(data_url, data_url_file, primary_exc)
                else:
                    raise
        else:
            _raise_fetch_error(hf_repo, split, subset, primary_exc)

    # Convert IterableDataset → Dataset by taking the first `rows` items
    if streaming:
        try:
            raw = Dataset.from_list(list(raw.take(rows)))
            return raw
        except Exception as e:
            _raise_fetch_error(hf_repo, split, subset, e)

    # Sample down to `rows` if the dataset is larger — deterministic via seed
    if len(raw) > rows:
        raw = raw.shuffle(seed=random_seed).select(range(rows))

    return raw


def _load_raw_files(hf_repo: str, token: Optional[str], original_exc: Exception) -> Dataset:
    """
    Fallback loader for legacy-script datasets.

    Lists the repo's files, finds loadable data files, and loads them directly
    using datasets.load_dataset with the appropriate format type.
    Prefers files with 'full' or 'train' in their name.
    """
    from huggingface_hub import HfApi, hf_hub_download

    try:
        api = HfApi(token=token)
        all_files = list(api.list_repo_files(hf_repo, repo_type="dataset"))
    except Exception as e:
        raise DatasetFetchError(
            f"Legacy script fallback failed for {hf_repo} — could not list repo files\n"
            f"  → Original error: {original_exc}\n"
            f"  → File listing error: {e}"
        ) from e

    # Find data files we can load directly
    candidates = []
    for f in all_files:
        _, ext = os.path.splitext(f.lower())
        if ext in _LOADABLE_EXTENSIONS and not f.startswith("."):
            candidates.append(f)

    if not candidates:
        raise DatasetFetchError(
            f"Legacy script fallback failed for {hf_repo} — no loadable data files found\n"
            f"  → The dataset only has a .py script and no raw data files\n"
            f"  → Consider finding a parquet mirror of this dataset on HF Hub"
        ) from original_exc

    # Prefer full/train files; otherwise take the first candidate
    chosen = None
    for pref in _FILE_PREFERENCE:
        match = next((f for f in candidates if pref in f.lower()), None)
        if match:
            chosen = match
            break
    if not chosen:
        chosen = candidates[0]

    _, ext = os.path.splitext(chosen.lower())
    fmt = _LOADABLE_EXTENSIONS[ext]
    logger.info(f"Loading raw file directly: {chosen} (format={fmt})")

    try:
        # Build the full URL for the file
        url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{chosen}"
        raw = load_dataset(fmt, data_files=url, split="train", token=token)
        logger.info(f"Raw file load succeeded: {len(raw)} rows from {chosen}")
        return raw
    except Exception as e:
        raise DatasetFetchError(
            f"Legacy script fallback failed for {hf_repo}\n"
            f"  → Tried loading: {chosen}\n"
            f"  → Error: {e}\n"
            f"  → Consider finding a parquet-format mirror of this dataset"
        ) from e


def _load_from_url(
    data_url: str,
    data_url_file: Optional[str],
    original_exc: Exception,
) -> Dataset:
    """
    Last-resort loader for datasets with no HF-hosted data files.

    Downloads the file at data_url. If it's a .tgz, extracts data_url_file
    from it. Supports TSV, CSV, and JSONL as the final format.
    """
    import tempfile
    import tarfile
    import urllib.request

    logger.info(f"Downloading external data source: {data_url}")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, os.path.basename(data_url))
            urllib.request.urlretrieve(data_url, archive_path)
            logger.info(f"Downloaded {os.path.getsize(archive_path) // 1024} KB")

            # Extract tgz if needed
            if archive_path.endswith(".tgz") or archive_path.endswith(".tar.gz"):
                with tarfile.open(archive_path, "r:gz") as tar:
                    members = tar.getnames()
                    logger.debug(f"Archive contents: {members}")

                    if data_url_file:
                        # Find the requested file (match by basename or full path)
                        target = next(
                            (m for m in members if m.endswith(data_url_file)),
                            None,
                        )
                        if not target:
                            raise DatasetFetchError(
                                f"File '{data_url_file}' not found in archive\n"
                                f"  → Archive contains: {members}"
                            )
                        tar.extract(target, tmpdir)
                        data_file = os.path.join(tmpdir, target)
                    else:
                        # Extract all, find first loadable file
                        tar.extractall(tmpdir)
                        data_file = next(
                            (
                                os.path.join(root, f)
                                for root, _, files in os.walk(tmpdir)
                                for f in files
                                if os.path.splitext(f.lower())[1] in _LOADABLE_EXTENSIONS
                            ),
                            None,
                        )
                        if not data_file:
                            raise DatasetFetchError(
                                f"No loadable files found in archive from {data_url}"
                            )
            else:
                data_file = archive_path

            _, ext = os.path.splitext(data_file.lower())
            fmt = _LOADABLE_EXTENSIONS.get(ext)
            if not fmt:
                raise DatasetFetchError(
                    f"Cannot load file type '{ext}' from {data_url}"
                )

            load_kwargs: dict = {"path": fmt, "data_files": data_file, "split": "train"}
            if ext in (".tsv",):
                load_kwargs["delimiter"] = "\t"

            raw = load_dataset(**load_kwargs)
            logger.info(f"External URL load succeeded: {len(raw)} rows")
            return raw

    except DatasetFetchError:
        raise
    except Exception as e:
        raise DatasetFetchError(
            f"External data_url load failed: {data_url}\n"
            f"  → Error: {e}\n"
            f"  → Original HF error: {original_exc}"
        ) from e


def _raise_fetch_error(hf_repo: str, split: str, subset: Optional[str], exc: Exception) -> None:
    """Translate a raw datasets exception into a DatasetFetchError with a fix hint."""
    msg = str(exc)

    if "404" in msg or "doesn't exist" in msg.lower():
        raise DatasetFetchError(
            f"Dataset not found: {hf_repo}\n"
            f"  → Check the repo ID spelling on huggingface.co/datasets\n"
            f"  → If it's a private/gated dataset, set HF_TOKEN in your .env"
        ) from exc

    if "401" in msg or "403" in msg or "authentication" in msg.lower():
        raise DatasetFetchError(
            f"Authentication failed for {hf_repo}\n"
            f"  → Set HF_TOKEN in your .env file\n"
            f"  → Ensure your token has read access to this dataset"
        ) from exc

    if "config" in msg.lower() or "subset" in msg.lower():
        raise DatasetFetchError(
            f"Dataset {hf_repo} requires a subset/config name\n"
            f"  → Set 'subset' in datasets.yaml for this dataset\n"
            f"  → Available configs can be found on the dataset's HF page"
        ) from exc

    if "split" in msg.lower():
        raise DatasetFetchError(
            f"Split '{split}' not found in {hf_repo}\n"
            f"  → Check available splits on huggingface.co/datasets/{hf_repo}\n"
            f"  → Update 'split' in datasets.yaml"
        ) from exc

    # Generic fallback — still human-readable
    raise DatasetFetchError(
        f"Failed to load {hf_repo} (split={split}, subset={subset})\n"
        f"  → Cause: {type(exc).__name__}: {msg}\n"
        f"  → Check your .env for HF_TOKEN\n"
        f"  → Check internet connectivity and HuggingFace status"
    ) from exc
