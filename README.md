# ols-dataset-prep

A production-grade dataset preparation pipeline for ML fine-tuning.

Fetches any HuggingFace dataset, resolves common compatibility issues
automatically, augments with LLMs, validates quality, and delivers clean
snappy-compressed parquet ready for any fine-tuning workflow.

Built for orchestration with Kestra as part of the Optimal Living Systems AI
Lab stack. Works standalone from the command line.

---

## What It Does

- Fetches datasets from HuggingFace, including multi-config, gated/private,
  large, zstd-compressed, and legacy-script repositories
- Filters columns and rows deterministically
- Optionally augments rows with LLM pipelines
- Validates schema, row counts, structured-output integrity, and basic PII
  signals
- Delivers clean parquet locally and/or to HuggingFace Hub with provenance in
  `manifest.json`

---

## Why This Exists

HuggingFace datasets are not packaged consistently across the ecosystem.
Production fine-tuning and research pipelines regularly run into:

- zstd-compressed parquet that some downstream tooling does not handle cleanly
- legacy `.py` loading scripts deprecated by newer `datasets` releases
- repositories that require direct file loading or external data sources

This pipeline normalizes those cases by routing datasets through
`datasets.load_dataset()` and compatible fallback loaders, then writing the
result back out as snappy parquet for broad downstream compatibility.

---

## Architecture

```text
Input: HF Dataset ID + config/datasets.yaml
              ↓
  Stage 1: Fetch      — load_dataset(), raw-file fallback, external data fallback
              ↓
  Stage 2: Filter     — column whitelist, row sampling, null removal
              ↓
  Stage 3: Augment    — Ollama / Anthropic-backed enrichment pipelines
              ↓
  Stage 4: Validate   — schema, PII scan, JSON checks, quality gates
              ↓
  Stage 5: Deliver    — snappy parquet locally + push to HF Hub
```

---

## Works With

- Any downstream ML training pipeline that consumes parquet or HuggingFace Hub
  datasets
- Kestra orchestration, custom Python workflows, RAG ingestion jobs, research
  notebooks, and fine-tuning stacks such as Unsloth Studio

---

## Installation

```bash
git clone https://github.com/Optimal-Living-Systems/ols-dataset-prep
cd ols-dataset-prep

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

---

## Quick Start

```bash
# Preview a dataset (fetch + filter, show 3 rows, no save)
ols-prep preview mmlu-sociology

# Process a single dataset
ols-prep run mmlu-sociology

# Process all pending datasets
ols-prep run --all

# Check status
ols-prep status
```

---

## Configuration

All datasets are defined in `config/datasets.yaml`.

To add a new dataset, add an entry like this:

```yaml
- id: my-new-dataset
  hf_repo: author/dataset-name
  known_issues: []          # zstd_compression | legacy_script
  split: train
  subset: null              # config name if multi-config dataset
  rows: 2000
  random_seed: 42
  columns_keep:
    - instruction
    - response
  augmentation: none        # none | instruction_from_answer | text_generation | ...
  augmentation_llm: null    # ollama | anthropic | litellm
  output_name: your-hf-username/my-output-dataset
  local_subdir: 01-instruction-from-answer
  recipe_target: instruction_from_answer   # output format / fine-tuning recipe type
  ols_project: my-project
  status: pending
```

`recipe_target` is the current runtime key used by the CLI and manifests.
Conceptually, it is the downstream fine-tuning target or output format target
for the prepared dataset.

Then run:

```bash
ols-prep run my-new-dataset
```

---

## CLI Reference

```text
ols-prep run [DATASET_ID]               Process one dataset
ols-prep run --recipe RECIPE_TARGET     Process all datasets for a target label
ols-prep run --project PROJECT          Process all datasets for a project
ols-prep run --all                      Process all pending datasets
ols-prep run --all --force              Re-process including complete datasets
ols-prep status                         Show status table
ols-prep preview DATASET_ID             Fetch + filter, show 3 rows, no save
ols-prep validate DATASET_ID            Re-validate a processed dataset
ols-prep push DATASET_ID                Push local parquet to HF Hub
```

---

## Output

Processed datasets are saved as snappy-compressed parquet:

```text
$OUTPUT_BASE_DIR/
├── 01-instruction-from-answer/
│   ├── prosocial-dialog.parquet
│   └── mmlu-sociology.parquet
├── 05-text-to-sql/
│   └── sql-create-context.parquet
└── 06-structured-outputs/
    └── moral-stories.parquet
```

Every run updates `manifest.json` with provenance metadata for reproducibility.

---

## Pipeline Examples

See the `pipelines/` directory for standalone examples for common downstream
fine-tuning targets:

| File | Target |
|------|--------|
| `pipelines/instruction_from_answer.py` | Instruction from Answer |
| `pipelines/structured_output.py`       | Structured Outputs |
| `pipelines/text_to_sql.py`             | Text to SQL |
| `pipelines/text_to_python.py`          | Text to Python |

---

## Roadmap

- **Phase 1** (current) — fetch, filter, validate, deliver, CLI
- **Phase 2** — compatibility test suite for registered datasets
- **Phase 3** — distilabel augmentation (Ollama + Anthropic backends)
- **Phase 4** — Kestra scheduling + Langfuse observability

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

Built by [Optimal Living Systems](https://github.com/Optimal-Living-Systems)
