# ols-dataset-prep

A production-grade dataset preparation pipeline for ML fine-tuning.

Fetches any HuggingFace dataset, resolves common compatibility issues automatically,
augments with LLMs, validates quality, and delivers clean snappy-compressed parquet
ready for any fine-tuning workflow.

Built for orchestration with Kestra. Works standalone from the command line.

---

## Why This Exists

Many HuggingFace datasets cannot be loaded directly by fine-tuning tools due to two
common compatibility issues that affect the entire ecosystem:

**Issue 1 — zstd compression**
HuggingFace is adopting zstd as its default compression format. Many tools read
parquet files directly and cannot handle it.
```
Error: "seed inspect failed: Compression type zstd not supported"
```

**Issue 2 — Legacy `.py` loading scripts**
`datasets >= 4.5.0` deprecated Python loading scripts. Datasets that still ship
one throw:
```
Error: "Dataset scripts are no longer supported, but found [name].py"
```

This pipeline solves both by routing every dataset through `load_dataset()`,
which handles both transparently, then saves as snappy parquet — compatible
with any ML training pipeline.

---

## Architecture

```
Input: HF Dataset ID + config/datasets.yaml
              ↓
  Stage 1: Fetch      — load_dataset() handles zstd, .py scripts, gated datasets
              ↓
  Stage 2: Filter     — column whitelist, row sampling, null removal
              ↓
  Stage 3: Augment    — distilabel LLM pipelines (Phase 3)
              ↓
  Stage 4: Validate   — schema, PII scan, quality checks
              ↓
  Stage 5: Deliver    — snappy parquet locally + push to HF Hub
```

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

**To add a new dataset**, add an entry:

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

Then run:
```bash
ols-prep run my-new-dataset
```

---

## CLI Reference

```
ols-prep run [DATASET_ID]               Process one dataset
ols-prep run --recipe RECIPE_TARGET     Process all datasets for a recipe type
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

```
$OUTPUT_BASE_DIR/
├── 01-instruction-from-answer/
│   ├── prosocial-dialog.parquet
│   └── mmlu-sociology.parquet
├── 05-text-to-sql/
│   └── sql-create-context.parquet
└── 06-structured-outputs/
    └── moral-stories.parquet
```

Every run updates `manifest.json` with full provenance metadata.

---

## Pipeline Examples

See the `pipelines/` directory for standalone examples for each fine-tuning format:

| File | Format |
|------|--------|
| `pipelines/instruction_from_answer.py` | Instruction from Answer |
| `pipelines/structured_output.py`       | Structured Outputs (Jinja) |
| `pipelines/text_to_sql.py`             | Text to SQL |
| `pipelines/text_to_python.py`          | Text to Python |

---

## Compatible With

Outputs clean snappy-compressed parquet compatible with any fine-tuning tool or
training pipeline, including Unsloth Studio, Axolotl, LLaMA-Factory, and
custom training scripts.

---

## Roadmap

- **Phase 1** (current) — fetch, filter, validate, deliver, CLI
- **Phase 2** — compatibility test suite for all 6 registered datasets
- **Phase 3** — distilabel augmentation (Ollama + Anthropic Claude backends)
- **Phase 4** — Kestra scheduling + Langfuse observability

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

Built by [Optimal Living Systems](https://github.com/Optimal-Living-Systems)
