# Phase 3 Brief — distilabel Augmentation

## What Phase 3 Adds

Phase 3 replaces the stub in `ols_dataset_prep/augmentor.py` with real
distilabel pipelines. Each augmentation type becomes a `TextGeneration`
wrapper that sends rows to an LLM and writes the output back into the dataset.

After Phase 3, `ols-prep run prosocial-dialog` will:
1. Fetch 2000 rows of prosocial dialog responses
2. Send each response to an Ollama-hosted LLM with the instruction prompt
3. Write the generated instruction back as a new `instruction` column
4. Deliver a dataset with `(instruction, response)` pairs ready for downstream fine-tuning

## Datasets That Get Augmented

| Dataset | Target Type | LLM Backend | Output Columns Added |
|---------|-------------|-------------|----------------------|
| prosocial-dialog | `instruction_from_answer` | Ollama (local) | `instruction` |
| social-bias-frames | `instruction_from_answer` | Ollama (local) | `instruction` |
| mmlu-sociology | `text_generation` | Anthropic Claude | `instruction`, `response` |
| moral-stories | `structured_output` | Anthropic Claude | `structured_json` |

Datasets with `augmentation: none` (sql-create-context, python-code-instructions)
are already in instruction/response format and need no augmentation.

## LLM Backend Assignment

**Ollama (local, free, bulk)**
- Used for: `instruction_from_answer` on large datasets (2000+ rows)
- Model: `qwen2.5:14b` (set via `OLLAMA_MODEL` in `.env`)
- Why: prosocial-dialog and social-bias-frames are high-volume; local
  inference keeps cost at zero for the bulk work

**Anthropic Claude (API, quality runs)**
- Used for: `text_generation` and `structured_output`
- Model: `claude-haiku-4-5` (fast, cheap, sufficient quality)
- Why: MMLU requires generating full Q&A pairs from MCQ stems — needs
  stronger reasoning. moral-stories requires structured JSON output —
  Claude is more reliable at schema adherence than local models.

## Prompt Template (instruction_from_answer)

Used for prosocial-dialog and social-bias-frames:

```
You are generating training data for a helpful AI assistant.

Given the response below, generate a realistic instruction — a question or
request that someone would naturally ask to receive this response.

Rules:
- Write in second or third person
- Do NOT include specific names, cities, or organization names
- 1-3 sentences only
- Make it feel like a genuine question a real person would ask

Response: {response}

Instruction:
```

The `{response}` field maps to:
- prosocial-dialog: `response` column
- social-bias-frames: `post` column (the original social media post)

## Prompt Template (text_generation — MMLU)

Used for mmlu-sociology:

```
You are a subject matter expert creating training data.

Given this multiple-choice sociology question and its correct answer,
generate a realistic instruction/response pair that captures the
knowledge being tested.

Question: {question}
Choices: {choices}
Correct answer index: {answer}

Write:
- instruction: A direct question someone would ask to learn this concept
- response: A clear, educational answer (2-4 sentences)

Do not reference "option A/B/C/D" — answer in plain prose.
```

## Prompt Template (structured_output — moral-stories)

Used for moral-stories:

```
You are formatting scenario data as structured JSON for AI training.

Given this moral scenario, produce a JSON object with these exact keys:
norm, situation, intention, moral_action, immoral_action,
moral_consequence, immoral_consequence.

Scenario data:
{row_as_json}

Return only valid JSON. No explanation. No markdown fences.
```

## Estimated Token Cost — Full Augmentation Run

Assumptions:
- Average response length: ~80 tokens input, ~40 tokens output
- Claude Haiku 4.5 pricing: $0.80/M input, $4.00/M output (approximate)
- Ollama: free (local)

| Dataset | Rows | Backend | Est. Input Tokens | Est. Output Tokens | Est. Cost |
|---------|------|---------|-------------------|--------------------|-----------|
| prosocial-dialog | 2,000 | Ollama | — | — | $0.00 |
| social-bias-frames | 2,953 | Ollama | — | — | $0.00 |
| mmlu-sociology | 201 | Claude Haiku | ~20,100 | ~12,060 | ~$0.07 |
| moral-stories | 2,000 | Claude Haiku | ~200,000 | ~80,000 | ~$0.48 |
| **Total** | | | **~220,100** | **~92,060** | **~$0.55** |

A full Phase 3 augmentation run costs approximately **$0.55 in API fees**.
The bulk of the work (prosocial-dialog, social-bias-frames) runs free on Ollama.

## Implementation Checklist

- [ ] Replace `augmentor.py` stub with real distilabel `TextGeneration` wrappers
- [ ] Add `instruction_from_answer()` — wraps `TextGeneration` with response→instruction prompt
- [ ] Add `text_generation()` — wraps `TextGeneration` with context→qa prompt
- [ ] Add `structured_output()` — wraps `TextGeneration` with JSON schema enforcement
- [ ] Wire Ollama backend: `AsyncOpenAI(base_url=OLLAMA_BASE_URL)`
- [ ] Wire Anthropic backend: `Anthropic(api_key=ANTHROPIC_API_KEY)`
- [ ] Add retry logic: 3 attempts, exponential backoff
- [ ] Add progress tracking: rows augmented / total
- [ ] Re-run all 4 augmented datasets and push updated parquets to HF Hub
- [ ] Write `tests/test_augmentor.py` with mocked LLM responses
