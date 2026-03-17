"""
Stage 3 — Augment (Phase 3 Implementation)

Replaces the Phase 1 stub with real distilabel pipelines.
Each augmentation type wraps TextGeneration and sends rows to an LLM,
then writes the output back into the dataset.

Backends:
  - Ollama (local, free) — used for high-volume instruction_from_answer tasks
  - Anthropic Claude     — used for text_generation and structured_output

Retry logic: 3 attempts, exponential backoff (1s, 2s, 4s).
Progress:    logged per batch (BATCH_SIZE rows at a time).
"""

import json
import logging
import os
import time
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)

BATCH_SIZE = 10  # rows per LLM call
MAX_RETRIES = 3

# ── Prompt Templates ────────────────────────────────────────────────────────

INSTRUCTION_FROM_ANSWER_PROMPT = """You are generating training data for a helpful AI assistant.

Given the response below, generate a realistic instruction — a question or \
request that someone would naturally ask to receive this response.

Rules:
- Write in second or third person
- Do NOT include specific names, cities, or organization names
- 1-3 sentences only
- Make it feel like a genuine question a real person would ask

Response: {response}

Instruction:"""

TEXT_GENERATION_PROMPT = """You are a subject matter expert creating training data.

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
Return only valid JSON with keys "instruction" and "response". No markdown fences."""

STRUCTURED_OUTPUT_PROMPT = """You are formatting scenario data as structured JSON for AI training.

Given this moral scenario, produce a JSON object with these exact keys:
norm, situation, intention, moral_action, immoral_action,
moral_consequence, immoral_consequence.

Scenario data:
{row_as_json}

Return only valid JSON. No explanation. No markdown fences."""

_STRUCTURED_KEYS = [
    "norm", "situation", "intention",
    "moral_action", "immoral_action",
    "moral_consequence", "immoral_consequence",
]


# ── LLM Factory ─────────────────────────────────────────────────────────────

def _make_ollama_llm(model: str, base_url: str):
    from distilabel.llms import OllamaLLM
    llm = OllamaLLM(model=model, host=base_url)
    llm.load()
    return llm


def _make_anthropic_llm(api_key: str, model: str = "claude-haiku-4-5-20251001"):
    from distilabel.llms import AnthropicLLM
    llm = AnthropicLLM(model=model, api_key=api_key)
    llm.load()
    return llm


# ── Generation Core ──────────────────────────────────────────────────────────

def _extract_text(generation) -> str:
    """Extract plain text from a distilabel generation output."""
    if generation is None:
        return ""
    if hasattr(generation, "text"):
        return generation.text or ""
    if isinstance(generation, str):
        return generation
    return str(generation)


def _generate_batch(llm, prompts: list[str]) -> list[str]:
    """Send a batch of prompts to the LLM with retry/backoff. Returns one string per prompt."""
    inputs = [[{"role": "user", "content": p}] for p in prompts]

    for attempt in range(MAX_RETRIES):
        try:
            responses = llm.generate(inputs=inputs, num_generations=1)
            results = []
            for resp in responses:
                # resp is List[GeneratedTextOutput] — one entry per num_generations
                if resp:
                    results.append(_extract_text(resp[0]))
                else:
                    results.append("")
            return results
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                logger.error("Generation failed after %d attempts: %s", MAX_RETRIES, exc)
                return [""] * len(prompts)
            wait = 2 ** attempt
            logger.warning("Attempt %d failed: %s. Retrying in %ds...", attempt + 1, exc, wait)
            time.sleep(wait)

    return [""] * len(prompts)


def _run_in_batches(llm, all_prompts: list[str], desc: str = "Augmenting") -> list[str]:
    """Process all prompts in BATCH_SIZE chunks, logging progress."""
    results: list[str] = []
    total = len(all_prompts)

    for i in range(0, total, BATCH_SIZE):
        batch = all_prompts[i : i + BATCH_SIZE]
        results.extend(_generate_batch(llm, batch))
        logger.info("%s: %d/%d rows processed", desc, min(i + BATCH_SIZE, total), total)

    return results


# ── Augmentation Functions ───────────────────────────────────────────────────

def instruction_from_answer(
    dataset: Dataset,
    llm_backend: str,
    ollama_model: Optional[str] = None,
) -> Dataset:
    """
    Generate an `instruction` column by inverting the response column.

    Works on prosocial-dialog (`response` col) and social-bias-frames (`post` col).
    Backend: Ollama (local, free).
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = ollama_model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

    llm = _make_ollama_llm(model=model, base_url=base_url)

    # prosocial-dialog uses "response"; social-bias-frames uses "post"
    resp_col = "response" if "response" in dataset.column_names else "post"
    logger.info("instruction_from_answer: col=%s, model=%s, rows=%d", resp_col, model, len(dataset))

    prompts = [
        INSTRUCTION_FROM_ANSWER_PROMPT.format(response=row[resp_col])
        for row in dataset
    ]

    raw = _run_in_batches(llm, prompts, desc="instruction_from_answer")

    # Strip echoed "Instruction:" prefix if the model included it
    instructions = [r.strip().removeprefix("Instruction:").strip() for r in raw]

    return dataset.add_column("instruction", instructions)


def text_generation(
    dataset: Dataset,
    llm_backend: str,
) -> Dataset:
    """
    Generate `instruction` + `response` pairs from MMLU MCQ rows.

    Used for mmlu-sociology. Backend: Anthropic Claude.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — required for text_generation augmentation")

    llm = _make_anthropic_llm(api_key=api_key)
    logger.info("text_generation: rows=%d", len(dataset))

    prompts = [
        TEXT_GENERATION_PROMPT.format(
            question=row["question"],
            choices=row["choices"],
            answer=row["answer"],
        )
        for row in dataset
    ]

    raw_outputs = _run_in_batches(llm, prompts, desc="text_generation (mmlu)")

    instructions: list[str] = []
    responses: list[str] = []
    for raw in raw_outputs:
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(clean)
            instructions.append(str(parsed.get("instruction", "")))
            responses.append(str(parsed.get("response", "")))
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.warning("Failed to parse JSON output: %r", raw[:120])
            instructions.append("")
            responses.append("")

    dataset = dataset.add_column("instruction", instructions)
    dataset = dataset.add_column("response", responses)
    return dataset


def structured_output(
    dataset: Dataset,
    llm_backend: str,
) -> Dataset:
    """
    Generate a `structured_json` column from moral-stories scenario rows.

    Used for moral-stories. Backend: Anthropic Claude.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — required for structured_output augmentation")

    llm = _make_anthropic_llm(api_key=api_key)
    logger.info("structured_output: rows=%d", len(dataset))

    prompts = []
    for row in dataset:
        row_dict = {k: row.get(k, "") for k in _STRUCTURED_KEYS if k in row}
        prompts.append(STRUCTURED_OUTPUT_PROMPT.format(row_as_json=json.dumps(row_dict, ensure_ascii=False)))

    raw_outputs = _run_in_batches(llm, prompts, desc="structured_output (moral-stories)")

    structured: list[str] = []
    for raw in raw_outputs:
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            json.loads(clean)   # validate parseable
            structured.append(clean)
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.warning("Failed to validate JSON output: %r", raw[:120])
            structured.append("{}")

    return dataset.add_column("structured_json", structured)


# ── Public API ───────────────────────────────────────────────────────────────

def augment_dataset(
    dataset: Dataset,
    augmentation_type: str,
    augmentation_llm: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> Dataset:
    """
    Apply augmentation to a dataset.

    Args:
        dataset:           Filtered Dataset from Stage 2.
        augmentation_type: One of: instruction_from_answer, text_generation,
                           structured_output, none.
        augmentation_llm:  Backend override: "ollama", "anthropic", or None.
        ollama_model:      Ollama model name override (defaults to $OLLAMA_MODEL).

    Returns:
        Dataset with new column(s) added, or unchanged if type is "none".
    """
    if not augmentation_type or augmentation_type == "none":
        logger.debug("Augmentation type is 'none' — passing through unchanged")
        return dataset

    logger.info("Starting augmentation: type=%s, llm=%s", augmentation_type, augmentation_llm)

    if augmentation_type == "instruction_from_answer":
        return instruction_from_answer(
            dataset=dataset,
            llm_backend=augmentation_llm or "ollama",
            ollama_model=ollama_model,
        )
    elif augmentation_type == "text_generation":
        return text_generation(
            dataset=dataset,
            llm_backend=augmentation_llm or "anthropic",
        )
    elif augmentation_type == "structured_output":
        return structured_output(
            dataset=dataset,
            llm_backend=augmentation_llm or "anthropic",
        )
    else:
        logger.warning("Unknown augmentation type '%s' — passing through unchanged", augmentation_type)
        return dataset
