"""
Stage 3 — Augment (STUB — Phase 1)

In Phase 1 this module is a graceful stub. Every augmentation type passes
data through unchanged with a console log. The pipeline completes end-to-end.

Phase 3 will replace the stub bodies with full distilabel implementations:
  - instruction_from_answer: InstructionBackTranslation-style via TextGeneration
  - text_generation: Free-form instruction+response pair generation
  - structured_output: JSON-formatted output generation
  - generate_sentence_pair: Positive/negative pairs for embedding training
  - quality_judge: LLM-based quality scoring and filtering

LLM backends (Phase 3): Ollama (local bulk), Anthropic Claude (quality runs),
LiteLLM (auto-route to best available).
"""

import logging
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)

# Sociology-specific prompt for instruction_from_answer (used in Phase 3)
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


def augment_dataset(
    dataset: Dataset,
    augmentation_type: str,
    augmentation_llm: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> Dataset:
    """
    Apply augmentation to a dataset.

    In Phase 1: all types are stubs that return the dataset unchanged.
    In Phase 3: each type runs a real distilabel pipeline.

    Args:
        dataset: Filtered Dataset from Stage 2.
        augmentation_type: One of: instruction_from_answer, text_generation,
            structured_output, generate_sentence_pair, quality_judge, none.
        augmentation_llm: Backend to use: "ollama", "anthropic", "litellm", or None.
        ollama_model: Ollama model name override (defaults to OLLAMA_MODEL env var).

    Returns:
        Dataset (unchanged in Phase 1).
    """
    if augmentation_type == "none" or augmentation_type is None:
        logger.debug("Augmentation type is 'none' — passing through unchanged")
        return dataset

    # Phase 1 stub — every non-none type logs and passes through
    logger.info(
        f"[STUB] Augmentation '{augmentation_type}' via '{augmentation_llm}' "
        f"not yet implemented — passing through unchanged. "
        f"This will be implemented in Phase 3."
    )
    return dataset
