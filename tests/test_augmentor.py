"""
Tests for Stage 3 — Augment (Phase 3)

All LLM calls are mocked so tests run without any running Ollama instance
or Anthropic API key. The mocks return realistic-looking outputs that
exercise JSON parsing, prefix stripping, and fallback handling.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from ols_dataset_prep.augmentor import (
    BATCH_SIZE,
    _extract_text,
    _generate_batch,
    augment_dataset,
    instruction_from_answer,
    structured_output,
    text_generation,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _prosocial_dataset(n: int = 3) -> Dataset:
    return Dataset.from_dict({
        "context": [f"ctx_{i}" for i in range(n)],
        "response": [f"This is response number {i}." for i in range(n)],
        "rots": [["be kind"] for _ in range(n)],
        "safety_label": ["safe"] * n,
    })


def _sbf_dataset(n: int = 3) -> Dataset:
    return Dataset.from_dict({
        "post": [f"Social post number {i}" for i in range(n)],
        "targetMinority": ["group_a"] * n,
        "targetStereotype": ["stereotype_x"] * n,
        "offensiveYN": ["1.0"] * n,
    })


def _mmlu_dataset(n: int = 3) -> Dataset:
    return Dataset.from_dict({
        "question": [f"What is concept {i}?" for i in range(n)],
        "choices": [["A", "B", "C", "D"] for _ in range(n)],
        "answer": [0] * n,
    })


def _moral_dataset(n: int = 3) -> Dataset:
    return Dataset.from_dict({
        "norm": [f"norm_{i}" for i in range(n)],
        "situation": [f"situation_{i}" for i in range(n)],
        "intention": [f"intention_{i}" for i in range(n)],
        "moral_action": [f"moral_{i}" for i in range(n)],
        "immoral_action": [f"immoral_{i}" for i in range(n)],
        "moral_consequence": [f"good_{i}" for i in range(n)],
        "immoral_consequence": [f"bad_{i}" for i in range(n)],
    })


def _make_generation(text: str):
    """Create a mock distilabel GeneratedTextOutput with a .text attribute."""
    g = MagicMock()
    g.text = text
    return g


# ── _extract_text ─────────────────────────────────────────────────────────────

def test_extract_text_from_object_with_text_attr():
    g = _make_generation("hello")
    assert _extract_text(g) == "hello"


def test_extract_text_from_string():
    assert _extract_text("direct string") == "direct string"


def test_extract_text_from_none():
    assert _extract_text(None) == ""


# ── _generate_batch ───────────────────────────────────────────────────────────

def test_generate_batch_success():
    llm = MagicMock()
    llm.generate.return_value = [
        [_make_generation("instruction A")],
        [_make_generation("instruction B")],
    ]
    result = _generate_batch(llm, ["prompt1", "prompt2"])
    assert result == ["instruction A", "instruction B"]


def test_generate_batch_empty_response():
    llm = MagicMock()
    llm.generate.return_value = [None, []]
    result = _generate_batch(llm, ["p1", "p2"])
    assert result == ["", ""]


def test_generate_batch_retries_then_returns_empty(monkeypatch):
    monkeypatch.setattr("ols_dataset_prep.augmentor.time.sleep", lambda _: None)
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("LLM down")
    result = _generate_batch(llm, ["p1"])
    assert result == [""]
    assert llm.generate.call_count == 3  # MAX_RETRIES


def test_generate_batch_succeeds_on_second_attempt(monkeypatch):
    monkeypatch.setattr("ols_dataset_prep.augmentor.time.sleep", lambda _: None)
    llm = MagicMock()
    llm.generate.side_effect = [
        RuntimeError("transient error"),
        [[_make_generation("recovered")]],
    ]
    result = _generate_batch(llm, ["p1"])
    assert result == ["recovered"]


# ── instruction_from_answer ───────────────────────────────────────────────────

def _mock_ollama_llm(generated_texts: list[str]):
    """Return a mock LLM that yields the given texts one per call (batch)."""
    llm = MagicMock()
    calls = iter(generated_texts)

    def _generate(inputs, num_generations=1):
        return [[_make_generation(next(calls))] for _ in inputs]

    llm.generate.side_effect = _generate
    return llm


@patch("ols_dataset_prep.augmentor._make_ollama_llm")
def test_instruction_from_answer_response_col(mock_make):
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation("How do you handle this situation?")],
        [_make_generation("What should someone do here?")],
        [_make_generation("How would you respond to this?")],
    ]
    mock_make.return_value = mock_llm

    ds = _prosocial_dataset(3)
    result = instruction_from_answer(ds, llm_backend="ollama")

    assert "instruction" in result.column_names
    assert len(result) == 3
    assert result["instruction"][0] == "How do you handle this situation?"


@patch("ols_dataset_prep.augmentor._make_ollama_llm")
def test_instruction_from_answer_post_col(mock_make):
    """social-bias-frames uses 'post' column, not 'response'."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation("What is being said here?")],
        [_make_generation("How is this post framed?")],
        [_make_generation("What does this imply?")],
    ]
    mock_make.return_value = mock_llm

    ds = _sbf_dataset(3)
    result = instruction_from_answer(ds, llm_backend="ollama")

    assert "instruction" in result.column_names
    assert "post" in result.column_names  # original col preserved


@patch("ols_dataset_prep.augmentor._make_ollama_llm")
def test_instruction_from_answer_strips_prefix(mock_make):
    """Model sometimes echoes 'Instruction:' — should be stripped."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation("Instruction: What are you looking for?")],
    ]
    mock_make.return_value = mock_llm

    ds = _prosocial_dataset(1)
    result = instruction_from_answer(ds, llm_backend="ollama")

    assert result["instruction"][0] == "What are you looking for?"


# ── text_generation ────────────────────────────────────────────────────────────

@patch("ols_dataset_prep.augmentor._make_anthropic_llm")
def test_text_generation_parses_json(mock_make, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation('{"instruction": "What is social stratification?", "response": "It is the layering of society."}')],
        [_make_generation('{"instruction": "Define deviance.", "response": "Deviance is behavior that violates norms."}')],
        [_make_generation('{"instruction": "What is socialization?", "response": "It is the process of learning norms."}')],
    ]
    mock_make.return_value = mock_llm

    ds = _mmlu_dataset(3)
    result = text_generation(ds, llm_backend="anthropic")

    assert "instruction" in result.column_names
    assert "response" in result.column_names
    assert result["instruction"][0] == "What is social stratification?"
    assert result["response"][1] == "Deviance is behavior that violates norms."


@patch("ols_dataset_prep.augmentor._make_anthropic_llm")
def test_text_generation_handles_bad_json(mock_make, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation("not json at all")],
    ]
    mock_make.return_value = mock_llm

    ds = _mmlu_dataset(1)
    result = text_generation(ds, llm_backend="anthropic")

    assert result["instruction"][0] == ""
    assert result["response"][0] == ""


@patch("ols_dataset_prep.augmentor._make_anthropic_llm")
def test_text_generation_strips_markdown_fences(mock_make, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation('```json\n{"instruction": "Q?", "response": "A."}\n```')],
    ]
    mock_make.return_value = mock_llm

    ds = _mmlu_dataset(1)
    result = text_generation(ds, llm_backend="anthropic")

    assert result["instruction"][0] == "Q?"
    assert result["response"][0] == "A."


def test_text_generation_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ds = _mmlu_dataset(1)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        text_generation(ds, llm_backend="anthropic")


# ── structured_output ─────────────────────────────────────────────────────────

@patch("ols_dataset_prep.augmentor._make_anthropic_llm")
def test_structured_output_valid_json(mock_make, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    valid_json = json.dumps({
        "norm": "be honest",
        "situation": "lying",
        "intention": "avoid trouble",
        "moral_action": "tell the truth",
        "immoral_action": "lie",
        "moral_consequence": "trust maintained",
        "immoral_consequence": "trust lost",
    })
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation(valid_json)],
        [_make_generation(valid_json)],
        [_make_generation(valid_json)],
    ]
    mock_make.return_value = mock_llm

    ds = _moral_dataset(3)
    result = structured_output(ds, llm_backend="anthropic")

    assert "structured_json" in result.column_names
    assert len(result) == 3
    parsed = json.loads(result["structured_json"][0])
    assert parsed["norm"] == "be honest"


@patch("ols_dataset_prep.augmentor._make_anthropic_llm")
def test_structured_output_falls_back_on_invalid_json(mock_make, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation("here is your json: {broken}")],
    ]
    mock_make.return_value = mock_llm

    ds = _moral_dataset(1)
    result = structured_output(ds, llm_backend="anthropic")

    assert result["structured_json"][0] == "{}"


def test_structured_output_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ds = _moral_dataset(1)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        structured_output(ds, llm_backend="anthropic")


# ── augment_dataset (public API) ──────────────────────────────────────────────

def test_augment_dataset_none_passthrough():
    ds = _prosocial_dataset(3)
    result = augment_dataset(ds, augmentation_type="none")
    assert result is ds


def test_augment_dataset_null_passthrough():
    ds = _prosocial_dataset(3)
    result = augment_dataset(ds, augmentation_type=None)
    assert result is ds


def test_augment_dataset_unknown_type_passthrough():
    ds = _prosocial_dataset(3)
    result = augment_dataset(ds, augmentation_type="not_a_real_type")
    assert result is ds
    assert result.column_names == ds.column_names


@patch("ols_dataset_prep.augmentor._make_ollama_llm")
def test_augment_dataset_dispatches_instruction_from_answer(mock_make):
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation("Generated instruction.")],
        [_make_generation("Another instruction.")],
        [_make_generation("Third instruction.")],
    ]
    mock_make.return_value = mock_llm

    ds = _prosocial_dataset(3)
    result = augment_dataset(ds, augmentation_type="instruction_from_answer", augmentation_llm="ollama")

    assert "instruction" in result.column_names


@patch("ols_dataset_prep.augmentor._make_anthropic_llm")
def test_augment_dataset_dispatches_text_generation(mock_make, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation('{"instruction": "Q?", "response": "A."}')],
        [_make_generation('{"instruction": "Q2?", "response": "A2."}')],
        [_make_generation('{"instruction": "Q3?", "response": "A3."}')],
    ]
    mock_make.return_value = mock_llm

    ds = _mmlu_dataset(3)
    result = augment_dataset(ds, augmentation_type="text_generation", augmentation_llm="anthropic")

    assert "instruction" in result.column_names
    assert "response" in result.column_names


@patch("ols_dataset_prep.augmentor._make_anthropic_llm")
def test_augment_dataset_dispatches_structured_output(mock_make, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    valid_json = json.dumps({k: "v" for k in [
        "norm", "situation", "intention", "moral_action",
        "immoral_action", "moral_consequence", "immoral_consequence",
    ]})
    mock_llm = MagicMock()
    mock_llm.generate.return_value = [
        [_make_generation(valid_json)],
        [_make_generation(valid_json)],
        [_make_generation(valid_json)],
    ]
    mock_make.return_value = mock_llm

    ds = _moral_dataset(3)
    result = augment_dataset(ds, augmentation_type="structured_output", augmentation_llm="anthropic")

    assert "structured_json" in result.column_names


# ── Batching behaviour ────────────────────────────────────────────────────────

@patch("ols_dataset_prep.augmentor._make_ollama_llm")
def test_batching_splits_large_dataset(mock_make):
    """Dataset larger than BATCH_SIZE should trigger multiple generate() calls."""
    n = BATCH_SIZE + 2
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = lambda inputs, num_generations=1: [
        [_make_generation(f"instruction for row")] for _ in inputs
    ]
    mock_make.return_value = mock_llm

    ds = _prosocial_dataset(n)
    result = instruction_from_answer(ds, llm_backend="ollama")

    assert len(result) == n
    assert mock_llm.generate.call_count == 2  # ceil(n / BATCH_SIZE)
