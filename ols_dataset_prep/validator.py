"""
Stage 4 — Validate

Runs quality checks on the processed dataset before delivery.
Returns a ValidationReport — never raises exceptions. The caller
decides whether to abort or warn based on the report.

Checks:
  1. Required columns are present
  2. No rows where all required columns are empty
  3. instruction != response (degenerate pairs check)
  4. Minimum row count: rows_out >= 80% of rows requested
  5. Basic PII scan: SSN patterns, phone numbers, email addresses
  6. JSON parseable if output_type == "structured"
"""

import re
import json
import logging
from dataclasses import dataclass, field

from datasets import Dataset

logger = logging.getLogger(__name__)

# PII detection patterns — intentionally conservative (flag, don't block)
_PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
}


@dataclass
class ValidationReport:
    """
    Result of validate_dataset(). passed=True only if all hard checks pass.
    PII findings are warnings, not failures — they require human review.
    """
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation {status} — {self.rows_out}/{self.rows_in} rows"]
        for check, result in self.checks.items():
            icon = "✓" if result else "✗"
            lines.append(f"  {icon} {check}")
        for w in self.warnings:
            lines.append(f"  ⚠  {w}")
        for e in self.errors:
            lines.append(f"  ✗ {e}")
        return "\n".join(lines)


def validate_dataset(
    dataset: Dataset,
    required_columns: list[str],
    min_rows: int,
    output_type: str = "default",
) -> ValidationReport:
    """
    Validate a processed dataset before delivery.

    Args:
        dataset: Dataset from Stage 3 (post-augmentation).
        required_columns: Columns that must be present and non-empty.
        min_rows: Minimum acceptable row count (80% of originally requested).
        output_type: "structured" triggers JSON validity check per row.

    Returns:
        ValidationReport with per-check results, warnings, and errors.
    """
    report = ValidationReport(rows_in=len(dataset), rows_out=len(dataset))

    # ── Check 1: Required columns present ─────────────────────────────────
    available = set(dataset.column_names)
    missing_cols = [c for c in required_columns if c not in available]
    report.checks["required_columns_present"] = len(missing_cols) == 0
    if missing_cols:
        report.errors.append(f"Missing required columns: {missing_cols}")

    # ── Check 2: No fully empty rows ──────────────────────────────────────
    present_required = [c for c in required_columns if c in available]
    empty_row_count = 0
    for row in dataset:
        if all(
            row.get(c) is None or (isinstance(row.get(c), str) and row[c].strip() == "")
            for c in present_required
        ):
            empty_row_count += 1
    report.checks["no_fully_empty_rows"] = empty_row_count == 0
    if empty_row_count:
        report.errors.append(f"{empty_row_count} rows have all required columns empty")

    # ── Check 3: instruction != response (degenerate pairs) ───────────────
    has_instruction = "instruction" in available
    has_response = "response" in available
    if has_instruction and has_response:
        degenerate = sum(
            1 for row in dataset
            if row.get("instruction") and row.get("response")
            and str(row["instruction"]).strip() == str(row["response"]).strip()
        )
        report.checks["no_degenerate_pairs"] = degenerate == 0
        if degenerate:
            report.warnings.append(f"{degenerate} rows have instruction == response")
    else:
        report.checks["no_degenerate_pairs"] = True  # not applicable

    # ── Check 4: Minimum row count ─────────────────────────────────────────
    report.checks["min_row_count"] = len(dataset) >= min_rows
    if len(dataset) < min_rows:
        report.errors.append(
            f"Row count {len(dataset)} is below minimum {min_rows} "
            f"(80% of requested). Dataset may be too small or heavily filtered."
        )

    # ── Check 5: Basic PII scan ────────────────────────────────────────────
    pii_found = {name: 0 for name in _PII_PATTERNS}
    for row in dataset:
        for col_val in row.values():
            if not isinstance(col_val, str):
                continue
            for name, pattern in _PII_PATTERNS.items():
                if pattern.search(col_val):
                    pii_found[name] += 1

    pii_detected = {k: v for k, v in pii_found.items() if v > 0}
    report.checks["pii_scan"] = len(pii_detected) == 0
    if pii_detected:
        report.warnings.append(
            f"Possible PII detected (review before publishing): {pii_detected}"
        )

    # ── Check 6: JSON valid (structured output type only) ─────────────────
    if output_type == "structured":
        json_col = next(
            (c for c in ["output", "response", "completion"] if c in available), None
        )
        if json_col:
            bad_json = 0
            for row in dataset:
                val = row.get(json_col, "")
                if isinstance(val, str):
                    try:
                        json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        bad_json += 1
            report.checks["json_valid"] = bad_json == 0
            if bad_json:
                report.warnings.append(
                    f"{bad_json} rows in '{json_col}' are not valid JSON"
                )
        else:
            report.checks["json_valid"] = True  # no output column to check
    else:
        report.checks["json_valid"] = True  # not applicable

    # ── Final pass/fail ────────────────────────────────────────────────────
    # Hard failures: missing columns, empty rows, min row count
    # Soft warnings: PII, degenerate pairs, invalid JSON
    hard_checks = ["required_columns_present", "no_fully_empty_rows", "min_row_count"]
    report.passed = all(report.checks.get(c, True) for c in hard_checks)

    return report
