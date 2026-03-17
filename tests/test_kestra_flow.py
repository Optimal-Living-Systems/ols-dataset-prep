"""Validation tests for the production Kestra flow."""

from pathlib import Path
import re

import yaml


FLOW_PATH = Path(__file__).resolve().parent.parent / "kestra" / "ols-dataset-prep-flow.yml"
SECRET_PATTERN = re.compile(r"\{\{\s*secret\('[A-Z0-9_]+'\)\s*\}\}")


def _load_flow() -> dict:
    with FLOW_PATH.open() as f:
        return yaml.safe_load(f)


def test_flow_yaml_is_parseable():
    flow = _load_flow()
    assert isinstance(flow, dict)


def test_required_top_level_fields_are_present():
    flow = _load_flow()
    assert flow["id"] == "ols-dataset-prep"
    assert flow["namespace"] == "ols.data"
    assert isinstance(flow["tasks"], list)
    assert flow["tasks"]


def test_all_task_ids_are_unique():
    flow = _load_flow()
    task_ids = [task["id"] for task in flow["tasks"]]
    assert len(task_ids) == len(set(task_ids))


def test_secret_references_use_kestra_syntax():
    flow = _load_flow()

    run_pipeline_env = next(task for task in flow["tasks"] if task["id"] == "run_pipeline")["env"]
    assert SECRET_PATTERN.fullmatch(run_pipeline_env["HF_TOKEN"])
    assert SECRET_PATTERN.fullmatch(run_pipeline_env["ANTHROPIC_API_KEY"])

    webhook_trigger = next(trigger for trigger in flow["triggers"] if trigger["id"] == "manual_webhook")
    assert SECRET_PATTERN.fullmatch(webhook_trigger["key"])
