"""
CLI Entry Point

Typer-based CLI for the OLS Dataset Preparation Pipeline.
All commands use Rich for progress display and error formatting.

Commands:
  ols-prep run [dataset_id]           — process one dataset by ID
  ols-prep run --recipe RECIPE        — process all datasets for a recipe type
  ols-prep run --project PROJECT      — process all datasets for an OLS project
  ols-prep run --all                  — process all pending datasets
  ols-prep run --all --force          — process all, including already-complete
  ols-prep status                     — show status table for all datasets
  ols-prep preview DATASET_ID         — fetch + filter + show 3 rows, don't save
  ols-prep validate DATASET_ID        — validate an already-processed dataset
  ols-prep push DATASET_ID            — push an existing local parquet to HF Hub
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import print as rprint
from rich.panel import Panel

from ols_dataset_prep.fetcher import fetch_dataset, DatasetFetchError
from ols_dataset_prep.filter import filter_dataset
from ols_dataset_prep.augmentor import augment_dataset
from ols_dataset_prep.validator import validate_dataset
from ols_dataset_prep.deliverer import deliver_dataset
from ols_dataset_prep import manifest as manifest_module
from ols_dataset_prep.logger import setup_logging, PipelineLogger

# Load .env from project root (parent of this package)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="ols-prep",
    help="OLS Dataset Preparation Pipeline — universal HuggingFace dataset prep for fine-tuning workflows",
    no_args_is_help=True,
)
console = Console()

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_registry() -> list[dict]:
    """Load and return all dataset configs from datasets.yaml."""
    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("datasets", [])


def _find_dataset(dataset_id: str, registry: list[dict]) -> Optional[dict]:
    """Look up a dataset config by its ID."""
    return next((d for d in registry if d["id"] == dataset_id), None)


def _run_pipeline(cfg: dict, force: bool = False, preview: bool = False) -> bool:
    """
    Run the full pipeline for a single dataset config.

    Returns True on success, False on failure.
    In preview mode: fetch + filter + print 3 rows, no save.
    """
    dataset_id = cfg["id"]
    run_log = PipelineLogger(dataset_id)

    # Skip already-complete datasets unless --force
    if not force and not preview and manifest_module.is_complete(dataset_id):
        console.print(f"[yellow]⏭  Skipping {dataset_id} (already complete — use --force to re-run)[/yellow]")
        return True

    console.rule(f"[bold cyan]{dataset_id}[/bold cyan]")
    run_log.start_run(metadata=cfg)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:

        # ── Stage 1: Fetch ─────────────────────────────────────────────
        task = progress.add_task("[cyan]Stage 1/5  Fetching...", total=None)
        try:
            dataset = fetch_dataset(
                hf_repo=cfg["hf_repo"],
                split=cfg["split"],
                subset=cfg.get("subset"),
                rows=cfg["rows"],
                random_seed=cfg.get("random_seed", 42),
                trust_remote_code=True,
                data_url=cfg.get("data_url"),
                data_url_file=cfg.get("data_url_file"),
            )
            progress.update(task, description=f"[green]Stage 1/5  Fetched {len(dataset)} rows", completed=1, total=1)
        except DatasetFetchError as e:
            progress.stop()
            console.print(Panel(str(e), title="[red]Fetch Failed[/red]", border_style="red"))
            run_log.finish_run(success=False, summary={"error": str(e)})
            return False

        # ── Stage 2: Filter ────────────────────────────────────────────
        task = progress.add_task("[cyan]Stage 2/5  Filtering...", total=None)
        dataset = filter_dataset(
            dataset=dataset,
            columns_keep=cfg["columns_keep"],
            rows=cfg["rows"],
            random_seed=cfg.get("random_seed", 42),
        )
        progress.update(task, description=f"[green]Stage 2/5  Filtered → {len(dataset)} rows", completed=1, total=1)

        # ── Preview mode: show 3 rows and exit ────────────────────────
        if preview:
            progress.stop()
            _show_preview(dataset, dataset_id)
            return True

        # ── Stage 3: Augment ───────────────────────────────────────────
        aug_type = cfg.get("augmentation", "none")
        aug_label = aug_type if aug_type and aug_type != "none" else "none (pass-through)"
        task = progress.add_task("[cyan]Stage 3/5  Augmenting...", total=None)
        dataset = augment_dataset(
            dataset=dataset,
            augmentation_type=aug_type,
            augmentation_llm=cfg.get("augmentation_llm"),
        )
        progress.update(task, description=f"[green]Stage 3/5  Augmented ({aug_label})", completed=1, total=1)

        # ── Stage 4: Validate ──────────────────────────────────────────
        task = progress.add_task("[cyan]Stage 4/5  Validating...", total=None)
        min_rows = cfg.get("min_rows_output") or int(cfg["rows"] * 0.8)
        report = validate_dataset(
            dataset=dataset,
            required_columns=cfg["columns_keep"],
            min_rows=min_rows,
            output_type="structured" if cfg.get("recipe_target") == "structured_outputs_jinja" else "default",
        )
        progress.update(task, description=f"[green]Stage 4/5  Validated ({'PASS' if report.passed else 'FAIL'})", completed=1, total=1)

        if not report.passed:
            progress.stop()
            console.print(Panel(report.summary(), title="[red]Validation Failed[/red]", border_style="red"))
            run_log.finish_run(success=False, summary={"validation": report.checks})
            return False

        if report.warnings:
            for w in report.warnings:
                console.print(f"  [yellow]⚠  {w}[/yellow]")

        # ── Stage 5: Deliver ───────────────────────────────────────────
        task = progress.add_task("[cyan]Stage 5/5  Delivering...", total=None)
        try:
            result = deliver_dataset(
                dataset=dataset,
                dataset_id=dataset_id,
                output_name=cfg["output_name"],
                local_subdir=cfg["local_subdir"],
                hf_repo=cfg["hf_repo"],
                augmentation=cfg.get("augmentation", "none"),
                recipe_target=cfg["recipe_target"],
                ols_project=cfg["ols_project"],
                push_to_hub=True,
            )
        except Exception as e:
            progress.stop()
            console.print(Panel(
                f"Delivery failed for {dataset_id}\n  → {e}",
                title="[red]Delivery Failed[/red]",
                border_style="red",
            ))
            run_log.finish_run(success=False, summary={"error": str(e)})
            return False

        progress.update(task, description="[green]Stage 5/5  Delivered", completed=1, total=1)

    # ── Update manifest ────────────────────────────────────────────────
    manifest_module.update_manifest(dataset_id, {
        "hf_repo": cfg["hf_repo"],
        "output_name": cfg["output_name"],
        "local_path": result["local_path"],
        "hf_url": result["hf_url"],
        "rows_in": len(dataset),
        "rows_out": result["rows"],
        "augmentation": cfg.get("augmentation", "none"),
        "augmentation_llm": cfg.get("augmentation_llm"),
        "validation_passed": report.passed,
        "processed_at": result["processed_at"],
        "recipe_target": cfg["recipe_target"],
        "ols_project": cfg["ols_project"],
        "known_issues_resolved": cfg.get("known_issues", []),
    })

    run_log.finish_run(success=True, summary=result)
    console.print(f"[bold green]✓ {dataset_id} complete[/bold green] — {result['local_path']}")
    return True


def _show_preview(dataset, dataset_id: str) -> None:
    """Print the first 3 rows of a dataset in a Rich table."""
    console.print(f"\n[bold]Preview: {dataset_id}[/bold] ({len(dataset)} rows, {len(dataset.column_names)} columns)\n")
    table = Table(show_header=True, header_style="bold magenta")
    cols = dataset.column_names
    for col in cols:
        table.add_column(col, max_width=60, overflow="fold")
    for row in dataset.select(range(min(3, len(dataset)))):
        table.add_row(*[str(row[c])[:200] for c in cols])
    console.print(table)


# ── Commands ───────────────────────────────────────────────────────────────

@app.command()
def run(
    dataset_id: Optional[str] = typer.Argument(None, help="Dataset ID from datasets.yaml"),
    recipe: Optional[str] = typer.Option(None, "--recipe", help="Process all datasets for this downstream target label"),
    project: Optional[str] = typer.Option(None, "--project", help="Process all datasets for this OLS project"),
    all_datasets: bool = typer.Option(False, "--all", help="Process all pending datasets"),
    force: bool = typer.Option(False, "--force", help="Re-process even if already complete"),
):
    """Process one or more datasets through the full pipeline."""
    registry = _load_registry()

    if dataset_id:
        cfg = _find_dataset(dataset_id, registry)
        if not cfg:
            console.print(f"[red]Dataset '{dataset_id}' not found in config/datasets.yaml[/red]")
            raise typer.Exit(1)
        targets = [cfg]

    elif recipe:
        targets = [d for d in registry if d.get("recipe_target") == recipe]
        if not targets:
            console.print(f"[red]No datasets found for target label '{recipe}'[/red]")
            raise typer.Exit(1)

    elif project:
        targets = [d for d in registry if d.get("ols_project") == project]
        if not targets:
            console.print(f"[red]No datasets found for ols_project '{project}'[/red]")
            raise typer.Exit(1)

    elif all_datasets:
        targets = registry if force else [d for d in registry if d.get("status") != "complete"]
    else:
        console.print("[red]Specify a dataset ID or use --all, --recipe, or --project[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]OLS Dataset Prep[/bold] — processing {len(targets)} dataset(s)\n")

    results = {"success": [], "failed": []}
    for cfg in targets:
        ok = _run_pipeline(cfg, force=force)
        (results["success"] if ok else results["failed"]).append(cfg["id"])

    # Summary
    console.print()
    console.print(f"[bold green]✓ Success:[/bold green] {len(results['success'])}  |  "
                  f"[bold red]✗ Failed:[/bold red] {len(results['failed'])}")
    if results["failed"]:
        console.print(f"  Failed: {', '.join(results['failed'])}")
        raise typer.Exit(1)


@app.command()
def preview(
    dataset_id: str = typer.Argument(..., help="Dataset ID to preview"),
):
    """Fetch and filter a dataset, show 3 rows. Does not save anything."""
    registry = _load_registry()
    cfg = _find_dataset(dataset_id, registry)
    if not cfg:
        console.print(f"[red]Dataset '{dataset_id}' not found in config/datasets.yaml[/red]")
        raise typer.Exit(1)
    _run_pipeline(cfg, force=True, preview=True)


@app.command()
def status():
    """Show processing status of all datasets in a table."""
    registry = _load_registry()
    mf = manifest_module.load_manifest()

    table = Table(title="OLS Dataset Status", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="bold")
    table.add_column("Status")
    table.add_column("Project")
    table.add_column("Target")
    table.add_column("Rows Out")
    table.add_column("Processed At")
    table.add_column("Known Issues")

    for cfg in registry:
        did = cfg["id"]
        entry = mf.get(did)
        if entry and entry.get("validation_passed"):
            st = "[green]complete[/green]"
            rows_out = str(entry.get("rows_out", "—"))
            proc = (entry.get("processed_at") or "")[:10]
        elif cfg.get("status") == "ready":
            st = "[blue]ready[/blue]"
            rows_out = "—"
            proc = "—"
        else:
            st = "[yellow]pending[/yellow]"
            rows_out = "—"
            proc = "—"

        issues = ", ".join(cfg.get("known_issues") or []) or "none"
        table.add_row(did, st, cfg["ols_project"], cfg["recipe_target"], rows_out, proc, issues)

    console.print(table)


@app.command()
def validate(
    dataset_id: str = typer.Argument(..., help="Dataset ID to validate"),
):
    """Validate an already-processed dataset from manifest."""
    entry = manifest_module.get_entry(dataset_id)
    if not entry:
        console.print(f"[red]{dataset_id} not found in manifest — run it first[/red]")
        raise typer.Exit(1)

    local_path = entry.get("local_path")
    if not local_path or not Path(local_path).exists():
        console.print(f"[red]Local file not found: {local_path}[/red]")
        raise typer.Exit(1)

    from datasets import load_dataset as hf_load
    dataset = hf_load("parquet", data_files=local_path, split="train")

    registry = _load_registry()
    cfg = _find_dataset(dataset_id, registry)
    required_columns = cfg["columns_keep"] if cfg else dataset.column_names

    report = validate_dataset(
        dataset=dataset,
        required_columns=required_columns,
        min_rows=1,
    )
    console.print(report.summary())


@app.command()
def push(
    dataset_id: str = typer.Argument(..., help="Dataset ID to push to HF Hub"),
):
    """Push an already-converted local parquet file to HuggingFace Hub."""
    entry = manifest_module.get_entry(dataset_id)
    if not entry:
        console.print(f"[red]{dataset_id} not found in manifest[/red]")
        raise typer.Exit(1)

    local_path = entry.get("local_path")
    if not local_path or not Path(local_path).exists():
        console.print(f"[red]Local file not found: {local_path}[/red]")
        raise typer.Exit(1)

    token = os.getenv("HF_TOKEN")
    if not token:
        console.print("[red]HF_TOKEN not set in .env[/red]")
        raise typer.Exit(1)

    from datasets import load_dataset as hf_load
    dataset = hf_load("parquet", data_files=local_path, split="train")
    output_name = entry["output_name"]

    console.print(f"Pushing {len(dataset)} rows → {output_name}")
    dataset.push_to_hub(output_name, token=token)
    console.print(f"[green]✓ Pushed → https://huggingface.co/datasets/{output_name}[/green]")


if __name__ == "__main__":
    app()
