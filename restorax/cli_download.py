"""
restorax download-models — download model weights from Hugging Face.

Usage:
  restorax download-models                        # show status table
  restorax download-models --model real_esrgan
  restorax download-models --group sr
  restorax download-models --all
  restorax download-models --all --force
"""
from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.command(name="download-models")
@click.option("--model", "models", multiple=True, metavar="TEXT", help="Model name to download (repeatable).")
@click.option(
    "--group",
    "groups",
    multiple=True,
    type=click.Choice(["sr", "face", "diffusion", "extras", "audio"]),
    help="Download all models in a group.",
)
@click.option("--all", "download_all", is_flag=True, default=False, help="Download every model in the catalog.")
@click.option("--force", is_flag=True, default=False, help="Re-download even if weights already present.")
def download_models_group(
    models: tuple[str, ...],
    groups: tuple[str, ...],
    download_all: bool,
    force: bool,
) -> None:
    """Download model weights from Hugging Face Hub."""
    from restorax.models_catalog import CATALOG, CATALOG_BY_NAME

    # No selector → print status table and exit
    if not models and not groups and not download_all:
        _print_status_table(CATALOG)
        return

    # Build target list
    targets: list = []
    if download_all:
        targets = list(CATALOG)
    else:
        for name in models:
            entry = CATALOG_BY_NAME.get(name)
            if entry is None:
                console.print(f"[yellow]Warning: unknown model '{name}', skipping.[/yellow]")
            else:
                targets.append(entry)
        for group in groups:
            targets.extend(m for m in CATALOG if m.group == group)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_targets = []
    for t in targets:
        if t.name not in seen:
            seen.add(t.name)
            unique_targets.append(t)

    if not unique_targets:
        console.print("[yellow]No models selected.[/yellow]")
        return

    # Guard: huggingface_hub must be importable
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        console.print(
            "[red]Error: 'huggingface_hub' is not installed. "
            "Run: pip install huggingface_hub[/red]"
        )
        sys.exit(1)

    from huggingface_hub import hf_hub_download, snapshot_download

    for entry in unique_targets:
        if entry.is_ready() and not force:
            console.print(f"[dim]  skipped {entry.name} (already present; use --force to re-download)[/dim]")
            continue

        weight_dir = entry.weight_dir()
        weight_dir.mkdir(parents=True, exist_ok=True)

        try:
            if entry.snapshot:
                snapshot_download(repo_id=entry.hf_repo, local_dir=str(weight_dir))
            else:
                for filename in entry.weight_files:
                    hf_hub_download(
                        repo_id=entry.hf_repo,
                        filename=filename,
                        local_dir=str(weight_dir),
                    )
            console.print(f"[green]✓ {entry.name}[/green]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning: failed to download '{entry.name}': {exc}[/yellow]")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_status_table(catalog: list) -> None:
    table = Table(title="Model Weights Status", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Group")
    table.add_column("Size (MB)", justify="right")
    table.add_column("HF Repo")
    table.add_column("Ready", justify="center")

    for entry in catalog:
        ready = "[green]✓[/green]" if entry.is_ready() else "[red]✗[/red]"
        table.add_row(entry.name, entry.group, str(entry.size_mb), entry.hf_repo, ready)

    console.print(table)
