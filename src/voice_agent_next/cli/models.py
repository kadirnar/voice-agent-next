"""``van models`` — list, download, verify and prune the models local providers use."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

app = typer.Typer(
    name="models",
    help="Model manager: list, download, verify and prune cached models.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()
err = Console(stderr=True)

_STATE_STYLE = {"cached": "green", "partial": "yellow", "missing": "dim"}


def human_size(n: int | None) -> str:
    """``1536`` -> ``"1.5 KB"`` (decimal units, like download pages)."""
    if n is None:
        return "?"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} GB"  # pragma: no cover


def _fail(message: str) -> typer.Exit:
    err.print(f"[red]{escape(message)}[/red]", highlight=False)
    return typer.Exit(1)


@app.callback()
def _models() -> None:
    """Models are cached in `van models path`; VAN_CACHE_DIR moves it, VAN_OFFLINE=1
    forbids downloads. faster-whisper and Smart Turn use the Hugging Face cache."""


@app.command("list")
def list_models(
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="Only this provider")
    ] = None,
    kind: Annotated[str | None, typer.Option("--kind", "-k", help="stt|tts|vad|turn")] = None,
    cached: Annotated[bool, typer.Option("--cached", help="Only cached models")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Available models: cache state, download size, license, languages."""
    from .. import models

    rows: list[dict[str, Any]] = []
    for info in models.catalog(provider=provider, kind=kind):
        status = models.model_status(info)
        if cached and not status.cached:
            continue
        rows.append(
            {
                "name": info.name,
                "provider": info.provider,
                "model": info.model,
                "kinds": list(info.kinds),
                "state": status.state,
                "size": info.size,
                "size_on_disk": status.size_on_disk,
                "hf_cache": info.in_hf_cache,
                "license": info.license,
                "languages": info.languages,
                "description": info.description,
            }
        )
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    n_cached = sum(1 for r in rows if r["state"] == "cached")
    table = Table(title=f"models ({len(rows)}, {n_cached} cached)", title_justify="left")
    for col in ("name", "kind", "state", "size", "license", "languages"):
        table.add_column(col, overflow="fold", justify="right" if col == "size" else "left")
    for r in rows:
        style = _STATE_STYLE[r["state"]]
        state = r["state"] + (" (HF)" if r["hf_cache"] and r["state"] != "missing" else "")
        table.add_row(
            escape(r["name"]),
            ",".join(r["kinds"]),
            f"[{style}]{state}[/{style}]",
            human_size(r["size"]),
            escape(r["license"]),
            escape(r["languages"]),
        )
    console.print(table)
    if models.is_offline():
        console.print("[yellow]VAN_OFFLINE is set: downloads are disabled[/yellow]")


@app.command()
def download(
    names: Annotated[
        list[str] | None, typer.Argument(help="Model names (provider/model) or specs")
    ] = None,
    for_: Annotated[
        list[str] | None,
        typer.Option(
            "--for",
            help="Everything a config file, 'stt=...,tts=...' or a spec needs (repeatable)",
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Download again even if cached")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-n", help="Only show what would be downloaded")
    ] = False,
) -> None:
    """Download models (with progress) into the cache, e.g. for Docker images or offline use."""
    from .. import models
    from ..errors import VoiceAgentError

    if not names and not for_:
        raise typer.BadParameter("give model names and/or --for <config|spec>")
    wanted: list[models.ModelInfo] = []
    notes: list[str] = []
    try:
        for name in names or []:
            wanted.extend(m for m in models.resolve_models(name) if m not in wanted)
        for target in for_ or []:
            req = models.models_for(target)
            wanted.extend(m for m in req.models if m not in wanted)
            notes.extend(req.notes)
    except VoiceAgentError as exc:
        raise _fail(str(exc)) from None
    for note in notes:
        console.print(f"[dim]note: {escape(note)}[/dim]")
    if not wanted:
        console.print("nothing to download")
        return
    todo = [(m, models.model_status(m)) for m in wanted]
    pending = [(m, s) for m, s in todo if force or not s.cached]
    for m, s in todo:
        if not force and s.cached:
            console.print(f"[green]cached[/green]  {escape(m.name)}")
    total = sum(m.size or 0 for m, _ in pending)
    if dry_run:
        for m, _ in pending:
            console.print(f"would download {escape(m.name)} ({human_size(m.size)})")
        console.print(f"{len(pending)} model(s), {human_size(total)}")
        return
    if pending and models.is_offline():
        missing = ", ".join(m.name for m, _ in pending)
        raise _fail(f"VAN_OFFLINE is set; not cached: {missing}")
    failed = 0
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as bar:
        for m, _ in pending:
            progress = _progress_reporter(bar, m.name)
            if m.in_hf_cache:
                bar.console.print(f"fetching {escape(m.name)} from Hugging Face ...")
            try:
                paths = models.download_model(m, force=force, progress=progress)
            except (VoiceAgentError, OSError) as exc:
                failed += 1
                bar.console.print(f"[red]failed[/red]  {escape(m.name)}: {escape(str(exc))}")
                continue
            where = paths[0].parent if paths else ""
            bar.console.print(f"[green]done[/green]    {escape(m.name)} -> {escape(str(where))}")
    if failed:
        raise typer.Exit(1)


def _progress_reporter(bar: Progress, model_name: str) -> Any:
    """A ``progress(file, done, total)`` callback with one progress bar per file."""
    tasks: dict[str, TaskID] = {}

    def progress(f: Any, done: int, size: int | None) -> None:
        key = f.label
        if key not in tasks:
            tasks[key] = bar.add_task(f"{model_name} [dim]{escape(key)}[/dim]", total=size)
        bar.update(tasks[key], completed=done, total=size)

    return progress


@app.command()
def verify(
    names: Annotated[
        list[str] | None, typer.Argument(help="Models to check (default: every cached one)")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Check cached files against their sha256; exit code 1 if anything is corrupt."""
    from .. import models
    from ..errors import VoiceAgentError

    try:
        if names:
            selected = [m for n in names for m in models.resolve_models(n)]
        else:
            selected = [m for m in models.catalog() if models.model_status(m).state != "missing"]
    except VoiceAgentError as exc:
        raise _fail(str(exc)) from None
    rows = []
    bad = 0
    for info in selected:
        for check in models.verify_model(info):
            bad += not check.good
            rows.append(
                {
                    "model": info.name,
                    "file": check.file.label,
                    "status": check.status,
                    "detail": check.detail,
                    "path": str(check.path),
                }
            )
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
    elif not rows:
        console.print("no cached models to verify")
    else:
        table = Table(title=f"verify ({len(selected)} models)", title_justify="left")
        for col in ("model", "file", "status", "detail"):
            table.add_column(col, overflow="fold")
        styles = {"ok": "green", "unverified": "yellow", "missing": "dim", "corrupt": "red"}
        for r in rows:
            style = styles[r["status"]]
            table.add_row(
                escape(r["model"]),
                escape(r["file"]),
                f"[{style}]{r['status']}[/{style}]",
                escape(r["detail"]),
            )
        console.print(table)
        if bad:
            console.print(
                "[red]re-download broken models with `van models download --force <name>`[/red]"
            )
    if bad:
        raise typer.Exit(1)


@app.command()
def prune(
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Delete (default: dry run, only list)")
    ] = False,
    model: Annotated[
        list[str] | None, typer.Option("--model", "-m", help="Also remove this model")
    ] = None,
    all_models: Annotated[
        bool, typer.Option("--all", help="Also remove every cached catalog model")
    ] = False,
    partial: Annotated[
        bool, typer.Option("--partial/--no-partial", help="Interrupted downloads")
    ] = True,
    unused: Annotated[
        bool,
        typer.Option("--unused/--no-unused", help="Files no catalog model uses (old versions)"),
    ] = True,
    older_than: Annotated[
        float | None,
        typer.Option("--older-than", help="Only entries not used for this many days"),
    ] = None,
    include_hf: Annotated[
        bool,
        typer.Option("--include-hf", help="Also delete selected models from the HF cache"),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Remove partial downloads, unused files and (with --model/--all) cached models."""
    from .. import models
    from ..errors import VoiceAgentError

    try:
        items = models.plan_prune(
            partial=partial,
            unused=unused,
            models=model or (),
            all_models=all_models,
            include_hf=include_hf,
            older_than=older_than * 86_400 if older_than is not None else None,
        )
    except VoiceAgentError as exc:
        raise _fail(str(exc)) from None
    total = sum(i.size for i in items)
    freed = models.apply_prune(items) if yes else 0
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "deleted": yes,
                    "bytes": freed if yes else total,
                    "items": [
                        {
                            "path": str(i.path),
                            "reason": i.reason,
                            "size": i.size,
                            "model": i.model,
                            "hf_cache": i.in_hf_cache,
                        }
                        for i in items
                    ],
                },
                indent=2,
            )
        )
        return
    if not items:
        console.print("nothing to prune")
    else:
        verb = "deleted" if yes else "would delete"
        table = Table(title=f"prune: {verb} {len(items)} entries", title_justify="left")
        for col in ("reason", "size", "path"):
            table.add_column(col, overflow="fold", justify="right" if col == "size" else "left")
        for i in items:
            reason = f"{i.reason} ({i.model})" if i.model else i.reason
            table.add_row(escape(reason), human_size(i.size), escape(str(i.path)))
        console.print(table)
        if yes:
            console.print(f"freed {human_size(freed)}")
        else:
            console.print(f"{human_size(total)} reclaimable; run again with --yes to delete")
    if (model or all_models) and not include_hf:
        kept = [
            m.name
            for m in (
                models.catalog()
                if all_models
                else [m for n in model or [] for m in models.resolve_models(n)]
            )
            if models.model_status(m).hf_size_on_disk
        ]
        if kept:
            console.print(
                f"[dim]kept in the Hugging Face cache (shared; add --include-hf): "
                f"{escape(', '.join(kept))}[/dim]"
            )


@app.command()
def path(
    name: Annotated[str | None, typer.Argument(help="Print this model's local path")] = None,
    hf: Annotated[bool, typer.Option("--hf", help="Print the Hugging Face cache")] = False,
) -> None:
    """Print the model cache directory (or a model's files, or the HF cache)."""
    from .. import models
    from ..errors import VoiceAgentError
    from ..utils.download import cache_dir

    if hf:
        typer.echo(str(models.hf_cache_dir()))
        return
    if name is None:
        typer.echo(str(cache_dir()))
        return
    try:
        found = models.resolve_models(name)
    except VoiceAgentError as exc:
        raise _fail(str(exc)) from None
    for info in found:
        for fs in models.model_status(info).files:
            typer.echo(str(fs.path))


@app.command()
def du(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Disk usage per provider (model cache and Hugging Face cache)."""
    from .. import models

    report = models.disk_usage()
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "cache_dir": str(report.cache_dir),
                    "hf_cache_dir": str(report.hf_cache_dir),
                    "by_provider": report.by_provider,
                    "hf_by_provider": report.hf_by_provider,
                    "unused": report.unused,
                    "partial": report.partial,
                    "other": report.other,
                    "total": report.total,
                },
                indent=2,
            )
        )
        return
    table = Table(title="disk usage", title_justify="left")
    table.add_column("provider")
    table.add_column("model cache", justify="right")
    table.add_column("HF cache", justify="right")
    for prov in sorted(set(report.by_provider) | set(report.hf_by_provider)):
        hf = report.hf_by_provider.get(prov)
        table.add_row(
            prov.replace("_", "-"),
            human_size(report.by_provider[prov]) if prov in report.by_provider else "-",
            human_size(hf) if hf else "-",
        )
    for label, value in (
        ("(unused)", report.unused),
        ("(partial)", report.partial),
        ("(other)", report.other),
    ):
        if value:
            table.add_row(f"[dim]{label}[/dim]", human_size(value), "-")
    hf_total = sum(report.hf_by_provider.values())
    table.add_row("[bold]total[/bold]", human_size(report.total), human_size(hf_total) or "-")
    console.print(table)
    if report.unused or report.partial:
        console.print("[dim]`van models prune` lists what can be removed[/dim]")
    console.print(f"[dim]model cache: {escape(str(report.cache_dir))}[/dim]")
    if hf_total:
        console.print(f"[dim]HF cache: {escape(str(report.hf_cache_dir))}[/dim]")
