"""Command-line interface (Typer + Rich)."""

from __future__ import annotations

import sys
import traceback
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from litreview import __version__
from litreview.events import (
    EventBus,
    RunCompleted,
    RunFailed,
    RunStarted,
    StepCompleted,
    StepStarted,
    StepProgress,
)
from litreview.orchestration import RunOptions, run_pipeline
from litreview.search import REGISTRY as SEARCH_REGISTRY

app = typer.Typer(
    name="litreview",
    help="Write critical, citation-grounded literature reviews from a topic.",
    no_args_is_help=True,
)
console = Console()


def _make_rich_handler(console: Console):
    def handler(event):
        if isinstance(event, RunStarted):
            console.rule(f"[bold cyan]{event.topic}[/]")
        elif isinstance(event, StepStarted):
            console.print(f"[cyan]▶ {event.title}[/]")
        elif isinstance(event, StepProgress):
            cur = f" ({event.current}/{event.total})" if event.total else ""
            console.print(f"   • {event.message}{cur}", highlight=False)
        elif isinstance(event, StepCompleted):
            console.print(f"   [green]✓[/] {event.message}", highlight=False)
        elif isinstance(event, RunCompleted):
            tbl = Table(show_header=False, box=None, padding=(0, 1))
            tbl.add_row("主题", event.summary.get("topic", ""))
            tbl.add_row("候选论文", str(event.summary.get("total_papers", 0)))
            tbl.add_row("筛选通过", str(event.summary.get("filtered_papers", 0)))
            tbl.add_row("主题", ", ".join(event.summary.get("themes", [])))
            tbl.add_row("带引用论断", str(event.summary.get("claims", 0)))
            tbl.add_row("参考文献", str(event.summary.get("references", 0)))
            prec = event.summary.get("citation_precision")
            if prec is not None:
                tbl.add_row("引用精度", f"{prec * 100:.0f}%")
            tbl.add_row("综述文件", event.review_path)
            console.print(Panel(tbl, title="[bold green]综述生成完成[/]", expand=False))
        elif isinstance(event, RunFailed):
            console.print(f"[bold red]✗ {event.step}: {event.message}[/]")

    return handler


@app.command()
def run(
    topic: list[str] = typer.Argument(None, help="研究主题 (research topic)"),
    n: Optional[int] = typer.Option(None, "-n", "--max-papers", help="最终保留论文数"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="候选池目标数"),
    sources: Optional[str] = typer.Option(None, "--sources", help="逗号分隔检索源"),
    no_cache: bool = typer.Option(False, "--no-cache", help="禁用缓存"),
    no_fulltext: bool = typer.Option(
        False, "--no-fulltext", help="不获取全文（仅用摘要溯源，精度较低）"
    ),
):
    """Generate a literature review for TOPIC."""
    topic_str = " ".join(topic) if topic else "大语言模型在金融领域的应用"
    src_list = [s.strip() for s in sources.split(",")] if sources else None

    bus = EventBus()
    bus.subscribe(_make_rich_handler(console))
    options = RunOptions(
        max_results=max_results,
        max_papers=n,
        sources=src_list,
        use_cache=not no_cache,
        fetch_fulltext=not no_fulltext,
    )
    try:
        run_pipeline(topic_str, options, bus)
    except Exception:  # noqa: BLE001
        console.print("[bold red]运行失败:[/]")
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def sources():
    """List available search sources."""
    tbl = Table(title="可用检索源 / Search sources")
    tbl.add_column("name", style="cyan")
    for name in SEARCH_REGISTRY:
        tbl.add_row(name)
    console.print(tbl)


@app.command()
def version():
    """Show version."""
    console.print(f"litreview {__version__}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":
    main()
