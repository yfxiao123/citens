"""Command-line interface (Typer + Rich)."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from citens import __version__
from citens.config import settings
from citens.events import (
    EventBus,
    RunCompleted,
    RunFailed,
    RunStarted,
    StepCompleted,
    StepProgress,
    StepStarted,
)
from citens.orchestration import RunOptions, run_pipeline
from citens.search import REGISTRY as SEARCH_REGISTRY

app = typer.Typer(
    name="citens",
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
    n: int | None = typer.Option(None, "-n", "--max-papers", help="最终保留论文数"),
    max_results: int | None = typer.Option(None, "--max-results", help="候选池目标数"),
    sources: str | None = typer.Option(None, "--sources", help="逗号分隔检索源"),
    mode: str | None = typer.Option(
        None, "--mode", help="运行模式: quick_scan/deep_review/interactive (默认自动检测)"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="禁用缓存"),
    no_fulltext: bool = typer.Option(
        False, "--no-fulltext", help="不获取全文（仅用摘要溯源，精度较低）"
    ),
    no_clarify: bool = typer.Option(
        False, "--no-clarify", help="跳过跑前澄清问题（直接运行）"
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="综述输出语言: en/zh (默认取 REVIEW_LANGUAGE 或 en)"
    ),
):
    """Generate a literature review for TOPIC."""
    from citens.models import RunMode

    topic_str = " ".join(topic) if topic else "大语言模型在金融领域的应用"
    src_list = [s.strip() for s in sources.split(",")] if sources else None
    if language:
        settings.review_language = language

    # Parse mode if provided
    run_mode = None
    if mode:
        try:
            run_mode = RunMode(mode)
        except ValueError:
            console.print(f"[red]无效的 mode: {mode}，可选值: quick_scan/deep_review/interactive[/]")
            raise typer.Exit(code=1) from None

    bus = EventBus()
    bus.subscribe(_make_rich_handler(console))
    options = RunOptions(
        max_results=max_results,
        max_papers=n,
        sources=src_list,
        use_cache=not no_cache,
        fetch_fulltext=not no_fulltext,
        mode=run_mode,
    )
    # pre-run clarification (interactive) — shape the search before it starts
    if not no_clarify:
        options.filters = _clarify_interactive(topic_str)
    try:
        run_pipeline(topic_str, options, bus)
    except Exception:  # noqa: BLE001
        console.print("[bold red]运行失败:[/]")
        traceback.print_exc()
        raise typer.Exit(code=1) from None


def _clarify_interactive(topic: str) -> dict:
    """Ask the user 2-4 clarifying questions before the run (CLI, blocking).

    Returns the chosen answers as {question_id: answer}. Requires a TTY;
    falls back to no filters (run proceeds) when stdin is not interactive.
    """
    from citens.agents.clarify import generate_clarifying_questions

    if not sys.stdin.isatty():
        return {}
    questions = generate_clarifying_questions(topic)
    if not questions:
        return {}
    filters: dict[str, str] = {}
    console.print("\n[bold cyan]跑前澄清 / Pre-run clarification[/]（Enter 使用默认值）")
    for q in questions:
        opts = q["options"]
        default = q.get("default") or opts[0]
        shown = "  ".join(f"{i + 1}. {o}" for i, o in enumerate(opts))
        console.print(f"\n[bold]{q['question']}[/]\n{shown}")
        try:
            raw = input(f"[默认 {default}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            chosen = default
        else:
            try:
                chosen = opts[int(raw) - 1] if 1 <= int(raw) <= len(opts) else raw
            except ValueError:
                chosen = raw
        filters[q["id"]] = chosen
    return filters


@app.command()
def sources():
    """List available search sources."""
    tbl = Table(title="可用检索源 / Search sources")
    tbl.add_column("name", style="cyan")
    for name in SEARCH_REGISTRY:
        tbl.add_row(name)
    console.print(tbl)


@app.command()
def eval(  # noqa: A001
    topics: list[str] = typer.Argument(None, help="要评测的主题（默认一组内置主题）"),
    n: int = typer.Option(8, "-n", "--max-papers", help="每主题论文数"),
    mode: str = typer.Option("deep_review", "--mode", help="运行模式"),
    from_runs: str = typer.Option(
        None, "--from-runs", help="不跑新任务，只汇总已有 run 目录（glob 模式）"
    ),
):
    """Eval harness: run topics, collect citation-precision metrics, write eval/report.md.

    Live LLM + network required — this is the maintainer tool behind the
    README's precision claims, not a CI step. --from-runs re-renders the
    table from existing runs offline.
    """
    from datetime import date

    from citens.eval import collect_metrics, render_table

    if from_runs:
        import glob as _glob

        dirs = _glob.glob(from_runs)
        rows = [collect_metrics(d) for d in dirs]
    else:
        from citens.models import RunMode

        topic_list = topics or [
            "limit order book modeling",
            "graph neural networks for molecule property prediction",
            "retrieval-augmented generation for question answering",
        ]
        rows = []
        for t in topic_list:
            console.print(f"\n[bold cyan]== eval run:[/] {t}")
            meta = run_pipeline(
                t,
                RunOptions(
                    max_papers=n,
                    mode=RunMode(mode),
                    allow_supplement=True,
                ),
            )
            rows.append(collect_metrics(meta.run_dir))

    if not rows:
        console.print("[yellow]没有可汇总的 run[/]")
        raise typer.Exit()

    table = render_table(rows)
    out = Path("eval")
    out.mkdir(exist_ok=True)
    report = (
        f"# Eval report — {date.today().isoformat()}\n\n"
        f"citation precision = (supported + partial) / verifiable claims\n\n"
        f"{table}\n"
    )
    (out / "report.md").write_text(report, encoding="utf-8")
    console.print(Panel(table, title=f"eval: {len(rows)} runs"))
    console.print(f"[green]✓[/] 写入 {out / 'report.md'}")


@app.command()
def sjr(
    force: bool = typer.Option(False, "--force", help="重新下载（即使文件已存在）"),
    mirror: bool = typer.Option(
        False, "--mirror", help="用纯 CSV 镜像（无需 pyreadr，分区为百分位近似）"
    ),
    official: bool = typer.Option(
        False, "--official", help="直接从 scimagojr.com 下载（常被反爬拦截且很慢）"
    ),
):
    """Download the SCImago journal-rank CSV (venue quartiles for ranking).

    Default: the ikashnitsky/sjrdata GitHub mirror — full official data
    (2025 edition, field-normalized quartiles) converted locally (needs
    ``pip install pyreadr``). --mirror uses a plain-CSV mirror with
    percentile-approximated quartiles; --official hits scimagojr.com directly.
    The dataset is CC BY-NC licensed, so it is fetched on demand instead of
    shipping with the package. Ranking degrades gracefully (neutral venue
    factor) when it is absent.
    """
    from citens.config import settings
    from citens.ranking import SJRIndex

    path = Path(settings.sjr_csv_path)
    if path.is_file() and not force:
        console.print(f"[green]✓[/] 已存在 {path}（--force 重新下载）")
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if official:
            url = "https://www.scimagojr.com/journalrank.php?out=xls"
            console.print(f"下载 SCImago SJR 数据 …\n  {url}")
            _download_to(url, path)
        elif mirror:
            url = (
                "https://raw.githubusercontent.com/Michael-E-Rose/"
                "SCImagoJournalRankIndicators/master/all.csv"
            )
            console.print(f"下载 SCImago SJR 数据（CSV 镜像）…\n  {url}")
            _download_to(url, path)
        else:
            url = (
                "https://raw.githubusercontent.com/ikashnitsky/sjrdata/"
                "master/data/sjr_journals.rda"
            )
            console.print(f"下载 SCImago SJR 数据（官方数据镜像，需 pyreadr）…\n  {url}")
            rda = path.with_suffix(".rda")
            _download_to(url, rda)
            from citens.ranking import convert_rda_to_csv

            n = convert_rda_to_csv(rda, path)
            console.print(f"转换完成：{n:,} journals")
            rda.unlink(missing_ok=True)
        index = SJRIndex.load(path)
        console.print(f"[green]✓[/] {len(index):,} journals -> {path}")
    except ImportError:
        console.print("[bold red]缺少 pyreadr[/]（默认数据源需要）：pip install pyreadr")
        console.print("或使用 --mirror 下载无需转换的 CSV 镜像")
        raise typer.Exit(code=1) from None
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]下载失败:[/] {e}")
        console.print(f"可手动下载后保存为 {settings.sjr_csv_path}")
        raise typer.Exit(code=1) from None


def _download_to(url: str, path: Path) -> None:
    from citens.net import sync_client

    with sync_client(url, timeout=300, headers={"User-Agent": "Mozilla/5.0 (citens)"}) as client:
        r = client.get(url)
        r.raise_for_status()
    if len(r.content) < 100_000 or b"Title" not in r.content[:8000]:
        raise ValueError(f"unexpected payload ({len(r.content)} bytes)")
    path.write_bytes(r.content)


@app.command()
def version():
    """Show version."""
    console.print(f"citens {__version__}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":
    main()
