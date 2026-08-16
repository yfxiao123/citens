"""Command-line interface (Typer + Rich)."""

from __future__ import annotations

import contextlib
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
    no_pool: bool = typer.Option(
        False, "--no-pool", help="不注入/回写 citens collect 的文献池"
    ),
    no_fulltext: bool = typer.Option(
        False, "--no-fulltext", help="不获取全文（仅用摘要溯源，精度较低）"
    ),
    no_clarify: bool = typer.Option(
        False, "--no-clarify", help="跳过跑前澄清问题（直接运行）"
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="综述输出语言: en/zh (默认取 REVIEW_LANGUAGE 或 en)"
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", "-c",
        help="并行 LLM 调用数 (默认取 LLM_CONCURRENCY 或 6)",
    ),
):
    """Generate a literature review for TOPIC."""
    from citens.models import RunMode

    topic_str = " ".join(topic) if topic else "大语言模型在金融领域的应用"
    src_list = [s.strip() for s in sources.split(",")] if sources else None
    if language:
        settings.review_language = language
    if concurrency:
        settings.llm_concurrency = concurrency
    elif (n or 0) >= 30 and settings.llm_concurrency < 8:
        console.print(
            "[yellow]提示: 大规模 run 建议提高并发 (--concurrency 12 或 .env 里 "
            "LLM_CONCURRENCY=12)，可显著缩短 filter/extract/verify 阶段耗时[/]"
        )

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
        use_pool=not no_pool,
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


# Default publisher tour for `citens login`: ARTICLE deep links, not homepages —
# the institutional entitlement handshake (SP-initiated SSO) only fires when an
# actual paywalled article is requested; a homepage visit sets no usable token.
_LOGIN_SITES = [
    "https://www.sciencedirect.com/science/article/pii/S0304405X1000102X",
    "https://link.springer.com/article/10.1007/s11579-012-0082-5",
    "https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413",
    "https://www.tandfonline.com/doi/full/10.1080/14697688.2016.1154244",
    "https://journals.sagepub.com/doi/10.1177/00220574231213461",
    "https://ieeexplore.ieee.org/document/8777151",
]


@app.command()
def login(
    url: list[str] = typer.Option(
        [], "--url", help="额外要登录的站点（可重复；不带则走默认出版商清单）"
    ),
    all_sites: bool = typer.Option(
        False, "--all", help="遍历全部默认出版商站点（首次 SSO 后其余自动登录）"
    ),
):
    """打开浏览器完成学校统一身份认证，把会话 Cookie 存入 data/cookies.json。

    密码只进浏览器、不进任何配置或代码；之后的 run 会自动带上这些
    Cookie 抓付费全文。默认只开 ScienceDirect；--all 会在同一浏览器
    会话里依次访问全部主流出版商——统一身份认证的会话在 IdP 上，
    第一个站登录后，后续站点自动放行，密码只需输一次。
    会话过期后重跑一次本命令即可。
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        console.print(
            "[yellow]![/] 需要 playwright: pip install 'citens[login]' "
            "&& playwright install chromium"
        )
        raise typer.Exit(code=1) from None

    from citens.net import save_cookie_jar

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx = browser.new_context()
        page = ctx.new_page()
        sites = url or (_LOGIN_SITES if all_sites else _LOGIN_SITES[:1])
        console.print(
            f"[cyan]将依次打开 {len(sites)} 篇付费文章（非首页——机构授权只在文章页触发）。[/]"
        )
        console.print("第一个弹出统一身份认证时登录；之后若出现'选择机构'页选南京大学。")
        for i, site in enumerate(sites, 1):
            try:
                page.goto(site, timeout=60000)
            except Exception as e:  # noqa: BLE001
                console.print(f"  [yellow]{site} 加载异常（跳过）:[/] {e}")
                continue
            console.print(f"  [{i}/{len(sites)}] {site}")
        console.print("确认各文章页显示为[已下载权限]状态（有 PDF 下载按钮）后，回这里按回车收 Cookie…")
        with contextlib.suppress(EOFError):
            input()
        cookies = ctx.cookies()
        browser.close()

    per_host: dict[str, list[str]] = {}
    for c in cookies:
        # empty-value cookies (e.g. sim-inst-token="") carry no entitlement
        if not c.get("name") or not c.get("value"):
            continue
        host = (c.get("domain") or "").lstrip(".").lower()
        if not host:
            continue
        per_host.setdefault(host, []).append(f"{c['name']}={c['value']}")
    jar = {h: "; ".join(v) for h, v in per_host.items()}
    save_cookie_jar(jar)
    hosts = ", ".join(sorted(jar)[:6])
    console.print(f"[green]✓[/] 已保存 {len(jar)} 个域的会话 Cookie（{hosts}…）")
    console.print("  之后 citens run 抓这些域的全文时会自动携带；过期后重跑 citens login。")


@app.command()
def collect(
    topic: str = typer.Argument(..., help="研究领域（中文或英文）"),
    target: int = typer.Option(100, "-n", help="文献池目标条数"),
    queries: str = typer.Option("", "--queries", help="追加自定义查询（逗号分隔）"),
    no_author: bool = typer.Option(False, "--no-author", help="跳过作者深耕信号补全"),
    audit_recall: bool = typer.Option(
        False, "--audit-recall", help="建池后用 top 综述的参考文献算池覆盖率"
    ),
):
    """按系统综述的方式建立该领域的文献池（只记录，不下载全文）。

    多批次关键词（维度覆盖 + 综述定向查询）→ 逐查询检索并去重 →
    记录细分领域/作者/年份/摘要/关键词/被引/期刊 → 补全一作深耕信号
    （works/h-index）→ 持久化到 data/litdb/<主题>.jsonl。
    之后 citens run 同主题会自动注入文献池，全文在筛选后分批获取。
    """
    from citens.collect import collect as _collect

    extra = [q.strip() for q in queries.split(",") if q.strip()] if queries else None

    def _prog(msg):
        console.print(f"  • {msg}")

    with console.status("检索并记录文献…", spinner="dots"):
        summary = _collect(
            topic, target=target, extra_queries=extra,
            enrich_authors=not no_author, on_progress=_prog,
        )
    console.print(
        f"[green]✓[/] 本轮发现 {summary['found']} 篇 · 新增 {summary['added']} 篇 · "
        f"文献池累计 {summary['pool_total']} 篇"
    )
    console.print(f"  文献池: {summary['pool_path']}")
    top = list(summary["subfields"].items())[:6]
    if top:
        console.print("  细分领域分布: " + " · ".join(f"{k} {v}" for k, v in top))
    dead = [q for q, n in summary["query_hits"].items() if n == 0]
    if dead:
        console.print(f"  [yellow]零命中查询:[/] {', '.join(dead[:5])}")
    console.print("  下次 citens run 该主题时自动注入文献池（--no-pool 可关闭）。")

    if audit_recall:
        from citens.collect import audit_recall as _audit

        rep = _audit(topic)
        if rep["reviews_checked"]:
            cov = rep["coverage"]
            cov_msg = f"{cov * 100:.0f}%" if cov is not None else "n/a"
            console.print(
                f"  [cyan]召回审计:[/] {rep['reviews_checked']} 篇综述的参考文献 "
                f"{rep['in_pool']}/{rep['refs_checked']} 已在池内（覆盖 {cov_msg}）"
            )
        else:
            console.print("  [yellow]召回审计:[/] 池内暂无带 DOI 的综述记录")


@app.command()
def resume(
    run_dir: str = typer.Argument(..., help="要续跑的 run 目录（含 steps/04_extracted.json）"),
    no_supplement: bool = typer.Option(False, "--no-supplement", help="跳过补检循环"),
    language: str | None = typer.Option(None, "--language", "-l", help="综述输出语言: en/zh"),
):
    """Resume an interrupted run: reuse its extracted papers, recompose the review.

    Retrieval (search/filter/snowball/extract) is skipped — the run directory's
    steps/04_extracted.json becomes the starting pool. Useful when a deep run
    dies during writing/verification, or when you want to re-write with a
    different language/model.
    """
    import json

    if language:
        settings.review_language = language

    topic_file = Path(run_dir) / "run.json"
    if topic_file.is_file():
        topic = json.loads(topic_file.read_text(encoding="utf-8"))["topic"]
    else:
        # run dirs are <topic>-<timestamp>; the slug preserves CJK characters
        topic = Path(run_dir).name.rsplit("-", 1)[0] or Path(run_dir).name
    console.print(f"[cyan]resume[/] {run_dir} · topic: {topic}")

    bus = EventBus()
    bus.subscribe(_make_rich_handler(console))
    try:
        run_pipeline(
            topic,
            RunOptions(resume_dir=run_dir, allow_supplement=not no_supplement),
            bus,
        )
    except FileNotFoundError as e:
        console.print(f"[bold red]无法续跑:[/] {e}")
        raise typer.Exit(code=1) from None
    except Exception:  # noqa: BLE001
        console.print("[bold red]续跑失败:[/]")
        traceback.print_exc()
        raise typer.Exit(code=1) from None


@app.command()
def reverify(
    run_dir: str = typer.Argument(..., help="要重新核验的 run 目录"),
):
    """Re-verify an existing run's claims against newly available full text.

    Drop the PDFs listed in the run's fetch_list.md into papers/ first — this
    command re-grounds every claim (full text where available) and rewrites
    verification.json / provenance.json, without re-running retrieval or
    writing. Reports the precision delta.
    """
    from citens.orchestration.reverify import reverify as _reverify

    bus = EventBus()
    bus.subscribe(_make_rich_handler(console))
    try:
        summary = _reverify(run_dir, bus)
    except FileNotFoundError as e:
        console.print(f"[bold red]无法重验:[/] {e}")
        raise typer.Exit(code=1) from None
    prev = summary.get("previous_precision")
    delta = ""
    if prev is not None:
        delta = f" (was {prev:.1%})"
    console.print(
        f"[green]✓[/] {summary['claims']} claims · {summary['fulltext']} full text · "
        f"precision {summary['precision']:.1%}{delta}"
    )


@app.command()
def audit(
    run_dir: str = typer.Argument(..., help="要审核的 run 目录"),
    ingest: str = typer.Option("", "--ingest", help="填好的审核清单路径: 回收人工判定并计算一致率"),
):
    """Generate (or ingest) a human audit sheet for a run's claims.

    No --ingest: writes 审核清单.md — every claim with its machine verdict,
    cited papers, and a fill-in 人工判定 slot (s/p/u).

    With --ingest: parses the filled sheet, computes human-vs-machine
    agreement / leniency, and writes audit_result.json.
    """
    from citens.audit import generate_audit_sheet, ingest_audit

    if not ingest:
        path = generate_audit_sheet(run_dir)
        console.print(f"[green]✓[/] 审核清单已生成: {path}")
        console.print("  填写每条'人工判定:'为 s/p/u 后, 运行:")
        console.print(f"  citens audit {run_dir} --ingest {path}")
        return
    try:
        report = ingest_audit(run_dir, ingest)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[bold red]审核失败:[/] {e}")
        raise typer.Exit(code=1) from None
    console.print(
        f"[green]✓[/] 人工判定 {report['judged']}/{report['of_total']} 条 · "
        f"一致率 {report['agreement_rate'] * 100:.0f}% · "
        f"机器偏宽 {report['machine_lenient']} 条 / 偏严 {report['machine_strict']} 条"
    )
    console.print(
        f"  人工grounded率 {report['human_grounded_rate'] * 100:.0f}% vs "
        f"机器自报精度 {report['machine_reported_precision'] * 100:.0f}% · "
        f"明细见 audit_result.json"
    )


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
