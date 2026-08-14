"""fetch_list.md — the honest fallback for paywalled full text.

Anti-bot walls (and SSO-only proxies) mean some PDFs simply cannot be fetched
by the agent. Instead of silently degrading to abstract-only grounding, each
run writes a fetch list: one line per unfetched paper with its DOI, landing
page and a suggested filename. The user opens the links in their own browser
(where their campus login lives), saves the PDFs into PAPERS_DIR, and the next
run picks them up automatically (see fulltext._local_pdf).
"""

from __future__ import annotations

from pathlib import Path

from litreview.config import settings
from litreview.grounding.fulltext import slugify
from litreview.models import Paper

_README = """# papers/ — 手动投递的 PDF / manually dropped PDFs

把你在浏览器里（用校园账号）下载的论文 PDF 放进本目录。下一次运行会自动
识别并用于全文溯源（文件名无需精确，含 DOI、arXiv 号或标题关键字即可）。

Drop PDFs you downloaded in your own browser (with your institutional login)
into this folder. The next run picks them up automatically — the filename just
needs to contain the DOI, arXiv id, or recognizable title words.
"""


def ensure_papers_dir() -> Path:
    """Create PAPERS_DIR (with a short README) if missing; return its path."""
    d = Path(settings.papers_dir)
    d.mkdir(parents=True, exist_ok=True)
    readme = d / "README.md"
    if not readme.exists():
        readme.write_text(_README, encoding="utf-8")
    return d


def suggested_filename(paper: Paper) -> str:
    if paper.doi:
        return f"{slugify(paper.doi)[:80]}.pdf"
    m = None
    if paper.url and "arxiv.org" in paper.url:
        tail = paper.url.rstrip("/").rsplit("/", 1)[-1]
        m = tail.replace(".pdf", "")
    if m:
        return f"arxiv-{slugify(m)[:60]}.pdf"
    return f"{slugify(paper.title)[:80]}.pdf"


def write_fetch_list(run_dir: str, papers: list[Paper]) -> str | None:
    """Write fetch_list.md for `papers` (those without fetched full text)."""
    if not papers:
        return None
    papers_dir = ensure_papers_dir()
    lines = [
        "# fetch_list — 待手动获取全文的论文 / papers needing manual full-text fetch",
        "",
        "以下论文没有公开可得（或通过你声明的代理可达）的全文，本次运行仅用摘要做引用核验。",
        "若你有权限：在浏览器中打开链接下载 PDF，按“建议文件名”放入",
        f"`{papers_dir}/`，重新运行即可自动纳入全文溯源。",
        "",
        "These papers have no openly fetchable full text; this run grounded them",
        "on abstracts only. If you have access: open each link in your browser,",
        f"save the PDF into `{papers_dir}/` using the suggested filename, and rerun.",
        "",
    ]
    for i, p in enumerate(papers):
        doi_url = f"https://doi.org/{p.doi}" if p.doi else ""
        links = [u for u in (doi_url, p.url) if u]
        year = f" ({p.year})" if p.year else ""
        lines.append(f"## [{i}] {p.title}{year}")
        lines.append("")
        if links:
            lines.extend(f"- {u}" for u in links)
        lines.append(f"- 建议文件名 / suggested filename: `{suggested_filename(p)}`")
        lines.append("")

    out = Path(run_dir) / "fetch_list.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)
