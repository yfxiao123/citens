"""Semi-automatic fulltext downloader: a visible, entitled browser that
fetches a run's missing PDFs into PAPERS_DIR.

Reverse-engineered PDF endpoints lose to publisher bot defenses (IEEE 202s,
SD 403s, Springer 303s) even with a valid session — but the HUMAN click
always works. This tool is that click, automated where possible and manual
where not: it opens each paper's article page in the headed login browser,
auto-clicks the PDF control when it can find one, waits for the user to
click manually when it cannot, and files every download into papers/ where
the grounding layer picks it up (then `citens reverify`).

Usage: citens fetch <run_dir>
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from citens.config import settings
from citens.models import Paper

# click candidates, most specific first
_PDF_SELECTORS = [
    "a[href*='stamp.jsp']",
    "a[href*='/content/pdf/']",
    "a[href*='pdf-direct']",
    "a[href*='/doi/pdf/']",
    "a[href*='pdfft']",
    "a[data-track-action='download pdf']",
    "a:has-text('Download PDF')",
    "button:has-text('Download PDF')",
    "a:has-text('PDF')",
]


def _load_run_papers(run_dir: str) -> list[Paper]:
    """Final paper set of a run (09_final_papers, else 04_extracted)."""
    import json

    steps = Path(run_dir) / "steps"
    for name in ("09_final_papers.json", "04_extracted.json"):
        p = steps / name
        if p.is_file():
            with p.open(encoding="utf-8") as fh:
                return [Paper(**r) for r in json.load(fh)]
    raise FileNotFoundError(f"no paper list found under {steps}")


def missing_papers(run_dir: str) -> list[Paper]:
    """Run papers that have neither a dropped PDF nor cached full text."""
    from citens import cache
    from citens.grounding.fulltext import _local_pdf

    out = []
    for p in _load_run_papers(run_dir):
        if _local_pdf(p) is not None:
            continue
        if cache.get("fulltext", p.id):  # a previous fetch succeeded
            continue
        if p.doi:
            out.append(p)
    return out


def fetch_run(run_dir: str) -> dict:
    """Open the headed browser and walk the user through the missing PDFs."""
    from playwright.sync_api import sync_playwright

    from citens.grounding.browserfetch import _jar_to_playwright_cookies
    from citens.grounding.fulltext import pdf_slugs

    todo = missing_papers(run_dir)
    if not todo:
        return {"missing": 0, "downloaded": 0}

    papers_dir = Path(settings.papers_dir)
    papers_dir.mkdir(parents=True, exist_ok=True)
    cookies = _jar_to_playwright_cookies()
    for c in cookies:
        if not c["domain"].startswith("."):
            c["domain"] = "." + c["domain"]

    downloaded = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx = browser.new_context(accept_downloads=True)
        if cookies:
            ctx.add_cookies(cookies)
        page = ctx.new_page()

        # capture downloads anywhere in the context (new tabs included)
        saved: list[Path] = []

        def _save(dl):
            try:
                slug = pdf_slugs(_current_paper[0])[0] if _current_paper[0] else "download"
                dest = papers_dir / f"{slug}.pdf"
                dl.save_as(str(dest))
                saved.append(dest)
            except Exception:  # noqa: BLE001
                pass

        _current_paper: list[Paper | None] = [None]
        ctx.on("page", lambda pg: pg.on("download", _save))
        page.on("download", _save)

        for i, paper in enumerate(todo, 1):
            _current_paper[0] = paper
            before = len(saved)
            print(f"[{i}/{len(todo)}] {paper.title[:70]}")
            try:
                page.goto(f"https://doi.org/{paper.doi}", timeout=60000,
                          wait_until="domcontentloaded")
                page.wait_for_timeout(4000)  # SPA hydration
            except Exception as e:  # noqa: BLE001
                print(f"    打开失败: {str(e)[:80]}")
            # auto-click pass
            for sel in _PDF_SELECTORS:
                try:
                    loc = page.locator(sel).first
                    if loc.count() == 0:
                        continue
                    loc.click(timeout=5000)
                    page.wait_for_timeout(6000)
                    break
                except Exception:  # noqa: BLE001
                    continue
            if len(saved) > before:
                downloaded += 1
                print(f"    ✓ 已保存 {saved[-1].name}")
                continue
            # manual pass — the window is visible, the user clicks
            print("    自动点击未成功。请在浏览器窗口手动点击 PDF 下载；")
            print("    完成后回车继续（直接回车 = 放弃此篇）。")
            with contextlib.suppress(EOFError):
                input()
            page.wait_for_timeout(2000)
            if len(saved) > before:
                downloaded += 1
                print(f"    ✓ 已保存 {saved[-1].name}")
                _current_paper[0] = None
            else:
                print("    ✗ 未捕获下载，跳过")
        browser.close()

    return {"missing": len(todo), "downloaded": downloaded,
            "files": [str(p) for p in saved]}
