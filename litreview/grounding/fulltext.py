"""Full-text retrieval for citation grounding.

The single biggest lever for citation precision: ground claims against a
paper's FULL TEXT (methods, results, numbers) rather than its ~150-word
abstract. This module tries, in order:

    0. a PDF the user dropped into PAPERS_DIR (see fetch_list.md — the honest
       fallback for paywalled content the agent itself cannot fetch)
    1. arXiv PDF (for arXiv-indexed papers — always free)
    2. the paper's known OA pdf_url (captured from OpenAlex)
    3. Unpaywall (free, DOI-based OA locator)

Publisher URLs additionally ride the user's declared access (proxy / EZproxy
rewrite) via litreview.net. If nothing yields a PDF, callers fall back to the
abstract.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from litreview import cache
from litreview.config import settings
from litreview.models import Paper
from litreview.net import rewrite_url, sync_client

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/\.[0-9]+)", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_CHUNK_SIZE = 1200
_HEADERS = {"User-Agent": "CiteLens/0.1 (open literature-review agent)"}
_md = None  # lazy MarkItDown singleton


def slugify(text: str) -> str:
    """Lowercase, non-alphanumeric runs -> single '-' (used to match dropped PDFs)."""
    return _NON_ALNUM_RE.sub("-", (text or "").lower()).strip("-")


def pdf_slugs(paper: Paper) -> list[str]:
    """Filename slugs a dropped PDF for this paper would plausibly carry."""
    slugs: list[str] = []
    if paper.doi:
        slugs.append(slugify(paper.doi))
    m = _ARXIV_ID_RE.search(paper.url or "")
    if m:
        slugs.append(slugify(m.group(1)))
    title_slug = slugify(paper.title)
    if len(title_slug) >= 20:  # too-short titles would match anything
        slugs.append(title_slug[:60])
    return [s for s in slugs if len(s) >= 8]


def _local_pdf(paper: Paper) -> Path | None:
    """A user-dropped PDF in PAPERS_DIR matching this paper, or None."""
    d = Path(settings.papers_dir)
    if not d.is_dir():
        return None
    slugs = pdf_slugs(paper)
    if not slugs:
        return None
    for f in sorted(d.glob("*.pdf")):
        stem = slugify(f.stem)
        if any(s in stem or stem in s for s in slugs):
            return f
    return None


def _markitdown():
    global _md
    if _md is None:
        from markitdown import MarkItDown

        _md = MarkItDown()
    return _md


def _arxiv_pdf_url(paper: Paper) -> str | None:
    m = _ARXIV_ID_RE.search(paper.url or "")
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    return None


def _unpaywall_pdf_url(doi: str) -> str | None:
    email = settings.openalex_email or "citelens@example.com"
    try:
        with sync_client(timeout=20) as client:
            r = client.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": email},
            )
            r.raise_for_status()
            loc = (r.json().get("best_oa_location") or {})
            return (loc.get("url_for_pdf") or loc.get("url") or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def _convert_pdf_file(path: str) -> str | None:
    """MarkItDown-convert a local PDF file to text (None if unusable)."""
    try:
        text = _markitdown().convert(path).text_content or ""
        return text.strip() if len(text) > 500 else None
    except Exception:  # noqa: BLE001
        return None


def _download_and_convert(url: str) -> str | None:
    try:
        url = rewrite_url(url)  # ride the user's EZproxy/declared access
        with sync_client(url, timeout=60, headers=_HEADERS) as client:
            r = client.get(url)
        if r.status_code != 200 or len(r.content) < 2000:
            return None
        ctype = r.headers.get("content-type", "").lower()
        if "pdf" not in ctype and not url.lower().endswith(".pdf") and r.content[:5] != b"%PDF-":
            # not a PDF (likely an HTML landing page) — skip
            return None
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(r.content)
            tmp = fh.name
        try:
            return _convert_pdf_file(tmp)
        finally:
            os.unlink(tmp)
    except Exception:  # noqa: BLE001
        return None


def fetch_fulltext(paper: Paper) -> str | None:
    """Return the paper's full text, or None if unavailable.

    Order: user-dropped PDF (PAPERS_DIR) -> cache -> open-access network fetch.
    The local check runs before the cache so a PDF dropped after a previous
    miss is still picked up.
    """
    local = _local_pdf(paper)
    if local is not None:
        text = _convert_pdf_file(str(local))
        if text:
            return text

    cached = cache.get("fulltext", paper.id)
    if cached is not None:
        return cached or None

    candidates: list[str] = []
    arxiv = _arxiv_pdf_url(paper)
    if arxiv:
        candidates.append(arxiv)
    if paper.pdf_url:
        candidates.append(paper.pdf_url)
    if paper.doi:
        upw = _unpaywall_pdf_url(paper.doi)
        if upw:
            candidates.append(upw)

    text = None
    for url in candidates:
        text = _download_and_convert(url)
        if text:
            break

    cache.put("fulltext", paper.id, text or "")
    return text


def chunk_text(text: str, size: int = _CHUNK_SIZE) -> list[str]:
    """Split text into ~`size`-char chunks on sentence boundaries."""
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 <= size or not buf:
            buf = (buf + " " + s).strip()
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks
