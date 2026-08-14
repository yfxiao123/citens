"""Full-text retrieval for citation grounding.

The single biggest lever for citation precision: ground claims against a
paper's FULL TEXT (methods, results, numbers) rather than its ~150-word
abstract. This module fetches an open-access PDF and converts it to clean text
with MarkItDown, trying several sources in order:

    1. arXiv PDF (for arXiv-indexed papers — always free)
    2. the paper's known OA pdf_url (captured from OpenAlex)
    3. Unpaywall (free, DOI-based OA locator)

If nothing yields a PDF, callers fall back to the abstract.
"""

from __future__ import annotations

import os
import re
import tempfile

import httpx

from litreview import cache
from litreview.config import settings
from litreview.models import Paper
from litreview.net import sync_client

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/\.[0-9]+)", re.IGNORECASE)
_CHUNK_SIZE = 1200
_HEADERS = {"User-Agent": "CiteLens/0.1 (open literature-review agent)"}
_md = None  # lazy MarkItDown singleton


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


def _download_and_convert(url: str) -> str | None:
    try:
        with sync_client(url, timeout=60) as client:
            r = client.get(url)
        if r.status_code != 200 or len(r.content) < 2000:
            return None
        ctype = r.headers.get("content-type", "").lower()
        if "pdf" not in ctype and not url.lower().endswith(".pdf") and not r.content[:5] == b"%PDF-":
            # not a PDF (likely an HTML landing page) — skip
            return None
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(r.content)
            tmp = fh.name
        try:
            text = _markitdown().convert(tmp).text_content or ""
        finally:
            os.unlink(tmp)
        return text.strip() if len(text) > 500 else None
    except Exception:  # noqa: BLE001
        return None


def fetch_fulltext(paper: Paper) -> str | None:
    """Return the paper's full text, or None if no OA PDF is available."""
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
