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
rewrite) via citens.net. If nothing yields a PDF, callers fall back to the
abstract.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from citens import cache
from citens.config import settings
from citens.models import Paper
from citens.net import rewrite_url, sync_client

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/\.[0-9]+)", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")  # keep CJK (zh topics)
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
    return _arxiv_title_lookup(paper)


def _norm_words(title: str) -> set[str]:
    import re as _re

    return {t for t in _re.split(r"[^a-z0-9]+", title.lower()) if len(t) > 2}


def _arxiv_title_lookup(paper: Paper) -> str | None:
    """Find the paper's arXiv preprint by title (Atom API, top hits).

    Many paywalled journal papers have a free arXiv version whose URL never
    appears in the metadata we collected — a title match recovers the PDF.
    Guards against weak matches (min token overlap) since the API ranks by
    relevance, not equality.
    """
    title = (paper.title or "").strip()
    if len(title) < 10:
        return None
    try:
        with sync_client(timeout=20, headers=_HEADERS) as client:
            r = client.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": f'ti:"{title}"', "max_results": 3},
            )
            r.raise_for_status()
    except Exception:  # noqa: BLE001
        return None
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(r.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        want = _norm_words(title)
        for entry in root.findall("a:entry", ns):
            hit = (entry.findtext("a:title", "", ns) or "").strip()
            got = _norm_words(hit)
            if want and got and len(want & got) / min(len(want), len(got)) >= 0.8:
                entry_id = (entry.findtext("a:id", "", ns) or "").strip()
                m = _ARXIV_ID_RE.search(entry_id.replace("/abs/", "/pdf/"))
                if m:
                    return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    except Exception:  # noqa: BLE001
        return None
    return None


def _core_pdf_url(doi: str) -> str | None:
    """Open-access PDF via CORE (aggregates repositories worldwide).

    Needs CORE_API_KEY (free registration); disabled when unset.
    """
    key = settings.core_api_key
    if not key or not doi:
        return None
    try:
        with sync_client(timeout=20, headers=_HEADERS) as client:
            r = client.get(
                "https://api.core.ac.uk/v3/search/works",
                params={"q": f'doi:"{doi}"', "limit": 3},
                headers={**_HEADERS, "Authorization": f"Bearer {key}"},
            )
            r.raise_for_status()
            for hit in r.json().get("results", []):
                for loc in hit.get("locations", []):
                    pdf = (loc.get("pdfUrl") or "").strip()
                    if pdf:
                        return pdf
    except Exception:  # noqa: BLE001
        return None
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
        core = _core_pdf_url(paper.doi)
        if core:
            candidates.append(core)

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
