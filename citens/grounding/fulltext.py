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
# arXiv's own DOI namespace: 10.48550/arXiv.<id> — the ONLY place some
# preprints expose their arXiv-ness (S2's openAccessPdf is empty for them),
# so parsing it into a direct pdf URL skips the throttled title lookup.
# New-style ids: 2501.12345; old-style: cs/0301012, math.GT/0309136
_ARXIV_DOI_RE = re.compile(
    r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[a-z\-]+)*/[0-9]{7})",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")  # keep CJK (zh topics)
_CHUNK_SIZE = 1200
_HEADERS = {"User-Agent": "CiteLens/0.1 (open literature-review agent)"}
_md = None  # lazy MarkItDown singleton
_warned_missing = False


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


def _auto_pdf_path(paper: Paper) -> Path | None:
    """Where a successfully fetched PDF is kept (PAPERS_DIR/auto-<slug>.pdf).

    Persisting fetched PDFs makes runs re-groundable offline: the text cache
    is disposable, URLs rot, and publisher versions drift. The auto- prefix
    still matches :func:`_local_pdf`'s slug scan, so the next run (or
    ``citens reverify``) treats an auto-fetched PDF exactly like a user drop.
    """
    if paper.doi:
        name = f"{slugify(paper.doi)[:80]}.pdf"
    else:
        m = _ARXIV_ID_RE.search(paper.url or "")
        if m:
            name = f"arxiv-{slugify(m.group(1))[:60]}.pdf"
        elif len(slugify(paper.title)) >= 20:
            name = f"{slugify(paper.title)[:80]}.pdf"
        else:
            return None
    return Path(settings.papers_dir) / f"auto-{name}"


def _markitdown():
    global _md
    if _md is None:
        try:
            from markitdown import MarkItDown
        except ImportError:
            # Loud and once: without this dependency every PDF (auto-fetched
            # or user-dropped) silently fails and grounding degrades to
            # abstracts — the run looks healthy but carries no full text.
            global _warned_missing
            if not _warned_missing:
                _warned_missing = True
                print(
                    "    [fulltext] WARNING: markitdown is not installed — "
                    "PDF grounding is DISABLED (abstracts only). "
                    "Fix: uv sync (markitdown[pdf] is a core dependency)."
                )
            raise
        _md = MarkItDown()
    return _md


def _arxiv_pdf_url(paper: Paper) -> str | None:
    m = _ARXIV_ID_RE.search(paper.url or "")
    if not m:
        # the arXiv DOI (10.48550/arXiv.<id>) appears in EITHER field
        # depending on which source reported the record
        m = _ARXIV_DOI_RE.search(paper.doi or "") or _ARXIV_DOI_RE.search(paper.url or "")
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
        with sync_client(timeout=12, headers=_HEADERS) as client:
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


def _s2_pdf_url(doi: str) -> list[str]:
    """Semantic Scholar's OA link for this DOI (graph API, single paper).

    S2 indexes author-homepage and repository copies that neither OpenAlex
    nor Unpaywall lists (measured: the cornell.edu copy of a paywalled
    T&F paper). Sends the configured SEMANTIC_SCHOLAR_API_KEY when present —
    the anonymous pool 429s exactly when a run harvests many papers in a
    row, silently emptying this leg (measured: a GOLD-OA ACM link missed).
    """
    if not doi:
        return []
    headers = dict(_HEADERS)
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    try:
        with sync_client(timeout=12, headers=headers) as client:
            r = client.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                params={"fields": "openAccessPdf"},
            )
            if r.status_code != 200:
                return []
            u = ((r.json().get("openAccessPdf") or {}).get("url") or "").strip()
            return [u] if u else []
    except Exception:  # noqa: BLE001
        return []


def _core_pdf_url(doi: str) -> str | None:
    """Open-access PDF via CORE (aggregates repositories worldwide).

    Needs CORE_API_KEY (free registration); disabled when unset.
    """
    key = settings.core_api_key
    if not key or not doi:
        return None
    try:
        with sync_client(timeout=12, headers=_HEADERS) as client:
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


def _unpaywall_pdf_url(doi: str) -> list[str]:
    """ALL OA pdf urls Unpaywall knows for this DOI (repository copies incl.).

    Reading only ``best_oa_location`` missed real PDFs: when the "best"
    location is a publisher landing page, the repository copy in
    ``oa_locations`` never gets tried (measured on a 14-paper finance run:
    4 of 9 missing full texts were sitting in non-best locations).
    ``url_for_pdf`` only — the ``url`` field is usually a landing page the
    download step's content-type check rejects anyway.
    """
    email = settings.openalex_email or "citelens@example.com"
    try:
        with sync_client(timeout=12) as client:
            r = client.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": email},
            )
            r.raise_for_status()
            urls = [
                (loc.get("url_for_pdf") or "").strip()
                for loc in (r.json().get("oa_locations") or [])
            ]
            return [u for u in dict.fromkeys(urls) if u]
    except Exception:  # noqa: BLE001
        return []


def _openalex_pdf_urls(doi: str) -> list[str]:
    """ALL pdf urls from the OpenAlex work record's locations.

    Same rationale as Unpaywall's full location list: institutional
    repository copies (ut-capitole, ACM OA, author homepages indexed as
    landing pages with pdf_url) live beyond the best_oa_location that the
    search-time harvest reads.
    """
    if not doi:
        return []
    try:
        with sync_client(timeout=12, headers=_HEADERS) as client:
            r = client.get(f"https://api.openalex.org/works/doi:{doi}")
            r.raise_for_status()
            urls = [
                (loc.get("pdf_url") or "").strip()
                for loc in (r.json().get("locations") or [])
            ]
            return [u for u in dict.fromkeys(urls) if u]
    except Exception:  # noqa: BLE001
        return []


def _convert_pdf_file(path: str) -> str | None:
    """MarkItDown-convert a local PDF file to text (None if unusable)."""
    try:
        text = _markitdown().convert(path).text_content or ""
        return text.strip() if len(text) > 500 else None
    except Exception:  # noqa: BLE001
        return None


def _pdf_bytes_to_text(content: bytes, paper: Paper | None = None) -> str | None:
    """Convert PDF bytes to text. With ``paper``, KEEP the PDF in PAPERS_DIR.

    A fetched-but-deleted PDF made every cache miss a re-download (URL rot,
    publisher version drift, cleared .cache). Kept files carry the auto-
    prefix so they ride the same local-match path as user drops.
    """
    if not content or content[:5] != b"%PDF-":
        return None
    import contextlib

    keep = _auto_pdf_path(paper) if paper is not None else None
    if keep is not None:
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep.write_bytes(content)
        path = str(keep)
    else:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(content)
            path = fh.name
    text = None
    try:
        text = _convert_pdf_file(path)
    finally:
        if keep is None:
            with contextlib.suppress(OSError):
                os.unlink(path)
    if keep is not None and not text:
        # unusable (scanned images, corrupt) — don't poison later runs'
        # local-first scan with a file that can never convert
        with contextlib.suppress(OSError):
            keep.unlink()
    return text


def _download_and_convert(
    url: str, paper: Paper | None = None
) -> tuple[str | None, str]:
    """Fetch one candidate URL -> (text, outcome).

    The outcome string is the audit trail the transcript shows when a paper
    ends up abstract-only ("arxiv.org HTTP 429", "dl.acm.org HTTP 403",
    "PDF解析失败"...) — "0/16 fulltext" runs need a visible why.
    """
    from urllib.parse import urlparse

    host = urlparse(url).hostname or url[:30]
    try:
        url = rewrite_url(url)  # ride the user's EZproxy/declared access
        with sync_client(url, timeout=30, headers=_HEADERS) as client:
            r = client.get(url)
        if r.status_code != 200:
            return None, f"{host} HTTP {r.status_code}"
        if len(r.content) < 2000:
            return None, f"{host} 内容过短"
        ctype = r.headers.get("content-type", "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf") or r.content[:5] == b"%PDF-":
            text = _pdf_bytes_to_text(r.content, paper)
            if text:
                return text, f"{host} ok"
            return None, f"{host} PDF解析失败"
        return None, f"{host} 非PDF"
    except Exception as e:  # noqa: BLE001 - timeout / conn refused / TLS ...
        return None, f"{host} {type(e).__name__}"


# per-paper harvest audit trail (paper.id -> compact outcome string); read by
# the pipeline's transcript lines so "why abstract-only" is visible in the UI
_FETCH_REPORTS: dict[str, str] = {}


def fetch_report(paper_id: str) -> str:
    """Why fetch_fulltext got (or didn't get) this paper's text, last run."""
    return _FETCH_REPORTS.get(paper_id, "")


def fetch_fulltext(paper: Paper) -> str | None:
    """Return the paper's full text, or None if unavailable.

    Order: user-dropped PDF (PAPERS_DIR) -> cache -> open-access network fetch.
    The local check runs before the cache so a PDF dropped after a previous
    miss is still picked up. Local conversions are cached by file mtime —
    re-parsing every dropped/auto PDF on every run was pure repeated work.
    """
    local = _local_pdf(paper)
    if local is not None:
        try:
            mtime = local.stat().st_mtime_ns
        except OSError:
            mtime = 0
        local_key = {"id": paper.id, "file": str(local), "mtime": mtime}
        text = cache.get("fulltext_local", local_key)
        if text is None:
            text = _convert_pdf_file(str(local))
            cache.put("fulltext_local", local_key, text or "")
        if text:
            _FETCH_REPORTS[paper.id] = f"本地PDF {len(text) // 1000}k字"
            return text

    cached = cache.get("fulltext", paper.id)
    if cached is not None:
        _FETCH_REPORTS[paper.id] = (
            f"缓存 {len(cached) // 1000}k字" if cached else "缓存未命中"
        )
        return cached or None

    candidates: list[tuple[str, str]] = []  # (leg label, url)
    arxiv = _arxiv_pdf_url(paper)
    if arxiv:
        candidates.append(("arxiv", arxiv))
    if paper.pdf_url:
        candidates.append(("pdf_url", paper.pdf_url))
    if paper.doi:
        # full location lists first (repository copies), then CORE
        candidates.extend(("s2", u) for u in _s2_pdf_url(paper.doi))
        candidates.extend(("openalex", u) for u in _openalex_pdf_urls(paper.doi))
        candidates.extend(("unpaywall", u) for u in _unpaywall_pdf_url(paper.doi))
        core = _core_pdf_url(paper.doi)
        if core:
            candidates.append(("core", core))

    text = None
    outcomes: list[str] = []
    for leg, url in candidates:
        text, outcome = _download_and_convert(url, paper)
        outcomes.append(f"{leg}:{outcome}")
        if text:
            break
    if text:
        _FETCH_REPORTS[paper.id] = f"{leg} ✓ {len(text) // 1000}k字"
    else:
        # cap at the first 3 tried legs — enough to answer "why" without
        # turning the transcript line into a wall
        _FETCH_REPORTS[paper.id] = " · ".join(outcomes[:3]) or "无OA候选"

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
