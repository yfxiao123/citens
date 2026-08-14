"""Crossref search + enrichment source (free, no key, huge DOI coverage).

Two roles:
  * a SearchSource registered as "crossref" (metadata + abstracts by relevance);
  * :func:`fetch_by_doi` — used by the enrichment step to fill missing abstracts
    via direct DOI lookup (Crossref deposits JATS abstracts for many works).
"""

from __future__ import annotations

import asyncio
import re

import httpx

from litreview.config import settings
from litreview.models import Paper
from litreview.search.base import SearchSource, register

_BASE = "https://api.crossref.org"
_JATS_RE = re.compile(r"<[^>]+>")


def _polite_params() -> dict:
    return {"mailto": settings.crossref_email} if settings.crossref_email else {}


def _clean_abstract(raw: str | None) -> str:
    if not raw:
        return ""
    return re.sub(r"\s+", " ", _JATS_RE.sub(" ", raw)).strip()


@register("crossref")
class CrossrefSearcher(SearchSource):
    name = "Crossref"

    async def search(self, keywords: list[str], max_results: int) -> list[Paper]:
        per_keyword = max(max_results // max(len(keywords), 1), 5)
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "CiteLens/0.1"}) as client:
            tasks = [self._one(client, kw, per_keyword) for kw in keywords]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[Paper] = []
        for res in results:
            if isinstance(res, list):
                out.extend(res)
        return out

    async def _one(self, client: httpx.AsyncClient, query: str, limit: int) -> list[Paper]:
        params = {"query.bibliographic": query, "rows": min(limit, 30), **_polite_params()}
        resp = await client.get(f"{_BASE}/works", params=params)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        return [self._to_paper(it) for it in items]

    @staticmethod
    def _to_paper(item: dict) -> Paper:
        authors = []
        for a in item.get("author", []):
            name = (a.get("given", "") + " " + a.get("family", "")).strip()
            if name:
                authors.append(name)
        doi = item.get("DOI")
        links = [
            l.get("URL") for l in item.get("link", []) if l.get("content-type") == "application/pdf"
        ]
        return Paper(
            title=(item.get("title") or [""])[0],
            authors=authors,
            year=_year(item),
            abstract=_clean_abstract(item.get("abstract")),
            source="Crossref",
            citation_count=item.get("is-referenced-by-count", 0) or 0,
            url=item.get("URL", "") or (f"https://doi.org/{doi}" if doi else ""),
            doi=doi,
            pdf_url=links[0] if links else None,
            venue=(item.get("container-title") or [""])[0] or "",
        )


def _year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (item.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def fetch_abstract_by_doi(doi: str) -> str:
    """Direct DOI lookup → cleaned abstract (empty string if unavailable)."""
    if not doi:
        return ""
    try:
        with httpx.Client(timeout=20, headers={"User-Agent": "CiteLens/0.1"}) as client:
            r = client.get(f"{_BASE}/works/{doi}", params=_polite_params())
            if r.status_code != 200:
                return ""
            return _clean_abstract(r.json()["message"].get("abstract"))
    except Exception:  # noqa: BLE001
        return ""
