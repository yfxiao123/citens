"""OpenAlex search source (open, free).

Uses ``filter=title.search:...`` with ``relevance_score`` ordering — a tight
title-term match that avoids the classic ``search=`` full-text pitfall where
high-cited papers matching a single generic word (e.g. "machine learning",
"simulation") swamp the results.
"""

from __future__ import annotations

import asyncio

import httpx

from citens.config import settings
from citens.models import Paper
from citens.search.base import SearchSource, register


@register("openalex")
class OpenAlexSearcher(SearchSource):
    name = "OpenAlex"
    BASE_URL = "https://api.openalex.org/works"

    def __init__(self) -> None:
        self.headers = {"User-Agent": "CiteLens/0.1"}
        if settings.openalex_email:
            self.headers["User-Agent"] = f"mailto:{settings.openalex_email}"

    async def search(self, keywords: list[str], max_results: int) -> list[Paper]:
        per_keyword = max(max_results // max(len(keywords), 1), 5)
        async with httpx.AsyncClient(timeout=30, headers=self.headers) as client:
            tasks = [self._one(client, kw, per_keyword) for kw in keywords]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[Paper] = []
        for res in results:
            if isinstance(res, list):
                out.extend(res)
        return out

    async def _one(self, client: httpx.AsyncClient, query: str, limit: int) -> list[Paper]:
        params: dict[str, str | int] = {
            "filter": f"title.search:{query}",
            "per_page": min(limit, 50),
            "sort": "relevance_score:desc",
            "select": (
                "id,title,authorships,publication_year,abstract_inverted_index,"
                "cited_by_count,doi,primary_location,open_access,relevance_score"
            ),
        }
        resp = await client.get(self.BASE_URL, params=params)
        resp.raise_for_status()
        return [self.to_paper(w) for w in resp.json().get("results", [])]

    @staticmethod
    def to_paper(work: dict) -> Paper:
        """Convert an OpenAlex work record to a Paper.

        Public because snowball.py and enrichment.py reuse the same conversion.
        """
        authors: list[str] = []
        for a in work.get("authorships", []):
            name = (a.get("author") or {}).get("display_name", "")
            if name:
                authors.append(name)
        abstract = OpenAlexSearcher.decode_abstract(work.get("abstract_inverted_index"))
        loc = work.get("primary_location") or {}
        source_obj = loc.get("source") or {}
        source_name = source_obj.get("display_name", "") or "OpenAlex"
        # Prefer the open-access PDF URL.
        pdf_url = (loc.get("pdf_url") or "").strip() or None
        oa = work.get("open_access") or {}
        if not pdf_url and oa.get("oa_url"):
            pdf_url = oa["oa_url"].strip()
        return Paper(
            title=work.get("title", ""),
            authors=authors,
            year=work.get("publication_year"),
            abstract=abstract,
            source=f"OpenAlex ({source_name})",
            citation_count=work.get("cited_by_count", 0) or 0,
            url=work.get("id", ""),
            doi=work.get("doi"),
            pdf_url=pdf_url,
            venue=source_name if source_name != "OpenAlex" else "",
        )

    @staticmethod
    def decode_abstract(inverted_index: dict | None) -> str:
        if not inverted_index:
            return ""
        words: dict[int, str] = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words.get(i, "") for i in range(len(words)))
