"""Semantic Scholar search source.

Uses the ``/paper/search/bulk`` endpoint which is markedly more reliable than
the classic ``/paper/search`` (the latter is rate-limited / connection-reset
under modest load). Bulk ignores ``limit`` and returns up to 1000 ranked docs,
so we slice client-side.
"""

from __future__ import annotations

import asyncio

import httpx

from citens.config import settings
from citens.models import Paper
from citens.search.base import SearchSource, register


@register("semantic_scholar")
class SemanticScholarSearcher(SearchSource):
    name = "Semantic Scholar"
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self) -> None:
        self.headers = {"User-Agent": "CiteLens/0.1"}
        if settings.semantic_scholar_api_key:
            self.headers["x-api-key"] = settings.semantic_scholar_api_key

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
        params = {
            "query": query,
            "fields": "title,authors,year,abstract,citationCount,externalIds,url,venue",
        }
        resp = await client.get(f"{self.BASE_URL}/paper/search/bulk", params=params)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [self._to_paper(item) for item in data[:limit]]

    @staticmethod
    def _to_paper(item: dict) -> Paper:
        authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
        ext_ids = item.get("externalIds") or {}
        return Paper(
            title=item.get("title", ""),
            authors=authors,
            year=item.get("year"),
            abstract=item.get("abstract") or "",
            source="Semantic Scholar",
            citation_count=item.get("citationCount", 0) or 0,
            url=item.get("url") or "",
            doi=ext_ids.get("DOI"),
            venue=item.get("venue") or "",
        )
