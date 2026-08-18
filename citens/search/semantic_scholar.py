"""Semantic Scholar search source.

Uses the ``/paper/search/bulk`` endpoint which is markedly more reliable than
the classic ``/paper/search`` (the latter is rate-limited / connection-reset
under modest load). Bulk ignores ``limit`` and returns up to 1000 ranked docs,
so we slice client-side.

Authenticated tier (``SEMANTIC_SCHOLAR_API_KEY``): the key MUST travel as an
``x-api-key`` header and is capped at 1 request/second **cumulative across all
endpoints** — so every S2 request in the process goes through one shared
throttle that spaces request starts ≥ ``_MIN_INTERVAL`` apart, key or no key
(the unauthenticated tier is throttled even harder). A 429 is retried once
after backing off, belt-and-suspenders style.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from citens.config import settings
from citens.models import Paper
from citens.search.base import SearchSource, register

# shared process-wide: S2 counts every endpoint against the same 1 rps bucket
_lock = asyncio.Lock()
_last_start = 0.0
_MIN_INTERVAL = 1.1  # 1 rps + clock-skew margin


async def _throttled_get(
    client: httpx.AsyncClient, url: str, *, params: dict
) -> httpx.Response:
    global _last_start
    for attempt in range(2):
        async with _lock:
            wait = _MIN_INTERVAL - (time.monotonic() - _last_start)
            if wait > 0:
                await asyncio.sleep(wait)
            _last_start = time.monotonic()
            resp = await client.get(url, params=params)
        if resp.status_code != 429:
            return resp
        await asyncio.sleep(2.0 * (attempt + 1))  # rejected anyway — back off
    return resp


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
        resp = await _throttled_get(client, f"{self.BASE_URL}/paper/search/bulk", params=params)
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
