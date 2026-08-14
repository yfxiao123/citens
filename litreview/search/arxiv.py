"""arXiv search source (free, no key). The official client is synchronous and
sleeps between pages; we run it in a worker thread so it never blocks the
event loop while other sources query concurrently."""

from __future__ import annotations

import asyncio

import arxiv

from litreview.models import Paper
from litreview.search.base import SearchSource, register


@register("arxiv")
class ArxivSearcher(SearchSource):
    name = "arXiv"

    async def search(self, keywords: list[str], max_results: int) -> list[Paper]:
        return await asyncio.to_thread(self._search_sync, keywords, max_results)

    def _search_sync(self, keywords: list[str], max_results: int) -> list[Paper]:
        client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
        per_kw = max(max_results // max(len(keywords), 1), 1)
        out: list[Paper] = []
        for kw in keywords:
            try:
                search = arxiv.Search(
                    query=kw,
                    max_results=per_kw,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                for result in client.results(search):
                    out.append(self._to_paper(result))
            except Exception:
                continue
        return out

    @staticmethod
    def _to_paper(result: arxiv.Result) -> Paper:
        return Paper(
            title=result.title,
            authors=[a.name for a in result.authors],
            year=result.published.year if result.published else None,
            abstract=result.summary,
            source="arXiv",
            citation_count=0,  # arXiv exposes no citation counts
            url=result.entry_id,
            doi=result.doi,
        )
