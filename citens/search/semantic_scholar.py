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


async def _throttled_post(
    client: httpx.AsyncClient, url: str, *, params: dict, json_body: dict
) -> httpx.Response:
    global _last_start
    for attempt in range(2):
        async with _lock:
            wait = _MIN_INTERVAL - (time.monotonic() - _last_start)
            if wait > 0:
                await asyncio.sleep(wait)
            _last_start = time.monotonic()
            resp = await client.post(url, params=params, json=json_body)
        if resp.status_code != 429:
            return resp
        await asyncio.sleep(2.0 * (attempt + 1))
    return resp


@register("semantic_scholar")
class SemanticScholarSearcher(SearchSource):
    name = "Semantic Scholar"
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self) -> None:
        super().__init__()
        self.headers = {"User-Agent": "CiteLens/0.1"}
        if settings.semantic_scholar_api_key:
            self.headers["x-api-key"] = settings.semantic_scholar_api_key

    async def search(self, keywords: list[str], max_results: int) -> list[Paper]:
        per_keyword = max(max_results // max(len(keywords), 1), 5)
        from citens.search.base import SEARCH_CONCURRENCY

        sem = asyncio.Semaphore(SEARCH_CONCURRENCY)

        async def _guarded(q: str) -> list[Paper]:
            async with sem:
                return await self._one(client, q, per_keyword)

        async with httpx.AsyncClient(timeout=30, headers=self.headers) as client:
            tasks = [_guarded(kw) for kw in keywords]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[Paper] = []
        for res in results:
            if isinstance(res, list):
                out.extend(res)
        return out

    async def _one(self, client: httpx.AsyncClient, query: str, limit: int) -> list[Paper]:
        params: dict[str, str] = {
            "query": query,
            # openAccessPdf: free OA pdf link (preprints + green-OA deposits).
            # Without it S2 records carry no pdf_url and fulltext grounding
            # falls back to arXiv title lookup + Unpaywall.
            "fields": (
                "title,authors,year,abstract,citationCount,externalIds,url,"
                "venue,openAccessPdf"
            ),
        }
        if self.constraints and (self.constraints.year_from or self.constraints.year_to):
            import datetime

            yf = self.constraints.year_from or 1900
            yt = self.constraints.year_to or datetime.date.today().year
            params["year"] = f"{yf}-{yt}"
        resp = await _throttled_get(client, f"{self.BASE_URL}/paper/search/bulk", params=params)
        resp.raise_for_status()
        data = resp.json().get("data", [])[:limit]
        papers = [self._to_paper(item) for item in data]
        for p in papers:
            p.matched_queries.append(query)  # retrieval provenance
        self.query_stats[query] = len(papers)
        return papers

    @staticmethod
    def _to_paper(item: dict) -> Paper:
        authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
        ext_ids = item.get("externalIds") or {}
        oa_pdf = (item.get("openAccessPdf") or {}).get("url") or ""
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
            pdf_url=oa_pdf or None,
        )


async def _batch_arxiv_lookup(ids: list[str]) -> dict[str, tuple[int, str | None]]:
    """One S2 batch call: arXiv id -> (citationCount, DOI), aligned response."""
    headers = {"User-Agent": "CiteLens/0.1"}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        resp = await _throttled_post(
            client,
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": "citationCount,externalIds"},
            json_body={"ids": [f"ARXIV:{i}" for i in ids]},
        )
        resp.raise_for_status()
        data = resp.json()
    out: dict[str, tuple[int, str | None]] = {}
    for pid, item in zip(ids, data, strict=False):
        if item:
            out[pid] = (
                item.get("citationCount") or 0,
                (item.get("externalIds") or {}).get("DOI"),
            )
    return out


async def enrich_citations(
    papers: list[Paper], max_ids: int = 200, _lookup=None
) -> int:
    """Fill citation_count/doi on arXiv-leg captures via one S2 batch join.

    The arXiv API exposes no citation counts, so every arXiv-only capture
    carries citation_count=0 — a field-defining work then sorts as if
    uncited and blend_pool's citation trim drops it before the LLM filter
    ever sees it (measured on the RAG bench run: Lewis 2020, captured via
    the arXiv leg only, cut at the 24-cap). Joining by arXiv id restores
    the true count and gives the published DOI.

    Degrades to a no-op on any failure (returns 0): enrichment is an
    optimization for ranking, never a dependency of the run.
    """
    from citens.search.base import paper_arxiv_id

    targets = [
        p for p in papers if p.citation_count == 0 and paper_arxiv_id(p)
    ][:max_ids]
    if not targets:
        return 0
    lookup = _lookup or _batch_arxiv_lookup
    info: dict = {}
    for attempt in range(2):
        try:
            info = await lookup([paper_arxiv_id(p) or "" for p in targets])
            break
        except Exception as e:  # noqa: BLE001 — a failed join must not fail the run
            # transient transport errors (empty-message disconnects) are the
            # common failure here, not 429s — one retry saves most of them
            if attempt == 1:
                print(
                    "[enrich_citations] S2 batch join failed, "
                    f"keeping citation_count=0: {e}"
                )
                return 0
            await asyncio.sleep(2.0)
    updated = 0
    for p in targets:
        hit = info.get(paper_arxiv_id(p) or "")
        if not hit:
            continue
        count, doi = hit
        p.citation_count = count
        if not p.doi and doi:
            p.doi = doi
        updated += 1
    return updated
