"""Seed-paper expansion: resolve LLM-named landmark papers into real records.

The planner can name 3-5 canonical papers for a topic, but a name alone does
not retrieve. This module resolves each title against OpenAlex (title search,
top hit) into a full :class:`Paper`, so the landmark itself enters the
candidate pool AND feeds the snowball seeds — pulling in its references and
citing papers even when the keyword queries missed the subfield entirely.
"""

from __future__ import annotations

import asyncio

import httpx

from citens import cache
from citens.config import settings
from citens.models import Paper

_OPENALEX_WORKS = "https://api.openalex.org/works"


def _polite_params() -> dict:
    return {"mailto": settings.openalex_email} if settings.openalex_email else {}


def _title_similarity(query: str, title: str) -> float:
    """Token-overlap similarity in [0, 1] (order-insensitive)."""
    import re

    toks = lambda s: {  # noqa: E731
        t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 2
    }
    a, b = toks(query), toks(title)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


async def _resolve_one(title: str) -> Paper | None:
    params = {
        "search": title,
        "per_page": 5,
        "select": "id,title,authorships,publication_year,abstract_inverted_index,"
        "cited_by_count,doi,primary_location",
        **_polite_params(),
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(_OPENALEX_WORKS, params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])
    except Exception:  # noqa: BLE001
        return None
    # OpenAlex relevance is good but not exact — guard against resolving to a
    # paper that merely mentions the landmark.
    for work in results:
        if _title_similarity(title, work.get("title") or "") >= 0.6:
            from citens.search.openalex import OpenAlexSearcher

            try:
                paper = OpenAlexSearcher.to_paper(work)
            except Exception:  # noqa: BLE001
                return None
            if paper.title.strip():
                paper.source = "seed-expansion"
                return paper
    return None


async def resolve_seeds(titles: list[str]) -> list[Paper]:
    """Resolve landmark-paper titles to Papers (deduplicated, best-first).

    Unresolvable titles are silently dropped — a seed is a bonus, not a
    guarantee (the model may hallucinate a title nothing matches).
    """
    titles = [t.strip() for t in titles if t and t.strip()][:5]
    if not titles:
        return []

    cache_ns = "seeds"
    cached = cache.get(cache_ns, {"titles": titles})
    if cached is not None:
        return [Paper(**p) for p in cached]

    results = await asyncio.gather(
        *(_resolve_one(t) for t in titles), return_exceptions=True
    )
    papers: dict[str, Paper] = {}
    for r in results:
        if isinstance(r, Paper) and r.id not in papers:
            papers[r.id] = r
    out = sorted(papers.values(), key=lambda p: p.citation_count, reverse=True)
    cache.put(cache_ns, {"titles": titles}, [p.model_dump() for p in out])
    return out
